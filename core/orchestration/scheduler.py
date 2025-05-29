"""
Distributed Task Scheduler - Hybrid Orchestrator for Agent Swarms
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
from datetime import datetime
import heapq
import logging
from typing import Dict, List, Optional, Tuple, AsyncGenerator
import uuid
from pydantic import BaseModel, Field, validator
import kubernetes_asyncio as k8s
from prometheus_client import Counter, Gauge, Histogram
from redis.asyncio import Redis

from core.agent.registry import AgentRegistry, AgentRecord
from core.workflow.dag import ExecutionGraph

# Metrics
SCHEDULER_METRICS = {
    "tasks_scheduled": Counter("scheduler_tasks_total", "Total scheduled tasks", ["priority"]),
    "scheduling_latency": Histogram("scheduler_latency_seconds", "Task scheduling latency"),
    "queue_depth": Gauge("scheduler_queue_depth", "Pending tasks in queue"),
    "resource_utilization": Gauge("scheduler_resource_util", "Resource utilization %", ["resource_type"])
}

@dataclass(frozen=True)
class ResourceProfile:
    cpu: float  # CPU cores
    memory: float  # GB
    gpu: int = 0
    ephemeral_storage: float = 0.0  # GB

class TaskRequest(BaseModel):
    payload: dict
    priority: int = Field(ge=0, le=3)  # 0:Critical, 1:High, 2:Normal, 3:Low
    deadline: Optional[datetime] = None
    dependencies: List[str] = Field(default_factory=list)
    retry_policy: Dict[str, int] = {"max_attempts": 3, "backoff": 5}
    affinity: Dict[str, List[str]] = Field(default_factory=dict)
    resource_limits: ResourceProfile = ResourceProfile(cpu=1.0, memory=2.0)

    @validator('dependencies')
    def check_cyclic_deps(cls, v):
        if len(v) != len(set(v)):
            raise ValueError("Duplicate dependencies detected")
        return v

class TaskState(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    status: str = "pending"  # pending -> scheduled -> running -> succeeded/failed
    assigned_agent: Optional[str] = None
    attempts: int = 0
    last_error: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None

class SchedulerConfig(BaseModel):
    scheduling_interval: float = 0.1  # Seconds between scheduling attempts
    max_batch_size: int = 100  # Max tasks per scheduling iteration
    eviction_timeout: int = 300  # Seconds before rescheduling stuck tasks
    kube_namespace: str = "aelion-agents"
    redis_url: str = "redis://localhost:6379/1"
    policy_module: str = "policies.binpacking_v1"  # Loadable policy module

class DistributedTaskScheduler:
    """
    Hybrid scheduler combining:
    - Priority queues for urgent tasks
    - Bin-packing for resource optimization
    - Deadline-aware scheduling
    - Kubernetes-native scaling
    """
    
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.registry = AgentRegistry()
        self.redis = Redis.from_url(config.redis_url)
        self.k8s_api = k8s.client.CoreV1Api()
        self.k8s_batch_api = k8s.client.BatchV1Api()
        
        # Priority queues: (priority, deadline, task)
        self._queues = {
            0: [],
            1: [],
            2: [],
            3: []
        }
        
        # State tracking
        self._lock = asyncio.Lock()
        self._scheduled_tasks: Dict[str, TaskState] = {}
        self._load_policy_module()
        
        # Background services
        self._scheduler_loop = asyncio.create_task(self._run_scheduler())
        self._eviction_monitor = asyncio.create_task(self._monitor_evictions())
        self._scaler = asyncio.create_task(self._autoscale_agents())
    
    def _load_policy_module(self):
        module_path = self.config.policy_module
        try:
            module = __import__(module_path, fromlist=['SchedulingPolicy'])
            self.policy = module.SchedulingPolicy()
        except ImportError as e:
            logging.error(f"Failed to load policy module {module_path}: {e}")
            raise
    
    async def submit_task(self, request: TaskRequest) -> TaskState:
        """Atomic task submission with persistence"""
        state = TaskState()
        
        async with self._lock:
            # Persist to Redis
            await self.redis.hset(
                f"tasks:{state.task_id}",
                mapping=state.dict()
            )
            
            # Queue based on priority
            deadline_ts = request.deadline.timestamp() if request.deadline else float('inf')
            heapq.heappush(
                self._queues[request.priority],
                (deadline_ts, state.task_id, request)
            )
            SCHEDULER_METRICS["queue_depth"].inc()
            
        return state
    
    async def _run_scheduler(self) -> None:
        """Core scheduling loop with backpressure control"""
        while True:
            try:
                async with self._lock:
                    # 1. Check resource availability
                    agents = await self.registry.find(status="active")
                    available_resources = [
                        (agent.agent_id, agent.metadata.get("resources", {}))
                        for agent in agents
                    ]
                    
                    # 2. Policy-based decision making
                    tasks_to_schedule = self._select_tasks(available_resources)
                    
                    # 3. Assign tasks using policy
                    assignments = self.policy.assign(
                        tasks=tasks_to_schedule,
                        agents=available_resources
                    )
                    
                    # 4. Dispatch assignments
                    await self._dispatch_assignments(assignments)
                    
                    # 5. Scale cluster if needed
                    pending_count = sum(len(q) for q in self._queues.values())
                    if pending_count > 100:
                        await self._trigger_scale_up()
                    
                await asyncio.sleep(self.config.scheduling_interval)
                
            except Exception as e:
                logging.error(f"Scheduling loop error: {e}")
                await asyncio.sleep(1)  # Backoff
    
    def _select_tasks(self, available_resources: list) -> List[Tuple[str, TaskRequest]]:
        """Select tasks that can be scheduled given current resources"""
        selected = []
        total_resources = self._aggregate_resources(available_resources)
        
        for priority in [0, 1, 2, 3]:  # Process queues in priority order
            while len(selected) < self.config.max_batch_size and self._queues[priority]:
                deadline_ts, task_id, request = heapq.heappop(self._queues[priority])
                
                # Check resource feasibility
                if self._resources_sufficient(request.resource_limits, total_resources):
                    selected.append((task_id, request))
                    total_resources = self._subtract_resources(total_resources, request.resource_limits)
                    SCHEDULER_METRICS["queue_depth"].dec()
                else:
                    heapq.heappush(self._queues[priority], (deadline_ts, task_id, request))
                    break  # Insufficient resources for higher priority tasks
        
        return selected
    
    async def _dispatch_assignments(self, assignments: Dict[str, List[str]]) -> None:
        """Dispatch tasks to agents with atomic state updates"""
        for agent_id, task_ids in assignments.items():
            for task_id in task_ids:
                state = TaskState.parse_raw(await self.redis.hgetall(f"tasks:{task_id}"))
                state.status = "scheduled"
                state.assigned_agent = agent_id
                state.scheduled_at = datetime.utcnow()
                
                # Update Redis and local state
                await self.redis.hset(f"tasks:{task_id}", mapping=state.dict())
                self._scheduled_tasks[task_id] = state
                
                # Notify agent via Redis Pub/Sub
                await self.redis.publish(
                    f"agent:{agent_id}:tasks",
                    task_id
                )
                
                SCHEDULER_METRICS["tasks_scheduled"].labels(priority=state.priority).inc()
    
    async def _trigger_scale_up(self) -> None:
        """Scale Kubernetes deployment based on pending tasks"""
        try:
            # Get current deployment
            dep = await self.k8s_api.read_namespaced_deployment(
                name="aelion-worker",
                namespace=self.config.kube_namespace
            )
            
            # Calculate desired replicas
            current_replicas = dep.spec.replicas or 0
            pending_tasks = sum(len(q) for q in self._queues.values())
            desired_replicas = min(
                current_replicas + (pending_tasks // 10),
                self.policy.max_scale
            )
            
            if desired_replicas > current_replicas:
                patch = {"spec": {"replicas": desired_replicas}}
                await self.k8s_api.patch_namespaced_deployment(
                    name="aelion-worker",
                    namespace=self.config.kube_namespace,
                    body=patch
                )
                logging.info(f"Scaled workers from {current_replicas} to {desired_replicas}")
        
        except k8s.client.exceptions.ApiException as e:
            logging.error(f"K8s scale error: {e}")
    
    async def _monitor_evictions(self) -> None:
        """Reschedule stuck or failed tasks"""
        while True:
            try:
                now = datetime.utcnow()
                evict_candidates = []
                
                async for task_id in self.redis.scan_iter("tasks:*"):
                    state = TaskState.parse_raw(await self.redis.hgetall(task_id))
                    if state.status == "running" and state.started_at:
                        if (now - state.started_at).seconds > self.config.eviction_timeout:
                            evict_candidates.append(task_id)
                
                for task_id in evict_candidates:
                    await self._reschedule_task(task_id, reason="timeout")
                    
                await asyncio.sleep(30)
            
            except Exception as e:
                logging.error(f"Eviction monitor error: {e}")
    
    async def _reschedule_task(self, task_id: str, reason: str) -> None:
        """Handle task rescheduling with backoff"""
        async with self._lock:
            state = TaskState.parse_raw(await self.redis.hgetall(f"tasks:{task_id}"))
            
            if state.attempts >= state.retry_policy.get("max_attempts", 3):
                state.status = "failed"
                state.last_error = f"Max retries exceeded: {reason}"
                await self.redis.hset(f"tasks:{task_id}", mapping=state.dict())
                return
                
            state.attempts += 1
            state.status = "pending"
            state.assigned_agent = None
            state.scheduled_at = None
            state.started_at = None
            
            # Apply exponential backoff
            backoff = state.retry_policy.get("backoff", 5) ** state.attempts
            await asyncio.sleep(backoff)
            
            # Re-enqueue task
            request = await self.redis.hget(f"task_requests:{task_id}")
            if request:
                heapq.heappush(
                    self._queues[request.priority],
                    (request.deadline.timestamp() if request.deadline else float('inf'), 
                     task_id, 
                     request)
                )
                await self.redis.hset(f"tasks:{task_id}", mapping=state.dict())
    
    async def _autoscale_agents(self) -> None:
        """Horizontal scaling based on resource utilization"""
        while True:
            try:
                # Get cluster metrics
                metrics = await self.k8s_api.list_namespaced_pod_metrics(
                    namespace=self.config.kube_namespace
                )
                
                # Calculate utilization
                cpu_usage = 0.0
                cpu_limit = 0.0
                for pod in metrics.items:
                    for container in pod.containers:
                        cpu_usage += container.usage.get("cpu", 0)
                        cpu_limit += container.limits.get("cpu", 0)
                
                utilization = (cpu_usage / cpu_limit) * 100 if cpu_limit > 0 else 0
                SCHEDULER_METRICS["resource_utilization"].labels(resource_type="cpu").set(utilization)
                
                # Scale decision
                if utilization > 80:
                    await self._trigger_scale_up()
                elif utilization < 20:
                    await self._trigger_scale_down()
                    
                await asyncio.sleep(60)
            
            except Exception as e:
                logging.error(f"Autoscaler error: {e}")
                await asyncio.sleep(30)
    
    def _resources_sufficient(self, required: ResourceProfile, available: Dict[str, float]) -> bool:
        """Check if aggregated resources can satisfy request"""
        return (
            available["cpu"] >= required.cpu and
            available["memory"] >= required.memory and
            available.get("gpu", 0) >= required.gpu
        )
    
    def _aggregate_resources(self, agents: List[Tuple[str, dict]]) -> Dict[str, float]:
        """Sum available resources across active agents"""
        totals = {"cpu": 0.0, "memory": 0.0, "gpu": 0}
        for _, resources in agents:
            if resources:
                totals["cpu"] += resources.get("cpu", 0)
                totals["memory"] += resources.get("memory", 0)
                totals["gpu"] += resources.get("gpu", 0)
        return totals
    
    def _subtract_resources(self, totals: Dict[str, float], request: ResourceProfile) -> Dict[str, float]:
        """Subtract task resources from available pool"""
        return {
            "cpu": totals["cpu"] - request.cpu,
            "memory": totals["memory"] - request.memory,
            "gpu": totals["gpu"] - request.gpu
        }

# Example Policy Implementation (separate file)
class SchedulingPolicy:
    """Bin-packing resource allocation strategy"""
    
    def __init__(self):
        self.max_scale = 1000  # Max agent instances
    
    def assign(self, tasks: List[Tuple[str, TaskRequest]], agents: List[Tuple[str, dict]]) -> Dict[str, List[str]]:
        assignments = {agent_id: [] for agent_id, _ in agents}
        agent_resources = {
            agent_id: {
                "cpu": float(res["cpu"]),
                "memory": float(res["memory"]),
                "gpu": int(res.get("gpu", 0))
            }
            for agent_id, res in agents
        }
        
        # Sort tasks by resource demand (descending)
        sorted_tasks = sorted(
            tasks,
            key=lambda x: (x[1].resource_limits.gpu, x[1].resource_limits.cpu),
            reverse=True
        )
        
        # Bin packing algorithm
        for task_id, request in sorted_tasks:
            req = request.resource_limits
            for agent_id in assignments:
                if (
                    agent_resources[agent_id]["cpu"] >= req.cpu and
                    agent_resources[agent_id]["memory"] >= req.memory and
                    agent_resources[agent_id]["gpu"] >= req.gpu
                ):
                    assignments[agent_id].append(task_id)
                    # Update remaining resources
                    agent_resources[agent_id]["cpu"] -= req.cpu
                    agent_resources[agent_id]["memory"] -= req.memory
                    agent_resources[agent_id]["gpu"] -= req.gpu
                    break
        
        return assignments

# Usage Example
async def main():
    config = SchedulerConfig()
    scheduler = DistributedTaskScheduler(config)
    
    # Submit sample task
    request = TaskRequest(
        payload={"type": "data_processing"},
        priority=0,
        resource_limits=ResourceProfile(cpu=2.0, memory=4.0)
    )
    state = await scheduler.submit_task(request)
    print(f"Scheduled task {state.task_id}")
    
    # Run indefinitely
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    k8s.config.load_kube_config()
    asyncio.run(main())
