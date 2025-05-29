"""
Supervisor Agent - Central Orchestrator for Agent Swarm Coordination
"""

from __future__ import annotations
import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import uuid
import numpy as np
import psutil
from pydantic import BaseModel, Field, validator
from .base import BaseAgent, AgentMessage, AgentID, AgentConfig, AgentRegistry
from .worker import WorkerAgent, WorkerMetrics, TaskResult, TaskRequest

# Custom Types
SwarmState = Dict[AgentID, np.ndarray]  # State vectors of all agents
PolicyVector = np.ndarray  # Output from RL policy network

class SupervisorConfig(AgentConfig):
    """Extended configuration for supervisor agents"""
    swarm_size_limit: int = Field(1000, gt=0)
    heartbeat_interval: int = 30  # seconds
    failure_threshold: int = 3  # Consecutive failures before remediation
    scheduling_algorithm: str = "rl_priority"  # Options: rr, priority, rl_priority
    resource_weights: Dict[str, float] = {"cpu": 1.0, "mem_gb": 0.5}
    
    @validator('scheduling_algorithm')
    def validate_algorithm(cls, v):
        allowed = ["rr", "priority", "rl_priority"]
        if v not in allowed:
            raise ValueError(f"Algorithm must be one of {allowed}")
        return v

class SwarmHealthReport(BaseModel):
    """Global swarm health metrics"""
    total_agents: int
    active_workers: int
    avg_cpu_util: float
    avg_mem_util: float
    pending_tasks: int
    dead_agents: List[AgentID]

class TaskAssignment(BaseModel):
    """Directive for task distribution"""
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}")
    worker_id: AgentID
    payload: Dict[str, Any]
    deadline: datetime
    priority: int = 1

class SupervisorAgent(BaseAgent):
    """
    Central coordination agent for swarm management
    
    Key Responsibilities:
    - Global state maintenance
    - RL-driven scheduling
    - Fault detection & recovery
    - Resource optimization
    - Swarm autoscaling
    """
    
    def __init__(self, agent_id: AgentID):
        super().__init__(agent_id)
        self.config = SupervisorConfig()
        self._swarm_state: SwarmState = {}
        self._task_queue = deque(maxlen=10000)
        self._failure_counts: Dict[AgentID, int] = {}
        self._policy_network = self._init_policy_network()
        self._last_heartbeat = datetime.utcnow()

    async def _process_message(self, message: AgentMessage) -> MessagePayload:
        """Handle swarm coordination messages"""
        if message.payload_type == "TaskResult":
            return await self._handle_task_result(TaskResult(**message.payload))
        elif message.payload_type == "WorkerMetrics":
            return await self._update_swarm_state(message.sender, message.payload)
        return {"status": "unhandled_message_type"}

    async def _handle_task_result(self, result: TaskResult) -> Dict[str, Any]:
        """Process task completion/failure events"""
        if not result.success:
            self._failure_counts[result.worker_id] = \
                self._failure_counts.get(result.worker_id, 0) + 1
            await self._trigger_remediation(result.worker_id)
        return {"action": "acknowledged"}

    async def _update_swarm_state(self, agent_id: AgentID, metrics: Dict) -> Dict:
        """Maintain real-time swarm state matrix"""
        state_vector = np.array([
            metrics["cpu_usage"],
            metrics["mem_usage_gb"],
            metrics["active_tasks"],
            metrics["queue_size"],
            datetime.utcnow().timestamp()
        ])
        self._swarm_state[agent_id] = state_vector
        return {"status": "state_updated"}

    async def _execute_policy(self, state: NDArray) -> NDArray:
        """Generate swarm-level coordination directives"""
        # Convert swarm state to policy input tensor
        state_tensor = np.stack(list(self._swarm_state.values()))
        async with self._policy_lock:
            policy_output = self._policy_network.predict(state_tensor)
        return policy_output

    def _init_policy_network(self) -> PolicyNetwork:
        """Initialize RL policy model (placeholder implementation)"""
        class MockPolicyNetwork:
            def predict(self, state: np.ndarray) -> np.ndarray:
                return np.random.rand(state.shape[0], 5)  # 5 actions per agent
        return MockPolicyNetwork()

    async def _coordinate_swarm(self) -> None:
        """Main coordination loop"""
        while self._is_running:
            # 1. Check swarm health
            health_report = self._generate_health_report()
            
            # 2. Execute RL policy
            policy_vector = await self._execute_policy(health_report)
            
            # 3. Dispatch tasks
            await self._dispatch_tasks(policy_vector)
            
            # 4. Handle autoscaling
            if len(self._swarm_state) < self.config.swarm_size_limit:
                await self._scale_swarm()
                
            # 5. Failure recovery
            await self._recover_failed_agents()
            
            await asyncio.sleep(1)

    async def _dispatch_tasks(self, policy_vector: PolicyVector) -> None:
        """Distribute tasks based on policy output"""
        for agent_id, actions in zip(self._swarm_state.keys(), policy_vector):
            if agent_id not in WorkerAgent.get_worker_metrics():
                continue
                
            # Decode policy actions
            task_capacity = int(actions[0] * 10)  # Max 10 tasks per dispatch
            for _ in range(task_capacity):
                if self._task_queue:
                    task = self._task_queue.popleft()
                    assignment = TaskAssignment(
                        worker_id=agent_id,
                        payload=task.payload,
                        deadline=datetime.utcnow() + timedelta(seconds=task.timeout)
                    )
                    await self._send_task_assignment(assignment)

    async def _send_task_assignment(self, assignment: TaskAssignment) -> None:
        """Direct task assignment to target worker"""
        try:
            await self._send_message(
                receiver=assignment.worker_id,
                payload_type="TaskAssignment",
                payload=assignment.dict()
            )
        except AgentNetworkError as e:
            self._logger.error(f"Failed to assign task {assignment.task_id}: {e}")
            self._task_queue.append(assignment)  # Requeue failed assignment

    async def _scale_swarm(self) -> None:
        """Autoscale worker agents based on load"""
        pending_tasks = len(self._task_queue)
        current_workers = len(WorkerAgent.get_worker_metrics())
        
        if pending_tasks > current_workers * 5:  # Scale-up threshold
            scale_count = min(
                (pending_tasks // 5) - current_workers,
                self.config.swarm_size_limit - current_workers
            )
            for _ in range(scale_count):
                worker_id = f"worker-{uuid.uuid4().hex[:8]}"
                await self._deploy_new_worker(worker_id)

    async def _deploy_new_worker(self, worker_id: AgentID) -> None:
        """Orchestrate new worker deployment (Kubernetes integration example)"""
        # TODO: Implement actual deployment logic
        worker = WorkerAgent(worker_id)
        self._registry.register(worker)
        asyncio.create_task(worker.start())

    async def _recover_failed_agents(self) -> None:
        """Handle agent failure recovery"""
        for agent_id, count in self._failure_counts.items():
            if count >= self.config.failure_threshold:
                await self._restart_agent(agent_id)
                self._failure_counts[agent_id] = 0

    async def _restart_agent(self, agent_id: AgentID) -> None:
        """Agent restart procedure"""
        self._logger.warning(f"Restarting agent {agent_id}")
        old_agent = self._registry.get(agent_id)
        if old_agent:
            await old_agent.shutdown()
            del self._swarm_state[agent_id]
        
        new_agent = WorkerAgent(agent_id)
        self._registry.register(new_agent)
        asyncio.create_task(new_agent.start())

    def _generate_health_report(self) -> SwarmHealthReport:
        """Generate system-wide health metrics"""
        worker_metrics = WorkerAgent.get_worker_metrics()
        return SwarmHealthReport(
            total_agents=len(self._swarm_state),
            active_workers=len(worker_metrics),
            avg_cpu_util=(
                sum(w.cpu_usage for w in worker_metrics.values()) 
                / len(worker_metrics) if worker_metrics else 0
            ),
            avg_mem_util=(
                sum(w.mem_usage_gb for w in worker_metrics.values()) 
                / len(worker_metrics) if worker_metrics else 0
            ),
            pending_tasks=len(self._task_queue),
            dead_agents=[
                aid for aid in self._swarm_state 
                if aid not in worker_metrics
            ]
        )

    async def submit_task(self, task: TaskRequest) -> str:
        """Public API for task submission"""
        self._task_queue.append(task)
        return task.task_id

    async def shutdown_swarm(self) -> None:
        """Graceful swarm shutdown"""
        for agent in self._registry.get_all_agents():
            if isinstance(agent, WorkerAgent):
                await agent.shutdown()
        await super().shutdown()

    @classmethod
    def get_global_health(cls) -> SwarmHealthReport:
        """Get current swarm health status"""
        supervisors = [
            agent for agent in cls._registry.values()
            if isinstance(agent, SupervisorAgent)
        ]
        if not supervisors:
            raise ValueError("No active supervisor")
        return supervisors[0]._generate_health_report()

# Kubernetes-enhanced Supervisor
class K8sSupervisor(SupervisorAgent):
    """Supervisor with Kubernetes cluster integration"""
    
    async def _deploy_new_worker(self, worker_id: AgentID) -> None:
        """Deploy workers using Kubernetes API"""
        from kubernetes import client, config  # Requires k8s SDK
        
        # Load cluster config
        config.load_incluster_config()
        api = client.AppsV1Api()
        
        # Create new worker deployment
        deployment = client.V1Deployment(
            metadata=client.V1ObjectMeta(name=f"aelion-worker-{worker_id}"),
            spec=client.V1DeploymentSpec(
                replicas=1,
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="worker",
                                image="aelionai/worker:latest",
                                env=[
                                    client.V1EnvVar(
                                        name="AGENT_ID", 
                                        value=worker_id
                                    )
                                ]
                            )
                        ]
                    )
                )
            )
        )
        
        api.create_namespaced_deployment(
            namespace="aelion", 
            body=deployment
        )
        self._logger.info(f"Deployed worker {worker_id} via Kubernetes")

    async def _restart_agent(self, agent_id: AgentID) -> None:
        """Kubernetes pod restart logic"""
        from kubernetes import client, config
        
        config.load_incluster_config()
        core_api = client.CoreV1Api()
        
        pods = core_api.list_namespaced_pod(
            namespace="aelion",
            label_selector=f"agent-id={agent_id}"
        )
        
        if pods.items:
            core_api.delete_namespaced_pod(
                name=pods.items[0].metadata.name,
                namespace="aelion"
            )
            self._logger.info(f"Restarted K8s pod for agent {agent_id}")
