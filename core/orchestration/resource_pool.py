"""
Resource Pool Manager - Dynamic Allocation & Autoscaling for Agent Swarms
"""

from __future__ import annotations
import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import uuid
import json

import kubernetes_asyncio as k8s
from kubernetes_asyncio.client import V1Deployment, V1Pod
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field, validator
from redis.asyncio import Redis

# Metrics
RESOURCE_METRICS = {
    "allocated_cpu": Gauge("resource_cpu_allocated", "Allocated CPU cores"),
    "allocated_mem": Gauge("resource_mem_allocated_gb", "Allocated memory in GB"),
    "allocation_errors": Counter("resource_allocation_errors", "Failed allocation attempts", ["reason"]),
    "scale_events": Counter("resource_scale_events", "Cluster scaling events", ["action"])
}

class ResourcePoolConfig(BaseModel):
    resource_types: List[str] = ["cpu", "memory", "gpu"]
    default_quota: Dict[str, float] = {"cpu": 100.0, "memory": 200.0}  # Per tenant
    max_overcommit: Dict[str, float] = {"cpu": 2.0, "memory": 1.2}  # Overprovisioning ratios
    autoscale_interval: int = 30  # Seconds between scaling checks
    scale_up_threshold: float = 0.8  # 80% utilization triggers scale-up
    scale_down_threshold: float = 0.3  # 30% utilization triggers scale-down
    kube_namespace: str = "aelion-resources"
    redis_url: str = "redis://localhost:6379/2"

class ResourceRequest(BaseModel):
    tenant_id: str
    duration: timedelta = Field(default=timedelta(minutes=30), description="TTL for resource reservation")
    resources: Dict[str, float] = Field(..., example={"cpu": 4.0, "memory": 8.0})
    priority: int = Field(1, ge=1, le=3) 1:BestEffort, 2:Burstable, 3:Guaranteed
    preemptible: bool = True

    @validator('resources')
    def validate_resources(cls, v):
        allowed = ["cpu", "memory", "gpu"]
        for key in v:
            if key not in allowed:
                raise ValueError(f"Invalid resource type: {key}")
        return v

class ResourceAllocation(BaseModel):
    allocation_id: str = Field(default_factory=lambda: f"alloc-{uuid.uuid4().hex[:8]}")
    expires_at: datetime
    assigned_nodes: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)

class ResourceExhaustedError(Exception):
    """Raised when resource cannot be allocated"""

class ResourceConflictError(Exception):
    """Raised when preemption fails"""

@dataclass
class NodeResources:
    total: Dict[str, float]
    allocated: Dict[str, float]

class ResourcePool:
    """
    Manages hybrid resource allocation with:
    - Multi-tenant quota enforcement
    - Priority-based preemption
    - Kubernetes-aware autoscaling
    - Overcommitment controls
    """
    
    def __init__(self, config: ResourcePoolConfig):
        self.config = config
        self.redis = Redis.from_url(config.redis_url)
        self.k8s_api = k8s.client.CoreV1Api()
        self.k8s_autoscaling_api = k8s.client.AutoscalingV1Api()
        
        # State tracking
        self.node_resources: Dict[str, NodeResources] = {}
        self.allocations: Dict[str, ResourceAllocation] = {}
        
        # Background services
        self._sync_task = asyncio.create_task(self._sync_with_k8s_loop())
        self._scaler_task = asyncio.create_task(self._autoscaler_loop())
    
    async def initialize_pool(self):
        """Bootstrap resource data from Kubernetes"""
        nodes = await self.k8s_api.list_node()
        for node in nodes.items:
            self._parse_node_resources(node)
        
        # Load existing allocations from Redis
        async for key in self.redis.scan_iter("alloc:*"):
            data = await self.redis.get(key)
            alloc = ResourceAllocation.parse_raw(data)
            self.allocations[alloc.allocation_id] = alloc
    
    def _parse_node_resources(self, node: V1Pod) -> None:
        """Extract allocatable resources from Node spec"""
        allocatable = node.status.allocatable
        self.node_resources[node.metadata.name] = NodeResources(
            total={
                "cpu": float(allocatable["cpu"]),
                "memory": float(allocatable["memory"].rstrip("Ki")) / 1024 / 1024,  # Convert to GB
                "gpu": int(allocatable.get("nvidia.com/gpu", 0))
            },
            allocated={"cpu": 0.0, "memory": 0.0, "gpu": 0}
        )
    
    async def allocate(self, request: ResourceRequest) -> ResourceAllocation:
        """
        Atomic resource allocation with:
        - Quota enforcement
        - Priority-based preemption
        - Overcommit checks
        """
        async with self.redis.pipeline(transaction=True) as pipe:
            try:
                # Check tenant quota
                tenant_key = f"tenant:{request.tenant_id}:usage"
                current_usage = await pipe.get(tenant_key)
                if current_usage:
                    current_usage = json.loads(current_usage)
                    if any(
                        current_usage.get(res, 0) + request.resources.get(res, 0) > 
                        self.config.default_quota.get(res, 0) * self.config.max_overcommit.get(res, 1.0)
                        for res in request.resources
                    ):
                        RESOURCE_METRICS["allocation_errors"].labels(reason="quota_exceeded").inc()
                        raise ResourceExhaustedError("Tenant quota exceeded")
                
                # Find suitable nodes
                allocation = await self._find_allocation(request)
                
                # Record allocation
                pipe.multi()
                pipe.set(
                    f"alloc:{allocation.allocation_id}",
                    allocation.json(),
                    ex=int(request.duration.total_seconds())
                )
                pipe.incrby(
                    tenant_key,
                    json.dumps({res: val for res, val in request.resources.items()})
                )
                await pipe.execute()
                
                self.allocations[allocation.allocation_id] = allocation
                self._update_metrics(allocation, "add")
                return allocation
            
            except Exception as e:
                await pipe.reset()
                RESOURCE_METRICS["allocation_errors"].labels(reason=str(e)).inc()
                raise
    
    async def _find_allocation(self, request: ResourceRequest) -> ResourceAllocation:
        """Priority-aware resource placement algorithm"""
        # 1. Try normal allocation
        for node_name, node in self.node_resources.items():
            if self._can_allocate(node, request.resources):
                return self._create_allocation(node_name, request)
        
        # 2. Attempt preemption
        if request.priority == 3 and request.preemptible:  # Guaranteed QoS
            victim = await self._find_preemption_candidate(request.resources)
            if victim:
                await self.release(victim.allocation_id)
                return self._create_allocation(victim.assigned_nodes[0], request)
        
        # 3. Trigger immediate scale-up
        await self._scale_cluster(request.resources)
        raise ResourceExhaustedError("Insufficient cluster resources")
    
    def _can_allocate(self, node: NodeResources, resources: Dict[str, float]) -> bool:
        """Check if node has enough free capacity"""
        return all(
            (node.total[res] * self.config.max_overcommit.get(res, 1.0)) - node.allocated[res] >= val
            for res, val in resources.items()
        )
    
    def _create_allocation(self, node_name: str, request: ResourceRequest) -> ResourceAllocation:
        """Record resource assignment"""
        alloc = ResourceAllocation(
            expires_at=datetime.utcnow() + request.duration,
            assigned_nodes=[node_name],
            metadata={
                "tenant": request.tenant_id,
                "priority": str(request.priority),
                "preemptible": str(request.preemptible)
            }
        )
        # Update node state
        for res, val in request.resources.items():
            self.node_resources[node_name].allocated[res] += val
        return alloc
    
    async def release(self, allocation_id: str) -> None:
        """Release resources and update accounting"""
        if allocation_id not in self.allocations:
            return
        
        alloc = self.allocations.pop(allocation_id)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.delete(f"alloc:{allocation_id}")
            tenant_key = f"tenant:{alloc.metadata['tenant']}:usage"
            pipe.decrby(tenant_key, json.dumps(alloc.resources))
            await pipe.execute()
        
        # Free node resources
        for node_name in alloc.assigned_nodes:
            for res, val in alloc.resources.items():
                self.node_resources[node_name].allocated[res] -= val
        
        self._update_metrics(alloc, "remove")
    
    async def _autoscaler_loop(self) -> None:
        """Reactive cluster scaling based on utilization"""
        while True:
            try:
                utilization = await self._calculate_utilization()
                
                if utilization > self.config.scale_up_threshold:
                    await self._scale_cluster()
                    RESOURCE_METRICS["scale_events"].labels(action="scale_up").inc()
                elif utilization < self.config.scale_down_threshold:
                    await self._scale_down_cluster()
                    RESOURCE_METRICS["scale_events"].labels(action="scale_down").inc()
                
                await asyncio.sleep(self.config.autoscale_interval)
            
            except Exception as e:
                logging.error(f"Autoscaler error: {e}")
                await asyncio.sleep(10)
    
    async def _calculate_utilization(self) -> float:
        """Calculate cluster-wide CPU utilization for scaling decisions"""
        total_cpu = sum(n.total["cpu"] for n in self.node_resources.values())
        allocated_cpu = sum(n.allocated["cpu"] for n in self.node_resources.values())
        return allocated_cpu / total_cpu if total_cpu > 0 else 0.0
    
    async def _scale_cluster(self, requested_resources: Dict[str, float] = None) -> None:
        """Intelligent scaling based on pending demand"""
        # Get current deployment
        dep = await self.k8s_api.read_namespaced_deployment(
            name="aelion-resource-pool",
            namespace=self.config.kube_namespace
        )
        current_replicas = dep.spec.replicas or 0
        
        # Calculate needed capacity
        if requested_resources:
            node_capacity = next(iter(self.node_resources.values())).total
            needed_nodes = max(
                [requested_resources[res] / node_capacity[res] for res in requested_resources]
            )
            new_replicas = current_replicas + int(needed_nodes) + 1  # Buffer
        else:
            new_replicas = current_replicas + 1
        
        # Apply scaling
        patch = {"spec": {"replicas": new_replicas}}
        await self.k8s_api.patch_namespaced_deployment(
            name="aelion-resource-pool",
            namespace=self.config.kube_namespace,
            body=patch
        )
    
    async def _scale_down_cluster(self) -> None:
        """Safe scale-down procedure"""
        # Get underutilized nodes
        candidates = [
            node_name for node_name, node in self.node_resources.items()
            if (node.allocated["cpu"] / node.total["cpu"]) < 0.1
        ]
        
        if candidates:
            # Drain node (k8s integration required)
            await self._drain_node(candidates[0])
            
            # Scale down deployment
            dep = await self.k8s_api.read_namespaced_deployment(
                name="aelion-resource-pool",
                namespace=self.config.kube_namespace
            )
            patch = {"spec": {"replicas": dep.spec.replicas - 1}}
            await self.k8s_api.patch_namespaced_deployment(
                name="aelion-resource-pool",
                namespace=self.config.kube_namespace,
                body=patch
            )
    
    async def _drain_node(self, node_name: str) -> None:
        """Safely evacuate allocations from a node"""
        # Find allocations on the node
        to_migrate = [
            alloc for alloc in self.allocations.values()
            if node_name in alloc.assigned_nodes
        ]
        
        # Attempt migration
        for alloc in to_migrate:
            try:
                await self.release(alloc.allocation_id)
                new_alloc = await self.allocate(ResourceRequest(
                    tenant_id=alloc.metadata["tenant"],
                    resources=alloc.resources,
                    priority=int(alloc.metadata["priority"]),
                    preemptible=alloc.metadata["preemptible"] == "True"
                ))
                logging.info(f"Migrated allocation {alloc.allocation_id} -> {new_alloc.allocation_id}")
            except ResourceExhaustedError:
                logging.warning(f"Failed to migrate allocation {alloc.allocation_id}")
    
    async def _sync_with_k8s_loop(self) -> None:
        """Periodically sync cluster state with Kubernetes"""
        while True:
            try:
                await self.initialize_pool()
                await asyncio.sleep(60)
            except Exception as e:
                logging.error(f"Sync error: {e}")
                await asyncio.sleep(30)
    
    def _update_metrics(self, alloc: ResourceAllocation, action: str) -> None:
        """Update Prometheus metrics"""
        sign = 1 if action == "add" else -1
        for res, val in alloc.resources.items():
            if res == "cpu":
                RESOURCE_METRICS["allocated_cpu"].inc(val * sign)
            elif res == "memory":
                RESOURCE_METRICS["allocated_mem"].inc(val * sign)

# Usage Example
async def main():
    k8s.config.load_kube_config()
    
    config = ResourcePoolConfig(
        default_quota={"cpu": 50, "memory": 100},
        max_overcommit={"cpu": 1.5, "memory": 1.2}
    )
    pool = ResourcePool(config)
    await pool.initialize_pool()
    
    # Allocate resources
    request = ResourceRequest(
        tenant_id="tenant-123",
        resources={"cpu": 4, "memory": 8},
        priority=3
    )
    alloc = await pool.allocate(request)
    print(f"Allocated resources: {alloc.allocation_id}")
    
    # Cleanup after 30m
    await asyncio.sleep(1800)
    await pool.release(alloc.allocation_id)

if __name__ == "__main__":
    asyncio.run(main())
