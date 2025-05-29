"""
Agent Registry - Centralized Service Discovery & State Management
"""

from __future__ import annotations
import asyncio
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Type, Any, Set
import uuid
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from sortedcontainers import SortedDict

# Custom Types
AgentID = str
AgentType = str
NodeID = str

class AgentRecord(BaseModel):
    """Immutable agent registration record"""
    agent_id: AgentID
    agent_type: AgentType
    node_id: NodeID
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "active"  # active | draining | terminated
    
    @validator('agent_id')
    def validate_agent_id(cls, v):
        if not v.startswith("agent-"):
            raise ValueError("AgentID must start with 'agent-'")
        return v

class RegistryConfig(BaseModel):
    """Dynamic registry configuration"""
    sync_interval: int = 5  # Seconds between syncs
    heartbeat_ttl: int = 30  # Seconds before marking offline
    max_local_cache_size: int = 1000
    storage_backend: str = "redis"  # redis | inmemory | consul
    redis_url: str = "redis://localhost:6379/0"
    
    @validator('storage_backend')
    def validate_backend(cls, v):
        if v not in {"redis", "inmemory", "consul"}:
            raise ValueError("Invalid storage backend")
        return v

class AgentRegistry:
    """
    Distributed agent registry with tiered caching
    
    Architecture:
    - L1: Local in-memory cache (LRU)
    - L2: Redis cluster (persistent)
    - Watch: Etcd-style change notifications
    """
    
    _instance: Optional[AgentRegistry] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.config = RegistryConfig()
        self._local_cache: SortedDict = SortedDict()
        self._redis: Optional[redis.Redis] = None
        self._watch_tasks: Set[asyncio.Task] = set()
        self._lock = asyncio.Lock()
        self._pubsub: Optional[redis.PubSub] = None
        self.logger = logging.getLogger("AgentRegistry")
        
        if self.config.storage_backend == "redis":
            self._redis = redis.from_url(self.config.redis_url)
        
        # Background tasks
        self._sync_task = asyncio.create_task(self._synchronize_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def register(self, agent: AgentRecord) -> None:
        """Atomic registration with CAS check"""
        async with self._lock:
            # Check existing registration
            existing = await self.get(agent.agent_id)
            if existing and existing.status != "terminated":
                raise ValueError(f"Agent {agent.agent_id} already registered")
            
            # Write-through to L2
            if self._redis:
                pipeline = self._redis.pipeline()
                pipeline.hset(
                    f"agent:{agent.agent_id}",
                    mapping=agent.dict()
                )
                pipeline.zadd(
                    "agent:heartbeats",
                    {agent.agent_id: agent.last_heartbeat.timestamp()}
                )
                await pipeline.execute()
            
            # Update L1
            self._update_local_cache(agent)
            
            # Publish registration event
            await self._publish_event("register", agent)
    
    async def deregister(self, agent_id: AgentID) -> None:
        """Safe deregistration with state marking"""
        async with self._lock:
            agent = await self.get(agent_id)
            if not agent:
                return
                
            agent.status = "terminated"
            
            if self._redis:
                await self._redis.hset(
                    f"agent:{agent_id}", 
                    "status", 
                    "terminated"
                )
            
            self._update_local_cache(agent)
            await self._publish_event("deregister", agent)
    
    async def get(self, agent_id: AgentID) -> Optional[AgentRecord]:
        """Multi-layer read with cache warming"""
        # L1 Check
        if agent_id in self._local_cache:
            return self._local_cache[agent_id]
        
        # L2 Check
        if self._redis:
            agent_data = await self._redis.hgetall(f"agent:{agent_id}")
            if agent_data:
                agent = AgentRecord(**agent_data)
                self._update_local_cache(agent)
                return agent
        
        return None
    
    async def find(
        self, 
        agent_type: Optional[AgentType] = None,
        status: str = "active",
        node_id: Optional[NodeID] = None
    ) -> List[AgentRecord]:
        """Distributed query with predicate pushdown"""
        # Redis-side filtering
        if self._redis:
            lua_script = """
                local results = {}
                local keys = redis.call('ZRANGE', 'agent:heartbeats', 0, -1)
                for _, key in ipairs(keys) do
                    local agent = redis.call('HGETALL', 'agent:'..key)
                    if #agent > 0 then
                        local fields = {}
                        for i=1,#agent,2 do fields[agent[i]] = agent[i+1] end
                        if (ARGV[1] == '' or fields['agent_type'] == ARGV[1]) 
                        and (ARGV[2] == '' or fields['status'] == ARGV[2])
                        and (ARGV[3] == '' or fields['node_id'] == ARGV[3]) then
                            table.insert(results, fields)
                        end
                    end
                end
                return results
            """
            args = [agent_type or "", status, node_id or ""]
            results = await self._redis.eval(
                lua_script, 
                keys=[], 
                args=args
            )
            return [AgentRecord(**r) for r in results]
        
        # Fallback to local cache
        return [
            agent for agent in self._local_cache.values()
            if (agent_type is None or agent.agent_type == agent_type) 
            and agent.status == status
            and (node_id is None or agent.node_id == node_id)
        ]
    
    async def update_heartbeat(self, agent_id: AgentID) -> None:
        """Optimized heartbeat update with coalescing"""
        async with self._lock:
            now = datetime.utcnow()
            if self._redis:
                await self._redis.zadd(
                    "agent:heartbeats",
                    {agent_id: now.timestamp()},
                    gt=True  # Only update if timestamp is newer
                )
            if agent_id in self._local_cache:
                self._local_cache[agent_id].last_heartbeat = now
    
    def _update_local_cache(self, agent: AgentRecord) -> None:
        """LRU cache update with size control"""
        if agent.agent_id in self._local_cache:
            del self._local_cache[agent.agent_id]
        self._local_cache[agent.agent_id] = agent
        
        if len(self._local_cache) > self.config.max_local_cache_size:
            oldest_key = next(iter(self._local_cache))
            del self._local_cache[oldest_key]
    
    async def _synchronize_loop(self) -> None:
        """Periodic L1/L2 cache synchronization"""
        while True:
            try:
                if self._redis:
                    # Refresh local cache with recent heartbeats
                    cutoff = datetime.utcnow() - timedelta(
                        seconds=self.config.heartbeat_ttl
                    )
                    active_agents = await self._redis.zrangebyscore(
                        "agent:heartbeats",
                        min=cutoff.timestamp(),
                        max="+inf"
                    )
                    for agent_id in active_agents:
                        await self.get(agent_id)  # Cache warm
            except Exception as e:
                self.logger.error(f"Sync error: {e}")
            await asyncio.sleep(self.config.sync_interval)
    
    async def _cleanup_loop(self) -> None:
        """Remove expired agents from registry"""
        while True:
            try:
                if self._redis:
                    cutoff = datetime.utcnow() - timedelta(
                        seconds=self.config.heartbeat_ttl * 2
                    )
                    expired = await self._redis.zrangebyscore(
                        "agent:heartbeats",
                        min="-inf",
                        max=cutoff.timestamp()
                    )
                    if expired:
                        await self._redis.delete(*[f"agent:{id}" for id in expired])
                        await self._redis.zrem("agent:heartbeats", *expired)
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(self.config.heartbeat_ttl)
    
    async def watch_events(self, callback: callable) -> None:
        """Subscribe to registry change notifications"""
        if not self._pubsub and self._redis:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe("registry_events")
        
        async def listener():
            async for message in self._pubsub.listen():
                if message['type'] == 'message':
                    event = message['data']
                    await callback(event)
        
        task = asyncio.create_task(listener())
        self._watch_tasks.add(task)
        task.add_done_callback(self._watch_tasks.discard)
    
    async def _publish_event(self, event_type: str, agent: AgentRecord) -> None:
        """Publish registry changes to subscribers"""
        if self._redis:
            await self._redis.publish(
                "registry_events",
                f"{event_type}:{agent.json()}"
            )
    
    async def shutdown(self) -> None:
        """Graceful registry termination"""
        self._sync_task.cancel()
        self._cleanup_task.cancel()
        if self._redis:
            await self._redis.close()
        for task in self._watch_tasks:
            task.cancel()

# Redis-backed registry with cluster support
class ClusterAgentRegistry(AgentRegistry):
    """Extension for multi-node Redis cluster"""
    
    def _initialize(self):
        super()._initialize()
        if self.config.storage_backend == "redis":
            self._redis = redis.RedisCluster.from_url(
                self.config.redis_url,
                decode_responses=False
            )

# Usage Example
async def main():
    registry = AgentRegistry()
    
    agent = AgentRecord(
        agent_id="agent-001",
        agent_type="worker",
        node_id="node-1",
        metadata={"capacity": 10}
    )
    
    await registry.register(agent)
    
    async def event_handler(event):
        print(f"Registry event: {event}")
    
    await registry.watch_events(event_handler)
    
    # Simulate heartbeat
    await registry.update_heartbeat("agent-001")
    
    # Query agents
    workers = await registry.find(agent_type="worker")
    print(f"Active workers: {workers}")

if __name__ == "__main__":
    asyncio.run(main())
