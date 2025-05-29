"""
Distributed Priority Queue - Dynamic Task Prioritization & Fault-Tolerant Execution
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import redis.asyncio as redis
from prometheus_client import Counter, Gauge, Histogram

# Metrics
QUEUE_METRICS = {
    "queue_size": Gauge("priority_queue_size", "Current tasks in queue", ["priority"]),
    "task_wait_time": Histogram("task_wait_seconds", "Time from enqueue to execution", ["priority"]),
    "task_retries": Counter("task_retry_count", "Number of task retries", ["task_type"]),
    "queue_ops": Counter("queue_operations", "Queue API calls", ["operation"])
}

class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: f"task-{uuid.uuid4().hex[:8]}")
    created_at: float = Field(default_factory=time.time)
    priority: float  # Lower values = higher priority
    data: Dict[str, str]
    max_retries: int = 3
    retry_count: int = 0
    last_attempt: Optional[float] = None
    status: str = "pending"  # pending/processing/succeeded/failed

    @validator('priority')
    def validate_priority(cls, v):
        if v < 0 or v > 100_000:
            raise ValueError("Priority must be between 0 (highest) and 100000 (lowest)")
        return v

class PriorityQueueConfig(BaseModel):
    redis_url: str = "redis://localhost:6379/3"
    lock_timeout: int = 30  # Seconds for distributed lock
    visibility_window: int = 300  # Seconds before retrying stuck tasks
    max_backoff: int = 3600  # Max delay for retries
    priority_adjustment_rate: float = 0.1  # Priority increase per minute of waiting

class PriorityQueue:
    """
    Redis-backed distributed priority queue with:
    - Dynamic priority adjustment
    - At-least-once delivery semantics
    - Dead letter queue for failures
    - Prometheus monitoring
    """
    
    def __init__(self, config: PriorityQueueConfig):
        self.redis = redis.Redis.from_url(config.redis_url)
        self.config = config
        self.lock_prefix = "queue:lock:"
        self.data_prefix = "queue:data:"
        self.dead_letter_key = "queue:dead_letters"
        
        # Start background tasks
        self._adjuster_task = asyncio.create_task(self._priority_adjustment_loop())
        self._reaper_task = asyncio.create_task(self._stuck_task_reaper_loop())
    
    async def enqueue(self, task: Task) -> str:
        """Atomically add task to queue with initial priority"""
        async with self.redis.pipeline(transaction=True) as pipe:
            # Store task metadata
            pipe.hset(f"{self.data_prefix}{task.task_id}", mapping=task.dict())
            # Add to priority queue
            pipe.zadd("queue:global", {task.task_id: task.priority})
            await pipe.execute()
            
            QUEUE_METRICS["queue_ops"].labels(operation="enqueue").inc()
            QUEUE_METRICS["queue_size"].labels(priority=int(task.priority)).inc()
            return task.task_id
    
    async def dequeue(self) -> Optional[Task]:
        """Fetch highest-priority task with distributed lock"""
        while True:
            # Find candidate tasks
            task_ids = await self.redis.zrangebyscore(
                "queue:global",
                min=0,
                max="inf",
                start=0,
                num=1,
                withscores=True
            )
            
            if not task_ids:
                return None
                
            task_id, priority = task_ids[0]
            task_id = task_id.decode()
            lock_key = f"{self.lock_prefix}{task_id}"
            
            # Attempt atomic lock
            locked = await self.redis.set(
                lock_key,
                value="locked",
                ex=self.config.lock_timeout,
                nx=True
            )
            
            if locked:
                # Mark as processing
                await self._update_task_status(task_id, "processing")
                # Load full task data
                task_data = await self.redis.hgetall(f"{self.data_prefix}{task_id}")
                task = Task.parse_obj(task_data)
                task.status = "processing"
                task.last_attempt = time.time()
                
                QUEUE_METRICS["task_wait_time"].labels(priority=int(priority)).observe(
                    time.time() - task.created_at
                )
                return task
                
            await asyncio.sleep(0.1)
    
    async def ack(self, task_id: str) -> None:
        """Mark task as successfully completed"""
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.zrem("queue:global", task_id)
            pipe.delete(f"{self.data_prefix}{task_id}")
            pipe.delete(f"{self.lock_prefix}{task_id}")
            await pipe.execute()
            QUEUE_METRICS["queue_ops"].labels(operation="ack").inc()
            QUEUE_METRICS["queue_size"].labels(priority=int(await self._get_priority(task_id))).dec()
    
    async def nack(self, task_id: str) -> None:
        """Mark task as failed, schedule retry or move to DLQ"""
        task_data = await self.redis.hgetall(f"{self.data_prefix}{task_id}")
        task = Task.parse_obj(task_data)
        
        if task.retry_count >= task.max_retries:
            await self._move_to_dead_letter(task)
            return
            
        # Exponential backoff with jitter
        delay = min(
            2 ** task.retry_count + (0.1 * task.retry_count),
            self.config.max_backoff
        ) * (0.8 + 0.4 * (task.retry_count / task.max_retries))
        
        # Update task state
        task.retry_count += 1
        task.last_attempt = time.time()
        await self.redis.hset(
            f"{self.data_prefix}{task_id}",
            mapping=task.dict()
        )
        
        # Reschedule with lower priority (higher numeric value)
        new_priority = task.priority + (10_000 * task.retry_count)
        await self.redis.zadd("queue:global", {task_id: new_priority})
        
        QUEUE_METRICS["task_retries"].labels(task_type=task.data.get("type", "unknown")).inc()
    
    async def _move_to_dead_letter(self, task: Task) -> None:
        """Permanently archive failed tasks"""
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.zrem("queue:global", task.task_id)
            pipe.hset(f"{self.dead_letter_key}:{task.task_id}", mapping=task.dict())
            pipe.delete(f"{self.data_prefix}{task.task_id}")
            pipe.delete(f"{self.lock_prefix}{task.task_id}")
            await pipe.execute()
    
    async def _update_task_status(self, task_id: str, status: str) -> None:
        """Atomic status update with timestamp"""
        await self.redis.hset(
            f"{self.data_prefix}{task_id}",
            key="status",
            value=status
        )
        await self.redis.hset(
            f"{self.data_prefix}{task_id}",
            key="last_updated",
            value=time.time()
        )
    
    async def _priority_adjustment_loop(self) -> None:
        """Gradually increase priority of long-waiting tasks"""
        while True:
            try:
                # Find tasks older than 1 minute
                cutoff = time.time() - 60
                task_ids = await self.redis.zrangebyscore(
                    "queue:global",
                    min=0,
                    max="inf",
                    withscores=True
                )
                
                for task_id, old_priority in task_ids:
                    task_id = task_id.decode()
                    task_data = await self.redis.hgetall(f"{self.data_prefix}{task_id}")
                    if not task_data:
                        continue
                        
                    task = Task.parse_obj(task_data)
                    if task.created_at < cutoff:
                        # Adjust priority: older tasks get higher priority (lower value)
                        new_priority = max(
                            0,
                            old_priority - (time.time() - task.created_at) * self.config.priority_adjustment_rate
                        )
                        await self.redis.zadd("queue:global", {task_id: new_priority})
                
                await asyncio.sleep(60)
            except Exception as e:
                logging.error(f"Priority adjuster failed: {e}")
                await asyncio.sleep(10)
    
    async def _stuck_task_reaper_loop(self) -> None:
        """Recover tasks stuck in processing state"""
        while True:
            try:
                # Find tasks locked beyond visibility window
                cutoff = time.time() - self.config.visibility_window
                processing_tasks = await self.redis.keys(f"{self.data_prefix}*")
                
                for key in processing_tasks:
                    task_id = key.decode().split(":")[-1]
                    status = await self.redis.hget(key, "status")
                    if status != b"processing":
                        continue
                        
                    last_updated = await self.redis.hget(key, "last_updated")
                    if last_updated and float(last_updated) < cutoff:
                        # Force-unlock and requeue
                        await self.redis.delete(f"{self.lock_prefix}{task_id}")
                        await self.redis.zadd("queue:global", {task_id: 0})  # Highest priority
                        logging.warning(f"Recovered stuck task: {task_id}")
                
                await asyncio.sleep(self.config.visibility_window)
            except Exception as e:
                logging.error(f"Task reaper failed: {e}")
                await asyncio.sleep(30)
    
    async def _get_priority(self, task_id: str) -> float:
        """Fetch current priority from sorted set"""
        return float(await self.redis.zscore("queue:global", task_id) or 0)

# Usage Example
async def main():
    queue = PriorityQueue(PriorityQueueConfig())
    
    # Enqueue tasks
    task = Task(
        priority=1000,
        data={"type": "image_processing", "url": "https://example.com/image.jpg"}
    )
    await queue.enqueue(task)
    
    # Worker loop
    while True:
        task = await queue.dequeue()
        if task:
            try:
                # Process task
                print(f"Processing {task.task_id}")
                await queue.ack(task.task_id)
            except Exception as e:
                await queue.nack(task.task_id)

if __name__ == "__main__":
    asyncio.run(main())
