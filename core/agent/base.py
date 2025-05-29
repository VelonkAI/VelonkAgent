"""
Base Agent Class - Core Abstraction for All AI Agents
"""

from __future__ import annotations
import abc
import asyncio
import logging
from typing import TypeGuard, Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel, BaseSettings, ValidationError
from numpy.typing import NDArray
import numpy as np

# Type Aliases
AgentID = str
MessagePayload = Dict[str, Any]
T = TypeVar('T', bound='BaseAgent')

class AgentConfig(BaseSettings):
    """Dynamic configuration for agent instances"""
    heartbeat_interval: int = 5
    max_retries: int = 3
    policy_engine_endpoint: str = "http://rl-engine:8080"
    enable_tracing: bool = True

    class Config:
        env_prefix = "AGENT_"
        case_sensitive = False

class AgentMessage(BaseModel):
    """Standardized message schema for inter-agent communication"""
    sender: AgentID
    receiver: AgentID
    payload_type: str
    payload: MessagePayload
    timestamp: float = None  # Populated on send
    trace_id: str = None     # For distributed tracing

class AgentNetworkError(Exception):
    """Custom exception for communication failures"""
    def __init__(self, url: str, status: int):
        super().__init__(f"Network error: {url} returned {status}")
        self.url = url
        self.status = status

class BaseAgent(abc.ABC):
    """
    Abstract base class defining agent lifecycle and core capabilities.
    
    Subclasses must implement:
    - _process_message()
    - _execute_policy()
    """

    _registry: Dict[AgentID, BaseAgent] = {}
    
    def __init__(self, agent_id: AgentID):
        self.agent_id = agent_id
        self.config = AgentConfig()
        self._message_queue = asyncio.Queue(maxsize=1000)
        self._is_running = False
        self._logger = logging.getLogger(f"agent.{agent_id}")
        self._current_state: NDArray = np.zeros((256,))  # State vector
        self._plugins = self._load_plugins()

        # Register instance
        self.__class__._registry[agent_id] = self

    @classmethod
    def get_agent(cls: Type[T], agent_id: AgentID) -> Optional[T]:
        """Retrieve agent instance from registry with type safety"""
        agent = cls._registry.get(agent_id)
        if isinstance(agent, cls):
            return agent
        return None

    async def start(self) -> None:
        """Start agent's main event loop"""
        self._is_running = True
        self._logger.info(f"Agent {self.agent_id} initializing...")
        
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._message_loop())
                tg.create_task(self._heartbeat())
                tg.create_task(self._sync_state())
        except ExceptionGroup as eg:
            self._logger.critical(f"Critical failure: {eg.exceptions}")
            await self.shutdown()

    async def _message_loop(self) -> None:
        """Core message processing loop"""
        while self._is_running:
            try:
                msg = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=self.config.heartbeat_interval
                )
                validated = self._validate_message(msg)
                response = await self._process_message(validated)
                await self._route_response(response)
            except asyncio.TimeoutError:
                continue  # Intentional periodic check
            except ValidationError as ve:
                self._logger.error(f"Invalid message: {ve.json()}")

    def _validate_message(self, raw_msg: dict) -> AgentMessage:
        """Validate incoming messages against schema"""
        try:
            return AgentMessage(**raw_msg)
        except ValidationError as ve:
            self._logger.error(f"Validation failed: {ve}")
            raise

    async def send_message(self, receiver_id: AgentID, payload: MessagePayload) -> None:
        """Thread-safe message sending with retry logic"""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver_id,
            payload_type=type(payload).__name__,
            payload=payload
        )
        
        for attempt in range(self.config.max_retries):
            try:
                receiver = self.__class__.get_agent(receiver_id)
                if receiver:
                    await receiver._message_queue.put(message.dict())
                    return
                else:
                    raise AgentNetworkError(receiver_id, 404)
            except AgentNetworkError as ane:
                if attempt == self.config.max_retries - 1:
                    self._logger.error(f"Failed to send after {attempt+1} attempts")
                    raise

    @abc.abstractmethod
    async def _process_message(self, message: AgentMessage) -> MessagePayload:
        """Process incoming message (implement in subclass)"""
        raise NotImplementedError

    @abc.abstractmethod
    async def _execute_policy(self, state: NDArray) -> NDArray:
        """Execute RL policy on current state (implement in subclass)"""
        raise NotImplementedError

    async def _heartbeat(self) -> None:
        """Periodic health reporting to orchestrator"""
        while self._is_running:
            await asyncio.sleep(self.config.heartbeat_interval)
            self._logger.debug("Sending heartbeat...")
            # TODO: Implement actual health reporting

    async def _sync_state(self) -> None:
        """Synchronize state with other agents"""
        while self._is_running:
            await asyncio.sleep(1)
            new_state = await self._execute_policy(self._current_state)
            self._current_state = new_state

    def _load_plugins(self) -> dict:
        """Dynamic plugin loader (extension point)"""
        # TODO: Implement plugin discovery/loading
        return {}

    async def shutdown(self) -> None:
        """Graceful termination procedure"""
        self._is_running = False
        self._logger.info("Shutting down...")
        # Cleanup resources
        self.__class__._registry.pop(self.agent_id, None)

    def __repr__(self) -> str:
        return f"<BaseAgent {self.agent_id} state_dim={self._current_state.shape}>"

# Example concrete implementation stub
class ConcreteAgent(BaseAgent):
    async def _process_message(self, message: AgentMessage) -> MessagePayload:
        """Example message handler"""
        return {"status": "ACK", "sender": self.agent_id}

    async def _execute_policy(self, state: NDArray) -> NDArray:
        """Example policy execution"""
        return np.clip(state + np.random.randn(*state.shape), -1, 1)
