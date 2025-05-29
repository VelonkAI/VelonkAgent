"""
Enterprise gRPC Client - Resilient & Adaptive Client for Agent Communication
"""

from __future__ import annotations
import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# gRPC
import grpc
from grpc import aio, ssl_channel_credentials

# Protobuf
from aelion.proto import agent_pb2, agent_pb2_grpc, common_pb2

# Internal modules
from utils.backoff import ExponentialBackoff
from utils.logger import get_logger
from utils.metrics import GRPC_CLIENT_METRICS

logger = get_logger(__name__)
MAX_MESSAGE_LENGTH = 100 * 1024 * 1024  # 100MB

@dataclass(frozen=True)
class ClientConfig:
    endpoints: List[str]  # Format: ["host1:port", "host2:port"]
    enable_ssl: bool = True
    cert_path: Optional[str] = None
    key_path: Optional[str] = None
    ca_path: Optional[str] = None
    jwt_token: Optional[str] = None
    max_retries: int = 5
    timeout: int = 30  # Seconds
    load_balance: str = "round_robin"  # round_robin|random|health_check

class AelionClient:
    """Smart gRPC client with enterprise features"""
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self._channel = None
        self._stub = None
        self._current_endpoint = None
        self._lb_index = 0
        self._backoff = ExponentialBackoff(
            max_retries=config.max_retries,
            max_delay=60
        )
        self._health_check_task = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def connect(self):
        """Establish optimized channel with load balancing"""
        target = self._select_endpoint()
        logger.debug(f"Connecting to {target}")
        
        # SSL Configuration
        credentials = None
        if self.config.enable_ssl:
            credentials = self._get_credentials()

        # Channel options
        options = [
            ('grpc.ssl_target_name_override', 'aelion-grpc'),
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.enable_retries', 1),
            ('grpc.service_config', 
             '{"loadBalancingConfig": [{"round_robin":{}}]}')
        ]

        # Create channel
        self._channel = aio.secure_channel(
            target,
            credentials,
            options=options
        ) if credentials else aio.insecure_channel(target, options=options)

        # Authentication headers
        if self.config.jwt_token:
            metadata_plugin = self._jwt_auth_metadata()
            self._stub = agent_pb2_grpc.AgentServiceStub(
                grpc.intercept_channel(
                    self._channel,
                    metadata_plugin
                )
            )
        else:
            self._stub = agent_pb2_grpc.AgentServiceStub(self._channel)

        # Start health check background task
        if "health_check" in self.config.load_balance:
            self._health_check_task = asyncio.create_task(
                self._health_monitor()
            )

    def _select_endpoint(self) -> str:
        """Load balancing endpoint selection"""
        strategy = self.config.load_balance
        if strategy == "random":
            return random.choice(self.config.endpoints)
        elif strategy == "health_check":
            return self._get_healthiest_endpoint()  # Implement health check
        else:  # round_robin
            self._lb_index = (self._lb_index + 1) % len(self.config.endpoints)
            return self.config.endpoints[self._lb_index]

    def _get_credentials(self) -> grpc.ChannelCredentials:
        """Build SSL credentials with mutual TLS support"""
        root_certs = open(self.config.ca_path, 'rb').read() if self.config.ca_path else None
        private_key = open(self.config.key_path, 'rb').read() if self.config.key_path else None
        cert_chain = open(self.config.cert_path, 'rb').read() if self.config.cert_path else None
        
        return ssl_channel_credentials(
            root_certificates=root_certs,
            private_key=private_key,
            certificate_chain=cert_chain
        )

    def _jwt_auth_metadata(
        self
    ) -> grpc.UnaryUnaryClientInterceptor:
        """JWT authentication interceptor"""
        def metadata(context, callback):
            callback([
                ('authorization', f'Bearer {self.config.jwt_token}'),
                ('x-client-id', 'aelion-python-sdk')
            ], None)
        
        return grpc.MetadataPlugin(metadata)

    async def _health_monitor(self):
        """Active health checking for load balancing"""
        while True:
            for endpoint in self.config.endpoints:
                try:
                    async with aio.insecure_channel(endpoint) as test_channel:
                        stub = agent_pb2_grpc.AgentServiceStub(test_channel)
                        await stub.CheckHealth(
                            common_pb2.HealthCheckRequest(),
                            timeout=5
                        )
                        self._mark_healthy(endpoint)
                except Exception as e:
                    self._mark_unhealthy(endpoint)
            await asyncio.sleep(30)

    async def register_agent(
        self,
        agent_type: str,
        capabilities: List[str],
        resource_profile: Dict[str, Any]
    ) -> str:
        """Agent registration with retry logic"""
        req = agent_pb2.AgentRegistrationRequest(
            agent_type=agent_type,
            capabilities=capabilities,
            resource_profile=str(resource_profile)
        )
        
        for attempt in self._backoff:
            try:
                GRPC_CLIENT_METRICS['requests_total'].labels(
                    method='RegisterAgent'
                ).inc()
                response = await self._stub.RegisterAgent(
                    req,
                    timeout=self.config.timeout,
                    metadata=(('x-retry-attempt', str(attempt)),)
                )
                return response.agent_id
            except grpc.RpcError as e:
                GRPC_CLIENT_METRICS['errors_total'].labels(
                    method='RegisterAgent',
                    code=e.code().name
                ).inc()
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    await self._reconnect()
                elif attempt == self.config.max_retries - 1:
                    raise

    async def submit_task(
        self,
        payload: bytes,
        priority: int = 0,
        deadline: Optional[datetime] = None
    ) -> str:
        """Task submission with deadline awareness"""
        req = agent_pb2.TaskRequest(
            payload=payload,
            priority=priority,
            deadline=int(deadline.timestamp()) if deadline else 0
        )
        
        try:
            start_time = datetime.now()
            response = await self._stub.SubmitTask(
                req,
                timeout=self.config.timeout
            )
            GRPC_CLIENT_METRICS['request_latency_seconds'].labels(
                method='SubmitTask'
            ).observe((datetime.now() - start_time).total_seconds())
            return response.message_id
        except grpc.RpcError as e:
            GRPC_CLIENT_METRICS['errors_total'].labels(
                method='SubmitTask',
                code=e.code().name
            ).inc()
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                logger.warning("Task submission timed out")
            raise

    async def _reconnect(self):
        """Reconnect with endpoint rotation"""
        await self.close()
        self._current_endpoint = None
        await self.connect()

    async def close(self):
        """Graceful shutdown"""
        if self._channel:
            await self._channel.close()
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

# Example usage
async def main():
    config = ClientConfig(
        endpoints=["grpc1.aelion.ai:443", "grpc2.aelion.ai:443"],
        cert_path="/path/to/client.crt",
        key_path="/path/to/client.key",
        ca_path="/path/to/ca.crt",
        jwt_token="eyJhbGciOi...",
        max_retries=3
    )
    
    async with AelionClient(config) as client:
        agent_id = await client.register_agent(
            agent_type="worker",
            capabilities=["nlp", "image_processing"],
            resource_profile={"gpu": 1}
        )
        print(f"Registered agent ID: {agent_id}")
        
        await client.submit_task(
            payload=b"task_data",
            priority=2,
            deadline=datetime.now() + timedelta(minutes=5)
        )

if __name__ == "__main__":
    asyncio.run(main())
