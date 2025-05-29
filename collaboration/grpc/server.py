"""
gRPC Server Implementation - High-performance RPC Layer for Agent Communication
"""

from __future__ import annotations
import asyncio
import logging
import signal
from concurrent import futures
from typing import Any, Dict, List, Optional

# gRPC
import grpc
from grpc import aio
from grpc_reflection.v1alpha import reflection

# Protobuf
from aelion.proto import agent_pb2, agent_pb2_grpc, common_pb2

# Internal modules
from core.agent.registry import AgentRegistry
from orchestration.resource_pool import ResourcePool
from utils.logger import get_logger
from utils.metrics import GRPC_SERVER_METRICS

logger = get_logger(__name__)
SERVICE_NAME = 'aelion.agent.v1.AgentService'

class AgentServicer(agent_pb2_grpc.AgentServiceServicer):
    """gRPC service implementation for agent communication"""
    
    def __init__(self, registry: AgentRegistry, resource_pool: ResourcePool):
        super().__init__()
        self.registry = registry
        self.resource_pool = resource_pool
        self._shutting_down = False

    async def RegisterAgent(
        self,
        request: agent_pb2.AgentRegistrationRequest,
        context: grpc.ServicerContext
    ) -> common_pb2.RegistrationResponse:
        """Handle agent registration requests"""
        try:
            GRPC_SERVER_METRICS['requests_total'].labels(
                method='RegisterAgent', status='started'
            ).inc()
            
            # Validate TLS client certificate
            if not self._validate_client_cert(context):
                context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, 
                    "Invalid client certificate"
                )
            
            agent_id = await self.registry.register(
                agent_type=request.agent_type,
                capabilities=list(request.capabilities),
                resource_profile=request.resource_profile
            )
            
            return common_pb2.RegistrationResponse(
                agent_id=agent_id,
                lease_duration=300  # 5-minute lease
            )
        except Exception as e:
            logger.error("Registration failed", exc_info=True)
            GRPC_SERVER_METRICS['errors_total'].labels(
                method='RegisterAgent'
            ).inc()
            context.abort(grpc.StatusCode.INTERNAL, str(e))

    async def SubmitTask(
        self,
        request: agent_pb2.TaskRequest,
        context: grpc.ServicerContext
    ) -> agent_pb2.TaskAcknowledgement:
        """Process task submissions with priority handling"""
        GRPC_SERVER_METRICS['task_queue_size'].inc()
        
        if self._shutting_down:
            context.abort(
                grpc.StatusCode.UNAVAILABLE,
                "Server is shutting down"
            )
        
        # Validate task payload
        if not request.payload:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Empty task payload"
            )
        
        # Submit to priority queue
        try:
            await self.resource_pool.submit_task(
                priority=request.priority,
                payload=request.payload,
                deadline=request.deadline
            )
            return agent_pb2.TaskAcknowledgement(
                status=common_pb2.Status.OK,
                message_id=request.message_id
            )
        except Exception as e:
            logger.error("Task submission failed", exc_info=True)
            return agent_pb2.TaskAcknowledgement(
                status=common_pb2.Status.ERROR,
                error_message=str(e)
            )

    def _validate_client_cert(self, context: grpc.ServicerContext) -> bool:
        """mTLS certificate validation"""
        try:
            cert = context.auth_context().get('x509_common_name')
            return cert in self._trusted_certs
        except:
            return False

class GracefulExitInterceptor(aio.ServerInterceptor):
    """Interceptor for graceful shutdown handling"""
    
    async def intercept_service(self, continuation, handler_call_details):
        if self.server._shutting_down:
            raise grpc.RpcError(
                code=grpc.StatusCode.UNAVAILABLE,
                details='Server shutting down'
            )
        return await continuation(handler_call_details)

class MetricsInterceptor(aio.ServerInterceptor):
    """Prometheus metrics collection"""
    
    async def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method.split('/')[-1]
        GRPC_SERVER_METRICS['requests_in_flight'].labels(method=method).inc()
        start_time = time.time()
        
        try:
            response = await continuation(handler_call_details)
            latency = time.time() - start_time
            GRPC_SERVER_METRICS['request_latency_seconds'].labels(
                method=method
            ).observe(latency)
            return response
        finally:
            GRPC_SERVER_METRICS['requests_in_flight'].labels(method=method).dec()

async def serve(
    registry: AgentRegistry,
    resource_pool: ResourcePool,
    port: int = 50051,
    ssl_cert_path: str = None,
    ssl_key_path: str = None,
    max_workers: int = 10
) -> None:
    """Start gRPC server with enterprise-grade configuration"""
    server = aio.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        interceptors=(
            GracefulExitInterceptor(),
            MetricsInterceptor(),
        ),
        options=[
            ('grpc.so_reuseport', 1),
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),
        ]
    )
    
    # Add services
    service = AgentServicer(registry, resource_pool)
    agent_pb2_grpc.add_AgentServiceServicer_to_server(service, server)
    
    # Enable reflection and health checking
    SERVICE_NAMES = (
        agent_pb2.DESCRIPTOR.services_by_name['AgentService'].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)
    
    # Configure SSL
    if ssl_cert_path and ssl_key_path:
        with open(ssl_key_path, 'rb') as f:
            private_key = f.read()
        with open(ssl_cert_path, 'rb') as f:
            certificate_chain = f.read()
        server_credentials = grpc.ssl_server_credentials(
            [(private_key, certificate_chain)]
        )
        listen_addr = f'[::]:{port}'
        server.add_secure_port(listen_addr, server_credentials)
    else:
        server.add_insecure_port(f'[::]:{port}')
    
    # Graceful shutdown
    async def shutdown(signal, loop):
        logger.info("Starting graceful shutdown...")
        service._shutting_down = True
        await server.stop(5)
        loop.stop()
    
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(sig, loop)))
    
    await server.start()
    logger.info(f"gRPC server started on port {port}")
    await server.wait_for_termination()

if __name__ == "__main__":
    # Initialize dependencies
    registry = AgentRegistry()
    resource_pool = ResourcePool()
    
    # Start server with SSL
    asyncio.run(serve(
        registry=registry,
        resource_pool=resource_pool,
        port=50051,
        ssl_cert_path="/etc/ssl/certs/aelion.crt",
        ssl_key_path="/etc/ssl/private/aelion.key"
    ))
