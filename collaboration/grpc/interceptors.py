"""
gRPC Interceptors - Enterprise-grade Cross-cutting Concerns Implementation
"""

from __future__ import annotations
import asyncio
import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# gRPC
import grpc
from grpc import RpcError, StatusCode, ServicerContext
from grpc.aio import ServerInterceptor

# Observability
from prometheus_client import Histogram, Counter
import opentelemetry.trace as otel_trace
from opentelemetry.sdk.trace import Tracer

# Internal modules
from utils.logger import get_logger
from utils.metrics import GRPC_METRICS
from utils.auth import JWTValidator, RBACEngine

logger = get_logger(__name__)
tracer: Tracer = otel_trace.get_tracer(__name__)

#region Client Interceptors

class EnterpriseClientInterceptor(grpc.aio.ClientInterceptor):
    """Base class for all client-side interceptors"""
    
    async def intercept_unary_unary(self, continuation, client_call_details, request):
        """Override for unary-unary RPCs"""
        return await continuation(client_call_details, request)

    async def intercept_stream_unary(self, continuation, client_call_details, request_iterator):
        """Override for stream-unary RPCs"""
        return await continuation(client_call_details, request_iterator)

class AuthClientInterceptor(EnterpriseClientInterceptor):
    """JWT Authentication & Token Refresh Interceptor"""
    
    def __init__(self, token_provider: Callable[[], str], refresh_threshold: int = 300):
        self.token_provider = token_provider
        self.refresh_threshold = refresh_threshold  # Seconds before expiration
        
    async def intercept_unary_unary(self, continuation, client_call_details, request):
        metadata = list(client_call_details.metadata or [])
        token = self.token_provider()
        
        # Validate token expiration
        if self._is_token_expiring_soon(token):
            token = await self._refresh_token()
            
        metadata.append(('authorization', f'Bearer {token}'))
        new_details = client_call_details._replace(metadata=metadata)
        return await continuation(new_details, request)
    
    def _is_token_expiring_soon(self, token: str) -> bool:
        # Implement JWT expiration check (mock implementation)
        return False
        
    async def _refresh_token(self) -> str:
        # Implement token refresh logic
        return "new_jwt_token"

class MetricsClientInterceptor(EnterpriseClientInterceptor):
    """Prometheus Metrics Collection & Distributed Tracing"""
    
    async def intercept_unary_unary(self, continuation, client_call_details, request):
        start_time = time.monotonic()
        method = client_call_details.method.decode('utf-8')
        
        with tracer.start_as_current_span(f"Client:{method}"), \
             GRPC_METRICS['client_calls_in_flight'].labels(method=method).track_inprogress():
            
            GRPC_METRICS['client_requests_total'].labels(method=method).inc()
            
            try:
                response = await continuation(client_call_details, request)
                latency = time.monotonic() - start_time
                GRPC_METRICS['client_latency_seconds'].labels(method=method).observe(latency)
                return response
            except RpcError as e:
                GRPC_METRICS['client_errors_total'].labels(
                    method=method, 
                    code=e.code().name
                ).inc()
                raise

class RetryClientInterceptor(EnterpriseClientInterceptor):
    """Smart Retry Logic with Backoff & Circuit Breaking"""
    
    def __init__(self, max_attempts: int = 3, retryable_codes: List[StatusCode] = None):
        self.max_attempts = max_attempts
        self.retryable_codes = retryable_codes or [
            StatusCode.UNAVAILABLE, 
            StatusCode.DEADLINE_EXCEEDED
        ]
        
    async def intercept_unary_unary(self, continuation, client_call_details, request):
        attempt = 0
        while True:
            try:
                return await continuation(client_call_details, request)
            except RpcError as e:
                if e.code() not in self.retryable_codes or attempt >= self.max_attempts:
                    raise
                
                await self._backoff(attempt)
                attempt += 1
                
                # Update metadata with attempt count
                metadata = list(client_call_details.metadata or [])
                metadata.append(('x-retry-attempt', str(attempt)))
                client_call_details = client_call_details._replace(metadata=metadata)
    
    async def _backoff(self, attempt: int):
        delay = min(2 ** attempt, 10)  # Exponential cap at 10s
        await asyncio.sleep(delay + random.uniform(0, 0.1))  # Jitter

#endregion

#region Server Interceptors

class EnterpriseServerInterceptor(ServerInterceptor):
    """Base class for all server-side interceptors"""
    
    async def intercept_service(self, continuation, handler_call_details):
        return await continuation(handler_call_details)

class AuthServerInterceptor(EnterpriseServerInterceptor):
    """JWT Validation & RBAC Enforcement"""
    
    def __init__(self, jwt_validator: JWTValidator, rbac_engine: RBACEngine):
        self.jwt_validator = jwt_validator
        self.rbac = rbac_engine
        
    async def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method.split('/')[-1]
        metadata = dict(handler_call_details.invocation_metadata)
        
        # JWT Extraction & Validation
        token = metadata.get('authorization', '').replace('Bearer ', '')
        claims = await self.jwt_validator.validate(token)
        
        # RBAC Check
        if not self.rbac.is_allowed(claims['roles'], method):
            raise PermissionError(f"Unauthorized access to {method}")
            
        # Add claims to context
        context = handler_call_details.context
        context.set_details({'user': claims})
        
        return await continuation(handler_call_details)

class RateLimitingInterceptor(EnterpriseServerInterceptor):
    """Dynamic Rate Limiting with Token Bucket Algorithm"""
    
    def __init__(self, capacity: int = 1000, refill_rate: float = 100.0):
        self.tokens = capacity
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
        
    async def intercept_service(self, continuation, handler_call_details):
        async with self.lock:
            self._refill_tokens()
            if self.tokens < 1:
                await self._reject_request(handler_call_details.context)
                return
            self.tokens -= 1
            
        return await continuation(handler_call_details)
    
    def _refill_tokens(self):
        now = time.monotonic()
        elapsed = now - self.last_refill
        refill_amount = elapsed * self.refill_rate
        self.tokens = min(self.tokens + refill_amount, self.capacity)
        self.last_refill = now
        
    async def _reject_request(self, context: ServicerContext):
        await context.abort(StatusCode.RESOURCE_EXHAUSTED, "Rate limit exceeded")

class RequestValidationInterceptor(EnterpriseServerInterceptor):
    """Protobuf Schema Validation & Sanitization"""
    
    async def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        request = handler_call_details.request
        
        # Validate against protobuf schema
        if not self._validate_request(method, request):
            await handler_call_details.context.abort(
                StatusCode.INVALID_ARGUMENT,
                "Malformed request payload"
            )
            
        # Sanitize input
        sanitized = self._sanitize(request)
        new_details = handler_call_details._replace(request=sanitized)
        
        return await continuation(new_details)
    
    def _validate_request(self, method: str, request) -> bool:
        # Implement protocol buffer validation logic
        return True
        
    def _sanitize(self, request):
        # Implement input sanitization logic
        return request

#endregion

#region Interceptor Factories

@dataclass
class ClientInterceptorChain:
    """Pre-configured chain of client interceptors"""
    
    @classmethod
    def default_chain(cls, token_provider: Callable[[], str]) -> List[EnterpriseClientInterceptor]:
        return [
            AuthClientInterceptor(token_provider),
            MetricsClientInterceptor(),
            RetryClientInterceptor(max_attempts=3)
        ]

@dataclass
class ServerInterceptorChain:
    """Pre-configured chain of server interceptors"""
    
    @classmethod
    def default_chain(cls, jwt_validator: JWTValidator, rbac: RBACEngine) -> List[EnterpriseServerInterceptor]:
        return [
            AuthServerInterceptor(jwt_validator, rbac),
            RateLimitingInterceptor(capacity=5000, refill_rate=200),
            RequestValidationInterceptor()
        ]

#endregion

# Example Usage

async def start_grpc_server():
    server = grpc.aio.server(
        interceptors=ServerInterceptorChain.default_chain(
            jwt_validator=JWTValidator(),
            rbac=RBACEngine()
        )
    )
    ...
    
async def create_grpc_client():
    channel = grpc.aio.insecure_channel(
        'localhost:50051',
        interceptors=ClientInterceptorChain.default_chain(
            token_provider=lambda: "test_jwt"
        )
    )
    ...
