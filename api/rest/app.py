"""
Enterprise REST API - Unified Interface for Agent System Management & Monitoring
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

import jwt
import orjson
import uvicorn
from cryptography.hazmat.primitives import serialization
from fastapi import (
    APIRouter,
    Body,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
    Security,
    status,
)
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
from pydantic import BaseModel, BaseSettings, Field, ValidationError
from pydantic.json import pydantic_encoder
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette_context import context
from starlette_context.middleware import ContextMiddleware

from core.agent.registry import AgentRegistry
from grpc.client import GRPCClient
from kafka.producer import KafkaProducer
from utils.logger import Logger
from utils.metrics import MetricsSystem
from utils.serialization import SecureSerializer

class APIConfig(BaseSettings):
    env: str = Field("prod", env="API_ENV")
    port: int = Field(8000, env="API_PORT")
    workers: int = Field(os.cpu_count(), env="API_WORKERS")
    jwt_secret: str = Field(..., env="API_JWT_SECRET")
    rsa_public_key: str = Field(..., env="API_RSA_PUBKEY")
    enable_swagger: bool = Field(False, env="API_ENABLE_SWAGGER")
    rate_limit: str = Field("100/minute", env="API_RATE_LIMIT")
    cors_origins: List[str] = Field(["*"], env="API_CORS_ORIGINS")

    class Config:
        env_file = ".env"
        json_loads = orjson.loads
        json_dumps = orjson.dumps

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        limiter = request.app.state.limiter
        identifier = f"{request.client.host}_{request.url.path}"
        
        if await limiter.exceeded(identifier):
            return ORJSONResponse(
                {"error": "Rate limit exceeded"},
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        
        response = await call_next(request)
        return response

class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.monotonic()
        audit_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host,
            "user_agent": request.headers.get("user-agent"),
        }
        
        try:
            response = await call_next(request)
            audit_log.update({
                "status": response.status_code,
                "duration": time.monotonic() - start_time,
            })
            await request.app.state.kafka.send("api_audit_log", audit_log)
            return response
        except Exception as exc:
            audit_log.update({
                "status": 500,
                "error": str(exc),
                "duration": time.monotonic() - start_time,
            })
            await request.app.state.kafka.send("api_audit_log", audit_log)
            raise

class JWTClaims(BaseModel):
    sub: str
    roles: List[str]
    exp: datetime
    iss: str = "aelion-ai"

async def validate_jwt(
    token: Annotated[str, Security(APIKeyHeader(name="Authorization"))],
    config: APIConfig = Depends(APIConfig),
) -> JWTClaims:
    try:
        payload = jwt.decode(
            token,
            config.rsa_public_key,
            algorithms=["RS256"],
            options={"require_exp": True},
        )
        return JWTClaims(**payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
        )
    except (jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid authentication credentials",
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.logger = Logger("api")
    app.state.metrics = MetricsSystem([
        "api_requests_total",
        "api_request_duration_seconds",
        "api_errors_total",
    ])
    app.state.grpc = GRPCClient()
    app.state.kafka = KafkaProducer()
    app.state.registry = AgentRegistry()
    app.state.limiter = Limiter(key_func=get_remote_address)
    
    await app.state.grpc.connect()
    await app.state.kafka.start()
    app.state.logger.info("API startup completed")
    
    yield
    
    # Shutdown
    await app.state.grpc.close()
    await app.state.kafka.stop()
    app.state.logger.info("API shutdown completed")

app = FastAPI(
    title="Aelion AI Control Plane API",
    version="2.0.0",
    default_response_class=ORJSONResponse,
    docs_url="/docs" if APIConfig().enable_swagger else None,
    redoc_url=None,
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig().cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ContextMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuditMiddleware)

# Routers
system_router = APIRouter(prefix="/system", tags=["System"])
agent_router = APIRouter(prefix="/agents", tags=["Agents"])
task_router = APIRouter(prefix="/tasks", tags=["Tasks"])

@system_router.get("/health", include_in_schema=False)
async def health_check():
    return {"status": "ok"}

@system_router.get("/metrics")
async def prometheus_metrics():
    return Response(
        content=app.state.metrics.generate(),
        media_type="text/plain",
    )

@agent_router.get("", dependencies=[Security(validate_jwt, scopes=["admin"])])
async def list_agents(
    agent_type: Optional[str] = None,
    status: Optional[str] = None,
    registry: AgentRegistry = Depends(lambda: app.state.registry),
):
    return await registry.list_agents(
        agent_type=agent_type,
        status=status,
    )

@agent_router.post("/{agent_id}/tasks")
async def submit_agent_task(
    agent_id: str,
    payload: Dict[str, Any] = Body(...),
    registry: AgentRegistry = Depends(lambda: app.state.registry),
    kafka: KafkaProducer = Depends(lambda: app.state.kafka),
):
    agent = await registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    task_id = SecureSerializer.generate_uuid()
    await kafka.send("agent_tasks", {
        "task_id": task_id,
        "agent_id": agent_id,
        "payload": payload,
        "timestamp": datetime.utcnow().isoformat(),
    })
    
    return {"task_id": task_id, "status": "queued"}

@task_router.get("/{task_id}")
async def get_task_status(
    task_id: str,
    grpc: GRPCClient = Depends(lambda: app.state.grpc),
):
    try:
        response = await grpc.call(
            service="task_manager",
            method="get_task_status",
            payload={"task_id": task_id},
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service unavailable: {str(e)}",
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    app.state.metrics.inc("api_errors_total")
    return ORJSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

# Register routers
app.include_router(system_router)
app.include_router(agent_router)
app.include_router(task_router)

if __name__ == "__main__":
    config = APIConfig()
    uvicorn.run(
        "api.rest.app:app",
        host="0.0.0.0",
        port=config.port,
        workers=config.workers,
        log_config=None,
        access_log=False,
        proxy_headers=True,
        timeout_keep_alive=120,
    )
