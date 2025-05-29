"""
Enterprise Agent Management API Routes - Full CRUD + Operational Endpoints
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, validator

from core.agent.registry import AgentRegistry
from core.agent.base import AgentStatus
from grpc.client import GRPCClient
from kafka.producer import KafkaProducer
from utils.logger import Logger
from utils.metrics import MetricsSystem
from utils.serialization import SecureSerializer
from api.rest.security import JWTClaims

#region Pydantic Models

class AgentCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=64, regex="^[a-zA-Z0-9_-]+$")
    agent_type: str = Field(..., alias="type", regex="^(worker|supervisor|coordinator)$")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        example={"max_concurrent_tasks": 10}
    )

    @validator("config")
    def validate_config(cls, v):
        if "unsafe_param" in v:
            raise ValueError("Forbidden config parameter")
        return v

class AgentUpdate(BaseModel):
    status: Optional[AgentStatus] = None
    config_update: Optional[Dict[str, Any]] = Field(None, alias="config")

class AgentResponse(BaseModel):
    id: UUID
    name: str
    agent_type: str = Field(..., alias="type")
    status: AgentStatus
    created_at: datetime
    last_heartbeat: Optional[datetime]
    config: Dict[str, Any]

    class Config:
        orm_mode = True
        allow_population_by_field_name = True

class PaginatedAgents(BaseModel):
    total: int
    agents: List[AgentResponse]
    next_page: Optional[str]

#endregion

router = APIRouter(
    prefix="/agents",
    tags=["Agent Management"],
    default_response_class=ORJSONResponse
)

#region Dependency Setup

async def get_registry() -> AgentRegistry:
    """Dependency that provides AgentRegistry instance"""
    return AgentRegistry.get_instance()

async def get_kafka() -> KafkaProducer:
    """Dependency that provides Kafka producer"""
    return KafkaProducer.get_instance()

async def get_grpc() -> GRPCClient:
    """Dependency that provides gRPC client"""
    return GRPCClient.get_instance()

async def require_admin(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    logger: Annotated[Logger, Depends(Logger.get_instance)]
) -> None:
    """Require admin role for sensitive operations"""
    if "admin" not in claims.roles:
        logger.warning(f"Unauthorized admin attempt by {claims.sub}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges"
        )

#endregion

@router.get(
    "",
    response_model=PaginatedAgents,
    summary="List all registered agents",
    description="Paginated list of agents with filtering capabilities",
    responses={
        401: {"description": "Missing or invalid authentication"},
        403: {"description": "Insufficient permissions"},
        500: {"description": "Internal server error"}
    }
)
async def list_agents(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    metrics: Annotated[MetricsSystem, Depends(MetricsSystem.get_instance)],
    agent_type: Optional[str] = None,
    status: Optional[AgentStatus] = None,
    limit: int = 100,
    offset: int = 0
) -> PaginatedAgents:
    metrics.inc("api_agent_list_requests")
    try:
        agents, total = await registry.list_agents(
            agent_type=agent_type,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return PaginatedAgents(
            total=total,
            agents=[
                AgentResponse.from_orm(agent) 
                for agent in agents
            ],
            next_page=f"?offset={offset+limit}&limit={limit}" if offset+limit < total else None
        )
    except Exception as e:
        metrics.inc("api_agent_list_errors")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agent list"
        ) from e

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=AgentResponse,
    dependencies=[Depends(require_admin)],
    summary="Create new agent instance",
    responses={
        400: {"description": "Invalid agent configuration"},
        409: {"description": "Agent name conflict"}
    }
)
async def create_agent(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    kafka: Annotated[KafkaProducer, Depends(get_kafka)],
    grpc: Annotated[GRPCClient, Depends(get_grpc)],
    logger: Annotated[Logger, Depends(Logger.get_instance)],
    agent_data: AgentCreate
) -> AgentResponse:
    try:
        # Check name uniqueness
        if await registry.agent_name_exists(agent_data.name):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Agent name already exists"
            )
            
        # Create via gRPC
        grpc_response = await grpc.call(
            service="agent_manager",
            method="create_agent",
            payload=agent_data.dict()
        )
        
        # Register in local registry
        agent = await registry.register_agent(
            agent_id=grpc_response["agent_id"],
            name=agent_data.name,
            agent_type=agent_data.agent_type,
            config=agent_data.config
        )
        
        # Emit creation event
        await kafka.send("agent_events", {
            "event_type": "agent_created",
            "agent_id": str(agent.id),
            "initiator": claims.sub,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Created new agent: {agent_data.name}")
        return AgentResponse.from_orm(agent)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent creation failed"
        ) from e

@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
    summary="Get agent details",
    responses={
        404: {"description": "Agent not found"}
    }
)
async def get_agent(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    agent_id: UUID = Path(..., description="UUID of the agent")
) -> AgentResponse:
    agent = await registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
        
    return AgentResponse.from_orm(agent)

@router.put(
    "/{agent_id}",
    response_model=AgentResponse,
    dependencies=[Depends(require_admin)],
    summary="Update agent configuration or status",
    responses={
        400: {"description": "Invalid update parameters"},
        404: {"description": "Agent not found"}
    }
)
async def update_agent(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    kafka: Annotated[KafkaProducer, Depends(get_kafka)],
    grpc: Annotated[GRPCClient, Depends(get_grpc)],
    agent_id: UUID = Path(..., description="UUID of the agent"),
    update_data: AgentUpdate = Body(...)
) -> AgentResponse:
    agent = await registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
        
    try:
        # Update via gRPC
        await grpc.call(
            service="agent_manager",
            method="update_agent",
            payload={
                "agent_id": str(agent_id),
                **update_data.dict(exclude_unset=True)
            }
        )
        
        # Update local registry
        updated_agent = await registry.update_agent(
            agent_id=agent_id,
            **update_data.dict(exclude_unset=True)
        )
        
        # Emit update event
        await kafka.send("agent_events", {
            "event_type": "agent_updated",
            "agent_id": str(agent_id),
            "fields_updated": list(update_data.dict().keys()),
            "initiator": claims.sub,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return AgentResponse.from_orm(updated_agent)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent update failed"
        ) from e

@router.delete(
    "/{agent_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin)],
    summary="Decommission an agent",
    responses={
        404: {"description": "Agent not found"},
        409: {"description": "Agent cannot be deleted (active state)"}
    }
)
async def delete_agent(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    kafka: Annotated[KafkaProducer, Depends(get_kafka)],
    grpc: Annotated[GRPCClient, Depends(get_grpc)],
    agent_id: UUID = Path(..., description="UUID of the agent")
) -> None:
    agent = await registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
        
    if agent.status == AgentStatus.ACTIVE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete active agent"
        )
        
    try:
        # Decommission via gRPC
        await grpc.call(
            service="agent_manager",
            method="delete_agent",
            payload={"agent_id": str(agent_id)}
        )
        
        # Remove from registry
        await registry.deregister_agent(agent_id)
        
        # Emit deletion event
        await kafka.send("agent_events", {
            "event_type": "agent_deleted",
            "agent_id": str(agent_id),
            "initiator": claims.sub,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent deletion failed"
        ) from e

@router.post(
    "/{agent_id}/restart",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Restart an agent instance",
    dependencies=[Depends(require_admin)],
    responses={
        404: {"description": "Agent not found"},
        409: {"description": "Restart not allowed in current state"}
    }
)
async def restart_agent(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    grpc: Annotated[GRPCClient, Depends(get_grpc)],
    agent_id: UUID = Path(..., description="UUID of the agent")
) -> Dict[str, str]:
    try:
        response = await grpc.call(
            service="agent_manager",
            method="restart_agent",
            payload={"agent_id": str(agent_id)}
        )
        
        return {"operation_id": response["operation_id"]}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent restart failed"
        ) from e

@router.get(
    "/{agent_id}/tasks",
    summary="Get recent tasks for an agent",
    responses={
        404: {"description": "Agent not found"}
    }
)
async def get_agent_tasks(
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    agent_id: UUID = Path(..., description="UUID of the agent"),
    limit: int = 50
) -> List[Dict[str, Any]]:
    agent = await registry.get_agent(agent_id)
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
        
    return await registry.get_agent_tasks(agent_id, limit=limit)
