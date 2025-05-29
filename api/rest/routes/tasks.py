"""
Enterprise Task Management API Routes - Full Task Lifecycle Operations
"""

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field, validator

from core.agent.registry import AgentRegistry
from core.agent.base import TaskStatus, TaskPriority
from grpc.client import GRPCClient
from kafka.producer import KafkaProducer
from utils.logger import Logger
from utils.metrics import MetricsSystem
from utils.serialization import SecureSerializer
from api.rest.security import JWTClaims

#region Pydantic Models

class TaskCreate(BaseModel):
    name: str = Field(..., min_length=3, max_length=128)
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        example={"type": "image_processing"}
    )
    priority: TaskPriority = TaskPriority.MEDIUM
    deadline: Optional[datetime] = None
    dependencies: List[UUID] = Field(default_factory=list)

    @validator("dependencies")
    def validate_dependencies(cls, v):
        if len(v) > 100:
            raise ValueError("Too many dependencies")
        return v

class TaskUpdate(BaseModel):
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    result: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    id: UUID
    name: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    deadline: Optional[datetime]
    result: Optional[Dict[str, Any]]
    dependencies: List[UUID]
    assigned_agent: Optional[UUID]

    class Config:
        orm_mode = True
        allow_population_by_field_name = True

class PaginatedTasks(BaseModel):
    total: int
    tasks: List[TaskResponse]
    next_page: Optional[str]

#endregion

router = APIRouter(
    prefix="/tasks",
    tags=["Task Management"],
    default_response_class=ORJSONResponse
)

#region Dependency Setup

async def get_registry() -> AgentRegistry:
    return AgentRegistry.get_instance()

async def get_kafka() -> KafkaProducer:
    return KafkaProducer.get_instance()

async def get_grpc() -> GRPCClient:
    return GRPCClient.get_instance()

async def require_admin(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    logger: Annotated[Logger, Depends(Logger.get_instance)]
) -> None:
    if "admin" not in claims.roles:
        logger.warning(f"Unauthorized admin attempt by {claims.sub}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges"
        )

#endregion

@router.get(
    "",
    response_model=PaginatedTasks,
    summary="List tasks with filtering",
    description="Paginated list of tasks with status/priority filtering",
    responses={
        400: {"description": "Invalid filter parameters"},
        500: {"description": "Internal server error"}
    }
)
async def list_tasks(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    metrics: Annotated[MetricsSystem, Depends(MetricsSystem.get_instance)],
    status: Optional[TaskStatus] = Query(None),
    priority: Optional[TaskPriority] = Query(None),
    agent_id: Optional[UUID] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
) -> PaginatedTasks:
    metrics.inc("api_task_list_requests")
    try:
        tasks, total = await registry.list_tasks(
            status=status,
            priority=priority,
            agent_id=agent_id,
            limit=limit,
            offset=offset
        )
        
        return PaginatedTasks(
            total=total,
            tasks=[TaskResponse.from_orm(task) for task in tasks],
            next_page=f"?offset={offset+limit}&limit={limit}" if offset+limit < total else None
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        metrics.inc("api_task_list_errors")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve task list"
        ) from e

@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=TaskResponse,
    summary="Create new task",
    responses={
        400: {"description": "Invalid task parameters"},
        409: {"description": "Task dependency conflict"}
    }
)
async def create_task(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    kafka: Annotated[KafkaProducer, Depends(get_kafka)],
    grpc: Annotated[GRPCClient, Depends(get_grpc)],
    logger: Annotated[Logger, Depends(Logger.get_instance)],
    task_data: TaskCreate
) -> TaskResponse:
    try:
        # Validate dependencies
        if task_data.dependencies:
            valid_deps = await registry.validate_dependencies(task_data.dependencies)
            if len(valid_deps) != len(task_data.dependencies):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Invalid task dependencies"
                )
        
        # Create via gRPC
        grpc_response = await grpc.call(
            service="task_manager",
            method="create_task",
            payload=task_data.dict()
        )
        
        # Register in local registry
        task = await registry.register_task(
            task_id=grpc_response["task_id"],
            name=task_data.name,
            priority=task_data.priority,
            deadline=task_data.deadline,
            dependencies=task_data.dependencies,
            payload=task_data.payload
        )
        
        # Emit creation event
        await kafka.send("task_events", {
            "event_type": "task_created",
            "task_id": str(task.id),
            "initiator": claims.sub,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Created new task: {task_data.name}")
        return TaskResponse.from_orm(task)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task creation failed"
        ) from e

@router.get(
    "/{task_id}",
    response_model=TaskResponse,
    summary="Get task details",
    responses={
        404: {"description": "Task not found"}
    }
)
async def get_task(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    task_id: UUID = Path(..., description="UUID of the task")
) -> TaskResponse:
    task = await registry.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
        
    return TaskResponse.from_orm(task)

@router.put(
    "/{task_id}",
    response_model=TaskResponse,
    dependencies=[Depends(require_admin)],
    summary="Update task status or results",
    responses={
        400: {"description": "Invalid update parameters"},
        404: {"description": "Task not found"},
        409: {"description": "Invalid state transition"}
    }
)
async def update_task(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    kafka: Annotated[KafkaProducer, Depends(get_kafka)],
    grpc: Annotated[GRPCClient, Depends(get_grpc)],
    task_id: UUID = Path(..., description="UUID of the task"),
    update_data: TaskUpdate = Body(...)
) -> TaskResponse:
    task = await registry.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
        
    try:
        # Update via gRPC
        grpc_response = await grpc.call(
            service="task_manager",
            method="update_task",
            payload={
                "task_id": str(task_id),
                **update_data.dict(exclude_unset=True)
            }
        )
        
        # Update local registry
        updated_task = await registry.update_task(
            task_id=task_id,
            **update_data.dict(exclude_unset=True)
        )
        
        # Emit update event
        event_type = "task_updated"
        if update_data.status == TaskStatus.COMPLETED:
            event_type = "task_completed"
            
        await kafka.send("task_events", {
            "event_type": event_type,
            "task_id": str(task_id),
            "fields_updated": list(update_data.dict().keys()),
            "initiator": claims.sub,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return TaskResponse.from_orm(updated_task)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task update failed"
        ) from e

@router.delete(
    "/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(require_admin)],
    summary="Cancel a task",
    responses={
        404: {"description": "Task not found"},
        409: {"description": "Task cannot be canceled in current state"}
    }
)
async def cancel_task(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    kafka: Annotated[KafkaProducer, Depends(get_kafka)],
    grpc: Annotated[GRPCClient, Depends(get_grpc)],
    task_id: UUID = Path(..., description="UUID of the task")
) -> None:
    task = await registry.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
        
    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot cancel completed/failed task"
        )
        
    try:
        # Cancel via gRPC
        await grpc.call(
            service="task_manager",
            method="cancel_task",
            payload={"task_id": str(task_id)}
        )
        
        # Update local registry
        await registry.update_task(
            task_id=task_id,
            status=TaskStatus.CANCELED
        )
        
        # Emit cancellation event
        await kafka.send("task_events", {
            "event_type": "task_canceled",
            "task_id": str(task_id),
            "initiator": claims.sub,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task cancellation failed"
        ) from e

@router.post(
    "/{task_id}/retry",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry failed task",
    dependencies=[Depends(require_admin)],
    responses={
        404: {"description": "Task not found"},
        409: {"description": "Task not in retryable state"}
    }
)
async def retry_task(
    claims: Annotated[JWTClaims, Depends(JWTClaims)],
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    grpc: Annotated[GRPCClient, Depends(get_grpc)],
    task_id: UUID = Path(..., description="UUID of the task")
) -> Dict[str, str]:
    task = await registry.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
        
    if task.status != TaskStatus.FAILED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only failed tasks can be retried"
        )
        
    try:
        response = await grpc.call(
            service="task_manager",
            method="retry_task",
            payload={"task_id": str(task_id)}
        )
        
        return {"operation_id": response["operation_id"]}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Task retry failed"
        ) from e

@router.get(
    "/{task_id}/dependencies",
    response_model=List[TaskResponse],
    summary="Get task dependencies",
    responses={
        404: {"description": "Task not found"}
    }
)
async def get_task_dependencies(
    registry: Annotated[AgentRegistry, Depends(get_registry)],
    task_id: UUID = Path(..., description="UUID of the task")
) -> List[TaskResponse]:
    task = await registry.get_task(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
        
    dependencies = await registry.get_task_dependencies(task_id)
    return [TaskResponse.from_orm(t) for t in dependencies]
