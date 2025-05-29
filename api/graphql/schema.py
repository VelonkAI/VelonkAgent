"""
Enterprise GraphQL Schema - Type-safe API Surface for Agent Orchestration
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import graphene
from graphene import relay, ObjectType, Interface, Mutation, Field
from graphene.types.datetime import DateTime
from graphene.types.generic import GenericScalar
from graphene_pydantic import PydanticObjectType

from core.models import Agent, Task, ResourcePool, TrainingRun, EvaluationMetrics
from core.pagination import PaginatedResponse
from core.filters import AgentFilter, TaskFilter
from core.types import JSONScalar

#region Enums

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    TRAINING = "training"
    DEGRADED = "degraded"
    OFFLINE = "offline"

class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class SortDirection(Enum):
    ASC = "asc"
    DESC = "desc"

#endregion

#region Interfaces

class AgentInterface(Interface):
    id = graphene.ID(required=True)
    status = graphene.Enum.from_enum(AgentStatus)(required=True)
    last_heartbeat = DateTime()
    capability_tags = graphene.List(graphene.String)

class ResourceUsageInterface(Interface):
    cpu_usage = graphene.Float()
    memory_usage = graphene.Float()
    gpu_utilization = graphene.Float()

#endregion

#region Types

class AgentType(PydanticObjectType):
    class Meta:
        model = Agent
        interfaces = (relay.Node, AgentInterface, ResourceUsageInterface)
        exclude = ("internal_state",)

    tasks = graphene.List(lambda: TaskType)
    resource_pool = Field(lambda: ResourcePoolType)
    metrics_history = graphene.List(lambda: AgentMetricsType)

    def resolve_tasks(self, info):
        return self.get_associated_tasks()

class TaskType(PydanticObjectType):
    class Meta:
        model = Task
        interfaces = (relay.Node,)
    
    progress = graphene.Float()
    dependencies = graphene.List(lambda: TaskType)
    assigned_agents = graphene.List(AgentType)

class ResourcePoolType(PydanticObjectType):
    class Meta:
        model = ResourcePool
        interfaces = (relay.Node,)

    utilization_history = graphene.List(JSONScalar)

class TrainingRunType(PydanticObjectType):
    class Meta:
        model = TrainingRun

    intermediate_metrics = graphene.List(lambda: TrainingMetricsType)

class EvaluationMetricsType(PydanticObjectType):
    class Meta:
        model = EvaluationMetrics

#endregion

#region Paginated Types

class AgentConnection(relay.Connection):
    class Meta:
        node = AgentType

    total_count = graphene.Int()
    page_info = graphene.Field(PaginatedResponse)

class TaskConnection(relay.Connection):
    class Meta:
        node = TaskType

#endregion

#region Input Types

class AgentFilterInput(graphene.InputObjectType):
    status = graphene.Enum.from_enum(AgentStatus)()
    capability_tags = graphene.List(graphene.String)
    min_cpu_usage = graphene.Float()
    last_heartbeat_after = DateTime()

class TaskCreateInput(graphene.InputObjectType):
    payload = JSONScalar(required=True)
    priority = graphene.Enum.from_enum(TaskPriority)(required=True)
    deadline = DateTime()
    required_capabilities = graphene.List(graphene.String)

class SortByInput(graphene.InputObjectType):
    field = graphene.String(required=True)
    direction = graphene.Enum.from_enum(SortDirection)(required=True)

#endregion

#region Queries

class Query(ObjectType):
    node = relay.Node.Field()

    agents = relay.ConnectionField(
        AgentConnection,
        filter=AgentFilterInput(),
        sort=SortByInput(),
        first=graphene.Int(),
        offset=graphene.Int()
    )

    task = graphene.Field(TaskType, id=graphene.ID(required=True))
    resource_pools = graphene.List(ResourcePoolType)
    training_runs = graphene.List(TrainingRunType)

    def resolve_agents(self, info, filter=None, sort=None, first=None, offset=None):
        return Agent.paginated_query(
            filter=filter,
            sort=sort,
            pagination={"first": first, "offset": offset}
        )

    def resolve_resource_pools(self, info):
        return ResourcePool.get_all()

#endregion

#region Mutations

class CreateTask(Mutation):
    class Arguments:
        input = TaskCreateInput(required=True)

    task = graphene.Field(TaskType)
    status = graphene.String()

    def mutate(self, info, input):
        task = Task.create_from_input(input)
        return CreateTask(task=task, status="ACCEPTED")

class TerminateTask(Mutation):
    class Arguments:
        id = graphene.ID(required=True)

    success = graphene.Boolean()
    timestamp = DateTime()

    def mutate(self, info, id):
        Task.terminate(id)
        return TerminateTask(success=True, timestamp=datetime.utcnow())

class Mutation(ObjectType):
    create_task = CreateTask.Field()
    terminate_task = TerminateTask.Field()
    register_agent = Field(AgentType)
    update_agent_state = Field(AgentType)
    start_training = Field(TrainingRunType)

#endregion

#region Subscriptions

class Subscription(ObjectType):
    agent_status_changed = graphene.Field(AgentType, agent_id=graphene.ID())
    task_progress = graphene.Field(TaskType, task_id=graphene.ID())
    resource_usage = graphene.Field(ResourcePoolType)

    def resolve_agent_status_changed(root, info, agent_id):
        return Agent.subscribe_to_status(agent_id)

#endregion

#region Schema Construction

class VersionedSchema(graphene.Schema):
    def __init__(self):
        super().__init__(
            query=Query,
            mutation=Mutation,
            subscription=Subscription,
            types=[
                AgentType,
                TaskType,
                TrainingMetricsType,
                EvaluationMetricsType
            ],
            auto_camelcase=True,
            middlewares=[ComplexityMiddleware()]
        )

#region Complexity Analysis

class ComplexityMiddleware:
    def resolve(self, next, root, info, **args):
        max_complexity = 1000
        current_complexity = calculate_complexity(info)
        
        if current_complexity > max_complexity:
            raise Exception(f"Query too complex: {current_complexity}/{max_complexity}")
        
        return next(root, info, **args)

#endregion

schema = VersionedSchema()
