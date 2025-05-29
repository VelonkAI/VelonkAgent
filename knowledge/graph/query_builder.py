"""
Enterprise Query Builder - Type-safe Cypher Query Construction & Optimization
"""

from __future__ import annotations
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    TypeVar,
    Generic,
    overload
)
from enum import Enum, auto
from dataclasses import dataclass
import inspect
import warnings
import textwrap
from functools import cached_property
from neo4j import Query
from pydantic import BaseModel, ValidationError
from ..utils.logger import Logger
from ..utils.metrics import MetricsSystem

T = TypeVar('T', bound=BaseModel)

class ClauseType(Enum):
    MATCH = auto()
    WHERE = auto()
    WITH = auto()
    RETURN = auto()
    CREATE = auto()
    MERGE = auto()
    SET = auto()
    DELETE = auto()
    ORDER_BY = auto()
    LIMIT = auto()
    UNWIND = auto()
    CALL = auto()
    UNION = auto()

@dataclass(frozen=True)
class QueryParam:
    name: str
    value: Any
    type: type

@dataclass
class ExecutionPlan:
    query: str
    parameters: Dict[str, Any]
    query_type: ClauseType
    complexity_score: float

class CypherClause:
    def __init__(self, clause_type: ClauseType):
        self.clause_type = clause_type
        self._fragments: List[str] = []
        self._params: Dict[str, QueryParam] = {}
        self._subqueries: List[CypherClause] = []

    def add_fragment(
        self,
        fragment: str,
        params: Optional[Dict[str, Any]] = None
    ) -> CypherClause:
        normalized = fragment.strip()
        if params:
            for name, value in params.items():
                param = QueryParam(name=name, value=value, type=type(value))
                self._params[name] = param
                normalized = normalized.replace(f"${name}", f"${param.name}")
        self._fragments.append(normalized)
        return self

    def chain(self, subclause: CypherClause) -> CypherClause:
        if subclause.clause_type != self.clause_type:
            raise ValueError("Cannot chain different clause types")
        self._subqueries.append(subclause)
        return self

    @cached_property
    def compiled(self) -> str:
        clauses = []
        main_clause = " ".join(self._fragments)
        clauses.append(f"{self.clause_type.name} {main_clause}")
        
        for sub in self._subqueries:
            clauses.append(sub.compiled)
        
        return "\n".join(clauses)

    @property
    def parameters(self) -> Dict[str, Any]:
        base_params = {p.name: p.value for p in self._params.values()}
        for sub in self._subqueries:
            base_params.update(sub.parameters)
        return base_params

    def explain(self) -> ExecutionPlan:
        return ExecutionPlan(
            query=self.compiled,
            parameters=self.parameters,
            query_type=self.clause_type,
            complexity_score=len(self._fragments) * 0.5 + len(self._subqueries) * 1.2
        )

class QueryBuilder(Generic[T]):
    def __init__(self, model: type[T]):
        self.model = model
        self._clauses: List[CypherClause] = []
        self._logger = Logger(__name__)
        self._cache: Dict[str, str] = {}
        self._param_counter = 0

    def match(self, pattern: str, **params: Any) -> QueryBuilder[T]:
        clause = CypherClause(ClauseType.MATCH)
        clause.add_fragment(pattern, params)
        self._clauses.append(clause)
        return self

    def where(self, condition: str, **params: Any) -> QueryBuilder[T]:
        clause = CypherClause(ClauseType.WHERE)
        clause.add_fragment(condition, params)
        self._clauses.append(clause)
        return self

    def with_clause(self, items: str, **params: Any) -> QueryBuilder[T]:
        clause = CypherClause(ClauseType.WITH)
        clause.add_fragment(items, params)
        self._clauses.append(clause)
        return self

    def return_clause(self, items: str, **params: Any) -> QueryBuilder[T]:
        clause = CypherClause(ClauseType.RETURN)
        clause.add_fragment(items, params)
        self._clauses.append(clause)
        return self

    def build(self, validate: bool = True) -> Query:
        full_query = []
        parameters = {}
        
        for clause in self._clauses:
            full_query.append(clause.compiled)
            parameters.update(clause.parameters)
        
        cypher = "\n".join(full_query)
        
        if validate:
            self._validate_model_compatibility(cypher)
            
        return Query(text=cypher, metadata={"params": parameters})

    def _validate_model_compatibility(self, cypher: str):
        return_fields = [
            line.split("RETURN")[1].strip()
            for line in cypher.splitlines()
            if "RETURN" in line
        ]
        
        if not return_fields:
            raise ValueError("Query must contain RETURN clause for model binding")
            
        expected_fields = set(self.model.__fields__.keys())
        returned_fields = set(",".join(return_fields).split(","))
        
        if not returned_fields.issuperset(expected_fields):
            missing = expected_fields - returned_fields
            raise ValueError(f"Query missing model fields: {missing}")

    def paginate(self, page: int, per_page: int) -> QueryBuilder[T]:
        skip = (page - 1) * per_page
        self._clauses.append(
            CypherClause(ClauseType.ORDER_BY)
            .add_fragment("SKIP $skip LIMIT $limit", {"skip": skip, "limit": per_page})
        )
        return self

    @overload
    def param(self, value: str) -> str: ...
    
    @overload
    def param(self, value: int) -> str: ...
    
    @overload
    def param(self, value: float) -> str: ...
    
    @overload
    def param(self, value: bool) -> str: ...
    
    def param(self, value: Any) -> str:
        self._param_counter += 1
        param_name = f"param_{self._param_counter}"
        self._clauses[-1].add_fragment(f"${param_name}", {param_name: value})
        return f"${param_name}"

    @classmethod
    def from_raw_cypher(cls, cypher: str, params: Dict[str, Any]) -> Query:
        return Query(text=cypher, metadata={"params": params})

    def cache_key(self) -> str:
        clauses = [clause.compiled for clause in self._clauses]
        return hash("".join(clauses))

class QueryOptimizer:
    _OPTIMIZATION_RULES = [
        ("MATCH (a) WHERE EXISTS(a.property)", "MATCH (a) WHERE a.property IS NOT NULL"),
        ("MERGE (a:Label)", "MERGE (a:Label) ON CREATE SET a.created_at = timestamp()"),
        ("LIMIT \d+ OFFSET \d+", "SKIP {skip} LIMIT {limit}")
    ]

    def __init__(self, query: Query):
        self.original = query
        self._applied_rules = []

    def optimize(self) -> Query:
        optimized_text = self.original.text
        for pattern, replacement in self._OPTIMIZATION_RULES:
            optimized_text, count = self._apply_rule(optimized_text, pattern, replacement)
            if count > 0:
                self._log_optimization(pattern, replacement, count)
        
        return Query(
            text=optimized_text,
            metadata=self.original.metadata
        )

    def _apply_rule(self, text: str, pattern: str, replacement: str) -> tuple[str, int]:
        import re
        compiled = re.compile(pattern)
        return compiled.subn(replacement, text)

    def _log_optimization(self, pattern: str, replacement: str, count: int):
        self._applied_rules.append({
            "pattern": pattern,
            "replacement": replacement,
            "count": count
        })
        Logger(__name__).info(f"Applied optimization: {pattern} → {replacement} ({count}x)")

class QueryExecutor:
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver
        self._metrics = MetricsSystem([
            "query_duration_seconds",
            "query_errors_total",
            "query_cache_hits_total"
        ])

    async def execute(
        self,
        query: Union[Query, QueryBuilder[T]],
        model: Optional[type[T]] = None
    ) -> List[T]:
        if isinstance(query, QueryBuilder):
            built = query.build()
            return await self._execute_built(built, query.model)
            
        return await self._execute_raw(query, model)

    async def _execute_built(self, query: Query, model: type[T]) -> List[T]:
        with self._metrics.timer("query_duration_seconds", labels=[model.__name__]):
            try:
                result = await self.driver.execute_query(query.text, query.metadata["params"])
                return [model(**item) for item in result]
            except Exception as e:
                self._metrics.inc("query_errors_total", labels=[model.__name__, type(e).__name__])
                raise

    async def _execute_raw(self, query: Query, model: Optional[type[T]]) -> List[Any]:
        with self._metrics.timer("query_duration_seconds", labels=["raw"]):
            try:
                result = await self.driver.execute_query(query.text, query.metadata["params"])
                if model:
                    return [model(**item) for item in result]
                return result
            except Exception as e:
                self._metrics.inc("query_errors_total", labels=["raw", type(e).__name__])
                raise

# Example Usage
if __name__ == "__main__":
    from pydantic import BaseModel
    from graph.neo4j_driver import Neo4jDriver

    class AgentModel(BaseModel):
        id: str
        type: str
        status: str

    async def main():
        driver = Neo4jDriver(config=Neo4jConfig())
        await driver.connect()
        
        # Example 1: Basic Query
        qb = QueryBuilder(AgentModel).match(
            "(a:Agent)",
            id="agent_001"
        ).where(
            "a.status = $status",
            status="active"
        ).return_clause("a")
        
        query = qb.build()
        print("Generated Query:\n", query.text)
        print("Parameters:", query.metadata["params"])
        
        # Example 2: Paginated Query
        qb = (
            QueryBuilder(AgentModel)
            .match("(a:Agent)")
            .where("a.type = $type", type="worker")
            .return_clause("a")
            .paginate(page=2, per_page=10)
        )
        print("Paginated Query:\n", qb.build().text)
        
        # Example 3: Execution
        executor = QueryExecutor(driver)
        agents = await executor.execute(qb)
        for agent in agents:
            print(f"Agent {agent.id} ({agent.type}) is {agent.status}")

    asyncio.run(main())
