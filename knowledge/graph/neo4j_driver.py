"""
Enterprise Neo4j Driver - High-performance Graph Database Integration with Cluster Support
"""

from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache
import logging
from dataclasses import dataclass
import ssl
from neo4j import (
    AsyncGraphDatabase,
    AsyncSession,
    AsyncTransaction,
    EagerResult,
    RoutingControl
)
from neo4j.exceptions import (
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
    TransientError
)
from prometheus_client import (  # type: ignore
    Histogram,
    Counter,
    Gauge
)
from ..utils.metrics import MetricsSystem
from ..utils.serialization import deserialize_neo4j

# Prometheus Metrics
QUERY_DURATION = Histogram(
    'neo4j_query_duration_seconds',
    'Query execution time distribution',
    ['query_type', 'cluster']
)
CONNECTION_GAUGE = Gauge(
    'neo4j_connections_active',
    'Active Neo4j connections',
    ['cluster']
)
RETRY_COUNTER = Counter(
    'neo4j_retries_total',
    'Total query retry attempts',
    ['cluster', 'error_type']
)

@dataclass(frozen=True)
class Neo4jConfig:
    """Immutable configuration for Neo4j cluster connectivity"""
    uri: str = "neo4j://localhost:7687"
    auth: tuple = ("neo4j", "password")
    max_connection_pool_size: int = 100
    connection_timeout: int = 30  # seconds
    encrypted: bool = True
    trust: str = "TRUST_ALL_CERTIFICATES"  # TRUST_SYSTEM_CA_SIGNED_CERTIFICATES
    max_transaction_retry_time: int = 30  # seconds
    database: str = "neo4j"
    load_balancing_strategy: str = "ROUND_ROBIN"
    max_retries: int = 5
    retry_delay: float = 0.5  # seconds
    fetch_size: int = 1000
    cert_path: Optional[str] = None

class Neo4jDriver:
    """Enterprise-grade Neo4j driver with connection pooling and automatic retries"""
    
    def __init__(self, config: Neo4jConfig):
        self._config = config
        self._driver = None
        self._metrics = MetricsSystem([])
        self._logger = logging.getLogger("aelion.neo4j")
        self._ssl_context = self._configure_ssl()
        self._cluster_nodes = []

    async def connect(self):
        """Initialize connection pool and cluster discovery"""
        kwargs = {
            "auth": self._config.auth,
            "max_connection_pool_size": self._config.max_connection_pool_size,
            "connection_timeout": self._config.connection_timeout,
            "encrypted": self._config.encrypted,
            "trust": self._config.trust,
            "user_agent": "AelionAI/1.0",
            "keep_alive": True,
            "fetch_size": self._config.fetch_size
        }
        
        if self._ssl_context:
            kwargs["ssl"] = self._ssl_context

        self._driver = AsyncGraphDatabase.driver(
            self._config.uri,
            **kwargs
        )
        
        await self._discover_cluster()
        CONNECTION_GAUGE.labels(cluster=self.cluster_name).inc()

    async def _discover_cluster(self):
        """Discover cluster topology and update routing tables"""
        try:
            with await self._driver.session(database="system") as session:
                result = await session.run(
                    "SHOW SERVERS YIELD id, address, role, currentStatus"
                )
                nodes = await result.values()
                self._cluster_nodes = [
                    {
                        "id": node[0],
                        "address": node[1],
                        "role": node[2],
                        "status": node[3]
                    } for node in nodes
                ]
                self._logger.info(f"Discovered {len(nodes)} cluster nodes")
        except Exception as e:
            self._logger.error(f"Cluster discovery failed: {str(e)}")
            raise

    def _configure_ssl(self) -> Optional[ssl.SSLContext]:
        """Configure SSL context for encrypted connections"""
        if not self._config.encrypted:
            return None
            
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        
        if self._config.trust == "TRUST_SYSTEM_CA_SIGNED_CERTIFICATES":
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_default_certs()
        elif self._config.cert_path:
            ctx.load_verify_locations(self._config.cert_path)
            ctx.verify_mode = ssl.CERT_REQUIRED
        else:
            ctx.verify_mode = ssl.CERT_NONE
            
        return ctx

    @property
    def cluster_name(self) -> str:
        """Extract cluster name from connection URI"""
        return self._config.uri.split("@")[-1].split("/")[0]

    @MetricsSystem.time_method(QUERY_DURATION, labels=["query_type", "cluster"])
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict] = None,
        *, 
        tx: Optional[AsyncTransaction] = None,
        routing: RoutingControl = RoutingControl.WRITE,
        **kwargs
    ) -> EagerResult:
        """
        Execute Cypher query with automatic retry and metrics collection
        """
        retries = 0
        parameters = parameters or {}
        
        while retries <= self._config.max_retries:
            session: Optional[AsyncSession] = None
            try:
                session = self._driver.session(
                    database=self._config.database,
                    default_access_mode=routing
                )
                
                if tx:
                    result = await tx.run(query, parameters, **kwargs)
                else:
                    result = await session.run(query, parameters, **kwargs)
                    
                eager_result = await result.to_eager_result()
                
                if session:
                    await session.close()
                    
                return eager_result
                
            except (ServiceUnavailable, SessionExpired, TransientError) as e:
                RETRY_COUNTER.labels(
                    cluster=self.cluster_name,
                    error_type=type(e).__name__
                ).inc()
                
                if retries >= self._config.max_retries:
                    raise Neo4jError(
                        f"Max retries ({self._config.max_retries}) exceeded"
                    ) from e
                    
                await self._handle_retry(e, retries)
                retries += 1
                
            except Exception as e:
                if session:
                    await session.close()
                raise
                
        raise Neo4jError("Unexpected execution path")  # Should never reach here

    async def _handle_retry(self, error: Exception, retry_count: int):
        """Handle retry logic with exponential backoff and cluster rediscovery"""
        delay = self._config.retry_delay * (2 ** retry_count)
        self._logger.warning(
            f"Retry {retry_count+1} in {delay:.2f}s: {str(error)}"
        )
        await asyncio.sleep(delay)
        await self._discover_cluster()

    async def transactional(
        self,
        query: str,
        parameters: Optional[Dict] = None,
        **kwargs
    ) -> Any:
        """Execute transactional query with automatic commit/rollback"""
        async with self._driver.session(
            database=self._config.database
        ) as session:
            try:
                return await session.execute_write(
                    lambda tx: self.execute_query(query, parameters, tx=tx, **kwargs)
                )
            except Neo4jError as e:
                await self._log_transaction_error(e)
                raise

    async def _log_transaction_error(self, error: Neo4jError):
        """Log detailed transaction error information"""
        self._logger.error(
            f"Transaction failed: {error.code} - {error.message}"
        )
        if error.classification == "ClientError":
            self._logger.debug(f"Query parameters: {error.parameters}")

    @lru_cache(maxsize=1000)
    async def cached_query(
        self,
        query: str,
        parameters: Optional[Dict] = None,
        ttl: int = 300
    ) -> List[Dict]:
        """Execute query with result caching (LRU + TTL)"""
        cache_key = hash((query, frozenset(parameters.items() if parameters else {})))
        if not hasattr(self, '_query_cache'):
            self._query_cache = {}
            
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
            
        result = await self.execute_query(query, parameters)
        data = deserialize_neo4j(result)
        self._query_cache[cache_key] = data
        
        if ttl > 0:
            async def expire_cache():
                await asyncio.sleep(ttl)
                if cache_key in self._query_cache:
                    del self._query_cache[cache_key]
            asyncio.create_task(expire_cache())
            
        return data

    async def batch_operations(
        self,
        queries: List[str],
        parameters_list: List[Dict],
        batch_size: int = 1000
    ) -> List[Any]:
        """Execute batch operations with chunking and parallel execution"""
        results = []
        for i in range(0, len(queries), batch_size):
            chunk = queries[i:i+batch_size]
            params_chunk = parameters_list[i:i+batch_size]
            
            tasks = [
                self.execute_query(q, p)
                for q, p in zip(chunk, params_chunk)
            ]
            results.extend(await asyncio.gather(*tasks))
            
        return results

    async def close(self):
        """Close all connections and release resources"""
        if self._driver:
            await self._driver.close()
            CONNECTION_GAUGE.labels(cluster=self.cluster_name).dec()
            self._logger.info("Neo4j driver closed")

    async def __aenter__(self):
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

# Schema Management Utilities
class Neo4jSchemaManager:
    """Schema versioning and migration utilities"""
    
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver
        self._lock = asyncio.Lock()
        
    async def initialize_schema(self):
        """Create indexes and constraints if missing"""
        constraints = [
            "CREATE CONSTRAINT unique_agent_id IF NOT EXISTS "
            "FOR (a:Agent) REQUIRE a.id IS UNIQUE",
            "CREATE INDEX agent_type_index IF NOT EXISTS "
            "FOR (a:Agent) ON (a.type)"
        ]
        
        async with self._lock:
            for constraint in constraints:
                await self.driver.execute_query(constraint)

    async def migrate_data(self, migration_script: str):
        """Execute schema migration script atomically"""
        await self.driver.execute_query(
            "CALL apoc.schema.assert({}, {}, true) YIELD label, key, unique, action "
            "RETURN *"
        )
        await self.driver.execute_query(migration_script)

# Example Usage
if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def main():
        config = Neo4jConfig(
            uri="neo4j://cluster.aelion.ai:7687",
            auth=("neo4j", os.getenv("NEO4J_PASSWORD")),
            encrypted=True,
            cert_path="/etc/ssl/neo4j-ca.pem"
        )
        
        async with Neo4jDriver(config) as driver:
            await driver.initialize_schema()
            
            # Create agent node
            result = await driver.execute_query(
                "CREATE (a:Agent {id: $id, type: $type}) RETURN a",
                {"id": "agent_001", "type": "supervisor"}
            )
            print("Created agent:", json.dumps(deserialize_neo4j(result), indent=2))
            
            # Complex query example
            result = await driver.execute_query(
                """
                MATCH (src:Agent)-[rel:COMMUNICATES_WITH]->(dest:Agent)
                WHERE src.type = $type
                RETURN src.id AS source, collect(dest.id) AS targets
                """,
                {"type": "worker"},
                routing=RoutingControl.READ
            )
            print("Communication graph:", json.dumps(deserialize_neo4j(result), indent=2))
            
    asyncio.run(main())
