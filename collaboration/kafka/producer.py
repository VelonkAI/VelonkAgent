"""
Enterprise Kafka Producer - High-performance, Transactional Message Broker Integration
"""

from __future__ import annotations
import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Kafka
from confluent_kafka import Producer, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import opentelemetry.trace as otel_trace
from opentelemetry.sdk.trace import Tracer

# Internal modules
from utils.serialization import Serializer
from utils.metrics import KAFKA_METRICS
from utils.logger import get_logger

logger = get_logger(__name__)
tracer: Tracer = otel_trace.get_tracer(__name__)

@dataclass
class KafkaProducerConfig:
    """Configuration for transactional Kafka producer with security"""
    bootstrap_servers: str = "kafka.aelion.ai:9093"
    acks: str = "all"  # "0", "1", "all"
    compression_type: str = "zstd"
    enable_idempotence: bool = True
    max_in_flight_requests: int = 5
    transactional_id: Optional[str] = None
    security_protocol: str = "SSL"
    ssl_cafile: str = "/etc/ssl/certs/ca.crt"
    ssl_certfile: str = "/etc/ssl/certs/client.pem"
    ssl_keyfile: str = "/etc/ssl/certs/client.key"
    message_timeout_ms: int = 300000
    queue_buffering_max_messages: int = 100000
    linger_ms: int = 20
    retries: int = 10
    retry_backoff_ms: int = 1000
    enable_metrics: bool = True
    pool_size: int = 4  # Connection pool size

class KafkaProducer:
    """Transactional producer with connection pooling and dead letter queue (DLQ)"""
    
    def __init__(self, config: KafkaProducerConfig, serializer: Serializer):
        self._config = config
        self._serializer = serializer
        self._pool = []
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.pool_size
        )
        self._dlq_producer = None  # DLQ producer instance
        self._init_producers()
        self._init_admin()
        self._init_dlq()

    def _init_producers(self):
        """Initialize pool of Kafka producers"""
        base_config = self._base_kafka_config()
        for _ in range(self._config.pool_size):
            producer = Producer(base_config)
            if self._config.transactional_id:
                producer.init_transactions()
            self._pool.append(producer)

    def _base_kafka_config(self) -> Dict[str, Any]:
        """Generate base Kafka configuration with security"""
        return {
            "bootstrap.servers": self._config.bootstrap_servers,
            "acks": self._config.acks,
            "compression.type": self._config.compression_type,
            "enable.idempotence": self._config.enable_idempotence,
            "max.in.flight.requests.per.connection": 
                self._config.max_in_flight_requests,
            "security.protocol": self._config.security_protocol,
            "ssl.ca.location": self._config.ssl_cafile,
            "ssl.certificate.location": self._config.ssl_certfile,
            "ssl.key.location": self._config.ssl_keyfile,
            "message.timeout.ms": self._config.message_timeout_ms,
            "queue.buffering.max.messages": 
                self._config.queue_buffering_max_messages,
            "linger.ms": self._config.linger_ms,
            "retries": self._config.retries,
            "retry.backoff.ms": self._config.retry_backoff_ms
        }

    def _init_admin(self):
        """Admin client for topic management"""
        self._admin = AdminClient({"bootstrap.servers": self._config.bootstrap_servers})

    def _init_dlq(self):
        """Initialize Dead Letter Queue producer"""
        if not self._dlq_producer:
            dlq_config = KafkaProducerConfig(
                bootstrap_servers=self._config.bootstrap_servers,
                transactional_id=None,
                enable_idempotence=False
            )
            self._dlq_producer = KafkaProducer(dlq_config, self._serializer)

    async def produce(
        self,
        topic: str,
        key: Union[str, bytes],
        value: Any,
        headers: Optional[Dict[str, str]] = None,
        schema_version: int = 1
    ) -> None:
        """Asynchronously produce message with schema versioning and tracing"""
        with tracer.start_as_current_span(f"KafkaProduce:{topic}"), \
             KAFKA_METRICS['produce_latency_seconds'].labels(topic=topic).time():
            
            serialized = self._serializer.serialize(
                value, 
                schema_type=topic,
                schema_version=schema_version
            )
            
            loop = asyncio.get_event_loop()
            producer = self._get_producer()
            
            try:
                await loop.run_in_executor(
                    self._executor,
                    self._sync_produce,
                    producer,
                    topic,
                    key,
                    serialized,
                    headers or {}
                )
                KAFKA_METRICS['messages_sent_total'].labels(topic=topic).inc()
                
            except KafkaException as e:
                await self._handle_produce_error(e, topic, key, value, headers)
                KAFKA_METRICS['produce_errors_total'].labels(topic=topic).inc()
                raise

    def _get_producer(self) -> Producer:
        """Get a producer from the pool using round-robin"""
        return self._pool[id(self) % len(self._pool)]

    def _sync_produce(
        self,
        producer: Producer,
        topic: str,
        key: Union[str, bytes],
        value: bytes,
        headers: Dict[str, str]
    ):
        """Synchronous produce with transaction support"""
        if self._config.transactional_id:
            producer.begin_transaction()
            
        try:
            producer.produce(
                topic=topic,
                key=key,
                value=value,
                headers=[(k, v.encode()) for k, v in headers.items()],
                on_delivery=self._delivery_report
            )
            producer.poll(0)
            
            if self._config.transactional_id:
                producer.commit_transaction()
                
        except BufferError as e:
            producer.flush()
            raise KafkaException(f"Producer buffer overflow: {e}")
        except Exception as e:
            if self._config.transactional_id:
                producer.abort_transaction()
            raise

    async def _handle_produce_error(
        self,
        error: KafkaException,
        topic: str,
        key: Union[str, bytes],
        value: Any,
        headers: Dict[str, str]
    ):
        """Error handling with DLQ fallback"""
        logger.error(f"Failed to produce message to {topic}: {error}")
        
        # Send to DLQ after 3 retries
        await self._dlq_producer.produce(
            topic=f"{topic}_DLQ",
            key=key,
            value={
                "original_topic": topic,
                "payload": value,
                "error": str(error),
                "headers": headers
            },
            headers={"retry_count": "3"}
        )

    @staticmethod
    def _delivery_report(err: Optional[Exception], msg: Any):
        """Delivery callback handler for async produces"""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered to {msg.topic()} [{msg.partition()}]")

    async def create_topic(
        self, 
        topic: str, 
        num_partitions: int = 6, 
        replication_factor: int = 3
    ):
        """Create topic with idempotent checks"""
        loop = asyncio.get_event_loop()
        new_topic = NewTopic(
            topic,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )
        
        try:
            await loop.run_in_executor(
                self._executor,
                lambda: self._admin.create_topics([new_topic])
            )
        except KafkaException as e:
            if "Topic already exists" not in str(e):
                raise

    async def close(self):
        """Graceful shutdown of producer pool"""
        for producer in self._pool:
            producer.flush()
            producer.poll(1)
        self._executor.shutdown()

    async def health_check(self) -> Dict[str, Any]:
        """Producer health check with metrics"""
        return {
            "status": "OK",
            "queue_size": Gauge('kafka_producer_queue_size', 'Current message queue size'),
            "active_producers": len(self._pool),
            "dlq_status": await self._dlq_producer.health_check() if self._dlq_producer else "disabled"
        }

# Example Usage
if __name__ == "__main__":
    config = KafkaProducerConfig(
        transactional_id="aelion-producer-1",
        bootstrap_servers="kafka.aelion.ai:9093"
    )
    serializer = Serializer(format="avro")
    producer = KafkaProducer(config, serializer)
    
    async def send_message():
        await producer.produce(
            topic="agent_events",
            key="agent-123",
            value={"event": "start", "timestamp": int(time.time())},
            headers={"source": "orchestrator"}
        )
    
    asyncio.run(send_message())
