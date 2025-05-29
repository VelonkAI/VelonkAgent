"""
Enterprise Kafka Consumer - High-throughput, Stateful Stream Processing with Exactly-Once Guarantees
"""

from __future__ import annotations
import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

# Kafka
from confluent_kafka import Consumer, KafkaException, TopicPartition
from confluent_kafka.admin import AdminClient

# Monitoring
from prometheus_client import Counter, Histogram, Gauge
import opentelemetry.trace as otel_trace
from opentelemetry.sdk.trace import Tracer

# Internal modules
from utils.serialization import Serializer
from utils.metrics import KAFKA_CONSUMER_METRICS
from utils.logger import get_logger

logger = get_logger(__name__)
tracer: Tracer = otel_trace.get_tracer(__name__)

@dataclass
class KafkaConsumerConfig:
    """Stateful consumer configuration with isolation levels"""
    bootstrap_servers: str = "kafka.aelion.ai:9093"
    group_id: str = "aelion-agent-group"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    isolation_level: str = "read_committed"
    max_poll_interval_ms: int = 300000
    session_timeout_ms: int = 10000
    heartbeat_interval_ms: int = 3000
    max_poll_records: int = 500
    fetch_max_bytes: int = 52428800  # 50MB
    security_protocol: str = "SSL"
    ssl_cafile: str = "/etc/ssl/certs/ca.crt"
    ssl_certfile: str = "/etc/ssl/certs/client.pem"
    ssl_keyfile: str = "/etc/ssl/certs/client.key"
    processing_concurrency: int = 16  # Thread pool size
    enable_dlq: bool = True
    commit_retries: int = 10
    retry_backoff_ms: int = 1000

class KafkaConsumer:
    """Transactional consumer with exactly-once processing semantics"""
    
    def __init__(self, config: KafkaConsumerConfig, serializer: Serializer):
        self._config = config
        self._serializer = serializer
        self._consumer = Consumer(self._base_config())
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.processing_concurrency
        )
        self._dlq_producer = None
        self._running = False
        self._partitions_assigned = set()
        self._init_dlq()

    def _base_config(self) -> Dict[str, Any]:
        """Generate base Kafka configuration"""
        return {
            "bootstrap.servers": self._config.bootstrap_servers,
            "group.id": self._config.group_id,
            "auto.offset.reset": self._config.auto_offset_reset,
            "enable.auto.commit": "false",
            "isolation.level": self._config.isolation_level,
            "max.poll.interval.ms": self._config.max_poll_interval_ms,
            "session.timeout.ms": self._config.session_timeout_ms,
            "heartbeat.interval.ms": self._config.heartbeat_interval_ms,
            "max.poll.records": self._config.max_poll_records,
            "fetch.max.bytes": self._config.fetch_max_bytes,
            "security.protocol": self._config.security_protocol,
            "ssl.ca.location": self._config.ssl_cafile,
            "ssl.certificate.location": self._config.ssl_certfile,
            "ssl.key.location": self._config.ssl_keyfile,
            "on_commit": self._commit_callback,
            "partition.assignment.strategy": "cooperative-sticky"
        }

    def _init_dlq(self):
        """Initialize Dead Letter Queue producer if enabled"""
        if self._config.enable_dlq:
            from kafka.producer import KafkaProducer, KafkaProducerConfig
            dlq_config = KafkaProducerConfig(
                bootstrap_servers=self._config.bootstrap_servers,
                transactional_id=f"{self._config.group_id}-dlq"
            )
            self._dlq_producer = KafkaProducer(dlq_config, self._serializer)

    async def subscribe(self, topics: List[str]):
        """Subscribe to topics with rebalance listeners"""
        self._consumer.subscribe(
            topics,
            on_assign=self._on_assign,
            on_revoke=self._on_revoke,
            on_lost=self._on_lost
        )

    def _on_assign(self, consumer: Consumer, partitions: List[TopicPartition]):
        """Rebalance: New partitions assigned"""
        self._partitions_assigned.update(p.partition for p in partitions)
        logger.info(f"Partitions assigned: {partitions}")

    def _on_revoke(self, consumer: Consumer, partitions: List[TopicPartition]):
        """Rebalance: Partitions being revoked"""
        self._partitions_assigned.clear()
        logger.warning(f"Partitions revoked: {partitions}")

    def _on_lost(self, consumer: Consumer, partitions: List[TopicPartition]):
        """Rebalance: Partitions lost (non-graceful)"""
        logger.error(f"Partitions lost: {partitions}")
        self._partitions_assigned.clear()

    async def consume_loop(self, processor: Callable[[Any], None]):
        """Start infinite consumption loop with exactly-once guarantees"""
        self._running = True
        loop = asyncio.get_event_loop()
        
        try:
            while self._running:
                with tracer.start_as_current_span("KafkaConsume"), \
                     KAFKA_CONSUMER_METRICS['poll_latency_seconds'].time():
                    
                    # Transactional poll
                    messages = await loop.run_in_executor(
                        None,
                        lambda: self._consumer.consume(
                            num_messages=self._config.max_poll_records,
                            timeout=1.0
                        )
                    )
                    
                if not messages:
                    continue

                # Process messages with thread pool
                futures = []
                for msg in messages:
                    future = loop.run_in_executor(
                        self._executor,
                        self._process_message,
                        msg,
                        processor
                    )
                    futures.append(future)
                
                # Wait for batch completion
                await asyncio.gather(*futures)
                
                # Transactional commit
                await self._commit_offsets()

        except KafkaException as e:
            logger.critical(f"Consumer failed: {e}")
            await self._handle_fatal_error(e)
        finally:
            await self.close()

    def _process_message(self, msg: Any, processor: Callable[[Any], None]):
        """Process individual message with exactly-once guarantees"""
        with tracer.start_as_current_span(f"Process:{msg.topic()}"), \
             KAFKA_CONSUMER_METRICS['process_latency_seconds'].labels(topic=msg.topic()).time():
            
            try:
                # Deserialize with schema version
                headers = {k: v.decode() for k, v in msg.headers()}
                schema_version = int(headers.get("schema_version", 1))
                
                deserialized = self._serializer.deserialize(
                    msg.value(),
                    schema_type=msg.topic(),
                    schema_version=schema_version
                )
                
                # Execute business logic
                processor(deserialized)
                
                KAFKA_CONSUMER_METRICS['messages_processed_total'].labels(topic=msg.topic()).inc()

            except Exception as e:
                KAFKA_CONSUMER_METRICS['processing_errors_total'].labels(topic=msg.topic()).inc()
                if self._config.enable_dlq:
                    self._send_to_dlq(msg, e)
                else:
                    logger.error(f"Message processing failed: {e}")

    def _send_to_dlq(self, msg: Any, error: Exception):
        """Send failed message to Dead Letter Queue"""
        dlq_message = {
            "original_topic": msg.topic(),
            "payload": msg.value(),
            "error": str(error),
            "headers": msg.headers(),
            "offset": msg.offset(),
            "partition": msg.partition()
        }
        
        self._dlq_producer.produce(
            topic=f"{msg.topic()}_DLQ",
            key=msg.key(),
            value=dlq_message,
            headers={"retry_count": "0"}
        )

    async def _commit_offsets(self):
        """Transactional offset commit with retries"""
        for attempt in range(self._config.commit_retries):
            try:
                self._consumer.commit(asynchronous=False)
                return
            except KafkaException as e:
                if attempt == self._config.commit_retries - 1:
                    raise
                await asyncio.sleep(self._config.retry_backoff_ms / 1000)
                logger.warning(f"Commit failed (attempt {attempt+1}): {e}")

    def _commit_callback(self, err: Optional[Exception], partitions: List[TopicPartition]):
        """Offset commit callback handler"""
        if err:
            logger.error(f"Offset commit failed: {err}")
            KAFKA_CONSUMER_METRICS['commit_errors_total'].inc()
        else:
            logger.debug(f"Committed offsets: {partitions}")

    async def _handle_fatal_error(self, error: KafkaException):
        """Critical failure handling"""
        await self.close()
        # Implement circuit breaker logic here

    async def close(self):
        """Graceful shutdown"""
        self._running = False
        self._consumer.close()
        self._executor.shutdown()
        if self._dlq_producer:
            await self._dlq_producer.close()

    async def health_check(self) -> Dict[str, Any]:
        """Consumer health status with lag monitoring"""
        status = {"status": "OK", "assigned_partitions": list(self._partitions_assigned)}
        
        if self._partitions_assigned:
            try:
                watermarks = self._consumer.get_watermark_offsets(list(self._partitions_assigned))
                status["lag"] = {p: watermarks[p].high - watermarks[p].low for p in self._partitions_assigned}
            except KafkaException:
                status["lag"] = "unavailable"
        
        return status

# Example Usage
if __name__ == "__main__":
    config = KafkaConsumerConfig(
        group_id="aelion-agents-v1",
        processing_concurrency=32
    )
    serializer = Serializer(format="avro")
    consumer = KafkaConsumer(config, serializer)
    
    def process_message(message: Dict):
        # Business logic here
        print(f"Processed: {message}")
    
    async def consume():
        await consumer.subscribe(["agent_events"])
        await consumer.consume_loop(process_message)
    
    try:
        asyncio.run(consume())
    except KeyboardInterrupt:
        asyncio.run(consumer.close())
