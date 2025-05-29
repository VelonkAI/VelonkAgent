"""
Unified Logging System with Structured Logging, Async Handlers & Metrics
"""

from __future__ import annotations
import asyncio
import logging
import sys
import os
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union
from concurrent.futures import ThreadPoolExecutor

# Third-party
from loguru import logger
import structlog
from prometheus_client import Counter, Histogram
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk

# Constants
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOG_FORMAT = "[{level}] [{timestamp}] [{module}:{line}] - {message}"
JSON_LOG_FORMAT = {
    "timestamp": "%(asctime)s",
    "level": "%(levelname)s",
    "module": "%(module)s",
    "line": "%(lineno)d",
    "message": "%(message)s",
    "context": "%(context)s"
}

# Prometheus Metrics
LOG_METRICS = {
    "log_count": Counter("aelion_log_messages_total", "Total log messages", ["level"]),
    "log_volume": Counter("aelion_log_bytes_total", "Total log volume in bytes", ["level"]),
    "log_latency": Histogram("aelion_log_latency_seconds", "Log processing latency")
}

class LogHandlerConfig(BaseModel):
    enabled: bool = True
    level: str = "INFO"
    format: str = "text"  # text/json
    file_path: Optional[Path] = None
    max_size: int = 100  # MB
    retention: int = 7  # Days
    elasticsearch_hosts: Optional[str] = None
    syslog_address: Optional[Tuple[str, int]] = None

class StructuredLogger:
    """Core logging facade with async-capable handlers"""
    
    def __init__(self, config: LogHandlerConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.es_client: Optional[AsyncElasticsearch] = None
        self._init_handlers()
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
    def _init_handlers(self):
        logger.remove()  # Remove default loguru handlers
        
        # Console handler
        if self.config.format == "text":
            logger.add(
                sys.stderr,
                format=DEFAULT_LOG_FORMAT,
                level=self.config.level,
                enqueue=True,  # Async-safe
                backtrace=True,
                diagnose=True
            )
        else:
            logger.add(
                sys.stderr,
                serialize=True,
                format=json.dumps(JSON_LOG_FORMAT),
                level=self.config.level,
                enqueue=True
            )
            
        # File handler with rotation
        if self.config.file_path:
            logger.add(
                str(self.config.file_path),
                rotation=f"{self.config.max_size} MB",
                retention=f"{self.config.retention} days",
                enqueue=True,
                compression="gz",
                level=self.config.level
            )
            
        # Elasticsearch handler
        if self.config.elasticsearch_hosts:
            self.es_client = AsyncElasticsearch(hosts=self.config.elasticsearch_hosts.split(','))
            logger.add(
                self._elasticsearch_sink,
                level=self.config.level,
                format="json",
                enqueue=True
            )
            
        # Syslog handler
        if self.config.syslog_address:
            logger.add(
                self._syslog_sink,
                level=self.config.level,
                enqueue=True
            )
            
    async def _elasticsearch_sink(self, message: Dict[str, Any]):
        """Async bulk insert to Elasticsearch"""
        if not self.es_client:
            return
            
        doc = {
            "_index": f"logs-{datetime.utcnow().strftime('%Y-%m-%d')}",
            "_source": {
                **message,
                "host": socket.gethostname(),
                "service": "aelion-ai"
            }
        }
        
        await async_bulk(self.es_client, [doc])
        
    async def _syslog_sink(self, message: Dict[str, Any]):
        """Async syslog transport (RFC 5424)"""
        # Implement syslog protocol here
        pass
        
    @staticmethod
    def get_logger(name: str = "velink") -> structlog.BoundLogger:
        return structlog.get_logger(name)
        
    def log_metric(self, level: str, message: str):
        """Update Prometheus metrics"""
        LOG_METRICS["log_count"].labels(level=level).inc()
        LOG_METRICS["log_volume"].labels(level=level).inc(len(message.encode('utf-8')))
        
    async def async_log(self, level: str, message: str, **context):
        """Threadpool-executed logging for CPU-bound formatting"""
        loop = asyncio.get_event_loop()
        with LOG_METRICS["log_latency"].time():
            await loop.run_in_executor(
                self.executor,
                lambda: self._log_sync(level, message, **context)
            )
            
    def _log_sync(self, level: str, message: str, **context):
        """Synchronous logging core"""
        log_method = getattr(logger, level.lower())
        log_method(message, **context)
        self.log_metric(level, message)
        
    async def shutdown(self):
        """Graceful cleanup"""
        await self.executor.shutdown(wait=True)
        if self.es_client:
            await self.es_client.close()

# Global instance (configure via environment variables)
def configure_logging():
    config = LogHandlerConfig(
        enabled=os.getenv("LOG_ENABLED", "true").lower() == "true",
        level=os.getenv("LOG_LEVEL", "INFO"),
        format=os.getenv("LOG_FORMAT", "json"),
        file_path=Path(os.getenv("LOG_PATH", "/var/log/aelion/aelion.log")),
        max_size=int(os.getenv("LOG_MAX_SIZE", "100")),
        retention=int(os.getenv("LOG_RETENTION", "7")),
        elasticsearch_hosts=os.getenv("ELASTICSEARCH_HOSTS"),
        syslog_address=(
            os.getenv("SYSLOG_HOST"), 
            int(os.getenv("SYSLOG_PORT", "514"))
        ) if os.getenv("SYSLOG_HOST") else None
    )
    return StructuredLogger(config)

# Usage Example
log = StructuredLogger.get_logger()

async def main():
    logging = configure_logging()
    try:
        log.info("System initialized", component="orchestrator", pid=os.getpid())
        # Contextual logging
        with log.bind(task_id="task-1234"):
            log.error("Failed to process task", error="Timeout", retries=3)
    finally:
        await logging.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
