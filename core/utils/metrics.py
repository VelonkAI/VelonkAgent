"""
Unified Metrics System - Multi-dimensional Telemetry Collection & Exposition
"""

from __future__ import annotations
import asyncio
import time
from contextlib import asynccontextmanager
from functools import wraps
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union
)
from collections import defaultdict
from datetime import datetime

# Third-party
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Summary,
    REGISTRY,
    start_http_server,
    generate_latest,
    CollectorRegistry,
    push_to_gateway
)
from prometheus_client.metrics import MetricWrapperBase
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Type variables
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Awaitable[Any]])

class MetricConfig:
    """Configuration for metric initialization"""
    
    def __init__(
        self,
        name: str,
        description: str,
        labels: List[str] = None,
        buckets: Tuple[float, ...] = None,
        namespace: str = "aelion",
        subsystem: str = "core"
    ):
        self.name = name
        self.description = description
        self.labels = labels or []
        self.buckets = buckets or Histogram.DEFAULT_BUCKETS
        self.namespace = namespace
        self.subsystem = subsystem

class TelemetryManager:
    """Core metrics registry with multi-backend support"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_metrics()
        return cls._instance
    
    def _init_self):
        """Initialize OpenTelemetry SDK if enabled"""
        if self.otel_enabled:
            self.otel_exporter = OTLPMetricExporter(endpoint=self.otel_endpoint)
            self.otel_reader = PeriodicExportingMetricReader(
                exporter=self.otel_exporter,
                export_interval_millis=5000
            )
            self.otel_provider = MeterProvider(metric_readers=[self.otel_reader])
            metrics.set_meter_provider(self.otel_provider)
            self.otel_meter = metrics.get_meter("aelion.ai")
    
    def create_metric(self, config: MetricConfig) -> MetricWrapperBase:
        """Factory method for metric creation with conflict detection"""
        metric_key = f"{config.namespace}_{config.subsystem}_{config.name}"
        
        if metric_key in self._metrics:
            raise ValueError(f"Metric {metric_key} already registered")
            
        if config.name.endswith("_total"):
            metric = Counter(
                config.name,
                config.description,
                config.labels,
                namespace=config.namespace,
                subsystem=config.subsystem
            )
        elif config.name.endswith("_seconds"):
            metric = Histogram(
                config.name,
                config.description,
                config.labels,
                buckets=config.buckets,
                namespace=config.namespace,
                subsystem=config.subsystem
            )
        else:
            metric = Gauge(
                config.name,
                config.description,
                config.labels,
                namespace=config.namespace,
                subsystem=config.subsystem
            )
            
        self._metrics[metric_key] = metric
        return metric
    
    @staticmethod
    def http_handler():
        """Generate Prometheus metrics endpoint for FastAPI/Starlette"""
        async def handler(request):
            return generate_latest(REGISTRY)
        return handler
    
    def push_gateway_job(self, job_name: str):
        """Configure periodic push to Prometheus PushGateway"""
        push_to_gateway(
            self.push_gateway,
            job=job_name,
            registry=REGISTRY,
            timeout=30
        )
    
    @asynccontextmanager
    async def latency_metric(self, metric_name: str, **labels):
        """Async context manager for latency measurement"""
        start_time = time.monotonic()
        try:
            yield
        finally:
            latency = time.monotonic() - start_time
            self._metrics[metric_name].labels(**labels).observe(latency)
    
    def track_concurrency(self, metric_name: str):
        """Decorator for tracking concurrent executions"""
        def decorator(func: AsyncF) -> AsyncF:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                concurrency_gauge = self._metrics[f"{metric_name}_concurrent"]
                concurrency_gauge.inc()
                try:
                    return await func(*args, **kwargs)
                finally:
                    concurrency_gauge.dec()
            return wrapper
        return decorator
    
    def error_counter(self, metric_name: str):
        """Decorator for counting exceptions"""
        def decorator(func: F) -> F:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    self._metrics[metric_name].labels(
                        error_type=e.__class__.__name__
                    ).inc()
                    raise
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._metrics[metric_name].labels(
                        error_type=e.__class__.__name__
                    ).inc()
                    raise
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def reset_metrics(self):
        """Clear all metric data (for testing only)"""
        for metric in self._metrics.values():
            metric.clear()

# Predefined core metrics
CORE_METRICS = [
    MetricConfig(
        name="agent_actions_total",
        description="Total agent actions executed",
        labels=["agent_type", "action_type"]
    ),
    MetricConfig(
        name="task_duration_seconds",
        description="Task processing latency distribution",
        labels=["task_type", "priority"],
        buckets=(0.01, 0.1, 0.5, 1, 5, 10, 30)
    ),
    MetricConfig(
        name="resource_usage_bytes",
        description="Memory/CPU resource consumption",
        labels=["resource_type", "node"]
    ),
    MetricConfig(
        name="api_errors_total",
        description="API endpoint errors",
        labels=["endpoint", "status_code"]
    )
]

# Initialize core metrics
manager = TelemetryManager()
for metric_config in CORE_METRICS:
    manager.create_metric(metric_config)

# Example usage in FastAPI
from fastapi import FastAPI

app = FastAPI()
app.add_route("/metrics", manager.http_handler())

@app.get("/health")
@manager.error_counter("api_errors_total")
async def health_check():
    with manager.latency_metric("http_request_duration_seconds", endpoint="/health"):
        return {"status": "ok"}

# Example usage in async tasks
@manager.track_concurrency("background_tasks")
@manager.error_counter("task_errors_total")
async def process_task(task_data: dict):
    async with manager.latency_metric("task_processing_time", task_type="data_ingest"):
        # Processing logic
        pass

# Push metrics to gateway (cron job)
manager.push_gateway_job("aelion-ai-prod")
