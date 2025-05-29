"""
Enterprise Load Testing Framework - Multi-protocol Distributed Benchmarking
"""

from __future__ import annotations
import os
import sys
import time
import random
import asyncio
import resource
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import gevent
from gevent import monkey; monkey.patch_all()
import locust
from locust import User, task, events, runners
from locust.env import Environment
from locust.runners import MasterRunner, WorkerRunner
import requests
import grpc
import pandas as pd
import matplotlib.pyplot as plt
from prometheus_client import start_http_server, Summary, Counter, Gauge
from kafka import KafkaProducer
from graphqlclient import GraphQLClient

# Configuration
CONFIG = {
    "target_host": os.getenv("LOAD_TEST_TARGET", "https://api.aelion.ai"),
    "max_users": int(os.getenv("MAX_USERS", "10000")),
    "spawn_rate": int(os.getenv("SPAWN_RATE", "100")),
    "duration": int(os.getenv("DURATION", "300")),
    "test_scenarios": {
        "grpc": 0.7,    # 70% gRPC traffic
        "rest": 0.2,    # 20% REST API
        "graphql": 0.05, # 5% GraphQL
        "kafka": 0.05   # 5% Kafka messaging
    },
    "prometheus_port": 9100,
    "distributed": os.getenv("DISTRIBUTED", "false").lower() == "true",
    "redis_host": os.getenv("REDIS_HOST", "redis://metrics.aelion.ai:6379/0")
}

# Prometheus Metrics
REQUEST_LATENCY = Summary('loadtest_latency_seconds', 'Request latency', ['method', 'endpoint'])
ERROR_COUNTER = Counter('loadtest_errors_total', 'Total errors', ['method', 'endpoint', 'error_code'])
ACTIVE_USERS = Gauge('loadtest_active_users', 'Concurrent active users')
THROUGHPUT = Gauge('loadtest_requests_per_second', 'Requests per second')

@dataclass
class TestResult:
    total_requests: int = 0
    failed_requests: int = 0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    avg_latency: float = 0.0
    percentile_95: float = 0.0
    percentile_99: float = 0.0

class ProtocolMixin:
    """Base class for protocol implementations"""
    
    @staticmethod
    def _record_metrics(start_time: float, method: str, endpoint: str):
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method, endpoint).observe(latency)
        THROUGHPUT.inc()

    @staticmethod
    def _record_error(method: str, endpoint: str, error_code: int):
        ERROR_COUNTER.labels(method, endpoint, str(error_code)).inc()

class GRPCTestUser(User, ProtocolMixin):
    """gRPC protocol load testing"""
    
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        channel = grpc.insecure_channel(CONFIG['target_host'])
        self.stub = agent_pb2_grpc.AgentServiceStub(channel)
    
    @task
    def execute_agent_task(self):
        start_time = time.time()
        try:
            request = agent_pb2.TaskRequest(
                task_id=f"loadtest-{random.randint(1, 1000000)}",
                payload=os.urandom(1024)  # 1KB random payload
            )
            response = self.stub.ExecuteTask(request)
            self._record_metrics(start_time, "grpc", "ExecuteTask")
        except grpc.RpcError as e:
            self._record_error("grpc", "ExecuteTask", e.code().value[0])

class RESTTestUser(User, ProtocolMixin):
    """REST API load testing"""
    
    abstract = True
    
    @task
    def create_agent(self):
        start_time = time.time()
        try:
            response = self.client.post(
                "/v1/agents",
                json={"type": "worker", "capabilities": ["nlp", "vision"]},
                headers={"Authorization": f"Bearer {os.getenv('API_TOKEN')}"}
            )
            if response.status_code != 201:
                self._record_error("rest", "create_agent", response.status_code)
            else:
                self._record_metrics(start_time, "rest", "create_agent")
        except requests.RequestException as e:
            self._record_error("rest", "create_agent", 500)

class GraphQLTestUser(User, ProtocolMixin):
    """GraphQL load testing"""
    
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client = GraphQLClient(CONFIG['target_host'])
    
    @task
    def query_agents(self):
        start_time = time.time()
        query = '''
        query {
            agents(filter: {status: ACTIVE}) {
                id
                tasks(first: 5) {
                    id
                    status
                }
            }
        }
        '''
        try:
            self.client.execute(query)
            self._record_metrics(start_time, "graphql", "query_agents")
        except Exception as e:
            self._record_error("graphql", "query_agents", 500)

class KafkaTestUser(User):
    """Kafka producer load testing"""
    
    abstract = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS'),
            security_protocol="SSL",
            ssl_cafile="certs/kafka/ca.pem",
            ssl_certfile="certs/kafka/service.cert",
            ssl_keyfile="certs/kafka/service.key"
        )
    
    @task
    def produce_message(self):
        start_time = time.time()
        try:
            future = self.producer.send(
                'agent_events',
                key=os.urandom(16),
                value=os.urandom(1024)
            )
            future.get(timeout=10)
            THROUGHPUT.inc()
        except Exception as e:
            ERROR_COUNTER.labels("kafka", "produce_message", str(type(e))).inc()

class MixedWorkload(locust.FastHttpUser):
    """Combined workload with adaptive ratios"""
    
    tasks = {
        GRPCTestUser: CONFIG['test_scenarios']['grpc'],
        RESTTestUser: CONFIG['test_scenarios']['rest'],
        GraphQLTestUser: CONFIG['test_scenarios']['graphql'],
        KafkaTestUser: CONFIG['test_scenarios']['kafka']
    }

class ResourceMonitor:
    """System resource utilization tracking"""
    
    def __init__(self):
        self.max_cpu = 0.0
        self.max_mem = 0.0
        
    def start(self):
        while True:
            self._record_resources()
            time.sleep(5)
    
    def _record_resources(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        current_cpu = usage.ru_utime + usage.ru_stime
        current_mem = usage.ru_maxrss / 1024  # Convert to MB
        
        self.max_cpu = max(self.max_cpu, current_cpu)
        self.max_mem = max(self.max_mem, current_mem)

class DistributedCoordinator:
    """Cross-node test coordination"""
    
    def __init__(self):
        self.redis = redis.Redis.from_url(CONFIG['redis_host'])
        self.lock = self.redis.lock('loadtest-coordination')
        
    def aggregate_results(self) -> TestResult:
        with self.lock:
            results = self.redis.get('aggregate_results') or TestResult()
            return TestResult(**msgpack.unpackb(results))
    
    def update_progress(self, partial_result: TestResult):
        with self.lock:
            current = self.aggregate_results()
            updated = TestResult(
                total_requests=current.total_requests + partial_result.total_requests,
                failed_requests=current.failed_requests + partial_result.failed_requests,
                min_latency=min(current.min_latency, partial_result.min_latency),
                max_latency=max(current.max_latency, partial_result.max_latency),
                avg_latency=(current.avg_latency * current.total_requests + 
                            partial_result.avg_latency * partial_result.total_requests) /
                          (current.total_requests + partial_result.total_requests)
            )
            self.redis.set('aggregate_results', msgpack.packb(updated))

def analyze_results(raw_data: List[Dict]) -> TestResult:
    """Statistical analysis of test results"""
    df = pd.DataFrame(raw_data)
    result = TestResult()
    
    if not df.empty:
        result.total_requests = len(df)
        result.failed_requests = df[df['success'] == False].count()
        result.min_latency = df['latency'].min()
        result.max_latency = df['latency'].max()
        result.avg_latency = df['latency'].mean()
        result.percentile_95 = df['latency'].quantile(0.95)
        result.percentile_99 = df['latency'].quantile(0.99)
    
    return result

def generate_report(result: TestResult):
    """Generate HTML/PDF report with visualizations"""
    plt.figure(figsize=(12, 8))
    
    # Latency distribution
    plt.subplot(2, 2, 1)
    plt.hist(result.latency_samples, bins=50, alpha=0.75)
    plt.title('Request Latency Distribution')
    
    # Error breakdown
    plt.subplot(2, 2, 2)
    error_counts = pd.Series(result.errors).value_counts()
    error_counts.plot.pie(autopct='%1.1f%%')
    
    # Throughput timeline
    plt.subplot(2, 2, 3)
    plt.plot(result.throughput_timeline)
    plt.title('Requests per Second')
    
    # Save report
    plt.tight_layout()
    plt.savefig('loadtest_report.pdf')
    plt.savefig('loadtest_report.png')

@events.init.add_listener
def on_locust_init(environment: Environment, **kwargs):
    """Initialization handler"""
    if isinstance(environment.runner, MasterRunner):
        start_http_server(CONFIG['prometheus_port'])
        environment.coordinator = DistributedCoordinator()
        environment.resource_monitor = ResourceMonitor()
        gevent.spawn(environment.resource_monitor.start)

@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """Test start handler"""
    environment.raw_results = []
    environment.start_time = time.time()

@events.request.add_listener
def on_request(**kwargs):
    """Record individual request metrics"""
    environment = locust.env.env
    if kwargs.get('exception'):
        error_code = getattr(kwargs['exception'], 'status_code', 500)
        ERROR_COUNTER.labels(
            kwargs.get('method', 'unknown'),
            kwargs.get('name', 'unknown'),
            str(error_code)
        ).inc()

@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Final analysis and reporting"""
    duration = time.time() - environment.start_time
    result = analyze_results(environment.raw_results)
    
    print(f"\n{'='*40}")
    print(f"Load Test Summary ({CONFIG['duration']}s)")
    print(f"Total Requests: {result.total_requests}")
    print(f"Failed Requests: {result.failed_requests} ({result.failed_requests/result.total_requests:.2%})")
    print(f"Latency (ms): Avg={result.avg_latency*1000:.2f} | 95%={result.percentile_95*1000:.2f} | 99%={result.percentile_99*1000:.2f}")
    print(f"Throughput: {result.total_requests/duration:.2f} req/s")
    print(f"Max CPU Usage: {environment.resource_monitor.max_cpu:.2f}%")
    print(f"Max Memory Usage: {environment.resource_monitor.max_mem:.2f}MB")
    
    generate_report(result)
    
    # Upload to S3
    os.system(f"aws s3 cp loadtest_report.pdf s3://aelion-loadtests/{time.strftime('%Y%m%d-%H%M%S')}.pdf")

if __name__ == "__main__":
    env = Environment(user_classes=[MixedWorkload])
    env.create_local_runner()
    
    if CONFIG['distributed']:
        env.create_worker_runner("worker-node-1", 8080)
        
    env.runner.start(
        user_count=CONFIG['max_users'],
        spawn_rate=CONFIG['spawn_rate'],
        host=CONFIG['target_host']
    )
    
    gevent.spawn_later(CONFIG['duration'], lambda: env.runner.quit())
    env.runner.greenlet.join()
