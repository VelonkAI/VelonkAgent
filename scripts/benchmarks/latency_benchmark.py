"""
Enterprise Latency Benchmark - Multi-protocol Performance Profiling & Statistical Analysis
"""

from __future__ import annotations
import os
import sys
import time
import random
import asyncio
import statistics
import resource
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Deque
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import prometheus_client as prom
from kafka import KafkaProducer, KafkaConsumer
import grpc
import requests
from graphqlclient import GraphQLClient
from hdrh.histogram import HdrHistogram
from hdrh.log import HistogramLogWriter
import psutil

# Configuration
CONFIG = {
    "target_host": os.getenv("BENCHMARK_TARGET", "https://api.aelion.ai"),
    "concurrency": int(os.getenv("CONCURRENCY", "100")),
    "duration": int(os.getenv("DURENCY_SECONDS", "300")),
    "warmup_seconds": int(os.getenv("WARMUP_SECONDS", "30")),
    "percentiles": [50, 75, 90, 95, 99, 99.9],
    "protocols": {
        "grpc": True,
        "rest": True,
        "graphql": True,
        "kafka": True
    },
    "histogram_precision": 3,
    "prometheus_port": 9101,
    "resolution_ms": 1000,
    "sliding_window_size": 60,
    "outlier_threshold": 3.0
}

# Prometheus Metrics
LATENCY_HISTOGRAM = prom.Histogram(
    'benchmark_latency_seconds',
    'Request latency distribution',
    ['protocol', 'operation'],
    buckets=(
        0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, float('inf')
    )
)
ERROR_COUNTER = prom.Counter(
    'benchmark_errors_total',
    'Total benchmark errors',
    ['protocol', 'error_type']
)
THROUGHPUT_GAUGE = prom.Gauge(
    'benchmark_throughput_rps',
    'Requests per second per protocol',
    ['protocol']
)

@dataclass
class LatencyStats:
    protocol: str
    count: int = 0
    min: float = float('inf')
    max: float = 0.0
    mean: float = 0.0
    std_dev: float = 0.0
    percentiles: Dict[float, float] = None
    histogram: HdrHistogram = None
    time_series: Deque[Tuple[float, float]] = None

class BaseBenchmark:
    def __init__(self):
        self.histogram = HdrHistogram(1, 120_000_000, CONFIG['histogram_precision'])
        self.time_series = deque(maxlen=CONFIG['sliding_window_size'])
        self._stop_flag = False
        
    async def warmup(self):
        """Protocol-specific warmup phase"""
        pass
        
    async def run_operation(self):
        """Protocol-specific operation to measure"""
        raise NotImplementedError
        
    def record_latency(self, latency: float):
        """Record latency with multiple precision mechanisms"""
        self.histogram.record_value(int(latency * 1000))  # Convert to ms
        self.time_series.append((time.time(), latency))
        LATENCY_HISTOGRAM.labels(self.protocol_name, self.op_name).observe(latency)
        
    def calculate_stats(self) -> LatencyStats:
        """Generate comprehensive statistics"""
        percentiles = {
            p: self.histogram.get_value_at_percentile(p) / 1000.0
            for p in CONFIG['percentiles']
        }
        
        return LatencyStats(
            protocol=self.protocol_name,
            count=self.histogram.total_count,
            min=self.histogram.get_min_value() / 1000.0,
            max=self.histogram.get_max_value() / 1000.0,
            mean=self.histogram.get_mean_value() / 1000.0,
            std_dev=self.histogram.get_stddev_value() / 1000.0,
            percentiles=percentiles,
            histogram=self.histogram,
            time_series=self.time_series
        )

class GRPCBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.protocol_name = "grpc"
        self.op_name = "ExecuteTask"
        channel = grpc.insecure_channel(CONFIG['target_host'])
        self.stub = agent_pb2_grpc.AgentServiceStub(channel)
        
    async def run_operation(self):
        start_time = time.time()
        try:
            request = agent_pb2.TaskRequest(
                task_id=f"bench-{random.randint(1, 1e6)}",
                payload=os.urandom(1024)
            )
            await self.stub.ExecuteTask(request)
            latency = time.time() - start_time
            self.record_latency(latency)
        except grpc.RpcError as e:
            ERROR_COUNTER.labels(self.protocol_name, e.code().name).inc()

class RESTBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.protocol_name = "rest"
        self.op_name = "CreateAgent"
        self.session = requests.Session()
        
    async def run_operation(self):
        start_time = time.time()
        try:
            response = self.session.post(
                f"{CONFIG['target_host']}/v1/agents",
                json={"type": "worker"},
                timeout=5
            )
            response.raise_for_status()
            latency = time.time() - start_time
            self.record_latency(latency)
        except requests.RequestException as e:
            error_type = "timeout" if isinstance(e, requests.Timeout) else "http_error"
            ERROR_COUNTER.labels(self.protocol_name, error_type).inc()

class KafkaBenchmark(BaseBenchmark):
    def __init__(self):
        super().__init__()
        self.protocol_name = "kafka"
        self.op_name = "ProduceMessage"
        self.producer = KafkaProducer(
            bootstrap_servers=os.getenv('KAFKA_BROKERS'),
            security_protocol="SSL",
            ssl_cafile="certs/kafka/ca.pem",
            ssl_certfile="certs/kafka/service.cert",
            ssl_keyfile="certs/kafka/service.key"
        )
        self.consumer = KafkaConsumer(
            'latency_metrics',
            group_id='benchmark-group',
            auto_offset_reset='earliest'
        )
        
    async def run_operation(self):
        correlation_id = str(random.randint(1, 1e9))
        start_time = time.time()
        
        try:
            future = self.producer.send(
                'agent_commands',
                key=correlation_id.encode(),
                value=os.urandom(1024)
            )
            future.get(timeout=10)
            
            # Wait for response
            for msg in self.consumer:
                if msg.key.decode() == correlation_id:
                    latency = time.time() - start_time
                    self.record_latency(latency)
                    break
        except Exception as e:
            ERROR_COUNTER.labels(self.protocol_name, "produce_error").inc()

class BenchmarkRunner:
    def __init__(self):
        self.benchmarks = []
        self.results = {}
        self.resource_stats = {
            'cpu': [],
            'memory': []
        }
        
        if CONFIG['protocols']['grpc']:
            self.benchmarks.append(GRPCBenchmark())
        if CONFIG['protocols']['rest']:
            self.benchmarks.append(RESTBenchmark())
        if CONFIG['protocols']['kafka']:
            self.benchmarks.append(KafkaBenchmark())
            
        self.executor = ThreadPoolExecutor(max_workers=CONFIG['concurrency'])
        prom.start_http_server(CONFIG['prometheus_port'])
        
    async def resource_monitor(self):
        """Track system resource utilization"""
        while True:
            self.resource_stats['cpu'].append(psutil.cpu_percent())
            self.resource_stats['memory'].append(psutil.virtual_memory().percent)
            await asyncio.sleep(1)
            
    async def run_benchmark(self):
        """Main benchmark execution loop"""
        # Warmup phase
        print(f"Starting {CONFIG['warmup_seconds']}s warmup...")
        warmup_tasks = [b.warmup() for b in self.benchmarks]
        await asyncio.gather(*warmup_tasks)
        
        # Main measurement
        print("Starting benchmark...")
        start_time = time.time()
        tasks = []
        
        # Create worker tasks
        for _ in range(CONFIG['concurrency']):
            for bench in self.benchmarks:
                tasks.append(asyncio.create_task(self._worker_loop(bench)))
                
        # Run until duration reached
        while time.time() - start_time < CONFIG['duration']:
            await asyncio.sleep(1)
            THROUGHPUT_GAUGE.labels('all').set(
                sum(b.histogram.total_count for b in self.benchmarks) / 
                (time.time() - start_time)
            )
            
        # Collect results
        for bench in self.benchmarks:
            self.results[bench.protocol_name] = bench.calculate_stats()
            
    async def _worker_loop(self, benchmark: BaseBenchmark):
        """Continuous operation executor"""
        while not benchmark._stop_flag:
            start_time = time.time()
            await benchmark.run_operation()
            
            # Maintain target rate
            elapsed = time.time() - start_time
            if elapsed < 1/CONFIG['concurrency']:
                await asyncio.sleep(1/CONFIG['concurrency'] - elapsed)
                
    def analyze_outliers(self):
        """Statistical outlier detection using Z-score"""
        for protocol, stats in self.results.items():
            if stats.count < 10:
                continue
                
            z_scores = np.abs(stats.zscores)
            outliers = np.where(z_scores > CONFIG['outlier_threshold'])[0]
            print(f"Found {len(outliers)} outliers in {protocol}")
            
    def generate_report(self):
        """Generate comprehensive latency report"""
        plt.figure(figsize=(15, 10))
        
        # Latency Distribution
        plt.subplot(2, 2, 1)
        for protocol, stats in self.results.items():
            counts, bins = np.histogram(
                [x/1000 for x in stats.histogram.get_recorded_values()], 
                bins=50
            )
            plt.stairs(counts, bins, label=protocol)
        plt.xlabel('Latency (s)')
        plt.title('Latency Distribution by Protocol')
        plt.legend()
        
        # Time Series
        plt.subplot(2, 2, 2)
        for protocol, stats in self.results.items():
            timestamps = [t[0] for t in stats.time_series]
            latencies = [t[1] for t in stats.time_series]
            plt.plot(timestamps, latencies, label=protocol, alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Latency (s)')
        plt.title('Latency Time Series')
        
        # Resource Usage
        plt.subplot(2, 2, 3)
        plt.plot(self.resource_stats['cpu'], label='CPU %')
        plt.plot(self.resource_stats['memory'], label='Memory %')
        plt.xlabel('Time (s)')
        plt.title('Resource Utilization')
        plt.legend()
        
        # Statistical Summary
        plt.subplot(2, 2, 4)
        data = []
        for protocol, stats in self.results.items():
            row = {
                'Protocol': protocol,
                'Count': stats.count,
                'Min (s)': stats.min,
                'Max (s)': stats.max,
                'Mean (s)': stats.mean,
                'p99 (s)': stats.percentiles[99]
            }
            data.append(row)
        df = pd.DataFrame(data)
        plt.table(cellText=df.values,
                 colLabels=df.columns,
                 loc='center',
                 cellLoc='center')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('latency_report.pdf')
        plt.savefig('latency_report.png')
        
        # Save raw histograms
        for protocol, stats in self.results.items():
            with open(f'{protocol}_histogram.log', 'w') as f:
                writer = HistogramLogWriter(f)
                writer.output_interval_histogram(
                    stats.histogram.get_start_time_stamp(),
                    stats.histogram.get_end_time_stamp(),
                    stats.histogram,
                    ticks_per_second=1000
                )

async def main():
    runner = BenchmarkRunner()
    monitor_task = asyncio.create_task(runner.resource_monitor())
    await runner.run_benchmark()
    monitor_task.cancel()
    
    runner.analyze_outliers()
    runner.generate_report()
    
    print("\n=== Benchmark Summary ===")
    for protocol, stats in runner.results.items():
        print(f"\n{protocol.upper()} Protocol:")
        print(f"  Requests: {stats.count}")
        print(f"  Latency (s): Min={stats.min:.3f}, Max={stats.max:.3f}")
        print(f"  p99: {stats.percentiles[99]:.3f}s")
        print(f"  Std Dev: {stats.std_dev:.3f}s")

if __name__ == "__main__":
    asyncio.run(main())
