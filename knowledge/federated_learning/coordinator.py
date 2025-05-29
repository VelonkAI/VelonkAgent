"""
Enterprise FL Coordinator - Secure Multi-party Federated Learning Orchestrator
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
import os
import pickle
from collections import defaultdict
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.shashes import SHA256
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, ValidationError
from ..utils.logger import Logger
from ..utils.metrics import MetricsSystem
from ..utils.serialization import SecureSerializer
from ..grpc.client import GRPCClient
from ..kafka.producer import KafkaProducer

@dataclass(frozen=True)
class FLNode:
    node_id: str
    public_key: bytes
    metadata: Dict[str, Any]
    last_contact: datetime = datetime.utcnow()

@dataclass
class FLRoundConfig:
    round_id: str
    model_class: str
    model_version: int
    node_selection: str = "random"
    sample_size: int = 10
    max_duration: timedelta = timedelta(minutes=30)
    aggregation_strategy: str = "fedavg"
    differential_privacy: Optional[Dict[str, float]] = None
    compression: Dict[str, Any] = field(default_factory=lambda: {"algorithm": "none"})

class FLNodeRegistry:
    def __init__(self):
        self._nodes: Dict[str, FLNode] = {}
        self._logger = Logger(__name__)
        self._metrics = MetricsSystem(["node_registration_total", "node_heartbeat_latency"])

    async def register_node(self, node_id: str, public_key: bytes, metadata: Dict) -> FLNode:
        if node_id in self._nodes:
            self._logger.warning(f"Node {node_id} attempted duplicate registration")
            raise ValueError("Node already registered")

        node = FLNode(
            node_id=node_id,
            public_key=public_key,
            metadata=metadata
        )
        self._nodes[node_id] = node
        self._metrics.inc("node_registration_total", labels=[metadata.get("node_type", "unknown")])
        return node

    async def deregister_node(self, node_id: str):
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._logger.info(f"Node {node_id} deregistered")

    async def update_heartbeat(self, node_id: str):
        if node_id in self._nodes:
            self._nodes[node_id] = FLNode(
                **{**self._nodes[node_id].__dict__, "last_contact": datetime.utcnow()}
            )
            self._metrics.timer("node_heartbeat_latency").observe(
                (datetime.utcnow() - self._nodes[node_id].last_contact).total_seconds()
            )

    async def select_nodes(self, strategy: str, count: int) -> List[FLNode]:
        active_nodes = [n for n in self._nodes.values() 
                      if datetime.utcnow() - n.last_contact < timedelta(minutes=5)]
        
        if strategy == "random":
            selected = np.random.choice(active_nodes, min(count, len(active_nodes)), replace=False)
        elif strategy == "bandwidth":
            sorted_nodes = sorted(active_nodes, 
                                key=lambda x: x.metadata.get("bandwidth", 0), 
                                reverse=True)
            selected = sorted_nodes[:count]
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        return list(selected)

class SecureAggregator:
    def __init__(self, coordinator_private_key: rsa.RSAPrivateKey):
        self.private_key = coordinator_private_key
        self._session_keys: Dict[str, bytes] = {}
        self._logger = Logger(__name__)
        self._metrics = MetricsSystem(["aggregation_errors_total", "model_decryption_time"])

    async def generate_session_key(self, node_public_key: bytes) -> bytes:
        shared_key = self.private_key.exchange(
            serialization.load_pem_public_key(node_public_key)
        )
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"fl_session_key",
        )
        return hkdf.derive(shared_key)

    async def decrypt_model_update(self, node_id: str, encrypted_data: bytes) -> Dict[str, torch.Tensor]:
        with self._metrics.timer("model_decryption_time"):
            try:
                session_key = self._session_keys[node_id]
                aesgcm = AESGCM(session_key)
                nonce = encrypted_data[:12]
                ciphertext = encrypted_data[12:]
                decrypted = aesgcm.decrypt(nonce, ciphertext, None)
                return SecureSerializer.deserialize(decrypted)
            except Exception as e:
                self._metrics.inc("aggregation_errors_total", labels=[type(e).__name__])
                self._logger.error(f"Decryption failed for {node_id}: {str(e)}")
                raise

    async def aggregate_updates(
        self,
        updates: List[Dict[str, torch.Tensor]],
        strategy: str = "fedavg",
        weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        if strategy == "fedavg":
            return self._fedavg(updates, weights)
        elif strategy == "fedprox":
            return self._fedprox(updates, weights)
        else:
            raise ValueError(f"Unsupported aggregation strategy: {strategy}")

    def _fedavg(self, updates: List[Dict[str, torch.Tensor]], weights: Optional[List[float]]) -> Dict[str, torch.Tensor]:
        if not weights:
            weights = [1/len(updates)] * len(updates)
            
        aggregated = {}
        for key in updates[0].keys():
            aggregated[key] = sum(w * update[key] for w, update in zip(weights, updates))
        return aggregated

    def _fedprox(self, updates: List[Dict[str, torch.Tensor]], weights: Optional[List[float]]) -> Dict[str, torch.Tensor]:
        global_model = self._load_global_model()  # Assume global model access
        mu = 0.01  # Proximal term weight
        
        aggregated = self._fedavg(updates, weights)
        for key in aggregated.keys():
            aggregated[key] += mu * (global_model[key] - aggregated[key])
        return aggregated

class ModelManager:
    def __init__(self, storage_path: str = "/models/fl"):
        self.storage_path = storage_path
        self._versions: Dict[str, int] = defaultdict(int)
        self._logger = Logger(__name__)
        os.makedirs(storage_path, exist_ok=True)

    async def initialize_model(self, model: nn.Module, model_class: str) -> int:
        version = self._versions[model_class] + 1
        path = self._model_path(model_class, version)
        torch.save(model.state_dict(), path)
        self._versions[model_class] = version
        return version

    async def load_model(self, model_class: str, version: int) -> nn.Module:
        path = self._model_path(model_class, version)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model {model_class} v{version} not found")
        return torch.load(path)

    async def save_checkpoint(self, model: nn.Module, round_id: str):
        path = os.path.join(self.storage_path, f"checkpoint_{round_id}.pt")
        torch.save(model.state_dict(), path)

    def _model_path(self, model_class: str, version: int) -> str:
        return os.path.join(self.storage_path, f"{model_class}_v{version}.pt")

class FederatedCoordinator:
    def __init__(self):
        self.node_registry = FLNodeRegistry()
        self.model_manager = ModelManager()
        self.grpc_client = GRPCClient()
        self.kafka_producer = KafkaProducer(topic="fl_events")
        self._current_round: Optional[FLRoundConfig] = None
        self._private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.aggregator = SecureAggregator(self._private_key)
        self._logger = Logger(__name__)
        self._metrics = MetricsSystem([
            "fl_rounds_total",
            "fl_round_duration_seconds",
            "model_update_size_bytes"
        ])

    async def start_round(self, config: FLRoundConfig):
        self._current_round = config
        self._logger.info(f"Starting FL round {config.round_id}")
        
        try:
            # Phase 1: Node Selection
            nodes = await self.node_registry.select_nodes(
                strategy=config.node_selection,
                sample_size=config.sample_size
            )
            
            # Phase 2: Model Distribution
            model = await self.model_manager.load_model(config.model_class, config.model_version)
            serialized_model = SecureSerializer.serialize(model.state_dict())
            
            # Phase 3: Secure Session Setup
            session_tasks = [self._establish_secure_session(node) for node in nodes]
            await asyncio.gather(*session_tasks)
            
            # Phase 4: Training Coordination
            update_tasks = [self._collect_update(node) for node in nodes]
            updates = await asyncio.gather(*update_tasks)
            
            # Phase 5: Secure Aggregation
            decrypted_updates = []
            for node, encrypted in updates:
                decrypted = await self.aggregator.decrypt_model_update(node.node_id, encrypted)
                decrypted_updates.append(decrypted)
                
            global_update = await self.aggregator.aggregate_updates(
                decrypted_updates,
                strategy=config.aggregation_strategy
            )
            
            # Phase 6: Model Update
            new_version = await self.model_manager.initialize_model(
                model.load_state_dict(global_update),
                config.model_class
            )
            
            # Phase 7: Broadcast Update
            await self._broadcast_new_model(config.model_class, new_version)
            
            self._metrics.inc("fl_rounds_total")
            
        except Exception as e:
            self._logger.error(f"FL round failed: {str(e)}")
            await self.kafka_producer.send({
                "event": "round_failed",
                "round_id": config.round_id,
                "error": str(e)
            })

    async def _establish_secure_session(self, node: FLNode) -> None:
        session_key = await self.aggregator.generate_session_key(node.public_key)
        self.aggregator._session_keys[node.node_id] = session_key
        
        encrypted_session_key = self._private_key.public_key().encrypt(
            session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        await self.grpc_client.call(
            service="fl_node",
            method="establish_session",
            payload={
                "session_key": encrypted_session_key,
                "round_id": self._current_round.round_id
            },
            node_id=node.node_id
        )

    async def _collect_update(self, node: FLNode) -> Tuple[FLNode, bytes]:
        response = await self.grpc_client.call(
            service="fl_node",
            method="submit_update",
            payload={"round_id": self._current_round.round_id},
            node_id=node.node_id,
            timeout=self._current_round.max_duration.total_seconds()
        )
        return (node, response["encrypted_update"])

    async def _broadcast_new_model(self, model_class: str, version: int):
        model_info = {
            "model_class": model_class,
            "version": version,
            "checksum": self._calculate_model_checksum(model_class, version)
        }
        await self.kafka_producer.send({
            "event": "model_updated",
            "timestamp": datetime.utcnow().isoformat(),
            **model_info
        })

    def _calculate_model_checksum(self, model_class: str, version: int) -> str:
        path = self.model_manager._model_path(model_class, version)
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    async def recover_round(self, round_id: str):
        checkpoint = await self.model_manager.load_checkpoint(round_id)
        # Implement recovery logic based on checkpoint state

# Example Usage
if __name__ == "__main__":
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

    async def main():
        coordinator = FederatedCoordinator()
        
        # Initialize global model
        model = SimpleModel()
        version = await coordinator.model_manager.initialize_model(model, "simple")
        
        # Start FL round
        config = FLRoundConfig(
            round_id="round_001",
            model_class="simple",
            model_version=version,
            node_selection="random",
            sample_size=5
        )
        await coordinator.start_round(config)

    asyncio.run(main())
