"""
Enterprise FL Client - Secure Federated Learning Agent with Privacy-Preserving Training
"""

from __future__ import annotations
import asyncio
import hashlib
import logging
import os
import pickle
import time
from contextlib import contextmanager
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader, Dataset
from ..utils.logger import Logger
from ..utils.metrics import MetricsSystem
from ..utils.serialization import SecureSerializer
from ..grpc.client import GRPCClient
from ..kafka.producer import KafkaProducer
from ..models.actor_critic import DynamicBatchNorm

class FLClientConfig(BaseModel):
    coordinator_url: str = Field(..., env="FL_COORDINATOR_URL")
    model_class: str = Field("default", env="FL_MODEL_CLASS")
    data_root: str = Field("/data/fl", env="FL_DATA_ROOT")
    train_batch_size: int = 32
    eval_batch_size: int = 64
    max_local_epochs: int = 5
    learning_rate: float = 0.001
    differential_privacy: Optional[Dict[str, float]] = None
    compression_level: int = 2
    cache_models: bool = True
    gpu_priority: List[int] = [0]

class SecureSessionHandler:
    def __init__(self):
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.session_keys: Dict[str, bytes] = {}
        self._logger = Logger(__name__)
    
    def get_public_key_pem(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    
    async def establish_session(self, encrypted_session_key: bytes, coordinator_public_key_pem: bytes) -> bytes:
        coordinator_key = serialization.load_pem_public_key(coordinator_public_key_pem)
        
        session_key = self.private_key.decrypt(
            encrypted_session_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        self.session_keys['current'] = session_key
        return hashlib.sha256(session_key).digest()

class ModelManager:
    def __init__(self, cache_dir: str = "/models/fl_cache"):
        self.cache_dir = cache_dir
        self._loaded_models: Dict[str, nn.Module] = {}
        self._logger = Logger(__name__)
        os.makedirs(cache_dir, exist_ok=True)
    
    async def load_model(self, model_class: str, version: int, architecture: type) -> nn.Module:
        cache_key = f"{model_class}_v{version}"
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        
        try:
            response = await GRPCClient().call(
                service="fl_coordinator",
                method="get_model",
                payload={"model_class": model_class, "version": version}
            )
            
            model = architecture()
            model.load_state_dict(SecureSerializer.deserialize(response["model_weights"]))
            self._loaded_models[cache_key] = model
            return model
        except Exception as e:
            self._logger.error(f"Model loading failed: {str(e)}")
            raise

class FLDataset(Dataset):
    def __init__(self, data_root: str, transform=None):
        self.data = self._load_and_preprocess(data_root)
        self.transform = transform
    
    def _load_and_preprocess(self, path: str) -> List[Tuple[Any, Any]]:
        # Implementation with privacy-preserving preprocessing
        return []
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int):
        sample = self.data[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

class FLClient:
    def __init__(self, config: FLClientConfig):
        self.config = config
        self.session = SecureSessionHandler()
        self.grpc = GRPCClient(base_url=config.coordinator_url)
        self.kafka = KafkaProducer(topic="fl_client_events")
        self.metrics = MetricsSystem([
            "training_epochs_total",
            "training_loss",
            "model_update_size_bytes"
        ])
        self._logger = Logger(__name__)
        self._setup_hardware()
    
    def _setup_hardware(self):
        torch.set_num_threads(os.cpu_count() // 2)
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.gpu_priority[0]}")
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
    
    async def initialize(self):
        await self._register_with_coordinator()
        await self._subscribe_to_model_updates()
    
    async def _register_with_coordinator(self):
        try:
            response = await self.grpc.call(
                service="fl_coordinator",
                method="register_node",
                payload={
                    "public_key": self.session.get_public_key_pem().decode(),
                    "capabilities": {
                        "max_batch_size": self.config.train_batch_size,
                        "gpu_available": torch.cuda.is_available()
                    }
                }
            )
            self.client_id = response["client_id"]
        except Exception as e:
            self._logger.error(f"Registration failed: {str(e)}")
            raise
    
    async def _subscribe_to_model_updates(self):
        async def callback(msg):
            await self._handle_new_model(
                msg["model_class"],
                msg["version"],
                msg["checksum"]
            )
        
        await self.kafka.subscribe("model_updates", callback)
    
    async def _handle_new_model(self, model_class: str, version: int, checksum: str):
        if model_class != self.config.model_class:
            return
        
        current_checksum = await self._get_local_model_checksum(model_class, version)
        if current_checksum != checksum:
            await self._download_model(model_class, version, checksum)
    
    async def participate_in_round(self, round_id: str):
        try:
            async with self._training_context(round_id):
                # Phase 1: Model Retrieval
                global_model = await self._get_current_model()
                
                # Phase 2: Local Training
                trained_model = await self._train_model(global_model)
                
                # Phase 3: Update Preparation
                update = self._compute_model_update(global_model, trained_model)
                encrypted_update = await self._encrypt_update(update)
                
                # Phase 4: Secure Submission
                await self._submit_update(round_id, encrypted_update)
                
        except Exception as e:
            self._logger.error(f"Round {round_id} failed: {str(e)}")
            await self.kafka.send({
                "event": "training_failed",
                "round_id": round_id,
                "error": str(e)
            })
    
    @contextmanager
    def _training_context(self, round_id: str):
        start_time = time.monotonic()
        try:
            yield
            await self.kafka.send({
                "event": "training_completed",
                "round_id": round_id,
                "duration": time.monotonic() - start_time
            })
        except asyncio.CancelledError:
            self._logger.warning(f"Training round {round_id} cancelled")
            raise
    
    async def _get_current_model(self) -> nn.Module:
        return await ModelManager().load_model(
            self.config.model_class,
            version="latest",
            architecture=DynamicBatchNorm
        )
    
    async def _train_model(self, model: nn.Module) -> nn.Module:
        model.train()
        model.to(self.device)
        
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=self.config.learning_rate
        )
        
        dataset = FLDataset(self.config.data_root)
        loader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=os.cpu_count()//2,
            pin_memory=torch.cuda.is_available()
        )
        
        for epoch in range(self.config.max_local_epochs):
            for batch in loader:
                inputs, targets = self._preprocess_batch(batch)
                outputs = model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                
                # Differential Privacy
                if self.config.differential_privacy:
                    noise = torch.randn_like(loss.grad) * self.config.differential_privacy["noise_scale"]
                    loss += noise
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping for Privacy
                if self.config.differential_privacy:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config.differential_privacy["max_grad_norm"]
                    )
                
                optimizer.step()
                
                self.metrics.record("training_loss", loss.item())
            
            self.metrics.inc("training_epochs_total")
        
        return model.cpu()
    
    def _compute_model_update(self, original: nn.Module, trained: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            name: trained.state_dict()[name] - original.state_dict()[name]
            for name in original.state_dict().keys()
        }
    
    async def _encrypt_update(self, update: Dict[str, torch.Tensor]) -> bytes:
        session_key = self.session.session_keys['current']
        aesgcm = AESGCM(session_key)
        nonce = os.urandom(12)
        
        serialized = SecureSerializer.serialize(update)
        encrypted = aesgcm.encrypt(nonce, serialized, None)
        return nonce + encrypted
    
    async def _submit_update(self, round_id: str, encrypted_update: bytes):
        await self.grpc.call(
            service="fl_coordinator",
            method="submit_update",
            payload={
                "round_id": round_id,
                "client_id": self.client_id,
                "encrypted_update": encrypted_update
            }
        )
        self.metrics.record("model_update_size_bytes", len(encrypted_update))
    
    async def _download_model(self, model_class: str, version: int, expected_checksum: str):
        # Implementation with checksum verification
        pass
    
    async def _get_local_model_checksum(self, model_class: str, version: int) -> str:
        # Implementation with local cache validation
        return ""

# Example Usage
if __name__ == "__main__":
    async def main():
        config = FLClientConfig(
            coordinator_url="grpc://fl-coordinator.aelion.ai:50051",
            model_class="supply_chain",
            data_root="/data/supply_chain",
            differential_privacy={"noise_scale": 0.01, "max_grad_norm": 1.0}
        )
        
        client = FLClient(config)
        await client.initialize()
        
        # Simulate participation
        await client.participate_in_round("round_001")

    asyncio.run(main())
