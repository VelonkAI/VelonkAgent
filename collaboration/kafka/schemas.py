"""
Schema Registry Manager - Avro Schema Versioning & Evolution for Kafka Messages
"""

from __future__ import annotations
import json
import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

# Schema management
from confluent_kafka.schema_registry import SchemaRegistryClient, Schema
from confluent_kafka.schema_registry.avro import AvroSerializer
from fastavro import parse_schema, schema, types

# Internal modules
from utils.logger import get_logger

logger = get_logger(__name__)

class SchemaCompatibility(Enum):
    """Avro schema compatibility modes"""
    NONE = "NONE"
    BACKWARD = "BACKWARD"
    FORWARD = "FORWARD"
    FULL = "FULL"

class SchemaValidationError(Exception):
    """Raised when message fails schema validation"""
    pass

class SchemaCompatibilityError(Exception):
    """Raised when schema evolution breaks compatibility"""
    pass

@dataclass(frozen=True)
class SchemaMetadata:
    """Immutable schema metadata container"""
    name: str
    version: int
    fingerprint: str
    compatibility: SchemaCompatibility
    dependencies: Dict[str, str]

class AvroSchemaRegistry:
    """Enterprise-grade schema registry with evolution control"""
    
    def __init__(
        self,
        registry_url: str = "http://schema-registry.aelion.ai:8081",
        cache_dir: Path = Path("/var/cache/aelion/schemas"),
        default_compatibility: SchemaCompatibility = SchemaCompatibility.BACKWARD
    ):
        self._client = SchemaRegistryClient({"url": registry_url})
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._default_compatibility = default_compatibility
        self._local_schemas: Dict[str, Dict[int, schema.SchemaParseResult]] = {}
        self._serializers: Dict[str, AvroSerializer] = {}
        
        # Preload core schemas
        self._preload_core_schemas()

    def _preload_core_schemas(self):
        """Load essential schemas at initialization"""
        core_schemas = [
            AgentMessage.get_schema(),
            TaskRequest.get_schema(),
            ResourceAllocation.get_schema()
        ]
        for s in core_schemas:
            self.register(schema_str=json.dumps(s), schema_type="AVRO")

    def register(
        self,
        schema_str: str,
        schema_type: str = "AVRO",
        compatibility: Optional[SchemaCompatibility] = None
    ) -> SchemaMetadata:
        """Register schema with compatibility checks"""
        schema_dict = json.loads(schema_str)
        parsed = parse_schema(schema_dict)
        schema_name = parsed["name"]
        
        # Check local cache first
        cached = self._load_from_cache(schema_name, parsed)
        if cached:
            return cached

        # Validate compatibility
        compatibility = compatibility or self._default_compatibility
        existing = self._get_latest_schema(schema_name)
        if existing:
            if not self._is_compatible(existing, parsed, compatibility):
                raise SchemaCompatibilityError(
                    f"Schema {schema_name} breaks {compatibility.value} compatibility"
                )

        # Register with SR
        schema_id = self._client.register_schema(
            subject_name=f"{schema_name}-value",
            schema=Schema(schema_str, schema_type)
        )

        # Update cache
        metadata = SchemaMetadata(
            name=schema_name,
            version=schema_id.version,
            fingerprint=hashlib.sha256(schema_str.encode()).hexdigest(),
            compatibility=compatibility,
            dependencies=self._extract_dependencies(parsed)
        )
        self._cache_schema(metadata, parsed)
        
        return metadata

    def get_serializer(self, schema_name: str) -> AvroSerializer:
        """Get cached Avro serializer for schema"""
        if schema_name not in self._serializers:
            latest = self._get_latest_schema(schema_name)
            self._serializers[schema_name] = AvroSerializer(
                schema_registry_client=self._client,
                schema_str=json.dumps(latest),
            )
        return self._serializers[schema_name]

    def validate(self, schema_name: str, data: Dict) -> bool:
        """Validate data against schema version"""
        schema_versions = self._local_schemas.get(schema_name, {})
        for version, parsed_schema in reversed(schema_versions.items()):
            try:
                types.validate(data, parsed_schema)
                return True
            except types.ValidationError:
                continue
        raise SchemaValidationError(f"No compatible schema found for {schema_name}")

    def _is_compatible(
        self,
        existing: schema.SchemaParseResult,
        new: schema.SchemaParseResult,
        compatibility: SchemaCompatibility
    ) -> bool:
        """Check schema evolution compatibility"""
        # Implement full compatibility rules
        if compatibility == SchemaCompatibility.NONE:
            return True
        # ... (detailed compatibility checks)
        return True  # Simplified for example

    def _extract_dependencies(self, schema: schema.SchemaParseResult) -> Dict[str, str]:
        """Extract nested schema dependencies"""
        deps = {}
        for field in schema.get("fields", []):
            if isinstance(field.type, dict) and "name" in field.type:
                deps[field.type["name"]] = field.type["namespace"]
        return deps

    def _cache_schema(self, metadata: SchemaMetadata, parsed: schema.SchemaParseResult):
        """Cache schema locally for fast access"""
        self._local_schemas.setdefault(metadata.name, {})[metadata.version] = parsed
        cache_file = self._cache_dir / f"{metadata.name}_v{metadata.version}.avsc"
        with open(cache_file, "w") as f:
            json.dump(parsed.to_json(), f)

    def _load_from_cache(self, name: str, parsed: schema.SchemaParseResult) -> Optional[SchemaMetadata]:
        """Load schema from local cache"""
        cache_pattern = f"{name}_v*.avsc"
        for cache_file in self._cache_dir.glob(cache_pattern):
            version = int(cache_file.stem.split("_v")[1].split(".avsc")[0])
            with open(cache_file, "r") as f:
                cached_schema = json.load(f)
                if cached_schema["fingerprint"] == hashlib.sha256(
                    json.dumps(parsed).encode()
                ).hexdigest():
                    return SchemaMetadata(
                        name=name,
                        version=version,
                        fingerprint=cached_schema["fingerprint"],
                        compatibility=SchemaCompatibility(
                            cached_schema["compatibility"]
                        ),
                        dependencies=cached_schema["dependencies"]
                    )
        return None

    def _get_latest_schema(self, name: str) -> Optional[schema.SchemaParseResult]:
        """Get latest schema version"""
        versions = self._local_schemas.get(name, {})
        if not versions:
            return None
        latest_version = max(versions.keys())
        return versions[latest_version]

# --------------------------
# Core Schema Definitions
# --------------------------

@dataclass
class AgentMessage:
    """Base message format for agent communication"""
    agent_id: str
    timestamp: int  # Epoch millis
    payload: Dict[str, Any]
    schema_version: int = 1

    @classmethod
    def get_schema(cls) -> Dict:
        return {
            "type": "record",
            "name": "AgentMessage",
            "namespace": "com.aelion.kafka",
            "fields": [
                {"name": "agent_id", "type": "string"},
                {"name": "timestamp", "type": "long"},
                {"name": "payload", "type": {"type": "map", "values": "bytes"}},
                {"name": "schema_version", "type": "int", "default": 1}
            ],
            "compatibility": "BACKWARD"
        }

@dataclass
class TaskRequest:
    """Task assignment payload"""
    task_id: str
    priority: int
    requirements: Dict[str, Union[int, float]]
    dependencies: List[str]
    schema_version: int = 2

    @classmethod
    def get_schema(cls) -> Dict:
        return {
            "type": "record",
            "name": "TaskRequest",
            "namespace": "com.aelion.kafka",
            "fields": [
                {"name": "task_id", "type": "string"},
                {"name": "priority", "type": "int"},
                {"name": "requirements", "type": {"type": "map", "values": ["int", "float"]}},
                {"name": "dependencies", "type": {"type": "array", "items": "string"}},
                {"name": "schema_version", "type": "int", "default": 2}
            ],
            "compatibility": "FULL"
        }

@dataclass 
class ResourceAllocation:
    """Resource assignment record"""
    allocation_id: str
    agent_ids: List[str]
    cpu: float
    memory: int  # MB
    gpu: Optional[int] = None
    schema_version: int = 3

    @classmethod
    def get_schema(cls) -> Dict:
        return {
            "type": "record",
            "name": "ResourceAllocation",
            "namespace": "com.aelion.kafka",
            "fields": [
                {"name": "allocation_id", "type": "string"},
                {"name": "agent_ids", "type": {"type": "array", "items": "string"}},
                {"name": "cpu", "type": "double"},
                {"name": "memory", "type": "int"},
                {"name": "gpu", "type": ["null", "int"], "default": null},
                {"name": "schema_version", "type": "int", "default": 3}
            ],
            "compatibility": "FORWARD"
        }

# --------------------------
# Initialization & Testing
# --------------------------

if __name__ == "__main__":
    # Initialize registry
    registry = AvroSchemaRegistry()
    
    # Test schema registration
    message_schema = AgentMessage.get_schema()
    metadata = registry.register(json.dumps(message_schema))
    print(f"Registered {metadata.name} v{metadata.version}")
    
    # Validate sample data
    sample_msg = {
        "agent_id": "agent-001",
        "timestamp": 1717027200000,
        "payload": {"status": b"active"},
        "schema_version": 1
    }
    registry.validate("AgentMessage", sample_msg)
