"""
Unified Serialization Engine - High-performance Multi-format Serialization with Schema Evolution
"""

from __future__ import annotations
import io
import json
import lzma
import zlib
import pickle
import struct
import warnings
from abc import ABC, abstractmethod
from collections import deque
from enum import IntEnum
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints
)

# Third-party
import msgpack
from pydantic import BaseModel, ValidationError
from google.protobuf.message import Message as ProtobufMessage

T = TypeVar('T')
SchemaType = TypeVar('SchemaType', bound=BaseModel)
ProtobufType = TypeVar('ProtobufType', bound=ProtobufMessage)

class SerializationFormat(IntEnum):
    """Supported serialization protocols"""
    JSON      = 0x1A
    MSGPACK   = 0x2B
    PROTOBUF  = 0x3C
    PICKLE    = 0x4D
    PYDANTIC  = 0x5E

class CompressionAlgorithm(IntEnum):
    """Supported compression methods"""
    NONE    = 0x00
    ZLIB    = 0x01
    LZMA    = 0x02
    BROTLI  = 0x03  # Reserved

class SerializationError(Exception):
    """Base exception for serialization failures"""
    def __init__(self, original_error: Exception, data: Any, format: SerializationFormat):
        super().__init__(f"Serialization failed for {format.name}: {str(original_error)}")
        self.original_error = original_error
        self.data = data
        self.format = format

class SchemaVersionMismatchError(SerializationError):
    """Schema evolution conflict"""
    def __init__(self, current_version: int, received_version: int):
        super().__init__(Exception(f"Schema version mismatch (current: {current_version}, received: {received_version})"), None, None)
        self.current_version = current_version
        self.received_version = received_version

class Serializer(ABC):
    """Abstract base class for serialization strategies"""
    
    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Convert object to bytes"""
        pass
    
    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Reconstruct object from bytes"""
        pass

class MultiFormatSerializer:
    """Main serialization facade with format auto-detection"""
    
    _VERSION_HEADER = b'AELION_SER_v1'
    _HEADER_FORMAT = '!4sBBIQ'  # Magic(4), Format(1), Compression(1), SchemaVersion(4), Checksum(8)
    
    def __init__(self):
        self._serializers: Dict[SerializationFormat, Serializer] = {
            SerializationFormat.JSON: JSONSerializer(),
            SerializationFormat.MSGPACK: MsgPackSerializer(),
            SerializationFormat.PROTOBUF: ProtobufSerializer(),
            SerializationFormat.PICKLE: PickleSerializer(),
            SerializationFormat.PYDANTIC: PydanticSerializer()
        }
        self._compression: Dict[CompressionAlgorithm, Callable[[bytes], bytes]] = {
            CompressionAlgorithm.ZLIB: lambda d: zlib.compress(d, level=3),
            CompressionAlgorithm.LZMA: lambda d: lzma.compress(d, preset=lzma.PRESET_DEFAULT)
        }
        self._decompression: Dict[CompressionAlgorithm, Callable[[bytes], bytes]] = {
            CompressionAlgorithm.ZLIB: zlib.decompress,
            CompressionAlgorithm.LZMA: lzma.decompress
        }
    
    def serialize(
        self,
        obj: Any,
        format: SerializationFormat = SerializationFormat.MSGPACK,
        compression: CompressionAlgorithm = CompressionAlgorithm.NONE,
        schema_version: int = 1,
        validate: bool = True
    ) -> bytes:
        header = struct.pack(
            self._HEADER_FORMAT,
            self._VERSION_HEADER,
            format.value,
            compression.value,
            schema_version,
            0  # Placeholder for checksum
        )
        
        try:
            # Serialize payload
            serializer = self._serializers[format]
            payload = serializer.serialize(obj)
            
            # Apply compression
            if compression != CompressionAlgorithm.NONE:
                payload = self._compression[compression](payload)
            
            # Add checksum (xxHash)
            checksum = self._compute_checksum(payload)
            header = struct.pack(
                self._HEADER_FORMAT,
                self._VERSION_HEADER,
                format.value,
                compression.value,
                schema_version,
                checksum
            )
            
            return header + payload
        except Exception as e:
            raise SerializationError(e, obj, format)
    
    def deserialize(
        self,
        data: bytes,
        expected_type: Optional[type] = None,
        schema_version: Optional[int] = None
    ) -> Any:
        try:
            header = data[:struct.calcsize(self._HEADER_FORMAT)]
            magic, format_val, compression_val, schema_ver, checksum = struct.unpack(
                self._HEADER_FORMAT, header
            )
            
            if magic != self._VERSION_HEADER:
                raise ValueError("Invalid serialization header")
            
            # Verify checksum
            payload = data[len(header):]
            if self._compute_checksum(payload) != checksum:
                raise ValueError("Data corruption detected")
            
            # Handle schema evolution
            if schema_version is not None and schema_ver != schema_version:
                raise SchemaVersionMismatchError(schema_version, schema_ver)
            
            # Decompress payload
            compression = CompressionAlgorithm(compression_val)
            if compression != CompressionAlgorithm.NONE:
                payload = self._decompression[compression](payload)
            
            # Deserialize content
            format = SerializationFormat(format_val)
            serializer = self._serializers[format]
            result = serializer.deserialize(payload)
            
            # Type validation
            if expected_type and not isinstance(result, expected_type):
                raise TypeError(f"Expected {expected_type}, got {type(result)}")
            
            return result
        except Exception as e:
            raise SerializationError(e, data, None)
    
    @staticmethod
    def _compute_checksum(data: bytes) -> int:
        """Non-cryptographic xxHash (placeholder implementation)"""
        return hash(data) & 0xFFFFFFFFFFFFFFFF

class JSONSerializer(Serializer):
    def serialize(self, obj: Any) -> bytes:
        return json.dumps(obj, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    
    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode('utf-8'))

class MsgPackSerializer(Serializer):
    def serialize(self, obj: Any) -> bytes:
        return msgpack.packb(obj, use_bin_type=True)
    
    def deserialize(self, data: bytes) -> Any:
        return msgpack.unpackb(data, raw=False)

class ProtobufSerializer(Serializer):
    def serialize(self, obj: ProtobufMessage) -> bytes:
        if not isinstance(obj, ProtobufMessage):
            raise TypeError("Protobuf serializer requires protobuf.Message instances")
        return obj.SerializeToString()
    
    def deserialize(self, data: bytes, proto_type: type[ProtobufType]) -> ProtobufType:
        proto = proto_type()
        proto.ParseFromString(data)
        return proto

class PickleSerializer(Serializer):
    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    
    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

class PydanticSerializer(Serializer):
    def serialize(self, model: BaseModel) -> bytes:
        if not isinstance(model, BaseModel):
            raise TypeError("Pydantic serializer requires BaseModel instances")
        return model.json(by_alias=True).encode('utf-8')
    
    def deserialize(self, data: bytes, model_type: type[SchemaType]) -> SchemaType:
        try:
            return model_type.parse_raw(data)
        except ValidationError as e:
            raise SerializationError(e, data, SerializationFormat.PYDANTIC)

# Example usage
if __name__ == "__main__":
    class SampleModel(BaseModel):
        id: int
        name: str
    
    serializer = MultiFormatSerializer()
    data = SampleModel(id=1, name="Aelion")
    
    # Serialization
    packed = serializer.serialize(
        data,
        format=SerializationFormat.PYDANTIC,
        compression=CompressionAlgorithm.ZLIB
    )
    
    # Deserialization
    unpacked = serializer.deserialize(packed, expected_type=SampleModel)
    print(unpacked)
