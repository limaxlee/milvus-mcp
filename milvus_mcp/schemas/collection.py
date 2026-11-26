from typing import Any

from pydantic import BaseModel


class ListCollectionsResult(BaseModel):
    collections: list[str] = []


class GetCollectionInfoResult(BaseModel):
    collection_name: str
    collection_id: int
    auto_id: bool
    description: str
    num_shards: int
    num_partitions: int = 0
    enable_namespace: bool
    enable_dynamic_field: bool
    aliases: list[Any] = []
    classes: dict[str, int] = {}
    fields: list[dict[Any, Any]] = []
    functions: list[Any] = []
    consistency_level: int = 0
    properties: dict[str, Any] = {}


class GetCollectionStatsResult(BaseModel):
    collection_name: str
    row_count: int


class CreateCollectionResult(BaseModel):
    collection_name: str
    collection_created: bool = True


class LoadCollectionResult(BaseModel):
    collection_name: str
    collection_loaded: bool = True
    replica_number: int = 1


class ReleaseCollectionResult(BaseModel):
    collection_name: str
    collection_released: bool = True
