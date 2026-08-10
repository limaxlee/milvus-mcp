from typing import Any
from pydantic import BaseModel


class ListCollectionsResult(BaseModel):
    collections: list[str] = []


class CollectionInfoResult(BaseModel):
    collection_name: str
    classes: dict[str, int] = {}
    collection_info: dict[str, Any] = {}


class CollectionStatsResult(BaseModel):
    collection_name: str
    collection_stats: dict[str, Any] = {}


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
