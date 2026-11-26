from typing import Any
from pydantic import BaseModel


class InsertDataResult(BaseModel):
    collection_name: str
    insert_count: int = 0
    primary_keys: list[Any] = []


class DeleteEntitiesResult(BaseModel):
    collection_name: str
    delete_count: int = 0


class QueryCollectionResult(BaseModel):
    collection_name: str
    query_data_count: int = 0
    query_data: list[dict[str, Any]] = []
