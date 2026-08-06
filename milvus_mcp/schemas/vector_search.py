from typing import Any
from pydantic import BaseModel


class VectorSearchResult(BaseModel):
    collection_name: str
    search_data_count: int = 0
    search_data_keys: list[Any] = []
    search_data_distance: list[float] = []
    search_data: dict[int, Any] = {}
