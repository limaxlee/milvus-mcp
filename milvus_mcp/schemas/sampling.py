from typing import Any
from pydantic import BaseModel


class SamplingResult(BaseModel):
    collection_name: str
    total_scanned_data: int = 0
    total_sampled_data_size: int = 0
    class_distribution: dict[str, int] = {}
    sampled_data_keys: list[int] = []
    sampled_data: dict[int, Any] = {}
