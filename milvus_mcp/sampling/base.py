import numpy as np
from typing import Any
from collections import Counter
from abc import ABC, abstractmethod

from common.constants import METRIC_TYPE, DataFields
from milvus_mcp.schemas import SamplingResult


class DataSampler(ABC):
    def sample(
            self,
            collection_name: str,
            data_rows: list[dict[str, Any]],
            sample_classes: dict[str, int],
            keep_classes: list[str] | None = None,
            vector_field: str = DataFields.VECTOR,
            metric_type: str = METRIC_TYPE
    ) -> SamplingResult:
        if any(sample_cap <= 0 for sample_cap in sample_classes.values()):
            raise ValueError(f"Sample cap for classes {sample_classes} must be positive")
        if keep_classes and set(keep_classes) & sample_classes.keys():
            raise ValueError(f"Overlapping classes in keep {keep_classes} and sample {sample_classes} classes")

        pk = [item[DataFields.PK] for item in data_rows]
        classes = [item.get(DataFields.PREDICTION) for item in data_rows]
        class_counts = Counter(classes)
        vectors = np.asarray([item[vector_field] for item in data_rows], dtype=np.float32)

        if sample_classes.keys() - class_counts.keys():
            raise ValueError(f"Sample classes {sample_classes} are not present in {collection_name} collection")
        if keep_classes and set(keep_classes) - class_counts.keys():
            raise ValueError(f"Keep classes {keep_classes} are not present in {collection_name} collection")

        class_caps = {class_name: class_counts[class_name] for class_name in keep_classes} if keep_classes else {}
        for class_name, sample_cap in sample_classes.items():
            class_caps[class_name] = min(sample_cap, class_counts[class_name])

        sampled_data = self._sample_indices(vectors, classes, class_caps, metric_type)
        sampled_data_pk = [pk[item] for item in sampled_data]

        return SamplingResult(
            collection_name=collection_name,
            total_scanned_data=len(data_rows),
            total_sampled_data_size=sum(class_caps.values()),
            class_distribution=class_caps,
            sampled_data_keys=sampled_data_pk,
            sampled_data={
                data_row[DataFields.PK]: {key: value for key, value in data_row.items() if key != vector_field}
                for data_row in data_rows if data_row[DataFields.PK] in sampled_data_pk
            }
        )

    @abstractmethod
    def _sample_indices(
            self,
            vectors: np.ndarray,
            classes: list[str],
            class_caps: dict[str, int],
            metric_type: str = METRIC_TYPE
    ) -> list[int]:
        ...
