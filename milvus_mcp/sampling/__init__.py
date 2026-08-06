from typing import Any

from common.constants import METRIC_TYPE, DataFields, SamplingMethods
from milvus_mcp.sampling.base import DataSampler
from milvus_mcp.sampling.k_center_greedy import KCenterGreedySampler
from milvus_mcp.schemas import SamplingResult


_DATA_SAMPLERS = {
    SamplingMethods.K_CENTER_GREEDY: KCenterGreedySampler
}


def sample_data(
        method: SamplingMethods,
        collection_name: str,
        data_rows: list[dict[str, Any]],
        sample_classes: dict[str, int],
        keep_classes: list[str] | None = None,
        vector_field: str = DataFields.VECTOR,
        metric_type: str = METRIC_TYPE
) -> SamplingResult:
    return _DATA_SAMPLERS[method]().sample(
        collection_name=collection_name,
        data_rows=data_rows,
        sample_classes=sample_classes,
        keep_classes=keep_classes,
        vector_field=vector_field,
        metric_type=metric_type
    )
