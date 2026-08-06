import numpy as np

from common.constants import METRIC_TYPE, MetricTypes
from milvus_mcp.sampling.base import DataSampler


class KCenterGreedySampler(DataSampler):
    def _sample_indices(
            self,
            vectors: np.ndarray,
            classes: list[str],
            class_caps: dict[str, int],
            metric_type: str = METRIC_TYPE
    ) -> list[int]:
        np_classes = np.array(classes)
        sampled_data = []

        for class_name, class_cap in class_caps.items():
            class_data_idx = np.where(np_classes == class_name)[0]
            local = self._k_center_greedy_sample(vectors[class_data_idx], class_cap, metric_type)
            selected_idx = [int(class_data_idx[i]) for i in local]

            sampled_data.extend(selected_idx)

        return sampled_data

    @staticmethod
    def _k_center_greedy_sample(vectors: np.ndarray, sample_size: int, metric_type: str = METRIC_TYPE) -> list[int]:
        feature_vectors = vectors.astype(np.float32, copy=True)

        if metric_type == MetricTypes.COSINE:
            normalized_vectors = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
            normalized_vectors[normalized_vectors == 0.0] = 1.0
            feature_vectors /= normalized_vectors

        def distance_to_center(center: np.ndarray) -> np.ndarray:
            if metric_type in (MetricTypes.COSINE, MetricTypes.IP):
                return -(feature_vectors @ center)
            diff = feature_vectors - center
            return np.einsum("ij,ij->i", diff, diff)

        start_data = int(np.argmin(distance_to_center(feature_vectors.mean(axis=0))))
        selected_data = [start_data]
        min_distance = distance_to_center(feature_vectors[start_data])
        min_distance[start_data] = -np.inf

        for _ in range(sample_size - 1):
            idx = int(np.argmax(min_distance))
            if not np.isfinite(min_distance[idx]):
                break

            selected_data.append(idx)
            min_distance = np.minimum(min_distance, distance_to_center(feature_vectors[idx]))
            min_distance[idx] = -np.inf

        return selected_data
