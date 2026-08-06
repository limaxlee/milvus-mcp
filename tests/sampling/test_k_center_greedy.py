import pytest
import numpy as np

from common.constants import MetricTypes
from milvus_mcp.sampling.k_center_greedy import KCenterGreedySampler

EUCLIDEAN_METRIC = "L2"


@pytest.fixture
def sampler():
    return KCenterGreedySampler()


class TestKCenterGreedySampler:
    def test_k_center_greedy_sample(self, sampler):
        line_vectors = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
            dtype=np.float32
        )

        assert sampler._k_center_greedy_sample(line_vectors, 3, EUCLIDEAN_METRIC) == [2, 4, 0]
        assert sampler._k_center_greedy_sample(line_vectors, 1, EUCLIDEAN_METRIC) == [2]
        assert sorted(sampler._k_center_greedy_sample(line_vectors, 10, EUCLIDEAN_METRIC)) == [0, 1, 2, 3, 4]

        original = line_vectors.copy()
        sampler._k_center_greedy_sample(line_vectors, 3, EUCLIDEAN_METRIC)
        assert np.array_equal(line_vectors, original)

        unit_vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
        assert sampler._k_center_greedy_sample(unit_vectors, 2, MetricTypes.COSINE) == [1, 0]
        assert sampler._k_center_greedy_sample(unit_vectors, 3, MetricTypes.COSINE) == [1, 0, 2]

        zero_vectors = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        assert sorted(sampler._k_center_greedy_sample(zero_vectors, 3, MetricTypes.COSINE)) == [0, 1, 2]

    def test_sample_indices(self, sampler):
        vectors = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [5.0, 5.0], [5.1, 5.1]],
            dtype=np.float32
        )
        classes = ["Good", "Good", "Good", "NG", "NG"]

        result = sampler._sample_indices(vectors, classes, {"Good": 2, "NG": 1}, EUCLIDEAN_METRIC)
        assert result == [1, 2, 4]
        assert [classes[i] for i in result].count("Good") == 2
        assert [classes[i] for i in result].count("NG") == 1

        result = sampler._sample_indices(vectors, classes, {"Good": 5}, EUCLIDEAN_METRIC)
        assert sorted(result) == [0, 1, 2]

        assert sampler._sample_indices(vectors, classes, {}, EUCLIDEAN_METRIC) == []
