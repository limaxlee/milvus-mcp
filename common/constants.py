import os
import pathlib
from enum import StrEnum

ROOT_DIR = pathlib.Path(__file__).parent.parent
TEMP_DIR = ROOT_DIR.joinpath("temp")
os.makedirs(TEMP_DIR, exist_ok=True)

LIMIT = 5
METRIC_TYPE = "COSINE"

# Feature extractor (dinov3_vit, hidden_size 1024 x layers [20, 24] -> patch_mean_concat)
DIMENSION = 2048

HEALTH_CHECK_TIMEOUT = 5.0
HEALTH_CHECK_PORT = 9091

BATCH_SIZE = 1000

DETECTION_COLLECTION_SUFFIX = "_object"
# A collection may be stored under its own name or with this prefix (ABC -> VFM_ABC)
VFM_COLLECTION_PREFIX = "VFM_"

FEATURE_EXTRACTION_TIMEOUT = 60.0
PRESIGNED_URL_EXPIRED_IN = 7200

ZIP_FETCH_CONCURRENCY = 16


class DataFields(StrEnum):
    PK = "pk"
    PREDICTION = "prediction"
    VECTOR = "feature_vector"
    DATA_URI = "data_uri"
    FILENAME = "filename"


class MetricTypes(StrEnum):
    COSINE = "COSINE"
    IP = "IP"
    L2 = "L2"


class SamplingMethods(StrEnum):
    K_CENTER_GREEDY = "k_center_greedy"

