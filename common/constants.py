import os
import pathlib
from enum import StrEnum

ROOT_DIR = pathlib.Path(__file__).parent.parent
TEMP_DIR = ROOT_DIR.joinpath("temp")
os.makedirs(TEMP_DIR, exist_ok=True)

LIMIT = 5
METRIC_TYPE = "COSINE"

DIMENSION = 2048

HEALTH_CHECK_TIMEOUT = 5.0
HEALTH_CHECK_PORT = 9091

BATCH_SIZE = 1000

DETECTION_COLLECTION_SUFFIX = "_object"
VFM_COLLECTION_PREFIX = "VFM_"

FEATURE_EXTRACTION_TIMEOUT = 60.0
PRESIGNED_URL_EXPIRED_IN = 7200         # 2 hours
MULTIPART_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

OS_PREFIX = "MilvusMCP"
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
