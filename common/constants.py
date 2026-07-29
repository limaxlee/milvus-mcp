import pathlib
from enum import StrEnum

ROOT_DIR = pathlib.Path(__file__).parent.parent

LIMIT = 5
METRIC_TYPE = "COSINE"

# FIXME: reflect feature extractor model dimension
DIMENSION = 768

HEALTH_CHECK_TIMEOUT = 5.0
HEALTH_CHECK_PORT = 9091

BATCH_SIZE = 1000

FEATURE_EXTRACTION_TIMEOUT = 60.0
PRESIGNED_URL_EXPIRED_IN = 7200


class DataFields(StrEnum):
    PK = "pk"
    PREDICTION = "prediction"
    DATA_URI = "data_uri"
