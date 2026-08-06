import logging
from httpx import AsyncClient, TimeoutException

from common.config import SETTINGS
from common.constants import FEATURE_EXTRACTION_TIMEOUT

logger = logging.getLogger(__name__)


# FIXME: Replace with the real API Endpoint to extract the feature vector
class FeatureExtractor:
    def __init__(self):
        self._url = f"http://{SETTINGS.feature_extractor.host}:{SETTINGS.feature_extractor.port}/feature-vector"

    async def extract_feature_vector(
            self,
            ai_model: str,
            filename: str,
            object_bytes: bytes,
            content_type: str
    ) -> list[float]:
        try:
            async with AsyncClient(timeout=FEATURE_EXTRACTION_TIMEOUT) as client:
                response = await client.post(
                    self._url,
                    data={"ai_model": ai_model},
                    files={"file": (filename, object_bytes, content_type)}
                )
                response.raise_for_status()

                vector = response.json()["feature_vector"]
                logger.info(f"Extracted {len(vector)}-length feature vector of data {filename}")
                return vector
        except TimeoutException:
            logger.exception(f"Failed to extract embeddings of data {filename}: timeout {FEATURE_EXTRACTION_TIMEOUT}s")
            raise
        except Exception as e:
            logger.exception(f"Failed to extract embeddings of data {filename}: {str(e)}")
            raise
