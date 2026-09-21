import logging
from typing import Any

from httpx import AsyncClient, TimeoutException

from common.config import SETTINGS
from common.constants import FEATURE_EXTRACTION_TIMEOUT

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Client for the feature extraction API (`POST /features`).

    The API accepts a multipart image upload and responds with a JSON document whose
    `feature_vector.data` holds the embedding as a list of floats.
    """

    def __init__(self):
        self._url = f"http://{SETTINGS.feature_extractor.host}:{SETTINGS.feature_extractor.port}/features"

    async def extract_feature_vector(
            self,
            filename: str,
            object_bytes: bytes,
            content_type: str,
            params: dict[str, Any] | None = None
    ) -> list[float]:
        """Extract the feature vector of an image.

        Args:
            filename: Name of the uploaded image (reported back by the API).
            object_bytes: Raw image bytes.
            content_type: MIME type of the image (jpeg, png, bmp, webp, tiff).
            params: Optional query parameters forwarded to the API (`ExtractFeaturesRequest`).
        """
        try:
            async with AsyncClient(timeout=FEATURE_EXTRACTION_TIMEOUT) as client:
                response = await client.post(
                    self._url,
                    params=params,
                    files={"file": (filename, object_bytes, content_type)}
                )
                response.raise_for_status()

                vector = response.json()["feature_vector"]["data"]
                logger.info(f"Extracted {len(vector)}-length feature vector of data {filename}")
                return vector
        except TimeoutException:
            logger.exception(f"Failed to extract embeddings of data {filename}: timeout {FEATURE_EXTRACTION_TIMEOUT}s")
            raise
        except Exception as e:
            logger.exception(f"Failed to extract embeddings of data {filename}: {str(e)}")
            raise
