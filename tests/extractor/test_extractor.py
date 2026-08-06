import pytest

from common.config import SETTINGS
from common.constants import FEATURE_EXTRACTION_TIMEOUT
from milvus_mcp.extractor import FeatureExtractor
from milvus_mcp.extractor import extractor as extractor_module


@pytest.fixture
def mock_async_client(mocker):
    mock_client = mocker.patch("milvus_mcp.extractor.extractor.AsyncClient")
    async_client_instance = mock_client.return_value
    async_client_instance.__aenter__.return_value = async_client_instance

    response = mocker.MagicMock(name="Response")
    response.raise_for_status = mocker.MagicMock(return_value=None)
    response.json = mocker.MagicMock(return_value={"feature_vector": [0.1, 0.2, 0.3]})
    async_client_instance.post = mocker.AsyncMock(return_value=response)

    return mock_client


class TestFeatureExtractor:
    def test_url_init(self):
        extractor = FeatureExtractor()
        host, port = SETTINGS.feature_extractor.host, SETTINGS.feature_extractor.port

        assert extractor._url == f"http://{host}:{port}/feature-vector"

    @pytest.mark.asyncio
    async def test_extract_feature_vector(self, mock_async_client):
        extractor = FeatureExtractor()
        client = mock_async_client.return_value

        vector = await extractor.extract_feature_vector("collection_a", "image.jpg", b"image-bytes", "image/jpeg")

        assert vector == [0.1, 0.2, 0.3]
        mock_async_client.assert_called_once_with(timeout=FEATURE_EXTRACTION_TIMEOUT)
        client.post.assert_awaited_once_with(
            extractor._url,
            data={"ai_model": "collection_a"},
            files={"file": ("image.jpg", b"image-bytes", "image/jpeg")}
        )
        client.__aexit__.assert_awaited_once()

        await extractor.extract_feature_vector("collection_b", "photo.png", b"photo-bytes", "image/png")

        assert client.post.await_count == 2
        assert client.post.await_args.kwargs["data"] == {"ai_model": "collection_b"}
        assert client.post.await_args.kwargs["files"] == {"file": ("photo.png", b"photo-bytes", "image/png")}

    @pytest.mark.asyncio
    async def test_extract_feature_vector_failure(self, mock_async_client):
        extractor = FeatureExtractor()
        client = mock_async_client.return_value
        response = client.post.return_value

        # raised through the module, so this test imports no third party package itself
        client.post.side_effect = extractor_module.TimeoutException("Extractor timeout")
        with pytest.raises(extractor_module.TimeoutException):
            await extractor.extract_feature_vector("collection_a", "image.jpg", b"image-bytes", "image/jpeg")

        client.post.side_effect = None
        response.raise_for_status.side_effect = RuntimeError("Extractor error")
        with pytest.raises(RuntimeError, match="Extractor error"):
            await extractor.extract_feature_vector("collection_a", "image.jpg", b"image-bytes", "image/jpeg")

        response.raise_for_status.side_effect = None
        response.json.return_value = {}
        with pytest.raises(KeyError):
            await extractor.extract_feature_vector("collection_a", "image.jpg", b"image-bytes", "image/jpeg")
