import pytest
from httpx import HTTPStatusError

from milvus_mcp.extractor import FeatureExtractor


@pytest.fixture
def mock_async_client(mocker):
    mock_client = mocker.patch("milvus_mcp.extractor.extractor.AsyncClient")
    async_client_instance = mock_client.return_value
    async_client_instance.__aenter__.return_value = async_client_instance

    response = mocker.MagicMock(name="Response")
    response.raise_for_status = mocker.MagicMock(return_value=None)
    response.json = mocker.MagicMock(return_value={"feature_vector": [0.1, 0.2, 0.3]})
    async_client_instance.post = mocker.AsyncMock(return_value=response)

    return async_client_instance


class TestFeatureExtractor:
    @pytest.mark.asyncio
    async def test_extract_feature_vector(self, mock_async_client, mocker):
        extractor = FeatureExtractor()
        vector = await extractor.extract_feature_vector("collection_a", "image.jpg", b"image-bytes", "image/jpeg")

        assert vector == [0.1, 0.2, 0.3]
        mock_async_client.post.assert_awaited_once_with(
            extractor._url,
            data={"ai_model": "collection_a"},
            files={"file": ("image.jpg", b"image-bytes", "image/jpeg")}
        )

        response = mock_async_client.post.return_value
        response.raise_for_status.side_effect = HTTPStatusError(
            "Server Error",
            request=mocker.MagicMock(),
            response=response
        )
        with pytest.raises(HTTPStatusError):
            await extractor.extract_feature_vector("collection_a", "image.jpg", b"image-bytes", "image/jpeg")

        response.raise_for_status.side_effect = None
        response.json.return_value = {}
        with pytest.raises(KeyError):
            await extractor.extract_feature_vector("collection_a", "image.jpg", b"image-bytes", "image/jpeg")
