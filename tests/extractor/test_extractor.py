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
    response.json = mocker.MagicMock(return_value={
        "image": {"filename": "image.jpg", "content_type": "image/jpeg"},
        "model": {"model_type": "dinov3_vit", "hidden_size": 1024, "layers": [20, 24]},
        "storage_dtype": "float16",
        "feature_vector": {
            "name": "patch_mean_concat",
            "shape": [3],
            "dtype": "float16",
            "encoding": "list",
            "data": [0.1, 0.2, 0.3]
        },
        "feature_maps": []
    })
    async_client_instance.post = mocker.AsyncMock(return_value=response)

    return async_client_instance


class TestFeatureExtractor:
    def test_url(self):
        extractor = FeatureExtractor()
        assert extractor._url.endswith("/features")

    @pytest.mark.asyncio
    async def test_extract_feature_vector(self, mock_async_client, mocker):
        extractor = FeatureExtractor()
        vector = await extractor.extract_feature_vector("image.jpg", b"image-bytes", "image/jpeg")

        assert vector == [0.1, 0.2, 0.3]
        mock_async_client.post.assert_awaited_once_with(
            extractor._url,
            params=None,
            files={"file": ("image.jpg", b"image-bytes", "image/jpeg")}
        )

        response = mock_async_client.post.return_value
        response.raise_for_status.side_effect = HTTPStatusError(
            "Server Error",
            request=mocker.MagicMock(),
            response=response
        )
        with pytest.raises(HTTPStatusError):
            await extractor.extract_feature_vector("image.jpg", b"image-bytes", "image/jpeg")

        response.raise_for_status.side_effect = None
        response.json.return_value = {}
        with pytest.raises(KeyError):
            await extractor.extract_feature_vector("image.jpg", b"image-bytes", "image/jpeg")

    @pytest.mark.asyncio
    async def test_extract_feature_vector_with_params(self, mock_async_client):
        extractor = FeatureExtractor()
        await extractor.extract_feature_vector("image.jpg", b"image-bytes", "image/jpeg", params={"layers": "20,24"})

        mock_async_client.post.assert_awaited_once_with(
            extractor._url,
            params={"layers": "20,24"},
            files={"file": ("image.jpg", b"image-bytes", "image/jpeg")}
        )
