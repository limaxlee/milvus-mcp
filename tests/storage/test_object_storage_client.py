import pytest
from contextlib import AsyncExitStack
from botocore.exceptions import ClientError

from common.constants import PRESIGNED_URL_EXPIRED_IN
from milvus_mcp.storage import ObjectStorageClient


@pytest.fixture
def storage_client(mocker):
    # ObjectStorageClient is a singleton, so the shared instance is reset around every test
    client = ObjectStorageClient()
    original_state = (client._session, client._exit_stack, client._client)

    client._session = mocker.MagicMock(name="AioSession")
    client._exit_stack = AsyncExitStack()
    client._client = None

    yield client

    client._session, client._exit_stack, client._client = original_state


@pytest.fixture
def mock_s3_client(mocker, storage_client):
    s3_client = mocker.MagicMock(name="S3Client")
    create_client = mocker.MagicMock(name="CreateClient")
    create_client.__aenter__ = mocker.AsyncMock(return_value=s3_client)
    create_client.__aexit__ = mocker.AsyncMock(return_value=False)
    storage_client._session.create_client.return_value = create_client

    return s3_client


class TestObjectStorage:
    def test_client_is_singleton(self):
        assert ObjectStorageClient() is ObjectStorageClient()

    @pytest.mark.asyncio
    async def test_connect(self, storage_client, mock_s3_client):
        result = await storage_client.connect()

        assert result is storage_client
        assert storage_client._client is mock_s3_client
        assert storage_client._session.create_client.call_count == 1

        # already connected, so the session is not asked for a second client
        await storage_client.connect()
        assert storage_client._session.create_client.call_count == 1

    @pytest.mark.asyncio
    async def test_connect_failure(self, storage_client):
        storage_client._session.create_client.side_effect = ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "CreateClient"
        )
        with pytest.raises(ClientError):
            await storage_client.connect()
        assert storage_client._client is None

        storage_client._session.create_client.side_effect = RuntimeError("Storage error")
        with pytest.raises(RuntimeError, match="Storage error"):
            await storage_client.connect()
        assert storage_client._client is None

    @pytest.mark.asyncio
    async def test_close(self, storage_client, mock_s3_client):
        await storage_client.connect()
        create_client = storage_client._session.create_client.return_value

        await storage_client.close()
        create_client.__aexit__.assert_awaited_once()
        assert storage_client._client is None

        # closing an already closed client is a no-op
        await storage_client.close()
        create_client.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_object(self, mocker, storage_client, mock_s3_client):
        await storage_client.connect()

        body = mocker.MagicMock(name="StreamingBody")
        body.__aenter__ = mocker.AsyncMock(return_value=body)
        body.__aexit__ = mocker.AsyncMock(return_value=False)
        body.read = mocker.AsyncMock(return_value=b"image-bytes")
        mock_s3_client.get_object = mocker.AsyncMock(return_value={"Body": body})

        data = await storage_client.retrieve_object("images/image.jpg")
        assert data == b"image-bytes"
        mock_s3_client.get_object.assert_awaited_once_with(
            Bucket=storage_client._bucket, Key="images/image.jpg"
        )

        await storage_client.retrieve_object("images/image.jpg", bucket="other-bucket")
        assert mock_s3_client.get_object.await_args.kwargs["Bucket"] == "other-bucket"

        mock_s3_client.get_object.side_effect = RuntimeError("Storage error")
        assert await storage_client.retrieve_object("images/image.jpg") is None

    @pytest.mark.asyncio
    async def test_get_presigned_url(self, mocker, storage_client, mock_s3_client):
        await storage_client.connect()
        mock_s3_client.generate_presigned_url = mocker.AsyncMock(
            return_value="https://storage.local/presigned/image.jpg"
        )

        url = await storage_client.get_presigned_url("images/image.jpg")
        assert url == "https://storage.local/presigned/image.jpg"
        mock_s3_client.generate_presigned_url.assert_awaited_once_with(
            ClientMethod="get_object",
            Params={"Bucket": storage_client._bucket, "Key": "images/image.jpg"},
            ExpiresIn=PRESIGNED_URL_EXPIRED_IN
        )

        await storage_client.get_presigned_url("images/image.jpg", bucket="other-bucket", expires_in=60)
        call_kwargs = mock_s3_client.generate_presigned_url.await_args.kwargs
        assert call_kwargs["Params"]["Bucket"] == "other-bucket"
        assert call_kwargs["ExpiresIn"] == 60

        mock_s3_client.generate_presigned_url.side_effect = RuntimeError("Storage error")
        assert await storage_client.get_presigned_url("images/image.jpg") == ""
