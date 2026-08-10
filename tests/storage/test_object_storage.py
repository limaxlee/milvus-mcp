import pytest

from common.config import SETTINGS
from common.constants import PRESIGNED_URL_EXPIRED_IN
from milvus_mcp.storage import ObjectStorage
from milvus_mcp.storage import object_storage as object_storage_module


@pytest.fixture
def storage_client(mocker):
    storage = ObjectStorage()
    storage._session = mocker.MagicMock()

    return storage


@pytest.fixture
def mock_s3_client(mocker, storage_client):
    s3_client = mocker.MagicMock(name="S3Client")
    create_client = mocker.MagicMock(name="CreateClient")
    create_client.__aenter__ = mocker.AsyncMock(return_value=s3_client)
    create_client.__aexit__ = mocker.AsyncMock(return_value=False)
    storage_client._session.create_client.return_value = create_client

    return s3_client


class TestObjectStorage:
    def test_config_init(self, storage_client):
        assert storage_client._client is None
        assert storage_client._bucket == SETTINGS.object_storage.bucket
        assert storage_client._config == {
            "bucket": SETTINGS.object_storage.bucket,
            "endpoint": SETTINGS.object_storage.endpoint,
            "access_key": SETTINGS.object_storage.access_key,
            "secret_key": SETTINGS.object_storage.secret_key
        }

    @pytest.mark.asyncio
    async def test_connect(self, storage_client, mock_s3_client):
        result = await storage_client.connect()

        assert result is storage_client
        assert storage_client._client is mock_s3_client
        assert storage_client._session.create_client.call_count == 1

        create_kwargs = storage_client._session.create_client.call_args.kwargs
        assert storage_client._session.create_client.call_args.args == ("s3",)
        assert create_kwargs["endpoint_url"] == SETTINGS.object_storage.endpoint
        assert create_kwargs["aws_access_key_id"] == SETTINGS.object_storage.access_key
        assert create_kwargs["aws_secret_access_key"] == SETTINGS.object_storage.secret_key

        await storage_client.connect()
        assert storage_client._session.create_client.call_count == 1

    @pytest.mark.asyncio
    async def test_connect_failure(self, storage_client):
        storage_client._session.create_client.side_effect = object_storage_module.ClientError(
            {"Error": {"Code": "403", "Message": "Forbidden"}}, "CreateClient"
        )
        with pytest.raises(object_storage_module.ClientError):
            await storage_client.connect()
        assert storage_client._client is None

        storage_client._session.create_client.side_effect = RuntimeError("Storage error")
        with pytest.raises(RuntimeError):
            await storage_client.connect()
        assert storage_client._client is None

    @pytest.mark.asyncio
    async def test_close(self, storage_client, mock_s3_client):
        await storage_client.connect()
        create_client = storage_client._session.create_client.return_value

        await storage_client.close()

        create_client.__aexit__.assert_awaited_once()
        assert storage_client._client is None

        await storage_client.close()
        create_client.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retrieve_object(self, mocker, storage_client, mock_s3_client):
        await storage_client.connect()

        body = mocker.MagicMock()
        body.__aenter__ = mocker.AsyncMock(return_value=body)
        body.__aexit__ = mocker.AsyncMock(return_value=False)
        body.read = mocker.AsyncMock(return_value=b"image-bytes")
        mock_s3_client.get_object = mocker.AsyncMock(return_value={"Body": body})

        data = await storage_client.retrieve_object("images/image.jpg")

        assert data == b"image-bytes"
        mock_s3_client.get_object.assert_awaited_once_with(Bucket=storage_client._bucket, Key="images/image.jpg")
        body.__aexit__.assert_awaited_once()

        await storage_client.retrieve_object("images/image.jpg", bucket="other-bucket")
        assert mock_s3_client.get_object.await_args.kwargs["Bucket"] == "other-bucket"

        mock_s3_client.get_object.side_effect = RuntimeError("Storage error")
        assert await storage_client.retrieve_object("images/image.jpg") is None

    @pytest.mark.asyncio
    async def test_upload_file(self, mocker, storage_client, mock_s3_client, tmp_path):
        await storage_client.connect()

        uploaded = {}

        async def capture(Bucket, Key, Body):
            uploaded.update({"bucket": Bucket, "key": Key, "body": Body.read()})
            return {}

        mock_s3_client.put_object = mocker.AsyncMock(side_effect=capture)

        zip_file = tmp_path / "sampled.zip"
        zip_file.write_bytes(b"zip-bytes")

        await storage_client.upload_file(str(zip_file), "MilvusMCP/collection_a/sampled.zip")

        assert uploaded == {
            "bucket": storage_client._bucket,
            "key": "MilvusMCP/collection_a/sampled.zip",
            "body": b"zip-bytes"
        }

        await storage_client.upload_file(str(zip_file), "sampled.zip", bucket="other-bucket")
        assert uploaded["bucket"] == "other-bucket"

        mock_s3_client.put_object.side_effect = RuntimeError("Storage error")
        with pytest.raises(RuntimeError, match="Storage error"):
            await storage_client.upload_file(str(zip_file), "sampled.zip")

    @pytest.mark.asyncio
    async def test_get_presigned_url(self, mocker, storage_client, mock_s3_client):
        await storage_client.connect()
        mock_s3_client.generate_presigned_url = mocker.AsyncMock(return_value="https://presigned/image.jpg")

        url = await storage_client.get_presigned_url("images/image.jpg")

        assert url == "https://presigned/image.jpg"
        mock_s3_client.generate_presigned_url.assert_awaited_once_with(
            ClientMethod="get_object",
            Params={"Bucket": storage_client._bucket, "Key": "images/image.jpg"},
            ExpiresIn=PRESIGNED_URL_EXPIRED_IN
        )

        await storage_client.get_presigned_url("images/image.jpg", bucket="test-bucket", expires_in=60)

        call_kwargs = mock_s3_client.generate_presigned_url.await_args.kwargs
        assert call_kwargs["Params"]["Bucket"] == "test-bucket"
        assert call_kwargs["ExpiresIn"] == 60

        mock_s3_client.generate_presigned_url.side_effect = RuntimeError("Storage error")
        with pytest.raises(RuntimeError):
            await storage_client.get_presigned_url("images/image.jpg")
