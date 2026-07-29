import sys
import yaml
import pytest
import pathlib

from milvus_mcp.connector.connector import MilvusConnector

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def pytest_configure(config):
    cfg = ROOT_DIR / "config.yaml"
    if not cfg.is_file():
        raise pytest.UsageError("Missing configuration file")


@pytest.fixture
def sample_config_dict():
    return {
        "server_port": 8443,
        "milvus_host": "10.240.35.173",
        "milvus_port": 19530,
        "milvus_user": "",
        "milvus_password": "",
        "milvus_db_name": "default"
    }


@pytest.fixture
def tmp_config_file(tmp_path, sample_config_dict):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(sample_config_dict))
    return cfg_path


@pytest.fixture
def mock_async_milvus_client(mocker):
    client = mocker.MagicMock(name="AsyncMilvusClient")
    client.list_databases = mocker.AsyncMock(return_value=["default", "test_db"])
    client.list_collections = mocker.AsyncMock(return_value=["collection_a", "collection_b"])
    client.describe_collection = mocker.AsyncMock(
        return_value={
            "collection_name": "collection_a",
            "auto_id": False,
            "num_shards": 1,
            "description": "test collection",
            "fields": [
                {"name": "id", "type": 5, "is_primary": True}
            ]
        }
    )
    client.get_collection_stats = mocker.AsyncMock(return_value={"row_count": 42})
    client.has_collection = mocker.AsyncMock(return_value=False)
    client.create_collection = mocker.AsyncMock(return_value=None)
    client.load_collection = mocker.AsyncMock(return_value=None)
    client.release_collection = mocker.AsyncMock(return_value=None)
    client.get_collection_stats = mocker.AsyncMock(return_value={"row_count": 42})
    client.insert_data = mocker.AsyncMock(return_value=["462025919837366264"])
    client.insert = mocker.AsyncMock(return_value=["462025919837366264"])
    client.delete_entities = mocker.AsyncMock(return_value={"delete_count": 1})
    client.delete = mocker.AsyncMock(return_value={"delete_count": 1})
    # built per call so that in-place mutation of the hits does not leak between searches
    client.search = mocker.AsyncMock(side_effect=lambda *args, **kwargs: [[
        {"id": 462025919837366269, "distance": 0.9999997615814209, "entity": {"data_uri": "images/image_1.jpg"}},
        {"id": 462025919837366277, "distance": 0.9594927430152893, "entity": {"data_uri": "images/image_2.jpg"}},
        {"id": 462025919837366263, "distance": 0.9590328335762024, "entity": {"data_uri": "images/image_3.jpg"}},
        {"id": 462025919837366274, "distance": 0.95219886302948, "entity": {"data_uri": "images/image_4.jpg"}},
        {"id": 462025919837366272, "distance": 0.951366126537323, "entity": {"data_uri": "images/image_5.jpg"}}
    ]])
    client.vector_search = mocker.AsyncMock(return_value=[[
        {"id": 462025919837366269, "distance": 0.9999997615814209, "entity": {}},
        {"id": 462025919837366277, "distance": 0.9594927430152893, "entity": {}},
        {"id": 462025919837366263, "distance": 0.9590328335762024, "entity": {}},
        {"id": 462025919837366274, "distance": 0.95219886302948, "entity": {}},
        {"id": 462025919837366272, "distance": 0.951366126537323, "entity": {}}
    ]])
    client.query = mocker.AsyncMock(return_value=[{"pk": "462025919837366269"}])
    client.query_collection = mocker.AsyncMock(return_value=[{"pk": "462025919837366269"}])
    client.close = mocker.AsyncMock(return_value=None)

    return client

@pytest.fixture
def mock_storage_client(mocker):
    storage = mocker.MagicMock(name="ObjectStorageClient")
    storage.connect = mocker.AsyncMock(return_value=storage)
    storage.close = mocker.AsyncMock(return_value=None)
    storage.retrieve_object = mocker.AsyncMock(return_value=b"image-bytes")
    storage.get_presigned_url = mocker.AsyncMock(return_value="https://storage.local/presigned/image.jpg")

    return storage


@pytest.fixture
def mock_feature_extractor(mocker):
    extractor = mocker.MagicMock(name="FeatureExtractor")
    extractor.extract_feature_vector = mocker.AsyncMock(return_value=[0.1, 0.2, 0.3])

    return extractor


@pytest.fixture
def mock_connector(mocker, mock_async_milvus_client, mock_storage_client, mock_feature_extractor):
    mocker.patch("milvus_mcp.connector.connector.AsyncMilvusClient", return_value=mock_async_milvus_client)
    connector = MilvusConnector()
    connector._storage = mock_storage_client
    connector._extractor = mock_feature_extractor
    connector._mock_client = mock_async_milvus_client
    connector._mock_storage = mock_storage_client
    connector._mock_extractor = mock_feature_extractor
    yield connector


@pytest.fixture
def mock_context(mocker, mock_connector):
    ctx = mocker.MagicMock()
    ctx.request_context.lifespan_context.connector = mock_connector
    return ctx
