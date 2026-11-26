import os
import pytest

from common.constants import DataFields, DETECTION_COLLECTION_SUFFIX
from milvus_mcp.schemas import SamplingResult, VectorSearchResult
from milvus_mcp.connector.connector import MilvusConnector


def mock_data_row(pk, vector, prediction, data_uri=None, filename=None):
    return {
        DataFields.PK: pk,
        DataFields.VECTOR: vector,
        DataFields.PREDICTION: prediction,
        DataFields.DATA_URI: data_uri if data_uri is not None else f"uri_{pk}",
        DataFields.FILENAME: filename if filename is not None else f"file_{pk}.jpg"
    }


@pytest.fixture
def stratified_sample_rows():
    return [
        mock_data_row(1, [1.0, 0.0], "Good"),
        mock_data_row(2, [0.9, 0.1], "Good"),
        mock_data_row(3, [0.0, 1.0], "Good"),
        mock_data_row(4, [5.0, 5.0], "NG"),
        mock_data_row(5, [5.1, 5.1], "NG")
    ]


class TestMilvusConnector:
    def test_client_init(self, mocker):
        mock_client = mocker.patch("milvus_mcp.connector.connector.AsyncMilvusClient")
        storage = mocker.MagicMock()
        connector = MilvusConnector(storage)
        assert connector._client is mock_client.return_value

    @pytest.mark.asyncio
    async def test_list_databases(self, mock_connector):
        result = await mock_connector.list_databases()
        assert result.databases == ["default", "test_db"]

        mock_connector._mock_client.list_databases.assert_awaited_once()

        mock_connector._mock_client.list_databases.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.list_databases()

    @pytest.mark.asyncio
    async def test_list_collections(self, mock_connector):
        result = await mock_connector.list_collections()
        assert result.collections == ["collection_a", "collection_b"]

        mock_connector._mock_client.list_collections.assert_awaited_once()

        mock_connector._mock_client.list_collections.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.list_collections()

    @pytest.mark.asyncio
    async def test_get_collection_info(self, mock_connector):
        result = await mock_connector.get_collection_info("collection_a")
        assert result.collection_name == "collection_a"
        assert isinstance(result.fields, list)

        mock_connector._mock_client.describe_collection.assert_awaited_once_with("collection_a")

        mock_connector._mock_client.describe_collection.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.get_collection_info("collection_b")

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, mock_connector):
        result = await mock_connector.get_collection_stats("collection_a")
        assert result.row_count == 42

        mock_connector._mock_client.get_collection_stats.assert_awaited_once_with("collection_a")

        mock_connector._mock_client.get_collection_stats.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.get_collection_stats("collection_b")

    @pytest.mark.asyncio
    async def test_create_collection(self, mock_connector):
        mock_connector._mock_client.has_collection.return_value = False
        await mock_connector.create_collection("collection_c")
        assert mock_connector._mock_client.create_collection.await_count == 1

        mock_connector._mock_client.create_collection.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.create_collection("collection_d")

    @pytest.mark.asyncio
    async def test_load_collection(self, mock_connector):
        await mock_connector.load_collection("collection_c")
        assert mock_connector._mock_client.load_collection.await_count == 1

        mock_connector._mock_client.load_collection.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.load_collection("collection_d")

    @pytest.mark.asyncio
    async def test_release_collection(self, mock_connector):
        await mock_connector.release_collection("collection_c")
        assert mock_connector._mock_client.release_collection.await_count == 1

        mock_connector._mock_client.release_collection.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.release_collection("collection_d")

    @pytest.mark.asyncio
    async def test_insert_data(self, mock_connector):
        result = await mock_connector.insert_data("collection_a", [{"vector": [1.0, 1.1, 1.2]}])
        assert result.insert_count == 1
        assert result.primary_keys == [462025919837366264]
        assert mock_connector._mock_client.insert.await_count == 1

        mock_connector._mock_client.insert.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.insert_data("collection_b", [{"vector": [2.0, 2.1, 2.2]}])

    @pytest.mark.asyncio
    async def test_delete_data(self, mock_connector):
        result = await mock_connector.delete_entities("collection_a", "image_name == 'image.jpg'")
        assert result.delete_count == 1
        assert mock_connector._mock_client.delete.await_count == 1

        mock_connector._mock_client.delete.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.delete_entities("collection_b", "image_name == 'image.jpg'")

    @pytest.mark.asyncio
    async def test_vector_search(self, mock_connector):
        results = await mock_connector.vector_search("collection_a", [1.0, 2.0], "vector")
        assert results.search_data_count == 5
        assert mock_connector._mock_client.search.await_count == 1

        mock_connector._mock_client.search.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.vector_search("collection_a", [1.0, 2.0], "vector")

    @pytest.mark.asyncio
    async def test_extract_embeddings_and_vector_search(self, mock_connector):
        client = mock_connector._mock_client
        storage = mock_connector._mock_storage
        extractor = mock_connector._mock_extractor

        client.search.return_value = [[
            {"id": 462025919837366269, "distance": 0.99, "entity": {DataFields.DATA_URI: "uri_a"}},
            {"id": 462025919837366277, "distance": 0.95, "entity": {DataFields.DATA_URI: "uri_b"}}
        ]]

        result = await mock_connector.extract_embeddings_and_vector_search(
            "collection_a", "vector", "image.jpg", "data/uri.jpg", "image/jpeg"
        )

        assert isinstance(result, VectorSearchResult)
        assert result.collection_name == "collection_a"
        assert result.search_data_count == 2
        assert result.search_data_keys == [462025919837366269, 462025919837366277]

        storage.retrieve_object.assert_awaited_once_with("data/uri.jpg")
        extractor.extract_feature_vector.assert_awaited_once_with(
            'image.jpg', b'image_bytes', 'image/jpeg'
        )
        assert client.search.await_count == 1
        assert storage.get_presigned_url.await_count == 2

        storage.retrieve_object.return_value = None
        with pytest.raises(ValueError):
            await mock_connector.extract_embeddings_and_vector_search(
                "collection_a", "vector", "image.jpg", "data/uri.jpg", "image/jpeg"
            )

        storage.retrieve_object.return_value = b"image_bytes"
        extractor.extract_feature_vector.return_value = []
        with pytest.raises(ValueError):
            await mock_connector.extract_embeddings_and_vector_search(
                "collection_a", "vector", "image.jpg", "data/uri.jpg", "image/jpeg"
            )

        extractor.extract_feature_vector.return_value = [1.0, 2.0, 3.0]
        client.search.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError):
            await mock_connector.extract_embeddings_and_vector_search(
                "collection_a", "vector", "image.jpg", "data/uri.jpg", "image/jpeg"
            )

    @pytest.mark.asyncio
    async def test_query_collection(self, mock_connector):
        results = await mock_connector.query_collection("collection_a", "image_name == 'image.jpg'")
        assert results.query_data_count == 1
        assert results.query_data == [{"pk": "462025919837366269"}]
        assert mock_connector._mock_client.query.await_count == 1

        mock_connector._mock_client.query.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.query_collection("collection_a", "image_name == 'image.jpg'")

    @pytest.mark.asyncio
    async def test_get_k_center_sampled_data(self, mock_connector, stratified_sample_rows):
        client = mock_connector._mock_client
        client.query.return_value = stratified_sample_rows

        client.has_collection.side_effect = [True, False]
        result = await mock_connector.get_k_center_sampled_data("collection_a", {"Good": 2, "NG": 1})

        assert isinstance(result, SamplingResult)
        assert result.collection_name == "collection_a"
        assert result.total_scanned_data == 5
        assert result.total_sampled_data_size == 3
        assert result.class_distribution == {"Good": 2, "NG": 1}
        assert len(result.sampled_data_keys) == 3
        assert set(result.sampled_data_keys) <= {1, 2, 3, 4, 5}
        assert all(DataFields.VECTOR not in row for row in result.sampled_data.values())

        predictions = [row[DataFields.PREDICTION] for row in result.sampled_data.values()]
        assert predictions.count("Good") == 2
        assert predictions.count("NG") == 1

        client.has_collection.side_effect = [True, True]
        detection_result = await mock_connector.get_k_center_sampled_data("collection_a", {"Good": 2})
        assert detection_result.collection_name == f"collection_a{DETECTION_COLLECTION_SUFFIX}"

        client.has_collection.side_effect = [True, False]
        client.query.return_value = []
        empty_result = await mock_connector.get_k_center_sampled_data("collection_a", {"Good": 2})
        assert empty_result.collection_name == "collection_a"
        assert not empty_result.sampled_data_keys

        client.has_collection.side_effect = None
        client.has_collection.return_value = False
        with pytest.raises(ValueError, match="doesn't exist"):
            await mock_connector.get_k_center_sampled_data("missing_collection", {"Good": 2})

        client.has_collection.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.get_k_center_sampled_data("collection_a", {"Good": 2})

    @pytest.mark.asyncio
    async def test_get_k_center_sampled_data_as_zip_file(self, mocker, mock_connector, stratified_sample_rows):
        client = mock_connector._mock_client
        client.query.return_value = stratified_sample_rows
        client.has_collection.side_effect = [True, True, False]

        packaged_data = {}

        async def fake_create_zip_file(zip_file, sampled_data, storage, **kwargs):
            packaged_data.update({"zip_file": zip_file, "sampled_data": sampled_data})
            with open(zip_file, "wb"):
                pass

        mocker.patch("milvus_mcp.connector.connector.create_sample_zip_file", side_effect=fake_create_zip_file)
        result = await mock_connector.get_k_center_sampled_data_as_zip_file("collection_a", {"Good": 2, "NG": 1})
        assert result.total_scanned_data == 5
        assert result.total_sampled_data_size == 3
        assert result.class_distribution == {"Good": 2, "NG": 1}
        assert result.sampled_zip_uri == "link-23b181f4e6"
        assert result.sampled_data == {}
        assert result.sampled_data_keys == []

        zip_file, zip_uri = mock_connector._mock_storage.upload_object.await_args.args
        assert zip_file == packaged_data["zip_file"]
        assert zip_uri.startswith("MilvusMCP/collection_a/collection_a")

        mock_connector._mock_storage.get_presigned_url.assert_awaited_once_with(zip_uri)
        assert not os.path.exists(zip_file)

        client.has_collection.side_effect = [True, True, False]

        mocker.patch("milvus_mcp.connector.connector.create_sample_zip_file", side_effect=RuntimeError("Error"))
        with pytest.raises(RuntimeError, match="Error"):
            await mock_connector.get_k_center_sampled_data_as_zip_file("collection_a", {"Good": 2})

    @pytest.mark.asyncio
    async def test_close(self, mock_connector):
        await mock_connector.close()
        mock_connector._mock_client.close.assert_called_once()

        mock_connector._mock_client.close.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.close()
