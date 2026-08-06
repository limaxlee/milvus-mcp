import pytest

from common.constants import DataFields, DETECTION_COLLECTION_SUFFIX
from milvus_mcp.schemas import SamplingResult
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
        assert result == ["default", "test_db"]

        mock_connector._mock_client.list_databases.assert_awaited_once()

        mock_connector._mock_client.list_databases.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.list_databases()

    @pytest.mark.asyncio
    async def test_list_collections(self, mock_connector):
        result = await mock_connector.list_collections()
        assert result == ["collection_a", "collection_b"]

        mock_connector._mock_client.list_collections.assert_awaited_once()

        mock_connector._mock_client.list_collections.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.list_collections()

    @pytest.mark.asyncio
    async def test_get_collection_info(self, mock_connector):
        result = await mock_connector.get_collection_info("collection_a")
        assert result["collection_name"] == "collection_a"
        assert "fields" in result

        mock_connector._mock_client.describe_collection.assert_awaited_once_with("collection_a")

        mock_connector._mock_client.describe_collection.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.get_collection_info("collection_b")

    @pytest.mark.asyncio
    async def test_get_collection_stats(self, mock_connector):
        result = await mock_connector.get_collection_stats("collection_a")
        assert result == {"row_count": 42}

        mock_connector._mock_client.get_collection_stats.assert_awaited_once_with("collection_a")

        mock_connector._mock_client.get_collection_stats.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.get_collection_stats("collection_b")

    @pytest.mark.asyncio
    async def test_create_collection(self, mock_connector):
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
        assert result == ["462025919837366264"]
        assert mock_connector._mock_client.insert.await_count == 1

        mock_connector._mock_client.insert.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.insert_data("collection_b", [{"vector": [2.0, 2.1, 2.2]}])

    @pytest.mark.asyncio
    async def test_delete_data(self, mock_connector):
        result = await mock_connector.delete_entities("collection_a", "image_name == 'image.jpg'")
        assert result == {"delete_count": 1}
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
    async def test_query_collection(self, mock_connector):
        results = await mock_connector.query_collection("collection_a", "image_name == 'image.jpg'")
        assert results == [{"pk": "462025919837366269"}]
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
    async def test_close(self, mock_connector):
        await mock_connector.close()
        mock_connector._mock_client.close.assert_called_once()

        mock_connector._mock_client.close.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.close()
