import pytest

from common.constants import DataFields
from milvus_mcp.connector.connector import MilvusConnector


class TestMilvusConnector:
    def test_client_init(self, mocker):
        mock_client = mocker.patch("milvus_mcp.connector.connector.AsyncMilvusClient")
        connector = MilvusConnector()
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
        assert len(results) == 5
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
    async def test_extract_embeddings_and_vector_search(self, mock_connector):
        presigned_url = mock_connector._mock_storage.get_presigned_url.return_value
        results = await mock_connector.extract_embeddings_and_vector_search(
            collection_name="collection_a",
            vector_field="vector",
            filename="image.jpg",
            data_uri="images/image.jpg",
            content_type="image/jpeg"
        )

        assert len(results) == 5
        assert all(result["entity"][DataFields.DATA_URI] == presigned_url for result in results)

        mock_connector._mock_storage.retrieve_object.assert_awaited_once_with("images/image.jpg")
        mock_connector._mock_extractor.extract_feature_vector.assert_awaited_once_with(
            "collection_a", "image.jpg", b"image-bytes", "image/jpeg"
        )
        assert mock_connector._mock_client.search.await_count == 1
        assert mock_connector._mock_storage.get_presigned_url.await_count == 5

        # each failure below happens one stage earlier in the pipeline than the previous one
        mock_connector._mock_storage.get_presigned_url.return_value = ""
        with pytest.raises(ValueError, match="Failed to generate presigned URL"):
            await mock_connector.extract_embeddings_and_vector_search(
                "collection_a", "vector", "image.jpg", "images/image.jpg", "image/jpeg"
            )

        mock_connector._mock_client.search.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.extract_embeddings_and_vector_search(
                "collection_a", "vector", "image.jpg", "images/image.jpg", "image/jpeg"
            )

        mock_connector._mock_extractor.extract_feature_vector.return_value = []
        with pytest.raises(ValueError, match="Failed to extract feature vector"):
            await mock_connector.extract_embeddings_and_vector_search(
                "collection_a", "vector", "image.jpg", "images/image.jpg", "image/jpeg"
            )

        mock_connector._mock_storage.retrieve_object.return_value = None
        with pytest.raises(ValueError, match="Failed to retrieve"):
            await mock_connector.extract_embeddings_and_vector_search(
                "collection_a", "vector", "image.jpg", "images/image.jpg", "image/jpeg"
            )

    @pytest.mark.asyncio
    async def test_extract_embeddings_and_vector_search_output_fields(self, mock_connector):
        await mock_connector.extract_embeddings_and_vector_search(
            "collection_a", "vector", "image.jpg", "images/image.jpg", "image/jpeg"
        )
        assert mock_connector._mock_client.search.await_args.kwargs["output_fields"] == [DataFields.DATA_URI]

        output_fields = ["prediction"]
        await mock_connector.extract_embeddings_and_vector_search(
            "collection_a", "vector", "image.jpg", "images/image.jpg", "image/jpeg", output_fields=output_fields
        )
        assert mock_connector._mock_client.search.await_args.kwargs["output_fields"] == \
               ["prediction", DataFields.DATA_URI]
        assert output_fields == ["prediction"]

        output_fields = ["prediction", DataFields.DATA_URI]
        await mock_connector.extract_embeddings_and_vector_search(
            "collection_a", "vector", "image.jpg", "images/image.jpg", "image/jpeg", output_fields=output_fields
        )
        assert mock_connector._mock_client.search.await_args.kwargs["output_fields"] == \
               ["prediction", DataFields.DATA_URI]

    @pytest.mark.asyncio
    async def test_close(self, mock_connector):
        await mock_connector.close()
        mock_connector._mock_client.close.assert_called_once()

        mock_connector._mock_client.close.side_effect = RuntimeError("Connector error")
        with pytest.raises(RuntimeError, match="Connector error"):
            await mock_connector.close()
            