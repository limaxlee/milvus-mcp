import pytest
import pytest_asyncio
from fastmcp.exceptions import ToolError

from common.constants import DataFields, VFM_COLLECTION_PREFIX
from milvus_mcp.tools import tools_mcp
from milvus_mcp.schemas import SamplingResult


@pytest_asyncio.fixture
async def tools_by_name():
    tools = await tools_mcp.list_tools()
    return {tool.name: tool.fn for tool in tools}


class TestMilvusMCPTools:
    @pytest.mark.asyncio
    async def test_milvus_list_databases(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_list_databases"]
        result = await tool(mock_context)

        assert result.databases == ["default", "test_db"]

        mock_context.request_context.lifespan_context.connector.list_databases = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_milvus_list_collections(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_list_collections"]
        result = await tool(mock_context)

        assert result.collections == ["collection_a", "collection_b"]

        mock_context.request_context.lifespan_context.connector.list_collections = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_milvus_get_collection_info(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_get_collection_info"]
        result = await tool("collection_a", mock_context)

        assert result.collection_name == "collection_a"
        assert result.classes == {"None": 1}
        assert result.collection_info["auto_id"] is False
        assert result.collection_info["fields"] == [{"name": "id", "type": 5, "is_primary": True}]
        assert "classes" not in result.collection_info

        await tool("my_collection", mock_context)

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.describe_collection.await_count == 2

        mock_context.request_context.lifespan_context.connector.get_collection_info = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool("invalid_collection", mock_context)

    @pytest.mark.asyncio
    async def test_milvus_get_collection_stats(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_get_collection_stats"]
        result = await tool("collection_a", mock_context)

        assert result.collection_name == "collection_a"
        assert result.collection_stats == {"row_count": 42}

        await tool("my_collection", mock_context)

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.get_collection_stats.await_count == 2

        mock_context.request_context.lifespan_context.connector.get_collection_stats = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool("invalid_collection", mock_context)

    @pytest.mark.asyncio
    async def test_milvus_create_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_create_collection"]
        mock_context.request_context.lifespan_context.connector._mock_client.has_collection.return_value = False

        result = await tool(mock_context, "collection_c", True, 768, "id", "vector", "COSINE", None, None, None)

        assert result.collection_name == "collection_c"
        assert result.collection_created is True

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.create_collection.await_count == 1
        mock_context.request_context.lifespan_context.connector.create_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_c", True, 768, "id", "vector", "COSINE", None, None, None)

    @pytest.mark.asyncio
    async def test_milvus_load_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_load_collection"]
        result = await tool(mock_context, "collection_c", 2)

        assert result.collection_name == "collection_c"
        assert result.collection_loaded is True
        assert result.replica_number == 2

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.load_collection.await_count == 1

        mock_context.request_context.lifespan_context.connector.load_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "load_collection", 1)

    @pytest.mark.asyncio
    async def test_milvus_release_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_release_collection"]
        result = await tool("collection_c", mock_context)

        assert result.collection_name == "collection_c"
        assert result.collection_released is True

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.release_collection.await_count == 1

        mock_context.request_context.lifespan_context.connector.release_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool("release_collection", mock_context)

    @pytest.mark.asyncio
    async def test_milvus_insert_data(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_insert_data"]
        result = await tool("collection_a", [{"vector": [1.0, 1.1, 1.2]}], mock_context)

        assert result.collection_name == "collection_a"
        assert result.insert_count == 1
        assert result.primary_keys == [462025919837366264]

        await tool("collection_b", [{"vector": [1.0, 1.1, 1.2]}], mock_context)

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.insert.await_count == 2

        mock_context.request_context.lifespan_context.connector.insert_data = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool("invalid_collection", [{"vector": [1.0, 1.1, 1.2]}], mock_context)

    @pytest.mark.asyncio
    async def test_milvus_delete_entities(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_delete_entities"]
        result = await tool("collection_a", "image_name == 'image.jpg'", mock_context)

        assert result.collection_name == "collection_a"
        assert result.delete_count == 1

        await tool("collection_b", "image_name == 'image.jpg'", mock_context)

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.delete.await_count == 2

        mock_context.request_context.lifespan_context.connector.delete_entities = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool("invalid_collection", "image_name == 'image.jpg'", mock_context)

    @pytest.mark.asyncio
    async def test_milvus_vector_search(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_vector_search"]
        await tool(mock_context, "collection_a", [1.0, 2.0], "vector")

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.search.await_count == 1

        mock_context.request_context.lifespan_context.connector.vector_search = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", [1.0, 2.0], "vector")

    @pytest.mark.asyncio
    async def test_milvus_vector_search_vfm_prefixed_fallback(self, tools_by_name, mock_context):
        tool = tools_by_name["milvus_vector_search"]
        client = mock_context.request_context.lifespan_context.connector._mock_client

        # ABC missing, VFM_ABC exists -> the search runs against VFM_ABC
        client.has_collection.side_effect = [False, True]
        result = await tool(mock_context, "ABC", [1.0, 2.0], "vector")
        assert result.collection_name == f"{VFM_COLLECTION_PREFIX}ABC"
        assert client.search.await_args.kwargs["collection_name"] == f"{VFM_COLLECTION_PREFIX}ABC"

        # Neither exists -> the tool error names the requested collection
        client.has_collection.side_effect = [False, False]
        with pytest.raises(ToolError, match="Collection ABC doesn't exist"):
            await tool(mock_context, "ABC", [1.0, 2.0], "vector")

    @pytest.mark.asyncio
    async def test_milvus_extract_embeddings_and_vector_search(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_extract_embeddings_and_vector_search"]
        connector = mock_context.request_context.lifespan_context.connector

        connector._mock_client.search.return_value = [[
            {"id": 462025919837366269, "distance": 0.99, "entity": {DataFields.DATA_URI: "uri_a"}},
            {"id": 462025919837366277, "distance": 0.95, "entity": {DataFields.DATA_URI: "uri_b"}}
        ]]

        result = await tool(
            mock_context, "collection_a", "vector", "image.jpg", "data/uri.jpg", "image/jpeg"
        )

        assert result.collection_name == "collection_a"
        assert result.search_data_count == 2
        assert connector._mock_client.search.await_count == 1

        connector._mock_storage.retrieve_object.assert_awaited_once_with("data/uri.jpg")
        connector._mock_extractor.extract_feature_vector.assert_awaited_once_with(
            "image.jpg", b"image_bytes", "image/jpeg"
        )

        mock_context.request_context.lifespan_context.connector.extract_embeddings_and_vector_search = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", "vector", "image.jpg", "data/uri.jpg", "image/jpeg")

    @pytest.mark.asyncio
    async def test_milvus_query_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_query_collection"]
        result = await tool(mock_context, "collection_a", "image_name == 'image.jpg'")

        assert result.collection_name == "collection_a"
        assert result.query_data_count == 1
        assert result.query_data == [{"pk": "462025919837366269"}]

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.query.await_count == 1

        mock_context.request_context.lifespan_context.connector.query_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", "image_name == 'image.jpg'")

    @pytest.mark.asyncio
    async def test_milvus_get_k_center_sampled_data(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_get_k_center_sampled_data"]
        connector = mock_context.request_context.lifespan_context.connector

        expected_result = SamplingResult(
            collection_name="collection_a",
            total_scanned_data=5,
            total_sampled_data_size=4,
            class_distribution={"Good": 2, "NG": 2},
            sampled_data_keys=[1, 2, 4, 5],
            sampled_data={}
        )
        connector.get_k_center_sampled_data = mocker.AsyncMock(return_value=expected_result)

        result = await tool(
            mock_context, "collection_a", "vector", {"Good": 2}, "COSINE", ["NG"], None, 1000
        )

        assert result == expected_result
        connector.get_k_center_sampled_data.assert_awaited_once_with(
            collection_name="collection_a",
            vector_field="vector",
            metric_type="COSINE",
            keep_classes=["NG"],
            sample_classes={"Good": 2},
            output_fields=None,
            batch_size=1000
        )

        connector.get_k_center_sampled_data = mocker.AsyncMock(return_value=expected_result)
        await tool(mock_context, "collection_a", "vector", {"Good": 2})
        connector.get_k_center_sampled_data.assert_awaited_once()

        connector.get_k_center_sampled_data = mocker.AsyncMock(side_effect=RuntimeError("Connector error"))
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", "vector", {"Good": 2})
