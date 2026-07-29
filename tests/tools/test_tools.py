import json
import pytest
import pytest_asyncio
from fastmcp.exceptions import ToolError

from milvus_mcp.tools import tools_mcp


@pytest_asyncio.fixture
async def tools_by_name():
    tools = await tools_mcp.list_tools()
    return {tool.name: tool.fn for tool in tools}


class TestMilvusMCPTools:
    @pytest.mark.asyncio
    async def test_milvus_list_databases(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_list_databases"]
        result = await tool(mock_context)

        assert "default" in result
        assert "test_db" in result

        mock_context.request_context.lifespan_context.connector.list_databases = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_milvus_list_collections(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_list_collections"]
        result = await tool(mock_context)

        assert "collection_a" in result
        assert "collection_b" in result

        mock_context.request_context.lifespan_context.connector.list_collections = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context)

    @pytest.mark.asyncio
    async def test_milvus_get_collection_info(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_get_collection_info"]
        result = await tool("collection_a", mock_context)

        prefix = "Collection information:\n"
        assert result.startswith(prefix)
        parsed_result = json.loads(result[len(prefix):])
        assert parsed_result["collection_name"] == "collection_a"

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

        assert result == {"row_count": 42}

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

        await tool(mock_context, "collection_c", True, 768, "id", "vector", "COSINE", None, None, None)

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.create_collection.await_count == 1
        mock_context.request_context.lifespan_context.connector.create_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_c", True, 768, "id", "vector", "COSINE", None, None, None)

    @pytest.mark.asyncio
    async def test_milvus_load_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_load_collection"]
        await tool(mock_context, "collection_c", 1)

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.load_collection.await_count == 1

        mock_context.request_context.lifespan_context.connector.load_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "load_collection", 1)

    @pytest.mark.asyncio
    async def test_milvus_release_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_release_collection"]
        await tool("collection_c", mock_context)

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

        assert result == f"Data inserted into collection collection_a with result: ['462025919837366264']"

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

        assert result == "Entities deleted from collection collection_a with result: {'delete_count': 1}"

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
    async def test_milvus_query_collection(self, mocker, tools_by_name, mock_context):
        tool = tools_by_name["milvus_query_collection"]
        await tool(mock_context, "collection_a", "image_name == 'image.jpg'")

        connector = mock_context.request_context.lifespan_context.connector
        assert connector._mock_client.query.await_count == 1

        mock_context.request_context.lifespan_context.connector.query_collection = \
            mocker.AsyncMock(side_effect=RuntimeError())
        with pytest.raises(ToolError):
            await tool(mock_context, "collection_a", "image_name == 'image.jpg'")
