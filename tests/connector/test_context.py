from milvus_mcp.connector.context import MilvusContext


class TestMilvusContext:
    def test_context_stores_connector(self, mocker):
        connector = mocker.MagicMock(name="connector")
        ctx = MilvusContext(connector)

        assert ctx.connector is connector
        