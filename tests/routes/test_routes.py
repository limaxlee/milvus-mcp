import pytest
import json
from starlette import status
from httpx import TimeoutException
from starlette.responses import Response

from milvus_mcp.routes.routes import check_health, download_logs


class TestCustomRoutes:
    @pytest.mark.asyncio
    async def test_check_health(self, mocker, mock_async_milvus_client):
        mock_async_client = mocker.patch("milvus_mcp.routes.routes.AsyncClient")
        async_client_instance = mock_async_client.return_value
        async_client_instance.__aenter__.return_value = async_client_instance
        async_client_instance.get = mocker.AsyncMock(return_value=Response(status_code=status.HTTP_200_OK))

        request = mocker.MagicMock()
        response = await check_health(request)
        assert response.status_code == status.HTTP_200_OK
        body = json.loads(response.body)
        assert body["MCP Server status"] == "healthy"
        assert body["DB status"] == "healthy"

        async_client_instance.get = mocker.AsyncMock(side_effect=TimeoutException("Request Timeout"))
        response = await check_health(request)
        assert response.status_code == status.HTTP_200_OK
        body = json.loads(response.body)
        assert body["MCP Server status"] == "healthy"
        assert body["DB status"] == "unhealthy"

        async_client_instance.get = mocker.AsyncMock(side_effect=Exception("MCP Error"))
        with pytest.raises(Exception, match="MCP Error"):
            await check_health(request)

    @pytest.mark.asyncio
    async def test_download_logs(self, mocker):
        request = mocker.MagicMock()

        mocker.patch("milvus_mcp.routes.routes.get_logs_zip_file", return_value=b'Log zip file')
        response = await download_logs(request)
        assert response.status_code == status.HTTP_200_OK
        assert response.media_type == "application/zip"
        assert response.body == b'Log zip file'

        mocker.patch("milvus_mcp.routes.routes.get_logs_zip_file", return_value=None)
        with pytest.raises(Exception, match="No log files found"):
            await download_logs(request)

        mocker.patch("milvus_mcp.routes.routes.get_logs_zip_file", side_effect=Exception("MCP Error"))
        with pytest.raises(Exception, match="MCP Error"):
            await download_logs(request)
