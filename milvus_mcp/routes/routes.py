import logging
import datetime
from fastmcp.server import FastMCP
from starlette import status
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.exceptions import HTTPException
from httpx import AsyncClient, TimeoutException

from common.config import SETTINGS
from common.constants import HEALTH_CHECK_PORT, HEALTH_CHECK_TIMEOUT
from milvus_mcp.utils import get_logs_zip_file

logger = logging.getLogger(__name__)

custom_mcp = FastMCP(name="custom_route")


@custom_mcp.custom_route("/health", methods=["GET"])
async def check_health(request: Request) -> JSONResponse:
    try:
        health_status = {"MCP Server status": "healthy"}
        health_check_uri = f"http://{SETTINGS.milvus_host}:{HEALTH_CHECK_PORT}/healthz"
        async with AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
            response = await client.get(health_check_uri)
            health_status["DB status"] = "healthy" if response.status_code == status.HTTP_200_OK else "unhealthy"

            logger.info(f"Checked server health status: {health_status}")
            return JSONResponse(health_status)
    except TimeoutException:
        logger.warning(f"Failed to check Milvus health: timed out after {HEALTH_CHECK_TIMEOUT}")
        return JSONResponse({"MCP Server status": "healthy", "DB status": "unhealthy"})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@custom_mcp.custom_route("/logs", methods=["GET"])
async def download_logs(request: Request) -> Response:
    try:
        zip_name = f"{datetime.datetime.now(datetime.UTC).strftime('%Y%m%d-%H%M%SZ')}.zip"
        zip_bytes = await get_logs_zip_file()
        if not zip_bytes:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No log files found")

        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_name}"'}
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
