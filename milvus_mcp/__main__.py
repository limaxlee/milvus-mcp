import uvicorn
from typing import AsyncIterator
from fastmcp.server import FastMCP
from contextlib import asynccontextmanager
from starlette.middleware.cors import CORSMiddleware

from common.config import SETTINGS
from milvus_mcp.connector import MilvusContext, MilvusConnector
from milvus_mcp.middleware import LoggingMiddleware
from milvus_mcp.storage import ObjectStorageClient
from milvus_mcp.routes import custom_mcp
from milvus_mcp.tools import tools_mcp
from milvus_mcp.utils import initialize_logger


@asynccontextmanager
async def lifespan(mcp: FastMCP) -> AsyncIterator[MilvusContext]:
    storage_client = ObjectStorageClient()
    await storage_client.__aenter__()

    connector = MilvusConnector()

    try:
        yield MilvusContext(connector)
    finally:
        await connector.close()
        await storage_client.__aexit__()


mcp = FastMCP(name="Milvus MCP Server", lifespan=lifespan)
mcp.mount(tools_mcp, prefix="mcp")
mcp.mount(custom_mcp)
mcp.add_middleware(LoggingMiddleware())

app = mcp.http_app()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["mcp-session-id"]
)

if __name__ == "__main__":
    logger = initialize_logger("milvus_mcp_server.log")
    logger.info("Starting Milvus MCP Server")

    uvicorn.run(app, host="0.0.0.0", port=SETTINGS.server_port)
