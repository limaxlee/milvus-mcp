FROM docker-remote.bart.sec.samsung.net/python:3.13.13

COPY ./cosmo-milvus-mcp-server /home/work/cosmo-milvus-mcp-server
COPY ./config.yaml /home/work/cosmo-milvus-mcp-server/config.yaml

WORKDIR /home/work/cosmo-milvus-mcp-server
RUN python -m pip install -r requirements_py313_prod.txt --no-cache-dir

CMD ["python", "-m", "milvus_mcp", "-c", "/home/work/cosmo-milvus-mcp-server/config.yaml"]
