import logging
from typing import Any
from collections import Counter
from pymilvus import AsyncMilvusClient, DataType

from common.config import SETTINGS
from common.constants import DIMENSION, METRIC_TYPE, LIMIT, BATCH_SIZE, DataFields
from milvus_mcp.storage import ObjectStorageClient
from milvus_mcp.extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class MilvusConnector:
    def __init__(self):
        self._uri = f"http://{SETTINGS.milvus_host}:{SETTINGS.milvus_port}"
        self._token = f"{SETTINGS.milvus_user}:{SETTINGS.milvus_password}"
        self._client = AsyncMilvusClient(uri=self._uri, token=self._token, db_name=SETTINGS.milvus_db_name)
        self._storage = ObjectStorageClient()
        self._extractor = FeatureExtractor()

    async def list_databases(self) -> list[str]:
        try:
            results = await self._client.list_databases()
            logger.info(f"Listed databases: {results}")

            return results
        except Exception as e:
            logger.exception(f"Failed to list databases: {str(e)}")
            raise

    async def list_collections(self) -> list[str]:
        try:
            result = await self._client.list_collections()
            logger.info(f"Listed collections: {result}")

            return result
        except Exception as e:
            logger.exception(f"Failed to list collections: {str(e)}")
            raise

    async def get_collection_info(self, collection_name: str) -> dict:
        try:
            result = await self._client.describe_collection(collection_name)
            result["classes"] = await self._get_class_distribution(collection_name)
            logger.info(f"Retrieved {collection_name} collection info: {result}")

            return result
        except Exception as e:
            logger.exception(f"Failed to get {collection_name} collection info: {str(e)}")
            raise

    async def _get_class_distribution(self, collection_name: str) -> dict[str, int]:
        counter = Counter()
        last_pk = None
        while True:
            filter_expression = "" if last_pk is None else f"{DataFields.PK} > {last_pk}"
            data_rows = await self._client.query(
                collection_name=collection_name,
                filter=filter_expression,
                output_fields=[DataFields.PK, DataFields.PREDICTION],
                limit=BATCH_SIZE
            )

            if not data_rows:
                break

            for row in data_rows:
                counter[row.get(DataFields.PREDICTION)] += 1

            last_pk = max(row[DataFields.PK] for row in data_rows)
            if len(data_rows) < BATCH_SIZE:
                break

        return {str(value): count for value, count in counter.most_common()}

    async def get_collection_stats(self, collection_name: str) -> dict[str, Any]:
        try:
            results = await self._client.get_collection_stats(collection_name)
            logger.info(f"Retrieved {collection_name} collection statistics: {results}")

            return results
        except Exception as e:
            logger.exception(f"Failed to get {collection_name} collection statistics: {str(e)}")
            raise

    async def create_collection(
            self,
            collection_name: str,
            auto_id: bool = True,
            dimension: int = DIMENSION,
            primary_field_name: str = "id",
            vector_field_name: str = "vector",
            metric_type: str = METRIC_TYPE,
            field_schema: list[dict[str, Any]] | None = None,
            index_params: list[dict[str, Any]] | None = None,
            **kwargs: Any
    ):
        try:
            if await self._client.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} already exists")

            schema_kwargs = {
                "auto_id": auto_id,
                "enable_dynamic_field": kwargs.get("enable_dynamic_field", True)
            }
            if "partition_key_isolation" in kwargs:
                schema_kwargs["partition_key_isolation"] = kwargs["partition_key_isolation"]

            if field_schema is not None:
                schema = self._client.create_schema(**schema_kwargs)
                for field_kwargs in field_schema:
                    field_kwargs["datatype"] = getattr(DataType, field_kwargs["datatype"].upper())
                    schema.add_field(**field_kwargs)
            else:
                schema = None

            built_index_params = self._client.prepare_index_params()
            if index_params is not None and len(index_params) > 0:
                for index_kwargs in index_params:
                    built_index_params.add_index(**index_kwargs)

            await self._client.create_collection(
                collection_name=collection_name,
                auto_id=auto_id,
                dimension=dimension,
                primary_field_name=primary_field_name,
                vector_field_name=vector_field_name,
                metric_type=metric_type,
                schema=schema,
                index_params=built_index_params if index_params is not None else None,
                **kwargs
            )
        except Exception as e:
            logger.exception(f"Failed to create {collection_name} collection: {str(e)}")
            raise

    async def load_collection(self, collection_name: str, replica_number: int = 1):
        try:
            await self._client.load_collection(collection_name=collection_name, replica_number=replica_number)
        except Exception as e:
            logger.exception(f"Failed to load {collection_name} collection: {str(e)}")
            raise

    async def release_collection(self, collection_name: str):
        try:
            await self._client.release_collection(collection_name=collection_name)
        except Exception as e:
            logger.exception(f"Failed to release {collection_name} collection: {str(e)}")
            raise

    async def insert_data(self, collection_name: str, data: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            return await self._client.insert(collection_name=collection_name, data=data)
        except Exception as e:
            logger.exception(f"Failed to insert data to {collection_name} collection: {str(e)}")
            raise

    async def delete_entities(self, collection_name: str, filter_expression: str) -> dict[str, Any]:
        try:
            return await self._client.delete(collection_name=collection_name, filter=filter_expression)
        except Exception as e:
            logger.exception(f"Failed to delete data from {collection_name} using filter {filter_expression}:{str(e)}")
            raise

    async def vector_search(
            self,
            collection_name: str,
            vector: list[float],
            vector_field: str,
            limit: int = LIMIT,
            output_fields: list[str] | None = None,
            metric_type: str = METRIC_TYPE,
            filter_expression: str | None = None,
            radius: float | None = None,
            range_filter: float | None = None
    ) -> list[dict]:
        try:
            search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
            if radius is not None:
                search_params["params"]["radius"] = radius
            if range_filter is not None:
                search_params["params"]["range_filter"] = range_filter

            results = await self._client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field=vector_field,
                search_params=search_params,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expression
            )

            return results[0]
        except Exception as e:
            logger.exception(f"Failed to vector search in {collection_name} collection using {vector[:3]}: {str(e)}")
            raise

    async def query_collection(
            self,
            collection_name: str,
            filter_expression: str,
            output_fields: list[str] | None = None,
            limit: int = LIMIT
    ) -> list[dict]:
        try:
            return await self._client.query(
                collection_name=collection_name,
                filter=filter_expression,
                output_fields=output_fields,
                limit=limit
            )
        except Exception as e:
            logger.exception(f"Failed to query from {collection_name} collection using {filter_expression}: {str(e)}")
            raise

    async def extract_embeddings_and_vector_search(
            self,
            collection_name: str,
            vector_field: str,
            filename: str,
            data_uri: str,
            content_type: str,
            limit: int = LIMIT,
            output_fields: list[str] | None = None,
            metric_type: str = METRIC_TYPE,
            filter_expression: str | None = None,
            radius: float | None = None,
            range_filter: float | None = None
    ) -> list[dict]:
        try:
            if not output_fields:
                output_fields = [DataFields.DATA_URI]
            elif DataFields.DATA_URI not in output_fields:
                output_fields.append(DataFields.DATA_URI)

            image_bytes = await self._storage.retrieve_object(data_uri)
            vector = await self._extractor.extract_feature_vector(collection_name, filename, image_bytes, content_type)

            results = await self.vector_search(
                collection_name=collection_name,
                vector=vector,
                vector_field=vector_field,
                limit=limit,
                output_fields=output_fields,
                metric_type=metric_type,
                filter_expression=filter_expression,
                radius=radius,
                range_filter=range_filter
            )

            for r in results:
                r["entity"][DataFields.DATA_URI] = await self._storage.get_presigned_url(r["entity"][DataFields.DATA_URI])

            logger.info(f"Extracted embedding and did vector search of data {data_uri} in {collection_name} collection")
            return results
        except Exception as e:
            logger.exception(f"Failed to extract embedding and vector search in collection {collection_name}: {str(e)}")
            raise

    async def close(self):
        try:
            await self._client.close()
            logger.info("Closed Milvus client")
        except Exception as e:
            logger.exception(f"Failed to close Milvus client: {str(e)}")
            raise
