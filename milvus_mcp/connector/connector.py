import logging
from typing import Any
from collections import Counter
from pymilvus import AsyncMilvusClient, DataType

from common.config import SETTINGS
from common.constants import (
    DIMENSION, METRIC_TYPE, LIMIT, BATCH_SIZE, DETECTION_COLLECTION_SUFFIX,
    DataFields, SamplingMethods
)
from milvus_mcp.extractor import FeatureExtractor
from milvus_mcp.storage import ObjectStorage
from milvus_mcp.sampling import sample_data
from milvus_mcp.schemas import SamplingResult, VectorSearchResult

logger = logging.getLogger(__name__)


class MilvusConnector:
    def __init__(self, storage: ObjectStorage):
        self._uri = f"http://{SETTINGS.milvus.host}:{SETTINGS.milvus.port}"
        self._token = f"{SETTINGS.milvus.user}:{SETTINGS.milvus.password}"
        self._client = AsyncMilvusClient(uri=self._uri, token=self._token, db_name=SETTINGS.milvus.db_name)
        self._storage = storage
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
            data_rows = await self._scan_data(collection_name, [])
            counter = Counter(row.get(DataFields.PREDICTION) for row in data_rows)

            result["classes"] = {str(value): count for value, count in counter.most_common()}
            logger.info(f"Retrieved {collection_name} collection info: {result}")

            return result
        except Exception as e:
            logger.exception(f"Failed to get {collection_name} collection info: {str(e)}")
            raise

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
    ) -> VectorSearchResult:
        try:
            search_params = {"metric_type": metric_type, "params": {"nprobe": 10}}
            if radius is not None:
                search_params["params"]["radius"] = radius
            if range_filter is not None:
                search_params["params"]["range_filter"] = range_filter

            search_results = await self._client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field=vector_field,
                search_params=search_params,
                limit=limit,
                output_fields=output_fields,
                filter=filter_expression
            )

            return VectorSearchResult(
                collection_name=collection_name,
                search_data_count=len(search_results[0]),
                search_data_keys=[item["id"] for item in search_results[0]],
                search_data_distance=[item["distance"] for item in search_results[0]],
                search_data={item["id"]: item["entity"] for item in search_results[0]}
            )
        except Exception as e:
            logger.exception(f"Failed to vector search in {collection_name} collection using {vector[:3]}: {str(e)}")
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
    ) -> VectorSearchResult:
        try:
            output_fields = list(
                set((output_fields or []) + [DataFields.DATA_URI, DataFields.FILENAME, DataFields.PREDICTION])
            )

            image_bytes = await self._storage.retrieve_object(data_uri)
            if not image_bytes:
                raise ValueError(f"Failed to retrieve {data_uri} object from storage")

            vector = await self._extractor.extract_feature_vector(collection_name, filename, image_bytes, content_type)
            if not vector:
                raise ValueError(f"Failed to extract feature vector from {data_uri} object")

            result = await self.vector_search(
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

            for key, item in result.search_data.items():
                item[DataFields.DATA_URI] = await self._storage.get_presigned_url(item[DataFields.DATA_URI])

            logger.info(f"Extracted embedding and did vector search of data {data_uri} in {collection_name} collection")
            return result
        except Exception as e:
            logger.exception(f"Failed to extract embedding and vector search in collection {collection_name}: {str(e)}")
            raise

    async def _scan_data(
            self,
            collection_name: str,
            output_fields: list[str],
            vector_field: str | None = None,
            batch_size: int = BATCH_SIZE
    ) -> list[dict[str, Any]]:
        fields = [DataFields.PK, DataFields.PREDICTION, *output_fields]
        if vector_field:
            fields.append(vector_field)
        fields = list(dict.fromkeys(fields))

        data_rows = []
        seen_data = set()
        cursor = None

        while True:
            filter_expression = "" if cursor is None else f"{DataFields.PK} > {cursor}"
            batch = await self._client.query(
                collection_name=collection_name,
                filter=filter_expression,
                output_fields=fields,
                limit=batch_size
            )
            if not batch:
                break

            data_count = 0
            batch.sort(key=lambda r: r[DataFields.PK])
            for row in batch:
                pk = row[DataFields.PK]
                if pk in seen_data:
                    continue

                seen_data.add(pk)
                data_rows.append(row)
                data_count += 1

            cursor = batch[-1][DataFields.PK]
            if len(batch) < batch_size or data_count == 0:
                break

        return data_rows

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

    async def get_k_center_sampled_data(
            self,
            collection_name: str,
            sample_classes: dict[str, int],
            vector_field: str = DataFields.VECTOR,
            metric_type: str = METRIC_TYPE,
            keep_classes: list[str] | None = None,
            output_fields: list[str] | None = None,
            batch_size: int = BATCH_SIZE
    ) -> SamplingResult:
        try:
            if not await self._client.has_collection(collection_name):
                raise ValueError(f"Collection {collection_name} doesn't exist")
            if await self._client.has_collection(f"{collection_name}{DETECTION_COLLECTION_SUFFIX}"):
                collection_name += DETECTION_COLLECTION_SUFFIX

            output_fields = list(set((output_fields or []) + [DataFields.DATA_URI, DataFields.FILENAME]))
            data_rows = await self._scan_data(collection_name, output_fields, vector_field, batch_size)
            if not data_rows:
                return SamplingResult(collection_name=collection_name)

            result = sample_data(
                method=SamplingMethods.K_CENTER_GREEDY,
                collection_name=collection_name,
                data_rows=data_rows,
                sample_classes=sample_classes,
                keep_classes=keep_classes,
                vector_field=vector_field,
                metric_type=metric_type
            )

            logger.info(f"K-center greedy sampled {len(result.sampled_data_keys)} data in {collection_name} collection")
            return result
        except Exception as e:
            logger.exception(f"Failed to k-center greedy sample  data in {collection_name} collection: {str(e)}")
            raise

    async def close(self):
        try:
            await self._client.close()
            logger.info("Closed Milvus client")
        except Exception as e:
            logger.exception(f"Failed to close Milvus client: {str(e)}")
            raise
