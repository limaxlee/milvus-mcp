import logging
import numpy as np
from typing import Any
from collections import Counter
from pymilvus import AsyncMilvusClient, DataType

from common.config import SETTINGS
from common.constants import DIMENSION, METRIC_TYPE, LIMIT, BATCH_SIZE, DataFields, DETECTION_COLLECTION_SUFFIX, MetricTypes
from milvus_mcp.storage import ObjectStorageClient
from milvus_mcp.extractor import FeatureExtractor
from milvus_mcp.schemas import SamplingResult

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
        data_rows = await self._scan_data(collection_name, [], vector_field=None)
        counter = Counter(row.get(DataFields.PREDICTION) for row in data_rows)

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
                output_fields = [*output_fields, DataFields.DATA_URI]

            image_bytes = await self._storage.retrieve_object(data_uri)
            if not image_bytes:
                raise ValueError(f"Failed to retrieve {data_uri} object from storage")

            vector = await self._extractor.extract_feature_vector(collection_name, filename, image_bytes, content_type)
            if not vector:
                raise ValueError(f"Failed to extract feature vector from {data_uri} object")

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
                presigned_url = await self._storage.get_presigned_url(r["entity"][DataFields.DATA_URI])
                if not presigned_url:
                    raise ValueError(f"Failed to generate presigned URL for {r['entity'][DataFields.DATA_URI]}")

                r["entity"][DataFields.DATA_URI] = presigned_url

            logger.info(f"Extracted embedding and did vector search of data {data_uri} in {collection_name} collection")
            return results
        except Exception as e:
            logger.exception(f"Failed to extract embedding and vector search in collection {collection_name}: {str(e)}")
            raise

    async def _scan_data(
            self,
            collection_name: str,
            output_fields: list[str],
            vector_field: str | None = DataFields.VECTOR,
            batch_size: int = BATCH_SIZE
    ) -> list[dict[str, Any]]:
        requested_fields = [DataFields.PK, DataFields.PREDICTION, *output_fields]
        if vector_field is not None:
            requested_fields.insert(1, vector_field)

        fields = list(dict.fromkeys(requested_fields))
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

    async def get_stratified_sampled_data(
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

            if any(sample_cap <= 0 for sample_cap in sample_classes.values()):
                raise ValueError(f"Sample cap for classes {sample_classes} must be positive")
            if keep_classes and set(keep_classes) & sample_classes.keys():
                raise ValueError(f"Overlapping classes in keep {keep_classes} and sample {sample_classes} classes")

            output_fields = list(set((output_fields or []) + [DataFields.DATA_URI, DataFields.FILENAME]))
            data_rows = await self._scan_data(collection_name, output_fields, vector_field, batch_size)
            if not data_rows:
                return SamplingResult(collection_name=collection_name)

            pk = [item[DataFields.PK] for item in data_rows]
            classes = [item.get(DataFields.PREDICTION) for item in data_rows]
            class_counts = Counter(classes)
            vectors = np.asarray([item[vector_field] for item in data_rows], dtype=np.float32)

            if sample_classes.keys() - class_counts.keys():
                raise ValueError(f"Sample classes {sample_classes} are not present in {collection_name} collection")
            if keep_classes and set(keep_classes) - class_counts.keys():
                raise ValueError(f"Keep classes {keep_classes} are not present in {collection_name} collection")

            class_caps = {class_name: class_counts[class_name] for class_name in keep_classes} if keep_classes else {}
            for class_name, sample_cap in sample_classes.items():
                class_caps[class_name] = min(sample_cap, class_counts[class_name])

            sampled_data = self._stratified_sample(vectors, classes, class_caps, metric_type)
            sampled_data_pk = [pk[item] for item in sampled_data]
            result = SamplingResult(
                collection_name=collection_name,
                vector_field=vector_field,
                total_scanned_data=len(data_rows),
                total_sampled_data_size=sum(class_caps.values()),
                class_distribution=class_caps,
                sampled_data_keys=sampled_data_pk,
                sampled_data={item[DataFields.PK]: item for item in data_rows if item[DataFields.PK] in sampled_data_pk}
            )

            logger.info(f"Sampled {len(sampled_data)} data in {collection_name} collection")
            return result
        except Exception as e:
            logger.exception(f"Failed to sample data in {collection_name} collection: {str(e)}")
            raise

    def _stratified_sample(
            self,
            vectors: np.ndarray,
            classes: list[str],
            class_caps: dict[str, int],
            metric_type: str = METRIC_TYPE
    ) -> list[int]:
        np_classes = np.array(classes)
        sampled_data = []

        for class_name, class_cap in class_caps.items():
            class_data_idx = np.where(np_classes == class_name)[0]
            local = self._k_center_greedy_sample(vectors[class_data_idx], class_cap, metric_type)
            chosen = [int(class_data_idx[i]) for i in local]

            sampled_data.extend(chosen)

        return sampled_data

    @staticmethod
    def _k_center_greedy_sample(vectors: np.ndarray, sample_size: int, metric_type: str = METRIC_TYPE) -> list[int]:
        feature_vectors = vectors.astype(np.float32, copy=True)

        if metric_type == MetricTypes.COSINE:
            normalized_vectors = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
            normalized_vectors[normalized_vectors == 0.0] = 1.0
            feature_vectors /= normalized_vectors

        def distance_to_center(center: np.ndarray) -> np.ndarray:
            if metric_type == MetricTypes.COSINE:
                return -(feature_vectors @ center)
            diff = feature_vectors - center
            return np.einsum("ij,ij->i", diff, diff)

        start_data = int(np.argmin(distance_to_center(feature_vectors.mean(axis=0))))
        selected_data = [start_data]
        min_distance = distance_to_center(feature_vectors[start_data])
        min_distance[start_data] = -np.inf

        for _ in range(sample_size - 1):
            idx = int(np.argmax(min_distance))
            if not np.isfinite(min_distance[idx]):
                break

            selected_data.append(idx)
            min_distance = np.minimum(min_distance, distance_to_center(feature_vectors[idx]))
            min_distance[idx] = -np.inf

        return selected_data

    async def close(self):
        try:
            await self._client.close()
            logger.info("Closed Milvus client")
        except Exception as e:
            logger.exception(f"Failed to close Milvus client: {str(e)}")
            raise
