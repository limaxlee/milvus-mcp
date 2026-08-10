import json
import logging
from typing import Any
from fastmcp.exceptions import ToolError
from fastmcp.server import Context, FastMCP

from common.constants import DIMENSION, METRIC_TYPE, LIMIT, BATCH_SIZE
from milvus_mcp.schemas import SamplingResult, VectorSearchResult

logger = logging.getLogger(__name__)

tools_mcp = FastMCP(name="tools")


@tools_mcp.tool()
async def milvus_list_databases(ctx: Context) -> str:
    """
    List all databases in the Milvus

    Returns:
        Comma separated names of every database in the Milvus instance
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        databases = await connector.list_databases()
        return f"Databases in Milvus:\n{', '.join(databases)}"
    except Exception as e:
        raise ToolError(f"Failed to list databases: {str(e)}")


@tools_mcp.tool()
async def milvus_list_collections(ctx: Context) -> str:
    """
    List all collections in the database

    Returns:
        Comma separated names of every collection in the configured database
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        collections = await connector.list_collections()
        return f"Collections in database:\n{', '.join(collections)}"
    except Exception as e:
        raise ToolError(f"Failed to list collections: {str(e)}")


@tools_mcp.tool()
async def milvus_get_collection_info(collection_name: str, ctx: Context) -> str:
    """
    Lists detailed information about a specific collection

    Args:
        collection_name: Name of collection

    Returns:
        JSON formatted collection description covering its schema, fields and settings, extended with a classes
        entry holding the number of data per prediction class
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        collection_info = await connector.get_collection_info(collection_name)
        info_str = json.dumps(collection_info, indent=2, default=list)
        return f"Collection information:\n{info_str}"
    except Exception as e:
        raise ToolError(f"Failed to get {collection_name} collection info: {str(e)}")


@tools_mcp.tool()
async def milvus_get_collection_stats(collection_name: str, ctx: Context) -> dict[str, Any]:
    """
    Lists the statistics collected on a specific collection

    Args:
        collection_name: Name of collection

    Returns:
        Collection statistics keyed by metric name, such as row_count
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        collection_stats = await connector.get_collection_stats(collection_name)
        return collection_stats
    except Exception as e:
        raise ToolError(f"Failed to get {collection_name} collection statistics: {str(e)}")


@tools_mcp.tool()
async def milvus_create_collection(
    ctx: Context,
    collection_name: str,
    auto_id: bool = True,
    dimension: int = DIMENSION,
    primary_field_name: str = "id",
    vector_field_name: str = "vector",
    metric_type: str = METRIC_TYPE,
    field_schema: list[dict[str, Any]] | None = None,
    index_params: list[dict[str, Any]] | None = None,
    other_kwargs: dict[str, Any] | None = None
) -> str:
    """
    Create a new collection with specified schema.

    Args:
        collection_name: Name for the new collection
        auto_id: Whether to auto generate id, default to True
        dimension: Vector dimension, default to 768
        primary_field_name: Name of the primary field, default to "id"
        vector_field_name: Name of the vector field, default to "vector"
        metric_type: Metric type, default to "COSINE"
        field_schema: List of field schema, each element is a dictionary with the following keys:
            - name: Name of the field
            - type: Type of the field
        index_params: Optional list of index parameters, each element is a dictionary with the following keys:
            - field_name: Name of the field to index
            - index_type: Index type
            - **kwargs: Other optional index parameters
        other_kwargs: Additional keyword arguments for the collection creation

    Returns:
        Confirmation message naming the created collection
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        await connector.create_collection(
            collection_name=collection_name,
            auto_id=auto_id,
            dimension=dimension,
            primary_field_name=primary_field_name,
            vector_field_name=vector_field_name,
            metric_type=metric_type,
            field_schema=field_schema,
            index_params=index_params,
            **(other_kwargs if other_kwargs is not None else {})
        )

        return f"Collection {collection_name} created successfully"
    except Exception as e:
        raise ToolError(f"Failed to create {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def milvus_load_collection(ctx: Context, collection_name: str, replica_number: int = 1) -> str:
    """
    Load a collection into memory for search and query.

    Args:
        collection_name: Name of collection to load
        replica_number: Number of replicas

    Returns:
        Confirmation message naming the loaded collection and its replica count
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        await connector.load_collection(collection_name=collection_name, replica_number=replica_number)

        return f"Collection {collection_name} loaded successfully with {replica_number} replica(s)"
    except Exception as e:
        raise ToolError(f"Failed to create {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def milvus_release_collection(collection_name: str, ctx: Context) -> str:
    """
    Release a collection from memory.

    Args:
        collection_name: Name of collection to release

    Returns:
        Confirmation message naming the released collection
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        await connector.release_collection(collection_name=collection_name)

        return f"Collection {collection_name} released successfully"
    except Exception as e:
        raise ToolError(f"Failed to release {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def milvus_insert_data(
        collection_name: str,
        data: list[dict[str, Any]],
        ctx: Context
) -> str:
    """
    Insert data into a collection.

    Args:
        collection_name: Name of collection
        data: List of dictionaries, each representing a record

    Returns:
        Confirmation message carrying the insert result, which holds the number of inserted data and their primary
        keys
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.insert_data(collection_name, data)

        return f"Data inserted into collection {collection_name} with result: {str(result)}"
    except Exception as e:
        raise ToolError(f"Failed to insert data to {collection_name} collection: {str(e)}")


@tools_mcp.tool()
async def milvus_delete_entities(
        collection_name: str,
        filter_expression: str,
        ctx: Context
) -> str:
    """
    Delete entities from a collection based on filter expression.

    Args:
        collection_name: Name of collection
        filter_expression: Filter expression to select entities to delete

    Returns:
        Confirmation message carrying the delete result, which holds the number of deleted entities
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        result = await connector.delete_entities(collection_name, filter_expression)

        return f"Entities deleted from collection {collection_name} with result: {str(result)}"
    except Exception as e:
        raise ToolError(f"Failed to delete data from {collection_name} using filter {filter_expression}: {str(e)}")


@tools_mcp.tool()
async def milvus_vector_search(
        ctx: Context,
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
    """
    Perform vector similarity search on a collection.

    Args:
        collection_name: Name of the collection to search
        vector: Query vector
        vector_field: Field containing vectors to search
        limit: Maximum number of results
        output_fields: Fields to include in results
        metric_type: Distance metric (COSINE, L2, IP)
        filter_expression: Optional filter expression
        radius: Optional lower bound for range search
        range_filter: Optional upper bound for range search

    Returns:
        Search result with the following fields:
            - collection_name: Name of the searched collection
            - search_data_count: Number of matches found
            - search_data_keys: Primary keys of the matches, ordered from nearest to furthest
            - search_data_distance: Distance of each match to the query vector, in the same order as search_data_keys
            - search_data: Requested output fields of each match, keyed by primary key
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.vector_search(
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
    except Exception as e:
        raise ToolError(f"Failed to vector search in {collection_name} collection using {vector[:3]}: {str(e)}")


@tools_mcp.tool()
async def milvus_extract_embeddings_and_vector_search(
        ctx: Context,
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
    """
    Extract the embeddings of the image and perform vector similarity search on a collection.

    Args:
        collection_name: Name of the collection to search
        vector_field: Field containing vectors to search
        filename: Name of the image file
        data_uri: Key of the image file within object storage bucket
        content_type: MIME type of the image (e.g. "image/jpeg", "image/png")
        limit: Maximum number of results
        output_fields: Fields to include in results
        metric_type: Distance metric (COSINE, L2, IP)
        filter_expression: Optional filter expression
        radius: Optional lower bound for range search
        range_filter: Optional upper bound for range search

    Returns:
        Search result with the following fields:
            - collection_name: Name of the searched collection
            - search_data_count: Number of matches found
            - search_data_keys: Primary keys of the matches, ordered from nearest to furthest
            - search_data_distance: Distance of each match to the query vector, in the same order as search_data_keys
            - search_data: Output fields of each match keyed by primary key, always including filename, prediction
              and data_uri, where data_uri is a presigned download url which expires in 2 hours
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.extract_embeddings_and_vector_search(
            collection_name=collection_name,
            vector_field=vector_field,
            filename=filename,
            data_uri=data_uri,
            content_type=content_type,
            limit=limit,
            output_fields=output_fields,
            metric_type=metric_type,
            filter_expression=filter_expression,
            radius=radius,
            range_filter=range_filter
        )
    except Exception as e:
        raise ToolError(
            f"Failed to extract embeddings and vector search in {collection_name} collection of {data_uri}: {str(e)}"
        )


@tools_mcp.tool()
async def milvus_query_collection(
        ctx: Context,
        collection_name: str,
        filter_expression: str,
        output_fields: list[str] | None = None,
        limit: int = LIMIT
) -> str:
    """
    Query collection using filter expressions.

    Args:
        collection_name: Name of the collection to query
        filter_expression: Filter expression
        output_fields: Fields to include in results
        limit: Maximum number of results

    Returns:
        Matching data rows one per block, or a message stating that the filter expression matched nothing
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        results = await connector.query_collection(
            collection_name=collection_name,
            filter_expression=filter_expression,
            output_fields=output_fields,
            limit=limit
        )

        if results:
            output = f"Query results for {filter_expression} in collection {collection_name}:\n\n"
            for result in results:
                output += f"{result}\n\n"
        else:
            output = f"No query result for {collection_name}"

        return output
    except Exception as e:
        raise ToolError(f"Failed to query from {collection_name} collection using {filter_expression}: {str(e)}")


@tools_mcp.tool()
async def milvus_get_k_center_sampled_data(
        ctx: Context,
        collection_name: str,
        vector_field: str,
        sample_classes: dict[str, int],
        metric_type: str = METRIC_TYPE,
        keep_classes: list[str] | None = None,
        output_fields: list[str] | None = None,
        batch_size: int = BATCH_SIZE
) -> SamplingResult:
    """
    Class-aware (stratified) sample collection using k-center greedy over its vector field

    Args:
        collection_name: Name of collection to sample
        vector_field: Field containing vectors to sample over
        sample_classes: Per-class sampling cap, e.g. {"Good": 800} — sample that class to 800
        metric_type: Distance metric (COSINE, L2, IP), default to COSINE
        keep_classes: If given, classes to retain entirely when sampling
        output_fields: If given, selected columns are returned with these fields
        batch_size: Number of data in single batch when scanning the collection data, default to 1000

    Returns:
        Sampling result with the following fields:
            - collection_name: Name of the sampled collection, suffixed with _object when the matching detection
              collection is sampled instead
            - total_scanned_data: Number of data scanned in the collection
            - total_sampled_data_size: Number of data selected by sampling
            - class_distribution: Number of sampled data per class
            - sampled_data_keys: Primary keys of the sampled data
            - sampled_data: Sampled data rows keyed by primary key, without the vector field
            - sampled_zip_uri: Always empty, only milvus_get_k_center_sampled_data_as_zip_file fills it
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.get_k_center_sampled_data(
            collection_name=collection_name,
            vector_field=vector_field,
            metric_type=metric_type,
            keep_classes=keep_classes,
            sample_classes=sample_classes,
            output_fields=output_fields,
            batch_size=batch_size
        )
    except Exception as e:
        raise ToolError(f"Failed to k-center greedy sample data in collection {collection_name}: {str(e)}")


@tools_mcp.tool()
async def milvus_get_k_center_sampled_data_as_zip_file(
        ctx: Context,
        collection_name: str,
        vector_field: str,
        sample_classes: dict[str, int],
        metric_type: str = METRIC_TYPE,
        keep_classes: list[str] | None = None,
        output_fields: list[str] | None = None,
        batch_size: int = BATCH_SIZE
) -> SamplingResult:
    """
    Class-aware (stratified) sample collection using k-center greedy over its vector field and create a zip file for
    sampled data and upload it to object storage

    Args:
        collection_name: Name of collection to sample
        vector_field: Field containing vectors to sample over
        sample_classes: Per-class sampling cap, e.g. {"Good": 800} — sample that class to 800
        metric_type: Distance metric (COSINE, L2, IP), default to COSINE
        keep_classes: If given, classes to retain entirely when sampling
        output_fields: If given, selected columns are returned with these fields
        batch_size: Number of data in single batch when scanning the collection data, default to 1000

    Returns:
        Sampling result with the following fields:
            - collection_name: Name of the sampled collection, suffixed with _object when the matching detection
              collection is sampled instead
            - total_scanned_data: Number of data scanned in the collection
            - total_sampled_data_size: Number of data selected by sampling
            - class_distribution: Number of sampled data per class
            - sampled_data_keys: Always empty, the sampled data is delivered through the zip file instead
            - sampled_data: Always empty, the sampled data is delivered through the zip file instead
            - sampled_zip_uri: Presigned download url for the zip file which expires in 2 hours, empty when sampling
              matched no data
    """
    try:
        connector = ctx.request_context.lifespan_context.connector
        return await connector.get_k_center_sampled_data_as_zip_file(
            collection_name=collection_name,
            vector_field=vector_field,
            metric_type=metric_type,
            keep_classes=keep_classes,
            sample_classes=sample_classes,
            output_fields=output_fields,
            batch_size=batch_size
        )
    except Exception as e:
        raise ToolError(f"Failed to package sampled data of collection {collection_name} as zip file: {str(e)}")
