import os
import asyncio
import logging
from typing import Any
from zipfile import ZipFile, ZIP_STORED

from common.constants import ZIP_FETCH_CONCURRENCY, DataFields
from milvus_mcp.storage import ObjectStorage

logger = logging.getLogger(__name__)


async def _retrieve_object(
        item: dict[str, Any],
        prefix: int,
        storage: ObjectStorage,
        queue: asyncio.Queue,
        semaphore: asyncio.Semaphore
) -> None:
    async with semaphore:
        data = await storage.retrieve_object(item[DataFields.DATA_URI])
        if data is None:
            raise ValueError(f"Failed to retrieve {item[DataFields.DATA_URI]} object from object storage")

        await queue.put((f"{prefix}_{item[DataFields.FILENAME]}", data))


async def _write_zip_file(zip_file: str, queue: asyncio.Queue, object_count: int) -> None:
    with ZipFile(zip_file, "w", ZIP_STORED) as archive:
        for _ in range(object_count):
            filename, data = await queue.get()
            archive.writestr(filename, data)


async def create_sample_zip_file(
        zip_file: str,
        sampled_data: dict[int, dict[str, Any]],
        storage: ObjectStorage,
        concurrency: int = ZIP_FETCH_CONCURRENCY
) -> None:
    queue = asyncio.Queue(maxsize=concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    try:
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(_write_zip_file(zip_file, queue, len(sampled_data)))
            for i, item in enumerate(sampled_data.values(), start=1):
                task_group.create_task(_retrieve_object(item, i, storage, queue, semaphore))

        logger.info(f"Packaged {len(sampled_data)} objects into {zip_file}")
    except Exception as e:
        logger.error(f"Failed to create zip file {zip_file}: {e}")
        raise


def remove_sample_zip_file(zip_file: str):
    if os.path.exists(zip_file):
        os.remove(zip_file)
