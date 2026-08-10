import asyncio
import logging
from typing import Any, TYPE_CHECKING
from zipfile import ZipFile, ZIP_STORED

from common.constants import ZIP_FETCH_CONCURRENCY, DataFields

if TYPE_CHECKING:
    from milvus_mcp.storage import ObjectStorage

logger = logging.getLogger(__name__)


async def _retrieve_object(
        item: dict[str, Any],
        storage: "ObjectStorage",
        queue: asyncio.Queue,
        semaphore: asyncio.Semaphore
) -> None:
    async with semaphore:
        data = await storage.retrieve_object(item[DataFields.DATA_URI])
        if data is None:
            raise ValueError(f"Failed to retrieve {item[DataFields.DATA_URI]} object from storage")

        await queue.put((item[DataFields.FILENAME], data))


async def _write_zip_file(zip_file: str, queue: asyncio.Queue, object_count: int) -> None:
    loop = asyncio.get_running_loop()
    with ZipFile(zip_file, "w", ZIP_STORED) as archive:
        for _ in range(object_count):
            filename, data = await queue.get()
            await loop.run_in_executor(None, archive.writestr, filename, data)


async def create_zip_file(
        zip_file: str,
        sampled_data: dict[int, dict[str, Any]],
        storage: "ObjectStorage",
        concurrency: int = ZIP_FETCH_CONCURRENCY
) -> None:
    """
    Retrieve sampled data objects from storage and write them into a zip file under their own filename

    Args:
        zip_file: Path of the zip file to create
        sampled_data: Sampled data rows keyed by primary key, each holding a filename and a data uri
        concurrency: Number of objects retrieved from storage at a time
    """
    queue = asyncio.Queue(maxsize=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    try:
        # ZipFile is not thread safe, so a single writer drains the queue and offloads each write to a worker thread
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(_write_zip_file(zip_file, queue, len(sampled_data)))
            for item in sampled_data.values():
                task_group.create_task(_retrieve_object(item, storage, queue, semaphore))
    except ExceptionGroup as eg:
        # The task group cancels the remaining tasks, so surface the failure that caused it instead of the group
        raise eg.exceptions[0] from eg

    logger.info(f"Packaged {len(sampled_data)} objects into {zip_file}")
