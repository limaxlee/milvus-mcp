import asyncio
import pytest
import zipfile

from common.constants import DataFields
from milvus_mcp.utils.zip import create_zip_file


def mock_sampled_item(pk, filename=None):
    return {DataFields.DATA_URI: f"uri_{pk}", DataFields.FILENAME: filename or f"file_{pk}.jpg"}


class FakeStorage:
    def __init__(self, objects, delay=0.0):
        self._objects = objects
        self._delay = delay
        self.in_flight = 0
        self.max_in_flight = 0

    async def retrieve_object(self, key):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self._delay)
            return self._objects.get(key)
        finally:
            self.in_flight -= 1


class TestCreateZipFile:
    @pytest.mark.asyncio
    async def test_writes_every_object_under_its_filename(self, tmp_path):
        storage = FakeStorage({"uri_1": b"first", "uri_2": b"second"})
        zip_file = str(tmp_path / "sampled.zip")

        await create_zip_file(zip_file, {1: mock_sampled_item(1), 2: mock_sampled_item(2)}, storage)

        with zipfile.ZipFile(zip_file) as archive:
            assert sorted(archive.namelist()) == ["file_1.jpg", "file_2.jpg"]
            assert archive.read("file_1.jpg") == b"first"
            assert archive.read("file_2.jpg") == b"second"
            assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())

    @pytest.mark.asyncio
    async def test_writes_empty_archive_for_no_sampled_data(self, tmp_path):
        zip_file = str(tmp_path / "empty.zip")

        await create_zip_file(zip_file, {}, FakeStorage({}))

        with zipfile.ZipFile(zip_file) as archive:
            assert archive.namelist() == []

    @pytest.mark.asyncio
    async def test_raises_when_object_is_missing(self, tmp_path):
        # retrieve_object returns None instead of raising, so a missing object must not silently short the archive
        storage = FakeStorage({"uri_1": b"first"})
        zip_file = str(tmp_path / "sampled.zip")

        with pytest.raises(ValueError, match="Failed to retrieve uri_2 object from storage"):
            await create_zip_file(zip_file, {1: mock_sampled_item(1), 2: mock_sampled_item(2)}, storage)

    @pytest.mark.asyncio
    async def test_propagates_storage_failure(self, mocker, tmp_path):
        storage = mocker.MagicMock()
        storage.retrieve_object = mocker.AsyncMock(side_effect=RuntimeError("Storage error"))

        with pytest.raises(RuntimeError, match="Storage error"):
            await create_zip_file(str(tmp_path / "sampled.zip"), {1: mock_sampled_item(1)}, storage)

    @pytest.mark.asyncio
    async def test_retrieves_concurrently_up_to_the_limit(self, tmp_path):
        sampled_data = {pk: mock_sampled_item(pk) for pk in range(20)}
        storage = FakeStorage({item[DataFields.DATA_URI]: b"data" for item in sampled_data.values()}, delay=0.01)

        await create_zip_file(str(tmp_path / "sampled.zip"), sampled_data, storage, concurrency=4)

        assert storage.max_in_flight == 4
