import pytest
import asyncio
import zipfile

from common.constants import DataFields
from milvus_mcp.utils.zip import create_sample_zip_file


def mock_sampled_item(pk, filename=None):
    return {DataFields.DATA_URI: f"uri_{pk}", DataFields.FILENAME: filename or f"file_{pk}.jpg"}


class MockObjectStorage:
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


class TestSampleCreateZipFile:
    @pytest.mark.asyncio
    async def test_create_zip_file(self, tmp_path):
        storage = MockObjectStorage({"uri_1": b"first", "uri_2": b"second"})
        zip_file = str(tmp_path / "sampled.zip")

        await create_sample_zip_file(zip_file, {1: mock_sampled_item(1), 2: mock_sampled_item(2)}, storage)

        with zipfile.ZipFile(zip_file) as archive:
            assert sorted(archive.namelist()) == ["1_file_1.jpg", "2_file_2.jpg"]
            assert archive.read("1_file_1.jpg") == b"first"
            assert archive.read("2_file_2.jpg") == b"second"
            assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())

        zip_file = str(tmp_path / "empty.zip")
        await create_sample_zip_file(zip_file, {}, MockObjectStorage({}))

        with zipfile.ZipFile(zip_file) as archive:
            assert archive.namelist() == []

        sampled_data = {pk: mock_sampled_item(pk) for pk in range(20)}
        storage = MockObjectStorage({item[DataFields.DATA_URI]: b"data" for item in sampled_data.values()}, delay=0.01)

        await create_sample_zip_file(str(tmp_path / "sampled.zip"), sampled_data, storage, concurrency=4)
        assert storage.max_in_flight == 4
