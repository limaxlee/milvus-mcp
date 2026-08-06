import logging
from contextlib import AsyncExitStack
from aiobotocore.session import get_session
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from common.config import SETTINGS
from common.constants import PRESIGNED_URL_EXPIRED_IN

logger = logging.getLogger(__name__)


class ObjectStorage:
    def __init__(self):
        self._client = None
        self._session = get_session()
        self._exit_stack = AsyncExitStack()
        self._config = {
            "bucket": SETTINGS.object_storage.bucket,
            "endpoint": SETTINGS.object_storage.endpoint,
            "access_key": SETTINGS.object_storage.access_key,
            "secret_key": SETTINGS.object_storage.secret_key
        }
        self._bucket = self._config["bucket"]

    async def connect(self):
        if self._client:
            return self

        try:
            self._client = await self._exit_stack.enter_async_context(self._session.create_client(
                "s3",
                endpoint_url=self._config["endpoint"],
                aws_access_key_id=self._config["access_key"],
                aws_secret_access_key=self._config["secret_key"],
                config=Config(signature_version='s3v4')
            ))
            logger.info("Connected to object storage successfully")

            return self
        except (BotoCoreError, ClientError) as e:
            logger.exception(f"Failed to connect to object storage: {str(e)}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error occurred during object storage connection: {str(e)}")
            raise

    async def close(self):
        if not self._client:
            return

        try:
            await self._exit_stack.aclose()
            logger.info("Closed object storage client")
        finally:
            self._client = None

    async def retrieve_object(self, key: str, bucket: str = None) -> bytes | None:
        bucket = bucket if bucket else self._bucket
        try:
            response = await self._client.get_object(Bucket=bucket, Key=key)
            async with response["Body"] as stream:
                data = await stream.read()

            logger.info(f"Data retrieved from {bucket}:{key}")
            return data
        except Exception as e:
            logger.exception(f"Failed to retrieve object {key} from bucket {bucket}: {e}")
            return None

    async def get_presigned_url(self, key: str, bucket: str = None, expires_in: int = PRESIGNED_URL_EXPIRED_IN) -> str:
        bucket = bucket if bucket else self._bucket
        try:
            response = await self._client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expires_in
            )

            logger.info(f"Generated presigned url for {key} from {bucket} which expires in {expires_in}")
            return response
        except Exception as e:
            logger.exception(f"Failed to generate presigned url for {key} from bucket {bucket}: {e}")
            return ""
