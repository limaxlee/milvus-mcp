import os
import yaml
import argparse
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from common.constants import ROOT_DIR

_ENV_MAP = {
    "server_port": ("SERVER_PORT", int),
    "milvus_host": ("MILVUS_HOST", str),
    "milvus_port": ("MILVUS_PORT", int),
    "milvus_user": ("MILVUS_USER", str),
    "milvus_password": ("MILVUS_PASSWORD", str),
    "milvus_db_name": ("MILVUS_DB_NAME", str),
    "OBJECT_STORAGE_BUCKET": ("object_storage.bucket", str),
    "OBJECT_STORAGE_ENDPOINT": ("object_storage.endpoint", str),
    "OBJECT_STORAGE_ACCESS_KEY": ("object_storage.access_key", str),
    "OBJECT_STORAGE_SECRET_KEY": ("object_storage.secret_key", str),
    "FEATURE_EXTRACTOR_HOST": ("feature_extractor.host", str),
    "FEATURE_EXTRACTOR_PORT": ("feature_extractor.port", str)
}

_REQUIRED_KEYS = (
    "server_port", "milvus_host", "milvus_port",
    "object_storage.bucket", "object_storage.endpoint",
    "object_storage.access_key", "object_storage.secret_key",
    "feature_extractor.host", "feature_extractor.port"
)


def load_config() -> dict:
    default_config = os.path.join(ROOT_DIR, "config.yaml")
    parser = argparse.ArgumentParser(description="Milvus MCP Server Configurations")
    parser.add_argument("--config", "-c", type=str, help="Path to the config.yaml", default=default_config)
    args, _ = parser.parse_known_args()

    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            config.update(yaml.safe_load(f))

    for env_name, (key, caster) in _ENV_MAP.items():
        env_value = os.getenv(env_name)
        if env_value is None:
            continue

        if "." in key:
            group, sub_key = key.split(".")
            config.setdefault(group, {})[sub_key] = caster(env_value)
        if env_value is not None:
            config[key] = caster(env_value)

    missing_config = []
    for key in _REQUIRED_KEYS:
        if "." in key:
            group, sub_key = key.split(".")
            value = config.get(group, {}).get(sub_key)
        else:
            value = config.get(key)

        if value in (None, ""):
            missing_config.append(key)

    if missing_config:
        raise ValueError(f"Missing required configuration: {missing_config}")

    return config


class ObjectStorageConfig(BaseModel):
    bucket: str
    endpoint: str
    access_key: str
    secret_key: str


class FeatureExtractorConfig(BaseModel):
    host: str
    port: int


class Settings(BaseSettings, extra="allow"):
    server_port: int
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_user: str = ""
    milvus_password: str = ""
    milvus_db_name: str = "default"

    object_storage: ObjectStorageConfig
    feature_extractor: FeatureExtractorConfig


SETTINGS = Settings(**load_config())
