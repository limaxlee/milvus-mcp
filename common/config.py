import os
import yaml
import argparse
from typing import Any
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from common.constants import ROOT_DIR

_ENV_MAP = {
    "SERVER_PORT": ("server_port", int),
    "MILVUS_HOST": ("milvus.host", str),
    "MILVUS_PORT": ("milvus.port", int),
    "MILVUS_USER": ("milvus.user", str),
    "MILVUS_PASSWORD": ("milvus.password", str),
    "MILVUS_DB_NAME": ("milvus.db_name", str),
    "FEATURE_EXTRACTOR_HOST": ("feature_extractor.host", str),
    "FEATURE_EXTRACTOR_PORT": ("feature_extractor.port", int),
    "OBJECT_STORAGE_BUCKET": ("object_storage.bucket", str),
    "OBJECT_STORAGE_ENDPOINT": ("object_storage.endpoint", str),
    "OBJECT_STORAGE_ACCESS_KEY": ("object_storage.access_key", str),
    "OBJECT_STORAGE_SECRET_KEY": ("object_storage.secret_key", str)
}


def _set_nested_config(config: dict[str, Any], key: str, value: Any) -> None:
    *groups, values = key.split(".")
    node = config
    for group in groups:
        node = node.setdefault(group, {})
        if not isinstance(node, dict):
            raise ValueError(f"Config key {group} is invalid: {type(node).__name__}")

    node[values] = value


def load_config() -> dict[str, Any]:
    default_config = os.path.join(ROOT_DIR, "config.yaml")
    parser = argparse.ArgumentParser(description="Milvus MCP Server Configurations")
    parser.add_argument("--config", "-c", type=str, help="Path to the config.yaml", default=default_config)
    args, _ = parser.parse_known_args()

    config = {}
    if os.path.isfile(args.config):
        with open(args.config, "r") as f:
            file_config = yaml.safe_load(f) or {}

        if not isinstance(file_config, dict):
            raise ValueError(f"Config file is invalid: {args.config}")

        config.update(file_config)

    for env_name, (key, caster) in _ENV_MAP.items():
        env_value = os.getenv(env_name)
        if env_value is None:
            continue

        try:
            value = caster(env_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid value for {env_name}: {env_value!r}") from error

        _set_nested_config(config, key, value)

    return config


class ObjectStorageConfig(BaseModel):
    bucket: str
    endpoint: str
    access_key: str
    secret_key: str


class FeatureExtractorConfig(BaseModel):
    host: str
    port: int


class MilvusDBConfig(BaseModel):
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    db_name: str = "default"


class Settings(BaseSettings, extra="allow"):
    server_port: int
    milvus: MilvusDBConfig
    object_storage: ObjectStorageConfig
    feature_extractor: FeatureExtractorConfig


SETTINGS = Settings(**load_config())
