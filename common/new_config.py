"""Configuration for the Milvus MCP server.

Settings are assembled from three sources, lowest precedence first:

1. model defaults declared on the config classes below
2. a YAML file -- ``--config/-c`` on our own command line, else ``$MILVUS_MCP_CONFIG``,
   else ``config.yaml`` next to the project root
3. individual ``$SERVER_PORT`` / ``$MILVUS_*`` / ``$OBJECT_STORAGE_*`` / ``$FEATURE_EXTRACTOR_*``
   overrides, as listed in ``_ENV_MAP``

``SETTINGS`` is built at import time, which is before ``initialize_logger`` runs, so the
log records emitted here only surface if logging was configured earlier in the process.
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Any, Callable
from pydantic import BaseModel, ConfigDict

from common.constants import ROOT_DIR

logger = logging.getLogger(__name__)

SERVER_PACKAGE = "milvus_mcp"
CONFIG_PATH_ENV = "MILVUS_MCP_CONFIG"
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"

# env var name -> (dotted path into the config mapping, how to parse the string)
_ENV_MAP: dict[str, tuple[str, Callable[[str], Any]]] = {
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


class _StrictModel(BaseModel):
    """Base for every config section: reject unknown keys instead of ignoring them.

    Without this, a typo such as ``milvus_hostt:`` is silently accepted and the real
    ``milvus.host`` quietly falls back to its default.
    """

    model_config = ConfigDict(extra="forbid")


class ObjectStorageConfig(_StrictModel):
    bucket: str
    endpoint: str
    access_key: str
    secret_key: str


class FeatureExtractorConfig(_StrictModel):
    host: str
    port: int


class MilvusDBConfig(_StrictModel):
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    db_name: str = "default"


class Settings(_StrictModel):
    server_port: int
    milvus: MilvusDBConfig
    object_storage: ObjectStorageConfig
    feature_extractor: FeatureExtractorConfig


def _running_as_server() -> bool:
    """Whether this process was started as the Milvus MCP server itself.

    Resolving the config path involves reading ``sys.argv``, but every program that
    imports this module brings its own command line, and ``-c`` is a popular flag
    elsewhere -- pytest uses it for its ini file. Parsing argv unconditionally lets a
    foreign ``-c`` silently become our config path, so gate on who ``__main__`` is.
    """
    main = sys.modules.get("__main__")
    if main is None:
        return False

    # python -m milvus_mcp  ->  __main__.__spec__.name == "milvus_mcp.__main__"
    spec_name = getattr(getattr(main, "__spec__", None), "name", "") or ""
    if spec_name.split(".")[0] == SERVER_PACKAGE:
        return True

    # python milvus_mcp/__main__.py  ->  no __spec__, but __file__ is one of ours
    main_file = getattr(main, "__file__", None)
    if not main_file:
        return False

    try:
        return Path(main_file).resolve().is_relative_to(ROOT_DIR.resolve() / SERVER_PACKAGE)
    except (OSError, ValueError):
        return False


def _config_path_from_argv() -> str | None:
    parser = argparse.ArgumentParser(description="Milvus MCP Server Configurations")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to the config.yaml")
    args, _ = parser.parse_known_args()

    return args.config


def _resolve_config_path() -> tuple[Path, bool]:
    """Return the config file to read and whether it was asked for explicitly.

    An explicitly requested file must exist; the bundled default may be absent so the
    server can run from environment variables alone.
    """
    if _running_as_server():
        cli_path = _config_path_from_argv()
        if cli_path:
            return Path(cli_path), True

    env_path = os.getenv(CONFIG_PATH_ENV)
    if env_path:
        return Path(env_path), True

    return DEFAULT_CONFIG_PATH, False


def _read_config_file(path: Path, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.warning("No config file at %s, falling back to environment and defaults", path)
        return {}

    with open(path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f) or {}

    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a mapping, got {type(content).__name__}: {path}")

    logger.info("Loaded config from %s", path)
    return content


def _set_nested_config(config: dict[str, Any], key: str, value: Any) -> None:
    *parents, leaf = key.split(".")
    node, walked = config, []
    for parent in parents:
        walked.append(parent)
        node = node.setdefault(parent, {})
        if not isinstance(node, dict):
            raise ValueError(f"Config key {'.'.join(walked)!r} must be a mapping, got {type(node).__name__}")

    node[leaf] = value


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    for env_name, (key, caster) in _ENV_MAP.items():
        env_value = os.getenv(env_name)
        if env_value is None:
            continue

        try:
            value = caster(env_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid value for {env_name}: {env_value!r}") from error

        _set_nested_config(config, key, value)
        logger.info("Config %r overridden by $%s", key, env_name)  # name only, values may be secrets

    return config


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Raw config mapping: the YAML file overlaid with environment overrides.

    Passing ``path`` skips command line and environment lookup entirely, which is how
    tests build a ``Settings`` without touching the process-wide state.
    """
    if path is not None:
        resolved, required = Path(path), True
    else:
        resolved, required = _resolve_config_path()

    return _apply_env_overrides(_read_config_file(resolved, required))


SETTINGS = Settings(**load_config())
