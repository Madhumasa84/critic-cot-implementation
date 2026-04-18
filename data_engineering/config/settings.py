"""
Central configuration for the Critic-CoT data-engineering pipeline.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
EXPORT_DIR = DATA_DIR / "exports"
REPORT_DIR = DATA_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

for directory in (DATA_DIR, CACHE_DIR, EXPORT_DIR, REPORT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


DATABASE_PATH = DATA_DIR / "reasoning_traces.db"
DAILY_LOG_PATH = LOG_DIR / "daily_results.csv"

PROJECT_CONFIG_PATH = PROJECT_ROOT / "config.py"
DOTENV_PATHS = (PROJECT_ROOT / ".env", BASE_DIR / ".env")


def _load_python_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    spec = importlib.util.spec_from_file_location("critic_cot_project_config", path)
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return {
        "OPENROUTER_API_KEY": getattr(module, "OPENROUTER_API_KEY", ""),
        "MODEL": getattr(module, "MODEL", ""),
        "OPENROUTER_BASE_URL": getattr(module, "OPENROUTER_BASE_URL", ""),
        "OPENROUTER_REFERER": getattr(module, "OPENROUTER_REFERER", ""),
        "OPENROUTER_TITLE": getattr(module, "OPENROUTER_TITLE", ""),
    }


def _load_dotenv(paths: tuple[Path, ...]) -> Dict[str, str]:
    values: Dict[str, str] = {}

    for path in paths:
        if not path.exists():
            continue

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("'\"")

    return values


def _first_non_empty(*values: Any, default: Any = "") -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return default


def _bool_from_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


PROJECT_CONFIG = _load_python_config(PROJECT_CONFIG_PATH)
DOTENV_CONFIG = _load_dotenv(DOTENV_PATHS)

OPENROUTER_API_KEY = _first_non_empty(
    os.getenv("OPENROUTER_API_KEY"),
    PROJECT_CONFIG.get("OPENROUTER_API_KEY"),
    DOTENV_CONFIG.get("OPENROUTER_API_KEY"),
    default="",
)

MODEL = _first_non_empty(
    os.getenv("MODEL"),
    PROJECT_CONFIG.get("MODEL"),
    DOTENV_CONFIG.get("MODEL"),
    default="openrouter/free",
)

OPENROUTER_BASE_URL = _first_non_empty(
    os.getenv("OPENROUTER_BASE_URL"),
    PROJECT_CONFIG.get("OPENROUTER_BASE_URL"),
    DOTENV_CONFIG.get("OPENROUTER_BASE_URL"),
    default="https://openrouter.ai/api/v1/chat/completions",
)

OPENROUTER_REFERER = _first_non_empty(
    os.getenv("OPENROUTER_REFERER"),
    PROJECT_CONFIG.get("OPENROUTER_REFERER"),
    DOTENV_CONFIG.get("OPENROUTER_REFERER"),
    default="https://localhost",
)

OPENROUTER_TITLE = _first_non_empty(
    os.getenv("OPENROUTER_TITLE"),
    PROJECT_CONFIG.get("OPENROUTER_TITLE"),
    DOTENV_CONFIG.get("OPENROUTER_TITLE"),
    default="Critic-CoT Data Pipeline",
)

REQUEST_TIMEOUT_SECONDS = int(
    _first_non_empty(os.getenv("REQUEST_TIMEOUT_SECONDS"), default="90")
)
DEFAULT_TEMPERATURE = float(
    _first_non_empty(os.getenv("DEFAULT_TEMPERATURE"), default="0.7")
)
DEFAULT_MAX_TOKENS = int(
    _first_non_empty(os.getenv("DEFAULT_MAX_TOKENS"), default="2000")
)


STRATEGIES = ("baseline", "iter_refine", "filter", "majority")
DEFAULT_MAX_ITERATIONS = 3
DEFAULT_FILTER_SAMPLES = 3
DEFAULT_MAJORITY_SAMPLES = 5
DEFAULT_EVAL_SAMPLES = 50
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_SAMPLE_SEED = 42

DEFAULT_SCHEDULER_DAILY_TIME = "09:00"
CONTINUOUS_SLEEP_SECONDS = 30
ENABLE_VERBOSE_LOGGING = _bool_from_env("CRITIC_COT_VERBOSE", default=True)


COST_PER_TOKEN = {
    "meta-llama/llama-3-70b-instruct": 0.0000010,
    "meta-llama/llama-3-8b-instruct": 0.0000002,
    "openai/gpt-4o-mini": 0.0000006,
    "openrouter/free": 0.0,
    "default": 0.0000010,
}


def resolve_cost_per_token(model_name: str) -> float:
    return COST_PER_TOKEN.get(model_name, COST_PER_TOKEN["default"])


def has_configured_api_key() -> bool:
    key = (OPENROUTER_API_KEY or "").strip()
    if not key:
        return False

    placeholders = {
        "your-openrouter-api-key",
        "your-api-key-here",
        "your_api_key",
        "your_api_key_here",
        "YOUR_API_KEY",
    }
    return key not in placeholders
