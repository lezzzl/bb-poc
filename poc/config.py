"""Настройки PoC из .env. Облачный трейсинг выключается всегда."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TRACING_OFF = {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING_V2": "false"}


@dataclass(frozen=True, slots=True)
class PocSettings:
    root: Path
    test_dir: Path
    data_dir: Path
    api_key: str = field(default="", repr=False)
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = ""
    timeout_s: float = 60.0
    max_tool_calls: int = 12
    reasoning_effort: str = ""  # "" — по умолчанию модели; low | medium | high


def load_settings(root: Path | None = None) -> PocSettings:
    root = Path(root or PROJECT_ROOT).resolve()
    os.environ.update(_TRACING_OFF)
    values = {**os.environ, **{k: v for k, v in dotenv_values(root / ".env").items() if v is not None}}
    data_dir = root / "poc_data"
    data_dir.mkdir(exist_ok=True)
    return PocSettings(
        root=root,
        test_dir=root / "TEST",
        data_dir=data_dir,
        api_key=(values.get("OPENROUTER_API_KEY") or "").strip(),
        base_url=(values.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip(),
        model=(values.get("OPENROUTER_MODEL") or "").strip(),
        timeout_s=float(values.get("POC_TIMEOUT_S") or 60),
        max_tool_calls=int(values.get("POC_MAX_TOOL_CALLS") or 12),
        reasoning_effort=(values.get("POC_REASONING_EFFORT") or "").strip(),
    )
