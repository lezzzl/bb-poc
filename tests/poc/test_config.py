import os
import pytest
from poc.config import load_settings
from poc.llm import create_model


def _env(tmp_path, text):
    (tmp_path / ".env").write_text(text, encoding="utf-8")
    return load_settings(tmp_path)


def test_tracing_disabled_after_load(tmp_path):
    os.environ["LANGSMITH_TRACING"] = "true"
    _env(tmp_path, "OPENROUTER_MODEL=x\n")
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_timeout_default_60(tmp_path):
    assert _env(tmp_path, "OPENROUTER_MODEL=x\n").timeout_s == 60


def test_create_model_requires_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    s = _env(tmp_path, "OPENROUTER_MODEL=x\nOPENROUTER_API_KEY=\n")
    with pytest.raises(ValueError):
        create_model(s)


def test_key_not_in_repr(tmp_path):
    s = _env(tmp_path, "OPENROUTER_MODEL=x\nOPENROUTER_API_KEY=sk-test\n")
    assert "sk-test" not in repr(s)


def test_model_timeout_in_milliseconds(tmp_path):
    s = _env(tmp_path, "OPENROUTER_MODEL=x\nOPENROUTER_API_KEY=sk-test\n")
    assert create_model(s).request_timeout == 60_000


def test_reasoning_effort_passed(tmp_path):
    s = _env(tmp_path, "OPENROUTER_MODEL=x\nOPENROUTER_API_KEY=sk-test\nPOC_REASONING_EFFORT=medium\n")
    assert s.reasoning_effort == "medium" and create_model(s).reasoning == {"effort": "medium"}
    s2 = _env(tmp_path, "OPENROUTER_MODEL=x\nOPENROUTER_API_KEY=sk-test\nPOC_REASONING_EFFORT=\n")
    assert create_model(s2).reasoning is None
