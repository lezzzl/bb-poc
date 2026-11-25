"""Модели-заглушки для тестов графа: без сети, с заранее заданными ответами."""

import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda


class ScriptedModel(BaseChatModel):
    script: list = []          # AIMessage или callable(i) -> AIMessage
    structured: list = []      # AgentResult | dict | Exception
    calls: dict = {}

    def model_post_init(self, _):
        self.calls = {"generate": 0, "structured": 0}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        i = self.calls["generate"]
        self.calls["generate"] += 1
        item = self.script[i] if i < len(self.script) else AIMessage(content="готово")
        msg = item(i) if callable(item) else item
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(self, schema, **kwargs):
        def run(_: Any):
            i = self.calls["structured"]
            self.calls["structured"] += 1
            item = self.structured[min(i, len(self.structured) - 1)]
            if isinstance(item, Exception):
                raise item
            return schema(**item) if isinstance(item, dict) else item
        return RunnableLambda(run)


class HangingModel(ScriptedModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        time.sleep(5)
        return super()._generate(messages)


def search_call(i: int, query: str = "TEST-APP-102") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "search", "args": {"query": query}, "id": f"c{i}"}])
