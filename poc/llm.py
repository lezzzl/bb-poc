"""Единственная точка выбора модели. Для перехода на Ollama меняется только эта функция."""

from langchain_core.language_models import BaseChatModel

from poc.config import PocSettings


def create_model(s: PocSettings) -> BaseChatModel:
    if not s.api_key:
        raise ValueError("Заполни OPENROUTER_API_KEY в .env.")
    if not s.model:
        raise ValueError("Заполни OPENROUTER_MODEL в .env.")
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(model=s.model, api_key=s.api_key, base_url=s.base_url, temperature=0,
                          timeout=int(s.timeout_s * 1000), max_retries=1,  # timeout в миллисекундах
                          reasoning={"effort": s.reasoning_effort} if s.reasoning_effort else None)
