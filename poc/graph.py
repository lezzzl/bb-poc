"""Граф эпизода: scope → agent⇄tools → finalize → check → review(interrupt) → save."""

import threading
import time
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from poc.config import PocSettings
from poc.dataset import Reference, load_case, load_mailbox, load_reference
from poc.index import ScopeIndex
from poc.models import AgentResult, Case, Decision, Material
from poc.scope import build_scope
from poc.tools import make_tools
from poc.validate import validate

TRANSITION_CODES = {"transition_not_allowed", "basis_wrong_type", "basis_wrong_sender", "basis_without_id"}
SaveHook = Callable[[Case, dict], None]


class EpisodeState(TypedDict, total=False):
    case_id: str
    mode: str
    messages: Annotated[list, add_messages]
    tool_calls: int
    raw: dict | None
    validated: dict | None
    decision: dict | None
    error: str | None
    error_detail: str | None
    model_latency_s: float


def system_prompt() -> str:
    return files("poc.prompts").joinpath("agent_system.txt").read_text(encoding="utf-8")


@lru_cache(maxsize=4)
def _mailbox(test_dir: Path) -> tuple[Material, ...]:
    return tuple(load_mailbox(test_dir))


@lru_cache(maxsize=4)
def _reference(test_dir: Path) -> Reference:
    return load_reference(test_dir)


def episode_context(settings: PocSettings, case_id: str, mode: str) -> tuple[Case, list[Material], Reference]:
    case = load_case(settings.test_dir, case_id, mode)
    ref = _reference(settings.test_dir)
    foreign = frozenset(c["email"] for c in ref.companies.values() if c.get("role") == "Клиент") - {case.client_email}
    return case, build_scope(case, list(_mailbox(settings.test_dir)), foreign), ref


def task_message(case: Case) -> str:
    return (f"Кейс {case.case_id}. Заявка {case.application_id}, инвойс {case.invoice_id or 'не указан'}, "
            f"клиент {case.client_name} ({case.client_email}). Статус в CRM: «{case.card_status.value}». "
            f"Материалы доступны на {case.cutoff_at.isoformat()}. Разбери эпизод и подготовь результат.")


def build_graph(settings: PocSettings, case_id: str, mode: str, model: BaseChatModel, checkpointer,
                on_save: SaveHook | None = None):
    case, scope, ref = episode_context(settings, case_id, mode)
    tools = make_tools(ScopeIndex(scope), case, ref)
    agent_model = model.bind_tools(tools, parallel_tool_calls=False)
    tool_node = ToolNode(tools, handle_tool_errors=True)
    sys = SystemMessage(content=system_prompt())

    def agent(state: EpisodeState) -> dict:
        new = [] if state.get("messages") else [HumanMessage(content=task_message(case))]
        t0 = time.monotonic()
        resp = agent_model.invoke([sys, *state.get("messages", []), *new])
        return {"messages": [*new, resp], "model_latency_s": state.get("model_latency_s", 0) + time.monotonic() - t0}

    def route(state: EpisodeState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state.get("tool_calls", 0) < settings.max_tool_calls:
            return "tools"
        return "finalize"

    def run_tools(state: EpisodeState) -> dict:
        out = tool_node.invoke(state)
        n = sum(isinstance(m, ToolMessage) for m in out["messages"])
        return {"messages": out["messages"], "tool_calls": state.get("tool_calls", 0) + n}

    def finalize(state: EpisodeState) -> dict:
        msgs = list(state.get("messages", []))
        if msgs and isinstance(msgs[-1], AIMessage) and msgs[-1].tool_calls:
            msgs = msgs[:-1]
        if not msgs:
            msgs = [HumanMessage(content=task_message(case))]
        structured = model.with_structured_output(AgentResult)
        ask = HumanMessage(content="Дай итоговый результат строго по схеме.")
        t0 = time.monotonic()
        error = None
        for attempt in range(2):
            try:
                res = structured.invoke([sys, *msgs, ask])
                if isinstance(res, dict):
                    res = AgentResult(**res)
                return {"raw": res.model_dump(mode="json"),
                        "model_latency_s": state.get("model_latency_s", 0) + time.monotonic() - t0}
            except Exception as e:  # невалидный ответ модели: один повтор с текстом ошибки
                error = e
                ask = HumanMessage(content=f"Ответ не прошёл проверку схемы: {str(e)[:500]}. Повтори строго по схеме.")
        return {"error": "invalid_output", "raw": None, "error_detail": f"{type(error).__name__}: {str(error)[:2000]}",
                "model_latency_s": state.get("model_latency_s", 0) + time.monotonic() - t0}

    def check(state: EpisodeState) -> dict:
        if state.get("error"):
            return {}
        return {"validated": validate(AgentResult(**state["raw"]), case, scope, ref).model_dump(mode="json")}

    def review(state: EpisodeState) -> dict:
        d = Decision(**interrupt({"validated": state["validated"]}))
        if d.action == "accept_edited":
            v = validate(d.result, case, scope, ref)
            dropped = d.result.proposed_transition is not None and v.checked.proposed_transition is None
            if dropped and not d.override:
                return {"validated": {**v.model_dump(mode="json"), "raw": state["validated"]["raw"]}, "decision": None}
            saved = v.checked
            if dropped:  # сотрудник принял переход под свою ответственность; в журнале override=1
                saved = saved.model_copy(update={"proposed_transition": d.result.proposed_transition})
            d = d.model_copy(update={"result": saved})
        return {"decision": d.model_dump(mode="json")}

    def after_review(state: EpisodeState) -> str:
        return "save" if state.get("decision") else "review"

    def save(state: EpisodeState) -> dict:
        if on_save:
            on_save(case, dict(state))
        return {}

    g = StateGraph(EpisodeState)
    for name, fn in [("agent", agent), ("tools", run_tools), ("finalize", finalize), ("check", check),
                     ("review", review), ("save", save)]:
        g.add_node(name, fn)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "agent")
    g.add_edge("finalize", "check")
    g.add_conditional_edges("check", lambda s: END if s.get("error") else "review", {END: END, "review": "review"})
    g.add_conditional_edges("review", after_review, {"save": "save", "review": "review"})
    g.add_edge("save", END)
    return g.compile(checkpointer=checkpointer)


def _summary(graph, config) -> dict[str, Any]:
    st = graph.get_state(config)
    v = st.values
    base = {"tool_calls": v.get("tool_calls", 0), "model_latency_s": round(v.get("model_latency_s", 0.0), 2),
            "validated": v.get("validated"), "decision": v.get("decision"), "error": v.get("error"),
            "error_detail": v.get("error_detail"),
            "messages": v.get("messages", [])}
    if v.get("error"):
        return {**base, "status": "failed"}
    if "review" in st.next:
        return {**base, "status": "review"}
    return {**base, "status": "done"}


def _with_timeout(settings: PocSettings, fn) -> dict | None:
    """Выполняет fn не дольше timeout_s. Фоновый поток не мешает завершению процесса после таймаута."""
    box: dict = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # передаём ошибку вызывающему
            box["error"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(settings.timeout_s)
    if th.is_alive():
        return None
    if "error" in box:
        raise box["error"]
    return box["value"]


def run_episode(settings, case_id, mode, model, checkpointer, thread_id: str, on_save: SaveHook | None = None) -> dict:
    graph = build_graph(settings, case_id, mode, model, checkpointer, on_save)
    config = {"configurable": {"thread_id": thread_id}}
    if graph.get_state(config).values:  # повторный запуск того же эпизода (перерисовка страницы): без нового вызова модели
        return _summary(graph, config)
    t0 = time.monotonic()
    try:
        done = _with_timeout(settings, lambda: graph.invoke({"case_id": case_id, "mode": mode, "tool_calls": 0}, config))
    except Exception as e:
        return {**_summary(graph, config), "status": "failed", "error": f"model_error: {type(e).__name__}",
                "error_detail": str(e)[:2000], "model_latency_s": round(time.monotonic() - t0, 2)}
    if done is None:
        return {**_summary(graph, config), "status": "failed", "error": "timeout", "error_detail": None,
                "model_latency_s": round(time.monotonic() - t0, 2)}
    return _summary(graph, config)


def resume_episode(settings, case_id, mode, model, checkpointer, thread_id: str, decision: Decision,
                   on_save: SaveHook | None = None) -> dict:
    graph = build_graph(settings, case_id, mode, model, checkpointer, on_save)
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke(Command(resume=decision.model_dump(mode="json")), config)
    return _summary(graph, config)
