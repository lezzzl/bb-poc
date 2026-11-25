"""Страница PoC: один эпизод одной заявки, режимы «с помощником» и «вручную», общий таймер и форма.

Запуск: streamlit run poc/app.py --server.address 127.0.0.1
"""

import html
import json
import sqlite3
import time
from datetime import datetime, timezone

import streamlit as st
from langgraph.checkpoint.sqlite import SqliteSaver

from poc.assignments import load_assignments, next_assignment
from poc.config import load_settings
from poc.dataset import load_case
from poc.graph import _mailbox, episode_context, resume_episode, run_episode
from poc.journal import Journal, dataset_version, edits_diff, poc_version
from poc.llm import create_model
from poc.models import TRIGGER_RU, AgentResult, Decision, Status
from poc.ui_form import empty_result, form_to_result, result_to_form

st.set_page_config(page_title="PoC · сопровождение заявки", layout="wide")
S = load_settings()
STATUSES = [s.value for s in Status]


@st.cache_resource
def journal() -> Journal:
    return Journal(S.data_dir / "poc.sqlite")


@st.cache_resource
def checkpointer() -> SqliteSaver:
    return SqliteSaver(sqlite3.connect(S.data_dir / "checkpoints.sqlite", check_same_thread=False))


@st.cache_resource
def model():
    return create_model(S)


def all_cases() -> list[str]:
    return sorted(p.name for p in (S.test_dir / "cases").iterdir() if p.is_dir())


ss = st.session_state
J = journal()

# ---------- шапка ----------
assignments = load_assignments(S.test_dir / "assignments.csv")
h1, h2, h3, h4, h5 = st.columns([2, 2, 2, 1.2, 1.5])
participant = h1.text_input("Участник", value=ss.get("participant", ""), disabled="run_id" in ss)
assigned = next_assignment(assignments, J.runs(), participant) if assignments and participant else None
if assignments:
    case_id = assigned.case_id if assigned else None
    mode_key = assigned.mode if assigned else "assistant"
    h2.text_input("Кейс", value=case_id or "нет заданий", disabled=True)
    h3.text_input("Режим", value={"assistant": "с помощником", "manual": "вручную"}[mode_key], disabled=True)
else:
    case_id = h2.selectbox("Кейс", all_cases(), index=all_cases().index("TEST-20"), disabled="run_id" in ss)
    mode_key = h3.radio("Режим", ["assistant", "manual"], horizontal=True, disabled="run_id" in ss,
                        format_func={"assistant": "с помощником", "manual": "вручную"}.get)


@st.fragment(run_every=1)
def timer():
    if "started_at" in ss:
        end = ss.get("finished_at", time.time())
        sec = int(end - ss["started_at"])
        st.markdown(f"### ⏱ {sec // 60:02d}:{sec % 60:02d}")
    else:
        st.markdown("### ⏱ 00:00")


with h5:
    timer()

if "run_id" not in ss:
    if h4.button("▶ Старт", type="primary", disabled=not (participant and case_id)):
        case = load_case(S.test_dir, case_id, "with_mail")
        ss.update(participant=participant, case_id=case_id, mode=mode_key, started_at=time.time(),
                  thread_id=None, summary=None, issues=[], submitted=False)
        ss["run_id"] = J.start_run(case=case, participant=participant, mode=mode_key,
                                   order_no=assigned.order_no if assigned else None,
                                   poc_version=poc_version(S), dataset_version=dataset_version(S.test_dir))
        st.rerun()
    st.info("Выберите участника, кейс и режим, затем нажмите «Старт». Материалы откроются после старта.")
    st.stop()

case, scope, ref = episode_context(S, ss["case_id"], "with_mail")

# ---------- запуск помощника ----------
if ss["mode"] == "assistant" and ss.get("summary") is None:
    ss["thread_id"] = f"{ss['case_id']}:{ss['run_id']}"
    with st.spinner("Помощник разбирает материалы заявки…"):
        try:
            summary = run_episode(S, ss["case_id"], "with_mail", model(), checkpointer(), ss["thread_id"])
        except Exception as e:  # конфигурация модели, сеть
            summary = {"status": "failed", "error": f"model_error: {type(e).__name__}", "validated": None,
                       "messages": [], "model_latency_s": 0, "tool_calls": 0}
    ss["summary"] = summary
    if summary["status"] == "failed":
        partial = [getattr(m, "content", "") for m in summary.get("messages", []) if getattr(m, "type", "") == "ai"]
        J.finish_run(ss["run_id"], secrets=(S.api_key,), failure=f"{summary['error']}: {summary.get('error_detail') or ''}",
                     raw_json={"partial_ai_messages": partial}, manual_fallback="1",
                     model_latency_s=summary["model_latency_s"], finished_at=None, duration_s=None)
    else:
        v = summary["validated"]
        tool_calls = [{"name": c["name"], "args": c["args"]} for m in summary["messages"]
                      for c in (getattr(m, "tool_calls", None) or [])]
        ss["tool_calls"] = tool_calls
        J.finish_run(ss["run_id"], secrets=(S.api_key,), raw_json=v["raw"], checked_json=v["checked"],
                     tool_calls_json=tool_calls, model_latency_s=summary["model_latency_s"],
                     finished_at=None, duration_s=None)
    st.rerun()

summary = ss.get("summary") or {}
failed = summary.get("status") == "failed"
validated = summary.get("validated")
manual_form = ss["mode"] == "manual" or failed

# ---------- карточка заявки ----------
crm = json.loads(next(m for m in case.start if m.source_id == "CRM").text)
with st.container(border=True):
    k1, k2, k3, k4 = st.columns([1.2, 1.4, 1.4, 1.6])
    k1.markdown(f"**Заявка**  \n{case.application_id}")
    k2.markdown(f"**Клиент**  \n{crm['client']['name']}")
    k3.markdown(f"**Инвойс контрагента**  \n{crm.get('invoice_id') or '—'} · "
                f"{crm.get('counterparty_invoice_amount', '')} {crm.get('currency', '')}  \n{crm['counterparty']['name']}")
    k4.markdown(f"**Статус в CRM**  \n{case.card_status.value}")
    docs = ", ".join(f"{m.source_id} ({m.kind})" for m in case.start if m.source_id != "CRM") or "нет"
    st.caption(f"Уже есть в папке заявки: {docs} · материалы на {case.cutoff_at:%d.%m.%Y %H:%M}")

# ---------- источники ----------
left, right = st.columns([1.25, 1], gap="large")
with left:
    if ss["mode"] == "manual" or failed:
        materials = [*case.start, *[m for m in _mailbox(S.test_dir) if m.at <= case.cutoff_at]]
    else:
        materials = scope
    used = set(validated["checked"]["used_mail_ids"]) if validated else set()
    children: dict[str, list] = {}
    for m in materials:
        if m.parent_id:
            children.setdefault(m.parent_id, []).append(m)
    tops = [m for m in materials if not m.parent_id]
    n_letters = sum(m.kind == "email" for m in tops)
    if ss["mode"] == "manual" or failed:
        st.caption(f"Стартовые материалы и весь ящик: {n_letters} писем")
        q = st.text_input("Поиск по тексту", key="q", placeholder="TEST-APP-102, поступление, акт…")
        if q:
            ql = q.lower()
            tops = [m for m in tops if any(ql in (x.source_id + " " + (x.subject or "") + " " + x.text).lower()
                                           for x in [m, *children.get(m.source_id, [])])]
    else:
        st.caption(f"Источники в области заявки: {n_letters} писем из 57")
    tops.sort(key=lambda m: (m.kind == "email", m.at))
    KIND = {"crm_snapshot": "CRM", "invoice": "Счёт", "application": "Заявка", "act": "Акт",
            "payment_confirmation": "Платёжка", "email": "Письмо"}
    rows = [{"✓": "✓" if m.source_id in used or any(c.source_id in used for c in children.get(m.source_id, [])) else "",
             "ID": m.source_id, "Время": f"{m.at:%H:%M}", "От": (m.sender or "—").replace("@example.com", ""),
             "Тема / тип": m.subject or KIND.get(m.kind, m.kind),
             "📎": str(len(children.get(m.source_id, [])) or "")} for m in tops]
    sel = st.dataframe(rows, hide_index=True, width="stretch", height=260, on_select="rerun",
                       selection_mode="single-row", key=f"src_{len(rows)}",
                       column_config={"✓": st.column_config.TextColumn(width=30), "ID": st.column_config.TextColumn(width=60),
                                      "Время": st.column_config.TextColumn(width=55), "От": st.column_config.TextColumn(width=110), "📎": st.column_config.TextColumn(width=30)})
    picked = sel.selection.rows[0] if sel and sel.selection.rows else None
    by_id = {x.source_id: x for x in [*materials, *[c for cs in children.values() for c in cs]]}

    def show(m):
        with st.container(height=520, border=True):
            st.markdown(f"**{m.subject or KIND.get(m.kind, m.kind)}**")
            st.caption(f"{m.source_id} · от {m.sender or '—'} → {m.recipient or '—'} · {m.at:%d.%m %H:%M}")
            if m.signatures:
                st.caption(f"Подписи (проверка кода): принципал {m.signatures.principal}, исполнитель {m.signatures.provider}")
            body = m.text.split("\n", 1)[1] if m.kind == "email" and "\n" in m.text else m.text
            st.html(f"<div style='white-space:pre-wrap;font-size:14px;line-height:1.5'>{html.escape(body[:8000])}</div>")
            for c in children.get(m.source_id, []):
                with st.expander(f"📎 {c.source_id.split('/')[-1]} · {KIND.get(c.kind, c.kind)}",
                                 expanded=ss.get("open_src") == c.source_id):
                    if c.signatures:
                        st.caption(f"Подписи (проверка кода): принципал {c.signatures.principal}, исполнитель {c.signatures.provider}")
                    st.html(f"<div style='white-space:pre-wrap;font-size:13px;line-height:1.45'>{html.escape(c.text[:8000])}</div>")

    opened = by_id.get(ss.get("open_src", ""))
    if opened is not None:
        head = st.columns([4, 1])
        head[0].caption(f"Открыт источник из сводки: {opened.source_id}")
        if head[1].button("✕", key="close_src", help="Закрыть"):
            ss.pop("open_src", None)
            st.rerun()
        # вложение показываем внутри письма, чтобы было видно, откуда оно
        show(by_id.get(opened.parent_id, opened) if opened.parent_id else opened)
    elif picked is not None and picked < len(tops):
        show(tops[picked])
    else:
        st.caption("Выберите строку, чтобы открыть письмо или документ.")

# ---------- результат и форма ----------
with right:
    if failed:
        st.error(f"Помощник не дал результат ({summary['error']}). Заполните форму вручную, таймер идёт.")
    if ss.get("submitted"):
        st.success("Результат сдан и записан в журнал.")
    base: AgentResult = AgentResult(**validated["checked"]) if (validated and not failed) else empty_result(case)
    f = result_to_form(base)

    if validated and not failed:
        top = st.columns([3, 2])
        top[0].markdown(f"#### {f['last_confirmed_status']}")
        if f["requires_review"]:
            why = ", ".join(TRIGGER_RU.get(x, x) for x in f.get("review_triggers", []))
            top[1].warning("Требуется проверка" + (f": {why}" if why else ""))
        issues = ss.get("issues") or validated.get("issues", [])
        if issues:
            st.caption("Проверка кода:")
            for i in issues:
                st.markdown(f"- `{i['code']}` {i['message']}")
        else:
            st.caption("Проверка кода: замечаний нет")
        refs = []
        for x in base.facts:
            refs += x.source_ids
        refs += base.proposed_transition.basis_source_ids if base.proposed_transition else []
        refs += [s for c in base.conflicts for s in (c.source_a, c.source_b)]
        refs = [r for i, r in enumerate(refs) if r not in refs[:i]]
        if refs:
            st.caption("Источники выводов — нажмите, чтобы открыть слева:")
            cols = st.columns(min(len(refs), 6))
            for i, r in enumerate(refs):
                if cols[i % len(cols)].button(r.split("/")[-1] if "/" in r else r, key=f"ref_{r}", help=r,
                                              type="primary" if ss.get("open_src") == r else "secondary"):
                    ss["open_src"] = r
                    st.rerun()
        with st.expander("Запросы агента к инструментам"):
            for c in ss.get("tool_calls", []):
                st.code(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})")

    disabled = ss.get("submitted", False)
    v = {}
    st.markdown("##### 1. Где заявка сейчас")
    v["last_confirmed_status"] = st.selectbox(
        "Подтверждённый этап", STATUSES, index=STATUSES.index(f["last_confirmed_status"]), disabled=disabled,
        help="Последний этап, который подтверждён. По умолчанию — статус в CRM. Меняйте, только если материалы прямо "
             "опровергают CRM (тогда укажите конфликт ниже).")
    nxt = STATUSES[STATUSES.index(case.card_status.value) + 1] if case.card_status.value != STATUSES[-1] else None
    to_opts = ["", nxt] if nxt else [""]
    if f["transition_to"] and f["transition_to"] not in to_opts:
        to_opts.append(f["transition_to"])
    v["transition_to"] = st.radio(
        "Можно ли перевести заявку на следующий этап?", to_opts, index=to_opts.index(f["transition_to"]),
        format_func=lambda x: f"Да → «{x}»" if x else "Нет, оставить как есть", disabled=disabled, horizontal=True,
        help="Только на один этап вперёд и только при подтверждении нужного типа: поступление и платёж — "
             "руководитель, получение денег контрагентом — клиент, подписание — документ с двумя подписями.")
    v["basis_source_ids"] = []
    if v["transition_to"]:
        src_ids = sorted({m.source_id for m in materials} | {c.source_id for cs in children.values() for c in cs}
                         | set(f["basis_source_ids"]))
        v["basis_source_ids"] = st.multiselect("Чем подтверждено (письмо или документ)", src_ids,
                                               default=f["basis_source_ids"], disabled=disabled,
                                               placeholder="Выберите источник, например M012")

    st.markdown("##### 2. Что нового и на чём основано")
    v["facts"] = st.text_area(
        "Факты — по одному в строке, источник в квадратных скобках", f["facts"], height=110, disabled=disabled,
        placeholder="Клиент сообщил о переводе по заявке [M019]\nПодтверждения руководителя о поступлении нет [CRM]")
    v["missing"] = st.text_area(
        "Чего не хватает — по одному в строке: что — зачем", f["missing"], height=70, disabled=disabled,
        placeholder="Подтверждение руководителя о поступлении — без него нельзя перейти к оплате")
    v["conflicts"] = st.text_area(
        "Противоречия — по одному в строке: источник A | источник B | в чём расхождение", f["conflicts"], height=70,
        disabled=disabled, placeholder="CRM | M041 | в CRM платёж отправлен, руководитель пишет, что нет")

    st.markdown("##### 3. Что делать дальше")
    v["next_action"] = st.text_input("Следующий шаг сотрудника", f["next_action"], disabled=disabled,
                                     placeholder="Уточнить у руководителя поступление денег по заявке")
    v["clarification"] = st.text_area("Черновик сообщения (если нужно что-то уточнить)", f["clarification"],
                                      height=80, disabled=disabled,
                                      placeholder="Кому: manager@… Подтвердите, поступили ли деньги по заявке …")
    v["requires_review"] = st.checkbox("Требуется проверка (нехватка данных, конфликт, неясная привязка)",
                                       value=f["requires_review"], disabled=disabled)
    v["review_reasons"] = st.text_area("Почему", f["review_reasons"], height=60, disabled=disabled) \
        if v["requires_review"] else ""
    v["used_mail_ids"] = f["used_mail_ids"]
    try:
        edited = form_to_result(v, case)
    except ValueError:  # pydantic ValidationError — подкласс ValueError
        st.error("Проверьте форму: не удалось разобрать заполненные поля.")
        edited = base

    def finish(decision: str, submitted: AgentResult, override: bool = False):
        ss["finished_at"] = time.time()
        J.finish_run(ss["run_id"], secrets=(S.api_key,), submitted_json=submitted.model_dump(mode="json"),
                     edits_json={k: list(x) for k, x in edits_diff(base, submitted).items()},
                     decision=decision, override="1" if override else "0",
                     manual_fallback="1" if failed else None)
        ss["submitted"] = True
        st.rerun()

    if ss.get("submitted"):
        if st.button("Новый эпизод"):
            for k in ["run_id", "started_at", "finished_at", "summary", "issues", "submitted", "thread_id",
                      "tool_calls", "facts", "missing", "conflicts", "pick", "q", "open_src"]:
                ss.pop(k, None)
            st.rerun()
    elif manual_form:
        if st.button("Сдать", type="primary"):
            finish("submitted", edited)
    else:
        override = bool(ss.get("issues")) and st.checkbox("Принимаю без основания нужного типа")
        b1, b2, b3 = st.columns(3)
        action = None
        if b1.button("Принять", type="primary"):
            action, result = "accept", base
        if b2.button("Принять с правками"):
            action, result = "accept_edited", edited
        if b3.button("Отклонить"):
            action, result = "reject", base
        if action:
            def on_save(c, state):
                J.apply_decision(c, Decision(**state["decision"]), ss["run_id"])
            out = resume_episode(S, ss["case_id"], "with_mail", model(), checkpointer(), ss["thread_id"],
                                 Decision(action=action, result=result, override=override), on_save=on_save)
            if out["status"] == "review":
                ss["issues"] = out["validated"]["issues"]
                st.rerun()
            finish(action, Decision(**out["decision"]).result if out.get("decision") else result, override)

    card = J.card(case.case_id, case)
    st.caption(f"Тестовая карточка: «{card['status']}»{' · требуется проверка' if card['requires_review'] else ''}")
