import json
from pathlib import Path
import pytest
from poc.dataset import load_case, load_mailbox
from poc.index import ScopeIndex
from poc.scope import build_scope
from poc.tools import make_tools

T = Path(__file__).resolve().parents[2] / "TEST"
BOX = load_mailbox(T)


def _scope(cid):
    return build_scope(load_case(T, cid, "with_mail"), BOX)


@pytest.mark.parametrize("n", range(1, 33))
def test_scope_covers_expected_mails_all_cases(n):
    cid = f"TEST-{n:02d}"
    scope = _scope(cid)
    exp = json.loads((T / f"reference/mail_pool_states/{cid}/expected.json").read_text())
    ids = {m.source_id for m in scope}
    assert set(exp["expected_mail_ids"]) <= ids
    letters = [m for m in scope if m.kind == "email" and m.parent_id is None]
    assert 6 <= len(letters) <= 10


def test_scope_excludes_other_client():
    ids = {m.source_id for m in _scope("TEST-20")}
    assert "M036" not in ids and "M019" in ids


def test_search_finds_application_id():
    idx = ScopeIndex(_scope("TEST-20"))
    assert "M019" in [h["source_id"] for h in idx.search("TEST-APP-102")]


def test_search_handles_short_and_symbols():
    idx = ScopeIndex(_scope("TEST-20"))
    assert isinstance(idx.search("а"), list)
    assert isinstance(idx.search('"('), list)


def _tool(name, scope):
    return next(t for t in make_tools(ScopeIndex(scope)) if t.name == name)


def test_get_outside_scope():
    assert _tool("get", _scope("TEST-20")).invoke({"source_id": "M036"}).startswith("Нет такого источника")


def test_tool_output_wrapped():
    assert '<data source_id="M019"' in _tool("get", _scope("TEST-20")).invoke({"source_id": "M019"})


def test_scope_excludes_foreign_sender_mentioning_our_client():
    from datetime import datetime
    from poc.dataset import load_reference
    from poc.models import Material
    c = load_case(T, "TEST-20", "with_mail")
    alien = Material(source_id="MX9", kind="email", at=datetime.fromisoformat("2026-11-20T09:00:00+03:00"),
                     sender="client-beacon@example.com", recipient="desk@example.com", text="Мы партнёры Север-Лаб")
    foreign = {x["email"] for x in load_reference(T).companies.values() if x["role"] == "Клиент"} - {c.client_email}
    ids = {m.source_id for m in build_scope(c, [*BOX, alien], frozenset(foreign))}
    assert "MX9" not in ids


from poc.dataset import load_reference
REF = load_reference(T)


def _tool2(name, cid):
    c = load_case(T, cid, "with_mail")
    return next(t for t in make_tools(ScopeIndex(build_scope(c, BOX)), c, REF) if t.name == name)


def test_signatures_tool_one_side():
    out = _tool2("signatures", "TEST-09").invoke({"source_id": "M045/act_act.pdf"})
    assert "принципал: нет" in out and "исполнитель: есть" in out and "обеими сторонами: НЕТ" in out


def test_signatures_tool_both():
    out = _tool2("signatures", "TEST-03").invoke({"source_id": "M002/act_act.pdf"})
    assert "обеими сторонами: ДА" in out


def test_signatures_tool_not_signable_and_outside():
    t = _tool2("signatures", "TEST-20")
    assert "не заявка и не акт" in t.invoke({"source_id": "M019"})
    assert t.invoke({"source_id": "M036"}).startswith("Нет такого источника")


def test_next_step_tool():
    out = _tool2("next_step_rule", "TEST-20").invoke({})
    assert "Ожидаем поступления денег" in out and "Ожидаем отправки платежа" in out
    assert "incoming_payment_received" in out and "руководител" in out


def test_next_step_tool_final_status():
    from poc.models import Status
    c = load_case(T, "TEST-03", "with_mail").model_copy(update={"card_status": Status.DONE})
    t = next(t for t in make_tools(ScopeIndex(c.start), c, REF) if t.name == "next_step_rule")
    assert "завершена" in t.invoke({}).lower()


def test_search_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor
    idx = ScopeIndex(_scope("TEST-20"))
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda q: idx.search(q), ["TEST-APP-102", "поступление", "акт", "TEST-INV-702"] * 50))
    assert all(isinstance(r, list) for r in res)


def test_get_includes_signatures_for_act():
    out = _tool2("get", "TEST-09").invoke({"source_id": "M045/act_act.pdf"})
    assert "Подписи (проверка кода)" in out and "обеими сторонами: НЕТ" in out
