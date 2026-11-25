from datetime import datetime
from pathlib import Path
from poc.dataset import load_case, load_mailbox, load_reference
from poc.models import AgentResult, Conflict, Fact, Material, ProposedTransition, Status
from poc.scope import build_scope
from poc.validate import validate

T = Path(__file__).resolve().parents[2] / "TEST"
BOX = load_mailbox(T)
REF = load_reference(T)


def ctx(cid):
    c = load_case(T, cid, "with_mail")
    return c, build_scope(c, BOX)


def res(c, to=None, event=None, basis=(), last=None, facts=(), conflicts=()):
    pt = ProposedTransition(to_status=to, event_code=event, basis_source_ids=list(basis)) if to else None
    return AgentResult(case_id=c.case_id, application_id=c.application_id, card_status=c.card_status,
                       last_confirmed_status=last or c.card_status, facts=list(facts), proposed_transition=pt,
                       conflicts=list(conflicts), next_action="x")


def codes(v):
    return {i.code for i in v.issues}


def test_valid_t03_passes():
    c, scope = ctx("TEST-20")
    mgr = Material(source_id="MX1", kind="email", at=datetime.fromisoformat("2026-11-20T11:30:00+03:00"),
                   sender="manager@example.com", recipient="desk@example.com",
                   text="Деньги по заявке TEST-APP-102 поступили.")
    v = validate(res(c, Status.AWAIT_PAYOUT, "incoming_payment_received", ["MX1"]), c, [*scope, mgr], REF)
    assert v.issues == [] and v.checked.proposed_transition is not None


def test_client_paid_message_not_basis_for_t03():
    c, scope = ctx("TEST-20")
    v = validate(res(c, Status.AWAIT_PAYOUT, "incoming_payment_received", ["M019"]), c, scope, REF)
    assert v.checked.proposed_transition is None and v.checked.requires_review
    assert codes(v) & {"basis_wrong_sender", "basis_wrong_type"}


def test_manager_message_without_id():
    c, _ = ctx("TEST-03")
    c = c.model_copy(update={"card_status": Status.AWAIT_FUNDS})
    m044 = next(m for m in BOX if m.source_id == "M044")
    crm = next(m for m in c.start if m.source_id == "CRM")
    v = validate(res(c, Status.AWAIT_PAYOUT, "incoming_payment_received", ["M044"]), c, [crm, m044], REF)
    assert "basis_without_id" in codes(v) and v.checked.proposed_transition is None


def test_close_without_two_signatures():
    c, scope = ctx("TEST-09")
    v = validate(res(c, Status.DONE, "act_signed", ["M045/act_act.pdf"]), c, scope, REF)
    assert v.checked.proposed_transition is None
    c3, scope3 = ctx("TEST-03")
    v3 = validate(res(c3, Status.DONE, "act_signed", ["M002/act_act.pdf"]), c3, scope3, REF)
    assert v3.issues == [] and v3.checked.proposed_transition is not None


def test_unknown_source_removed():
    c, scope = ctx("TEST-20")
    v = validate(res(c, facts=[Fact(fact="x", source_ids=["M999", "M019"])]), c, scope, REF)
    assert v.checked.facts[0].source_ids == ["M019"] and "unknown_source" in codes(v)


def test_status_mismatch_requires_conflict():
    c, scope = ctx("TEST-21")
    last = Status.AWAIT_PAYOUT
    v = validate(res(c, last=last), c, scope, REF)
    assert "status_mismatch_without_conflict" in codes(v) and v.checked.requires_review
    v2 = validate(res(c, Status.ACT, "counterparty_received_payment", ["CRM"], last=last,
                      conflicts=[Conflict(source_a="CRM", source_b="M041", note="платёж не выполнен")]), c, scope, REF)
    assert "status_mismatch_without_conflict" not in codes(v2) and v2.checked.proposed_transition is None


def test_raw_unchanged():
    c, scope = ctx("TEST-20")
    r = res(c, Status.AWAIT_PAYOUT, "incoming_payment_received", ["M019"], facts=[Fact(fact="x", source_ids=["M999"])])
    before = r.model_copy(deep=True)
    v = validate(r, c, scope, REF)
    assert v.raw == before


def test_used_mail_ids_recomputed():
    c, scope = ctx("TEST-01")
    v = validate(res(c, facts=[Fact(fact="x", source_ids=["M020/application.pdf", "CRM"])]), c, scope, REF)
    assert v.checked.used_mail_ids == ["M020"]


def test_other_request_refs_dropped():
    c, scope = ctx("TEST-20")   # M012 — руководитель про TEST-APP-101 того же клиента
    v = validate(res(c, facts=[Fact(fact="Поступление по другой заявке", source_ids=["M012"]),
                               Fact(fact="Клиент сообщил о переводе", source_ids=["M019"])]), c, scope, REF)
    assert [f.fact for f in v.checked.facts] == ["Клиент сообщил о переводе"]
    assert v.checked.used_mail_ids == ["M019"]
    assert "other_request" in codes(v)
    assert not v.checked.requires_review or v.raw.requires_review


def test_conflict_drops_transition():
    c, scope = ctx("TEST-20")
    mgr = Material(source_id="MX1", kind="email", at=datetime.fromisoformat("2026-11-20T11:30:00+03:00"),
                   sender="manager@example.com", recipient="desk@example.com",
                   text="Деньги по заявке TEST-APP-102 поступили.")
    r = res(c, Status.AWAIT_PAYOUT, "incoming_payment_received", ["MX1"],
            conflicts=[Conflict(source_a="MX1", source_b="M019", note="разные даты")])
    v = validate(r, c, [*scope, mgr], REF)
    assert v.checked.proposed_transition is None and v.checked.requires_review
    assert "conflict_blocks_transition" in codes(v)


def test_review_follows_triggers():
    c, scope = ctx("TEST-20")
    r = res(c).model_copy(update={"requires_review": True, "review_triggers": [], "review_reasons": ["мелочь"]})
    assert validate(r, c, scope, REF).checked.requires_review is False
    r2 = res(c).model_copy(update={"requires_review": False, "review_triggers": ["missing_confirmation"]})
    assert validate(r2, c, scope, REF).checked.requires_review is True


def test_code_issue_forces_review():
    c, scope = ctx("TEST-20")
    r = res(c, Status.AWAIT_PAYOUT, "incoming_payment_received", ["M019"]).model_copy(update={"review_triggers": []})
    assert validate(r, c, scope, REF).checked.requires_review is True


def test_unknown_trigger_rejected():
    import pytest
    from pydantic import ValidationError
    c, _ = ctx("TEST-20")
    with pytest.raises(ValidationError):
        res(c).model_copy(update={}).__class__(**{**res(c).model_dump(), "review_triggers": ["other_requests"]})


def test_missing_without_transition_adds_trigger():
    from poc.models import Missing
    c, scope = ctx("TEST-20")
    r = res(c).model_copy(update={"missing": [Missing(what="Подтверждение руководителя", why="нужно для перехода")]})
    v = validate(r, c, scope, REF)
    assert v.checked.requires_review and "missing_confirmation" in v.checked.review_triggers


def test_document_basis_must_belong_to_request():
    c, scope = ctx("TEST-15")   # письмо M049 про TEST-APP-127, а акт внутри — про DEMO-OTHER-INV-TEST-15
    v = validate(res(c, Status.DONE, "act_signed", ["M049/act_act.pdf"]), c, scope, REF)
    assert v.checked.proposed_transition is None and "document_other_request" in codes(v)


def test_own_document_basis_still_passes():
    c3, scope3 = ctx("TEST-03")
    v3 = validate(res(c3, Status.DONE, "act_signed", ["M002/act_act.pdf"]), c3, scope3, REF)
    assert v3.issues == []
