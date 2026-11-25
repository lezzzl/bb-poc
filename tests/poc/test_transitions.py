from pathlib import Path
import pytest
from pydantic import ValidationError
from poc.models import AgentResult, Status
from poc.transitions import load_rules, rule_for

RULES = Path(__file__).resolve().parents[2] / "TEST/reference/transitions.json"


def test_six_rules_loaded():
    rules = load_rules(RULES)
    assert len(rules) == 6
    t03 = next(r for r in rules if r.id == "T03")
    assert t03.evidence_types == ["manager_message", "manager_confirmation_file"]


def test_rule_forward_only():
    rules = load_rules(RULES)
    s = Status
    assert rule_for(rules, s("Ожидаем поступления денег"), "incoming_payment_received", s("Ожидаем отправки платежа")).id == "T03"
    assert rule_for(rules, s("Новая заявка"), "incoming_payment_received", s("Ожидаем отправки платежа")) is None
    assert rule_for(rules, s("Ожидаем поступления денег"), "incoming_payment_received", s("Платёж отправлен")) is None


def test_agent_result_rejects_unknown_status():
    with pytest.raises(ValidationError):
        AgentResult(case_id="X", application_id="A", card_status="Новая заявка", last_confirmed_status="Оплачено",
                    facts=[], proposed_transition=None, requires_review=False, next_action="x")


def test_event_code_is_enum():
    from poc.models import ProposedTransition
    with pytest.raises(ValidationError):
        ProposedTransition(to_status="Ожидаем отправки платежа", event_code="PAYMENT_RECEIPT_CONFIRMED")
    assert ProposedTransition(to_status="Ожидаем отправки платежа", event_code="incoming_payment_received")


def test_agent_result_accepts_json_string_lists():
    r = AgentResult(case_id="X", application_id="A", card_status="Новая заявка", last_confirmed_status="Новая заявка",
                    conflicts='[{"source_a": "M1", "source_b": "M2", "note": "n"}]', facts="[]", next_action="x")
    assert r.conflicts[0].source_a == "M1" and r.facts == []
