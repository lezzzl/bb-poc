from pathlib import Path
from poc.assignments import Assignment, load_assignments, next_assignment
from poc.dataset import load_case
from poc.models import AgentResult, Conflict, Fact, Missing, ProposedTransition, Status
from poc.ui_form import empty_result, form_to_result, parse_facts, result_to_form

CASE = load_case(Path(__file__).resolve().parents[2] / "TEST", "TEST-20", "with_mail")


def test_form_roundtrip():
    r = AgentResult(case_id="TEST-20", application_id="TEST-APP-102", card_status=Status.AWAIT_FUNDS,
                    last_confirmed_status=Status.AWAIT_FUNDS, facts=[Fact(fact="f", source_ids=["M019"])],
                    proposed_transition=ProposedTransition(to_status=Status.AWAIT_PAYOUT,
                                                           event_code="incoming_payment_received", basis_source_ids=["M1"]),
                    requires_review=True, review_reasons=["r"], missing=[Missing(what="w", why="y")],
                    conflicts=[Conflict(source_a="CRM", source_b="M1", note="n")], next_action="a",
                    clarification="c", used_mail_ids=["M019"])
    assert form_to_result(result_to_form(r), CASE) == r


def test_empty_result_keeps_crm_status():
    e = empty_result(CASE)
    assert e.last_confirmed_status == CASE.card_status and e.proposed_transition is None and e.facts == []


def _write(tmp_path):
    p = tmp_path / "assignments.csv"
    p.write_text("participant,order_no,case_id,mode\nМаксим,1,TEST-01,manual\nМаксим,2,TEST-02,assistant\n"
                 "Артём,1,TEST-02,manual\n", encoding="utf-8")
    return p


def test_no_file_returns_none(tmp_path):
    assert load_assignments(tmp_path / "nope.csv") is None


def test_next_assignment_skips_done(tmp_path):
    a = load_assignments(_write(tmp_path))
    runs = [{"participant": "Максим", "case_id": "TEST-01", "finished_at": "x"}]
    assert next_assignment(a, runs, "Максим") == Assignment(participant="Максим", order_no=2, case_id="TEST-02", mode="assistant")


def test_participant_cannot_repeat_case(tmp_path):
    a = load_assignments(_write(tmp_path))
    runs = [{"participant": "Артём", "case_id": "TEST-02", "finished_at": "x"}]
    assert next_assignment(a, runs, "Артём") is None


def test_parse_facts_lines():
    facts = parse_facts("Клиент сообщил о переводе [M019]\nНет подтверждения руководителя\n\n  Акт подписан [M002, M002/act_act.pdf] ")
    assert [f.fact for f in facts] == ["Клиент сообщил о переводе", "Нет подтверждения руководителя", "Акт подписан"]
    assert facts[0].source_ids == ["M019"] and facts[1].source_ids == [] and facts[2].source_ids == ["M002", "M002/act_act.pdf"]


def test_event_derived_from_status():
    v = result_to_form(empty_result(CASE))
    v["transition_to"] = "Ожидаем отправки платежа"
    v["basis_source_ids"] = ["M019"]
    assert form_to_result(v, CASE).proposed_transition.event_code == "incoming_payment_received"
