import json
import shutil
from pathlib import Path
from poc.config import load_settings
from poc.dataset import load_case
from poc.journal import Journal, edits_diff, poc_version
from poc.models import AgentResult, Decision, ProposedTransition, Status

ROOT = Path(__file__).resolve().parents[2]
CASE = load_case(ROOT / "TEST", "TEST-20", "with_mail")


def r(**kw):
    base = dict(case_id="TEST-20", application_id="TEST-APP-102", card_status=Status.AWAIT_FUNDS,
                last_confirmed_status=Status.AWAIT_FUNDS, next_action="x")
    return AgentResult(**{**base, **kw})


def moved():
    return r(proposed_transition=ProposedTransition(to_status=Status.AWAIT_PAYOUT,
             event_code="incoming_payment_received", basis_source_ids=["MX"]))


def test_card_unchanged_on_reject(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    j.apply_decision(CASE, Decision(action="reject", result=moved()), "run1")
    assert j.card("TEST-20", CASE)["status"] == Status.AWAIT_FUNDS.value


def test_card_changes_on_accept_with_transition(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    j.apply_decision(CASE, Decision(action="accept", result=moved()), "run1")
    assert j.card("TEST-20", CASE)["status"] == Status.AWAIT_PAYOUT.value


def test_review_flag_not_status(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    j.apply_decision(CASE, Decision(action="accept", result=r(requires_review=True)), "run1")
    c = j.card("TEST-20", CASE)
    assert c["status"] == Status.AWAIT_FUNDS.value and c["requires_review"] == 1


def test_no_secrets_in_runs(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    rid = j.start_run(case=CASE, participant="a", mode="assistant", order_no=1, poc_version="v", dataset_version="d")
    j.finish_run(rid, raw_json=json.dumps({"x": 1}), failure=None, secrets=("sk-secret",))
    assert "sk-secret" not in json.dumps(j.runs(), ensure_ascii=False)
    rid2 = j.start_run(case=CASE, participant="a", mode="assistant", order_no=2, poc_version="v", dataset_version="d")
    j.finish_run(rid2, failure="error sk-secret leaked", secrets=("sk-secret",))
    assert "sk-secret" not in json.dumps(j.runs(), ensure_ascii=False)


def test_version_changes_with_prompt(tmp_path):
    s = load_settings(ROOT)
    shutil.copytree(ROOT / "poc", tmp_path / "poc")
    v1 = poc_version(s, tmp_path / "poc")
    p = tmp_path / "poc/prompts/agent_system.txt"
    p.write_text(p.read_text(encoding="utf-8") + "\nX", encoding="utf-8")
    assert poc_version(s, tmp_path / "poc") != v1


def test_edits_diff():
    d = edits_diff(r(), r(next_action="y"))
    assert d == {"next_action": ("x", "y")}
