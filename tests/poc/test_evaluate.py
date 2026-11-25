import json
from pathlib import Path
from poc.dataset import load_case
from poc.evaluate import compare, export_blind
from poc.journal import Journal
from poc.models import AgentResult, Fact, Status

T = Path(__file__).resolve().parents[2] / "TEST"
EXP = json.loads((T / "reference/mail_pool_states/TEST-20/expected.json").read_text())


def test_compare_matches_expected():
    r = AgentResult(case_id="TEST-20", application_id="TEST-APP-102", card_status=Status.AWAIT_FUNDS,
                    last_confirmed_status=Status.AWAIT_FUNDS, facts=[Fact(fact="x", source_ids=["M019"])],
                    requires_review=True, next_action="x", used_mail_ids=["M019"])
    c = compare(r, EXP)
    assert c["status_ok"] and c["proposed_ok"] and c["review_ok"] and c["mails_found"]


def test_compare_none_result_counts_as_failure():
    c = compare(None, EXP)
    assert not any(c[k] for k in ("status_ok", "proposed_ok", "review_ok", "mails_found"))


def test_export_blind_has_no_participant_or_mode(tmp_path):
    j = Journal(tmp_path / "j.sqlite")
    rid = j.start_run(case=load_case(T, "TEST-20", "with_mail"), participant="Максим", mode="manual", order_no=1,
                      poc_version="v", dataset_version="d")
    j.finish_run(rid, submitted_json={"a": 1})
    out = tmp_path / "blind.jsonl"
    export_blind(j, out)
    row = json.loads(out.read_text().splitlines()[0])
    assert set(row) == {"run_id", "case_id", "submitted"} and "Максим" not in out.read_text()
