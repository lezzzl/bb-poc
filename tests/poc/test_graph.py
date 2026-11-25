import sqlite3
from pathlib import Path
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from poc.config import load_settings
from poc.graph import run_episode, resume_episode
from poc.models import AgentResult, Decision, ProposedTransition, Status
from fakes import HangingModel, ScriptedModel, search_call

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def settings(tmp_path, monkeypatch):
    s = load_settings(ROOT)
    return s.__class__(**{**{f: getattr(s, f) for f in s.__slots__}, "data_dir": tmp_path})


def saver(tmp_path):
    return SqliteSaver(sqlite3.connect(tmp_path / "cp.sqlite", check_same_thread=False))


def result20(**kw):
    base = dict(case_id="TEST-20", application_id="TEST-APP-102", card_status=Status.AWAIT_FUNDS,
                last_confirmed_status=Status.AWAIT_FUNDS, facts=[{"fact": "Клиент сообщил о переводе", "source_ids": ["M019"]}],
                requires_review=True, review_triggers=["missing_confirmation"],
                next_action="Уточнить у руководителя поступление денег")
    return AgentResult(**{**base, **kw})


def test_graph_stops_at_review(settings, tmp_path):
    m = ScriptedModel(script=[search_call], structured=[result20()])
    out = run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t1")
    assert out["status"] == "review"
    assert out["validated"]["checked"]["requires_review"] is True


def test_resume_from_new_graph_object(settings, tmp_path):
    m = ScriptedModel(script=[search_call], structured=[result20()])
    run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t2")
    n = dict(m.calls)
    out = resume_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t2",
                         Decision(action="accept", result=result20()))
    assert out["status"] == "done" and out["decision"]["action"] == "accept"
    assert m.calls == n


def test_tool_limit(settings, tmp_path):
    m = ScriptedModel(script=[search_call] * 50, structured=[result20()])
    out = run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t3")
    assert out["status"] == "review" and out["tool_calls"] == settings.max_tool_calls


def test_invalid_output_retry_then_fail(settings, tmp_path):
    m = ScriptedModel(script=[], structured=[{"bad": 1}, {"bad": 2}])
    out = run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t4")
    assert out["status"] == "failed" and out["error"] == "invalid_output"
    assert m.calls["structured"] == 2


def test_timeout(settings, tmp_path):
    s = settings.__class__(**{**{f: getattr(settings, f) for f in settings.__slots__}, "timeout_s": 1})
    out = run_episode(s, "TEST-20", "with_mail", HangingModel(structured=[result20()]), saver(tmp_path), "t5")
    assert out["status"] == "failed" and out["error"] == "timeout"


def test_edit_rechecked(settings, tmp_path):
    m = ScriptedModel(script=[], structured=[result20()])
    run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t6")
    bad = result20(proposed_transition=ProposedTransition(to_status=Status.AWAIT_PAYOUT,
                   event_code="incoming_payment_received", basis_source_ids=["M019"]))
    out = resume_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t6",
                         Decision(action="accept_edited", result=bad))
    assert out["status"] == "review"
    assert {i["code"] for i in out["validated"]["issues"]} & {"basis_wrong_sender", "basis_wrong_type"}
    out2 = resume_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t6",
                          Decision(action="accept_edited", result=bad, override=True))
    assert out2["status"] == "done" and out2["decision"]["override"] is True


def test_invalid_output_keeps_error_detail(settings, tmp_path):
    m = ScriptedModel(script=[], structured=[{"bad": 1}, {"bad": 2}])
    out = run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t7")
    assert out["error"] == "invalid_output" and "validation" in (out["error_detail"] or "").lower()


def test_rerun_does_not_call_model_again(settings, tmp_path):
    m = ScriptedModel(script=[search_call], structured=[result20()])
    run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t8")
    n = dict(m.calls)
    out = run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t8")
    assert out["status"] == "review" and m.calls == n


def test_edit_saves_checked_result(settings, tmp_path):
    from poc.models import Fact
    m = ScriptedModel(script=[], structured=[result20()])
    run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t9")
    edited = result20(facts=[Fact(fact="x", source_ids=["M999", "M019"])])
    out = resume_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t9",
                         Decision(action="accept_edited", result=edited))
    assert out["decision"]["result"]["facts"][0]["source_ids"] == ["M019"]


def test_edit_status_mismatch_blocks_transition(settings, tmp_path):
    m = ScriptedModel(script=[], structured=[result20()])
    run_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t10")
    bad = result20(last_confirmed_status=Status.AWAIT_PAYOUT, proposed_transition=ProposedTransition(
        to_status=Status.PAID_OUT, event_code="outgoing_payment_executed", basis_source_ids=["CRM"]))
    out = resume_episode(settings, "TEST-20", "with_mail", m, saver(tmp_path), "t10",
                         Decision(action="accept_edited", result=bad))
    assert out["status"] == "review"
