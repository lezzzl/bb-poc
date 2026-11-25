"""CLI: run одного кейса, batch по набору, eval по эталонам, export-blind для оценщика."""

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from poc.config import load_settings
from poc.dataset import load_case
from poc.evaluate import compare, export_blind
from poc.graph import run_episode
from poc.journal import Journal, dataset_version, poc_version
from poc.llm import create_model
from poc.models import AgentResult


def _episode(S, model, case_id: str, mode: str) -> dict:
    cp = SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False))
    try:
        return run_episode(S, case_id, mode, model, cp, f"cli:{case_id}")
    except Exception as e:
        return {"status": "failed", "error": f"error: {type(e).__name__}", "validated": None, "messages": [],
                "model_latency_s": 0, "tool_calls": 0}


def _record(J: Journal, S, case_id: str, mode: str, out: dict, versions: tuple[str, str]) -> None:
    case = load_case(S.test_dir, case_id, mode)
    rid = J.start_run(case=case, participant="cli", mode=f"batch:{mode}", order_no=None,
                      poc_version=versions[0], dataset_version=versions[1])
    v = out.get("validated") or {}
    calls = [{"name": c["name"], "args": c["args"]} for m in out.get("messages", [])
             for c in (getattr(m, "tool_calls", None) or [])]
    J.finish_run(rid, secrets=(S.api_key,), raw_json=v.get("raw"), checked_json=v.get("checked"),
                 tool_calls_json=calls, model_latency_s=out.get("model_latency_s"),
                 failure=f"{out['error']}: {out.get('error_detail') or ''}" if out.get("error") else None)


def _cases(S, which: str) -> list[str]:
    ids = sorted(p.name for p in (S.test_dir / "cases").iterdir() if p.is_dir())
    if which == "all":
        return ids
    return [c for c in ids if load_case(S.test_dir, c, "prepared").dataset == which]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m poc")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("case"); r.add_argument("--mode", default="with_mail", choices=["with_mail", "prepared"])
    b = sub.add_parser("batch"); b.add_argument("--set", default="development", choices=["holdout", "development", "all"])
    b.add_argument("--mode", default="with_mail", choices=["with_mail", "prepared"]); b.add_argument("--workers", type=int, default=4)
    b.add_argument("--model", help="модель OpenRouter вместо OPENROUTER_MODEL из .env")
    b.add_argument("--effort", choices=["low", "medium", "high"], help="уровень рассуждений модели")
    e = sub.add_parser("eval"); e.add_argument("--poc-version")
    x = sub.add_parser("export-blind"); x.add_argument("out")
    a = p.parse_args(argv)

    S = load_settings()
    if getattr(a, "model", None):
        import dataclasses
        S = dataclasses.replace(S, model=a.model)
    if getattr(a, "effort", None):
        import dataclasses
        S = dataclasses.replace(S, reasoning_effort=a.effort)
    J = Journal(S.data_dir / "poc.sqlite")
    versions = (poc_version(S), dataset_version(S.test_dir))
    if a.cmd == "run":
        out = _episode(S, create_model(S), a.case, a.mode)
        _record(J, S, a.case, a.mode, out, versions)
        print(f"status={out['status']} error={out.get('error')} tools={out['tool_calls']} latency={out['model_latency_s']}s")
        if out.get("validated"):
            print(json.dumps(out["validated"]["checked"], ensure_ascii=False, indent=1))
            for i in out["validated"]["issues"]:
                print(f"! {i['code']}: {i['message']}")
    elif a.cmd == "batch":
        model, cases = create_model(S), _cases(S, a.set)
        with ThreadPoolExecutor(a.workers) as ex:
            results = list(ex.map(lambda c: (c, _episode(S, model, c, a.mode)), cases))
        for c, out in results:
            _record(J, S, c, a.mode, out, versions)
            print(f"{c}: {out['status']} {out.get('error') or ''} {out['model_latency_s']}s")
    elif a.cmd == "eval":
        latest: dict[str, dict] = {}
        for row in J.runs():
            if row["participant"] == "cli" and (not a.poc_version or row["poc_version"] == a.poc_version):
                latest[row["case_id"]] = row
        keys = ["status_ok", "proposed_ok", "review_ok", "mails_found"]
        totals = {k: 0 for k in keys}
        for cid, row in sorted(latest.items()):
            exp = json.loads((S.test_dir / f"reference/mail_pool_states/{cid}/expected.json").read_text(encoding="utf-8"))
            res = AgentResult(**json.loads(row["checked_json"])) if row["checked_json"] else None
            c = compare(res, exp)
            for k in keys:
                totals[k] += c[k]
            flags = " ".join("✓" if c[k] else "✗" for k in keys)
            print(f"{cid} {flags} fail={row['failure'] or '-'} latency={row['model_latency_s']} extra={c['extra_mails']}")
        n = len(latest) or 1
        print("итого (" + str(len(latest)) + "): " + ", ".join(f"{k} {totals[k]}/{len(latest)} ({100 * totals[k] / n:.0f}%)" for k in keys))
    elif a.cmd == "export-blind":
        print(f"выгружено {export_blind(J, Path(a.out))}")


if __name__ == "__main__":
    main()
