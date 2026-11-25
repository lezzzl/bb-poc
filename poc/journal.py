"""Тестовые карточки и журнал замера (SQLite). Одна строка runs = одно выполнение (протокол 2.5.2)."""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from poc.config import PocSettings
from poc.models import AgentResult, Case, Decision

RUN_FIELDS = ["run_id", "dataset_version", "poc_version", "case_id", "category", "participant", "mode", "order_no",
              "started_at", "finished_at", "duration_s", "model_latency_s", "failure", "manual_fallback", "raw_json",
              "checked_json", "submitted_json", "edits_json", "tool_calls_json", "decision", "override",
              "grader_errors", "grader_verdict"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash(parts: list[bytes]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:12]


def poc_version(settings: PocSettings, package_dir: Path | None = None) -> str:
    pkg = package_dir or Path(__file__).resolve().parent
    files = sorted([*pkg.rglob("*.py"), *pkg.glob("prompts/*.txt")])
    return _hash([f.read_bytes() for f in files] +
                 [f"{settings.model}|{settings.timeout_s}|{settings.max_tool_calls}|{settings.reasoning_effort}".encode()])


def dataset_version(test_dir: Path) -> str:
    files = sorted([*test_dir.glob("reference/*.json"), *test_dir.glob("cases/*/*/input.json"),
                    *test_dir.glob("cases/*/*/case.json")])
    return _hash([f.read_bytes() for f in files])


def edits_diff(before: AgentResult, after: AgentResult) -> dict[str, tuple]:
    a, b = before.model_dump(mode="json"), after.model_dump(mode="json")
    return {k: (a[k], b[k]) for k in a if a[k] != b.get(k)}


class Journal:
    def __init__(self, db_path: Path):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("CREATE TABLE IF NOT EXISTS cards (case_id TEXT PRIMARY KEY, status TEXT, "
                        "requires_review INTEGER, run_id TEXT, updated_at TEXT)")
        cols = ", ".join(f"{f} TEXT" if f != "run_id" else "run_id TEXT PRIMARY KEY" for f in RUN_FIELDS)
        self.db.execute(f"CREATE TABLE IF NOT EXISTS runs ({cols})")
        self.db.commit()

    def start_run(self, case: Case, participant: str, mode: str, order_no: int | None,
                  poc_version: str, dataset_version: str) -> str:
        rid = uuid.uuid4().hex[:12]
        self.db.execute("INSERT INTO runs (run_id, dataset_version, poc_version, case_id, category, participant, "
                        "mode, order_no, started_at) VALUES (?,?,?,?,?,?,?,?,?)",
                        (rid, dataset_version, poc_version, case.case_id, case.category, participant, mode,
                         order_no, _now()))
        self.db.commit()
        return rid

    def finish_run(self, run_id: str, secrets: tuple[str, ...] = (), **fields) -> None:
        fields.setdefault("finished_at", _now())
        row = self.db.execute("SELECT started_at FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row and "duration_s" not in fields:
            start = datetime.fromisoformat(row["started_at"])
            fields["duration_s"] = round((datetime.fromisoformat(fields["finished_at"]) - start).total_seconds(), 1)
        clean = {}
        for k, v in fields.items():
            if k not in RUN_FIELDS:
                raise KeyError(k)
            if v is not None and not isinstance(v, str):
                v = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            for s in secrets:
                if s and v:
                    v = v.replace(s, "***")
            clean[k] = v
        sets = ", ".join(f"{k}=?" for k in clean)
        self.db.execute(f"UPDATE runs SET {sets} WHERE run_id=?", (*clean.values(), run_id))
        self.db.commit()

    def apply_decision(self, case: Case, decision: Decision, run_id: str) -> None:
        status = self.card(case.case_id, case)["status"]
        pt = decision.result.proposed_transition
        if decision.action != "reject" and pt is not None:
            status = pt.to_status.value
        review = int(decision.result.requires_review) if decision.action != "reject" else 1
        self.db.execute("INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?)", (case.case_id, status, review, run_id, _now()))
        self.db.commit()

    def card(self, case_id: str, case: Case | None = None) -> dict:
        row = self.db.execute("SELECT * FROM cards WHERE case_id=?", (case_id,)).fetchone()
        if row:
            return dict(row)
        return {"case_id": case_id, "status": case.card_status.value if case else None, "requires_review": 0,
                "run_id": None, "updated_at": None}

    def runs(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM runs ORDER BY started_at")]
