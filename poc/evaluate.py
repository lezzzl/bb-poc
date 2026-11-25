"""Сверка с эталонами expected.json (для отладки) и слепой экспорт сданных результатов."""

import json
from pathlib import Path

from poc.journal import Journal
from poc.models import AgentResult


def compare(result: AgentResult | None, expected: dict) -> dict:
    if result is None:
        return {"status_ok": False, "proposed_ok": False, "review_ok": False, "mails_found": False, "extra_mails": []}
    proposed = result.proposed_transition.to_status.value if result.proposed_transition else None
    used = set(result.used_mail_ids)
    exp_mails = set(expected["expected_mail_ids"])
    return {"status_ok": result.last_confirmed_status.value == expected["last_confirmed_status"],
            "proposed_ok": proposed == expected["proposed_status"],
            "review_ok": result.requires_review == expected["requires_review"],
            "mails_found": exp_mails <= used,
            "extra_mails": sorted(used - exp_mails)}


def export_blind(journal: Journal, out: Path) -> int:
    rows = [r for r in journal.runs() if r.get("submitted_json")]
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"run_id": r["run_id"], "case_id": r["case_id"],
                                "submitted": json.loads(r["submitted_json"])}, ensure_ascii=False) + "\n")
    return len(rows)
