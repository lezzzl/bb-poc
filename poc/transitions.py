"""Таблица 2.10: разрешённые переходы статусов и типы оснований."""

import json
from pathlib import Path

from pydantic import BaseModel

from poc.models import Status


class Rule(BaseModel):
    id: str
    from_status: Status
    to_status: Status
    event_code: str
    evidence_types: list[str]


def load_rules(path: Path) -> list[Rule]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Rule(id=t["id"], from_status=t["from"], to_status=t["to"], event_code=t["event_code"],
                 evidence_types=t["evidence_types"]) for t in data["transitions"]]


def rule_for(rules: list[Rule], from_status: Status, event_code: str, to_status: Status) -> Rule | None:
    for r in rules:
        if r.from_status == from_status and r.event_code == event_code and r.to_status == to_status:
            return r
    return None
