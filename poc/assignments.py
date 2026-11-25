"""Назначения замера из TEST/assignments.csv (готовит Никита)."""

import csv
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class Assignment(BaseModel):
    participant: str
    order_no: int
    case_id: str
    mode: Literal["manual", "assistant"]


def load_assignments(path: Path) -> list[Assignment] | None:
    if not Path(path).exists():
        return None
    with open(path, encoding="utf-8") as f:
        return [Assignment(**row) for row in csv.DictReader(f)]


def next_assignment(assignments: list[Assignment], runs: list[dict], participant: str) -> Assignment | None:
    done = {r["case_id"] for r in runs if r.get("participant") == participant and r.get("finished_at")}
    for a in sorted((a for a in assignments if a.participant == participant), key=lambda a: a.order_no):
        if a.case_id not in done:
            return a
    return None
