"""Загрузка кейса (оба режима), общего ящика, справочников, текста PDF и отметок подписей."""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pypdf import PdfReader

from poc.models import Case, Material, SignatureMarks
from poc.transitions import Rule, load_rules

SIGNED = "Подписано · тестовая отметка"
UNSIGNED = ("Подпись отсутствует", "Ожидается подпись")
SIGNABLE = {"act", "application"}


class Reference(BaseModel):
    rules: list[Rule]
    roles: list[dict]
    companies: dict[str, dict]


def _json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_reference(test_dir: Path) -> Reference:
    ref = test_dir / "reference"
    return Reference(rules=load_rules(ref / "transitions.json"), roles=_json(ref / "roles.json"),
                     companies=_json(ref / "companies.json"))


def pdf_text(path: Path) -> str:
    try:
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    except Exception:
        return ""


def parse_signatures(text: str) -> SignatureMarks | None:
    marks: dict[str, bool] = {}
    party = None
    for line in text.splitlines():
        s = line.strip()
        if "Поставщик услуг /" in s:
            party = "provider"
        elif "Принципал /" in s:
            party = "principal"
        if party is None:
            continue
        if SIGNED in s:
            marks[party] = True
        elif any(s.startswith(u) or u in s for u in UNSIGNED):
            marks[party] = False
    if party is None:
        return None
    return SignatureMarks(**marks)


def doc_kind(file_name: str, declared: str | None = None) -> str:
    if declared:
        return declared
    n = file_name.lower()
    if n.startswith("act"):
        return "act"
    if "application" in n:
        return "application"
    if n.startswith("payment_confirmation"):
        return "payment_confirmation"
    if "invoice" in n:
        return "invoice"
    return "attachment"


def _file_material(source_id: str, path: Path, kind: str, at: datetime, sender=None, recipient=None,
                   parent_id=None, subject=None) -> Material:
    text = pdf_text(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    sig = parse_signatures(text) if kind in SIGNABLE else None
    if not text.strip():
        text = "[текст не извлечён]"
    return Material(source_id=source_id, kind=kind, at=at, sender=sender, recipient=recipient, subject=subject,
                    text=text, path=path, parent_id=parent_id, signatures=sig)


def _parse_email_txt(path: Path, source_id: str) -> Material:
    head, _, body = path.read_text(encoding="utf-8").partition("\n\n")
    h = {}
    for line in head.splitlines():
        k, _, v = line.partition(":")
        h[k.strip()] = v.strip()
    return Material(source_id=source_id, kind="email", at=datetime.fromisoformat(h["Дата"]), sender=h.get("От"),
                    recipient=h.get("Кому"), subject=h.get("Тема"), text=f"{h.get('Тема', '')}\n{body.strip()}",
                    path=path)


def load_case(test_dir: Path, case_id: str, mode: Literal["with_mail", "prepared"]) -> Case:
    base = test_dir / "cases" / case_id
    meta = _json(base / "without_mail_search" / "case.json")
    if mode == "with_mail":
        folder = base / "with_mail_search"
        inp = _json(folder / "input.json")
        crm_path = folder / inp["crm_snapshot"]
        docs = inp["documents"]
    else:
        folder = base / "without_mail_search"
        inp = meta
        crm_path = next(folder / s["path"] for s in meta["sources"] if s["kind"] == "crm_snapshot")
        docs = [s for s in meta["sources"] if s["kind"] != "crm_snapshot"]
    crm = _json(crm_path)
    start = [Material(source_id="CRM", kind="crm_snapshot", at=datetime.fromisoformat(crm["recorded_at"]),
                      text=json.dumps(crm, ensure_ascii=False, indent=1), path=crm_path)]
    for d in docs:
        p = folder / d["path"]
        if d["kind"] == "email":
            start.append(_parse_email_txt(p, d["source_id"]))
        else:
            start.append(_file_material(d["source_id"], p, doc_kind(p.name, d["kind"]),
                                        datetime.fromisoformat(d["available_at"]), sender=d.get("source_address")))
    name = crm["client"]["name"]
    short = name.split("«")[-1].rstrip("»") if "«" in name else name
    return Case(case_id=case_id, application_id=crm["application_id"], invoice_id=crm.get("invoice_id"),
                client_id=crm["client"]["company_id"], client_email=crm["client"]["email"], client_name=short,
                cutoff_at=datetime.fromisoformat(inp["cutoff_at"]), card_status=crm["status"],
                dataset=meta.get("dataset"), category=meta.get("category"), mode=mode, start=start)


def load_mailbox(test_dir: Path) -> list[Material]:
    pool = test_dir / "mail_pool"
    out: list[Material] = []
    for entry in _json(pool / "index.json"):
        m = _json(pool / entry["input_path"])
        at = datetime.fromisoformat(m["date"])
        out.append(Material(source_id=m["mail_id"], kind="email", at=at, sender=m["from"], recipient=m["to"],
                            subject=m["subject"], text=f"{m['subject']}\n{m['text']}",
                            path=pool / entry["message_path"]))
        for rel in entry.get("attachment_paths", []):
            p = pool / rel
            out.append(_file_material(f"{m['mail_id']}/{p.name}", p, doc_kind(p.name), at, m["from"], m["to"],
                                      parent_id=m["mail_id"], subject=m["subject"]))
    return out
