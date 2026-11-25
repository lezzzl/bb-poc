"""Общая форма результата для режима с помощником и ручного режима.

Списки в форме — обычный текст, по одному пункту в строке:
- факты: «Клиент сообщил о переводе [M019]» — источники в квадратных скобках через запятую;
- недостающее: «что нужно — зачем»;
- конфликты: «CRM | M041 | в чём расхождение».
"""

import re

from poc.models import AgentResult, Case, Conflict, Fact, Missing

EVENT_FOR_STATUS = {
    "Ожидаем подписания заявки": "application_sent",
    "Ожидаем поступления денег": "application_signed",
    "Ожидаем отправки платежа": "incoming_payment_received",
    "Платёж отправлен": "outgoing_payment_executed",
    "Оформление и подписание акта": "counterparty_received_payment",
    "Заявка завершена": "act_signed",
}
_SRC = re.compile(r"\[([^\]]*)\]\s*$")


def empty_result(case: Case) -> AgentResult:
    return AgentResult(case_id=case.case_id, application_id=case.application_id, card_status=case.card_status,
                       last_confirmed_status=case.card_status, next_action="")


def _lines(text: str) -> list[str]:
    return [x.strip() for x in (text or "").splitlines() if x.strip()]


def parse_facts(text: str) -> list[Fact]:
    out = []
    for line in _lines(text):
        m = _SRC.search(line)
        ids = [s.strip() for s in m.group(1).split(",") if s.strip()] if m else []
        out.append(Fact(fact=line[: m.start()].strip() if m else line, source_ids=ids))
    return out


def facts_to_text(facts: list[Fact]) -> str:
    return "\n".join(f"{f.fact} [{', '.join(f.source_ids)}]" if f.source_ids else f.fact for f in facts)


def parse_missing(text: str) -> list[Missing]:
    return [Missing(what=a.strip(), why=b.strip()) for a, _, b in (l.partition(" — ") for l in _lines(text))]


def parse_conflicts(text: str) -> list[Conflict]:
    out = []
    for line in _lines(text):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[0] and parts[1]:
            out.append(Conflict(source_a=parts[0], source_b=parts[1], note=" | ".join(parts[2:])))
    return out


def result_to_form(r: AgentResult) -> dict:
    pt = r.proposed_transition
    return {"last_confirmed_status": r.last_confirmed_status.value,
            "transition_to": pt.to_status.value if pt else "",
            "basis_source_ids": list(pt.basis_source_ids) if pt else [],
            "facts": facts_to_text(r.facts),
            "missing": "\n".join(f"{m.what} — {m.why}" if m.why else m.what for m in r.missing),
            "conflicts": "\n".join(f"{c.source_a} | {c.source_b} | {c.note}" for c in r.conflicts),
            "requires_review": r.requires_review,
            "review_triggers": list(r.review_triggers),
            "review_reasons": "\n".join(r.review_reasons),
            "next_action": r.next_action,
            "clarification": r.clarification or "",
            "used_mail_ids": list(r.used_mail_ids)}


def form_to_result(v: dict, case: Case) -> AgentResult:
    to = v.get("transition_to") or ""
    pt = {"to_status": to, "event_code": EVENT_FOR_STATUS[to], "basis_source_ids": v.get("basis_source_ids", [])} if to else None
    return AgentResult(case_id=case.case_id, application_id=case.application_id, card_status=case.card_status,
                       last_confirmed_status=v["last_confirmed_status"], facts=parse_facts(v.get("facts", "")),
                       proposed_transition=pt, requires_review=bool(v.get("requires_review")),
                       review_triggers=v.get("review_triggers", []),
                       review_reasons=_lines(v.get("review_reasons", "")), missing=parse_missing(v.get("missing", "")),
                       conflicts=parse_conflicts(v.get("conflicts", "")), next_action=v.get("next_action", ""),
                       clarification=(v.get("clarification") or "").strip() or None,
                       used_mail_ids=v.get("used_mail_ids", []))
