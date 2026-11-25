"""Программная проверка результата агента (спека 5.5). Модель не может обойти эти правила."""

from poc.dataset import Reference
from poc.models import AgentResult, Case, Issue, Material, Validated
from poc.transitions import rule_for

import re

MANAGER_EVENTS = {"incoming_payment_received", "outgoing_payment_executed"}
_IDS = re.compile(r"TEST-(?:APP|INV)-[0-9]+(?:-[A-Z0-9]+)?")


def _role(email: str | None, ref: Reference) -> dict | None:
    return next((r for r in ref.roles if r.get("email") == email), None)


def _client_emails(ref: Reference) -> set[str]:
    return {c["email"] for c in ref.companies.values() if c.get("role") == "Клиент"}


def evidence_types(m: Material, case: Case, ref: Reference, scope: list[Material] | None = None) -> set[str]:
    types: set[str] = set()
    if m.kind == "crm_snapshot":
        return {"crm_record"}
    role = _role(m.sender, ref) or {}
    confirms = set(role.get("can_confirm", []))
    if m.signatures and m.signatures.both:
        types.add("application_signed_by_both" if m.kind == "application" else "act_signed_by_both")
    if role.get("role") == "employee" and m.recipient == case.client_email:
        sent_app = m.kind == "application" or any(
            a.parent_id == m.source_id and a.kind == "application" for a in (scope or []))
        if sent_app:
            types.add("application_sent_documents")
    if confirms & MANAGER_EVENTS:
        if m.parent_id is None and m.kind == "email":
            types.add("manager_message")
        else:
            types.add("manager_confirmation_file")
        if m.kind == "payment_confirmation":
            types.add("transfer_document")
    if m.kind == "payment_confirmation" and m.source_id in {s.source_id for s in case.start}:
        types.add("transfer_document")
    if m.sender == case.client_email and m.kind == "email":
        types.add("client_confirmation")
    return types


DOCUMENT_KINDS = {"application", "act", "payment_confirmation"}


def _own_text_mentions(m: Material, case: Case) -> bool:
    """Принадлежность документа (2.3.5): номер заявки или инвойса в тексте самого документа, а не письма."""
    return any(k in m.text for k in (case.application_id, case.invoice_id) if k)


def _mentions_case(m: Material, case: Case, by_id: dict[str, Material]) -> bool:
    keys = [k for k in (case.application_id, case.invoice_id) if k]
    texts = [m.text, m.subject or ""]
    if m.parent_id and m.parent_id in by_id:
        texts.append(by_id[m.parent_id].text)
    return any(k in t for k in keys for t in texts)


def _other_request(m: Material, case: Case, by_id: dict[str, Material]) -> bool:
    """Материал прямо про другую заявку: в нём есть номера заявок/инвойсов, но ни одного нашего."""
    if m.kind == "crm_snapshot" or m.source_id in {s.source_id for s in case.start}:
        return False
    text = m.text + " " + (m.subject or "")
    if m.parent_id and m.parent_id in by_id:
        text += " " + by_id[m.parent_id].text
    found = set(_IDS.findall(text))
    ours = {k for k in (case.application_id, case.invoice_id) if k}
    return bool(found) and not (found & ours)


def _mail_id(source_id: str, by_id: dict[str, Material]) -> str | None:
    m = by_id.get(source_id)
    if m is None or m.kind == "crm_snapshot":
        return None
    if m.parent_id:
        return m.parent_id
    return source_id if source_id[:1] == "M" and source_id[1:].isdigit() else None


def validate(result: AgentResult, case: Case, scope: list[Material], ref: Reference) -> Validated:
    raw = result.model_copy(deep=True)
    r = result.model_copy(deep=True)
    by_id = {m.source_id: m for m in scope}
    issues: list[Issue] = []
    reasons: list[str] = []

    def flag(code: str, msg: str) -> None:
        issues.append(Issue(code=code, message=msg))
        reasons.append(msg)

    foreign = _client_emails(ref) - {case.client_email}

    def clean(ids: list[str], where: str) -> list[str]:
        out = []
        for sid in ids:
            m = by_id.get(sid)
            if m is not None and _other_request(m, case, by_id):
                issues.append(Issue(code="other_request", message=f"{where}: {sid} относится к другой заявке, ссылка убрана"))
            elif m is None:
                flag("unknown_source", f"{where}: ссылка {sid} вне области заявки, удалена")
            elif m.sender in foreign:
                flag("foreign_client", f"{where}: {sid} от другого клиента ({m.sender}), удалена")
            else:
                out.append(sid)
        return out

    if r.application_id != case.application_id or r.case_id != case.case_id:
        flag("wrong_application", f"Результат указывает {r.application_id}, кейс — {case.application_id}")
        r.application_id, r.case_id = case.application_id, case.case_id
    r.card_status = case.card_status

    kept_facts = []
    for f in r.facts:
        had = bool(f.source_ids)
        f.source_ids = clean(f.source_ids, "Факт")
        if f.source_ids or not had:
            kept_facts.append(f)
    r.facts = kept_facts
    kept_conflicts = []
    for c in r.conflicts:
        ids = clean([c.source_a, c.source_b], "Конфликт")
        if len(ids) == 2:
            kept_conflicts.append(c)
    r.conflicts = kept_conflicts

    drop_transition = False
    if r.last_confirmed_status != case.card_status:
        if not any("CRM" in (c.source_a, c.source_b) for c in r.conflicts):
            flag("status_mismatch_without_conflict",
                 f"Подтверждённый этап «{r.last_confirmed_status.value}» расходится с CRM без указанного конфликта")
        else:
            reasons.append("CRM и материалы расходятся; статус исправляет сотрудник по правилам компании")
        drop_transition = True

    pt = r.proposed_transition
    if pt is not None:
        pt.basis_source_ids = clean(pt.basis_source_ids, "Основание")
        rule = rule_for(ref.rules, case.card_status, pt.event_code, pt.to_status)
        if rule is None:
            flag("transition_not_allowed",
                 f"Переход «{case.card_status.value}» → «{pt.to_status.value}» ({pt.event_code}) не разрешён таблицей 2.10")
            drop_transition = True
        else:
            basis = [by_id[s] for s in pt.basis_source_ids]
            typed = [m for m in basis if set(rule.evidence_types) & evidence_types(m, case, ref, scope)]
            if not typed:
                if not basis:
                    flag("basis_wrong_type", "Нет основания для перехода")
                elif rule.event_code in MANAGER_EVENTS and any(m.kind in ("email", "payment_confirmation") for m in basis):
                    flag("basis_wrong_sender", "Поступление и платёж подтверждает только руководитель")
                else:
                    flag("basis_wrong_type", f"Основание не подходит: нужно одно из {rule.evidence_types}")
                drop_transition = True
            elif all(m.kind in DOCUMENT_KINDS and not _own_text_mentions(m, case) for m in typed):
                flag("document_other_request", "Документ-основание не содержит номера этой заявки или инвойса — "
                                               "возможно, он относится к другой заявке")
                drop_transition = True
            elif not any(m.kind == "crm_snapshot" or _mentions_case(m, case, by_id) for m in typed):
                flag("basis_without_id", "В основании нет номера этой заявки или инвойса")
                drop_transition = True
    if r.proposed_transition is not None and r.conflicts and not drop_transition:
        flag("conflict_blocks_transition", "Есть противоречие в материалах: переход не предлагается до проверки сотрудником")
        drop_transition = True
    if drop_transition and r.proposed_transition is not None:
        r.proposed_transition = None
    if r.proposed_transition is None and r.missing and "missing_confirmation" not in r.review_triggers:
        r.review_triggers = [*r.review_triggers, "missing_confirmation"]  # модель сама указала нехватку данных
    if r.conflicts and "conflict" not in r.review_triggers:
        r.review_triggers = [*r.review_triggers, "conflict"]
    # Пометка ставится только по причине из закрытого списка или по замечанию проверки кода
    r.requires_review = bool(r.review_triggers) or bool(reasons)
    if reasons:
        r.review_reasons = [*r.review_reasons, *reasons]

    ids = [s for f in r.facts for s in f.source_ids]
    ids += [s for c in r.conflicts for s in (c.source_a, c.source_b)]
    ids += r.proposed_transition.basis_source_ids if r.proposed_transition else []
    used = []
    for sid in ids:
        mid = _mail_id(sid, by_id)
        if mid and mid not in used:
            used.append(mid)
    r.used_mail_ids = used
    return Validated(raw=raw, checked=r, issues=issues)
