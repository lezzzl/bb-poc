"""Область доступа: что модель вообще может увидеть по кейсу. Выполняется кодом до модели."""

from poc.models import Case, Material


def _keys(case: Case) -> list[str]:
    return [k for k in (case.application_id, case.invoice_id, case.client_name) if k]


def build_scope(case: Case, mailbox: list[Material], foreign_emails: frozenset[str] = frozenset()) -> list[Material]:
    """foreign_emails — адреса других клиентов: их письма в область не попадают никогда."""
    if case.mode == "prepared":
        return list(case.start)
    keys = _keys(case)
    letters = [m for m in mailbox if m.parent_id is None and m.at <= case.cutoff_at
               and not ({m.sender, m.recipient} & foreign_emails) and (
        case.client_email in (m.sender, m.recipient) or any(k in m.text for k in keys))]
    ids = {m.source_id for m in letters}
    attachments = [m for m in mailbox if m.parent_id in ids]
    return [*case.start, *letters, *attachments]
