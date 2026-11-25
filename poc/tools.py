"""Инструменты агента. Работают только по индексу области; содержимое отдаётся как данные."""

import re

from langchain_core.tools import BaseTool, tool

from poc.index import ScopeIndex
from poc.models import Case, Material, Status

EVIDENCE_RU = {
    "crm_record": "запись в CRM",
    "application_sent_documents": "письмо сотрудника (desk@) клиенту с заявкой во вложении",
    "application_signed_by_both": "заявка с подписями обеих сторон",
    "manager_message": "сообщение руководителя (manager@) с номером заявки или инвойса",
    "manager_confirmation_file": "файл-подтверждение от руководителя",
    "transfer_document": "документ о переводе (платёжка) от руководителя или из папки заявки",
    "client_confirmation": "письмо клиента этой заявки",
    "act_signed_by_both": "акт с подписями обеих сторон",
}


def _mark(v: bool | None) -> str:
    return "есть" if v is True else "нет" if v is False else "не найдена отметка"


def _attrs(m: Material) -> str:
    parts = [f'source_id="{m.source_id}"', f'kind="{m.kind}"', f'at="{m.at.isoformat()}"']
    if m.sender:
        parts.append(f'from="{m.sender}"')
    if m.recipient:
        parts.append(f'to="{m.recipient}"')
    if m.parent_id:
        parts.append(f'attached_to="{m.parent_id}"')
    return " ".join(parts)


def wrap(m: Material, body: str) -> str:
    return f"<data {_attrs(m)}>\n{body}\n</data>"


def _signature_line(m: Material) -> str:
    s = m.signatures
    if s is None:
        return "блоки подписей в тексте не найдены."
    return (f"{'заявка' if m.kind == 'application' else 'акт'}: принципал: {_mark(s.principal)}; "
            f"исполнитель: {_mark(s.provider)}; подписан обеими сторонами: {'ДА' if s.both else 'НЕТ'}.")


def make_tools(index: ScopeIndex, case: Case | None = None, ref=None) -> list[BaseTool]:
    @tool
    def list_sources() -> str:
        """Перечень всех материалов области заявки: id, тип, время, отправитель, тема."""
        return "\n".join(f"{m.source_id} | {m.kind} | {m.at.isoformat()} | {m.sender or '-'} | {m.subject or '-'}"
                         for m in index.materials())

    @tool
    def search(query: str, kind: str | None = None, sender: str | None = None) -> str:
        """Полнотекстовый поиск по материалам области. query — слова или номера (TEST-APP-..., TEST-INV-...).
        kind — фильтр типа (email, invoice, application, act, payment_confirmation, crm_snapshot), sender — email."""
        hits = index.search(query, kind=kind, sender=sender)
        if not hits:
            return "Ничего не найдено."
        return "\n".join(wrap(index.get(h["source_id"]), h["snippet"]) for h in hits)

    @tool
    def get(source_id: str) -> str:
        """Полный текст материала области по source_id."""
        m = index.get(source_id)
        if m is None:
            return f"Нет такого источника в области заявки: {source_id}"
        out = wrap(m, m.text)
        if m.kind in ("application", "act", "payment_confirmation"):
            ids = sorted(set(re.findall(r"[A-Z]+(?:-[A-Z0-9]+)*-(?:APP|INV)-[A-Z0-9-]+", m.text)))
            out += "\nНомера в самом документе (проверка кода): " + (", ".join(ids) if ids else "не найдены")
        if m.kind in ("application", "act"):
            out += "\nПодписи (проверка кода): " + _signature_line(m)
        return out

    @tool
    def signatures(source_id: str) -> str:
        """Подписи на заявке или акте по отметкам в документе (определяет код, не модель).
        Используй перед тем, как считать документ подписанным обеими сторонами."""
        m = index.get(source_id)
        if m is None:
            return f"Нет такого источника в области заявки: {source_id}"
        if m.kind not in ("application", "act"):
            return f"{source_id}: это не заявка и не акт ({m.kind}), подписи не проверяются."
        return f"{source_id}: " + _signature_line(m)

    tools = [list_sources, search, get, signatures]
    if case is not None and ref is not None:
        @tool
        def next_step_rule() -> str:
            """Какой переход возможен из текущего статуса CRM и какое основание для него нужно (таблица 2.10)."""
            rule = next((r for r in ref.rules if r.from_status == case.card_status), None)
            if rule is None:
                return f"Статус в CRM «{case.card_status.value}»: заявка завершена, переходов дальше нет."
            ev = "; ".join(EVIDENCE_RU.get(e, e) for e in rule.evidence_types)
            return (f"Статус в CRM: «{rule.from_status.value}». Следующий этап: «{rule.to_status.value}» "
                    f"(event_code={rule.event_code}). "
                    f"Подходящее основание (любое из): {ev}. Основание должно прямо относиться к заявке "
                    f"{case.application_id}" + (f" / инвойсу {case.invoice_id}" if case.invoice_id else "") + ".")
        tools.append(next_step_rule)
    return tools
