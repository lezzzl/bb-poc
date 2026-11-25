"""Модели данных PoC: материалы, кейс, результат агента, проверка, решение сотрудника."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import json

from pydantic import BaseModel, Field, field_validator


class Status(str, Enum):
    NEW = "Новая заявка"
    AWAIT_SIGN = "Ожидаем подписания заявки"
    AWAIT_FUNDS = "Ожидаем поступления денег"
    AWAIT_PAYOUT = "Ожидаем отправки платежа"
    PAID_OUT = "Платёж отправлен"
    ACT = "Оформление и подписание акта"
    DONE = "Заявка завершена"


class SignatureMarks(BaseModel):
    principal: bool | None = None
    provider: bool | None = None

    @property
    def both(self) -> bool:
        return self.principal is True and self.provider is True


class Material(BaseModel):
    source_id: str
    kind: str
    at: datetime
    sender: str | None = None
    recipient: str | None = None
    subject: str | None = None
    text: str = ""
    path: Path | None = None
    parent_id: str | None = None
    signatures: SignatureMarks | None = None


class Case(BaseModel):
    case_id: str
    application_id: str
    invoice_id: str | None = None
    client_id: str
    client_email: str
    client_name: str
    cutoff_at: datetime
    card_status: Status
    dataset: str | None = None
    category: str | None = None
    mode: str = "with_mail"
    start: list[Material]


class Fact(BaseModel):
    fact: str
    source_ids: list[str] = Field(default_factory=list)


EventCode = Literal["application_sent", "application_signed", "incoming_payment_received",
                    "outgoing_payment_executed", "counterparty_received_payment", "act_signed"]


ReviewTrigger = Literal["missing_confirmation", "conflict", "ambiguous_binding", "incomplete_signature",
                        "refused_or_returned", "wrong_status"]
TRIGGER_RU = {
    "missing_confirmation": "нет подтверждения нужного типа",
    "conflict": "противоречие в материалах",
    "ambiguous_binding": "неясно, к какой заявке относится",
    "incomplete_signature": "не хватает подписи",
    "refused_or_returned": "отказ, возврат или отклонение",
    "wrong_status": "статус в CRM опровергается",
}


class ProposedTransition(BaseModel):
    to_status: Status
    event_code: EventCode
    basis_source_ids: list[str] = Field(default_factory=list)


class Missing(BaseModel):
    what: str
    why: str = ""


class Conflict(BaseModel):
    source_a: str
    source_b: str
    note: str


class AgentResult(BaseModel):
    case_id: str
    application_id: str
    card_status: Status
    last_confirmed_status: Status
    facts: list[Fact] = Field(default_factory=list)
    proposed_transition: ProposedTransition | None = None
    requires_review: bool = False
    review_triggers: list[ReviewTrigger] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    missing: list[Missing] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    next_action: str
    clarification: str | None = None
    used_mail_ids: list[str] = Field(default_factory=list)

    @field_validator("facts", "review_triggers", "review_reasons", "missing", "conflicts", "used_mail_ids", mode="before")
    @classmethod
    def _json_string_list(cls, v):
        """Модель иногда возвращает список строкой с JSON внутри — разбираем его."""
        if isinstance(v, str):
            try:
                return json.loads(v) if v.strip() else []
            except json.JSONDecodeError:
                return [v] if v.strip() else []
        return v


class Issue(BaseModel):
    code: str
    message: str


class Validated(BaseModel):
    raw: AgentResult
    checked: AgentResult
    issues: list[Issue] = Field(default_factory=list)


class Decision(BaseModel):
    action: Literal["accept", "accept_edited", "reject"]
    result: AgentResult
    override: bool = False
