from pathlib import Path
from poc.dataset import load_case, load_mailbox, parse_signatures, pdf_text
from poc.models import Status

T = Path(__file__).resolve().parents[2] / "TEST"


def test_load_case_test01_with_mail():
    c = load_case(T, "TEST-01", "with_mail")
    assert c.application_id == "TEST-APP-113"
    assert c.card_status == Status("Новая заявка")
    assert {m.source_id for m in c.start} == {"CRM", "INV"}
    assert c.client_email == "client-spectrum@example.com"


def test_load_case_test02_prepared():
    c = load_case(T, "TEST-02", "prepared")
    emails = [m for m in c.start if m.kind == "email"]
    assert emails and emails[0].sender == "manager@example.com"
    assert c.dataset in {"holdout", "development"}


def test_mailbox_has_57_letters_and_attachments():
    box = load_mailbox(T)
    letters = [m for m in box if m.parent_id is None]
    assert len(letters) == 57
    att = next(m for m in box if m.source_id == "M020/application.pdf")
    assert att.parent_id == "M020" and att.kind == "application"


def sig(p):
    return parse_signatures(pdf_text(T / "mail_pool" / p))


def test_signatures_act_both():
    assert sig("M002/act_act.pdf").both is True


def test_signatures_act_principal_missing():
    s = sig("M045/act_act.pdf")
    assert s.principal is False and s.provider is True


def test_signatures_application_pending():
    assert sig("M020/application.pdf").principal is False


def test_signatures_returned_application():
    assert sig("M011/returned_application.pdf").both is True
