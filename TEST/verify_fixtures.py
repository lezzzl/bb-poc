"""Проверка согласованности входных материалов учебного набора PoC."""
from datetime import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def verify():
    directory = read_json(ROOT / "reference/companies.json")
    roles = read_json(ROOT / "reference/roles.json")
    known_addresses = {row["email"] for row in roles}
    known_addresses.update(company["email"] for company in directory.values())
    company_fields = {
        "client": "client_id",
        "counterparty": "counterparty_id",
        "service_provider": "service_provider_id",
    }
    sources_checked = 0
    case_paths = sorted((ROOT / "cases").glob("*/without_mail_search/case.json"))
    require(bool(case_paths), "Не найдены карточки кейсов")
    application_ids = set()

    for path in case_paths:
        case = read_json(path)
        case_id = case["case_id"]
        application_id = case["application_id"]
        require(path.parent.parent.name == case_id, f"{case_id}: неверное имя папки")
        require(application_id not in application_ids, f"{case_id}: номер заявки уже использован")
        application_ids.add(application_id)

        sources = {item["source_id"]: item for item in case["sources"]}
        require(len(sources) == len(case["sources"]), f"{case_id}: повтор ID источника")
        crm = read_json(path.parent / sources["CRM"]["path"])
        require(crm["application_id"] == application_id and crm["status"] == case["initial_status"],
                f"{case_id}: карточка и снимок CRM расходятся")
        require(crm["recorded_at"] == sources["CRM"]["available_at"], f"{case_id}: расходится время снимка CRM")

        for snapshot_field, card_field in company_fields.items():
            company_id = case[card_field]
            require(crm[snapshot_field] == {"company_id": company_id, **directory[company_id]},
                    f"{case_id}: поля {company_id} в CRM расходятся со справочником")

        for source_id, source in sources.items():
            label = f"{case_id}/{source_id}"
            content = (path.parent / source["path"]).resolve()
            require(content.is_relative_to(path.parent) and content.is_file(), f"{label}: нет локального файла")
            require(datetime.fromisoformat(source["available_at"]) <= datetime.fromisoformat(case["cutoff_at"]),
                    f"{label}: источник позже отсечения")
            address = source.get("source_address")
            require(address is None or address in known_addresses, f"{label}: неизвестный адрес источника")
            if source["kind"] == "email":
                lines = content.read_text(encoding="utf-8").splitlines()
                require(lines[0] == f"От: {address}" and lines[2] == f"Дата: {source['available_at']}",
                        f"{label}: отправитель или время не совпадает с карточкой")
                require(lines[1].removeprefix("Кому: ") in known_addresses, f"{label}: неизвестный получатель")
            sources_checked += 1

    print(f"Проверено кейсов: {len(case_paths)}. Источников: {sources_checked}. Компаний: {len(directory)}.")


if __name__ == "__main__":
    verify()
