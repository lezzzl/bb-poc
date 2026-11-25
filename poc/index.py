"""Поиск BM25 по материалам области (SQLite FTS5, trigram). Индекс живёт один эпизод."""

import re
import sqlite3
import threading

from poc.models import Material


class ScopeIndex:
    def __init__(self, materials: list[Material]):
        self._by_id = {m.source_id: m for m in materials}
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._lock = threading.Lock()  # модель может вызвать инструменты параллельно: одно соединение — по очереди
        self._db.execute("CREATE VIRTUAL TABLE docs USING fts5(source_id UNINDEXED, kind UNINDEXED, "
                         "sender UNINDEXED, body, tokenize='trigram')")
        self._db.executemany("INSERT INTO docs VALUES (?,?,?,?)",
                             [(m.source_id, m.kind, m.sender or "", f"{m.subject or ''}\n{m.text}") for m in materials])

    def ids(self) -> set[str]:
        return set(self._by_id)

    def get(self, source_id: str) -> Material | None:
        return self._by_id.get(source_id)

    def materials(self) -> list[Material]:
        return list(self._by_id.values())

    @staticmethod
    def _fts_query(query: str) -> str | None:
        words = [w for w in re.findall(r"[\w\-·]+", query) if len(w) >= 3]
        return " OR ".join('"' + w.replace('"', "") + '"' for w in words) or None

    def search(self, query: str, kind: str | None = None, sender: str | None = None, limit: int = 8) -> list[dict]:
        q = self._fts_query(query)
        if q is None:
            return []
        sql = "SELECT source_id, kind, sender, snippet(docs, 3, '[', ']', '…', 24) FROM docs WHERE docs MATCH ?"
        args: list = [q]
        if kind:
            sql += " AND kind = ?"
            args.append(kind)
        if sender:
            sql += " AND sender = ?"
            args.append(sender)
        sql += " ORDER BY bm25(docs) LIMIT ?"
        args.append(limit)
        try:
            with self._lock:
                rows = self._db.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"source_id": r[0], "kind": r[1], "sender": r[2] or None,
                 "at": self._by_id[r[0]].at.isoformat(), "snippet": r[3]} for r in rows]
