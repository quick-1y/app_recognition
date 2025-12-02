from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from app.config import DatabaseConfig


@dataclass
class PlateRecord:
    plate: str
    region: str
    confidence: float
    source: str
    created_at: datetime


class PlateDatabase:
    """SQLite wrapper for storing recognized plates and watchlists."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.path = Path(config.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._init_schema()
        if self.config.vacuum_on_start:
            self.conn.execute("VACUUM")
        if self.config.retention_days:
            self.prune_old_records()

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate TEXT NOT NULL,
                region TEXT,
                confidence REAL,
                source TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS lists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS list_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                list_id INTEGER NOT NULL,
                plate TEXT NOT NULL,
                FOREIGN KEY(list_id) REFERENCES lists(id) ON DELETE CASCADE
            );
            """
        )
        self.conn.commit()

    def prune_old_records(self) -> None:
        cutoff = datetime.utcnow() - timedelta(days=self.config.retention_days)
        self.conn.execute("DELETE FROM plates WHERE created_at < ?", (cutoff.isoformat(),))
        self.conn.commit()

    def add_plate(self, plate: str, region: str, confidence: float, source: str) -> None:
        self.conn.execute(
            "INSERT INTO plates (plate, region, confidence, source) VALUES (?, ?, ?, ?)",
            (plate, region, confidence, source),
        )
        self.conn.commit()

    def recent_plates(self, limit: int = 100) -> List[PlateRecord]:
        cur = self.conn.execute(
            "SELECT plate, region, confidence, source, created_at FROM plates ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            PlateRecord(
                plate=row[0],
                region=row[1] or "",
                confidence=float(row[2]) if row[2] is not None else 0.0,
                source=row[3] or "",
                created_at=datetime.fromisoformat(row[4]) if row[4] else datetime.utcnow(),
            )
            for row in rows
        ]

    def ensure_list(self, name: str, description: str = "") -> int:
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO lists (name, description) VALUES (?, ?)",
            (name, description),
        )
        if cur.lastrowid:
            self.conn.commit()
        list_row = self.conn.execute("SELECT id FROM lists WHERE name=?", (name,)).fetchone()
        return int(list_row[0])

    def add_to_list(self, list_name: str, plate: str, description: str = "") -> None:
        list_id = self.ensure_list(list_name, description)
        self.conn.execute(
            "INSERT INTO list_items (list_id, plate) VALUES (?, ?)",
            (list_id, plate.upper()),
        )
        self.conn.commit()

    def list_members(self, list_name: str) -> List[str]:
        row = self.conn.execute("SELECT id FROM lists WHERE name=?", (list_name,)).fetchone()
        if not row:
            return []
        list_id = int(row[0])
        cur = self.conn.execute("SELECT plate FROM list_items WHERE list_id=? ORDER BY plate", (list_id,))
        return [r[0] for r in cur.fetchall()]

    def lists(self) -> List[Tuple[str, str]]:
        cur = self.conn.execute("SELECT name, description FROM lists ORDER BY name")
        return [(row[0], row[1] or "") for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


__all__ = ["PlateDatabase", "PlateRecord"]
