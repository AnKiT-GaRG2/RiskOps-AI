"""Phase 6 — append-only, hash-chained audit log.

Every row carries prev_hash = SHA-256 of the previous row's canonical
fields, so any edit to a past row breaks every entry_hash after it —
tampering is detectable, not merely discouraged.

Three event types share one table (so the chain covers all of them in
true append order):
  - "decision"         a scored transaction and the action taken on it
  - "threshold_change" T_LOW/T_HIGH changing is its own logged event,
                        so "why did the block rate jump Tuesday" doesn't
                        require archaeology across code deploys
  - "outcome_update"    a later-known ground-truth outcome for a past
                        transaction_id — appended, never an UPDATE to
                        the original decision row (that would require
                        mutating an already-hashed entry)
"""
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

GENESIS_HASH = "0" * 64
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "audit_log.db"

_CHAIN_FIELDS = [
    "event_type", "timestamp", "transaction_id", "model_version",
    "raw_score", "calibrated_score", "band", "t_low", "t_high",
    "top_attributions", "action", "outcome", "prev_hash",
]

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    transaction_id TEXT,
    model_version TEXT,
    raw_score REAL,
    calibrated_score REAL,
    band TEXT,
    t_low REAL,
    t_high REAL,
    top_attributions TEXT,
    action TEXT,
    outcome TEXT,
    prev_hash TEXT NOT NULL,
    entry_hash TEXT NOT NULL
);
"""


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def _canonical(fields: dict) -> str:
    return json.dumps({k: fields.get(k) for k in _CHAIN_FIELDS}, sort_keys=True, default=str)


def _entry_hash(fields: dict) -> str:
    return hashlib.sha256(_canonical(fields).encode("utf-8")).hexdigest()


def _last_hash(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT entry_hash FROM decisions ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else GENESIS_HASH


def _append(conn: sqlite3.Connection, **fields) -> int:
    fields.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    fields["prev_hash"] = _last_hash(conn)
    fields["entry_hash"] = _entry_hash(fields)

    cols = _CHAIN_FIELDS + ["entry_hash"]
    values = [fields.get(c) for c in cols]
    placeholders = ",".join(["?"] * len(cols))
    cur = conn.execute(f"INSERT INTO decisions ({','.join(cols)}) VALUES ({placeholders})", values)
    conn.commit()
    return cur.lastrowid


def append_decision(conn, transaction_id: str, model_version: str, raw_score: float,
                     calibrated_score: float, band: str, t_low: float, t_high: float,
                     top_attributions: list, action: str) -> int:
    return _append(
        conn, event_type="decision", transaction_id=str(transaction_id), model_version=model_version,
        raw_score=raw_score, calibrated_score=calibrated_score, band=band,
        t_low=t_low, t_high=t_high, top_attributions=json.dumps(top_attributions), action=action,
    )


def append_threshold_change(conn, model_version: str, t_low: float, t_high: float, reason: str = None) -> int:
    return _append(
        conn, event_type="threshold_change", model_version=model_version,
        t_low=t_low, t_high=t_high, action=reason,
    )


def append_outcome_update(conn, transaction_id: str, outcome: str) -> int:
    return _append(conn, event_type="outcome_update", transaction_id=str(transaction_id), outcome=outcome)


def verify_chain(conn: sqlite3.Connection) -> dict:
    rows = conn.execute(
        "SELECT id, event_type, timestamp, transaction_id, model_version, raw_score, calibrated_score, "
        "band, t_low, t_high, top_attributions, action, outcome, prev_hash, entry_hash "
        "FROM decisions ORDER BY id ASC"
    ).fetchall()

    cols = ["id", "event_type", "timestamp", "transaction_id", "model_version", "raw_score",
            "calibrated_score", "band", "t_low", "t_high", "top_attributions", "action", "outcome",
            "prev_hash", "entry_hash"]

    expected_prev = GENESIS_HASH
    broken_at = []
    for row in rows:
        r = dict(zip(cols, row))
        recomputed = _entry_hash(r)
        if r["prev_hash"] != expected_prev or recomputed != r["entry_hash"]:
            broken_at.append(r["id"])
        expected_prev = r["entry_hash"]

    return {"valid": len(broken_at) == 0, "rows_checked": len(rows), "broken_row_ids": broken_at}


def tamper_row(conn: sqlite3.Connection, row_id: int, **field_overrides) -> None:
    """DEV-ONLY. Directly mutates a row without recomputing the hash
    chain, so verify_chain() can be shown failing live on stage."""
    if not field_overrides:
        field_overrides = {"action": "TAMPERED"}
    set_clause = ",".join(f"{k}=?" for k in field_overrides)
    conn.execute(f"UPDATE decisions SET {set_clause} WHERE id=?", list(field_overrides.values()) + [row_id])
    conn.commit()


def _demo() -> None:
    """Phase 6 checkpoint: score a batch, verify the chain, tamper one
    row, watch verification fail. Run: python -m src.audit.chain"""
    import tempfile
    db_path = Path(tempfile.mkdtemp()) / "demo_audit.db"
    conn = connect(db_path)

    append_threshold_change(conn, model_version="lgbm_v1", t_low=0.3, t_high=0.7, reason="initial deploy")
    for i in range(5):
        append_decision(
            conn, transaction_id=f"TXN{i}", model_version="lgbm_v1",
            raw_score=0.5 + i * 0.1, calibrated_score=0.4 + i * 0.1, band="HOLD" if i > 2 else "ALLOW",
            t_low=0.3, t_high=0.7,
            top_attributions=[{"name": "velocity_1h", "value": i, "contribution": 0.2}],
            action="HOLD" if i > 2 else "ALLOW",
        )
    append_outcome_update(conn, transaction_id="TXN4", outcome="confirmed_fraud")

    report = verify_chain(conn)
    print(f"Before tamper: valid={report['valid']} rows_checked={report['rows_checked']}")
    assert report["valid"], "chain should be valid before tampering"

    print("Tampering row 6 (rewriting a HOLD decision's action to ALLOW)...")
    tamper_row(conn, row_id=6, action="ALLOW")

    report = verify_chain(conn)
    print(f"After tamper: valid={report['valid']} broken_row_ids={report['broken_row_ids']}")
    assert not report["valid"], "tampering should have broken the chain"
    print("PASS: tamper detected.")


if __name__ == "__main__":
    _demo()
