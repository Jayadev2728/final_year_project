"""
database.py
------------
SQLite database layer for ClassSentinel.

This module is the single source of truth for the schema and all
read/write operations. Two different programs use it:
  1. detection_system.py  -> WRITES data (attendance, alerts, engagement)
  2. backend/app.py       -> READS data to serve the dashboard API

Keeping all SQL in one place means both programs always agree on the
schema, and the dashboard can never see half-written rows.
"""

import sqlite3
from datetime import datetime
from contextlib import contextmanager
import os

# Single shared DB file at the project root, regardless of which
# script (detection_system.py or backend/app.py) imports this module.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classentinel.db")


def normalize_id(student_id):
    """Uppercase + strip, so '4bd23IS050' and '4BD23IS050' are treated as the same student."""
    return student_id.strip().upper()


def normalize_name(student_name):
    """Capitalizes each word, but preserves short all-caps words (KL, JS, BS)
    as initials instead of mangling them — Python's plain .title() turns
    'Prashanth KL' into 'Prashanth Kl', which is wrong for Indian naming
    conventions that commonly include initials."""
    words = student_name.strip().split()
    fixed = []
    for word in words:
        if len(word) <= 3 and word.isalpha():
            fixed.append(word.upper())   # treat short words as initials: kl/KL/Kl -> KL
        else:
            fixed.append(word.capitalize())
    return " ".join(fixed)


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # allows detection script + API to read/write concurrently
    return conn


@contextmanager
def db_cursor():
    conn = get_connection()
    try:
        cur = conn.cursor()
        yield cur
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Creates all tables if they don't already exist. Safe to call every startup."""
    with db_cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                status TEXT NOT NULL,
                UNIQUE(session_id, student_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS phone_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                time TEXT NOT NULL,
                student_id TEXT,
                student_name TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS drowsy_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                time TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS engagement_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                time TEXT NOT NULL,
                score INTEGER NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS session_presence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                student_id TEXT NOT NULL,
                student_name TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                UNIQUE(session_id, student_id)
            )
        """)
    _migrate_add_column("phone_alerts", "student_id", "TEXT")
    _migrate_add_column("phone_alerts", "student_name", "TEXT")
    _migrate_add_column("drowsy_alerts", "student_id", "TEXT")
    _migrate_add_column("drowsy_alerts", "student_name", "TEXT")


def _migrate_add_column(table, column, coltype):
    """Adds a column to an existing table if it's not already there. Needed
    because init_db()'s CREATE TABLE IF NOT EXISTS won't add new columns to
    a database file created by an older version of this schema."""
    with db_cursor() as cur:
        cur.execute(f"PRAGMA table_info({table})")
        existing_columns = [row["name"] for row in cur.fetchall()]
        if column not in existing_columns:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")


# ── Sessions ─────────────────────────────────────────────────────
def start_session():
    """Call once when the detection script starts. Returns the new session_id."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (start_time) VALUES (?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"),)
        )
        return cur.lastrowid


def end_session(session_id):
    """Call once when the detection script exits (e.g. on 'q')."""
    with db_cursor() as cur:
        cur.execute(
            "UPDATE sessions SET end_time = ? WHERE id = ?",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), session_id)
        )


def get_latest_session():
    with db_cursor() as cur:
        cur.execute("SELECT * FROM sessions ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else None


def get_session(session_id):
    with db_cursor() as cur:
        cur.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def get_all_sessions():
    with db_cursor() as cur:
        cur.execute("SELECT * FROM sessions ORDER BY id DESC")
        return [dict(r) for r in cur.fetchall()]


# ── Attendance ───────────────────────────────────────────────────
def mark_attendance(session_id, student_id, student_name, status):
    """Insert-or-ignore so a student is only ever marked once per session.
    IDs/names are normalized so casing differences don't create duplicate students."""
    student_id   = normalize_id(student_id)
    student_name = normalize_name(student_name)
    now = datetime.now()
    with db_cursor() as cur:
        cur.execute(
            """INSERT OR IGNORE INTO attendance
               (session_id, student_id, student_name, date, time, status)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, student_id, student_name,
             now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), status)
        )
        return cur.rowcount > 0  # True only if this was a new insert


def is_marked(session_id, student_id):
    student_id = normalize_id(student_id)
    with db_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM attendance WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        )
        return cur.fetchone() is not None


def is_marked_today(student_id):
    """Checks ALL sessions today, not just the current one — this is what stops
    re-running main.py several times in one day from creating duplicate rows
    for the same student (the exact issue visible in the old attendance.csv)."""
    student_id = normalize_id(student_id)
    today = datetime.now().strftime("%Y-%m-%d")
    with db_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND date = ? LIMIT 1",
            (student_id, today)
        )
        return cur.fetchone() is not None


def get_attendance(session_id):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM attendance WHERE session_id = ? ORDER BY time",
            (session_id,)
        )
        return [dict(r) for r in cur.fetchall()]


def get_todays_attendance_status(student_id):
    """Looks up a student's official attendance status for TODAY, regardless
    of which session actually recorded it. This is what lets a later
    session's report correctly show someone as 'Late'/'On Time' even though
    same-day dedup meant THIS session didn't write a new attendance row for
    them — they were already marked earlier today."""
    student_id = normalize_id(student_id)
    today = datetime.now().strftime("%Y-%m-%d")
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM attendance WHERE student_id = ? AND date = ? LIMIT 1",
            (student_id, today)
        )
        row = cur.fetchone()
        return dict(row) if row else None


# ── Session presence (separate from once-a-day attendance) ─────────
def mark_seen(session_id, student_id, student_name):
    """Upserts a 'this student was seen in this session' record — separate
    from mark_attendance's once-a-day dedup. This is what a session's
    report/dashboard uses to show 'who was actually present', since the
    attendance table deliberately won't write a new row for someone
    already marked earlier the same day."""
    student_id   = normalize_id(student_id)
    student_name = normalize_name(student_name)
    now = datetime.now().strftime("%H:%M:%S")
    with db_cursor() as cur:
        cur.execute(
            """INSERT INTO session_presence (session_id, student_id, student_name, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id, student_id) DO UPDATE SET last_seen = excluded.last_seen""",
            (session_id, student_id, student_name, now, now)
        )


def get_session_presence(session_id):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM session_presence WHERE session_id = ? ORDER BY first_seen",
            (session_id,)
        )
        return [dict(r) for r in cur.fetchall()]


# ── Alerts ───────────────────────────────────────────────────────
def log_phone_alert(session_id, student_id=None, student_name=None):
    """student_id/student_name are the closest recognized face to the phone
    at detection time (best-effort proximity match, not guaranteed correct
    if multiple people are close together) — pass None/None if no
    recognized face was near enough to attribute confidently."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO phone_alerts (session_id, time, student_id, student_name) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now().strftime("%H:%M:%S"), student_id, student_name)
        )


def log_drowsy_alert(session_id, student_id=None, student_name=None):
    """student_id/student_name are the closest recognized face to the
    drowsy landmark set at detection time — None/None if no recognized
    face was near enough to attribute confidently."""
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO drowsy_alerts (session_id, time, student_id, student_name) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now().strftime("%H:%M:%S"), student_id, student_name)
        )


def get_phone_alerts(session_id, limit=50):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM phone_alerts WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        )
        return [dict(r) for r in cur.fetchall()]


def get_drowsy_alerts(session_id, limit=50):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM drowsy_alerts WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        )
        return [dict(r) for r in cur.fetchall()]


def get_phone_alert_count(session_id, student_id):
    student_id = normalize_id(student_id)
    with db_cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as c FROM phone_alerts WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        )
        return cur.fetchone()["c"]


def get_drowsy_alert_count(session_id, student_id):
    student_id = normalize_id(student_id)
    with db_cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) as c FROM drowsy_alerts WHERE session_id = ? AND student_id = ?",
            (session_id, student_id)
        )
        return cur.fetchone()["c"]


def get_student_summary(session_id):
    """THE unified per-student table: one row per student actually present
    in this session, combining attendance status, phone alerts, drowsy
    alerts, and a derived engagement score. This is what the report and
    dashboard show instead of several disconnected tables."""
    import config
    presence = get_session_presence(session_id)
    rows = []
    for p in presence:
        sid, sname  = p["student_id"], p["student_name"]
        attendance  = get_todays_attendance_status(sid)
        status      = attendance["status"] if attendance else "Not marked"
        phone_count  = get_phone_alert_count(session_id, sid)
        drowsy_count = get_drowsy_alert_count(session_id, sid)
        engagement   = max(0, 100 - phone_count * config.PHONE_ALERT_PENALTY
                                    - drowsy_count * config.DROWSY_ALERT_PENALTY)
        rows.append({
            "student_id":    sid,
            "student_name":  sname,
            "attendance":    status,
            "first_seen":    p["first_seen"],
            "last_seen":     p["last_seen"],
            "phone_alerts":  phone_count,
            "drowsy_alerts": drowsy_count,
            "engagement":    engagement
        })
    return rows


# ── Engagement ───────────────────────────────────────────────────
def log_engagement(session_id, score):
    with db_cursor() as cur:
        cur.execute(
            "INSERT INTO engagement_log (session_id, time, score) VALUES (?, ?, ?)",
            (session_id, datetime.now().strftime("%H:%M:%S"), score)
        )


def get_engagement_log(session_id):
    with db_cursor() as cur:
        cur.execute(
            "SELECT * FROM engagement_log WHERE session_id = ? ORDER BY id",
            (session_id,)
        )
        return [dict(r) for r in cur.fetchall()]


def export_attendance_csv(session_id, path="attendance.csv"):
    """Writes attendance for one session out to CSV, matching the format your
    evaluators/report already expect. The database stays the single source of
    truth; CSV is just a export view of it, so the two can never drift apart
    the way attendance.csv and attendence.csv did before."""
    import csv
    rows = get_attendance(session_id)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["student_id", "student_name", "date", "time", "status"])
        for r in rows:
            writer.writerow([r["student_id"], r["student_name"], r["date"], r["time"], r["status"]])


# ── Summary (used for the dashboard's top cards) ───────────────────
def get_summary(session_id):
    presence   = get_session_presence(session_id)
    phone      = get_phone_alerts(session_id, limit=100000)
    drowsy     = get_drowsy_alerts(session_id, limit=100000)
    engagement = get_engagement_log(session_id)
    avg_score  = round(sum(e["score"] for e in engagement) / len(engagement), 1) if engagement else 0
    return {
        "total_marked":   len(presence),
        "phone_alerts":   len(phone),
        "drowsy_alerts":  len(drowsy),
        "avg_engagement": avg_score
    }
