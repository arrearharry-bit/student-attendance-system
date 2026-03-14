"""
models.py – SQLite database helper functions for the attendance system.
All DB access goes through these helpers for consistency.
"""
import sqlite3
import os
from contextlib import contextmanager
from datetime import date

DB_PATH = 'information.db'

@contextmanager
def get_db():
    """Context manager: yields a Row-factory SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ─── Schema ──────────────────────────────────────────────────────────────────

def init_db():
    """Create all tables and migrate existing data."""
    with get_db() as conn:
        # Users table (RBAC)
        conn.execute('''CREATE TABLE IF NOT EXISTS Users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    UNIQUE NOT NULL,
            password    TEXT    NOT NULL,
            name        TEXT    DEFAULT '',
            email       TEXT    DEFAULT '',
            role        TEXT    DEFAULT 'student',
            status      TEXT    DEFAULT 'active',
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        )''')

        # Students table (extended profile)
        conn.execute('''CREATE TABLE IF NOT EXISTS Students (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             INTEGER REFERENCES Users(id),
            student_id          TEXT    UNIQUE,
            name                TEXT    NOT NULL,
            status              TEXT    DEFAULT 'active',
            registration_date   TEXT    DEFAULT CURRENT_TIMESTAMP,
            image_filename      TEXT    DEFAULT ''
        )''')

        # Attendance table (enhanced)
        conn.execute('''CREATE TABLE IF NOT EXISTS Attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            NAME        TEXT    NOT NULL,
            Time        TEXT    NOT NULL,
            Date        TEXT    NOT NULL,
            STATUS      TEXT    DEFAULT 'Present',
            notes       TEXT    DEFAULT ''
        )''')
        # Migrate old schema: ensure STATUS column exists
        try:
            conn.execute("ALTER TABLE Attendance ADD COLUMN STATUS TEXT DEFAULT 'Present'")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE Attendance ADD COLUMN notes TEXT DEFAULT ''")
        except Exception:
            pass

        # Audit log
        conn.execute('''CREATE TABLE IF NOT EXISTS AuditLogs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_username  TEXT,
            action          TEXT,
            target          TEXT,
            details         TEXT,
            timestamp       TEXT    DEFAULT CURRENT_TIMESTAMP
        )''')
    print("Database initialized.")

# ─── User helpers ─────────────────────────────────────────────────────────────

def get_user_by_username(username):
    with get_db() as conn:
        return conn.execute("SELECT * FROM Users WHERE username=?", (username,)).fetchone()

def get_user_by_id(user_id):
    with get_db() as conn:
        return conn.execute("SELECT * FROM Users WHERE id=?", (user_id,)).fetchone()

def create_user(username, hashed_pw, name='', email='', role='student'):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO Users (username, password, name, email, role) VALUES (?,?,?,?,?)",
            (username, hashed_pw, name, email, role)
        )

def update_user(user_id, name, email, status, hashed_pw=None):
    with get_db() as conn:
        if hashed_pw:
            conn.execute("UPDATE Users SET name=?,email=?,status=?,password=? WHERE id=?",
                         (name, email, status, hashed_pw, user_id))
        else:
            conn.execute("UPDATE Users SET name=?,email=?,status=? WHERE id=?",
                         (name, email, status, user_id))

def get_all_students():
    """Return all Users with role=student."""
    with get_db() as conn:
        return conn.execute("SELECT * FROM Users WHERE role='student' ORDER BY name").fetchall()

def search_students(query='', status=''):
    with get_db() as conn:
        sql = "SELECT * FROM Users WHERE role='student'"
        params = []
        if query:
            sql += " AND (name LIKE ? OR username LIKE ?)"
            params += [f'%{query}%', f'%{query}%']
        if status:
            sql += " AND status=?"
            params.append(status)
        return conn.execute(sql, params).fetchall()

# ─── Attendance helpers ───────────────────────────────────────────────────────

def mark_present(name):
    """Insert a Present record if not already marked today."""
    today = str(date.today())
    from datetime import datetime
    now_str = datetime.now().strftime('%I:%M %p')
    with get_db() as conn:
        existing = conn.execute(
            "SELECT rowid FROM Attendance WHERE NAME=? AND Date=?", (name, today)
        ).fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO Attendance (NAME, Time, Date, STATUS) VALUES (?,?,?,?)",
                (name.upper(), now_str, today, 'Present')
            )
            return True  # newly marked
    return False  # already marked

def get_today_attendance():
    today = str(date.today())
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM Attendance WHERE Date=? ORDER BY Time DESC", (today,)
        ).fetchall()

def get_all_attendance(name=None, from_date=None, to_date=None):
    with get_db() as conn:
        sql = "SELECT rowid, * FROM Attendance WHERE 1=1"
        params = []
        if name:
            sql += " AND UPPER(NAME)=?"
            params.append(name.upper())
        if from_date:
            sql += " AND Date >= ?"
            params.append(from_date)
        if to_date:
            sql += " AND Date <= ?"
            params.append(to_date)
        sql += " ORDER BY Date DESC, Time DESC"
        return conn.execute(sql, params).fetchall()

def update_attendance_status(rowid, new_status):
    with get_db() as conn:
        conn.execute("UPDATE Attendance SET STATUS=? WHERE rowid=?", (new_status, rowid))

def get_student_stats(display_name, from_date=None):
    records = get_all_attendance(name=display_name, from_date=from_date)
    total = len(records)
    present = sum(1 for r in records if r['STATUS'] == 'Present')
    absent = total - present
    pct = round((present / total * 100), 1) if total > 0 else 0
    return total, present, absent, pct, records

# ─── Audit log ───────────────────────────────────────────────────────────────

def log_audit(admin_username, action, target, details=""):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO AuditLogs (admin_username, action, target, details) VALUES (?,?,?,?)",
            (admin_username, action, target, details)
        )

def get_audit_logs(limit=50):
    with get_db() as conn:
        return conn.execute(
            "SELECT * FROM AuditLogs ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
