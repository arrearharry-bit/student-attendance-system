"""
init_db.py – One-time database setup + seed script.
Run: python init_db.py
"""
from models import init_db, get_db
from werkzeug.security import generate_password_hash

def seed():
    init_db()
    with get_db() as conn:
        # Seed default admin
        existing = conn.execute("SELECT id FROM Users WHERE username='admin'").fetchone()
        if not existing:
            conn.execute(
                "INSERT INTO Users (username, password, name, role, status) VALUES (?,?,?,?,?)",
                ('admin', generate_password_hash('admin123'), 'Administrator', 'admin', 'active')
            )
            print("Admin created: username=admin, password=admin123")
        else:
            # Update to werkzeug hash format
            conn.execute(
                "UPDATE Users SET password=? WHERE username='admin'",
                (generate_password_hash('admin123'),)
            )
            print("Admin password refreshed.")

        # Seed sample students (Removed for hard reset)
        count = conn.execute("SELECT COUNT(*) FROM Users WHERE role='student'").fetchone()[0]
        if count == 0:
            print("Skipping student seeding (Hard Reset mode).")


    print("\n✅ Database ready! Run: python app.py")
    print("   Admin login: admin / admin123")

if __name__ == '__main__':
    seed()
