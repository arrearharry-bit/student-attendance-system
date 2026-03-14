# Face Recognition Attendance System 📸🎓

A professional, full-stack attendance management system using Flask, OpenCV, and the `face_recognition` library. Featuring a high-accuracy local recognition engine, interactive dashboards, and role-based access control.

## 🚀 Key Features

- **Dual Face Recognition**:
  - **Local Camera (High Accuracy)**: Uses a local OpenCV window for the fastest matching and best resolution.
  - **Webcam (Remote)**: Browser-based AJAX recognition for flexibility.
- **Role-Based Access Control (RBAC)**:
  - **Admin**: Manage students, view audit logs, edit attendance, and view global stats.
  - **Student**: View personal attendance history and performance dashboards.
- **Enhanced Accuracy**: Encodings generated with 10x Jittering and 50% recognition resolution.
- **Modern UI**: Dark-themed, responsive dashboard built with Bootstrap 5 and Chart.js.
- **Production Ready**: Support for environment variables (`.env`), HTTPS (adhoc SSL), and custom error handling.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/amlanmohanty1/face-recognition-attendance-management-system-with-PowerBI-dashboard.git
   cd face-recognition-attendance-management-system-with-PowerBI-dashboard
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**:
   Create a `.env` file in the root directory:
   ```env
   SECRET_KEY=your_secure_random_key
   FLASK_ENV=development
   ```

## 📸 Usage

### 1. Register Students
Place student photos in the `Training images/` folder. Ensure the filename is precisely the student's name (e.g., `John_Doe.png`).

### 2. Initialize System
Run the app to initialize the database and generate high-quality face encodings:
```bash
python app.py
```

### 3. Record Attendance
- Open the web interface (default: `https://127.0.0.1:5000`).
- Login as a student or admin.
- Go to the **Punch Attendance** page.
- Choose **Local Camera** for maximum accuracy.

## 🛡️ Security
- **Passwords**: Hashed using Werkzeug security (PBKDF2).
- **HTTPS**: Automatically enabled via `ssl_context='adhoc'` for secure camera access.
- **Audit Logs**: Every admin action (edits, re-encodes) is logged.

## 📂 Project Structure
- `app.py`: Main Flask application core.
- `models.py`: Database schema and SQLite helper functions.
- `face_utils.py`: Face encoding and recognition logic.
- `templates/`: HTML templates with modern styling.
- `static/`: CSS and JavaScript (Webcam logic, Chart.js).
- `information.db`: SQLite database storing users and attendance.

---
*Built with ❤️ for Educational Institutions.*
