"""
app.py – Main Flask application.
Clean, modular entry point. All routes organized cleanly.
Run:   python init_db.py   (first time)
Then:  python app.py
"""
from flask import (Flask, render_template, request, session,
                   redirect, url_for, jsonify, send_file, flash)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os, io, json, base64
import sqlite3 # Added sqlite3 import
import numpy as np
from datetime import date, datetime, timedelta
from functools import wraps
from dotenv import load_dotenv # Added dotenv import
import time # Added for camera slowing

load_dotenv() # Call load_dotenv()

from models import (init_db, get_db, get_user_by_username, get_user_by_id,
                    create_user, update_user, get_all_students, search_students,
                    mark_present, get_today_attendance, get_all_attendance,
                    update_attendance_status, get_student_stats, log_audit, get_audit_logs,
                    auto_mark_absent_for_today)
from face_utils import (recognize_frame, encode_all_images, get_encodings, 
                        load_encodings, encode_single_image)
import cv2
# import face_recognition (Removed for stability)

# ─── App setup ────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default-dev-key') # Updated secret key loading
app.config['ENV'] = os.getenv('FLASK_ENV', 'development') # Added FLASK_ENV config

TRAINING_PATH = 'Training images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ─── Auth decorators ──────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def student_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'student':
            flash('Student access required.', 'danger')
            return redirect(url_for('student_dashboard'))
        return f(*args, **kwargs)
    return decorated

# ─── Public routes ────────────────────────────────────────────────────────────

@app.route('/')
def home():
    today_count = len(get_today_attendance())
    student_count = len(get_all_students())
    return render_template('home.html', today_count=today_count, student_count=student_count)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('admin_dashboard') if session.get('role') == 'admin' else url_for('student_dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        user = get_user_by_username(username)
        if user and user['status'] == 'active':
            pw = user['password']
            # Support both old SHA-256 and new Werkzeug format
            import hashlib
            valid = (check_password_hash(pw, password) if pw.startswith('pbkdf2:') or pw.startswith('scrypt:')
                     else pw == hashlib.sha256(password.encode()).hexdigest())
            if valid:
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                session['user_name'] = user['name'] or user['username']
                flash(f"Welcome back, {session['user_name']}!", 'success')
                return redirect(url_for('admin_dashboard') if user['role'] == 'admin' else url_for('student_dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    name = session.get('user_name', '')
    session.clear()
    flash(f'Goodbye, {name}!', 'info')
    return redirect(url_for('home'))

# Backward compat
@app.route('/how')
def how():
    return redirect(url_for('login'))

@app.route('/checklogin')
def checklogin():
    return session.get('username', 'False')

# ─── Webcam / Face Recognition ────────────────────────────────────────────────

@app.route('/punch')
def punch():
    """Page offering choice between Browser Camera and Local Server Camera."""
    return render_template('punch.html')

@app.route('/recognize_local')
@login_required
def recognize_local():
    """Optimized 'Old Way': Opens a local OpenCV window using the LBPH recognizer."""
    video_capture = cv2.VideoCapture(0)
    
    # Ensure encodings are loaded
    load_encodings()
    
    flash("Local Camera Started. Press 'q' to close the window.", "info")
    
    frame_count = 0
    current_results = []
    
    while True:
        ret, frame = video_capture.read()
        if not ret: break
        
        # Processing delay to slow the camera rate as requested
        time.sleep(0.05)
        frame_count += 1
        
        # Only perform heavy recognition every 8th frame for maximum stability
        if frame_count % 8 == 0:
            # Recognition using OpenCV LBPH
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_results = recognize_frame(rgb_frame)

        for (name, dist, (top, right, bottom, left)) in current_results:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            if name != "Unknown":
                mark_present(name)
            
            # Draw box & Label
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            label = f"{name} ({dist:.2f})"
            cv2.putText(frame, label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Face Recognition (Press Q to Quit)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    return redirect(url_for('home'))



@app.route('/recognize_ajax', methods=['POST'])
def recognize_ajax():
    """AJAX endpoint: receives base64 image, returns recognition result."""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image provided'})

        img_b64 = data['image'].split(',')[-1]  # strip data:image/... prefix
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)

        import cv2
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'success': False, 'message': 'Cannot decode image'})

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = recognize_frame(rgb_frame)

        if not results or results[0][0] == 'Unknown':
            return jsonify({'success': True, 'name': 'Unknown', 'confidence': 0})

        name, dist, _ = results[0]
        already = not mark_present(name)
        return jsonify({
            'success': True,
            'name': name.title(),
            'confidence': round(1 - dist, 2),
            'already_marked': already
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# Legacy streaming route (keep for backward compat)
@app.route('/recognize', methods=['GET', 'POST'])
def recognize():
    return redirect(url_for('punch'))

# ─── Student Dashboard ────────────────────────────────────────────────────────

@app.route('/student/dashboard')
@student_required
def student_dashboard():
    username = session['username']
    user = get_user_by_id(session['user_id'])
    display_name = (user['name'] or username).upper()

    # Ensure daily absent status is marked
    auto_mark_absent_for_today()

    filter_type = request.args.get('filter', 'all')
    today = date.today()
    if filter_type == 'month':
        from_date = str(today.replace(day=1))
    elif filter_type == 'semester':
        from_date = str(today - timedelta(days=120))
    else:
        from_date = None

    total, present, absent, pct, records = get_student_stats(display_name, from_date)

    # Chart data
    daily = {}
    for r in records:
        d = str(r['Date'])
        if d not in daily:
            daily[d] = {'Present': 0, 'Absent': 0}
        
        status_key = r['STATUS']
        if status_key in daily[d]:
            daily[d][status_key] += 1
            
    recent_dates = sorted(daily.keys())
    chart_labels = recent_dates[-30:] if len(recent_dates) > 30 else recent_dates
    chart_present = [daily[d].get('Present', 0) for d in chart_labels]
    chart_absent  = [daily[d].get('Absent', 0) for d in chart_labels]

    return render_template('student/dashboard.html',
        display_name=display_name, total=total, present=present,
        absent=absent, pct=pct, records=list(records)[:20],
        chart_labels=json.dumps(chart_labels),
        chart_present=json.dumps(chart_present),
        chart_absent=json.dumps(chart_absent),
        filter_type=filter_type)

# ─── Admin Dashboard ──────────────────────────────────────────────────────────

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    # Ensure daily absent status is marked for all students
    auto_mark_absent_for_today()
    
    students = get_all_students()
    today_rows = get_today_attendance()
    audit_logs = get_audit_logs(20)

    total_students = len(students)
    today_present = len([r for r in today_rows if r['STATUS'] == 'Present'])
    attendance_rate = round((today_present / total_students * 100), 1) if total_students else 0

    # Weekly trend – last 7 days
    trend_labels, trend_counts = [], []
    for i in range(6, -1, -1):
        d = str(date.today() - timedelta(days=i))
        with get_db() as conn:
            cnt = conn.execute("SELECT COUNT(*) FROM Attendance WHERE Date=? AND STATUS='Present'", (d,)).fetchone()[0]
        trend_labels.append(d[-5:])
        trend_counts.append(cnt)

    return render_template('admin/dashboard.html',
        total_students=total_students,
        today_present=today_present,
        attendance_rate=attendance_rate,
        today_rows=today_rows,
        audit_logs=audit_logs,
        trend_labels=json.dumps(trend_labels),
        trend_counts=json.dumps(trend_counts))

# ─── Admin: Student Management ────────────────────────────────────────────────

@app.route('/admin/students')
@admin_required
def admin_students():
    search = request.args.get('search', '').strip()
    status_filter = request.args.get('status', '')
    users = search_students(search, status_filter)

    students = []
    for u in users:
        sname = (u['name'] or u['username']).upper()
        total, present, absent, pct, _ = get_student_stats(sname)
        students.append({
            'id': u['id'], 'username': u['username'], 'name': u['name'],
            'email': u['email'], 'status': u['status'],
            'total': total, 'present': present, 'absent': absent, 'pct': pct
        })
    return render_template('admin/students.html', students=students,
                           search=search, status_filter=status_filter)

@app.route('/admin/student/<int:sid>', methods=['GET'])
@admin_required
def admin_student_detail(sid):
    user = get_user_by_id(sid)
    if not user:
        return "Student not found", 404
    sname = (user['name'] or user['username']).upper()
    date_from = request.args.get('from', '')
    date_to   = request.args.get('to', '')
    records = get_all_attendance(name=sname, from_date=date_from or None, to_date=date_to or None)
    total = len(records)
    present = sum(1 for r in records if r['STATUS'] == 'Present')
    pct = round((present / total * 100), 1) if total else 0
    return render_template('admin/student_detail.html',
        student=user, records=records,
        total=total, present=present, absent=total-present, pct=pct,
        date_from=date_from, date_to=date_to)

@app.route('/admin/student/<int:sid>/edit', methods=['GET', 'POST'])
@admin_required
def admin_student_edit(sid):
    user = get_user_by_id(sid)
    if not user:
        return "Student not found", 404
    if request.method == 'POST':
        name   = request.form.get('name', user['name'])
        email  = request.form.get('email', user['email'])
        status = request.form.get('status', user['status'])
        new_pw = request.form.get('new_password', '').strip()
        hashed = generate_password_hash(new_pw) if new_pw else None
        update_user(sid, name, email, status, hashed)
        log_audit(session['username'], 'EDIT_STUDENT', user['username'], f'status={status}')
        flash(f'Profile updated for {name}.', 'success')
        return redirect(url_for('admin_student_detail', sid=sid))
    return render_template('admin/student_edit.html', student=user)

@app.route('/admin/student/register', methods=['GET', 'POST'])
@admin_required
def admin_register_student():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip()
        photo    = request.files.get('photo')

        if not username or not password:
            error = 'Username and password are required.'
        else:
            try:
                create_user(username, generate_password_hash(password), name, email, 'student')
                log_audit(session['username'], 'REGISTER_STUDENT', username)
                # Handle photo upload
                if photo and allowed_file(photo.filename):
                    filename = secure_filename(f"{name or username}.png")
                    save_path = os.path.join(TRAINING_PATH, filename)
                    photo.save(save_path)
                    if encode_single_image(save_path, name or username):
                        flash(f'Student {name} registered and face encoded!', 'success')
                    else:
                        flash(f'Student {name} registered, but FACE DETECTION FAILED. Please upload a clearer photo to enable recognition.', 'warning')
                else:
                    flash(f'Student {name} registered. Upload a photo later to enable face recognition.', 'info')
                return redirect(url_for('admin_students'))
            except Exception as e:
                error = str(e)
    return render_template('admin/register_student.html', error=error)

@app.route('/admin/attendance/edit', methods=['POST'])
@admin_required
def admin_attendance_edit():
    id      = request.form.get('id')
    status  = request.form.get('status')
    sid     = request.form.get('student_id')
    update_attendance_status(id, status)
    log_audit(session['username'], 'EDIT_ATTENDANCE', f'id={id}', f'status={status}')
    return redirect(url_for('admin_student_detail', sid=sid))

@app.route('/admin/edit', methods=['GET'])
@admin_required
def admin_edit():
    today = str(date.today())
    with get_db() as conn:
        present_today = [r['NAME'].upper() for r in
            conn.execute("SELECT NAME FROM Attendance WHERE Date=?", (today,)).fetchall()]
        all_students = conn.execute(
            "SELECT name, username FROM Users WHERE role='student' AND status='active'"
        ).fetchall()

    student_status = []
    for s in all_students:
        display_name = (s['name'] or s['username']).strip()
        is_present = display_name.upper() in present_today
        student_status.append({
            'name': display_name,
            'is_present': is_present
        })

    return render_template('admin_edit.html', students=student_status)

@app.route('/admin/mark_absent', methods=['POST'])
@admin_required
def mark_absent():
    absent_students = request.form.getlist('absent_students')
    today = str(date.today())
    now_str = datetime.now().strftime('%H:%M')
    with get_db() as conn:
        for name in absent_students:
            conn.execute("INSERT OR IGNORE INTO Attendance (NAME,Time,Date,STATUS) VALUES (?,?,?,?)",
                         (name.upper(), now_str, today, 'Absent'))
    flash(f'Marked {len(absent_students)} student(s) as Absent.', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/reencode', methods=['POST'])
@admin_required
def admin_reencode():
    count = encode_all_images()
    log_audit(session['username'], 'RE_ENCODE', 'all', f'{count} faces encoded')
    flash(f'Re-encoded {count} face(s) successfully!', 'success')
    return redirect(url_for('admin_dashboard'))

# ─── Reports & Exports ────────────────────────────────────────────────────────

@app.route('/data', methods=['GET', 'POST'])
@admin_required
def data():
    if request.method == 'POST':
        rows = get_today_attendance()
        return render_template('admin/today_attendance.html', rows=rows)
    return render_template('admin/today_attendance.html', rows=get_today_attendance())

@app.route('/whole')
@admin_required
def whole():
    rows = get_all_attendance()
    return render_template('admin/whole_attendance.html', rows=rows)

@app.route('/export/csv')
@admin_required
def export_csv():
    import pandas as pd
    rows = get_all_attendance()
    data = [{'Date': str(r['Date']), 'Name': r['NAME'], 'Time': r['Time'], 'Status': r['STATUS']} for r in rows]
    df = pd.DataFrame(data)
    
    # Explicitly ensure column order and fix encoding for Excel
    if not df.empty:
        df = df[['Date', 'Name', 'Time', 'Status']]
    
    output = io.StringIO() # Use StringIO for text-based CSV
    df.to_csv(output, index=False, encoding='utf-8-sig')
    
    buf = io.BytesIO()
    buf.write(output.getvalue().encode('utf-8-sig'))
    buf.seek(0)
    
    return send_file(buf, mimetype='text/csv',
                     as_attachment=True, download_name='attendance_report.csv')

# ─── Live Analytics (PowerBI Replacement) ──────────────────────────────────────

@app.route('/dashboard')
@admin_required
def dashboard():
    """Live Analytics Dashboard (replaces external PowerBI iframe)."""
    # 1. Weekly stats
    labels, present_counts, absent_counts = [], [], []
    for i in range(6, -1, -1):
        d = str(date.today() - timedelta(days=i))
        labels.append(d)
        with get_db() as conn:
            p = conn.execute("SELECT COUNT(*) FROM Attendance WHERE Date=? AND STATUS='Present'", (d,)).fetchone()[0]
            a = conn.execute("SELECT COUNT(*) FROM Attendance WHERE Date=? AND STATUS='Absent'", (d,)).fetchone()[0]
        present_counts.append(p)
        absent_counts.append(a)

    # 2. Overall distribution
    with get_db() as conn:
        total_p = conn.execute("SELECT COUNT(*) FROM Attendance WHERE STATUS='Present'").fetchone()[0]
        total_a = conn.execute("SELECT COUNT(*) FROM Attendance WHERE STATUS='Absent'").fetchone()[0]
    
    # 3. Student performance (top 10)
    students = get_all_students()
    perf = []
    for s in students:
        sname = (s['name'] or s['username']).upper()
        total, pre, abs_, pct, _ = get_student_stats(sname)
        if total > 0:
            perf.append({'name': sname, 'pct': pct})
    
    perf = sorted(perf, key=lambda x: x['pct'], reverse=True)[:10]

    return render_template('dashboard.html',
                           labels=json.dumps(labels),
                           present_counts=json.dumps(present_counts),
                           absent_counts=json.dumps(absent_counts),
                           total_p=total_p, total_a=total_a,
                           perf=perf)

# ─── Error Handlers ───────────────────────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', code=404, message="Page Not Found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', code=500, message="Internal Server Error"), 500

# ─── Startup ─────────────────────────────────────────────────────────────────

init_db()
load_encodings()

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'True').lower() == 'true'
    app.run(debug=debug, host=host, port=port)
