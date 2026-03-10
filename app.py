import os
from datetime import datetime  # <--- NEW: Required for timestamps
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response,jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from report_generator import generate_pdf_bytes

# --- 1. IMPORT ANALYTICS & CAMERA ---
try:
    from detection import VideoCamera
    # We only need calculate_engagement now. We don't need get_all_reports anymore.
    from analytics import calculate_engagement 
except ImportError:
    print("WARNING: detection.py or analytics.py not found.")
    class VideoCamera: pass
    def calculate_engagement(f): return 0

app = Flask(__name__)
app.secret_key = "mca_project_secret_key"

# --- 2. GLOBAL CAMERA VARIABLE ---
global_camera = None 

# --- DATABASE SETUP ---
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- MODELS ---

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    # Relationship to access reports easily (optional but good practice)
    reports = db.relationship('Report', backref='author', lazy=True)

# --- NEW CLASS: REPORT ---
# This table stores the history for each user
class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    # Link this report to a specific user
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)

# Create DB if not exists
with app.app_context():
    db.create_all()

# --- ROUTES ---

@app.route('/')
def home():
    if 'user_id' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(email=email).first():
            flash("Email already exists. Please login.")
            return redirect(url_for('register'))
        
        hashed_pw = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_pw)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration Successful! Please Login.")
        return redirect(url_for('login'))
        
    return render_template('register_dark.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home'))
        else:
            flash("Invalid Email or Password")
            
    return render_template('login_dark.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# --- CAMERA LOGIC ---

def gen(camera):
    while True:
        frame = camera.get_frame() 
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

@app.route('/monitor')
def monitor():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('monitor.html')

@app.route('/video_feed')
def video_feed():
    global global_camera 
    if global_camera is None:
        global_camera = VideoCamera()
    return Response(gen(global_camera), mimetype='multipart/x-mixed-replace; boundary=frame')


# --- STOP ANALYSIS & SAVE TO DB ---
@app.route('/stop_analysis')
def stop_analysis():
    global global_camera 
    final_score = 0
    report_id = None
    
    violation_reason = request.args.get('violation')

    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    current_user_id = session['user_id']

    if global_camera:
        filename = global_camera.stop_and_save()
        global_camera = None 
        
        if filename:
            final_score = calculate_engagement(filename)
            
            # --- PENALTY LOGIC ---
            if violation_reason == 'tab_switch':
                print("Session terminated due to Tab Switching.")
                final_score = 0  # <--- THIS MUST BE UNCOMMENTED
            
            new_report = Report(
                user_id=current_user_id,
                filename=os.path.basename(filename),
                score=final_score
            )
            db.session.add(new_report)
            db.session.commit()
            report_id = new_report.id

    return render_template('report.html', score=final_score, report_id=report_id, violation=violation_reason)
# --- ARCHIVES ROUTE  ---
@app.route('/archives')
def archives():
    # 1. Security Check
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # 2. Query Database
    # Fetch only reports that match the current User ID
    user_reports = Report.query.filter_by(user_id=session['user_id']).order_by(Report.timestamp.desc()).all()
    
    # 3. Format Data for Template
    formatted_reports =[]
    for report in user_reports:
        formatted_reports.append({
            'id': report.id, # <--- FIXED: Added ID here so the download button works!
            'date': report.timestamp.strftime("%b %d, %Y %I:%M %p"), # Format: Feb 12, 2026 01:30 PM
            'score': report.score,
            'filename': report.filename
        })
    
    # 4. Send the filtered list
    return render_template('archives.html', reports=formatted_reports)

# --- HEATMAP DATA API ---
@app.route('/heatmap_data')
def heatmap_data():
    if 'user_id' not in session:
        return jsonify([]) 

    # 1. Get all reports for the current user
    reports = Report.query.filter_by(user_id=session['user_id']).all()

    # 2. Prepare a 7x24 Grid (7 Days, 24 Hours)
    # Day 0 = Mon, Day 6 = Sun
    data_grid = {} # Key: "Day-Hour", Value: {total_score, count}
    
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Initialize grid with 0
    for day in days:
        for hour in range(24):
            data_grid[f"{day}-{hour}"] = {'total': 0, 'count': 0}

    # 3. Fill with Database Data
    for r in reports:
        # Get Day Name (e.g., "Mon") and Hour (e.g., 14)
        day_name = r.timestamp.strftime("%a") 
        hour = r.timestamp.hour
        key = f"{day_name}-{hour}"
        
        if key in data_grid:
            data_grid[key]['total'] += r.score
            data_grid[key]['count'] += 1

    # 4. Format for ApexCharts
    series_data = []
    for day in days:
        day_points = []
        for hour in range(24):
            key = f"{day}-{hour}"
            avg = 0
            if data_grid[key]['count'] > 0:
                avg = round(data_grid[key]['total'] / data_grid[key]['count'], 1)
            
            day_points.append({
                'x': f"{hour}:00",
                'y': avg
            })
        series_data.append({'name': day, 'data': day_points})

    return jsonify(series_data)
@app.route('/download_report/<int:report_id>')
def download_report(report_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # 1. Get Data from DB
    report = Report.query.get_or_404(report_id)
    user = User.query.get(session['user_id'])
    
    # 2. Security Check
    if report.user_id != user.id:
        return "Unauthorized Access", 403

    # 3. Use the function from the other file
    pdf_content = generate_pdf_bytes(
        username=user.username,
        email=user.email,
        date_str=report.timestamp.strftime('%B %d, %Y'),
        time_str=report.timestamp.strftime('%I:%M %p'),
        score=report.score,
        report_id=report.id
    )

    # 4. Download the file
    return Response(
        pdf_content, 
        mimetype='application/pdf', 
        headers={'Content-Disposition': f'attachment;filename=Report_{report.id}.pdf'}
    )
if __name__ == '__main__':
    app.run(debug=True)