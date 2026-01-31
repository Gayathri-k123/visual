import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# --- 1. IMPORT ANALYTICS & CAMERA ---
try:
    from detection import VideoCamera
    # THIS LINE IS CRITICAL: We import BOTH functions here
    from analytics import calculate_engagement, get_all_reports 
except ImportError:
    print("WARNING: detection.py or analytics.py not found.")
    class VideoCamera: pass
    def calculate_engagement(f): return 0
    def get_all_reports(): return []

app = Flask(__name__)
app.secret_key = "mca_project_secret_key"

# --- 2. GLOBAL CAMERA VARIABLE ---
# This holds the camera so we can stop it later
global_camera = None 

# --- DATABASE SETUP ---
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# USER MODEL 
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

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

# --- STOP ANALYSIS & SHOW REPORT ---
@app.route('/stop_analysis')
def stop_analysis():
    global global_camera 
    final_score = 0
    filename = None
    
    if global_camera:
        # 1. Stop Camera and Save CSV
        filename = global_camera.stop_and_save()
        
        # 2. Reset Camera
        global_camera = None 
        
        # 3. Calculate Score
        if filename:
            final_score = calculate_engagement(filename)
    
    # 4. Show Report Page
    return render_template('report.html', score=final_score)

# --- ARCHIVES ROUTE ---
@app.route('/archives')
def archives():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # 1. Get list of files from analytics.py
    past_reports = get_all_reports()
    
    # 2. Show the list
    return render_template('archives.html', reports=past_reports)

if __name__ == '__main__':
    app.run(debug=True)