import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Import the Camera Logic
try:
    from detection import VideoCamera
except ImportError:
    print("WARNING: detection.py not found. Camera will not work.")
    class VideoCamera: pass

app = Flask(__name__)
app.secret_key = "mca_project_secret_key"

# --- DATABASE SETUP ---
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_path
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- USER MODEL ---
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
        
        # Check if user exists
        if User.query.filter_by(email=email).first():
            flash("Email already exists. Please login.")
            return redirect(url_for('register'))
        
        # Save new user
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

# --- CAMERA FEED ---
def gen(camera):
    while True:
        # Get frame from detection.py
        frame = camera.get_frame() 
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/monitor')
def monitor():
    # Security check: User must be logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('monitor.html')
if __name__ == '__main__':
    app.run(debug=True)