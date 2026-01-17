from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

# ---------------- EAR calculation ----------------
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ---------------- MediaPipe setup ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.25

# ---------------- Global variables ----------------
blink_count = 0
frame_count = 0
blink_rate = 0
engagement_status = "Analyzing"
alert_count = 0
closed_frames = 0   # 👈 IMPORTANT
start_time = time.time()

cap = cv2.VideoCapture(0)

# ---------------- Video generator ----------------
def generate_frames():
    global blink_count, frame_count, blink_rate
    global engagement_status, alert_count, closed_frames

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        frame_count += 1

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape

            left_eye = np.array([
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in LEFT_EYE
            ])
            right_eye = np.array([
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in RIGHT_EYE
            ])

            ear = (eye_aspect_ratio(left_eye) +
                   eye_aspect_ratio(right_eye)) / 2.0

            # ---------------- Eye & Attention logic ----------------
            if ear < EAR_THRESHOLD:
                closed_frames += 1
                eye_status = "EYES CLOSED"

                # Eyes closed too long → inattentive
                if closed_frames > 45:  # ~1.5 seconds
                    engagement_status = "Low Attention"
                    alert_count += 1
            else:
                closed_frames = 0
                eye_status = "EYES OPEN"

            # Blink rate update every ~3 seconds
            if frame_count % 90 == 0:
                blink_rate = blink_count
                blink_count = 0

                if engagement_status != "Low Attention":
                    if blink_rate <= 5:
                        engagement_status = "Attentive"
                    elif blink_rate <= 12:
                        engagement_status = "Possibly Tired"
                    else:
                        engagement_status = "Low Attention"
                        alert_count += 1

            # ---------------- Display text ----------------
            cv2.putText(frame, f"Eye Status: {eye_status}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 0, 0), 2)

            cv2.putText(frame, f"Engagement: {engagement_status}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

            if engagement_status == "Low Attention":
                cv2.putText(frame, "⚠️ NOT ATTENTIVE",
                            (180, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 255), 3)

        else:
            engagement_status = "Not Attentive"
            closed_frames = 0
            cv2.putText(frame, "Face Not Detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ---------------- Flask routes ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(debug=False)
