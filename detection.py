import cv2
import mediapipe as mp
import pandas as pd
import pickle
import numpy as np
from scipy.spatial import distance as dist
import os
import time
from datetime import datetime

class VideoCamera(object):
    def __init__(self):
        # --- MODEL & MEDIAPIPE SETUP ---
        self.model_path = 'engagement_model.pkl'
        self.model = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
        
        # --- SLEEP LOGIC ---
        self.eye_closed_start = None
        self.SLEEP_TIME_THRESHOLD = 1.5

        # --- NEW: DATA LOGGING ---
        self.session_data = []  # List to store frame data
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def calculate_EAR(self, eye_points, landmarks):
        # ... (Keep your existing EAR code here) ...
        try:
            A = dist.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
            B = dist.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
            C = dist.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
            return (A + B) / (2.0 * C)
        except:
            return 0.0

    # --- NEW: FUNCTION TO SAVE CSV ---
    def stop_and_save(self):
        self.video.release()
        if len(self.session_data) > 0:
            if not os.path.exists('reports'):
                os.makedirs('reports')
            
            filename = f"reports/session_{self.start_time}.csv"
            df = pd.DataFrame(self.session_data, columns=['timestamp', 'status'])
            df.to_csv(filename, index=False)
            return filename # Return the filename so Flask knows what to read
        return None

    def get_frame(self):
        success, frame = self.video.read()
        if not success: return None

        # ... (Standard Preprocessing) ...
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        status = "Searching..."
        box_color = (200, 200, 200)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # ... (Landmark extraction logic) ...
                h, w, c = image.shape
                face_2d = []
                face_row = []
                for lm in face_landmarks.landmark:
                    face_2d.append((int(lm.x * w), int(lm.y * h)))
                for i in range(468):
                    lm = face_landmarks.landmark[i]
                    face_row.append(lm.x); face_row.append(lm.y); face_row.append(lm.z)

                # --- CALCULATE STATUS ---
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                leftEAR = self.calculate_EAR(LEFT_EYE, face_2d)
                rightEAR = self.calculate_EAR(RIGHT_EYE, face_2d)
                avgEAR = (leftEAR + rightEAR) / 2.0

                if avgEAR < 0.25:
                    if self.eye_closed_start is None:
                        self.eye_closed_start = time.time()
                    elapsed = time.time() - self.eye_closed_start
                    
                    if elapsed >= self.SLEEP_TIME_THRESHOLD:
                        status = "Sleeping"
                        box_color = (0, 0, 255)
                    else:
                        status = "Blinking"
                        box_color = (255, 255, 0)
                else:
                    self.eye_closed_start = None
                    if self.model:
                        row_df = pd.DataFrame([face_row])
                        pred = self.model.predict(row_df)[0]
                        if pred == 'Distracted':
                            status = "Looking Away"
                            box_color = (0, 165, 255)
                        else:
                            status = "Attentive"
                            box_color = (0, 255, 0)

                # --- NEW: SAVE STATUS TO LIST ---
                self.session_data.append({
                    'timestamp': time.time(),
                    'status': status
                })

                # Drawing
                cv2.rectangle(image, (0,0), (320, 60), box_color, -1)
                cv2.putText(image, status, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()