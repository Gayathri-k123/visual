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
        # --- 1. MODEL & MEDIAPIPE SETUP ---
        self.model_path = 'engagement_model.pkl'
        self.model = None
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        self.mp_face_mesh = mp.solutions.face_mesh
        # refine_landmarks=True is CRITICAL for gaze tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=2, 
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5, 
            refine_landmarks=True)
        
        # --- 2. THRESHOLDS & LOGIC ---
        self.eye_closed_start = None
        self.SLEEP_TIME_THRESHOLD = 3.0  
        self.EAR_THRESHOLD = 0.20        
        self.HEAD_TILT_THRESHOLD = 0.13  
        # Gaze thresholds (0.5 is center, <0.35 or >0.65 is looking away)
        self.GAZE_LEFT_LIMIT = 0.38
        self.GAZE_RIGHT_LIMIT = 0.62

        # --- 3. DATA LOGGING ---
        self.session_data = []  
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def calculate_EAR(self, eye_points, landmarks):
        try:
            A = dist.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
            B = dist.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
            C = dist.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
            return (A + B) / (2.0 * C)
        except:
            return 0.0

    # --- NEW: GAZE CALCULATION FUNCTION ---
    def get_gaze_ratio(self, eye_points, iris_center, landmarks):
        try:
            # eye_points[0] is inner corner, eye_points[3] is outer corner
            # iris_center is the middle of the eyeball
            eye_width = dist.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
            iris_dist = dist.euclidean(landmarks[eye_points[0]], landmarks[iris_center])
            
            if eye_width == 0: return 0.5
            return iris_dist / eye_width
        except:
            return 0.5

    def stop_and_save(self):
        self.video.release()
        if len(self.session_data) > 0:
            if not os.path.exists('reports'):
                os.makedirs('reports')
            
            filename = f"reports/session_{self.start_time}.csv"
            df = pd.DataFrame(self.session_data, columns=['timestamp', 'status'])
            df.to_csv(filename, index=False)
            return filename
        return None

    def get_frame(self):
        success, frame = self.video.read()
        if not success: return None

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        status = "Searching..."
        box_color = (200, 200, 200)

        if results.multi_face_landmarks:
            if len(results.multi_face_landmarks) > 1:
                status = "WARNING: Multiple Faces!"
                box_color = (0, 0, 255)
                self.session_data.append({'timestamp': time.time(), 'status': 'Cheating'})
                cv2.rectangle(image, (0,0), (640, 60), box_color, -1)
                cv2.putText(image, status, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

            else:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, c = image.shape
                    face_2d = []
                    face_row = []
                    
                    for i, lm in enumerate(face_landmarks.landmark):
                        face_2d.append((int(lm.x * w), int(lm.y * h)))
                        if i < 468:
                            face_row.append(lm.x); face_row.append(lm.y); face_row.append(lm.z)

                    # --- HEAD TILT ---
                    nose_tip = face_landmarks.landmark[1]
                    chin = face_landmarks.landmark[152]
                    head_tilt = chin.y - nose_tip.y 

                    # --- EAR (Blinks) ---
                    LEFT_EYE = [33, 160, 158, 133, 153, 144]
                    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                    avgEAR = (self.calculate_EAR(LEFT_EYE, face_2d) + self.calculate_EAR(RIGHT_EYE, face_2d)) / 2.0

                    # --- NEW: GAZE DETECTION ---
                    # Left iris center: 468 | Right iris center: 473
                    left_gaze = self.get_gaze_ratio([133, 160, 158, 33], 468, face_2d)
                    right_gaze = self.get_gaze_ratio([362, 385, 387, 263], 473, face_2d)

                    is_side_looking = False
                    if left_gaze < self.GAZE_LEFT_LIMIT or left_gaze > self.GAZE_RIGHT_LIMIT or \
                       right_gaze < self.GAZE_LEFT_LIMIT or right_gaze > self.GAZE_RIGHT_LIMIT:
                        is_side_looking = True

                    # --- DECISION LOGIC ---
                    if avgEAR < self.EAR_THRESHOLD:
                        if head_tilt < self.HEAD_TILT_THRESHOLD:
                            status = "Typing/Reading"
                            box_color = (255, 255, 0)
                            self.eye_closed_start = None
                        else:
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
                        
                        # PRIORITY: Check if user is looking away with eyes first
                        if is_side_looking:
                            status = "Looking Away "
                            box_color = (0, 165, 255) # Orange
                        
                        elif self.model:
                            row_df = pd.DataFrame([face_row])
                            pred = self.model.predict(row_df)[0]
                            if pred == 'Distracted':
                                status = "Looking Away"
                                box_color = (0, 165, 255)
                            else:
                                status = "Attentive"
                                box_color = (0, 255, 0)
                        else:
                            status = "Attentive"
                            box_color = (0, 255, 0)

                    self.session_data.append({'timestamp': time.time(), 'status': status})
                    cv2.rectangle(image, (0,0), (450, 60), box_color, -1)
                    cv2.putText(image, status, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()