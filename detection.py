import cv2
import mediapipe as mp
import pandas as pd
import pickle
import numpy as np
from scipy.spatial import distance as dist
import os

class VideoCamera(object):
    def __init__(self):
        # --- 1. SETUP & LOAD MODEL ---
        self.model_path = 'engagement_model.pkl'
        self.model = None

        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            print("WARNING: Model not found. Head tracking disabled.")

        #  SETUP MEDIAPIPE 
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            refine_landmarks=True 
        )
        
        # Start Camera
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    #  HELPER FUNCTION: EAR 
    def calculate_EAR(self, eye_points, landmarks):
        try:
            A = dist.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
            B = dist.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
            C = dist.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
            return (A + B) / (2.0 * C)
        except:
            return 0.0

    def get_frame(self):
        success, frame = self.video.read()
        if not success:
            return None

        # Convert to RGB
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        status = "Searching..."
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # --- A. PREPARE DATA ---
                h, w, c = image.shape
                face_2d = [] 
                face_row = [] 
                
                # Get Pixel Coords for Math
                for lm in face_landmarks.landmark:
                    face_2d.append((int(lm.x * w), int(lm.y * h)))

                # Get Normalized Coords for ML (Only first 468 points)
                for i in range(468):
                    lm = face_landmarks.landmark[i]
                    face_row.append(lm.x)
                    face_row.append(lm.y)
                    face_row.append(lm.z)

                # CHECK EYES 
                LEFT_EYE = [33, 160, 158, 133, 153, 144] 
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]

                leftEAR = self.calculate_EAR(LEFT_EYE, face_2d)
                rightEAR = self.calculate_EAR(RIGHT_EYE, face_2d)
                avgEAR = (leftEAR + rightEAR) / 2.0

                if avgEAR < 0.25:
                    status = " Sleeping"
                    box_color = (0, 0, 255) # Red
                else:
                    # CHECK HEAD POSE (ML) 
                    if self.model:
                        row_df = pd.DataFrame([face_row])
                        prediction = self.model.predict(row_df)[0]
                        
                        if prediction == 'Distracted':
                            status = "Looking Away"
                            box_color = (0, 165, 255) # Orange
                        else:
                            status = "Attentive"
                            box_color = (0, 255, 0) # Green
                    else:
                        status = "Model Missing"

                
                # Draw Box
                cv2.rectangle(image, (0,0), (320, 60), (245, 117, 16), -1)
                # Draw Text
                cv2.putText(image, status, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode the frame so Flask can read it
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()