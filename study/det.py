import cv2
import mediapipe as mp
import pandas as pd
import pickle
import numpy as np
from scipy.spatial import distance as dist
import os

# --- 1. SETUP & LOAD MODEL ---
model_path = 'engagement_model.pkl'

if not os.path.exists(model_path):
    print("❌ ERROR: 'engagement_model.pkl' not found.")
    print("Please run train_model.py first.")
    exit()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# --- 2. SETUP MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
# refine_landmarks=True gives us detailed eye points (478 points total)
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_landmarks=True 
)

# --- 3. HELPER FUNCTION: CALCULATE EYE ASPECT RATIO (EAR) ---
def calculate_EAR(eye_points, landmarks):
    try:
        # Vertical distances (Eyelid height)
        A = dist.euclidean(landmarks[eye_points[1]], landmarks[eye_points[5]])
        B = dist.euclidean(landmarks[eye_points[2]], landmarks[eye_points[4]])
        # Horizontal distance (Eye width)
        C = dist.euclidean(landmarks[eye_points[0]], landmarks[eye_points[3]])
        # EAR Formula
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception as e:
        return 0.0

# --- 4. START CAMERA ---
cap = cv2.VideoCapture(0)

print("--------------------------------------")
print(" COMPLETE SYSTEM ACTIVE ")
print(" 1. Head Tracking (ML Model)")
print(" 2. Eye/Gaze Tracking (EAR Math)")
print(" Press 'q' to Quit")
print("--------------------------------------")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Convert to RGB
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    status = "Searching..."
    box_color = (128, 128, 128) # Gray
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            # --- A. PREPARE DATA ---
            h, w, c = image.shape
            face_2d = [] # For Eyes (Pixel math)
            face_row = [] # For Head (ML Model)
            
            # Populate Face 2D for Math (Use all points)
            for lm in face_landmarks.landmark:
                face_2d.append((int(lm.x * w), int(lm.y * h)))

            # Populate Face Row for ML (ONLY USE FIRST 468 POINTS)
            # This fixes the "1434 vs 1404" error
            for i in range(468):
                lm = face_landmarks.landmark[i]
                face_row.append(lm.x)
                face_row.append(lm.y)
                face_row.append(lm.z)

            # --- B. CHECK EYES (IS USER SLEEPING?) ---
            # Landmark indices for Left and Right Eye
            LEFT_EYE = [33, 160, 158, 133, 153, 144] 
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]

            leftEAR = calculate_EAR(LEFT_EYE, face_2d)
            rightEAR = calculate_EAR(RIGHT_EYE, face_2d)
            avgEAR = (leftEAR + rightEAR) / 2.0

            # THRESHOLD: If EAR < 0.25, Eyes are closed
            if avgEAR < 0.25:
                status = "Drowsy / Sleeping"
                box_color = (0, 0, 255) # Red (High Alert)
            
            else:
                # --- C. CHECK HEAD POSE (ML MODEL) ---
                # Now face_row has exactly 1404 features, so this will work
                row_df = pd.DataFrame([face_row])
                prediction = model.predict(row_df)[0]
                
                if prediction == 'Distracted':
                    status = "Looking Away"
                    box_color = (0, 165, 255) # Orange
                else:
                    status = "Attentive"
                    box_color = (0, 255, 0) # Green

            # --- D. DRAW UI ---
            # Draw the colored status box
            cv2.rectangle(image, (0,0), (320, 60), (245, 117, 16), -1)
            
            # Draw Status Text
            cv2.putText(image, status, (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Optional: Draw small value for EAR to debug
            cv2.putText(image, f"Eye: {round(avgEAR,2)}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Full Detection System', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()