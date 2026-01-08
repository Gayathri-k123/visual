import cv2
import mediapipe as mp
import csv
import os

# Setup MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# CSV File Name
csv_file = 'engagement_dataset.csv'

# Create CSV File with Headers if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        headers = []
        for i in range(468): # Face Mesh has 468 points
            headers += [f'x{i}', f'y{i}', f'z{i}']
        headers.append('class') # The label (Attentive/Distracted)
        writer.writerow(headers)

cap = cv2.VideoCapture(0)

print("--- INSTRUCTIONS ---")
print("1. Look at screen and HOLD 'a' to record 'Attentive'")
print("2. Look away and HOLD 'd' to record 'Distracted'")
print("3. Press 'q' to Quit")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Convert to RGB and process
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw Face Mesh
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Check for Key Press
            k = cv2.waitKey(1)
            if k == ord('a') or k == ord('d'):
                row = []
                for lm in face_landmarks.landmark:
                    row.append(lm.x)
                    row.append(lm.y)
                    row.append(lm.z)
                
                if k == ord('a'):
                    row.append('Attentive')
                    cv2.putText(image, "RECORDING: ATTENTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif k == ord('d'):
                    row.append('Distracted')
                    cv2.putText(image, "RECORDING: DISTRACTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Save to File
                with open(csv_file, 'a', newline='') as f:
                    csv.writer(f).writerow(row)

    cv2.imshow('Step 1: Data Collection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()