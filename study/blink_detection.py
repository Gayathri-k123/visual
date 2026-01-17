import cv2
import mediapipe as mp
import numpy as np

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

cap = cv2.VideoCapture(0)

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ---------------- Variables ----------------
blink_count = 0
eye_closed = False
EAR_THRESHOLD = 0.25

frame_count = 0
blink_rate = 0
engagement_status = "Analyzing"

# ---------------- Main loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
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

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # -------- Blink detection --------
        if ear < EAR_THRESHOLD:
            if not eye_closed:
                blink_count += 1
                eye_closed = True
            eye_status = "EYES CLOSED"
        else:
            eye_closed = False
            eye_status = "EYES OPEN"

        # -------- Blink rate & engagement (every ~3 seconds) --------
        if frame_count % 90 == 0:
            blink_rate = blink_count
            blink_count = 0

            if blink_rate <= 5:
                engagement_status = "Attentive"
            elif blink_rate <= 12:
                engagement_status = "Possibly Tired"
            else:
                engagement_status = "Low Attention"

        # -------- Display text --------
        cv2.putText(frame, f"Eye Status: {eye_status}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 0, 0), 2)

        cv2.putText(frame, f"Blink Rate: {blink_rate}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)

        cv2.putText(frame, f"Engagement: {engagement_status}",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "Face Not Detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)
        engagement_status = "Not Attentive"

    cv2.imshow("Blink Detection with Engagement", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
