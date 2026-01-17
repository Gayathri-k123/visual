import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) == 13:  # Press Enter to exit
        break

cap.release()
cv2.destroyAllWindows()
