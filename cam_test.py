import cv2

cap = cv2.VideoCapture(0)      # try index 0 first
if not cap.isOpened():
    print("❌ Webcam index 0 not found.")
    exit()

print("✅ Webcam opened.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed."); break
    cv2.imshow("Test Cam – press q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
