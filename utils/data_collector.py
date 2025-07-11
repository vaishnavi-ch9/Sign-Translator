import cv2
import mediapipe as mp
import csv
import os

# CONFIG
gesture_label = "hello"  # 👈 CHANGE this per gesture
samples_to_collect = 200
output_dir = "../dataset"

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Create output file
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, f"{gesture_label}.csv")
csv_file = open(csv_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

print(f"Starting data collection for '{gesture_label}'... Show your hand in front of the camera.")

while cap.isOpened() and count < samples_to_collect:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                csv_writer.writerow(landmarks)
                count += 1
                print(f"Sample {count}/{samples_to_collect}")

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Data Collector', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

print(f"\nCollection for '{gesture_label}' complete! Saved at: {csv_path}")
cap.release()
csv_file.close()
cv2.destroyAllWindows()
