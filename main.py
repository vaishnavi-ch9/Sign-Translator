# main.py – Continuous‑sentence sign translator (lower threshold & 2‑sec cooldown)
import cv2, mediapipe as mp, joblib, pyttsx3, time
from collections import deque, Counter
from pathlib import Path

# ───── Load model & TTS ─────
model  = joblib.load(Path("models/gesture_model.pkl"))
engine = pyttsx3.init()
def speak(txt): engine.say(txt); engine.runAndWait()

# ───── MediaPipe hands ─────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1)
drawer   = mp.solutions.drawing_utils

# ───── Parameters ─────
FRAME_WINDOW   = 20     # # of frames to vote over
CONF_THRESHOLD = 0.45   # accept if ≥ 45 % confident
MAJORITY_RATIO = 0.50   # word must be ≥ 50 % of confident frames
COOLDOWN_SEC   = 2      # can repeat same word after 2 s

pred_buffer = deque(maxlen=FRAME_WINDOW)
sent_words  = []

cap = cv2.VideoCapture(0)
last_announced, last_time = None, 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for lm_set in res.multi_hand_landmarks:
            row   = [c for lm in lm_set.landmark for c in (lm.x, lm.y, lm.z)]
            probs = model.predict_proba([row])[0]
            best  = probs.argmax()
            word  = model.classes_[best]
            conf  = probs[best]

            # optional: print confidences to console for debugging
            print({cls: f"{p:.2f}" for cls, p in zip(model.classes_, probs)})

            drawer.draw_landmarks(frame, lm_set, mp_hands.HAND_CONNECTIONS)
            pred_buffer.append((word, conf))

            # draw raw prediction
            cv2.putText(frame, f"{word} {conf:.2f}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (180, 255, 180) if conf >= CONF_THRESHOLD else (0, 0, 255), 2)
            break   # process only first detected hand

    # ── Majority‑vote smoothing ──
    if len(pred_buffer) == FRAME_WINDOW:
        confident = [w for w, c in pred_buffer if c >= CONF_THRESHOLD]
        if confident:
            maj_word, maj_ct = Counter(confident).most_common(1)[0]
            if maj_ct / len(confident) >= MAJORITY_RATIO:
                now = time.time()
                if maj_word != last_announced or (now - last_time) > COOLDOWN_SEC:
                    sent_words.append(maj_word)
                    speak(maj_word)
                    last_announced, last_time = maj_word, now
                pred_buffer.clear()

    # ── Display running sentence ──
    sentence = " ".join(sent_words[-10:])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (0, 0, 0), -1)
    cv2.putText(frame, sentence, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    cv2.imshow("Vaish's Sentence Translator (q quit, c clear)", frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        sent_words.clear()
        last_announced, last_time = None, 0
        pred_buffer.clear()

cap.release()
cv2.destroyAllWindows()
