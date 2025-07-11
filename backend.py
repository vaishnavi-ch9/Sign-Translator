# backend.py – sentence‑level sign‑language API (dynamic mode + debug)

import base64, cv2, numpy as np, joblib, time
from flask import Flask, request, jsonify
from flask_cors import CORS
import mediapipe as mp
from pathlib import Path
from collections import deque

# ── Flask app ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)                                # allow React calls (localhost)

# ── Load trained model ────────────────────────────────────
MODEL_PATH = Path("models/gesture_model.pkl")
model   = joblib.load(MODEL_PATH)
classes = model.classes_.tolist()
print("Loaded model with classes:", classes)

# ── MediaPipe hands (dynamic mode) ────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(max_num_hands=1)        # dynamic tracking ✅

# ── Sentence buffer & cooldown ────────────────────────────
buffer          = deque(maxlen=20)
last_word       = None
last_timestamp  = 0
COOLDOWN_SEC    = 2

# ── Helper functions ─────────────────────────────────────
def b64_to_cv2(b64: str):
    _, enc = b64.split(",", 1) if "," in b64 else ("", b64)
    img = cv2.imdecode(np.frombuffer(base64.b64decode(enc), np.uint8),
                       cv2.IMREAD_COLOR)
    return img

def xyz_feats(bgr):
    res = hands.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        print("⚠️  No hand detected.")
        return None
    lm = res.multi_hand_landmarks[0]
    return [c for p in lm.landmark for c in (p.x, p.y, p.z)]  # 63 features

def beautify(txt: str):
    txt = txt.strip()
    if not txt:
        return ""
    txt = txt.replace(" i ", " I ").replace(" i'", " I'")
    if not txt.endswith("."):
        txt += "."
    return txt.capitalize()

# ── Routes ────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    global last_word, last_timestamp
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "no image"}), 400

    feats = xyz_feats(b64_to_cv2(data["image"]))
    if feats is None:
        return jsonify({
            "label": "none",
            "conf": 0.0,
            "sentence": beautify(" ".join(buffer))
        })

    probs = model.predict_proba([feats])[0]
    print("Model probs:", dict(zip(classes, probs.round(2))))  # debug

    idx, conf = int(np.argmax(probs)), float(np.max(probs))
    label     = classes[idx]

    now = time.time()
    if conf >= 0.60 and (label != last_word or now - last_timestamp > COOLDOWN_SEC):
        buffer.append(label)
        last_word, last_timestamp = label, now

    return jsonify({
        "label": label,
        "conf":  conf,
        "sentence": beautify(" ".join(buffer))
    })

@app.route("/clear", methods=["POST"])
def clear_sentence():
    buffer.clear()
    return jsonify({"status": "cleared"})

# ── Start server ──────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
