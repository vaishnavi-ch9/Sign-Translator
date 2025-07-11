# app.py  –  Collect labelled hand‑landmark samples
import streamlit as st
import cv2, mediapipe as mp, csv, os, time
from datetime import datetime

# ──────────────────────────────
# Persist a simple gesture log
# ──────────────────────────────
if "gesture_log" not in st.session_state:
    st.session_state["gesture_log"] = []

# ───────────── UI ─────────────
st.set_page_config(page_title="Gesture Collector", layout="wide")
st.title("🖐️  Real‑Time Gesture Collector")

label   = st.selectbox("Choose / type a gesture label", ["hello", "yes", "no", "love"])
samples = st.slider("Samples to collect", 50, 400, 200, 50)
start   = st.button("Start collection")

# ─────────── Logic ────────────
if start:
    cap        = cv2.VideoCapture(0)
    mp_hands   = mp.solutions.hands
    hands      = mp_hands.Hands(max_num_hands=1)
    mp_draw    = mp.solutions.drawing_utils
    save_dir   = os.path.join("dataset")
    os.makedirs(save_dir, exist_ok=True)
    save_path  = os.path.join(save_dir, f"{label}_{int(time.time())}.csv")
    csv_file   = open(save_path, "w", newline="")
    writer     = csv.writer(csv_file)

    count      = 0
    frame_ph   = st.empty()
    status_ph  = st.empty()
    ts_start   = datetime.now().strftime("%H:%M:%S")
    st.session_state["gesture_log"].append(f"{ts_start} – started **{label}**")

    while cap.isOpened() and count < samples:
        ret, frame = cap.read()
        if not ret:
            break

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res  = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                row = [c for lm in hand.landmark for c in (lm.x, lm.y, lm.z)]
                writer.writerow(row)
                count += 1
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        frame_ph.image(frame, channels="BGR", caption=f"{count}/{samples} frames")
        status_ph.info(f"Collecting **{label}** — {count}/{samples}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    csv_file.close()
    status_ph.success(f"Saved {count} samples to `{os.path.abspath(save_path)}` ✅")
    ts_end = datetime.now().strftime("%H:%M:%S")
    st.session_state["gesture_log"].append(f"{ts_end} – finished **{label}** ({count})")

# ─────────── Log panel ─────────
with st.expander("📝 Gesture Log", expanded=True):
    for entry in st.session_state["gesture_log"][::-1]:
        st.markdown(f"- {entry}")
    if st.button("🗑️ Clear log"):
        st.session_state["gesture_log"].clear()
        st.success("Log cleared")
