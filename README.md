# Sign Language Translator

A real-time sign language sentence translator that uses computer vision and machine learning to convert hand gestures into complete English sentences. Built with Python (Flask) and React, this tool also speaks out the translated sentence for accessibility.

---

## Tech Stack

- **Frontend**: React, Webcam API, Vite, Text-to-Speech (TTS)
- **Backend**: Python, Flask, MediaPipe, scikit-learn
- **Model**: Trained on hand landmark features for gesture classification

---

## Features

- Live webcam-based gesture detection
- Converts gesture sequences into structured sentences
- Confidence scores for predictions
- Real-time translation with speech synthesis
- Minimal and intuitive UI

---

## 🗂️ Folder Structure

sign_language_translator/
│
├── backend.py # Flask API for gesture prediction
├── models/ # Trained ML model (gesture_model.pkl)
├── dataset/ # CSV data for each gesture
├── utils/ # Scripts for data collection & training
│
└── sign-translator/ # React frontend (Vite project)

yaml
Copy
Edit

---

## 🚀 How to Run the Project

### 1. Backend (Python Flask)
```bash
cd sign_language_translator
pip install -r requirements.txt
python backend.py
2. Frontend (React + Vite)
bash
Copy
Edit
cd sign-translator
npm install
npm run dev
Visit: http://localhost:5173

Training Your Own Gestures
Collect new gesture data using:

bash
Copy
Edit
python utils/data_collector.py
Then train the model:

bash
Copy
Edit
python utils/train_model.py
This creates gesture_model.pkl in the models/ folder.

Supported Gestures (Sample)
Gesture	Meaning
👋	Hello
❤️	Love
👍	Yes
👎	No

You can extend it with your own custom gestures.

License
This project is open-source and free to use. MIT License.

Author
Made by Vaishnavi
