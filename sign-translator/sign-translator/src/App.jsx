import { useState, useRef } from "react";
import Webcam from "react-webcam";
import "./App.css";

export default function App() {
  const camRef = useRef(null);
  const [result, setResult] = useState("");

  const translate = async () => {
    const shot = camRef.current?.getScreenshot();
    if (!shot) return alert("Could not capture webcam frame 😢");

    try {
      const r = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: shot }),
      });
      const j = await r.json();
      setResult(`${j.label}  (${(j.conf * 100).toFixed(1)} %)`);
    } catch (err) {
      console.error(err);
      setResult("Server error");
    }
  };

  return (
    <div style={styles.page}>
      <h1 style={styles.h1}>🖐️ Sign Translator</h1>

      <Webcam
        ref={camRef}
        screenshotFormat="image/jpeg"
        videoConstraints={{ facingMode: "user" }}
        style={styles.cam}
      />

      <button style={styles.btn} onClick={translate}>📸 Translate</button>

      {result && <h2 style={styles.h2}>🗨️ Detected: {result}</h2>}
    </div>
  );
}

const styles = {
  page: { textAlign: "center", fontFamily: "sans-serif", padding: 24 },
  h1: { marginBottom: 20 },
  cam: { width: 400, borderRadius: 12, boxShadow: "0 0 12px rgba(0,0,0,0.3)" },
  btn: {
    marginTop: 16,
    padding: "10px 24px",
    fontSize: 18,
    borderRadius: 8,
    border: "none",
    background: "#4caf50",
    color: "#fff",
    cursor: "pointer",
  },
  h2: { marginTop: 24 },
};
