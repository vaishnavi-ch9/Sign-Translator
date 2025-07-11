import { useEffect, useRef, useState } from "react";
import Webcam from "react-webcam";

/* poll every 700 ms */
const POLL_MS = 700;

export default function App() {
  const camRef = useRef(null);
  const [word, setWord]         = useState("—");
  const [conf, setConf]         = useState(0);
  const [sentence, setSentence] = useState("");
  const lastSentenceRef         = useRef("");     // for TTS dedupe

  /* poll backend */
  useEffect(() => {
    const id = setInterval(async () => {
      if (!camRef.current) return;
      const shot = camRef.current.getScreenshot();
      if (!shot) return;

      try {
        const r = await fetch("http://127.0.0.1:8000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: shot }),
        });
        const j = await r.json();
        setWord(j.label);
        setConf((j.conf * 100).toFixed(1));
        setSentence(j.sentence);

        /* speak only when the sentence actually changes */
        if (j.sentence && j.sentence !== lastSentenceRef.current) {
          lastSentenceRef.current = j.sentence;
          const utter = new SpeechSynthesisUtterance(j.sentence);
          speechSynthesis.cancel();
          speechSynthesis.speak(utter);
        }
      } catch (err) {
        console.error(err);
      }
    }, POLL_MS);
    return () => clearInterval(id);
  }, []);

  /* clear sentence */
  const clearSentence = async () => {
    setSentence("");
    lastSentenceRef.current = "";
    await fetch("http://127.0.0.1:8000/clear", { method: "POST" });
  };

  return (
    <div style={styles.page}>
      <h1>🖐️ Sign‑Language Sentence Translator</h1>

      <Webcam
        ref={camRef}
        videoConstraints={{ facingMode: "user" }}
        screenshotFormat="image/jpeg"
        style={styles.cam}
      />

      <h3>
        📸 Word:&nbsp;
        <span style={{ color: "green" }}>{word}</span>{" "}
        {conf > 0 && <span>({conf}%)</span>}
      </h3>

      <h3 style={{ marginTop: 20 }}>📝 Sentence</h3>
      <div style={styles.box}>{sentence || "Waiting for gestures…"}</div>

      <button onClick={clearSentence} style={styles.btn}>
        🧹 Clear
      </button>
    </div>
  );
}

const styles = {
  page: {
    fontFamily: "Arial, sans-serif",
    textAlign: "center",
    padding: 32,
    maxWidth: 720,
    margin: "0 auto",
  },
  cam: {
    width: 420,
    borderRadius: 12,
    boxShadow: "0 0 12px rgba(0,0,0,0.25)",
    marginBottom: 16,
  },
  box: {
    background: "#eee",
    borderRadius: 10,
    padding: 16,
    minHeight: 48,
    fontSize: 20,
  },
  btn: {
    marginTop: 12,
    padding: "8px 24px",
    fontSize: 16,
    borderRadius: 8,
    border: "none",
    background: "#ff5c5c",
    color: "#fff",
    cursor: "pointer",
  },
};
