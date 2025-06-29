import React, { useState } from "react";
import "./App.css"; // CSS’in aynen kullanılmaya devam ediliyor

function App() {
  const [question, setQuestion] = useState("");
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    if (!question.trim()) return;
    setLoading(true);
    const currentTime = new Date().toLocaleTimeString();
    const userEntry = { question, answer: null, sources: [], time: currentTime };
    setHistory([...history, userEntry]);

    try {
      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      const data = await res.json();
      setHistory(prev => {
        const newHistory = [...prev];
        newHistory[newHistory.length - 1] = {
          question,
          answer: data.answer,
          sources: data.sources,
          time: currentTime
        };
        return newHistory;
      });
    } catch (e) {
      setHistory(prev => {
        const newHistory = [...prev];
        newHistory[newHistory.length - 1].answer = "Bir hata oluştu. Lütfen tekrar deneyin.";
        return newHistory;
      });
    } finally {
      setLoading(false);
      setQuestion("");
    }
  };

  const handleRetry = originalIdx => {
    setQuestion(history[originalIdx].question);
    setHistory(prev => prev.filter((_, i) => i !== originalIdx));
  };

  return (
    <div className="app-container">
      <div className="chat-box">
        <header style={{ textAlign: "center", marginBottom: 16 }}>
          <h1>Hyundai i20 Asistan</h1>
        </header>

        <div style={{ display: "flex", marginBottom: 12 }}>
          <input
            style={{ flex: 1, padding: 10, fontSize: 16 }}
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Sorunuzu yazın..."
          />
          <button
            style={{ marginLeft: 8, padding: "10px 16px" }}
            onClick={ask}
            disabled={loading || !question.trim()}
          >
            Sor
          </button>
        </div>

        <div>
          {history.slice().reverse().map((h, revIdx) => {
            const originalIdx = history.length - 1 - revIdx;
            return (
              <div key={originalIdx} style={{ marginBottom: 24 }}>
                <div style={{ textAlign: "right" }}>
                  <div className="message-user">{h.question}</div>
                  <div style={{ fontSize: 12, color: "#ddd", marginTop: 4 }}>{h.time}</div>
                </div>

                <div style={{ textAlign: "left", marginTop: 8 }}>
                  {h.answer ? (
                    <div className="message-bot">
                      <div style={{ whiteSpace: "pre-wrap" }}>{h.answer}</div>
                      <h4 style={{ marginTop: 12 }}>Kaynaklar:</h4>
                      <ul style={{ paddingLeft: 20 }}>
                        {h.sources.map((src, sidx) => (
                          <li key={sidx}>
                            <b>Sayfa {src.page || "?"}:</b>{" "}
                            {src.text.length > 200 ? src.text.slice(0, 200) + "..." : src.text}
                          </li>
                        ))}
                      </ul>
                      <button
                        style={{
                          marginTop: 8,
                          padding: "6px 12px",
                          background: "#e0e0e0",
                          border: "none",
                          borderRadius: 4,
                          cursor: "pointer"
                        }}
                        onClick={() => handleRetry(originalIdx)}
                      >
                        Tekrar Sor
                      </button>
                    </div>
                  ) : h.answer === null ? (
                    revIdx === 0 && loading ? <div>Cevap aranıyor...</div> : null
                  ) : (
                    <div style={{ color: "red" }}>{h.answer}</div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default App;
