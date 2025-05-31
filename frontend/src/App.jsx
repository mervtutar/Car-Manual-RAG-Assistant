import React, { useState } from "react";

function App() {
  const [question, setQuestion] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);

  const ask = async () => {
    setLoading(true);
    const res = await fetch("http://localhost:3000/api/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    setResponse(data);
    setLoading(false);
  };

  return (
    <div style={{ maxWidth: 700, margin: "auto", fontFamily: "sans-serif" }}>
      <h1>Hyundai i20 Soru-Cevap Asistanı</h1>
      <input
        style={{ width: 400, padding: 8, fontSize: 18 }}
        value={question}
        onChange={e => setQuestion(e.target.value)}
        placeholder="Sorunuzu yazın..."
      />
      <button style={{ marginLeft: 16, padding: 8 }} onClick={ask} disabled={loading}>
        Sor
      </button>
      {loading && <div>Cevap aranıyor...</div>}
      {response && (
        <div style={{ marginTop: 32 }}>
          <h2>Cevap</h2>
          <div style={{ background: "#f7f7f7", padding: 16, borderRadius: 6 }}>
            {response.answer}
          </div>
          <h3>Kaynak Chunklar:</h3>
          <ul>
            {response.sources.map((src, idx) => (
              <li key={idx}><b>Sayfa {src.page}:</b> {src.text.slice(0, 120)}...</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default App;
