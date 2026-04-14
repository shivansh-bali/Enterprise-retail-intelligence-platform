import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {

  const [userId, setUserId] = useState(10);
  const [recs, setRecs] = useState([]);

  const [analytics, setAnalytics] = useState(null);
  const [stock, setStock] = useState([]);
  const [summary, setSummary] = useState(null);

  const API = "http://127.0.0.1:8000";

  //  FEEDBACK FUNCTION
  const logFeedback = async (user_id, product_id, event) => {
    try {
      await axios.post(`${API}/feedback`, {
        user_id,
        product_id,
        event
      });
    } catch (err) {
      console.error("Feedback error");
    }
  };

  // ── FETCH RECS (TOGGLE) ──
  const fetchRecs = async () => {

    if (recs.length > 0) {
      setRecs([]);
      return;
    }

    const res = await axios.get(`${API}/recommend`, {
      params: { user_id: userId, top_k: 100 }
    });

    setRecs(res.data);
  };

  // ── ANALYTICS ──
  const fetchAnalytics = async () => {
    if (analytics) {
      setAnalytics(null);
      return;
    }
    const res = await axios.get(`${API}/analytics/products`);
    setAnalytics(res.data);
  };

  const fetchStock = async () => {
    if (stock.length > 0) {
      setStock([]);
      return;
    }
    const res = await axios.get(`${API}/analytics/stock`);
    setStock(res.data);
  };

  const fetchSummary = async () => {
    if (summary) {
      setSummary(null);
      return;
    }
    const res = await axios.get(`${API}/analytics/summary`);
    setSummary(res.data);
  };

  return (
    <div className="dashboard">

      <h1>Retail AI Dashboard</h1>

      {/* ── USER RECOMMENDATIONS ── */}
      <div className="section">
        <h2>Recommendations</h2>

        <input
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
        />

        <button onClick={fetchRecs}>
          {recs.length > 0 ? "Hide Recommendations" : "Get Recommendations"}
        </button>

        {recs.length === 0 && (
          <div className="muted">No recommendations yet</div>
        )}

        {recs.length > 0 && (
          <div className="grid">
            {recs.map((r, i) => (
              <div key={i} className="card">

                <div className="product-row">
                  <span className="product-icon">📦</span>
                  <span className="product-id">{r.product_id}</span>
                </div>
                <div className="muted">
                  Score: {r.final_score.toFixed(3)}
                </div>

                {/*  FEEDBACK BUTTONS */}
                <div style={{ marginTop: "10px", display: "flex", gap: "6px" }}>

                  <button
                    onClick={() => logFeedback(userId, r.product_id, "click")}
                  >
                    Click
                  </button>

                  <button
                    onClick={() => logFeedback(userId, r.product_id, "purchase")}
                  >
                    Buy
                  </button>

                </div>

              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── PRODUCT ANALYTICS ── */}
      <div className="section">
        <h2>Product Analytics</h2>

        <button onClick={fetchAnalytics}>
          {analytics ? "Hide Analytics" : "Load Analytics"}
        </button>

        {analytics && (
          <div className="grid">

            <div className="card">
              <h3>Top Products</h3>
              {analytics.top_products.map((p, i) => (
                <div key={i}>{p.product_id} → {p.final_score.toFixed(3)}</div>
              ))}
            </div>

            <div className="card">
              <h3>Top Demand</h3>
              {analytics.top_demand.map((p, i) => (
                <div key={i}>{p.product_id} → {p.forecast_qty.toFixed(2)}</div>
              ))}
            </div>

            <div className="card">
              <h3>High Impact</h3>
              {analytics.high_impact.map((p, i) => (
                <div key={i}>{p.product_id} → {p.impact.toFixed(2)}</div>
              ))}
            </div>

          </div>
        )}
      </div>

      {/* ── STOCK PLAN ── */}
      <div className="section">
        <h2>Stock Plan</h2>

        <button onClick={fetchStock}>
          {stock.length > 0 ? "Hide Stock Plan" : "Load Stock Plan"}
        </button>

        {stock.length > 0 && (
          <div className="grid">
            {stock.map((s, i) => (
              <div key={i} className="card">
                <div>{s.product_id}</div>
                <div className={`badge badge-${s.priority === "HIGH" ? "green" : "amber"}`}>
                  {s.priority}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* ── SUMMARY ── */}
      <div className="section">
        <h2>Business Summary</h2>

        <button onClick={fetchSummary}>
          {summary ? "Hide Summary" : "Load Summary"}
        </button>

        {summary && (
          <div className="grid">
            <div className="card">Products: {summary.total_products}</div>
            <div className="card">Avg Score: {summary.avg_score.toFixed(3)}</div>
            <div className="card">Avg Demand: {summary.avg_demand.toFixed(2)}</div>
            <div className="card">Avg Impact: {summary.avg_impact.toFixed(2)}</div>
          </div>
        )}
      </div>

    </div>
  );
}

export default App;