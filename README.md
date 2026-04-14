# 🚀 Enterprise Retail Intelligence Platform

An end-to-end AI system that combines **recommendation systems**, **demand forecasting**, **ranking optimization**, and a **learning feedback loop** to power intelligent retail decision-making.

---

# 🧠 Overview

Modern retail systems struggle to align:

* What users **want** (personalization)
* What businesses should **stock** (demand)

This project solves that by integrating:

```text
User Behavior + Demand Forecasting + Ranking Optimization
```

---

# ⚙️ System Architecture

```text
                ┌──────────────────────┐
                │  User Interactions   │
                │ (view/click/buy)     │
                └─────────┬────────────┘
                          ↓
                ┌──────────────────────┐
                │   Feedback Logger    │
                └─────────┬────────────┘
                          ↓
        ┌──────────────────────────────────┐
        │                                  │
        ↓                                  ↓
┌──────────────────┐              ┌────────────────────┐
│ Recommendation   │              │ Demand Forecasting │
│ (Two-Tower DL)   │              │ (TFT Model)        │
└────────┬─────────┘              └────────┬───────────┘
         ↓                                 ↓
         └──────────────┬──────────────────┘
                        ↓
              ┌──────────────────────┐
              │ Ranking Engine      │
              │ (Weighted Scoring)  │
              └─────────┬────────────┘
                        ↓
              ┌──────────────────────┐
              │ API (FastAPI)        │
              └─────────┬────────────┘
                        ↓
              ┌──────────────────────┐
              │ React Dashboard      │
              └──────────────────────┘
                        ↓
              ┌──────────────────────┐
              │ Learning Pipeline    │
              │ (BPR / Regression)   │
              └──────────────────────┘
```

---

# 🔑 Key Features

## 🧠 1. Deep Learning Recommendations

* Two-Tower Neural Network (PyTorch)
* Learns user-product affinity
* Generates personalized recommendations

---

## 📈 2. Demand Forecasting

* Temporal Fusion Transformer (TFT)
* Predicts product demand
* Enables stock planning

---

## ⚖️ 3. Hybrid Ranking Engine

Combines:

```text
Final Score = w1 * affinity_score + w2 * forecast_norm
```

* Dynamic weights (learned from data)
* Handles missing forecast gracefully

---

## 🔁 4. Feedback Learning Loop

Tracks:

* 👁 Views
* 🖱 Clicks
* 🛒 Purchases

Used for:

* A/B testing
* Weight optimization
* Model improvement

---

## 🧪 5. Real A/B Testing

* Users split deterministically (A / B)
* Different ranking strategies tested
* Performance measured via:

  * CTR (Click-through rate)
  * Conversion rate

---

## 🤖 6. Ranking Optimization (Advanced)

### ✅ Regression-based learning

* Learns optimal weights automatically

### 🔥 BPR (Bayesian Personalized Ranking)

* Learns ranking instead of scores
* Uses (user, positive, negative) pairs
* Industry-standard approach

---

# 🗂 Project Structure

```bash
enterprise-retail-ai/
│
├── data/
│   ├── processed/
│   ├── feedback/
│   ├── metrics/
│   └── experiments/
│
├── recommender/
│   └── two_tower_model.py
│
├── serving/
│   └── api.py
│
├── scripts/
│   ├── generate_global_recs.py
│   ├── ab_testing.py
│   ├── feedback_metrics.py
│   ├── ranking_optimization.py
│   └── train_bpr.py
│
├── config/
│   └── ranking_weights.csv
│
├── frontend/
│   └── React dashboard
│
└── README.md
```

---

# 🚀 Getting Started

## 🔧 Backend Setup

```bash
pip install -r requirements.txt
```

Run API:

```bash
uvicorn serving.api:app --reload
```

---

## 💻 Frontend Setup

```bash
cd frontend
npm install
npm start
```

---

# 📡 API Endpoints

## 🔹 Get Recommendations

```http
GET /recommend?user_id=123
```

Response:

```json
{
  "group": "A",
  "recommendations": [
    {
      "product_id": "22035",
      "final_score": 0.93
    }
  ]
}
```

---

## 🔹 Log Feedback

```http
POST /feedback
```

```json
{
  "user_id": 123,
  "product_id": "22035",
  "event": "click"
}
```

---

# 🔄 Training & Optimization

## Generate Recommendations

```bash
python scripts/generate_global_recs.py
```

---

## Run A/B Testing

```bash
python scripts/ab_testing.py
```

---

## Compute Metrics

```bash
python scripts/feedback_metrics.py
```

---

## Learn Ranking Weights (Regression)

```bash
python scripts/ranking_optimization.py
```

---

## Train BPR Model (Advanced)

```bash
python scripts/train_bpr.py
```

---

# 🧠 Key Concepts

## 🔹 Affinity Score

* Learned from user behavior
* Output of deep learning model

## 🔹 Forecast Norm

* Normalized demand signal
* Computed using min-max scaling

## 🔹 Final Score

```text
Final Score = affinity_weight * affinity_score
            + forecast_weight * forecast_norm
```

---

# 📊 Metrics

| Metric          | Meaning                   |
| --------------- | ------------------------- |
| CTR             | Click-through rate        |
| Conversion Rate | Purchases / Clicks        |
| Engagement      | User-product interactions |

---

# ⚡ Performance Highlights

* ⚡ Real-time recommendation API
* 🧠 Self-improving ranking system
* 🔁 Continuous feedback loop
* 📊 Data-driven optimization

---

# 🚀 Future Improvements

* 🔥 XGBoost / LightGBM ranking
* 🧠 User-level personalization weights
* 📊 Real-time streaming pipeline
* 🏆 NDCG / MAP evaluation metrics

---

# 👨‍💻 Tech Stack

* **Backend:** Python, FastAPI
* **ML:** PyTorch, Scikit-learn
* **Frontend:** React
* **Data:** Pandas, NumPy

---

# 📌 Summary

```text
This project builds a complete intelligent retail system that:
- Recommends products
- Predicts demand
- Learns from user behavior
- Continuously improves ranking
```

---

# ⭐ Key Insight

```text
Personalization + Demand + Learning Loop = Smart Retail Intelligence
```

---
