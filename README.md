# 🚀 Enterprise Retail Intelligence Platform

An end-to-end AI system that combines **recommendation systems**, **demand forecasting**, **ranking optimization**, and a **learning feedback loop** to power intelligent retail decision-making.

---

# 🧠 Overview

Modern retail systems struggle to align:

* What users **want** (personalization)
* What businesses should **stock** (demand)

This project solves that by integrating:

```
User Behavior + Demand Forecasting + Ranking Optimization
```

---

# ⚙️ System Architecture

```
User → Interaction → Feedback → Learning → Better Ranking
```

---

# 🔑 Key Features

## 🧠 1. Deep Learning Recommendations

* Two-Tower Neural Network (PyTorch)
* Learns user-product affinity

## 📈 2. Demand Forecasting

* Predicts product demand

## ⚖️ 3. Hybrid Ranking Engine

```
Final Score = w1 * affinity_score + w2 * forecast_norm
```

* Weights are **learned automatically**

## 🔁 4. Feedback Learning Loop

Tracks:

* Views
* Clicks
* Purchases

Used for:

* Model learning
* Ranking optimization

## 🤖 5. Ranking Optimization (Core)

### Regression-based

* Learns weights from data

### BPR (Pairwise Ranking)

* Learns ordering instead of scores
* Industry-standard approach

---

# 🗂 Project Structure

```
enterprise-retail-ai/
│
├── app/                     
│   ├── api.py
│   ├── routes/
│   │   ├── recommend.py
│   │   └── feedback.py
│   ├── services/
│   │   ├── ranking.py
│   │   └── recommendation.py
│   └── utils/
│       └── loaders.py
│
├── ml/                    
│   ├── models/
│   │   ├── two_tower_model.py
│   │   └── checkpoints/
│   │       ├── two_tower_1.ckpt
│   │       ├── two_tower_2.ckpt
│   │       └── two_tower_3.ckpt
│   │
│   ├── pipelines/
│   │   ├── ranking_pipeline.py
│   │   ├── generate_recs.py
│   │   └── forecast_pipeline.py
│   │
│   ├── training/
│   │   ├── train_bpr.py
│   │   └── retrain_model.py
│   │
│   └── features/
│       └── feature_engineering.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── feedback/
│
├── experiments/             
│   ├── metrics/
│   └── logs/
│
├── config/
│   └── ranking_weights.csv
│
├── frontend/               
│   ├── src/
│   ├── public/
│   └── package.json
│
├── notebooks/              
│
├── scripts/                
│   ├── run_pipeline.py
│   └── generate_global_recs.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🚀 Getting Started

## Backend

```
pip install -r requirements.txt
uvicorn serving.api:app --reload
```

## Frontend

```
cd frontend
npm install
npm start
```

---

# 📡 API

## Get Recommendations

```
GET /recommend?user_id=123
```

## Log Feedback

```
POST /feedback
```

---

# 🔄 Training

```
python scripts/generate_global_recs.py
python scripts/feedback_metrics.py
python scripts/ranking_optimization.py
python scripts/train_bpr.py
```

---

# 🧠 Key Idea

```
Learn from user behavior → update ranking → improve recommendations
```

---

# ⭐ Summary

A self-improving recommendation system that combines:

* Personalization
* Demand awareness
* Continuous learning

---
