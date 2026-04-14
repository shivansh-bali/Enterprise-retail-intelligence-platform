# 🚀 Enterprise Retail Intelligence Platform

An end-to-end AI system that combines **demand forecasting**, **deep learning recommendations**, **ranking optimization**, and a **business copilot** to power intelligent retail decision-making.

---

## 🧠 Overview

Modern retail systems struggle to align **what users want** with **what should be stocked**.

This platform solves that by integrating:

* 📈 Demand Forecasting (TFT)
* 🎯 Personalized Recommendations (Two-Tower Model)
* ⚖️ Hybrid Ranking (Affinity + Demand)
* 🔁 Feedback-driven Optimization (A/B Testing)
* 🤖 Business Copilot (Decision Intelligence)
* 📊 Executive Dashboard (Streamlit)
* ⚡ Real-time Serving (FastAPI)

---

## 🏗️ System Architecture

```
Raw Data
   ↓
Preprocessing
   ↓
Forecasting Engine (TFT)
   ↓
Candidate Generation (Collaborative Filtering)
   ↓
Ranking Model (Two-Tower)
   ↓
Hybrid Scoring (Affinity + Demand)
   ↓
Serving API (FastAPI)
   ↓
Dashboard (Streamlit)
   ↓
Business Copilot
```

---

## 🔑 Key Features

### 📈 Demand Forecasting

* Temporal Fusion Transformer (TFT)
* Multi-step time-series prediction
* Handles real-world sparse retail demand

---

### 🎯 Recommendation System

* Collaborative Filtering (candidate generation)
* Two-Tower Deep Learning Model
* Learns user-product embeddings

---

### ⚖️ Hybrid Ranking Engine

* Combines:

  * User affinity (ML model)
  * Forecasted demand (time-series)
* Config-driven weights (auto-optimized)

---

### 🔁 Optimization Loop

* Feedback logging (views, clicks, purchases)
* A/B testing engine
* Automatic ranking weight tuning

---

### 🤖 Business Copilot

Answer business questions like:

* Top demand products next week
* Which items need restocking
* High affinity + high demand products
* Over-recommended products

---

### 📊 Executive Dashboard

* Demand visualization
* Recommendation insights
* Experiment tracking
* Fusion analytics (Demand vs Affinity)

---

### ⚡ Real-Time API

* FastAPI-based serving
* Loads latest model artifacts
* Supports real-time recommendations

---

## 📂 Project Structure

```
enterprise-retail-intelligence-platform/

├── data/
│   ├── raw/
│   └── processed/

├── forecasting/
├── recommender/
├── serving/
├── intelligence/
├── dashboard/
├── copilot/
├── config/

├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ⚙️ Tech Stack

* **Python**
* **PyTorch** (Deep Learning)
* **PyTorch Forecasting (TFT)**
* **FastAPI** (Serving)
* **Streamlit** (Dashboard)
* **Pandas / NumPy**

---

## 🚀 Getting Started

### 1️⃣ Clone Repo

```bash
git clone https://github.com/your-username/enterprise-retail-intelligence-platform.git
cd enterprise-retail-intelligence-platform
```

---

### 2️⃣ Setup Environment

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Run API

```bash
uvicorn serving.api:app --reload
```

---

### 4️⃣ Run Dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📊 Example Insights

* Demand is highly skewed (long-tail distribution)
* Only a small % of SKUs drive majority of forecast demand
* Hybrid ranking improves prioritization vs standalone models

---

## 🧠 Key Learnings

* Handling real-world sparse retail data
* Feature scale imbalance in hybrid models
* Multi-stage ML system design
* Debugging large-scale pipelines
* Bridging forecasting with recommendation systems

---

## 📌 Future Improvements

* LLM-powered natural language copilot
* Cloud deployment (AWS/GCP)
* Feature store integration
* Real-time streaming feedback (Kafka)
* Advanced ranking signals (price, margin, inventory)

---

## 💼 Use Cases

* Inventory planning
* Personalized marketing
* Demand-driven recommendations
* Supply chain optimization

---

## 🏁 Conclusion

This project demonstrates how modern AI systems evolve from:

```
Standalone Models → Integrated Intelligence Platforms
```

It goes beyond predictions to enable **continuous learning and decision-making**.

---

## ⭐ If You Like This Project

Give it a star ⭐ — and feel free to connect!
