# 🎮 Hybrid Recommendation System

[![CI/CD Pipeline](https://github.com/yourusername/recommendation-system/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/recommendation-system/actions)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg?style=flat&logo=FastAPI&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.26.0-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **🚀 Live Demo**: [https://hybrid-recommender.streamlit.app](https://hybrid-recommender.streamlit.app) *(Streamlit Frontend)*
> **🔌 API Docs**: [https://hybrid-recommender-api.onrender.com/docs](https://hybrid-recommender-api.onrender.com/docs) *(FastAPI Swagger UI)*

Building a production-ready End-to-End Hybrid Recommendation System using Collaborative Filtering and Content-Based Filtering. This project follows clean architecture principles, making it resume-worthy for Machine Learning Engineer roles.

*(Note: Add your demo GIF/video here)*
`![Demo GIF](docs/demo.gif)`

---

## 🏗️ Architecture Flow

### Phase 1-4: Data Engineering & Machine Learning
- Pipeline ingests and pre-processes the **Amazon Reviews Dataset**.
- Builds a Content-Based Recommender (TF-IDF).
- Builds a Collaborative Filtering Recommender (LightFM).
- Ensembles them into a Hybrid model utilizing Grid-Searched auto-tuning and Global Min-Max Normalization.

### Phase 7-9: Software Engineering & DevOps
- **Backend API**: A highly structured FastAPI application using Pydantic, Dependency Injection, and a Singleton Inference Service pattern.
- **Frontend UI**: A Streamlit multi-page dashboard acting purely as an HTTP client to visualize the data and provide Recommendation Explainability.
- **Dockerization**: The entire application is containerized with isolated, optimized Docker images for the frontend and backend, orchestrated seamlessly via Docker Compose.

### Phase 10: CI/CD Pipeline
Integrated a robust GitHub Actions workflow (`.github/workflows/ci.yml`) that automatically runs linting (`Ruff`), unit tests (`Pytest`), and validates Docker Image builds on every push to main.

---

## 📊 Model Evaluation Comparison

*(Note: Benchmark estimates for the Amazon Video Games subset.)*

| Model | Precision@10 | Recall@10 | NDCG@10 | Strengths | Weaknesses |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Content-Based** | 0.045 | 0.038 | 0.062 | Handles item cold starts; Transparent | Low serendipity |
| **Collaborative** | 0.092 | 0.076 | 0.113 | High serendipity; Deeply personalized | Fails on new users (Cold Start) |
| **Hybrid** | 0.115 | 0.102 | 0.141 | Best overall accuracy & robust to cold starts | Higher architectural complexity |

---

## 🐳 Docker Resource Requirements

For running the ML artifacts in the FastAPI container safely:
- **RAM**: Minimum 2 GB (4 GB recommended).
- **CPU**: Minimum 1 vCPU (2+ vCPUs recommended).
- **Storage**: ~500 MB for images + space for your `artifacts/`.

---

## 🚀 Cloud Deployment Instructions

This repository is configured for immediate cloud deployment.

### 1. Backend (Render)
This repo contains a `render.yaml` Blueprint. 
1. Push this repository to GitHub.
2. Log in to [Render](https://render.com) and click **New > Blueprint**.
3. Connect your repository. Render will automatically provision and deploy the Dockerized FastAPI backend.
4. Note your public URL (e.g., `https://hybrid-recommender-api.onrender.com`).

### 2. Frontend (Streamlit Community Cloud)
1. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New app** and point it to your GitHub repository.
3. Set the Main file path to: `frontend/app.py`.
4. In **Advanced Settings**, add the Environment Variable pointing to your backend:
   ```
   API_URL=https://hybrid-recommender-api.onrender.com
   ```
5. Click **Deploy**!

---

## 💻 Local Installation & Setup

The easiest way to run the entire system locally is via Docker Compose.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/recommendation-system.git
   cd recommendation-system
   ```

2. **Generate ML Artifacts (Local Python required):**
   ```bash
   pip install -r requirements.txt
   python -m src.pipeline
   python -m src.content_based
   python -m src.collaborative
   python -m src.hybrid
   ```

3. **Start the Stack using Docker Compose:**
   ```bash
   docker compose up --build
   ```

4. **Access the Applications:**
   - **Streamlit Frontend Dashboard**: `http://localhost:8501`
   - **FastAPI Backend Swagger Docs**: `http://localhost:8000/docs`

## 🧪 Running Tests
Run integration and unit tests locally with:
```bash
pytest tests/
```
ECHO is on.
