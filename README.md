# 🚀 SHL Conversational Assessment Recommender

An AI-powered conversational agent that leverages Retrieval-Augmented Generation (RAG) to guide hiring managers and recruiters from vague hiring intents to a concrete, evidence-based shortlist of 1 to 10 SHL assessments. Built with **FastAPI**, **LangChain**, **Groq (Llama 3.3 70B)**, and **FAISS**.

---

## 🛠️ Tech Stack & Key Technologies
* **Framework**: FastAPI (Python 3.10)
* **LLM Engine**: LangChain ChatGroq with `llama-3.3-70b-versatile` (Temperature = 0)
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFaceEmbeddings)
* **Vector Store**: FAISS (Local Index)
* **Containerization**: Docker (optimized for Hugging Face Spaces deployment)

---

## ✨ Features
1. **Interactive Conversational Flows**: Guides users from general inputs (e.g., *"I need to hire a Java developer"*) to specific requirements (seniority, framework knowledge) via targeted clarifying questions.
2. **Catalog-Grounded Shortlists**: Restricts recommendations strictly to the provided SHL Product Catalog (377 assessments) to prevent hallucinated product names.
3. **Stateless Turn Management**: The `/chat` endpoint accepts the complete conversation history array for stateless API interactions.
4. **Strict JSON Schemas**: Employs Pydantic models to guarantee that all responses strictly conform to the expected format:
   ```json
   {
     "reply": "Assistant message text...",
     "recommendations": [
       {
         "name": "Assessment Name",
         "url": "https://www.shl.com/.../",
         "test_type": "K"
       }
     ],
     "end_of_conversation": false
   }
   ```
5. **Robust Evaluation Harness**: Included evaluation methods to benchmark Retrieval Quality, Relevance, Groundedness (LLM-as-a-judge), and Overall Effectiveness against standard reference traces.

---

## 🏗️ Architecture

```
                  ┌───────────────────────┐
                  │   Recruiter Chat UI   │
                  └───────────┬───────────┘
                              │
                    JSON payload (/chat)
                              ▼
                  ┌───────────────────────┐
                  │    FastAPI Backend    │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │   RAG Agent Engine    │
                  │      (agent.py)       │
                  └─────┬───────────▲─────┘
                        │           │
           Query Rewrite│           │ Retrieve Relevant
           & Embed      │           │ Context (Top-20)
                        ▼           │
                  ┌───────────┐     │
                  │   FAISS   ├─────┘
                  │ Vector DB │
                  └───────────┘
```

---

## 📂 Project Directory Structure

```bash
├── faiss_index/             # Pre-built FAISS Vector Store index files
├── sample_conversations/    # 10 Reference conversation traces for evaluation
├── .env.example             # Example environment configuration
├── Dockerfile               # Production Dockerfile for Hugging Face Spaces
├── agent.py                 # Core LangChain RAG & LLM orchestration logic
├── app.py                   # FastAPI Application endpoints (/health, /chat)
├── evaluate.py              # Automated evaluation metric script
├── explore_json.py          # Utility script for parsing & analyzing raw catalog data
├── ingest.py                # Data processing and FAISS vectorization pipeline
├── requirements.txt         # Project dependencies
├── test_chat.py             # Script simulating multi-turn local chat queries
└── walkthrough.md           # Detailed development walkthrough
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10** installed. We recommend using `conda` for environment isolation:
```bash
conda create -p shl_env python=3.10 -y
conda activate ./shl_env
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables
Create a `.env` file in the root directory (or use your deployment environment secrets configuration):
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the API Locally
Start the FastAPI development server:
```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

* Swagger API documentation will be available at: **`http://127.0.0.1:8000/docs`**

---

## 🧪 Testing & Evaluation

### Local Integration Test
To run a simulated 3-turn client chat workflow (vague query -> clarification -> refinement & recommendation) against the local backend:
```bash
python test_chat.py
```

### Evaluation Script
To run the automated RAG evaluation metrics against the 10 reference traces in `sample_conversations/`:
```bash
python evaluate.py
```
This script computes:
* **Retrieval Recall@20**
* **Recommendation IoU (Intersection-over-Union)**
* **Groundedness Score** (1-5 range using LLM-as-a-judge)
* **Conversation Effectiveness** (1-5 range using LLM-as-a-judge)

---

## 🌐 Production Deployment

This project is configured for serverless production deployment via **Hugging Face Spaces (Docker SDK)**.

### Hugging Face Spaces Configuration:
1. **SDK**: Docker
2. **Hardware**: CPU Basic (Free Tier - 16GB RAM)
3. **Environment Secrets**: Add `GROQ_API_KEY` under Settings -> Secrets.
4. **Port**: Automatically binds to port `7860` as configured in the `Dockerfile`.
