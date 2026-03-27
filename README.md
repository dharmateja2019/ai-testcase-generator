# 🤖 AI Test Case Generator (Local LLM + Streamlit)

## 🚀 Overview

An AI-powered test case generator that converts user stories into structured test cases using local LLMs.

This project simulates real-world challenges in AI systems such as handling non-deterministic outputs, ensuring structured responses, and improving reliability.

---

## 🎯 Features

- Generate test cases from user stories
- Supports **single input** and **batch processing**
- Structured output in **JSON and CSV**
- **Retry mechanism** for handling invalid LLM responses
- **Custom JSON parsing with fallback handling**
- **History tracking** of previous runs
- **Logging system** for monitoring
- **Multi-model support** (Phi-3, LLaMA)
- **Confidence scoring** for test case quality

---

## 🧠 Architecture

- **UI Layer:** Streamlit (`app_ui.py`)
- **Backend Layer:** LLM integration + parsing (`main.py`)
- **Utilities:** Logging + history (`utils.py`, `logger.py`)

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Ollama (Local LLM)
- Pandas
- JSON / Regex

---

## ⚙️ How It Works

1. User enters a user story
2. Prompt is generated with strict JSON schema
3. Local LLM generates test cases
4. Output is parsed and validated
5. Retry mechanism ensures structured output
6. Results are displayed and saved

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app_ui.py
```

---

## 📊 Example Output

- Test Case ID
- Description
- Steps
- Expected Result
- Priority
- Confidence Score

---

## 🔥 Key Challenges Solved

- Handling inconsistent LLM outputs
- Ensuring structured JSON responses
- Designing retry + fallback parsing mechanisms

---

## 🎯 Future Improvements

- RAG integration for domain-specific test cases
- API deployment (FastAPI)
- Database integration
- LLM-based evaluation system

---

## 📌 Author

Dharmateja – QA Engineer transitioning into AI-powered QA 🚀
