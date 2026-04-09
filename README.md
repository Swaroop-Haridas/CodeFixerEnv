# 🔧 CodeFixer

An **OpenEnv-based Reinforcement Learning environment** designed to train AI agents to **debug Python code automatically**.

Built as part of the **Meta × PyTorch × Hugging Face OpenEnv Hackathon**, this project focuses on creating a structured environment where models learn to fix buggy code using feedback-driven rewards.

---

## 🚀 Overview

CodeFixer provides an interactive environment where an AI agent:

- Receives **buggy Python functions**
- Understands the **intended functionality**
- Submits **corrected code**
- Gets evaluated through **automated test cases**
- Learns using a **reward-based feedback system**

This setup mimics real-world debugging scenarios and is ideal for training RL-based code assistants.

---

## 🧠 Key Features

- ✅ Reinforcement Learning environment (OpenEnv compatible)
- ✅ Automated grading using deterministic test cases
- ✅ Dense reward signals for better learning
- ✅ Modular architecture for adding new tasks
- ✅ Docker support for easy deployment

---

## 🏗️ Project Structure

```
CodeFixer/
├── env/
│   ├── environment.py     # Core RL environment logic
│   ├── models.py          # Data models (Observation, Action, Reward)
│
├── tasks/
│   ├── tasks.py           # Buggy coding tasks & test cases
│
├── server/
│   ├── app.py             # API server (if applicable)
│
├── tests/
│   ├── test_env.py        # Unit tests for environment
│
├── inference.py           # Inference / evaluation script
├── server.py              # Entry point for server
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container setup
├── openenv.yaml           # OpenEnv configuration
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd CodeFixer
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🐳 Docker Setup (Recommended)

```bash
docker build -t codefixer .
docker run -p 8000:8000 codefixer
```

---

## ▶️ Usage

### Run Environment Locally

```bash
python inference.py
```

### Run Server

```bash
python server.py
```

---

## 🔄 How It Works

1. Environment provides:
   - Buggy code
   - Problem description

2. Agent submits:
   - Fixed Python function

3. Environment evaluates:
   - Correctness (test cases)
   - Efficiency
   - Code validity

4. Reward is computed and returned

---

## 🧪 Testing

Run unit tests using:

```bash
pytest
```

---

## 📌 Use Cases

- Training RL agents for **code debugging**
- Building **AI coding assistants**
- Research in **program synthesis & repair**
- Educational tools for **learning debugging**


---

> "Teaching machines to debug code is one step closer to autonomous programming." 🚀

