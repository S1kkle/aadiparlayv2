# Underdog Prop Predictor (Underdog + ESPN + Ollama)

Local web app that ranks Underdog Pick'em props using:
- a simple stats model (edge/EV/volatility) from ESPN-derived history
- local qualitative analysis via Ollama

## Prereqs
- Node.js (for `web/`)
- Python 3.11+ (you have Python already)
- (Optional) Ollama running locally: `http://localhost:11434`

## Setup

### 1) Backend

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy ..\.env.example .env
python -m uvicorn app.main:app --reload --port 8000
```

Backend health: `http://localhost:8000/health`

### 2) Frontend

```bash
cd web
npm install
npm run dev
```

Open: `http://localhost:3000`

## Configuration notes
- **Do not put Underdog auth tokens in the frontend**. The backend reads `.env`.
- If your Underdog endpoint requires extra query params (like `product_experience_id` / `state_config_id`), set them in `backend/.env`.

