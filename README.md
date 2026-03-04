# Restarting without typing (Windows)

You can restart both the backend + frontend by double-clicking:

- `scripts/restart-dev.cmd`

This will:

- stop anything listening on ports `8000`, `3000`, `3001`
- remove the Next.js dev lock file if it exists
- start FastAPI + Next.js in separate PowerShell windows

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

## Publishing to a website

- **Always-on (no PC):** Backend on **Render** + AI via **Groq** (free tier). See **[DEPLOYMENT.md](DEPLOYMENT.md)** → Option 1.
- **PC + tunnel:** Backend and Ollama on your computer; expose with Cloudflare/ngrok. See **[DEPLOYMENT.md](DEPLOYMENT.md)** → Option 2.

