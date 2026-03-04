# Publishing the app to a website (everything working)

You can run the site in two ways:

- **Option 1 (recommended): Always-on** – Backend + AI on **Render** (free tier) using **Groq** (free-tier API). No PC or tunnel. Site works 24/7.
- **Option 2: PC + tunnel** – Backend + Ollama on your computer; expose with Cloudflare/ngrok. Site works only when your PC and tunnel are running.

---

# Option 1: Always-on (Backend on Render + Groq)

No computer or tunnel. Frontend on Vercel, backend on Render, AI via Groq’s free-tier API.

## 1. Get a Groq API key (free)

1. Sign up at [console.groq.com](https://console.groq.com).
2. Create an API key (e.g. **API Keys** in the dashboard).
3. Copy the key; you’ll use it on Render.

## 2. Deploy the backend to Render

1. Push your repo to **GitHub** (if needed).
2. Go to [render.com](https://render.com) and sign in with GitHub.
3. **New** → **Blueprint** → connect the repo. Render will read `render.yaml` at the repo root.
4. When Render shows the services, it will prompt for **secret** env vars (because of `sync: false` in the blueprint). Set:
   - **GROQ_API_KEY** – your Groq API key.
   - **UD_AUTH_TOKEN** – your Underdog auth token (from your existing `.env`).
   - **UD_USER_LOCATION_TOKEN** – your Underdog user-location token.
   - **UD_PRODUCT_EXPERIENCE_ID** and **UD_STATE_CONFIG_ID** – same as in your `backend/.env`.
5. Deploy. Wait for the backend to build and start. Your backend URL will be like `https://aadiparlay-backend.onrender.com`.

**Note:** On the free tier, Render spins down the service after ~15 minutes of no traffic. The first request after that may take 30–60 seconds (cold start). For always-warm service, use a paid plan or another host.

## 3. Deploy the frontend to Vercel

1. At [vercel.com](https://vercel.com), import the same repo.
2. Set **Root Directory** to **`web`**.
3. Add an environment variable:
   - **NEXT_PUBLIC_BACKEND_URL** = your Render backend URL (e.g. `https://aadiparlay-backend.onrender.com`) — **no trailing slash**.
4. Deploy.

Your site is now live and always on. The frontend calls the backend on Render; the backend uses Groq for AI. No PC or tunnel needed.

---

# Option 2: PC + tunnel (Backend and Ollama on your computer)

Site works only when your PC is on and the tunnel is running.

## Step 1: Deploy the frontend to Vercel

1. Push your project to **GitHub** (if you haven’t already).
2. Go to [vercel.com](https://vercel.com) and sign in with GitHub.
3. **Import** the repo. When asked:
   - **Root Directory:** click “Edit” and set it to **`web`** (not the repo root).
   - **Build Command:** `npm run build` (default).
   - **Output Directory:** `.next` (default).
4. Add **Environment variables** in the Vercel project:
   - **`NEXT_PUBLIC_BACKEND_URL`** – leave this **empty for now**. You’ll set it in Step 3 to your tunnel URL.
5. Deploy. The site will be at `https://your-project.vercel.app` (or your custom domain). It will show “Backend error” until the backend URL is set and your backend is exposed.

---

## Step 2: Expose your backend with a tunnel

Your backend runs on your PC. A tunnel gives it a public URL (e.g. `https://abc123.trycloudflare.com`) so Vercel can call it.

### Option A: Cloudflare Tunnel (recommended, free)

1. Install **cloudflared**:  
   [https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation)  
   On Windows you can use the standalone executable or `winget install cloudflare.cloudflared`.
2. Start your **backend** and **Ollama** on your PC (e.g. run `scripts\restart-dev.cmd` for backend; keep Ollama running).
3. In a terminal, run:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
4. You’ll see a line like:
   ```text
   Your quick Tunnel has been created! Visit it at:
   https://random-name-here.trycloudflare.com
   ```
5. That URL is your **public backend URL**. Use it in Step 3.

**Note:** With the free “quick” tunnel, the URL changes each time you restart `cloudflared`. For a stable URL you can create a free Cloudflare account and a named tunnel (see Cloudflare docs).

### Option B: ngrok (free tier)

1. Sign up at [ngrok.com](https://ngrok.com) and install ngrok.
2. Start backend + Ollama on your PC.
3. Run:
   ```bash
   ngrok http 8000
   ```
4. Copy the **HTTPS** URL ngrok shows (e.g. `https://abc123.ngrok-free.app`). That’s your public backend URL.

Free ngrok URLs change each time you start ngrok unless you have a paid plan.

---

## Step 3: Point the frontend at your backend

1. In **Vercel** → your project → **Settings** → **Environment Variables**:
   - Set **`NEXT_PUBLIC_BACKEND_URL`** to your tunnel URL **with no trailing slash**, e.g.  
     `https://random-name.trycloudflare.com`  
     or  
     `https://abc123.ngrok-free.app`
2. **Redeploy** the frontend (Deployments → … on latest deployment → Redeploy), so the new env var is picked up.

After redeploy, the live site will call your backend at that URL. The backend will use Ollama at `http://127.0.0.1:11434` on your PC, so Ollama does **not** need to be exposed to the internet.

---

## Step 4: Run everything when you want the site to work

When you want the published site to work:

1. **Start Ollama** on your PC (if not already running).
2. **Start the backend**, e.g.:
   ```bash
   cd backend
   .\.venv\Scripts\activate
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
   Using `--host 0.0.0.0` lets the tunnel reach the app.
3. **Start the tunnel** (in another terminal), e.g.:
   ```bash
   cloudflared tunnel --url http://localhost:8000
   ```
4. If you use a **new** tunnel URL (e.g. new Cloudflare quick tunnel URL), update **`NEXT_PUBLIC_BACKEND_URL`** in Vercel and redeploy once.

Then open `https://your-project.vercel.app` and use the app. Picks will load and AI summaries will run via your local Ollama.

---

## Summary (Option 2)

| Part        | Where it runs        | Cost   |
|------------|----------------------|--------|
| Frontend   | Vercel               | Free   |
| Backend    | Your PC              | —      |
| Ollama (AI)| Your PC              | —      |
| Tunnel     | Your PC (cloudflared or ngrok) | Free   |

- **No purchases needed.** The site works for anyone as long as your PC is on and the tunnel is running.
- To avoid “Backend error,” keep backend + tunnel running whenever you or others need the site.
- For a **stable** public backend URL so you don’t have to change Vercel env after each restart, use a **named** Cloudflare Tunnel (free account). See quick comparison below.

---

## Quick comparison

| | Option 1 (Always-on) | Option 2 (PC + tunnel) |
|---|----------------------|------------------------|
| **Backend** | Render (free tier) | Your PC |
| **AI** | Groq API (free tier) | Ollama (local) |
| **PC required?** | No | Yes |
| **Tunnel required?** | No | Yes |
| **Cold starts** | Render spins down after ~15 min idle | None (while PC is on) |
