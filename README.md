# ASPRS — AI Based Student Performance Review System

## 🚀 Deploy to Render.com (Free, Permanent Hosting)

### Step 1 — Upload to GitHub
1. Go to **github.com** and create a free account if you don't have one
2. Click **New Repository** → name it `asprs` → set to Public → click **Create**
3. Download **GitHub Desktop** from desktop.github.com
4. Open GitHub Desktop → **Add Existing Repository** → select your `asprs` folder
5. Click **Publish Repository** → make sure it's Public → click **Publish**

### Step 2 — Deploy on Render
1. Go to **render.com** → Sign up with your GitHub account
2. Click **New +** → **Web Service**
3. Click **Connect** next to your `asprs` repository
4. Fill in the settings:
   - **Name:** asprs (or any name you like)
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click **Create Web Service**
6. Wait 3–5 minutes for the build to finish
7. Your site is live at: `https://asprs.onrender.com` (or similar)

> ⚠️ Free tier note: Render free tier sleeps after 15 minutes of no traffic.
> The first visit after sleeping takes ~30 seconds to wake up. This is normal.

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

---

## 📁 Project Structure
```
asprs/
├── app.py                  ← Flask backend (all API routes)
├── requirements.txt        ← Python dependencies
├── render.yaml             ← Render deployment config
├── models/                 ← All 8 .pkl model files
│   ├── kt_next_semester_model.pkl
│   ├── kt_next_semester_scaler.pkl
│   ├── dropout_model.pkl
│   ├── dropout_scaler.pkl
│   ├── weak_subject_model.pkl
│   ├── weak_subject_scaler.pkl
│   ├── placement.pkl
│   └── placement_scaler.pkl
└── templates/
    ├── index.html          ← Landing page
    ├── predict.html        ← Single student prediction
    └── bulk.html           ← Bulk Excel upload + charts
```

---

## 🤖 ML Models

| Model | Algorithm | Features | Output |
|-------|-----------|----------|--------|
| KT Predictor | RandomForest (350 trees) | 6 | KT Likely / No KT |
| Dropout Risk | RandomForest (350 trees) | 6 | High / Medium / Low Risk |
| Placement | RandomForest (250 trees) | 7 | High / Moderate / Low |
| Weak Subject | RandomForest (400 trees) | 5 | Grade A / B / C / D |

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/kt` | KT risk prediction |
| POST | `/api/dropout` | Dropout risk prediction |
| POST | `/api/placement` | Placement prediction |
| POST | `/api/weak-subject` | Weak subject grade prediction |
| POST | `/api/bulk/preview` | Bulk prediction (returns JSON) |
| POST | `/api/bulk/download` | Bulk prediction (returns Excel) |
| GET  | `/api/template/<model>` | Download blank input template |
