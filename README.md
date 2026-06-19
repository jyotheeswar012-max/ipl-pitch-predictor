# 🏏 IPL Pitch Predictor — ML Player Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ipl-pitch-predictor.streamlit.app)
![scikit-learn](https://img.shields.io/badge/ML-RandomForest-orange?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

> **ML-powered Streamlit dashboard that recommends the best IPL players for any given pitch condition, venue, and match phase — and explains *why* they fit.**

---

## 🚀 Live Demo

### 👉 [https://ipl-pitch-predictor.streamlit.app](https://ipl-pitch-predictor.streamlit.app)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ipl-pitch-predictor.streamlit.app)

---

## ✨ Features

| Feature | Details |
|---|---|
| 🏟️ Pitch-aware ranking | 5 pitch types with intelligent role weighting |
| 🤖 ML model | Random Forest Regressor (200 trees) predicts suitability 0–100 |
| 💡 Explainability | Each recommendation includes a natural-language reason |
| 📊 Interactive charts | Bar, Scatter, Radar, Pie — all in dark theme via Plotly |
| 🗺️ Venue analytics | Per-venue avg scores and role distribution |
| 🎯 Role filter | Filter by Batsman, Bowler, All-Rounder, WK-Batter |
| 🪙 Toss + Phase | Match phase (powerplay/death) and toss decision inputs |

---

## 📐 Pitch Types Supported

| Pitch Type | Best For | Key Players |
|---|---|---|
| 🟢 Flat/Batting | Top-order batters | Virat Kohli, Rohit Sharma, Jos Buttler |
| 🟡 Hard/True Bounce | Power hitters + pace bowlers | Suryakumar Yadav, Jasprit Bumrah |
| 🌿 Green/Grassy | Seam & swing bowlers | Bumrah, Shami, Bhuvneshwar, Boult |
| 🟠 Slow/Dry Spin | Spinners + anchors | Rashid, Chahal, Jadeja, Narine |
| 💧 Wet/Dew Heavy | Chasers + death specialists | Arshdeep, Buttler, Suryakumar |

---

## 🧠 ML Model Architecture

```
Features → RandomForestRegressor → Suitability Score (0–100)
```

**Features used:**
- `batting_avg`, `strike_rate`, `economy`, `wickets_per_match`
- `pitch_type` (encoded), `venue` (encoded), `toss` (encoded), `role` (encoded)

**Training data:** 1,500+ synthetic IPL-style records with pitch-role suitability logic modeled on real player strengths.

---

## 🛠️ Run Locally

```bash
git clone https://github.com/jyotheeswar012-max/ipl-pitch-predictor
cd ipl-pitch-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deploy to Streamlit Cloud (Steps)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub (`jyotheeswar012-max`)
3. **New App** → Repo: `ipl-pitch-predictor` → Branch: `main` → File: `app.py`
4. Click **Deploy** 🚀

✅ Live URL:
```
https://ipl-pitch-predictor.streamlit.app
```

---

## 📂 Project Structure

```
ipl-pitch-predictor/
├── app.py               # Streamlit UI (4 tabs)
├── model.py             # RandomForest training & prediction
├── data_generator.py    # Synthetic IPL dataset with pitch logic
├── requirements.txt
├── .streamlit/
│   └── config.toml      # Dark theme config
└── README.md
```

---

## 👨‍💻 Built By
**Jyotheeswar Reddy** — [GitHub](https://github.com/jyotheeswar012-max)

---

*Made with ❤️ in Hyderabad | IPL Pitch Predictor v1.0*
