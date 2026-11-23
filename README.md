# SafeRoute NCR

**An AI-powered traffic-fine risk map and route planner for Delhi NCR**

| | |
|---|---|
| **Name** | Vishesh Kumar Bachchas |
| **Roll No.** | 25SCS1003005456 |
| **Class** | 2CSE4 |
| **University** | IILM University |
| **Subject** | Generative AI |
| **Project type** | Final project (prototype) |

---

## 1. Overview

SafeRoute NCR is a web app that shows where traffic fines are most likely to be issued across Delhi NCR, at a chosen day and hour. A machine-learning model predicts fines per hour for 40 checkpoints. The app uses those predictions to colour a live map from green (low risk) to red (high risk), and to suggest a lower-risk alternative to the fastest route.

A built-in AI assistant explains the risk, the chosen route and typical fines in plain language. It runs in a rule-based mode with no setup, or in an optional Gemini mode for open-ended questions.

> **Note:** This is an academic prototype built for a class presentation. The data is synthetic and the checkpoint locations are approximate. It is not official enforcement data and should not be treated as real police or traffic information.

## 2. Features

- **Interactive map:** real OpenStreetMap tiles (Leaflet) with 40 checkpoints coloured by predicted fine risk.
- **Time control:** a slider and day selector show how risk changes through the day and week.
- **Personalised risk:** tick or untick licence, helmet and documents, and the map, risk list and routes update for your situation.
- **Route comparison:** shows the fastest route and a lower-risk route side by side, with extra minutes, extra distance and the reduction in checkpoint exposure.
- **Detour limit:** a slider sets how many extra minutes you will accept (default 15).
- **Place search:** type a place or pick a point on the map for the start and destination.
- **Checkpoint details:** click any checkpoint to see its risk, likely violation and average fine.
- **Ask AI chatbot:** answers about the riskiest spots now, the best time to travel, why a route was chosen, typical fines and what to watch for.

## 3. Where generative AI is used

1. **Dataset generation:** the 60,000-row synthetic dataset was designed with a generative AI prompt (Google Gemini), which produced the Python generator script.
2. **Code assistance:** the notebook, the website and the routing logic were built with generative AI assistance (Claude).
3. **Gemini-powered assistant:** the chatbot can call the Gemini API. It is grounded on the model's predictions, the selected time and the last route, so its answers reflect the app's data. If the API fails, it falls back to built-in rule-based answers.

## 4. Dataset

- **File:** `traffic_fines_ncr.csv`, created by `generate_traffic_fines_ncr.py` (pandas + numpy, seed 42).
- **Size:** about 60,000 rows and 32 columns.
- **Granularity:** one row is one checkpoint, one hour and one date.
- **Period:** 1 January 2025 to 30 June 2025.
- **Coverage:** 40 well-known Delhi NCR junctions and roads, with approximate latitude and longitude.

**Column groups**

| Group | Columns |
|---|---|
| Checkpoint | checkpoint_id, checkpoint_name, latitude, longitude, road_type, speed_limit_kmph, is_near_metro |
| Enforcement and traffic | police_presence, camera_density, traffic_density, checkpoint_active, vehicles_checked |
| Time and context | date, hour, day_of_week, is_weekend, is_holiday, month, weather, temperature_c, event_nearby |
| Fines | fines_issued, total_fine_amount_inr, top_violation |
| Violation counts | helmet, no_licence, drunk_driving, speeding, signal_jumping, wrong_parking, no_documents |
| Target score | fine_risk_score (0 to 100) |

**Realism rules built into the generator**

- Fines rise with police presence, camera density and traffic density.
- Peaks at 8–11 and 17–21, with more drunk-driving checks at night, especially on Friday and Saturday.
- Rain and fog slightly reduce checkpoint activity. Holidays and special drives increase fines.
- The six violation counts sum to `fines_issued`, and `top_violation` is the largest count.
- Fine amounts follow typical Indian challan values.
- About 1% missing values, in weather and temperature only.

## 5. Machine-learning model

The notebook `traffic_risk_model.ipynb` explores the data, trains and compares three models, and exports the predictions used by the website.

- **Task:** predict fines issued per hour for each checkpoint, hour and day of week.
- **Split:** time-based. Train on January–April, test on May–June.
- **Models compared:** Ridge Regression (baseline), Random Forest and Gradient Boosting.

| Model | R² (held-out May–June) | MAE |
|---|---|---|
| Ridge Regression | 0.38 | – |
| Random Forest | 0.465 | – |
| **Gradient Boosting (best)** | **0.495** | **about 5.0 fines per hour** |

Gradient Boosting performed best, so its predictions are written to `website/data.js`. Re-running the notebook regenerates that file.

## 6. How routing works

1. The user enters a start and a destination.
2. The app requests real road routes from the public OSRM service and adds detour candidates that steer around high-risk checkpoints.
3. For each candidate it calculates the total fine-risk exposure along the path, using the user's personal risk (licence, helmet, documents) and the chosen time.
4. It selects the route with the lowest exposure within the allowed extra minutes, and shows it next to the fastest route.

## 7. Project structure

```
proj/
├── generate_traffic_fines_ncr.py   # synthetic dataset generator
├── traffic_fines_ncr.csv           # generated dataset (about 60,000 rows)
├── traffic_risk_model.ipynb        # data exploration, training, model comparison
├── README.md
└── website/
    ├── index.html                  # map, routing, personal risk, chatbot
    └── data.js                     # model predictions exported by the notebook
```

## 8. How to run

### 8.1 Train the model (VS Code notebook)

Requires Python 3.9 or newer (tested on 3.12).

1. Open the `proj` folder in VS Code (File → Open Folder) with the Python and Jupyter extensions installed.
2. Open `traffic_risk_model.ipynb` and select your Python kernel.
3. Add a first cell and run it, then restart the kernel:
   ```python
   %pip install numpy pandas scikit-learn matplotlib joblib
   ```
4. Click **Run All**. The last cell saves `website/data.js`.

### 8.2 Run the website locally

Open a terminal inside the `website` folder and run:

```
py -m http.server 8080
```

Then open <http://localhost:8080> in Chrome and press Ctrl+F5 to refresh. An internet connection is needed for the map tiles, search and routing.

### 8.3 Enable the Gemini chatbot (optional)

1. Create a free API key at <https://aistudio.google.com>.
2. In the app, click **Ask AI**, then **AI settings**.
3. Paste the key, enter a current model ID from AI Studio (lowercase with hyphens, for example `gemini-3.8-flash`), and click **Save**.

The key stays in your own browser only. Without a key, the chatbot still answers from the built-in rule-based mode. Model names change often, so use the ID shown in AI Studio's code sample.

## 9. Hosting

The website is static, so it can be hosted free:

- **Netlify:** log in at <https://app.netlify.com/drop> and drag the `website` folder onto the page. To update, drag the folder onto the site's Deploys tab again.
- **GitHub Pages:** push `website` to a repository, then enable Pages under Settings.

**Live link:**(https://saferoute-ncr-website.vercel.app/)

## 10. Technology stack

| Area | Tools |
|---|---|
| Data and ML | Python, pandas, NumPy, scikit-learn, matplotlib, joblib |
| Frontend | HTML, CSS, JavaScript, Leaflet |
| Map and routing | OpenStreetMap, OSRM (routing), Nominatim (place search) |
| Generative AI | Google Gemini (dataset prompt and chatbot), Claude (code assistance) |
| Hosting | Netlify |

## 11. Limitations

- The dataset is **synthetic**, and checkpoint locations are approximate, so risk scores are illustrative only.
- The model's R² of about 0.5 reflects noise built into the synthetic data.
- OSRM and Nominatim are free public services with rate limits. If routing is busy, wait a few seconds and retry.
- The Gemini mode depends on the user's API key and on the model name still being available.
- In real life, riding without a helmet or licence is unsafe and illegal. The personal-risk checkboxes exist only to demonstrate how inputs change predictions in this prototype.

## 12. Future work

- Replace the synthetic data with real challan data and retrain.
- Add weather and event inputs as selectable conditions on the map.
- Add live traffic data and a self-hosted routing server.
- Add a shareable route link and a mobile-friendly layout.

## 13. Author

**Vishesh Kumar Bachchas**
Roll No. 25SCS1003005456 · Class 2CSE4
IILM University · Generative AI Project
