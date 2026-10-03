import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATA_PATH = Path(__file__).with_name("advertising.csv")
FEATURES = [
    "Daily Time Spent on Site",
    "Age",
    "Area Income",
    "Daily Internet Usage",
    "Male",
]
TARGET = "Clicked on Ad"
REQUIRED_COLUMNS = [*FEATURES, TARGET]

app = FastAPI(
    title="Ad Engagement | Logistic Regression",
    description="Explore advertising data and score visitor profiles.",
    version="1.0.0",
)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Could not find {DATA_PATH.name} next to app.py.",
        )

    data = pd.read_csv(DATA_PATH)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail="The CSV is missing required columns: " + ", ".join(missing_columns),
        )
    return data


def analyze_data(test_fraction: float, random_seed: int) -> dict[str, Any]:
    data = load_data()
    features = data[FEATURES]
    target = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_fraction,
        random_state=random_seed,
    )
    model = LogisticRegression(solver="liblinear")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    return {
        "data": data,
        "model": model,
        "features": features,
        "target": target,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "report": report,
    }


def dashboard_result(test_fraction: float, random_seed: int) -> dict[str, Any]:
    analysis = analyze_data(test_fraction, random_seed)
    data = analysis["data"]
    target = analysis["target"]
    report = analysis["report"]
    outcome_counts = target.value_counts().reindex([0, 1], fill_value=0)
    feature_stats = {
        feature: {
            "minimum": float(data[feature].min()),
            "maximum": float(data[feature].max()),
            "median": float(data[feature].median()),
        }
        for feature in FEATURES
    }

    return {
        "title": "Who clicks on an ad?",
        "description": "Explore the advertising audience, review model performance, and score a visitor profile.",
        "metrics": {
            "visitors": len(data),
            "click_through_rate": float(target.mean() * 100),
            "model_accuracy": float(report["accuracy"]),
            "test_observations": len(analysis["y_test"]),
        },
        "audience_preview": data.head(10).to_dict(orient="records"),
        "click_outcomes": {
            "did_not_click": int(outcome_counts.loc[0]),
            "clicked": int(outcome_counts.loc[1]),
        },
        "feature_stats": feature_stats,
        "model": {
            "classification_report": report,
            "confusion_matrix": confusion_matrix(
                analysis["y_test"], analysis["predictions"]
            ).tolist(),
            "train_observations": len(analysis["X_train"]),
            "test_observations": len(analysis["X_test"]),
            "test_fraction": test_fraction,
            "random_seed": random_seed,
        },
        "default_profile": {
            feature: float(data[feature].median()) for feature in FEATURES
        },
    }


def dashboard_page(test_fraction: float, random_seed: int) -> HTMLResponse:
    result = dashboard_result(test_fraction, random_seed)
    payload = json.dumps(result)
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Ad Engagement Dashboard</title>
    <style>
        :root {
            --bg: #07111f;
            --bg-alt: #0d1d2f;
            --panel: rgba(14, 25, 40, 0.9);
            --panel-light: rgba(19, 34, 53, 0.9);
            --primary: #4cc9f0;
            --accent: #7b61ff;
            --success: #39d98a;
            --warning: #ffb703;
            --text: #eaf5ff;
            --muted: #8ea8c6;
            --border: rgba(142, 168, 198, 0.18);
            --shadow: 0 20px 50px rgba(2, 9, 17, 0.45);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, var(--bg) 0%, #0d1a2f 40%, #110f1f 100%);
            color: var(--text);
        }
        .page {
            max-width: 1280px;
            margin: 0 auto;
            padding: 32px 20px 48px;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            display: grid;
            place-items: center;
            box-shadow: var(--shadow);
            font-size: 20px;
            color: #031522;
        }
        .pill {
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.02);
            border-radius: 999px;
            padding: 8px 14px;
            color: var(--muted);
            font-size: 0.85rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(76,201,240,0.12), rgba(123,97,255,0.18));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 28px 28px 20px;
            box-shadow: var(--shadow);
            margin-bottom: 30px;
        }
        .hero h1 {
            margin: 0 0 10px;
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.1;
        }
        .hero p {
            margin: 0;
            color: var(--muted);
            max-width: 760px;
            font-size: 1.04rem;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(200px, 1fr));
            gap: 18px;
            margin: 30px 0;
        }
        .card {
            background: rgba(12, 24, 39, 0.8);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 20px;
            box-shadow: var(--shadow);
        }
        .card-label {
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 12px;
        }
        .metric-value {
            font-size: clamp(1.8rem, 2vw, 2.4rem);
            font-weight: 700;
            margin: 0;
        }
        .metric-sub {
            font-size: 0.82rem;
            color: var(--muted);
            margin-top: 8px;
        }
        .layout {
            display: grid;
            grid-template-columns: 1.5fr 1fr;
            gap: 22px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            color: var(--text);
        }
        th, td {
            text-align: left;
            padding: 12px 10px;
            border-bottom: 1px solid var(--border);
            font-size: 0.92rem;
        }
        th {
            color: var(--muted);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            font-size: 0.72rem;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(160px, 1fr));
            gap: 14px;
            margin-top: 15px;
        }
        .feature-stat {
            padding: 14px 14px 12px;
            border-radius: 14px;
            background: var(--panel-light);
            border: 1px solid var(--border);
        }
        .feature-stat .name {
            font-size: 0.75rem;
            color: var(--muted);
            display: block;
            margin-bottom: 8px;
        }
        .feature-stat .value {
            font-weight: 700;
            font-size: 1.2rem;
        }
        .form-card {
            background: linear-gradient(180deg, rgba(11, 24, 38, 0.94), rgba(15, 28, 45, 0.92));
            border: 1px solid var(--border);
            border-radius: 22px;
            padding: 20px;
            box-shadow: var(--shadow);
        }
        form {
            display: grid;
            gap: 14px;
        }
        .field-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        label {
            display: flex;
            flex-direction: column;
            gap: 8px;
            color: var(--muted);
            font-size: 0.82rem;
        }
        input, select {
            width: 100%;
            background: rgba(255,255,255,0.02);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 11px 12px;
            font-size: 0.96rem;
        }
        button {
            border: none;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: #041019;
            border-radius: 12px;
            padding: 12px 14px;
            font-weight: 700;
            font-size: 0.95rem;
            cursor: pointer;
            box-shadow: 0 14px 25px rgba(76,201,240,0.25);
        }
        .result-box {
            margin-top: 20px;
            background: rgba(57,217,138,0.08);
            border: 1px solid rgba(57,217,138,0.3);
            border-radius: 16px;
            padding: 16px;
            display: none;
        }
        .result-box.visible { display: block; }
        .result-box strong {
            display: block;
            font-size: 1.25rem;
            margin-bottom: 8px;
        }
        .mini-label {
            color: var(--muted);
            display: inline-block;
            margin-right: 10px;
            font-size: 0.78rem;
        }
        .tags {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 20px;
        }
        .tag {
            padding: 7px 10px;
            border-radius: 999px;
            border: 1px solid var(--border);
            color: var(--muted);
            font-size: 0.75rem;
        }
        @media (max-width: 980px) {
            .stats-grid, .layout { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 700px) {
            .stats-grid, .layout, .field-row, .feature-grid { grid-template-columns: 1fr; }
            .topbar { flex-direction: column; align-items: flex-start; }
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark">AI</div>
                <span>Ad Engagement</span>
            </div>
            <div class="pill">Logistic Regression model</div>
        </div>

        <section class="hero">
            <h1 id="title">__TITLE__</h1>
            <p id="description">__DESCRIPTION__</p>
            <div class="tags">
                <span class="tag">Audience analysis</span>
                <span class="tag">Model evaluation</span>
                <span class="tag">Profile scoring</span>
            </div>
        </section>

        <section class="stats-grid" id="statsGrid"></section>

        <section class="layout">
            <div class="card">
                <div class="card-label">Audience preview</div>
                <div style="overflow-x:auto;">
                    <table id="audienceTable"></table>
                </div>
            </div>

            <div class="form-card">
                <div class="card-label">Predict visitor behavior</div>
                <form id="predictForm">
                    <div class="field-row">
                        <label>
                            Daily Time Spent on Site
                            <input id="siteTime" name="Daily Time Spent on Site" type="number" step="any" value="0" />
                        </label>
                        <label>
                            Age
                            <input id="age" name="Age" type="number" step="1" value="0" />
                        </label>
                    </div>
                    <div class="field-row">
                        <label>
                            Area Income
                            <input id="income" name="Area Income" type="number" step="any" value="0" />
                        </label>
                        <label>
                            Daily Internet Usage
                            <input id="internet" name="Daily Internet Usage" type="number" step="any" value="0" />
                        </label>
                    </div>
                    <label>
                        Male
                        <select id="male" name="Male">
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </label>
                    <button type="submit">Score profile</button>
                </form>
                <div id="resultBox" class="result-box"></div>
            </div>
        </section>

        <section class="card" style="margin-top: 22px;">
            <div class="card-label">Feature statistics</div>
            <div id="featureStats" class="feature-grid"></div>
        </section>
    </div>

    <script>
        const dashboardData = __PAYLOAD__;
        const defaults = dashboardData.default_profile;
        document.getElementById('siteTime').value = defaults['Daily Time Spent on Site'];
        document.getElementById('age').value = defaults.Age;
        document.getElementById('income').value = defaults['Area Income'];
        document.getElementById('internet').value = defaults['Daily Internet Usage'];
        document.getElementById('male').value = String(defaults.Male);

        const metricConfig = [
            ['Visitors', dashboardData.metrics.visitors, 'Total records in the ad dataset'],
            ['Click-through rate', `${dashboardData.metrics.click_through_rate.toFixed(2)}%`, 'Share of visitors who clicked'],
            ['Model accuracy', `${(dashboardData.metrics.model_accuracy * 100).toFixed(2)}%`, 'Accuracy on the test split'],
            ['Test observations', dashboardData.metrics.test_observations, 'Rows used for evaluation']
        ];

        const statsGrid = document.getElementById('statsGrid');
        statsGrid.innerHTML = metricConfig.map(([label, value, sub]) => `
            <div class="card">
                <div class="card-label">${label}</div>
                <p class="metric-value">${value}</p>
                <div class="metric-sub">${sub}</div>
            </div>
        `).join('');

        const previewRows = dashboardData.audience_preview;
        document.getElementById('audienceTable').innerHTML = `
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Age</th>
                    <th>Income</th>
                    <th>Usage</th>
                    <th>Clicked</th>
                </tr>
            </thead>
            <tbody>
                ${previewRows.map((row) => `
                    <tr>
                        <td>${row['Daily Time Spent on Site'].toFixed(2)}</td>
                        <td>${row.Age}</td>
                        <td>${row['Area Income'].toFixed(2)}</td>
                        <td>${row['Daily Internet Usage'].toFixed(2)}</td>
                        <td>${row['Clicked on Ad']}</td>
                    </tr>
                `).join('')}
            </tbody>
        `;

        const featureStatsEl = document.getElementById('featureStats');
        const featureEntries = Object.entries(dashboardData.feature_stats);
        featureStatsEl.innerHTML = featureEntries.map(([feature, stats]) => `
            <div class="feature-stat">
                <span class="name">${feature}</span>
                <div class="value">${stats.median.toFixed(2)}</div>
                <div class="metric-sub">Range ${stats.minimum.toFixed(2)} - ${stats.maximum.toFixed(2)}</div>
            </div>
        `).join('');

        const form = document.getElementById('predictForm');
        const resultBox = document.getElementById('resultBox');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const formData = new FormData(form);
            const payload = Object.fromEntries(formData.entries());
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await response.json();
            resultBox.classList.add('visible');
            if (!response.ok) {
                resultBox.style.background = 'rgba(255, 138, 128, 0.08)';
                resultBox.style.borderColor = 'rgba(255, 138, 128, 0.35)';
                resultBox.innerHTML = `<strong>Validation error</strong><span>${data.detail}</span>`;
                return;
            }
            resultBox.style.background = 'rgba(57, 217, 138, 0.08)';
            resultBox.style.borderColor = 'rgba(57, 217, 138, 0.3)';
            resultBox.innerHTML = `
                <strong>${data.result}</strong>
                <span><span class="mini-label">Probability</span>${(data.click_probability * 100).toFixed(2)}%</span>
            `;
        });
    </script>
</body>
</html>
"""
    html = html.replace("__TITLE__", result["title"])
    html = html.replace("__DESCRIPTION__", result["description"])
    html = html.replace("__PAYLOAD__", payload)
    return HTMLResponse(html, status_code=200)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index(
    test_fraction: float = Query(0.33, ge=0.20, le=0.40),
    random_seed: int = Query(101, ge=0),
) -> HTMLResponse:
    return dashboard_page(test_fraction, random_seed)


@app.get("/api/dashboard")
def get_dashboard(
    test_fraction: float = Query(0.33, ge=0.20, le=0.40),
    random_seed: int = Query(101, ge=0),
) -> dict[str, Any]:
    return dashboard_result(test_fraction, random_seed)


@app.get("/api/data")
def get_data() -> dict[str, Any]:
    data = load_data()
    return {
        "columns": data.columns.tolist(),
        "rows": data.to_dict(orient="records"),
    }


@app.post("/api/predict")
def predict_visitor(
    visitor: dict[str, Any],
    test_fraction: float = Query(0.33, ge=0.20, le=0.40),
    random_seed: int = Query(101, ge=0),
) -> dict[str, Any]:
    missing_features = [feature for feature in FEATURES if feature not in visitor]
    unexpected_features = [feature for feature in visitor if feature not in FEATURES]
    if missing_features or unexpected_features:
        details = []
        if missing_features:
            details.append("Missing features: " + ", ".join(missing_features))
        if unexpected_features:
            details.append("Unexpected features: " + ", ".join(unexpected_features))
        raise HTTPException(status_code=422, detail=". ".join(details))

    try:
        values = {feature: float(visitor[feature]) for feature in FEATURES}
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="All visitor features must be numeric.")

    data = load_data()
    for feature, value in values.items():
        minimum = float(data[feature].min())
        maximum = float(data[feature].max())
        if not pd.notna(value) or value < minimum or value > maximum:
            raise HTTPException(
                status_code=422,
                detail=f"{feature} must be between {minimum} and {maximum}.",
            )
    if values["Male"] not in (0, 1):
        raise HTTPException(status_code=422, detail="Male must be 0 or 1.")

    analysis = analyze_data(test_fraction, random_seed)
    profile = pd.DataFrame([values], columns=FEATURES)
    probability = float(analysis["model"].predict_proba(profile)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "result": "Likely to click" if prediction else "Unlikely to click",
        "click_probability": probability,
    }