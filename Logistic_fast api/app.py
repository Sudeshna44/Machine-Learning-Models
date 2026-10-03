from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
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
DATASETS: dict[str, pd.DataFrame] = {}

app = FastAPI(
    title="Ad Engagement Logistic Regression API",
    description="Explore advertising data, evaluate a logistic regression model, and score visitor profiles.",
    version="1.0.0",
)


class VisitorProfile(BaseModel):
    daily_time_spent_on_site: float = Field(alias="Daily Time Spent on Site")
    age: float = Field(alias="Age")
    area_income: float = Field(alias="Area Income")
    daily_internet_usage: float = Field(alias="Daily Internet Usage")
    male: int = Field(alias="Male", ge=0, le=1)


def validate_data(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing_columns))

    cleaned = data.copy()
    for column in REQUIRED_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if cleaned[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Required feature and target columns must contain numeric, non-missing values.")
    if not cleaned[TARGET].isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 or 1.")
    if cleaned[TARGET].nunique() < 2:
        raise ValueError("The target column must contain both click outcomes.")
    return cleaned


def get_data(dataset_id: str) -> pd.DataFrame:
    if dataset_id in DATASETS:
        return DATASETS[dataset_id]
    if dataset_id != "default":
        raise HTTPException(status_code=404, detail="Dataset not found. Upload a CSV to obtain a dataset_id.")
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail=f"Could not find {DATA_PATH.name} next to app.py.")
    try:
        data = validate_data(pd.read_csv(DATA_PATH))
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    DATASETS["default"] = data
    return data


def analyze(data: pd.DataFrame, test_fraction: float, random_seed: int) -> dict:
    features = data[FEATURES]
    target = data[TARGET]
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=test_fraction,
            random_state=random_seed,
        )
        model = LogisticRegression(solver="liblinear")
        model.fit(x_train, y_train)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=f"Unable to train model: {error}") from error

    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    return {
        "model": model,
        "report": report,
        "confusion_matrix": matrix,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "random_seed": random_seed,
        "test_fraction": test_fraction,
    }


def json_records(data: pd.DataFrame) -> list[dict]:
    safe_data = data.astype(object).where(pd.notna(data), None)
    return safe_data.to_dict(orient="records")


def run_analysis(
    dataset_id: str,
    test_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, dict]:
    data = get_data(dataset_id)
    return data, analyze(data, test_fraction, random_seed)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(Path(__file__).with_name("dashboard.html"), media_type="text/html")


@app.post("/data/upload", status_code=201)
async def upload_csv(file: UploadFile = File(...)) -> dict:
    if file.filename is None or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    try:
        data = validate_data(pd.read_csv(BytesIO(await file.read())))
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    dataset_id = str(uuid4())
    DATASETS[dataset_id] = data
    return {"dataset_id": dataset_id, "rows": len(data), "columns": list(data.columns)}


@app.get("/overview")
def overview(
    dataset_id: str = "default",
    test_fraction: float = Query(0.33, ge=0.20, le=0.40),
    random_seed: int = Query(101, ge=0),
) -> dict:
    data, result = run_analysis(dataset_id, test_fraction, random_seed)
    report = result["report"]
    return {
        "visitors": len(data),
        "click_through_rate": float(data[TARGET].mean()),
        "model_accuracy": report["accuracy"],
        "test_observations": result["test_size"],
        "outcome_counts": {
            "did_not_click": int((data[TARGET] == 0).sum()),
            "clicked": int((data[TARGET] == 1).sum()),
        },
        "profile_defaults": {feature: float(data[feature].median()) for feature in FEATURES},
        "feature_ranges": {
            feature: [float(data[feature].min()), float(data[feature].max())]
            for feature in FEATURES
        },
        "sample": json_records(data.head(10)),
    }


@app.get("/explore/age")
def explore_age(dataset_id: str = "default", bins: int = Query(30, ge=1, le=100)) -> dict:
    data = get_data(dataset_id)
    edges = np.histogram_bin_edges(data["Age"], bins=bins)
    counts = {
        str(outcome): np.histogram(data.loc[data[TARGET] == outcome, "Age"], bins=edges)[0].tolist()
        for outcome in [0, 1]
    }
    return {"bin_edges": edges.tolist(), "counts_by_click_outcome": counts}


@app.get("/explore/scatter")
def explore_scatter(dataset_id: str = "default") -> dict:
    data = get_data(dataset_id)
    return {
        "x": "Daily Time Spent on Site",
        "y": "Daily Internet Usage",
        "points": json_records(data[[FEATURES[0], FEATURES[3], TARGET]]),
    }


@app.get("/explore/relationship")
def explore_relationship(
    feature: str = Query(FEATURES[0]),
    dataset_id: str = "default",
) -> dict:
    if feature not in FEATURES:
        raise HTTPException(status_code=422, detail=f"feature must be one of: {', '.join(FEATURES)}")
    data = get_data(dataset_id)
    return {
        "feature": feature,
        "target": TARGET,
        "points": json_records(data[[feature, TARGET]]),
    }


@app.get("/model/performance")
def model_performance(
    dataset_id: str = "default",
    test_fraction: float = Query(0.33, ge=0.20, le=0.40),
    random_seed: int = Query(101, ge=0),
) -> dict:
    _, result = run_analysis(dataset_id, test_fraction, random_seed)
    report = result["report"]
    return {
        "accuracy": report["accuracy"],
        "classes": {
            "did_not_click": report["0"],
            "clicked": report["1"],
        },
        "confusion_matrix": {
            "labels": ["No click", "Clicked"],
            "values": result["confusion_matrix"].tolist(),
        },
        "train_observations": result["train_size"],
        "test_observations": result["test_size"],
        "random_seed": result["random_seed"],
        "test_fraction": result["test_fraction"],
    }


@app.post("/predict")
def predict(
    visitor: VisitorProfile,
    dataset_id: str = "default",
    test_fraction: float = Query(0.33, ge=0.20, le=0.40),
    random_seed: int = Query(101, ge=0),
) -> dict:
    _, result = run_analysis(dataset_id, test_fraction, random_seed)
    profile = pd.DataFrame(
        [[
            visitor.daily_time_spent_on_site,
            visitor.age,
            visitor.area_income,
            visitor.daily_internet_usage,
            visitor.male,
        ]],
        columns=FEATURES,
    )
    model = result["model"]
    probability_column = list(model.classes_).index(1)
    probability = float(model.predict_proba(profile)[0, probability_column])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "label": "Likely to click" if prediction else "Unlikely to click",
        "click_probability": probability,
        "threshold": 0.5,
    }