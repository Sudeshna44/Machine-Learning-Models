from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
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

st.set_page_config(
    page_title="Ad Engagement | Logistic Regression",
    page_icon="📊",
    layout="wide",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --ink: #172b24;
        --muted: #43574e;
        --green: #17634d;
        --green-dark: #104b3b;
        --line: #c6d2c8;
        --canvas: #f3f6f1;
    }
    html, body, [class*="st-"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Manrope', sans-serif; letter-spacing: 0; color: var(--ink); }
    .stApp, [data-testid="stAppViewContainer"] { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: rgba(243, 246, 241, 0.96); }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stCaptionContainer"] p,
    [data-testid="stWidgetLabel"] p,
    label { color: var(--muted); }
    [data-testid="stCaptionContainer"] p { font-size: 0.92rem; }
    [data-testid="stMetric"] {
        background: #ffffff; border: 1px solid var(--line); border-top: 3px solid var(--green);
        padding: 14px 16px; border-radius: 6px;
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--green-dark); }
    [data-testid="stSidebar"] { background: #e5ede6; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    [data-testid="stSidebar"] label { color: var(--ink); }
    [data-testid="stTabs"] button { color: var(--muted); }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--green-dark); }
    [data-testid="stNumberInputContainer"],
    [data-testid="stTextInputRootElement"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #e5ede6 !important; border-color: var(--line) !important;
    }
    [data-testid="stNumberInputContainer"] input,
    [data-testid="stTextInputRootElement"] input,
    [data-testid="stSelectbox"] [data-baseweb="select"] * { color: var(--ink) !important; }
    [data-testid="stNumberInputContainer"] button { background-color: #e5ede6; color: var(--ink); }
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primaryFormSubmit"] {
        background: var(--green-dark) !important; border-color: var(--green-dark) !important; color: #ffffff !important;
    }
    [data-testid="stBaseButton-primaryFormSubmit"] p { color: #ffffff !important; }
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primaryFormSubmit"]:hover {
        background: var(--green-dark) !important; border-color: var(--green-dark) !important; color: #ffffff !important;
    }
    .st-key-_visitor_profile_Male [data-testid="stSelectbox"] .react-aria-ComboBox > div {
        background-color: var(--green-dark) !important; border-color: var(--green-dark) !important;
    }
    .st-key-_visitor_profile_Male [role="combobox"],
    .st-key-_visitor_profile_Male button,
    .st-key-_visitor_profile_Male svg { color: #ffffff !important; }
    [data-testid="stFileUploaderDropzone"] { background: #ffffff; border: 1px dashed var(--line); }
    [data-testid="stFileUploaderDropzone"] * { color: var(--muted); }
    [data-testid="stFileUploaderDropzone"] button {
        background: #ffffff !important; border-color: var(--line) !important; color: var(--ink) !important;
    }
    [data-testid="stFileUploaderDropzone"] button p { color: var(--ink) !important; }
    [data-testid="stFileUploaderDropzone"] [data-testid="stIconMaterial"] { display: none !important; }
    :focus-visible { outline: 3px solid #d06a43 !important; outline-offset: 2px; }
    .eyebrow { color: var(--green-dark); font-weight: 700; font-size: 0.78rem; text-transform: uppercase; }
    </style>
    """,
    unsafe_allow_html=True,
)
sns.set_theme(
    style="whitegrid",
    palette=["#17634d", "#b94f2d"],
    rc={
        "text.color": "#172b24",
        "axes.labelcolor": "#344a40",
        "xtick.color": "#43574e",
        "ytick.color": "#43574e",
        "axes.edgecolor": "#c6d2c8",
    },
)


@st.cache_data

def load_default_data(path: str, modified_time: float) -> pd.DataFrame:
    del modified_time
    return pd.read_csv(path)


@st.cache_resource

def fit_logistic_model(features: pd.DataFrame, target: pd.Series) -> LogisticRegression:
    model = LogisticRegression(solver="liblinear")
    return model.fit(features, target)


st.sidebar.markdown("## Ad engagement")
st.sidebar.caption("Logistic regression workspace")
uploaded_file = st.sidebar.file_uploader("Use another CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
elif DATA_PATH.exists():
    data = load_default_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
else:
    st.error(f"Could not find {DATA_PATH.name} next to app.py.")
    st.stop()

missing_columns = [column for column in [*FEATURES, TARGET] if column not in data.columns]
if missing_columns:
    st.error("The selected CSV is missing required columns: " + ", ".join(missing_columns))
    st.stop()

st.sidebar.divider()
test_fraction = st.sidebar.slider("Test set", min_value=0.20, max_value=0.40, value=0.33, step=0.01)
random_seed = st.sidebar.number_input("Random seed", min_value=0, value=101, step=1)

features = data[FEATURES]
target = data[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=test_fraction,
    random_state=int(random_seed),
)
model = fit_logistic_model(X_train, y_train)
predictions = model.predict(X_test)
report = classification_report(y_test, predictions, output_dict=True, zero_division=0)

click_rate = target.mean() * 100

def render_header(title: str, description: str) -> None:
    st.markdown('<p class="eyebrow">Campaign analytics / Classification</p>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def save_explore_feature() -> None:
    st.session_state["explore_selected_feature"] = st.session_state["_explore_selected_feature"]


def render_overview() -> None:
    render_header("Who clicks on an ad?", "A snapshot of the advertising audience and campaign response.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Visitors", f"{len(data):,}")
    metric_columns[1].metric("Click-through rate", f"{click_rate:.1f}%")
    metric_columns[2].metric("Model accuracy", f"{report['accuracy']:.1%}")
    metric_columns[3].metric("Test observations", f"{len(y_test):,}")

    left_column, right_column = st.columns([1.15, 0.85], gap="large")
    with left_column:
        st.subheader("Audience snapshot")
        st.dataframe(data.head(10), width="stretch", hide_index=True)
    with right_column:
        st.subheader("Click outcomes")
        outcome_counts = target.value_counts().reindex([0, 1], fill_value=0)
        outcome_frame = pd.DataFrame(
            {"Outcome": ["Did not click", "Clicked"], "Visitors": outcome_counts.values}
        )
        figure, axis = plt.subplots(figsize=(6, 3.8))
        sns.barplot(data=outcome_frame, x="Outcome", y="Visitors", hue="Outcome", legend=False, ax=axis)
        axis.set_xlabel("")
        axis.set_ylabel("Visitors")
        axis.set_title("Observed ad response")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)


def render_explore() -> None:
    render_header("Explore the audience", "Compare visitor behavior and feature distributions by ad response.")
    chart_column, scatter_column = st.columns(2, gap="large")
    with chart_column:
        st.subheader("Age by click outcome")
        figure, axis = plt.subplots(figsize=(7, 4.3))
        sns.histplot(data=data, x="Age", hue=TARGET, bins=30, element="step", ax=axis)
        axis.set_xlabel("Age (years)")
        axis.set_ylabel("Visitors")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with scatter_column:
        st.subheader("Site time and internet use")
        figure, axis = plt.subplots(figsize=(7, 4.3))
        sns.scatterplot(
            data=data,
            x="Daily Time Spent on Site",
            y="Daily Internet Usage",
            hue=TARGET,
            alpha=0.72,
            s=35,
            ax=axis,
        )
        axis.set_xlabel("Daily time on site (minutes)")
        axis.set_ylabel("Daily internet use (minutes)")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    st.subheader("Feature relationships")
    if "_explore_selected_feature" not in st.session_state:
        st.session_state["_explore_selected_feature"] = st.session_state.get(
            "explore_selected_feature", FEATURES[0]
        )
    selected_x = st.selectbox(
        "Compare a feature",
        FEATURES,
        key="_explore_selected_feature",
        on_change=save_explore_feature,
    )
    figure, axis = plt.subplots(figsize=(10, 4.2))
    sns.boxplot(data=data, x=TARGET, y=selected_x, hue=TARGET, legend=False, ax=axis)
    axis.set_xlabel("Clicked on ad (0 = no, 1 = yes)")
    axis.set_ylabel(selected_x)
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def render_model_performance() -> None:
    render_header("Model performance", "Review how logistic regression classifies held-out visitors.")
    st.subheader("Test-set performance")
    score_column, matrix_column = st.columns([1, 1], gap="large")
    with score_column:
        report_frame = pd.DataFrame(report).T.loc[["0", "1"], ["precision", "recall", "f1-score", "support"]]
        report_frame.index = ["Did not click", "Clicked"]
        st.dataframe(report_frame.style.format({
            "precision": "{:.2f}",
            "recall": "{:.2f}",
            "f1-score": "{:.2f}",
            "support": "{:.0f}",
        }), width="stretch")
        st.caption(f"Train: {len(X_train):,} visitors · Test: {len(X_test):,} visitors · Seed: {int(random_seed)}")
    with matrix_column:
        figure, axis = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(
            confusion_matrix(y_test, predictions),
            annot=True,
            fmt="d",
            cmap="YlGn",
            cbar=False,
            xticklabels=["No click", "Clicked"],
            yticklabels=["No click", "Clicked"],
            ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title("Confusion matrix")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)


def render_visitor_scoring() -> None:
    render_header("Score a visitor", "Estimate the probability that a visitor will click an ad.")
    st.caption("Model: logistic regression · Inputs: site time, age, area income, internet use, and gender flag")
    saved_values = st.session_state.get("visitor_profile_values", {})
    with st.form("visitor_profile"):
        input_columns = st.columns(3)
        visitor_values = {}
        for index, feature in enumerate(FEATURES):
            minimum = float(data[feature].min())
            maximum = float(data[feature].max())
            default = float(data[feature].median())
            widget_key = f"_visitor_profile_{feature}"
            if widget_key not in st.session_state:
                initial_value = saved_values.get(feature, 0 if feature == "Male" else default)
                if feature == "Age":
                    initial_value = min(max(int(initial_value), int(minimum)), int(maximum))
                elif feature == "Male":
                    initial_value = int(initial_value) if int(initial_value) in (0, 1) else 0
                else:
                    initial_value = min(max(float(initial_value), minimum), maximum)
                st.session_state[widget_key] = initial_value

            if feature == "Male":
                visitor_values[feature] = input_columns[index % 3].selectbox(
                    "Male",
                    options=[0, 1],
                    format_func=lambda value: "Yes" if value else "No",
                    key=widget_key,
                )
            elif feature == "Age":
                age_minimum = int(minimum)
                age_maximum = int(maximum)
                current_age = int(st.session_state[widget_key])
                if current_age < age_minimum or current_age > age_maximum:
                    st.session_state[widget_key] = min(max(current_age, age_minimum), age_maximum)
                visitor_values[feature] = input_columns[index % 3].number_input(
                    "Age",
                    min_value=age_minimum,
                    max_value=age_maximum,
                    step=1,
                    key=widget_key,
                )
            else:
                current_value = float(st.session_state[widget_key])
                if current_value < minimum or current_value > maximum:
                    st.session_state[widget_key] = min(max(current_value, minimum), maximum)
                visitor_values[feature] = input_columns[index % 3].number_input(
                    feature,
                    min_value=minimum,
                    max_value=maximum,
                    step=0.5 if feature != "Area Income" else 100.0,
                    key=widget_key,
                )
        submitted = st.form_submit_button("Estimate click likelihood", type="primary")

    if submitted:
        st.session_state["visitor_profile_values"] = visitor_values.copy()

    saved_values = st.session_state.get("visitor_profile_values")
    if saved_values:
        visitor = pd.DataFrame([saved_values], columns=FEATURES)
        probability = float(model.predict_proba(visitor)[0, 1])
        prediction = int(probability >= 0.5)
        result_column, probability_column = st.columns([1, 2])
        with result_column:
            st.success("Likely to click" if prediction else "Unlikely to click")
        with probability_column:
            st.metric("Estimated click probability", f"{probability:.1%}")
        st.progress(probability)


navigation = st.navigation(
    [
        st.Page(render_overview, title="Overview"),
        st.Page(render_explore, title="Explore"),
        st.Page(render_model_performance, title="Model performance"),
        st.Page(render_visitor_scoring, title="Score a visitor"),
    ]
)
navigation.run()
