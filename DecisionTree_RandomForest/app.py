from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path(__file__).resolve().parent / "loan_data.csv"
TARGET = "not.fully.paid"
PAID_COLOR = "#147D73"
UNPAID_COLOR = "#D6654F"

st.set_page_config(
    page_title="Lending Club | Portfolio Lab",
    page_icon="LC",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    :root {
        color-scheme: light;
        --ink: #173431;
        --body: #294541;
        --muted: #536b65;
        --teal: #147d73;
        --coral: #c4513b;
        --line: #d6e4de;
        --surface: #ffffff;
    }
    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
        font-family: 'DM Sans', sans-serif;
        letter-spacing: 0;
        color: var(--body);
    }
    [data-testid="stAppViewContainer"] h1,
    [data-testid="stAppViewContainer"] h2,
    [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] h4 {
        color: var(--ink) !important;
        font-family: 'DM Sans', sans-serif !important;
        letter-spacing: 0;
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"] p,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] {
        color: var(--body) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #edf6f2 0%, #ffffff 58%, #f1f6f2 100%);
    }
    [data-testid="stHeader"] { background: rgba(255,255,255,0.72); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #173431 0%, #204d47 100%);
        color: #f2f8f5;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #f2f8f5; }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] { color: #d4e7df; }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] { color: #ffffff; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #ffffff; }
    [data-testid="stSidebar"] [data-baseweb="select"] input { color: #173431; }
    [data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #536b65; }
    .block-container { max-width: 1440px; padding-top: 2.2rem; padding-bottom: 3rem; }
    h1, h2, h3, h4 { color: var(--ink); letter-spacing: 0; }
    h1 { font-weight: 700; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--body); }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.94);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 16px 18px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--ink); }
    [data-testid="stTabs"] button { color: var(--muted); font-weight: 600; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal); }
    [data-testid="stWidgetLabel"] p { color: var(--ink); font-weight: 600; }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
    .stButton button { color: #ffffff; background: var(--teal); border-color: var(--teal); }
    .stButton button:hover { color: #ffffff; background: #0f685f; border-color: #0f685f; }
    .eyebrow {
        color: var(--teal);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .lede { color: var(--muted); font-size: 1.02rem; }
    .section-note { color: var(--muted); font-size: 0.9rem; }
    .sidebar-kicker {
        color: #9dd2c2;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        text-transform: uppercase;
    }
    .sidebar-copy { color: #d6e8e1 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #d6e8e1 !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h4 {
        color: #ffffff !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_loans() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.name} beside app.py.")
    data = pd.read_csv(DATA_PATH)
    required = {
        "credit.policy",
        "purpose",
        "int.rate",
        "installment",
        "fico",
        TARGET,
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"The CSV is missing required columns: {', '.join(sorted(missing))}")
    return data


@st.cache_resource(show_spinner="Training model...")
def train_model(
    model_name: str,
    test_fraction: float,
    random_seed: int,
    forest_trees: int,
    balance_classes: bool,
):
    data = load_loans()
    features = pd.get_dummies(
        data.drop(columns=[TARGET]), columns=["purpose"], drop_first=True
    )
    target = data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=target,
    )
    class_weight = "balanced" if balance_classes else None

    if model_name == "Random forest":
        model = RandomForestClassifier(
            n_estimators=forest_trees,
            random_state=random_seed,
            class_weight=class_weight,
            n_jobs=-1,
        )
    else:
        model = DecisionTreeClassifier(
            random_state=random_seed,
            class_weight=class_weight,
        )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Paid in full", "Not fully paid"],
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    return model, x_test, y_test, predictions, report, matrix


def style_plot_axis(axis, grid_axis="y") -> None:
    axis.tick_params(colors="#536b65")
    axis.xaxis.label.set_color("#294541")
    axis.yaxis.label.set_color("#294541")
    axis.title.set_color("#173431")
    for spine in axis.spines.values():
        spine.set_color("#d6e4de")
    if grid_axis:
        axis.grid(axis=grid_axis, color="#e5eeea", linewidth=0.7)
        axis.set_axisbelow(True)


try:
    loans = load_loans()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

unpaid_rate = loans[TARGET].mean()

with st.sidebar:
    st.markdown('<div class="sidebar-kicker">Portfolio snapshot</div>', unsafe_allow_html=True)
    st.markdown("### Lending Club")
    st.markdown('<p class="sidebar-copy">2007-2010 loan performance sample</p>', unsafe_allow_html=True)
    st.metric("Loans analyzed", f"{len(loans):,}")
    st.progress(unpaid_rate, text=f"{unpaid_rate:.1%} not fully paid")
    st.markdown(
        '<p class="sidebar-copy">The unpaid class is the minority. Use recall and the confusion matrix alongside accuracy.</p>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown('<div class="sidebar-kicker">Data source</div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-copy">loan_data.csv<br>14 fields, no missing values</p>', unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Lending Club / Portfolio Lab</div>', unsafe_allow_html=True)
st.title("Loan performance, made visible")
st.markdown(
    '<p class="lede">Explore the 2007-2010 loan portfolio and test how tree models identify loans not paid in full.</p>',
    unsafe_allow_html=True,
)

overview_tab, explore_tab, model_tab, data_tab = st.tabs(
    ["Overview", "Explore portfolio", "Model lab", "Data"]
)

with overview_tab:
    st.subheader("Portfolio at a glance")
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("Loans in sample", f"{len(loans):,}")
    metric_b.metric("Not fully paid", f"{int(loans[TARGET].sum()):,}")
    metric_c.metric("Unpaid share", f"{unpaid_rate:.1%}")
    metric_d.metric("Median FICO", f"{loans['fico'].median():.0f}")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Loan purpose")
        purpose_counts = loans["purpose"].value_counts().sort_values()
        st.bar_chart(purpose_counts, horizontal=True, color=PAID_COLOR)
    with right:
        st.markdown("#### Payment outcome")
        outcomes = pd.Series(
            {
                "Paid in full": int((loans[TARGET] == 0).sum()),
                "Not fully paid": int((loans[TARGET] == 1).sum()),
            },
            name="Loans",
        )
        st.bar_chart(outcomes, color=UNPAID_COLOR)
        st.markdown(
            '<p class="section-note">Only about one loan in six is marked not fully paid. Accuracy alone can hide missed cases.</p>',
            unsafe_allow_html=True,
        )

with explore_tab:
    st.subheader("Explore the portfolio")
    filter_a, filter_b, filter_c = st.columns([1.4, 1.1, 1.5])
    with filter_a:
        selected_purposes = st.multiselect(
            "Loan purpose",
            options=sorted(loans["purpose"].unique()),
            default=sorted(loans["purpose"].unique()),
        )
    with filter_b:
        payment_status = st.selectbox(
            "Payment status",
            ["All loans", "Paid in full", "Not fully paid"],
        )
    with filter_c:
        fico_range = st.slider(
            "FICO range",
            min_value=int(loans["fico"].min()),
            max_value=int(loans["fico"].max()),
            value=(int(loans["fico"].min()), int(loans["fico"].max())),
        )

    filtered = loans[
        loans["purpose"].isin(selected_purposes)
        & loans["fico"].between(fico_range[0], fico_range[1])
    ]
    if payment_status == "Paid in full":
        filtered = filtered[filtered[TARGET] == 0]
    elif payment_status == "Not fully paid":
        filtered = filtered[filtered[TARGET] == 1]

    st.caption(f"Showing {len(filtered):,} of {len(loans):,} loans")
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown("#### FICO score by payment outcome")
        figure, axis = plt.subplots(figsize=(8, 4.4))
        axis.hist(
            filtered.loc[filtered[TARGET] == 0, "fico"],
            bins=28,
            alpha=0.72,
            color=PAID_COLOR,
            label="Paid in full",
        )
        axis.hist(
            filtered.loc[filtered[TARGET] == 1, "fico"],
            bins=28,
            alpha=0.72,
            color=UNPAID_COLOR,
            label="Not fully paid",
        )
        axis.set_xlabel("FICO score")
        axis.set_ylabel("Number of loans")
        axis.spines[["top", "right"]].set_visible(False)
        style_plot_axis(axis)
        axis.legend(frameon=False)
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    with chart_b:
        st.markdown("#### Interest rate and FICO")
        figure, axis = plt.subplots(figsize=(8, 4.4))
        for outcome, label, color in [
            (0, "Paid in full", PAID_COLOR),
            (1, "Not fully paid", UNPAID_COLOR),
        ]:
            subset = filtered[filtered[TARGET] == outcome]
            if len(subset) > 1800:
                subset = subset.sample(1800, random_state=101)
            axis.scatter(
                subset["fico"],
                subset["int.rate"],
                s=13,
                alpha=0.38,
                label=label,
                color=color,
                edgecolors="none",
            )
        axis.set_xlabel("FICO score")
        axis.set_ylabel("Interest rate")
        axis.spines[["top", "right"]].set_visible(False)
        style_plot_axis(axis)
        axis.legend(frameon=False)
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    if filtered.empty:
        st.info("No loans match these filters. Adjust the purpose, status, or FICO range.")
    else:
        with st.expander("View matching loans"):
            st.dataframe(filtered, use_container_width=True, hide_index=True)

with model_tab:
    st.subheader("Train and evaluate a classifier")
    st.markdown(
        '<p class="section-note">The target is <code>not.fully.paid</code>: 1 means the loan was not paid in full.</p>',
        unsafe_allow_html=True,
    )

    control_a, control_b, control_c, control_d = st.columns(4)
    with control_a:
        model_name = st.selectbox("Model", ["Random forest", "Decision tree"])
    with control_b:
        test_fraction = st.slider("Test-set share", 0.15, 0.40, 0.30, 0.05)
    with control_c:
        forest_trees = st.slider("Trees in forest", 100, 800, 600, 100)
    with control_d:
        balance_classes = st.toggle("Balance class weights", value=False)

    model, x_test, y_test, predictions, report, matrix = train_model(
        model_name,
        test_fraction,
        101,
        forest_trees,
        balance_classes,
    )

    accuracy = (predictions == y_test).mean()
    unpaid_recall = report["Not fully paid"]["recall"]
    score_a, score_b, score_c = st.columns(3)
    score_a.metric("Test accuracy", f"{accuracy:.1%}")
    score_b.metric("Unpaid-loan recall", f"{unpaid_recall:.1%}")
    score_c.metric("Test loans", f"{len(y_test):,}")

    report_column, matrix_column = st.columns([1.1, 1])
    with report_column:
        st.markdown("#### Classification report")
        report_table = pd.DataFrame(report).T.loc[
            ["Paid in full", "Not fully paid", "macro avg", "weighted avg"],
            ["precision", "recall", "f1-score", "support"],
        ]
        st.dataframe(
            report_table.style.format(
                {"precision": "{:.2f}", "recall": "{:.2f}", "f1-score": "{:.2f}", "support": "{:.0f}"}
            ),
            use_container_width=True,
        )
    with matrix_column:
        st.markdown("#### Confusion matrix")
        figure, axis = plt.subplots(figsize=(5.5, 4.2))
        image = axis.imshow(matrix, cmap="GnBu")
        axis.set_xticks([0, 1], labels=["Paid", "Not paid"])
        axis.set_yticks([0, 1], labels=["Paid", "Not paid"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title("Loan payment outcome")
        style_plot_axis(axis, grid_axis=None)
        threshold = matrix.max() / 2
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}",
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "#173431",
                    fontsize=12,
                    fontweight="bold",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        st.pyplot(figure, use_container_width=True)
        plt.close(figure)

    if hasattr(model, "feature_importances_"):
        st.markdown("#### Most influential features")
        importances = pd.Series(model.feature_importances_, index=x_test.columns)
        importances = importances.sort_values(ascending=False).head(10).sort_values()
        st.bar_chart(importances, horizontal=True, color=PAID_COLOR)

    st.info(
        "The outcome is imbalanced: a model can score well by predicting 'paid in full' most of the time. "
        "Compare unpaid-loan recall and the confusion matrix, not accuracy alone. Class weighting changes this trade-off."
    )

with data_tab:
    st.subheader("Dataset")
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Sample rows")
        st.dataframe(loans.head(25), use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Numeric summary")
        st.dataframe(loans.describe().T, use_container_width=True)
    st.caption(f"Source: {DATA_PATH.name} | {len(loans):,} rows | {len(loans.columns)} columns")
