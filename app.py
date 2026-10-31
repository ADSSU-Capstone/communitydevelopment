import re
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from openpyxl import load_workbook

st.set_page_config(page_title="Barangay Priority Dashboard", layout="wide")

st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 8px 16px;
    background: #f8fafc;
    color: #1e293b;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1e40af !important;
    color: #ffffff !important;
    border-color: #1e40af !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }
h1, h2, h3 { color: #1e3a8a; }
</style>
""", unsafe_allow_html=True)

EXAMPLE_ROWS = [
    ("Esperanza", "Concordia", 161, 623, 0.0, 4.8, 88.2, 51.3, "High", "Low", "Medium", "Medium", "High Priority"),
    ("Esperanza", "Sinakungan", 81, 319, 72.0, 7.9, 76.1, 25.0, "High", "Low", "High", "High", "High Priority"),
    ("Esperanza", "Agsabu", 80, 327, 63.0, 8.8, 72.9, 25.5, "Medium", "Low", "Low", "Low", "Low Priority"),
    ("Esperanza", "Aguinaldo", 130, 515, 93.0, 6.8, 75.4, 29.0, "Low", "Low", "Medium", "Medium", "Medium Priority"),
    ("Esperanza", "Anolingan", 90, 369, 83.0, 7.7, 71.1, 23.5, "Medium", "Medium", "Medium", "Medium", "Medium Priority"),
    ("La Paz", "Bataan", 390, 1603, 80.0, 5.3, 85.6, 47.3, "Medium", "Low", "Low", "Low", "Low Priority"),
    ("La Paz", "Sabang Adgawan", 373, 1569, 85.0, 5.2, 88.1, 42.8, "High", "High", "Medium", "Medium", "High Priority"),
    ("Loreto", "San Mariano", 666, 2693, 599.0, 9.3, 72.0, 18.8, "High", "Low", "High", "High", "High Priority"),
    ("Prosperidad", "Aurora", 792, 3028, 752.0, 8.3, 73.8, 22.2, "Medium", "Low", "Medium", "Low", "Medium Priority"),
    ("San Luis", "Balit", 226, 871, 0.0, 4.0, 94.5, 53.5, "High", "High", "High", "High", "High Priority"),
    ("Bayugan City", "Berseba", 460, 1871, 323.0, 8.0, 78.5, 26.0, "Medium", "Low", "Low", "Low", "Low Priority"),
]

def strip_invisible(value):
    if not isinstance(value, str):
        return value
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Cf")
    value = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

with st.sidebar:
    uploaded = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    use_example = st.checkbox("Use example dataset (if no upload)", value=False)

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".csv"):
            raw = pd.read_csv(uploaded)
        else:
            wb = load_workbook(uploaded, read_only=True)
            if "Table 1" in wb.sheetnames:
                raw = pd.read_excel(uploaded, sheet_name="Table 1")
            else:
                raw = pd.read_excel(uploaded)
    except Exception as exc:
        st.error(f"Unable to read uploaded file: {exc}")
        st.stop()
elif use_example:
    raw = pd.DataFrame(EXAMPLE_ROWS, columns=[
        "City Name", "Barangay Name", "# of total Households", "Popolation",
        "Poverty_Rate (%)", "Unemployment_Rate (%)", "Access_to_Water (%)",
        "Internet_Access (%)", "Road_Access", "Health_Facility_Access",
        "School_Access", "Disaster_Risk_Level", "Priority_Level"])
    raw.insert(0, "Region Name", "Region XIII (Caraga)")
    raw.insert(1, "Province Name", "Agusan del Sur")
    raw.insert(3, "Year", 2025)
else:
    st.info("Upload a dataset or enable the example dataset to begin.")
    st.stop()

log = []
df = raw.copy()
df.columns = [str(c).strip() for c in df.columns]
df = df.rename(columns={"Popolation": "Population", "# of total Households": "Households"})

if "Priority_Level" not in df.columns:
    st.error("Priority_Level column not found.")
    st.stop()

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].map(strip_invisible)

before = len(df)
df = df[df["Priority_Level"].notna()]
df = df[df["Priority_Level"].astype(str).str.strip() != ""]
log.append(("Drop blank Priority_Level", before - len(df)))

before = len(df)
df = df[df["Barangay Name"].notna()]
df = df[df["Barangay Name"].astype(str).str.strip() != ""]
log.append(("Drop blank Barangay Name", before - len(df)))

priority_map = {
    "High Priority": "High Priority", "High": "High Priority",
    "Medium Priority": "Medium Priority", "Moderate Priority": "Medium Priority",
    "Medium": "Medium Priority", "Moderate": "Medium Priority",
    "Midium Priority": "Medium Priority",
    "Low Priority": "Low Priority", "Low": "Low Priority",
}
df["Priority_Level"] = df["Priority_Level"].astype(str).str.strip().map(priority_map)
before = len(df)
df = df.dropna(subset=["Priority_Level"])
log.append(("Normalize Priority_Level labels", before - len(df)))

if "Province Name" in df.columns:
    before = len(df)
    df = df[df["Province Name"].astype(str).str.strip() == "Agusan del Sur"]
    log.append(("Province whitelist (Agusan del Sur)", before - len(df)))

key_cols = [c for c in ["Region Name", "Province Name", "City Name", "Barangay Name", "Year"] if c in df.columns]
before = len(df)
df = df.drop_duplicates(subset=key_cols)
log.append(("Exact duplicates removed", before - len(df)))

num_cols = ["Population", "Households", "Poverty_Rate (%)", "Unemployment_Rate (%)", "Access_to_Water (%)", "Internet_Access (%)"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        miss = int(df[col].isna().sum())
        if miss > 0:
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0.0)
            log.append((f"Median imputation: {col}", miss))

cat_cols = ["Road_Access", "Health_Facility_Access", "School_Access", "Disaster_Risk_Level"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace("", np.nan)
        miss = int(df[col].isna().sum())
        if miss > 0:
            mode_val = df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else "Medium"
            df[col] = df[col].fillna(mode_val)
            log.append((f"Mode imputation: {col}", miss))

for col in ["Poverty_Rate (%)", "Unemployment_Rate (%)", "Access_to_Water (%)", "Internet_Access (%)"]:
    if col in df.columns:
        clipped = int(((df[col] < 0) | (df[col] > 100)).sum())
        df[col] = df[col].clip(0, 100)
        if clipped > 0:
            log.append((f"Clip out-of-range: {col}", clipped))

df = df.reset_index(drop=True)

ORD = {
    "Road_Access": {"High": 0, "Medium": 1, "Low": 2},
    "Health_Facility_Access": {"High": 0, "Medium": 1, "Low": 2},
    "School_Access": {"High": 0, "Medium": 1, "Low": 2},
    "Disaster_Risk_Level": {"High": 2, "Medium": 1, "Low": 0},
}
FEATURES = ["Population", "Households", "Poverty_Rate (%)", "Unemployment_Rate (%)",
            "Access_to_Water (%)", "Internet_Access (%)",
            "Road_Access", "Health_Facility_Access", "School_Access", "Disaster_Risk_Level"]

X_raw = df[FEATURES].copy()
for col, mapping in ORD.items():
    X_raw[col] = X_raw[col].map(mapping)
X_raw = X_raw.fillna(X_raw.median() if not X_raw.isna().all() else 0.0)

TARGET_MAP = {"High Priority": 2, "Medium Priority": 1, "Low Priority": 0}
y = df["Priority_Level"].map(TARGET_MAP).values

scaler = MinMaxScaler()
X_all_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=FEATURES, index=df.index)
mi_all = mutual_info_classif(X_all_scaled, y, random_state=42)

with st.sidebar:
    k_features = st.slider("Retained features (k)", 3, len(FEATURES), 8)

selector = SelectKBest(mutual_info_classif, k=k_features)
X = selector.fit_transform(X_all_scaled, y)
selected_mask = selector.get_support()
selected_features = [f for f, s in zip(FEATURES, selected_mask) if s]

try:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
except ValueError:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(random_state=42, kernel="rbf"),
}

results = []
fitted = {}
for name, mdl in models.items():
    try:
        mdl.fit(X_tr, y_tr)
        fitted[name] = mdl
        tr_pred = mdl.predict(X_tr)
        te_pred = mdl.predict(X_te)
        try:
            cv_scores = cross_val_score(mdl, X_tr, y_tr, cv=cv, scoring="accuracy")
            cv_mean = float(cv_scores.mean())
        except Exception:
            cv_mean = np.nan
        results.append({
            "Model": name,
            "Train Accuracy": accuracy_score(y_tr, tr_pred),
            "CV Accuracy": cv_mean,
            "Test Accuracy": accuracy_score(y_te, te_pred),
            "Precision": precision_score(y_te, te_pred, average="macro", zero_division=0),
            "Recall": recall_score(y_te, te_pred, average="macro", zero_division=0),
            "F1": f1_score(y_te, te_pred, average="macro", zero_division=0),
        })
    except Exception:
        results.append({"Model": name, "Train Accuracy": np.nan, "CV Accuracy": np.nan, "Test Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan, "F1": np.nan})

results_df = pd.DataFrame(results).set_index("Model")

rf = fitted.get("Random Forest")
if rf is not None:
    X_full_sel = selector.transform(X_all_scaled)
    pred_codes = rf.predict(X_full_sel)
    code_label = {0: "Low Priority", 1: "Medium Priority", 2: "High Priority"}
    df["Predicted_Priority"] = [code_label[c] for c in pred_codes]
    rf_importance = pd.Series(rf.feature_importances_, index=selected_features)
else:
    df["Predicted_Priority"] = df["Priority_Level"]
    rf_importance = pd.Series(dtype=float)

severity = pd.DataFrame(index=df.index)
severity["Poverty"] = df["Poverty_Rate (%)"].rank(pct=True)
severity["Unemployment"] = df["Unemployment_Rate (%)"].rank(pct=True)
severity["Water"] = 1 - df["Access_to_Water (%)"].rank(pct=True)
severity["Internet"] = 1 - df["Internet_Access (%)"].rank(pct=True)
severity["Road"] = df["Road_Access"].map(ORD["Road_Access"]).rank(pct=True)
severity["Health"] = df["Health_Facility_Access"].map(ORD["Health_Facility_Access"]).rank(pct=True)
severity["School"] = df["School_Access"].map(ORD["School_Access"]).rank(pct=True)
severity["Disaster"] = df["Disaster_Risk_Level"].map(ORD["Disaster_Risk_Level"]).rank(pct=True)
df["Primary_Gap"] = severity.idxmax(axis=1)

palette = {"High Priority": "#b91c1c", "Medium Priority": "#f59e0b", "Low Priority": "#16a34a"}

tabs = st.tabs([
    "Priority Classification",
    "Model Performance",
    "Feature Importance",
    "Confusion Matrix",
    "Priority Registry",
    "Policy Recommendations",
    "Data Preparation Log",
])

with tabs[0]:
    counts = df["Priority_Level"].value_counts().reindex(["High Priority", "Medium Priority", "Low Priority"], fill_value=0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Barangays evaluated", len(df))
    c2.metric("High Priority", int(counts["High Priority"]))
    c3.metric("Medium Priority", int(counts["Medium Priority"]))
    c4.metric("Low Priority", int(counts["Low Priority"]))

    breakdown = st.selectbox("Breakdown by", ["City Name", "Disaster_Risk_Level", "Road_Access", "Health_Facility_Access", "School_Access"], key="breakdown_select")

    ct = pd.crosstab(df[breakdown], df["Priority_Level"])
    order = ["High Priority", "Medium Priority", "Low Priority"]
    ct = ct.reindex(columns=[c for c in order if c in ct.columns], fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ct.plot(kind="bar", stacked=True, ax=ax, color=[palette[c] for c in ct.columns])
    ax.set_ylabel("Barangay count")
    ax.set_xlabel(breakdown)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Priority", frameon=False)
    plt.xticks(rotation=40, ha="right")
    st.pyplot(fig)
    plt.close(fig)

    st.write(f"The dataset covers {len(df)} barangays, of which {int(counts['High Priority'])} ({counts['High Priority']/max(len(df),1):.1%}) are High Priority under the LGU classification standard.")

    try:
        valid_features = [f for f in FEATURES if f in df.columns]
        summary = df.groupby("Priority_Level")[valid_features].mean().round(2)
        summary = summary.reindex([c for c in order if c in summary.index])
        st.dataframe(summary, use_container_width=True)
    except Exception:
        pass

with tabs[1]:
    disp = results_df.copy().round(3)
    st.dataframe(disp.style.highlight_max(axis=0, color="#dbeafe"), use_container_width=True)

    f1_series = results_df["F1"].sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    f1_series.plot(kind="barh", ax=ax, color="#1e40af")
    ax.set_xlabel("F1 score (macro)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    best_model = results_df["F1"].idxmax() if not results_df["F1"].isna().all() else "n/a"
    best_f1 = results_df.loc[best_model, "F1"] if best_model in results_df.index else float("nan")
    st.write(f"With k={k_features} retained features, {best_model} records the highest macro F1 ({best_f1:.3f}) among the four classifiers evaluated.")

with tabs[2]:
    mi_sorted = pd.Series(mi_all, index=FEATURES).sort_values(ascending=False)
    mi_top = mi_sorted.head(k_features).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    mi_top.plot(kind="barh", ax=ax, color="#0e7490")
    ax.set_xlabel("Mutual information score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)
    st.write(f"The three highest mutual-information predictors are {', '.join(mi_sorted.head(3).index)} with scores {', '.join(f'{v:.3f}' for v in mi_sorted.head(3).values)}.")

    if not rf_importance.empty:
        rf_sorted = rf_importance.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        rf_sorted.plot(kind="barh", ax=ax, color="#7c3aed")
        ax.set_xlabel("Random Forest importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
        top_rf = rf_importance.sort_values(ascending=False).head(3)
        st.write(f"Random Forest assigns the largest importance to {', '.join(top_rf.index)} with normalized values {', '.join(f'{v:.3f}' for v in top_rf.values)}.")

with tabs[3]:
    cm_model = st.selectbox("Model", list(fitted.keys()), key="cm_select")
    if cm_model in fitted:
        te_pred = fitted[cm_model].predict(X_te)
        cm = confusion_matrix(y_te, te_pred, labels=[0, 1, 2])
        labels = ["Low", "Medium", "High"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels)
        ax.set_yticks(range(3))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        st.pyplot(fig)
        plt.close(fig)

        acc = accuracy_score(y_te, te_pred)
        high_total = int((y_te == 2).sum())
        high_correct = int(cm[2, 2]) if high_total else 0
        st.write(f"{cm_model} achieves test accuracy {acc:.3f} and correctly identifies {high_correct} of {high_total} actual High Priority barangays.")

with tabs[4]:
    registry = df[["City Name", "Barangay Name", "Priority_Level", "Predicted_Priority", "Poverty_Rate (%)", "Access_to_Water (%)", "Internet_Access (%)", "Road_Access", "Disaster_Risk_Level", "Primary_Gap"]].copy()
    registry = registry.rename(columns={
        "Priority_Level": "Actual Priority",
        "Predicted_Priority": "Predicted Priority",
        "Poverty_Rate (%)": "Poverty Rate",
        "Access_to_Water (%)": "Water Access",
        "Internet_Access (%)": "Internet Access",
        "Road_Access": "Road Access",
        "Disaster_Risk_Level": "Disaster Risk",
        "Primary_Gap": "Primary Gap",
    })

    city_filter = st.selectbox("Filter by city", ["All"] + sorted(registry["City Name"].dropna().unique().tolist()), key="city_filter")
    name_filter = st.text_input("Search barangay", key="name_filter")

    view = registry.copy()
    if city_filter != "All":
        view = view[view["City Name"] == city_filter]
    if name_filter:
        view = view[view["Barangay Name"].str.contains(name_filter, case=False, na=False)]

    priority_rank = {"High Priority": 2, "Medium Priority": 1, "Low Priority": 0}
    view = view.assign(_rank=view["Actual Priority"].map(priority_rank)).sort_values(["_rank", "City Name", "Barangay Name"], ascending=[False, True, True]).drop(columns="_rank")

    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button("Download registry as CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="barangay_priority_registry.csv", mime="text/csv")

    st.write(f"{len(view)} barangays match the current filter; {int((view['Actual Priority']=='High Priority').sum())} are High Priority.")

with tabs[5]:
    hp = df[df["Predicted_Priority"] == "High Priority"].copy()

    if hp.empty:
        st.write("No barangays are currently predicted as High Priority under the selected feature set.")
    else:
        gap_city = hp.groupby(["City Name", "Primary_Gap"]).size().reset_index(name="Barangays")
        gap_city = gap_city.sort_values(["City Name", "Barangays"], ascending=[True, False])
        st.dataframe(gap_city, use_container_width=True, hide_index=True)

        top_gaps = hp["Primary_Gap"].value_counts()
        fig, ax = plt.subplots(figsize=(9, 3.4))
        top_gaps.sort_values().plot(kind="barh", ax=ax, color="#dc2626")
        ax.set_xlabel("High-priority barangays")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        top_gap_name = top_gaps.index[0]
        top_gap_count = int(top_gaps.iloc[0])
        st.write(f"Across the {len(hp)} barangays predicted High Priority, {top_gap_name.lower()} is the most frequent primary service gap, affecting {top_gap_count} barangays.")

        detail = hp[["City Name", "Barangay Name", "Primary_Gap", "Poverty_Rate (%)", "Access_to_Water (%)", "Internet_Access (%)", "Road_Access", "Disaster_Risk_Level"]].copy()
        detail = detail.rename(columns={
            "Poverty_Rate (%)": "Poverty Rate",
            "Access_to_Water (%)": "Water Access",
            "Internet_Access (%)": "Internet Access",
            "Road_Access": "Road Access",
            "Disaster_Risk_Level": "Disaster Risk",
            "Primary_Gap": "Primary Gap",
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

        st.download_button("Download recommendations as CSV", data=detail.to_csv(index=False).encode("utf-8"), file_name="barangay_policy_recommendations.csv", mime="text/csv")

with tabs[6]:
    log_df = pd.DataFrame(log, columns=["Step", "Rows affected"])
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    st.write(f"Cleaning retained {len(df)} of {len(raw)} original rows across {log_df['Step'].nunique()} distinct preparation steps.")
