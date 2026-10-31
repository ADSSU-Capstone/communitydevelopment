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
.dashboard-subtitle { font-size: 1.1rem; color: #475569; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.title("Barangay Priority Dashboard")
st.markdown('<div class="dashboard-subtitle">Classification-Based Techniques for Identifying Priority Barangays for Community Development</div>', unsafe_allow_html=True)

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

def clean_col_name(c):
    c = str(c).strip()
    c = re.sub(r"[^a-zA-Z0-9_]", "_", c)
    c = re.sub(r"_+", "_", c)
    return c.strip("_")

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
                raw = pd.read_excel(uploaded, sheet_name="Table 1", header=1)
            else:
                raw = pd.read_excel(uploaded, header=1)
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

# Clean column names
df.columns = [clean_col_name(c) for c in df.columns]

# Rename to standard internal names
col_map = {}
for col in df.columns:
    if "priority" in col.lower(): col_map[col] = "Priority_Level"
    elif "pop" in col.lower(): col_map[col] = "Population"
    elif "household" in col.lower() or col == "of_total_Households": col_map[col] = "Households"
    elif "city" in col.lower(): col_map[col] = "City_Name"
    elif "barangay" in col.lower() and "name" in col.lower(): col_map[col] = "Barangay_Name"
    elif "province" in col.lower(): col_map[col] = "Province_Name"
    elif "poverty" in col.lower(): col_map[col] = "Poverty_Rate"
    elif "unemployment" in col.lower(): col_map[col] = "Unemployment_Rate"
    elif "water" in col.lower(): col_map[col] = "Access_to_Water"
    elif "internet" in col.lower(): col_map[col] = "Internet_Access"
    elif "road" in col.lower(): col_map[col] = "Road_Access"
    elif "health" in col.lower(): col_map[col] = "Health_Facility_Access"
    elif "school" in col.lower(): col_map[col] = "School_Access"
    elif "disaster" in col.lower(): col_map[col] = "Disaster_Risk_Level"

df = df.rename(columns=col_map)

if "Priority_Level" not in df.columns:
    st.error("Priority_Level column not found in the dataset headers.")
    st.stop()

for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].map(strip_invisible)

# Drop completely empty rows
before = len(df)
df = df.dropna(how="all")
if before - len(df) > 0:
    log.append(("Drop completely empty rows", before - len(df)))

before = len(df)
df = df[df["Priority_Level"].notna()]
df = df[df["Priority_Level"].astype(str).str.strip() != ""]
df = df[~df["Priority_Level"].astype(str).str.contains("High Priority - Area", na=False)]
log.append(("Drop blank or descriptive Priority_Level", before - len(df)))

if "Barangay_Name" in df.columns:
    before = len(df)
    df = df[df["Barangay_Name"].notna()]
    df = df[df["Barangay_Name"].astype(str).str.strip() != ""]
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

if "Province_Name" in df.columns:
    before = len(df)
    df = df[df["Province_Name"].astype(str).str.strip() == "Agusan del Sur"]
    log.append(("Province whitelist (Agusan del Sur)", before - len(df)))

key_cols = [c for c in ["Region_Name", "Province_Name", "City_Name", "Barangay_Name", "Year"] if c in df.columns]
if key_cols:
    before = len(df)
    df = df.drop_duplicates(subset=key_cols)
    log.append(("Exact duplicates removed", before - len(df)))

num_cols = ["Population", "Households", "Poverty_Rate", "Unemployment_Rate", "Access_to_Water", "Internet_Access"]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        miss = int(df[col].isna().sum())
        if miss > 0:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
            log.append((f"Median imputation: {col}", miss))

cat_cols = ["Road_Access", "Health_Facility_Access", "School_Access", "Disaster_Risk_Level"]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace("", np.nan)
        df[col] = df[col].astype(str).str.split().str[0].replace("nan", np.nan)
        miss = int(df[col].isna().sum())
        if miss > 0:
            mode_series = df[col].mode(dropna=True)
            mode_val = mode_series.iloc[0] if not mode_series.empty else "Medium"
            df[col] = df[col].fillna(mode_val)
            log.append((f"Mode imputation: {col}", miss))

for col in ["Poverty_Rate", "Unemployment_Rate", "Access_to_Water", "Internet_Access"]:
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
FEATURES = ["Population", "Households", "Poverty_Rate", "Unemployment_Rate",
            "Access_to_Water", "Internet_Access",
            "Road_Access", "Health_Facility_Access", "School_Access", "Disaster_Risk_Level"]

valid_features = [f for f in FEATURES if f in df.columns]
X_raw = df[valid_features].copy()
for col in valid_features:
    if col in ORD:
        X_raw[col] = X_raw[col].map(ORD[col])

# Fill any remaining NaNs strictly with 0 to avoid scaler crashes
X_raw = X_raw.fillna(0.0)

TARGET_MAP = {"High Priority": 2, "Medium Priority": 1, "Low Priority": 0}
y = df["Priority_Level"].map(TARGET_MAP).values

scaler = MinMaxScaler()
X_all_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=valid_features, index=df.index)
mi_all = mutual_info_classif(X_all_scaled, y, random_state=42)

with st.sidebar:
    st.markdown("---")
    max_k = min(8, len(valid_features))
    k_features = st.slider("Retained features (k)", 3, len(valid_features), max_k)

selector = SelectKBest(mutual_info_classif, k=k_features)
X = selector.fit_transform(X_all_scaled, y)
selected_mask = selector.get_support()
selected_features = [f for f, s in zip(valid_features, selected_mask) if s]

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
if "Poverty_Rate" in df.columns: severity["Poverty"] = df["Poverty_Rate"].rank(pct=True)
if "Unemployment_Rate" in df.columns: severity["Unemployment"] = df["Unemployment_Rate"].rank(pct=True)
if "Access_to_Water" in df.columns: severity["Water"] = 1 - df["Access_to_Water"].rank(pct=True)
if "Internet_Access" in df.columns: severity["Internet"] = 1 - df["Internet_Access"].rank(pct=True)
if "Road_Access" in df.columns: severity["Road"] = df["Road_Access"].map(ORD["Road_Access"]).rank(pct=True)
if "Health_Facility_Access" in df.columns: severity["Health"] = df["Health_Facility_Access"].map(ORD["Health_Facility_Access"]).rank(pct=True)
if "School_Access" in df.columns: severity["School"] = df["School_Access"].map(ORD["School_Access"]).rank(pct=True)
if "Disaster_Risk_Level" in df.columns: severity["Disaster"] = df["Disaster_Risk_Level"].map(ORD["Disaster_Risk_Level"]).rank(pct=True)

df["Primary_Gap"] = severity.idxmax(axis=1) if not severity.empty else "Unknown"

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

    st.markdown("### Priority Distribution Analysis")
    
    colA, colB = st.columns(2)
    with colA:
        bd_options = [c for c in ["City_Name", "Disaster_Risk_Level", "Road_Access", "Health_Facility_Access", "School_Access"] if c in df.columns]
        breakdown = st.selectbox("Barangay Count Breakdown by:", bd_options, key="breakdown_select")
        
        ct = pd.crosstab(df[breakdown], df["Priority_Level"])
        order = ["High Priority", "Medium Priority", "Low Priority"]
        ct = ct.reindex(columns=[c for c in order if c in ct.columns], fill_value=0)

        fig, ax = plt.subplots(figsize=(8, 4))
        ct.plot(kind="bar", stacked=True, ax=ax, color=[palette[c] for c in ct.columns])
        ax.set_ylabel("Barangay count")
        ax.set_xlabel(breakdown)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="Priority", frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)

    with colB:
        line_opts = [c for c in ["Poverty_Rate", "Unemployment_Rate", "Access_to_Water", "Internet_Access"] if c in df.columns]
        line_var = st.selectbox("Trend Indicator (Line Graph):", line_opts, key="line_var")
        
        city_col = "City_Name"
        line_df = df.groupby([city_col, "Priority_Level"])[line_var].mean().unstack()
        line_df = line_df.reindex(columns=[c for c in order if c in line_df.columns])
        
        line_df = line_df.interpolate(method='linear', limit_direction='both')
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        line_df.plot(kind="line", marker="o", ax=ax2, color=[palette[c] for c in line_df.columns])
        ax2.set_ylabel(f"Average {line_var}")
        ax2.set_xlabel(city_col)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(title="Priority", frameon=False, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig2)
        plt.close(fig2)

    st.write(f"Across the {len(df)} evaluated barangays, {int(counts['High Priority'])} ({counts['High Priority']/max(len(df),1):.1%}) fall into the High Priority category, requiring immediate intervention. The line graph illustrates the average {line_var} across different cities, showing how specific priority levels correlate with this indicator.")

    st.markdown("### Indicator Averages by Priority Level")
    try:
        summary_df = df[valid_features].apply(pd.to_numeric, errors="coerce")
        summary_df["Priority_Level"] = df["Priority_Level"]
        summary = summary_df.groupby("Priority_Level")[valid_features].mean().round(2)
        summary = summary.reindex([c for c in order if c in summary.index])
        st.dataframe(summary, use_container_width=True)
        st.write("This table provides the average scores of all socioeconomic, infrastructure, and disaster indicators grouped by their actual priority classification, allowing for a direct comparison of living conditions between High, Medium, and Low priority barangays.")
    except Exception:
        pass

with tabs[1]:
    st.markdown("### Algorithm Performance Metrics")
    disp = results_df.copy().round(3)
    st.dataframe(disp.style.highlight_max(axis=0, color="#dbeafe"), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 3.5))
    f1_series = results_df["F1"].sort_values()
    f1_series.plot(kind="barh", ax=ax, color="#1e40af")
    ax.set_xlabel("Macro F1 Score")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    best_model = results_df["F1"].idxmax() if not results_df["F1"].isna().all() else "n/a"
    best_f1 = results_df.loc[best_model, "F1"] if best_model in results_df.index else float("nan")
    st.write(f"With {k_features} retained features, the {best_model} model achieves the highest macro F1 score of {best_f1:.3f}. While accuracy shows overall correctness, the F1 score ensures we balance the correct identification of priority barangays (precision) with not missing any truly vulnerable communities (recall).")

with tabs[2]:
    st.markdown("### Feature Importance Analysis")
    
    mi_sorted = pd.Series(mi_all, index=valid_features).sort_values(ascending=False)
    mi_top = mi_sorted.head(k_features).sort_values(ascending=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        mi_top.plot(kind="barh", ax=ax, color="#0e7490")
        ax.set_xlabel("Mutual Information Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        if not rf_importance.empty:
            rf_sorted = rf_importance.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(6, 4))
            rf_sorted.plot(kind="barh", ax=ax, color="#7c3aed")
            ax.set_xlabel("Random Forest Importance")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

    top_mi = mi_sorted.head(3)
    st.write(f"The Mutual Information filter highlights {', '.join(top_mi.index)} as the top three predictors. The Random Forest model relies heavily on these selected features, confirming that factors like {', '.join(top_mi.index[:2])} are the strongest statistical drivers for classifying a barangay's development priority level.")

with tabs[3]:
    st.markdown("### Confusion Matrix")
    cm_model = st.selectbox("Select Classifier to Audit:", list(fitted.keys()), key="cm_select")
    if cm_model in fitted:
        te_pred = fitted[cm_model].predict(X_te)
        cm = confusion_matrix(y_te, te_pred, labels=[0, 1, 2])
        labels = ["Low", "Medium", "High"]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(4, 3.5))
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
            
        with c2:
            st.write("")
            st.write("")
            acc = accuracy_score(y_te, te_pred)
            high_total = int((y_te == 2).sum())
            high_correct = int(cm[2, 2]) if high_total else 0
            st.markdown(f"**Test Accuracy:** {acc:.3f}")
            st.markdown(f"**High Priority Recall:** {high_correct} out of {high_total}")
            st.write("The confusion matrix provides a detailed audit of the model's predictions during testing. A high number in the bottom-right diagonal indicates the model successfully identifies actual High Priority barangays without falsely downgrading them to lower tiers, which is critical for ensuring vulnerable communities are not missed.")

with tabs[4]:
    st.markdown("### Barangay Priority Registry")
    reg_cols = ["City_Name", "Barangay_Name", "Priority_Level", "Predicted_Priority", "Poverty_Rate", "Access_to_Water", "Internet_Access", "Road_Access", "Disaster_Risk_Level", "Primary_Gap"]
    reg_cols = [c for c in reg_cols if c in df.columns]
    
    registry = df[reg_cols].copy()
    registry = registry.rename(columns={
        "Priority_Level": "Actual Priority",
        "Predicted_Priority": "Predicted Priority",
        "Poverty_Rate": "Poverty Rate",
        "Access_to_Water": "Water Access",
        "Internet_Access": "Internet Access",
        "Road_Access": "Road Access",
        "Disaster_Risk_Level": "Disaster Risk",
        "Primary_Gap": "Primary Gap",
        "City_Name": "City",
        "Barangay_Name": "Barangay"
    })

    colA, colB = st.columns(2)
    with colA:
        city_filter = st.selectbox("Filter by City", ["All"] + sorted(registry["City"].dropna().unique().tolist()), key="city_filter")
    with colB:
        name_filter = st.text_input("Search Barangay Name", key="name_filter")

    view = registry.copy()
    if city_filter != "All":
        view = view[view["City"] == city_filter]
    if name_filter:
        view = view[view["Barangay"].str.contains(name_filter, case=False, na=False)]

    priority_rank = {"High Priority": 2, "Medium Priority": 1, "Low Priority": 0}
    view = view.assign(_rank=view["Actual Priority"].map(priority_rank)).sort_values(["_rank", "City", "Barangay"], ascending=[False, True, True]).drop(columns="_rank")

    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button("Download Registry as CSV", data=view.to_csv(index=False).encode("utf-8"), file_name="barangay_priority_registry.csv", mime="text/csv")

    st.write(f"Currently displaying {len(view)} barangays, of which {int((view['Actual Priority']=='High Priority').sum())} are classified as High Priority.")

with tabs[5]:
    st.markdown("### Policy Recommendations & Gap Analysis")
    hp = df[df["Predicted_Priority"] == "High Priority"].copy()

    if hp.empty:
        st.write("No barangays are currently predicted as High Priority under the selected feature set.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            gap_city = hp.groupby(["City_Name", "Primary_Gap"]).size().reset_index(name="Barangays")
            gap_city = gap_city.sort_values(["City_Name", "Barangays"], ascending=[True, False])
            st.dataframe(gap_city, use_container_width=True, hide_index=True)
            
        with c2:
            top_gaps = hp["Primary_Gap"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 3))
            top_gaps.sort_values().plot(kind="barh", ax=ax, color="#dc2626")
            ax.set_xlabel("Number of Barangays")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            st.pyplot(fig)
            plt.close(fig)

        top_gap_name = top_gaps.index[0]
        top_gap_count = int(top_gaps.iloc[0])
        
        st.markdown("#### Automated Recommendations")
        st.write(f"Across the {len(hp)} barangays predicted as High Priority, **{top_gap_name.lower()}** is the most frequent primary service gap, affecting {top_gap_count} barangays. This indicates that immediate resource allocation and intervention should be directed towards addressing this specific development dimension.")
        
        top_city = gap_city.iloc[0]["City_Name"]
        top_city_gap = gap_city.iloc[0]["Primary_Gap"]
        st.write(f"The municipality of **{top_city}** requires the most urgent attention for **{top_city_gap.lower()}** initiatives, as it holds the highest concentration of High Priority barangays facing this specific gap.")

        detail = hp[["City_Name", "Barangay_Name", "Primary_Gap", "Poverty_Rate", "Access_to_Water", "Internet_Access", "Road_Access", "Disaster_Risk_Level"]].copy()
        detail = detail.rename(columns={
            "Poverty_Rate": "Poverty Rate",
            "Access_to_Water": "Water Access",
            "Internet_Access": "Internet Access",
            "Road_Access": "Road Access",
            "Disaster_Risk_Level": "Disaster Risk",
            "Primary_Gap": "Primary Gap",
            "City_Name": "City",
            "Barangay_Name": "Barangay"
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)

        st.download_button("Download Recommendations as CSV", data=detail.to_csv(index=False).encode("utf-8"), file_name="barangay_policy_recommendations.csv", mime="text/csv")

with tabs[6]:
    st.markdown("### Data Preparation Log")
    log_df = pd.DataFrame(log, columns=["Step", "Rows affected"])
    st.dataframe(log_df, use_container_width=True, hide_index=True)
    st.write(f"Data cleaning retained {len(df)} of {len(raw)} original rows across {log_df['Step'].nunique()} distinct preparation steps. Missing values were resolved using median/mode imputation, categorical text was standardized, and out-of-range percentages were clipped.")
