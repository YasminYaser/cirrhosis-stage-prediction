"""
=====================================================================
 CIRRHOSIS STAGE PREDICTION - STREAMLIT APP
=====================================================================
Run locally with:
    streamlit run app.py

Requires the artifacts produced by train_deployment_model.py:
    cirrhosis_model.pkl, scaler.pkl, feature_columns.pkl,
    numeric_features.pkl, metrics.json
=====================================================================
"""

import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Cirrhosis Stage Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Styling - soft, muted palette consistent with the notebook charts
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: #FAF7F5; }
        h1, h2, h3 { color: #6B4F53; }
        .result-card {
            padding: 1.6rem; border-radius: 14px; text-align: center;
            border: 1px solid #E5D6D0; background: #FFFFFF;
            box-shadow: 0 2px 10px rgba(160,130,120,0.10);
        }
        .result-stage { font-size: 3.2rem; font-weight: 700; margin: 0; }
        .result-label { font-size: 1.05rem; color: #8C6A5D; margin-top: .3rem; }
        .metric-box {
            background: #FFFFFF; border: 1px solid #EFE2DC;
            border-radius: 10px; padding: .9rem 1.1rem;
        }
        .disclaimer {
            background: #FDF3EE; border-left: 4px solid #C97B84;
            padding: .9rem 1.1rem; border-radius: 6px;
            font-size: .9rem; color: #6B4F53;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

STAGE_COLORS = {1: "#A8C0A0", 2: "#D4A574", 3: "#C97B84", 4: "#8C5A63"}
STAGE_TEXT = {
    1: "Early fibrosis - minimal scarring of the liver.",
    2: "Moderate fibrosis - scarring is spreading but still limited.",
    3: "Severe fibrosis - extensive scarring, approaching cirrhosis.",
    4: "Cirrhosis - advanced, widespread scarring of the liver.",
}


# ---------------------------------------------------------------------
# Load artifacts (cached so they load once, not on every interaction)
# ---------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("cirrhosis_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    numeric_features = joblib.load("numeric_features.pkl")
    with open("metrics.json") as f:
        metrics = json.load(f)
    return model, scaler, feature_columns, numeric_features, metrics


try:
    model, scaler, feature_columns, numeric_features, metrics = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train_deployment_model.py` first "
        "to generate the model artifacts, then reload this page."
    )
    st.stop()


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("🩺 Cirrhosis Stage Prediction")
st.markdown(
    "Predicts the **fibrosis stage (1-4)** of a liver-disease patient from "
    "clinical signs and lab results, using an XGBoost model trained on the "
    "Mayo Clinic primary biliary cirrhosis trial dataset."
)

st.markdown(
    '<div class="disclaimer"><b>Academic project - not a medical device.</b> '
    'This tool is a graduation project built for educational purposes. It must '
    'not be used to diagnose, treat, or make any decision about a real patient. '
    'Always consult a qualified clinician.</div>',
    unsafe_allow_html=True,
)
st.write("")


# ---------------------------------------------------------------------
# Sidebar - patient inputs
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Patient Data")
    st.caption("Enter the patient's information below, then press Predict.")

    st.subheader("Demographics")
    age_years = st.slider("Age (years)", 20, 90, 50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    drug = st.selectbox("Treatment", ["D-penicillamine", "Placebo"])

    st.subheader("Clinical Signs")
    ascites = st.selectbox("Ascites (fluid in abdomen)", ["No", "Yes"])
    hepatomegaly = st.selectbox("Hepatomegaly (enlarged liver)", ["No", "Yes"])
    spiders = st.selectbox("Spider angiomas (skin lesions)", ["No", "Yes"])
    edema = st.selectbox(
        "Edema (swelling)",
        ["None", "Present, responds to treatment", "Present, no response to treatment"],
    )

    st.subheader("Laboratory Results")
    bilirubin = st.number_input("Bilirubin (mg/dl)", 0.1, 30.0, 1.4, 0.1)
    albumin = st.number_input("Albumin (g/dl)", 1.5, 5.0, 3.53, 0.01)
    prothrombin = st.number_input("Prothrombin time (s)", 8.0, 20.0, 10.6, 0.1)
    copper = st.number_input("Copper (ug/day)", 0.0, 700.0, 73.0, 1.0)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 100.0, 1800.0, 309.5, 1.0)
    alk_phos = st.number_input("Alkaline Phosphatase (U/l)", 200.0, 14000.0, 1259.0, 10.0)
    sgot = st.number_input("SGOT / AST (U/ml)", 20.0, 500.0, 114.7, 1.0)
    trig = st.number_input("Triglycerides (mg/dl)", 20.0, 700.0, 108.0, 1.0)
    platelets = st.number_input("Platelets (per 1000/ml)", 50.0, 800.0, 251.0, 1.0)

    predict_clicked = st.button("Predict Stage", type="primary", use_container_width=True)


# ---------------------------------------------------------------------
# Build the model input row
# ---------------------------------------------------------------------
def build_input_row():
    """Assemble a single-row DataFrame matching the training schema exactly.

    The encoding here MUST mirror train_deployment_model.py exactly:
    same binary maps, same ordinal order for Edema, same engineered
    features. Any mismatch silently produces wrong predictions.
    """
    ascites_v = 1 if ascites == "Yes" else 0
    hepatomegaly_v = 1 if hepatomegaly == "Yes" else 0
    spiders_v = 1 if spiders == "Yes" else 0
    edema_v = {
        "None": 0,
        "Present, responds to treatment": 1,
        "Present, no response to treatment": 2,
    }[edema]

    row = {
        "Age": age_years * 365.25,          # model was trained on age in days
        "Sex": 1 if sex == "Male" else 0,
        "Ascites": ascites_v,
        "Hepatomegaly": hepatomegaly_v,
        "Spiders": spiders_v,
        "Edema": edema_v,
        "Bilirubin": bilirubin,
        "Cholesterol": cholesterol,
        "Albumin": albumin,
        "Copper": copper,
        "Alk_Phos": alk_phos,
        "SGOT": sgot,
        "Tryglicerides": trig,
        "Platelets": platelets,
        "Prothrombin": prothrombin,
        "Age_Years": age_years,
        # Engineered features - recreated with the same rules as training.
        # 3.4 is the dataset's 75th-percentile Bilirubin used at training time.
        "High_Severity": int(bilirubin > 3.4 and ascites_v == 1),
        "Signs_Count": ascites_v + hepatomegaly_v + spiders_v + (1 if edema_v > 0 else 0),
        "Low_Albumin": int(albumin < 2.8),
        "Drug_Placebo": 1 if drug == "Placebo" else 0,
    }

    X_row = pd.DataFrame([row])
    # Reindex to the exact training column order; fill any missing dummy with 0.
    X_row = X_row.reindex(columns=feature_columns, fill_value=0)
    X_row[numeric_features] = scaler.transform(X_row[numeric_features])
    return X_row


# ---------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------
tab_predict, tab_model, tab_about = st.tabs(["Prediction", "Model Performance", "About the Data"])

with tab_predict:
    if predict_clicked:
        X_row = build_input_row()
        pred = int(model.predict(X_row)[0]) + 1          # shift 0-3 back to 1-4
        proba = model.predict_proba(X_row)[0]

        col_a, col_b = st.columns([1, 1.4])

        with col_a:
            st.markdown(
                f'<div class="result-card">'
                f'<p class="result-stage" style="color:{STAGE_COLORS[pred]}">Stage {pred}</p>'
                f'<p class="result-label">{STAGE_TEXT[pred]}</p>'
                f'<p class="result-label">Model confidence: <b>{proba[pred-1]*100:.1f}%</b></p>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Confidence is the model's probability for the predicted class. "
                "Because adjacent stages overlap clinically, values below ~60% "
                "are common and expected."
            )

        with col_b:
            fig = go.Figure(
                go.Bar(
                    x=[f"Stage {i}" for i in range(1, 5)],
                    y=proba * 100,
                    marker_color=[STAGE_COLORS[i] for i in range(1, 5)],
                    text=[f"{p*100:.1f}%" for p in proba],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Probability across all stages",
                yaxis_title="Probability (%)",
                yaxis_range=[0, max(proba * 100) * 1.25],
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                height=340,
                margin=dict(t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Patient summary")
        signs_count = int(X_row["Signs_Count"].iloc[0]) if "Signs_Count" in X_row else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bilirubin", f"{bilirubin:.1f} mg/dl", "high" if bilirubin > 2 else "normal")
        c2.metric("Albumin", f"{albumin:.2f} g/dl", "low" if albumin < 2.8 else "normal")
        c3.metric("Prothrombin", f"{prothrombin:.1f} s", "prolonged" if prothrombin > 11 else "normal")
        c4.metric("Clinical signs present", f"{signs_count} / 4")
    else:
        st.info("Fill in the patient data in the sidebar, then press **Predict Stage**.")

with tab_model:
    st.subheader("How well does this model actually perform?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
    c2.metric("Majority-class baseline", f"{metrics['baseline_accuracy']*100:.1f}%")
    c3.metric("F1-macro", f"{metrics['f1_macro']:.3f}")

    st.markdown(
        f"""
The model correctly predicts the stage about **{metrics['accuracy']*100:.0f}%** of the time,
compared with **{metrics['baseline_accuracy']*100:.0f}%** for a trivial model that always guesses
the most common stage. The gap is the model's real added value.

**Why the accuracy is modest, and why that is expected here:**

- **Small dataset** - roughly 400 patients in total ({metrics['n_train']} used for training,
  {metrics['n_test']} held out for testing). Stage 1 has only 21 records in the entire dataset.
- **Class imbalance** - Stages 3 and 4 dominate the data while Stage 1 is rare, so the model
  sees very few examples of the early stage.
- **Clinically overlapping stages** - adjacent stages (Stage 2 vs Stage 3, for example) often
  share very similar lab values. The boundary between them is genuinely blurred, for clinicians
  as well as for the model.
- **Published results on this same dataset** report comparable accuracy ranges for 4-class stage
  prediction, so this outcome matches what is generally achievable with this data.

**Selected hyperparameters** (tuned with GridSearchCV, 5-fold cross-validation, scoring F1-macro):
"""
    )
    st.json(metrics["best_params"])

    st.divider()
    st.subheader("A deliberate design decision: avoiding data leakage")
    st.markdown(
        """
The research notebook explored every column in the dataset, including `Status`
(whether the patient died or received a transplant) and `N_Days` (length of follow-up).
Those columns are **deliberately excluded from this deployed model**.

The reason: for a new patient being assessed today, neither value exists yet. A model
relying on them would look accurate in testing but would be unusable in practice — a
classic case of data leakage. This app is trained only on information a clinician
genuinely has at the time of assessment: demographics, physical signs, and lab results.
"""
    )

with tab_about:
    st.subheader("About the dataset")
    st.markdown(
        """
The data comes from a **Mayo Clinic clinical trial** on primary biliary cirrhosis,
conducted between 1974 and 1984. It contains **418 patients**, each described by
demographics, physical examination findings, and laboratory measurements.

**Target variable — `Stage`:** the fibrosis stage of the liver, graded 1 to 4, where
1 is early fibrosis and 4 is full cirrhosis.
"""
    )

    st.markdown("**Features used by this model**")
    feature_info = pd.DataFrame(
        [
            ("Age", "Patient age", "Cirrhosis risk and severity increase with age."),
            ("Sex", "Male / Female", "Primary biliary cirrhosis is far more common in women."),
            ("Drug", "D-penicillamine or Placebo", "Treatment arm assigned in the original trial."),
            ("Ascites", "Fluid accumulation in the abdomen", "A hallmark sign of advanced liver disease."),
            ("Hepatomegaly", "Enlarged liver", "Common physical finding in liver disease."),
            ("Spiders", "Spider angiomas on the skin", "Small vascular skin lesions linked to liver dysfunction."),
            ("Edema", "Swelling from fluid retention", "Graded by whether it responds to treatment."),
            ("Bilirubin", "Bile pigment in blood", "Rises when the liver cannot clear it; causes jaundice."),
            ("Albumin", "Protein made by the liver", "Falls as the liver's synthetic function declines."),
            ("Prothrombin", "Blood clotting time", "Lengthens as the liver makes fewer clotting factors."),
            ("Copper", "Urine copper", "Accumulates when biliary excretion is impaired."),
            ("Cholesterol", "Blood cholesterol", "Liver disease alters lipid metabolism."),
            ("Alk_Phos", "Alkaline phosphatase", "Liver enzyme, elevated in biliary obstruction."),
            ("SGOT", "AST liver enzyme", "Released when liver cells are damaged."),
            ("Tryglicerides", "Blood triglycerides", "Another lipid affected by liver function."),
            ("Platelets", "Platelet count", "Declines as portal hypertension develops."),
        ],
        columns=["Feature", "What it is", "Why it matters clinically"],
    )
    st.dataframe(feature_info, use_container_width=True, hide_index=True)

    st.markdown("**Engineered features** (created from the raw columns to help the model)")
    st.markdown(
        """
- **`Age_Years`** — age converted from days to years for readability.
- **`Signs_Count`** — how many of the four clinical signs are present (0–4); a simple severity score.
- **`Low_Albumin`** — flag for albumin below 2.8 g/dl, a recognized clinical threshold.
- **`High_Severity`** — flag combining high bilirubin with the presence of ascites, two markers
  that together indicate advanced disease.
"""
    )
