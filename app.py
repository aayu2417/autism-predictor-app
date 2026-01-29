import streamlit as st
import pandas as pd
import pickle

# ==============================
# Load trained model
# ==============================
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# ==============================
# Configuration
# ==============================
FEATURES = [
    "A1_Score","A2_Score","A3_Score","A4_Score","A5_Score",
    "A6_Score","A7_Score","A8_Score","A9_Score","A10_Score",
    "autism_in_family"
]

# Reverse-scored AQ questions
REVERSE_QUESTIONS = [
    "A2_Score","A3_Score","A4_Score",
    "A5_Score","A6_Score","A9_Score"
]

# ==============================
# Helper functions
# ==============================
def apply_reverse_scoring(df):
    df = df.copy()
    for q in REVERSE_QUESTIONS:
        df[q] = 1 - df[q]
    return df

def aq_interpretation(score):
    if score >= 7:
        return "High AQ traits detected"
    elif score >= 4:
        return "Moderate AQ traits detected"
    else:
        return "Low AQ traits detected"

def ml_interpretation(prob):
    if prob >= 0.75:
        return "High model confidence"
    elif prob >= 0.35:
        return "Moderate model confidence"
    else:
        return "Low model confidence"

def yes_no(question):
    return st.selectbox(question, ["No", "Yes"]) == "Yes"

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Autism Screening Tool", layout="centered")
st.title("Autism Trait Screening Tool \nBased on the AQ-10 Score")
st.caption("This is a screening tool, not a medical diagnosis.")

# ------------------------------
# Basic Info (not used in model)
# ------------------------------
st.subheader("Basic Information")
age = st.number_input("Age", min_value=1, max_value=120, value=18)
gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])

# ------------------------------
# AQ Questions
# ------------------------------
st.subheader("AQ Screening Questions")

A1 = yes_no("I often notice small sounds when others do not.")
A2 = yes_no("I usually concentrate more on the whole picture rather than small details.")
A3 = yes_no("I find it easy to do more than one thing at once.")
A4 = yes_no("If there is an interruption, I can switch back very quickly.")
A5 = yes_no("I find it easy to read between the lines when someone is talking to me.")
A6 = yes_no("I know how to tell if someone listening to me is getting bored.")
A7 = yes_no("When reading a story, I find it difficult to work out characters’ intentions.")
A8 = yes_no("I like to collect information about categories of things.")
A9 = yes_no("I find it easy to work out what someone is thinking or feeling just by looking at their face.")
A10 = yes_no("I find it difficult to work out people’s intentions.")

st.subheader("Family History")
family_history = st.selectbox("Family history of autism?", ["No", "Yes"])

# ==============================
# Prediction
# ==============================
if st.button("Predict"):
    # Raw input
    raw = {
        "A1_Score": int(A1),
        "A2_Score": int(A2),
        "A3_Score": int(A3),
        "A4_Score": int(A4),
        "A5_Score": int(A5),
        "A6_Score": int(A6),
        "A7_Score": int(A7),
        "A8_Score": int(A8),
        "A9_Score": int(A9),
        "A10_Score": int(A10),
        "autism_in_family": 1 if family_history == "Yes" else 0
    }

    # ------------------------------
    # AQ deterministic score
    # ------------------------------
    aq_score = (
        raw["A1_Score"] +
        (1 - raw["A2_Score"]) +
        (1 - raw["A3_Score"]) +
        (1 - raw["A4_Score"]) +
        (1 - raw["A5_Score"]) +
        (1 - raw["A6_Score"]) +
        raw["A7_Score"] +
        raw["A8_Score"] +
        (1 - raw["A9_Score"]) +
        raw["A10_Score"]
    )

    # ------------------------------
    # ML prediction
    # ------------------------------
    df = pd.DataFrame([raw])
    df = apply_reverse_scoring(df)
    df = df[FEATURES]

    autism_prob = model.predict_proba(df)[0][1]

    # ==============================
    # Results
    # ==============================
    st.subheader("Results")

    st.markdown("### 🧮 AQ Screening Result")
    st.write(f"**AQ Score:** {aq_score} / 10")
    st.write(f"**Interpretation:** {aq_interpretation(aq_score)}")

    st.markdown("---")

    st.markdown("### 🤖 ML Model Assessment")
    st.write(f"**Autism probability:** {autism_prob*100:.2f}%")
    st.write(f"**Interpretation:** {ml_interpretation(autism_prob)}")

    st.markdown("---")

    # Combined explanation
    if aq_score >= 7 and autism_prob >= 0.6:
        st.error("🔴 High AQ traits detected with strong model confidence.")
    elif aq_score >= 7 and autism_prob < 0.6:
        st.warning("⚠️ High AQ traits detected, but model confidence is moderate.")
    elif aq_score < 4 and autism_prob < 0.3:
        st.success("✅ Low AQ traits and low model confidence.")
    else:
        st.info("🟡 Mixed indicators detected.")

    st.caption(
        "This tool is intended for educational and screening purposes only. "
        "It does not provide a medical diagnosis."
    )
