# app.py
import streamlit as st
from predict import predict_single

st.markdown("""
### ⚠️ Disclaimer
This tool is for **screening and educational purposes only**.
It is **not a medical diagnosis**.
Please consult a qualified healthcare professional.
""")


st.set_page_config(page_title="Autism Predictor", layout="centered")
st.title("Autism Screening — Quick Predictor")

st.write("Fill the questionnaire values and click **Predict**.")

# Build UI inputs for each feature. Adjust options to match your data.
A1 = st.selectbox("A1_Score", [0, 1], index=0)
A2 = st.selectbox("A2_Score", [0, 1], index=0)
A3 = st.selectbox("A3_Score", [0, 1], index=0)
A4 = st.selectbox("A4_Score", [0, 1], index=0)
A5 = st.selectbox("A5_Score", [0, 1], index=0)
A6 = st.selectbox("A6_Score", [0, 1], index=0)
A7 = st.selectbox("A7_Score", [0, 1], index=0)
A8 = st.selectbox("A8_Score", [0, 1], index=0)
A9 = st.selectbox("A9_Score", [0, 1], index=0)
A10 = st.selectbox("A10_Score", [0, 1], index=0)

age = st.number_input("Age", min_value=1, max_value=120, value=25)
gender = st.selectbox("Gender", ["M", "F", "Other"])
ethnicity = st.text_input("Ethnicity", "Select or type...")     # better: use the exact set used in training
jaundice = st.selectbox("Jaundice", ["Yes", "No"])
autism_in_family = st.selectbox("Autism in family", ["Yes", "No"])
country = st.text_input("Country", "Type country")
used_app_before = st.selectbox("Used app before", ["Yes", "No"])
relation = st.selectbox("Relation", ["Self", "Relative", "Parent", "Others"])

if st.button("Predict"):
    input_row = {
        "A1_Score": A1, "A2_Score": A2, "A3_Score": A3, "A4_Score": A4, "A5_Score": A5,
        "A6_Score": A6, "A7_Score": A7, "A8_Score": A8, "A9_Score": A9, "A10_Score": A10,
        "age": age, "gender": gender, "ethnicity": ethnicity, "jaundice": jaundice,
        "autism_in_family": autism_in_family, "country": country,
        "used_app_before": used_app_before, "relation": relation
    }

    out = predict_single(input_row)
    st.markdown("### Prediction")
    if out["prediction"] == "Yes":
        st.error(f"Prediction: **{out['prediction']}** — likely ASD")
    else:
        st.success(f"Prediction: **{out['prediction']}** — unlikely ASD")

    if out["proba"] is not None:
        st.write("Class probabilities:", out["proba"])
    st.write("Raw model output:", out["raw_pred"])
