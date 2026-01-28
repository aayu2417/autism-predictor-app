import streamlit as st
import pandas as pd
import pickle

# --------------------------------------------------
# Load model and encoders
# --------------------------------------------------
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# --------------------------------------------------
# Helper function for AQ-10 scoring
# --------------------------------------------------
def score_answer(user_answer, autistic_answer):
    return 1 if user_answer == autistic_answer else 0


st.set_page_config(page_title="Autism Screening Tool", layout="centered")
st.title("🧠 Autism Screening Tool")
st.caption("This is a screening tool, not a medical diagnosis.")

st.markdown("---")

# --------------------------------------------------
# AQ-10 QUESTIONS
# --------------------------------------------------
A1 = score_answer(
    st.selectbox("I often notice small sounds when others do not.", ["No", "Yes"]),
    "Yes"
)

A2 = score_answer(
    st.selectbox("I usually concentrate more on the whole picture rather than small details.", ["No", "Yes"]),
    "No"
)

A3 = score_answer(
    st.selectbox("I find it easy to do more than one thing at once.", ["No", "Yes"]),
    "No"
)

A4 = score_answer(
    st.selectbox("If there is an interruption, I can switch back very quickly.", ["No", "Yes"]),
    "No"
)

A5 = score_answer(
    st.selectbox("I find it easy to read between the lines when someone is talking.", ["No", "Yes"]),
    "No"
)

A6 = score_answer(
    st.selectbox("I know how to tell if someone listening to me is getting bored.", ["No", "Yes"]),
    "No"
)

A7 = score_answer(
    st.selectbox("When reading a story, I find it difficult to work out characters’ intentions.", ["No", "Yes"]),
    "Yes"
)

A8 = score_answer(
    st.selectbox("I like to collect information about categories of things.", ["No", "Yes"]),
    "Yes"
)

A9 = score_answer(
    st.selectbox("I find it easy to work out what someone is thinking from their face.", ["No", "Yes"]),
    "No"
)

A10 = score_answer(
    st.selectbox("I find it difficult to work out people’s intentions.", ["No", "Yes"]),
    "Yes"
)

st.markdown("---")

# --------------------------------------------------
# OTHER INPUTS
# --------------------------------------------------
age = st.number_input("Age", min_value=1, max_value=100, value=18)

gender = st.selectbox("Gender", encoders["gender"].classes_)
ethnicity = st.selectbox("Ethnicity", encoders["ethnicity"].classes_)
jaundice = st.selectbox("Jaundice at birth?", encoders["jaundice"].classes_)
autism_in_family = st.selectbox("Family history of autism?", encoders["autism_in_family"].classes_)
country = st.selectbox("Country", encoders["country"].classes_)
used_app_before = st.selectbox("Used screening app before?", encoders["used_app_before"].classes_)
relation = st.selectbox("Who is filling the form?", encoders["relation"].classes_)

# --------------------------------------------------
# Encode categorical variables
# --------------------------------------------------
gender = encoders["gender"].transform([gender])[0]
ethnicity = encoders["ethnicity"].transform([ethnicity])[0]
jaundice = encoders["jaundice"].transform([jaundice])[0]
autism_in_family = encoders["autism_in_family"].transform([autism_in_family])[0]
country = encoders["country"].transform([country])[0]
used_app_before = encoders["used_app_before"].transform([used_app_before])[0]
relation = encoders["relation"].transform([relation])[0]

# --------------------------------------------------
# Build input DataFrame (ORDER MUST MATCH TRAINING)
# --------------------------------------------------
input_df = pd.DataFrame([[
    A1, A2, A3, A4, A5,
    A6, A7, A8, A9, A10,
    age, gender, ethnicity, jaundice,
    autism_in_family, country, used_app_before, relation
]], columns=[
    'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score',
    'A6_Score','A7_Score','A8_Score','A9_Score','A10_Score',
    'age','gender','ethnicity','jaundice','autism_in_family',
    'country','used_app_before','relation'
])

# --------------------------------------------------
# Prediction + Probability (label-safe)
# --------------------------------------------------
if st.button("🔍 Predict"):

    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]

    class_labels = model.classes_
    prob_map = dict(zip(class_labels, pred_proba))

    autism_prob = prob_map.get(1, 0) * 100

    st.markdown("## 📊 Screening Result")

    if autism_prob >= 75:
        st.error("🔴 **High likelihood of autism traits**")
        st.write("The model indicates a strong presence of autism-related traits.")
    elif autism_prob >= 50:
        st.warning("🟠 **Moderate likelihood of autism traits**")
        st.write("Some autism-related traits are present. Further evaluation is advised.")
    elif autism_prob >= 25:
        st.info("🟡 **Low likelihood of autism traits**")
        st.write("Few autism-related traits are present.")
    else:
        st.success("🟢 **Very low likelihood of autism traits**")
        st.write("Autism-related traits are unlikely.")

    # Probability bar
    st.markdown("### 🔢 Model Confidence")
    st.progress(int(autism_prob))
    st.write(f"**Autism probability:** {autism_prob:.2f}%")

    st.warning(
        "⚠️ This tool is for screening purposes only and does not provide a medical diagnosis."
    )
