# 🧠 Autism Trait Screening Tool (ML + AQ-Based)

A machine learning–powered autism trait screening application that combines a deterministic AQ (Autism Quotient) scoring system with a trained ML model to provide a balanced, interpretable, and responsible assessment of autism-related traits.

⚠️ Disclaimer: This project is intended only for educational and screening purposes. It is not a medical diagnostic tool.

## 🚀 Project Overview

This application allows users to:

  Answer 10 AQ-style behavioral questions (A1–A10)

  Provide limited background information

Receive:

A rule-based AQ risk score (transparent & explainable)

A machine learning–based probability estimate (data-driven)

The goal is to mimic how real-world medical screening tools work:

combining deterministic questionnaires with probabilistic ML confidence.

## 🧩 Why Two Outputs?

During development, a key insight emerged:

Machine learning models should not replace validated screening logic — they should complement it.

So the system provides:

AQ-based Risk (Deterministic)

Based purely on the AQ questionnaire logic

Easy to understand and clinically interpretable

ML Probability (Learned)

Trained on historical data

Conservative by design

Reflects uncertainty and data limitations

This avoids overconfident predictions while maintaining logical consistency.

## 📊 Dataset Description

The dataset used is a public autism screening dataset (commonly available on Kaggle), containing:

AQ screening question responses (A1–A10)

Demographic & background information

A binary label indicating autism traits (self-reported)

## ⚠️ Important Dataset Limitations

Labels are self-reported, not clinical diagnoses

Autism is underreported in certain populations

Some features reflect social or reporting bias, not biological reality

These limitations heavily influenced design decisions in this project.

## 🛠️ Feature Engineering Decisions
✅ Retained Features

A1–A10 AQ scores (primary behavioral signal)

Family history of autism (genetic relevance)

❌ Removed Features

Ethnicity
Removed after discovering it introduced systematic bias
(e.g., autism underreporting in certain regions)

Derived “result” score
Found to be inconsistent and noisy
Removing it improved cross-validation accuracy

## 🧪 Sanity Testing (Non-Negotiable Step)

To validate logical behavior (not just accuracy), manual sanity tests were created:

Test Case	Expected Behavior	Result
Strong autistic traits	High probability (>75%)	✅ Passed
Strong non-autistic traits	Very low probability (<25%)	✅ Passed
Mixed traits	Moderate probability (30–60%)	✅ Passed

These tests helped uncover:

Feature mismatch bugs

Reverse-scoring errors

Bias introduced by non-behavioral features

## ⚙️ Model Details

Model: Random Forest Classifier

Class imbalance: Handled using SMOTE

Evaluation: Cross-validation (not single split)

Final behavior: Conservative, bias-aware, interpretable

Accuracy alone was not trusted — logical consistency was prioritized.

## 🖥️ Application Interface

Built using Streamlit, the app:

Collects AQ responses using user-friendly Yes/No inputs

Asks for age & gender (for realism only — not used in prediction)

Displays:

AQ Score + interpretation

ML probability + confidence level

A combined, human-readable conclusion

## 🧠 Key Challenges Faced
1. Logical vs Statistical Correctness

High accuracy did not mean logical predictions

Solved using deterministic + ML hybrid approach

2. Reverse-Scored AQ Questions

Some AQ questions are negatively phrased

Incorrect handling caused misleading outputs

Fixed via explicit reverse scoring

3. Bias in Demographic Features

Ethnicity introduced unfair suppression of predictions

Identified and removed after sanity testing

4. Feature Mismatch Bugs

Inconsistent column names between training & inference

Solved by strict feature alignment and sanity checks

## 📌 Technologies Used

Python

Pandas

Scikit-learn

Streamlit

SMOTE (imbalanced-learn)

## 📈 Future Improvements

Probability calibration (e.g., isotonic calibration)

Explainability using SHAP

Larger, clinically validated datasets

Deployment with clinician-facing disclaimers

Longitudinal tracking instead of one-time screening

## 🧠 Final Note

This project focuses on responsible ML:

Transparent logic

Bias awareness

Conservative predictions

Ethical UX design

It reflects how real-world ML systems should be built, especially in sensitive domains like healthcare.
