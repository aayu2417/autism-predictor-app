# predict.py
import pickle
import pandas as pd
import numpy as np
from typing import Dict

MODEL_PATH = "best_model.pkl"
ENCODERS_PATH = "encoders.pkl"

# Put here the exact feature order used during training (no target column)
FEATURE_COLUMNS = [
    'A1_Score','A2_Score','A3_Score','A4_Score','A5_Score','A6_Score',
    'A7_Score','A8_Score','A9_Score','A10_Score','age','gender',
    'ethnicity','jaundice','autism_in_family','country','used_app_before','result','relation'
]

def load_artifacts(model_path=MODEL_PATH, encoders_path=ENCODERS_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)    # expected: dict { column_name: LabelEncoder() }
    return model, encoders

def safe_label_transform(le, value):
    """
    Transform a single value with a LabelEncoder, handling unseen classes.
    Strategy: if unseen, map to the most frequent (first) seen class as fallback.
    (Better: train an OrdinalEncoder/OneHotEncoder inside a pipeline with handle_unknown,
    but this is a practical fallback.)
    """
    # If value is nan, replace with first class
    if pd.isna(value):
        return le.transform([le.classes_[0]])[0]
    # If value seen in training:
    if value in le.classes_:
        return le.transform([value])[0]
    # Unknown category — fallback
    return le.transform([le.classes_[0]])[0]

def preprocess_input(input_dict: Dict, encoders: dict, feature_columns=FEATURE_COLUMNS):
    """
    Build a 1-row dataframe in the correct column order and apply label encoders.
    Returns a DataFrame (1 x n_features) ready for model.predict().
    """
    # Make sure feature_columns contains all keys needed
    row = {col: input_dict.get(col, np.nan) for col in feature_columns}
    df = pd.DataFrame([row], columns=feature_columns)

    # Apply encoders for categorical columns that we have encoders for
    for col, le in encoders.items():
        if col not in df.columns:
            continue
        # do safe transform of the single value
        transformed = safe_label_transform(le, df.at[0, col])
        df.at[0, col] = transformed

    # Ensure numeric-like types are numeric
    # (convert score columns and age to numeric if they were passed as strings)
    for c in df.columns:
        # try to coerce to numeric if possible
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            # keep as-is (already encoded above if categorical)
            pass

    return df

def map_prediction_to_yes_no(model, raw_pred):
    """
    Map model output to a 'Yes'/'No' prediction string.
    Handles common cases:
      - classifier that returns 0/1
      - classifier that returns encoded string labels
    """
    # raw_pred is an array-like; get single value
    val = raw_pred[0]
    # If model has classes_ we can inspect them
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        # if classes look numeric {0,1}
        if all(isinstance(c, (int, np.integer)) for c in classes):
            # assume 1 means positive (autism), 0 negative
            try:
                return "Yes" if int(val) == 1 else "No"
            except:
                return "Yes" if val == classes[1] else "No"
        else:
            # classes are strings (e.g., 'yes'/'no', 'autism'/'no_autism')
            # assume the second class is the positive one if binary
            if len(classes) == 2:
                positive = classes[1]
                return "Yes" if val == positive else "No"
            else:
                # multi-class fallback: return raw label
                return str(val)
    else:
        # If it's numeric prediction (regressor used as classifier) fallback:
        try:
            return "Yes" if int(val) == 1 else "No"
        except:
            return str(val)

def predict_single(input_dict: Dict):
    """
    Given a dict of input features (keys should be feature names), returns prediction 'Yes' or 'No'.
    Example input_dict:
      {"A1_Score": 1, "A2_Score": 0, ..., "age": 25, "gender": "M", "ethnicity": "Asian", ...}
    """
    model, encoders = load_artifacts()
    X = preprocess_input(input_dict, encoders, FEATURE_COLUMNS)
    raw_pred = model.predict(X)       # shape (1,)
    result = map_prediction_to_yes_no(model, raw_pred)
    # We can also return probability if available
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
    return {"prediction": result, "raw_pred": raw_pred[0], "proba": prob}
