
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = "kidney_cicd_model.pkl"


def get_sample_input():

    data = pd.read_csv("kidney_disease.csv")

    data.columns = data.columns.str.strip().str.lower()
    data = data.loc[:, ~data.columns.str.contains("^unnamed")]
    data = data.drop(columns=["id"], errors="ignore")

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].str.strip().str.lower()

    numeric_columns = [
        "age", "bp", "sg", "al", "su", "bgr",
        "bu", "sc", "sod", "pot", "hemo",
        "pcv", "wc", "rc"
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["classification"])

    X = data.drop(columns=["classification"])

    return X.iloc[[0]]


def test_model_file_exists():
    assert Path(MODEL_PATH).exists()


def test_model_prediction():

    model = joblib.load(MODEL_PATH)

    sample = get_sample_input()

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in model.classes_
