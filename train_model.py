
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


def train_model():

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

    data["classification"] = (
        data["classification"].astype("string").str.strip().str.lower()
    )

    data = data.dropna(subset=["classification"])

    X = data.drop(columns=["classification"])
    y = data["classification"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    numerical_columns = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    joblib.dump(model, "kidney_cicd_model.pkl")

    print("Model trained and saved successfully!")

    return model


if __name__ == "__main__":
    train_model()
