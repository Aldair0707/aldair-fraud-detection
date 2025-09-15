from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import pathlib

# Cargar dataset de fraude
df = pd.read_csv(pathlib.Path("data/fraud-detection.csv"))

# Eliminar columnas que no aportan al modelo
df = df.drop(columns=["id", "Group"])

# Separar variable objetivo
y = df.pop("Target")
X = df

# Dividir train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
clf = RandomForestClassifier(
    n_estimators=100,  # más árboles
    max_depth=None,    # que aprenda más patrones
    random_state=42,
    class_weight="balanced"  # para el desbalance de clases típico en fraude
)
clf.fit(X_train, y_train)

print("Saving model...")
dump(clf, pathlib.Path("model/fraud-detection-v1.joblib"))
