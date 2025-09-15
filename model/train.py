from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
import pandas as pd
import pathlib

 
df = pd.read_csv(pathlib.Path("data/fraud-detection.csv"))

 
df = df.drop(columns=["id", "Group"])

 
y = df.pop("Target")
X = df

 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
clf = RandomForestClassifier(
    n_estimators=100,   
    max_depth=None,     
    random_state=42,
    class_weight="balanced"   
)
clf.fit(X_train, y_train)

print("Saving model...")
dump(clf, pathlib.Path("model/fraud-detection-v1.joblib"))
