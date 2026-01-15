import os, sys, json, warnings
warnings.filterwarnings('ignore')
WORKSPACE = 'E:\\ai-safety\\bioworld\\workspace'
ARTIFACTS = 'E:\\ai-safety\\bioworld\\workspace\\artifacts'
os.makedirs(ARTIFACTS, exist_ok=True)
os.chdir(WORKSPACE)
def save_metrics(m):
    import json
    open(os.path.join(ARTIFACTS,'metrics.json'),'w').write(json.dumps(m, indent=2, default=str))
def append_leaderboard(row):
    import json, os
    p=os.path.join(ARTIFACTS,'leaderboard.csv')
    import csv
    exists=os.path.exists(p)
    with open(p,'a',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(row.keys()));
        if not exists: w.writeheader()
        w.writerow(row)

import pandas as pd, numpy as np, os, joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bioworld.mltasks import load_dataframe

# Load dataset
path = r"E:\ai-safety\data\debrecen\messidor_features.arff"
df = load_dataframe(path)
y = df["Class"].astype(int)
X = df.drop(columns=["Class"])
print("Loaded", X.shape, "target dist:", dict(y.value_counts()))

models = {
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    "histgb": HistGradientBoostingClassifier(max_iter=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"{name:8s} AUC {scores.mean():.3f} +/- {scores.std():.3f}")
    append_leaderboard({
        "model": name, "auc_mean": round(float(scores.mean()),4),
        "auc_std": round(float(scores.std()),4), "n_folds": 5,
        "cpu_only": True, "peak_ram_mb": "na",
    })

best = "histgb"
m = models[best].fit(X, y)
joblib.dump(m, os.path.join(ARTIFACTS, "best_model.joblib"))
save_metrics({"best": best, "n": int(len(df))})
print("done; saved best_model.joblib")
