"""Concrete ML helper functions used by the profiling tool and as a starting
point the agent iterates from. Kept tiny and dependency-light for the i5/4GB
POC: scikit-learn, xgboost, lightgbm only. CPU-only."""
from __future__ import annotations

import json
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff


def load_dataframe(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".arff":
        data, _ = arff.loadarff(str(p))
        df = pd.DataFrame(data)
        for c in df.select_dtypes(["object"]).columns:
            df[c] = df[c].str.decode("utf-8") if df[c].dtype == object else df[c]
        # Debrecen stores the label as bytes '0'/'1'.
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip("b'")
        return df
    return pd.read_csv(p)


def profile(path: str) -> dict:
    df = load_dataframe(path)
    buf = StringIO()
    df.info(buf=buf)
    info = buf.getvalue()
    numeric = df.select_dtypes("number")
    target_candidates = [c for c in df.columns if c.lower() in {"class", "label", "target", "outcome"}]
    target = target_candidates[0] if target_candidates else df.columns[-1]
    return {
        "path": path,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missing_pct": {
            c: round(float(df[c].isna().mean() * 100), 2) for c in df.columns
        },
        "head": df.head(3).to_dict(orient="records"),
        "describe": json.loads(numeric.describe().to_json()),
        "target": target,
        "target_distribution": {
            str(k): int(v) for k, v in df[target].value_counts().items()
        },
        "info": info,
    }


def starter_train_code(dataset_path: str) -> str:
    """A minimal, working baseline script the agent can edit and re-run."""
    return f'''
import pandas as pd, numpy as np, os, joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from bioworld.mltasks import load_dataframe

df = load_dataframe(r"{dataset_path}")
y = df["Class"].astype(int)
X = df.drop(columns=["Class"])
print("Loaded", X.shape, "target dist:", dict(y.value_counts()))

models = {{
    "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "rf": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    "histgb": HistGradientBoostingClassifier(max_iter=200, random_state=42),
}}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, m in models.items():
    scores = cross_val_score(m, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    print(f"{{name:8s}} AUC {{scores.mean():.3f}} +/- {{scores.std():.3f}}")
    append_leaderboard({{
        "model": name, "auc_mean": round(float(scores.mean()),4),
        "auc_std": round(float(scores.std()),4), "n_folds": 5,
        "cpu_only": True, "peak_ram_mb": "na",
    }})

best = "histgb"
m = models[best].fit(X, y)
joblib.dump(m, os.path.join(ARTIFACTS, "best_model.joblib"))
save_metrics({{"best": best, "n": int(len(df))}})
print("done; saved best_model.joblib")
'''