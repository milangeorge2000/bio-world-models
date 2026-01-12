---
name: biomedical-ml
description: Structured approach to running a CPU-only biomedical ML pipeline in the BioWorld sandbox, profiling data, iterating a leaderboard, explaining with SHAP, and writing an honest report.
license: MIT
---

# Biomedical ML Skill (CPU-only laptop POC)

Use this skill whenever a biomedical prediction question is asked.

## When to Use
- A researcher asks to predict an outcome from a biomedical dataset.
- The dataset is small (≤ a few thousand rows) and the machine is CPU-only.

## Hardware contract (hard)
- CPU only, i5-class, ~4GB RAM, NO GPU. Never call a transformer/ViT. If a model
  needs a GPU, SKIP it and state why in prose.
- Single-threaded BLAS (sandbox sets this). Prefer:
  LogisticRegression, RandomForest, LightGBM, XGBoost, scikit-learn.
- 5-fold stratified CV for small n; report AUC mean +/- std, calibration, and a
  bootstrap 95% CI for the headline metric.

## Loop (do this every time)
1. `profile_dataset(path)` — confirm rows, target, imbalance, missingness.
2. `starter_script(path)` → `run_code(code)` baseline; observe stdout.
3. Iterate 2-4 model families; each run calls `append_leaderboard({...})`.
4. Pick winner; fit on full data; save `best_model.joblib` to ARTIFACTS.
5. `run_code` a SHAP explainer (TreeExplainer for trees); summarize importances.
6. `run_code` 2-3 what-if perturbations; print risk deltas + CI note.
7. `web_search` to link top features to biomedical literature.
8. `run_code` writes `report.md` to ARTIFACTS and prints the path.

## Honesty rules
- Always print peak RAM (approx) and wall-clock.
- If n < 1000, end the report with: "Low sample size — calibration applied,
  external validation recommended."
- Never claim clinical-grade accuracy in a POC.