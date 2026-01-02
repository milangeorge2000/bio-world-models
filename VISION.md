# BioWorld OS

**The Operating System for Executable Biomedical Intelligence**

---

## Core Principle

Everyone else stops at:

```
Question
   ↓
LLM
   ↓
Answer
```

We don't. In BioWorld, the answer is *earned by computation*:

```
Question
   ↓
Planner Agent
   ↓
Knowledge Graph
   ↓
Evidence Search
   ↓
Prediction Models
   ↓
Simulation
   ↓
Statistical Validation
   ↓
Report
```

The model never answers from vibes. It answers from an executed pipeline, and every step is visible.

---

## Example

A researcher asks:

> "Will this compound reduce liver toxicity?"

We don't reply "based on literature…" We run the pipeline live:

```
Searching evidence...
  Found:
    • 42 papers
    • 18 clinical studies
    • 3 public datasets
  ↓
Existing prediction model found.
  ↓
Running prediction...
  Confidence: 84%
  ↓
Simulation finished.
  ↓
Evidence agrees with prediction.
  ↓
Generated report.
```

Everything is visible. Every number has a source. Every step has a log.

---

## If No Model Exists

This is where the project becomes genuinely unique.

The system says honestly:

> "No validated prediction model exists for this outcome."

Then it asks:

> "Can you provide a dataset?"

**Accepted formats:** CSV, Parquet, Excel, SQL, FHIR, OMOP.

The user uploads the dataset. A new agent starts — and the user watches it work:

```
Dataset uploaded
   ↓
Schema Detection
   ↓
Missing Value Analysis
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Selection
   ↓
Hyperparameter Search
   ↓
Calibration
   ↓
Explainability
   ↓
Leaderboard
```

Nothing is hidden. This is the whole point of the platform.

---

## Live Execution

Imagine a console. Not a "submitted… done." A living transcript:

```
✓ Reading dataset
✓ Found 14,782 patients
✓ Missing values
    Age:        4%
    ALT:       18%
    Albumin:    1%
  ↓
Imputing
  ↓
Encoding
  ↓
Training XGBoost ...
Training LightGBM ...
Training Random Forest ...
  ↓
Cross Validation
  ↓
Winner: LightGBM
  AUC 0.91
```

**Nothing is hidden.** The researcher watches every decision the system makes, and can intervene at any point.

---

## Agents

### Dataset Agent
Reads **CSV, Excel, SQL, images, DICOM, PDFs**. Automatically understands the schema before anyone configures anything.

### Data Quality Agent
Checks for **duplicates, missing values, outliers, data leakage, target leakage, bias, class imbalance** — and reports them as the first thing the user sees.

### Feature Engineering Agent
Creates **interaction features, time features, clinical scores, normalized units**.

### Model Selection Agent
Picks the right task — **classification, survival analysis, regression, segmentation, forecasting, graph learning** — based on the problem, not a default.

### Explainability Agent
Runs **SHAP, feature importance, counterfactual explanations, confidence intervals**. Every prediction explains itself.

### Validation Agent
Runs **bootstrapping, cross-validation, external validation (when available), calibration curves**. We don't report a score; we report a score with a warrant.

### Simulation Agent
After training, runs "what-if":

```
Age +5 years      → Risk 38%
ALT doubles       → Risk 61%
Add Drug A        → Risk 44%
```

Each simulation carries a confidence interval and links back to the evidence that supports it.

---

## World Model

A trained model is not a silo. It becomes a node in a living graph:

```
Liver Toxicity Model
   ↓ trained on
Dataset
   ↓ predicts
Outcome
   ↓ linked to
Evidence, Clinical Trials, Papers
```

Every prediction is explainable, because every prediction is *reachable* — from model → dataset → evidence → paper.

### Continuous Learning
Later, someone uploads another dataset. The platform doesn't silently retrain. It asks:

> Retrain · Fine-tune · Create Ensemble · Compare

**Nothing is overwritten. Everything is versioned.**

---

## Beyond Tables

### Computer Vision
The pipeline adapts when the dataset is images:

```
Images
   ↓
Segmentation
   ↓
Feature Extraction
   ↓
CNN / ViT
   ↓
Embeddings
   ↓
Fusion
   ↓
Tabular Data
   ↓
Prediction
```

### Time-Series
ECG, vitals, wearables, ICU monitoring:

```
Signals → Transformer → Prediction → Simulation
```

### Knowledge Graph
Every concept is connected — **Disease, Drug, Gene, Pathway, Prediction Model, Dataset, Evidence, Simulation** — and the graph itself is *executable*. Nodes aren't decoration; they're the source code of the answer.

---

## User Interface

Four synchronized panels:

- **Knowledge Graph** — interactive biomedical relationships.
- **Agent Activity** — every agent logs its reasoning, tool use, and outputs.
- **Training Dashboard** — data profiling, model training, metrics, explainability.
- **Simulation Workspace** — interactive what-if analysis with confidence intervals and evidence links.

The researcher watches the graph update while models train and simulations run — live, in one screen.

---

## Plugin Ecosystem

Every capability is a plugin:

```
Prediction Plugin · Survival Plugin · Image Plugin · Genome Plugin
Drug Discovery Plugin · Pathology Plugin · Simulation Plugin
Knowledge Graph Plugin
```

Anyone can contribute a plugin. The platform is built to grow.

---

## Example Workflow

A researcher uploads the **HANCOCK head-and-neck cancer dataset** — 763 patients, each with **histopathology whole-slide images**, **blood labs**, **structured clinical data**, **pathology reports** — and asks:

> "Predict 5-year survival."

The system:

1. Profiles all datasets.
2. Detects image and tabular modalities.
3. Builds a multimodal training pipeline (histology embeddings + clinical + blood).
4. Trains several candidate models.
5. Evaluates and compares them.
6. Explains predictions with SHAP and attention maps.
7. Links influential features to biomedical pathways and literature.
8. Runs what-if simulations (e.g., biomarker changes).
9. Saves the trained model, lineage, metrics, and evidence into the world model.

---

# The POC That Runs on a Laptop: HANCOCK

**The hard constraint, stated plainly:** we don't have a GPU. We have an i5 with 4 GB RAM. And that's fine — because **we picked a use case where a laptop is the right machine, not a handicap.**

HANCOCK is a real, public head-and-neck cancer dataset: **763 patients**, each with histopathology whole-slide images, blood labs, structured clinical data, and pathology reports. It was built specifically for **multimodal outcome prediction** — exactly BioWorld's core competence — and its size is a *feature*: a single academic center, uniformly treated. Small-n is the scientific norm here, not a limitation we're hiding.

This POC proves BioWorld runs **wherever a researcher actually works**.

The rule of this POC:

> **Shrink the data. Never shrink the tech.**

### What stays 100% intact in the POC

- All agents — Dataset, Data Quality, Feature Engineering, Model Selection, Explainability, Validation, Simulation.
- The knowledge graph and the world model.
- Full visibility — agent logs, training dashboard, live console.
- Explainability (SHAP), calibration, confidence intervals, bootstrap CIs.
- Versioning and continuous learning (retrain / fine-tune / ensemble / compare).
- The "If No Model Exists" flow — upload → AutoML → leaderboard.
- The four-panel UI and the plugin architecture.

The only thing that changes is **the size of the inputs**.

### Why HANCOCK fits a laptop (the honest CV story)

Computer vision is in the pipeline — but in the form a real multimodal lab actually uses:

- **The heavy lifting is pre-computed.** HANCOCK ships **pre-extracted UNI histology embeddings** of the whole-slide images (from a pathology foundation model). The expensive CV step happened once, upstream, by the dataset authors. BioWorld consumes the embeddings — a 512–2048-dim vector per patient — and fuses them with tabular data. No GPU needed, no ViT training, and this is *the* standard practice in multimodal pathology.
- **We still do real CV where it's cheap:** tissue-microarray cell-density measurements (CD3/CD8 counts — image-derived features) and, if it fits in 4 GB, a tiny CNN on downscaled TMA cores. The Model Selection Agent decides live.
- **Task:** predict **5-year survival** and **recurrence risk** — the exact two tasks the HANCOCK challenge uses.

### The hardware budget, stated up front

4 GB RAM is the real design constraint, not the i5. Everything below is sized to fit comfortably in it:

- **n ≈ 300–400 of 763 patients.** Enough for a learning curve, not enough to fool anyone.
- **Feature vectors are tiny.** UNI embeddings + ~50 structured features (age, sex, smoking, TNM stage, grade, blood labs like CRP, hemoglobin, leukocytes). All fit in memory in a few MB.
- **Images only if they fit:** TMA cores downscaled to 128×128, loaded one batch at a time. If a CNN doesn't fit under 4 GB, the Model Selection Agent says so and falls back to embeddings-only. That fallback is a feature, not a failure.
- **CPU-first models only:** LightGBM, Random Forest, Logistic Regression, a small scikit-learn MLP, Cox proportional hazards for survival. No transformer training, no ViT, no GPU-dependent stack.

### The efficiency story becomes the whole story

On a laptop, efficiency isn't a side metric. It's the product:

```
✓ Reading dataset
✓ Found 763 patients (schema + 3 modalities)
✓ Memory-safe: sampling 350 patients for training
✓ Modality detection:
    UNI histology embeddings (pre-computed)  ✓ accepted
    TMA cell density (CD3/CD8)               ✓ accepted
    Structured clinical + blood              ✓ accepted
✓ Not enough samples for holdout → using 5-fold CV
✓ Model Selection Agent:
    ViT/Transformer → SKIPPED (CPU-only, no GPU detected)
    Tiny CNN on TMA cores → candidate (must fit in 4 GB)
    LightGBM on fused embeddings+clinical → candidate
    Cox PH (survival) → candidate
✓ Training ...
✓ Winner: LightGBM (early fusion)
  AUC 0.84 (95% CI 0.77–0.90, bootstrap) · C-index 0.71
  Peak RAM: 1.2 GB · Wall-clock: 3m 12s
  Warning: low sample size — calibration applied, external validation recommended.
```

That last line is not a bug. **It's the brand.** A system that reports its own limits — including "skipped the GPU models because there's no GPU" — is a system you can trust. That is exactly what "nothing is hidden" means in practice.

### What we actually demo (and what we prove)

1. **The full pipeline runs on a laptop.** Every agent, every panel, every step — on hardware a lab would actually have. That IS the demo.
2. **Real multimodal research on real data.** A published benchmark dataset, a clinically meaningful task (5-year survival), and true image+tabular fusion — not a toy.
3. **Hardware-aware model selection.** The system *notices* there's no GPU and 4 GB RAM, and picks models accordingly — live, in the console.
4. **Sample efficiency.** Show the learning curve live — model performance as n grows from 100 → 400. That's a demo of *efficiency* no GPU cluster can show.
5. **Cost and latency as first-class metrics.** RAM in MB, wall-clock in seconds, model size in KB. BioWorld's pitch becomes: *the same software, sized to the machine you actually have.*
6. **Honest statistics.** Wide confidence intervals, calibration curves, a printed "statistical power" warning, and a one-click **external validation** request.
7. **What-if simulation with clinical meaning.** "If this patient's CRP were normal, would 5-year survival risk change?" — grounded in SHAP on the fused model, linked to head-and-neck literature in the knowledge graph.

### The optional scale-probe (still no GPU needed)

To show the pipeline *can* absorb 20k+ patients, stream the **full 763-patient cohort** plus **synthetic extrapolation** (procedurally varied embeddings and clinical records) through a **scale dry-run**:

- Full volume ingests and profiles in batches, staying under the RAM ceiling.
- A small subset actually trains. The rest proves the throughput path exists.
- Real inference and explanation happen only on the small curated sample at demo time.

This separates two things we should never conflate: **pipeline scale** (proveable on a laptop) and **model quality** (honest only at small n).

### What we do NOT claim in the POC

- No clinical-grade accuracy.
- No validated model for head-and-neck 5-year survival.
- No GPU training, no transformer training.
- No production deployment.

We claim: **a complete, visible, executable biomedical intelligence pipeline — hardware-aware model selection, memory-safe data handling, real multimodal fusion (histology embeddings + clinical + blood), live agent logs, explainability, and simulation — running on a 4 GB laptop, on a published benchmark dataset, in minutes, with every step logged and honest confidence bounds.** That is a stronger demo than a fake 20k run would ever be.

---

## One-Sentence Summary

BioWorld OS is the platform where a biomedical question doesn't get an answer until the pipeline — agents, graph, evidence, models, simulation, validation — has *actually run*, and where a 4 GB laptop proves the whole system, by shrinking the data and never the tech.
