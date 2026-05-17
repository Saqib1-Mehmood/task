# task
Seizure Prediction
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings, os, json
warnings.filterwarnings("ignore")

from sklearn.datasets         import make_classification
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition    import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline         import Pipeline
from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                       learning_curve, validation_curve)
from sklearn.metrics          import (accuracy_score, f1_score,
                                       average_precision_score, precision_score,
                                       recall_score, classification_report,
                                       confusion_matrix)
from sklearn.utils            import resample
from imblearn.over_sampling   import SMOTE
from imblearn.under_sampling  import RandomUnderSampler

np.random.seed(42)
os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ── DATASET SIMULATION
# (Real project: load from CHB-MIT / UCI / Bonn; here we replicate their
#  published statistics so every algorithm runs end-to-end reproducibly.)
# ─────────────────────────────────────────────────────────────────────────────

def make_dataset(name, n_samples, n_features, seizure_ratio, noise, seed):
    """
    Generate a synthetic EEG-feature dataset that mirrors the class balance
    and feature dimensionality of a real epilepsy dataset.
    """
    rng = np.random.RandomState(seed)
    n_seizure   = int(n_samples * seizure_ratio)
    n_normal    = n_samples - n_seizure

    # Normal EEG: low-amplitude, higher-frequency features
    X_normal    = rng.randn(n_normal,   n_features) * 0.8
    # Seizure EEG: higher variance, shifted mean on a subset of features
    X_seizure   = rng.randn(n_seizure,  n_features) * 1.4 + rng.uniform(0.3, 0.9, n_features)
    X_seizure  += rng.randn(n_seizure,  n_features) * noise   # add noise

    X = np.vstack([X_normal, X_seizure])
    y = np.array([0]*n_normal + [1]*n_seizure)
    return X, y

print("=" * 60)
print("  GENERATING DATASETS")
print("=" * 60)

# Dataset A — UCI Epileptic Seizure Recognition analogue
#   11 500 samples, 179 time-domain features, ~20 % seizure
DATASETS = {}
DATASETS["UCI-EEG"] = make_dataset(
    "UCI-EEG", n_samples=11500, n_features=30,
    seizure_ratio=0.20, noise=0.3, seed=0)

# Dataset B — CHB-MIT analogue (heavily imbalanced, spectral features)
DATASETS["CHB-MIT"] = make_dataset(
    "CHB-MIT", n_samples=8000, n_features=25,
    seizure_ratio=0.06, noise=0.6, seed=1)

# Dataset C — Bonn University analogue (binary, balanced, waveform)
DATASETS["Bonn-EEG"] = make_dataset(
    "Bonn-EEG", n_samples=5000, n_features=20,
    seizure_ratio=0.50, noise=0.1, seed=2)

for name, (X, y) in DATASETS.items():
    ratio = y.mean() * 100
    print(f"  {name:12s}: {X.shape[0]:6d} samples | {X.shape[1]:3d} features | "
          f"seizure={ratio:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ── PREPROCESSING PIPELINES
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PREPROCESSING PIPELINES")
print("=" * 60)

def pipeline_A(X_train, X_test, k_features=15):
    """
    Pipeline A: Normalisation → Band-pass noise attenuation (robust scaling
    removes outlier artefacts) → Univariate feature selection (ANOVA F-test).

    Rationale: Clean the signal first so that feature scoring is not
    corrupted by artefact-driven variance.
    """
    # Step 1 – Min-Max normalisation to [0,1]
    scaler1 = MinMaxScaler()
    X_tr = scaler1.fit_transform(X_train)
    X_te = scaler1.transform(X_test)

    # Step 2 – Robust scaling as proxy for noise / artefact removal
    scaler2 = RobustScaler()
    X_tr = scaler2.fit_transform(X_tr)
    X_te = scaler2.transform(X_te)

    # Step 3 – Univariate feature selection
    k = min(k_features, X_tr.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_tr = selector.fit_transform(X_tr, y_train_global)
    X_te = selector.transform(X_te)
    return X_tr, X_te, f"Pipeline-A (Norm→Robust→ SelectK={k})"


def pipeline_B(X_train, X_test, n_components=0.95):
    """
    Pipeline B: Feature extraction (polynomial interactions) → Standard
    scaling → PCA dimensionality reduction.

    Rationale: Expand the feature space to capture non-linear EEG dynamics,
    then use PCA to compress while preserving 95 % variance.
    NOTE: Ordering matters — PCA on un-scaled data is dominated by
    high-variance features, which is why scaling precedes PCA here.
    """
    # Step 1 – Standard scaling (zero mean, unit variance)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Step 2 – PCA (95 % variance retained)
    pca = PCA(n_components=n_components, random_state=42)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    n_comp = pca.n_components_
    return X_tr, X_te, f"Pipeline-B (Scale→PCA, {n_comp} components)"


# Global reference for SelectKBest inside pipeline_A (set per dataset below)
y_train_global = None

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ── METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_tr, y_tr, X_te, y_te, label=""):
    y_pred     = model.predict(X_te)
    y_prob     = model.predict_proba(X_te)[:, 1]
    acc        = accuracy_score(y_te, y_pred)
    f1         = f1_score(y_te, y_pred, zero_division=0)
    pr_auc     = average_precision_score(y_te, y_prob)
    prec       = precision_score(y_te, y_pred, zero_division=0)
    rec        = recall_score(y_te, y_pred, zero_division=0)
    train_acc  = accuracy_score(y_tr, model.predict(X_tr))
    return dict(label=label, acc=acc, f1=f1, pr_auc=pr_auc,
                prec=prec, rec=rec, train_acc=train_acc)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ── BASELINE LOGISTIC REGRESSION PER DATASET × PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  BASELINE LOGISTIC REGRESSION")
print("=" * 60)

all_results = {}   # {dataset: {pipeline: metrics}}

for ds_name, (X, y) in DATASETS.items():
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    y_train_global = y_train          # for SelectKBest inside pipeline_A

    ds_results = {}
    for pipe_fn, pipe_tag in [(pipeline_A, "PipeA"), (pipeline_B, "PipeB")]:
        X_tr, X_te, desc = pipe_fn(X_tr_raw, X_te_raw)
        model = LogisticRegression(C=1.0, solver="lbfgs",
                                   max_iter=1000, random_state=42)
        model.fit(X_tr, y_train)
        m = evaluate(model, X_tr, y_train, X_te, y_test, label=desc)
        ds_results[pipe_tag] = m
        print(f"  {ds_name:12s} | {pipe_tag} | "
              f"Acc={m['acc']:.3f}  F1={m['f1']:.3f}  PR-AUC={m['pr_auc']:.3f}")

    all_results[ds_name] = ds_results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ── OVERFITTING & UNDERFITTING (UCI dataset for illustration)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  OVERFITTING & UNDERFITTING DEMONSTRATION")
print("=" * 60)

X_uci, y_uci = DATASETS["UCI-EEG"]
X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
    X_uci, y_uci, test_size=0.2, stratify=y_uci, random_state=42)

y_train_global = y_train
X_tr, X_te, _ = pipeline_A(X_tr_raw, X_te_raw)

scenarios = {
    "Overfit  (C=1e4, no reg)":  LogisticRegression(C=1e4,  penalty="l2", max_iter=2000, random_state=42),
    "Baseline (C=1.0)":          LogisticRegression(C=1.0,  penalty="l2", max_iter=1000, random_state=42),
    "Underfit (C=1e-4)":         LogisticRegression(C=1e-4, penalty="l2", max_iter=1000, random_state=42),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Learning Curves: Overfitting vs Baseline vs Underfitting",
             fontsize=13, fontweight="bold")

for ax, (label, model) in zip(axes, scenarios.items()):
    train_sizes, train_sc, val_sc = learning_curve(
        model, X_tr, y_train, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="f1", n_jobs=-1)
    tr_mean = train_sc.mean(axis=1)
    va_mean = val_sc.mean(axis=1)
    tr_std  = train_sc.std(axis=1)
    va_std  = val_sc.std(axis=1)

    ax.plot(train_sizes, tr_mean, "o-",  color="#E63946", label="Train F1")
    ax.fill_between(train_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color="#E63946")
    ax.plot(train_sizes, va_mean, "s--", color="#457B9D", label="Val F1")
    ax.fill_between(train_sizes, va_mean-va_std, va_mean+va_std, alpha=0.15, color="#457B9D")
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlabel("Training samples"); ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig1_learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig1_learning_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 ── REGULARISATION STUDY  (L1 / L2 / ElasticNet)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  REGULARISATION STUDY")
print("=" * 60)

C_values = np.logspace(-4, 4, 20)   # λ = 1/C

reg_results = {}   # {dataset: {penalty: {C: metrics}}}

for ds_name, (X, y) in DATASETS.items():
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    y_train_global = y_train
    X_tr, X_te, _ = pipeline_A(X_tr_raw, X_te_raw)

    ds_reg = {}
    for penalty, solver in [("l1", "liblinear"), ("l2", "lbfgs"),
                             ("elasticnet", "saga")]:
        scores = []
        n_zeros = []
        for C in C_values:
            kw = dict(C=C, penalty=penalty, solver=solver,
                      max_iter=2000, random_state=42)
            if penalty == "elasticnet":
                kw["l1_ratio"] = 0.5
            m = LogisticRegression(**kw)
            m.fit(X_tr, y_train)
            y_prob = m.predict_proba(X_te)[:, 1]
            f1     = f1_score(y_test, m.predict(X_te), zero_division=0)
            prauc  = average_precision_score(y_test, y_prob)
            coef   = m.coef_[0]
            zeros  = int((np.abs(coef) < 1e-4).sum())
            scores.append(dict(C=C, f1=f1, pr_auc=prauc))
            n_zeros.append(zeros)
        ds_reg[penalty] = {"scores": scores, "sparsity": n_zeros}
        best = max(scores, key=lambda x: x["f1"])
        print(f"  {ds_name:12s} | {penalty:10s} | best C={best['C']:.3e} "
              f"F1={best['f1']:.3f} PR-AUC={best['pr_auc']:.3f}")
    reg_results[ds_name] = ds_reg

# ── Regularisation plot ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Regularisation Study: F1 vs C across Datasets",
             fontsize=13, fontweight="bold")
colors = {"l1": "#E63946", "l2": "#2A9D8F", "elasticnet": "#F4A261"}

for ax, (ds_name, ds_reg) in zip(axes, reg_results.items()):
    for penalty, data in ds_reg.items():
        cs  = [s["C"]  for s in data["scores"]]
        f1s = [s["f1"] for s in data["scores"]]
        ax.semilogx(cs, f1s, "o-", color=colors[penalty],
                    label=penalty.upper(), linewidth=1.8, markersize=4)
    ax.set_title(ds_name, fontweight="bold")
    ax.set_xlabel("C  (1/λ — regularisation strength →)"); ax.set_ylabel("F1")
    ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("figures/fig2_regularisation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig2_regularisation.png")

# ── Sparsity plot (feature-selection effect of L1) ───────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
for ds_name, ds_reg in reg_results.items():
    for penalty in ["l1", "elasticnet"]:
        sp = ds_reg[penalty]["sparsity"]
        ax.semilogx(C_values, sp,
                    label=f"{ds_name} / {penalty.upper()}",
                    linewidth=1.6)
ax.set_xlabel("C  (higher C = less regularisation)"); ax.set_ylabel("Zero coefficients")
ax.set_title("Sparsity: Number of zeroed features vs C", fontweight="bold")
ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3_sparsity.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig3_sparsity.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 ── CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  CLASS IMBALANCE HANDLING")
print("=" * 60)

X_chb, y_chb = DATASETS["CHB-MIT"]
X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
    X_chb, y_chb, test_size=0.2, stratify=y_chb, random_state=42)
y_train_global = y_train
X_tr, X_te, _ = pipeline_A(X_tr_raw, X_te_raw)

imbalance_results = {}

for strategy, label in [
        ("none",           "No strategy"),
        ("smote",          "SMOTE (oversample)"),
        ("undersample",    "Random undersample"),
        ("class_weight",   "Class weight balanced"),
]:
    X_res, y_res = X_tr.copy(), y_train.copy()

    if strategy == "smote":
        X_res, y_res = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_tr, y_train)
    elif strategy == "undersample":
        X_res, y_res = RandomUnderSampler(random_state=42).fit_resample(X_tr, y_train)

    cw = "balanced" if strategy == "class_weight" else None
    model = LogisticRegression(C=1.0, solver="lbfgs",
                               class_weight=cw,
                               max_iter=2000, random_state=42)
    model.fit(X_res, y_res)
    m = evaluate(model, X_res, y_res, X_te, y_test, label=label)
    imbalance_results[strategy] = m
    print(f"  {label:30s} | Prec={m['prec']:.3f}  Rec={m['rec']:.3f}  "
          f"F1={m['f1']:.3f}  PR-AUC={m['pr_auc']:.3f}")

# ── Precision–Recall comparison bar chart ─────────────────────────────────────
strats = list(imbalance_results.keys())
precs  = [imbalance_results[s]["prec"]   for s in strats]
recs   = [imbalance_results[s]["rec"]    for s in strats]
f1s    = [imbalance_results[s]["f1"]     for s in strats]
labels = [imbalance_results[s]["label"]  for s in strats]

x = np.arange(len(strats)); w = 0.25
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - w,   precs, w, label="Precision", color="#2A9D8F")
ax.bar(x,       recs,  w, label="Recall",    color="#E63946")
ax.bar(x + w,   f1s,   w, label="F1",        color="#F4A261")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Score"); ax.set_ylim(0, 1)
ax.set_title("Impact of Imbalance Strategy on CHB-MIT (6 % seizures)",
             fontweight="bold")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig4_imbalance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig4_imbalance.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 ── PIPELINE ORDER COMPARISON (Research Insight)
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PIPELINE ORDER COMPARISON")
print("=" * 60)

def pipeline_B_wrong_order(X_train, X_test):
    """
    Wrong order: PCA BEFORE scaling — inflated by high-variance raw features.
    This is the anti-pattern we contrast against Pipeline-B.
    """
    pca = PCA(n_components=0.95, random_state=42)
    X_tr = pca.fit_transform(X_train)
    X_te = pca.transform(X_test)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, "PCA→Scale (wrong order)"

pipe_order_results = {}
for ds_name, (X, y) in DATASETS.items():
    X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    y_train_global = y_train

    row = {}
    for pipe_fn, tag in [
            (lambda a, b: pipeline_A(a, b),            "PipeA"),
            (lambda a, b: pipeline_B(a, b),            "PipeB (correct)"),
            (lambda a, b: pipeline_B_wrong_order(a, b),"PipeB (wrong order)"),
    ]:
        X_tr, X_te, desc = pipe_fn(X_tr_raw, X_te_raw)
        model = LogisticRegression(C=1.0, solver="lbfgs",
                                   max_iter=1000, random_state=42)
        model.fit(X_tr, y_train)
        m = evaluate(model, X_tr, y_train, X_te, y_test)
        row[tag] = {"f1": m["f1"], "pr_auc": m["pr_auc"]}
        print(f"  {ds_name:12s} | {tag:25s} | F1={m['f1']:.3f}  PR-AUC={m['pr_auc']:.3f}")
    pipe_order_results[ds_name] = row

# ── Pipeline order grouped bar chart ─────────────────────────────────────────
ds_names = list(pipe_order_results.keys())
pipe_tags = ["PipeA", "PipeB (correct)", "PipeB (wrong order)"]
clrs = ["#457B9D", "#2A9D8F", "#E63946"]
x = np.arange(len(ds_names)); w = 0.25

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, metric, title in zip(axes, ["f1", "pr_auc"], ["F1 Score", "PR-AUC"]):
    for i, (tag, clr) in enumerate(zip(pipe_tags, clrs)):
        vals = [pipe_order_results[ds][tag][metric] for ds in ds_names]
        ax.bar(x + (i-1)*w, vals, w, label=tag, color=clr)
    ax.set_xticks(x); ax.set_xticklabels(ds_names)
    ax.set_ylabel(title); ax.set_ylim(0, 1)
    ax.set_title(f"Pipeline Order Effect on {title}", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig5_pipeline_order.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig5_pipeline_order.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 ── CROSS-DATASET REGULARISATION STABILITY HEATMAP
# ─────────────────────────────────────────────────────────────────────────────

# Best F1 per (dataset × regulariser) at optimal C
best_f1_table = {}
for ds_name, ds_reg in reg_results.items():
    best_f1_table[ds_name] = {}
    for penalty, data in ds_reg.items():
        best_f1_table[ds_name][penalty] = max(s["f1"] for s in data["scores"])

penalties = ["l1", "l2", "elasticnet"]
matrix = np.array([[best_f1_table[ds][p] for p in penalties]
                    for ds in DATASETS])

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels(["L1", "L2", "ElasticNet"], fontsize=11)
ax.set_yticks(range(3)); ax.set_yticklabels(list(DATASETS.keys()), fontsize=11)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{matrix[i,j]:.3f}", ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="black" if matrix[i,j] < 0.7 else "white")
plt.colorbar(im, ax=ax, label="Best F1")
ax.set_title("Regularisation Generalisation Heatmap", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/fig6_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig6_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 ── IMBALANCE × REGULARISATION INTERACTION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  IMBALANCE × REGULARISATION INTERACTION")
print("=" * 60)

X_chb, y_chb = DATASETS["CHB-MIT"]
X_tr_raw, X_te_raw, y_train, y_test = train_test_split(
    X_chb, y_chb, test_size=0.2, stratify=y_chb, random_state=42)
y_train_global = y_train
X_tr, X_te, _ = pipeline_A(X_tr_raw, X_te_raw)

interaction = {}
for imb_strat, imb_label in [("none","None"), ("smote","SMOTE"), ("class_weight","CW")]:
    X_res, y_res = X_tr.copy(), y_train.copy()
    if imb_strat == "smote":
        X_res, y_res = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_tr, y_train)
    interaction[imb_label] = {}
    for penalty, solver, kw in [
            ("l1",        "liblinear", {}),
            ("l2",        "lbfgs",     {}),
            ("elasticnet","saga",      {"l1_ratio": 0.5}),
    ]:
        cw = "balanced" if imb_strat == "class_weight" else None
        model = LogisticRegression(C=1.0, penalty=penalty, solver=solver,
                                   class_weight=cw, max_iter=2000, random_state=42, **kw)
        model.fit(X_res, y_res)
        f1 = f1_score(y_test, model.predict(X_te), zero_division=0)
        interaction[imb_label][penalty] = f1
        print(f"  Imbalance={imb_label:5s} | {penalty:10s} | F1={f1:.3f}")

# Grouped bar
x = np.arange(3); w = 0.25
fig, ax = plt.subplots(figsize=(8, 4))
for i, (imb_label, reg_dict) in enumerate(interaction.items()):
    f1_vals = [reg_dict[p] for p in ["l1","l2","elasticnet"]]
    ax.bar(x + (i-1)*w, f1_vals, w, label=imb_label, color=clrs[i])
ax.set_xticks(x); ax.set_xticklabels(["L1","L2","ElasticNet"])
ax.set_ylabel("F1 Score"); ax.set_ylim(0, 1)
ax.set_title("Imbalance Handling × Regularisation (CHB-MIT)", fontweight="bold")
ax.legend(title="Imbalance Strategy"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig7_interaction.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: figures/fig7_interaction.png")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE SUMMARY JSON (used by report generator)
# ─────────────────────────────────────────────────────────────────────────────

summary = {
    "baseline":        all_results,
    "best_f1_table":   best_f1_table,
    "pipeline_order":  pipe_order_results,
    "imbalance":       {k: {kk: round(vv,4) for kk,vv in v.items()
                            if isinstance(vv, float)}
                        for k,v in imbalance_results.items()},
    "interaction":     interaction,
}

# Convert to JSON-serialisable form
def to_serialisable(obj):
    if isinstance(obj, dict):   return {k: to_serialisable(v) for k,v in obj.items()}
    if isinstance(obj, list):   return [to_serialisable(v) for v in obj]
    if isinstance(obj, (np.floating, float)): return round(float(obj), 4)
    if isinstance(obj, (np.integer, int)):    return int(obj)
    return obj

with open("figures/summary.json", "w") as f:
    json.dump(to_serialisable(summary), f, indent=2)

print("\n" + "=" * 60)
print("  ALL FIGURES + SUMMARY SAVED TO ./figures/")
print("=" * 60)
