#Everyone write a comment to make sure can access edit this thing

import os
import numpy as np
import pandas as pd

from sklearn import __version__ as skver
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

np.random.seed(42)

# ----------------------------
# 0) Config
# ----------------------------
TRAIN_URL = "https://raw.githubusercontent.com/yjw2807/NSL-KDD-Model/main/KDDTrain+.txt"
TEST_URL  = "https://raw.githubusercontent.com/yjw2807/NSL-KDD-Model/main/KDDTest+.txt"
OUTPUT_DIR = "./nslkdd_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEAT41 = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot',
    'num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count',
    'serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate'
]

def load_nsl_kdd(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url, header=None)
    if df.shape[1] == len(FEAT41) + 2:      # 43 cols => label + difficulty
        df.columns = FEAT41 + ['labels', 'difficulty']
        df = df.drop(columns=['difficulty'])
    elif df.shape[1] == len(FEAT41) + 1:    # 42 cols => label only
        df.columns = FEAT41 + ['labels']
    else:
        raise ValueError(f"Unexpected column count: {df.shape[1]} (expected 42 or 43).")
    return df

print("scikit-learn version:", skver)
print(">>> Loading NSL-KDD train/test ...")
df_train = load_nsl_kdd(TRAIN_URL)
df_test  = load_nsl_kdd(TEST_URL)
print(f"Train shape (raw): {df_train.shape}")
print(f"Test  shape (raw): {df_test.shape}")

# ----------------------------
# 1) Quick null report (before cleaning)
# ----------------------------
def quick_null_report(df: pd.DataFrame, name: str):
    nulls = df.isna().sum().sum()
    print(f"[{name}] total NaNs: {nulls}")

quick_null_report(df_train, "TRAIN (raw)")
quick_null_report(df_test,  "TEST  (raw)")

# ----------------------------
# 2) Split features/labels
# ----------------------------
X_train_df = df_train.drop(columns=['labels'])
y_train = df_train['labels']
X_test_df  = df_test.drop(columns=['labels'])
y_test  = df_test['labels']

cat_cols = ['protocol_type', 'service', 'flag']
num_cols = [c for c in X_train_df.columns if c not in cat_cols]

# ----------------------------
# 3) Version-safe OneHotEncoder
# ----------------------------
def make_ohe():
    # Newer sklearn (>=1.4): use sparse_output
    try:
        return OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False)
    except TypeError:
        # Older sklearn (<1.4): fall back to 'sparse'
        return OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False)

ohe = make_ohe()

preprocess = ColumnTransformer(
    transformers=[
        ('cat', ohe, cat_cols),
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ]), num_cols),
    ],
    remainder='drop'
)

# ----------------------------
# 4) Fit preprocessing (TRAIN only) and transform both
# ----------------------------
print(">>> Fitting preprocessing on TRAIN and transforming TRAIN/TEST ...")
X_train = preprocess.fit_transform(X_train_df)
X_test  = preprocess.transform(X_test_df)

# Ensure dense (in case an older version returned sparse)
if hasattr(X_train, "toarray"):
    X_train = X_train.toarray()
    X_test  = X_test.toarray()

# Build cleaned DataFrame previews
try:
    cat_names = preprocess.named_transformers_['cat'].get_feature_names_out(cat_cols)
except AttributeError:
    # Very old sklearn
    cat_names = preprocess.named_transformers_['cat'].get_feature_names(cat_cols)
feat_names = np.concatenate([cat_names, np.array(num_cols)])

clean_train_df = pd.DataFrame(X_train, columns=feat_names)
clean_test_df  = pd.DataFrame(X_test,  columns=feat_names)

# ----------------------------
# 5) Show “after cleaning” results + save previews
# ----------------------------
print("\n=== AFTER DATA CLEANING (ENCODE + SCALE) ===")
print("Train (clean) shape:", clean_train_df.shape)
print("Test  (clean) shape:", clean_test_df.shape)
print("Sample columns:", list(clean_train_df.columns[:10]), "...")
print("Train (clean) head():")
print(clean_train_df.head(5))

clean_train_df.head(2000).to_csv(os.path.join(OUTPUT_DIR, "cleaned_train_preview.csv"), index=False)
clean_test_df.head(2000).to_csv(os.path.join(OUTPUT_DIR, "cleaned_test_preview.csv"), index=False)
print(f"Saved cleaned previews to: {OUTPUT_DIR}/cleaned_train_preview.csv and cleaned_test_preview.csv")


#------------MODEL TRAINING-------------------



#---------------- Evaluation Metrics-----------------
import os, time, numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
import numpy as np, math


from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    roc_curve, precision_recall_curve, log_loss
)
from scipy.stats import binomtest

# ---- Safety check ----
needed = ["model","preprocess_full","le","X_test_df","y_test_5","X_test_full","y_test_int","y_test_oh","y_train_5"]
missing = [v for v in needed if v not in globals()]
if missing:
    raise RuntimeError(f"Missing variables from previous cells: {missing}")

OUTDIR = "./eval_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ===== Predictions =====
y_proba = model.predict(X_test_full, verbose=0)      # shape: [N, C]
y_pred  = np.argmax(y_proba, axis=1)                 # int labels
y_true  = y_test_int                                 # int labels aligned with `le`
class_names = list(le.classes_)                      # e.g. ['DoS','Normal','Probe','R2L','U2R']
num_classes = len(class_names)

# ===== Metrics (multi-class) =====
acc  = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)

macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro', zero_division=0
)
micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average='micro', zero_division=0
)
per_p, per_r, per_f1, per_support = precision_recall_fscore_support(
    y_true, y_pred, average=None, labels=np.arange(num_classes), zero_division=0
)

mcc   = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

# --- Stable log loss (fixes your error) ---
eps = 1e-12
y_proba_safe = np.clip(y_proba, eps, 1 - eps)
y_proba_safe = (y_proba_safe.T / y_proba_safe.sum(axis=1)).T  # re-normalize rows
if y_test_oh.ndim == 2:
    # If we already have one-hot true labels, do NOT pass `labels=` (prevents the error)
    lloss = log_loss(y_test_oh, y_proba_safe)
else:
    # If you only had int labels, you'd use this:
    lloss = log_loss(y_true, y_proba_safe, labels=np.arange(num_classes))

# Multi-class ROC/PR AUC (OvR)
try:
    roc_auc_macro = roc_auc_score(y_test_oh, y_proba_safe, multi_class='ovr', average='macro')
    roc_auc_micro = roc_auc_score(y_test_oh, y_proba_safe, multi_class='ovr', average='micro')
except Exception:
    roc_auc_macro = np.nan
    roc_auc_micro = np.nan

try:
    pr_auc_macro  = average_precision_score(y_test_oh, y_proba_safe, average='macro')
    pr_auc_micro  = average_precision_score(y_test_oh, y_proba_safe, average='micro')
except Exception:
    pr_auc_macro = np.nan
    pr_auc_micro = np.nan

# ===== Print tables =====
print("=== Classification Report (per-class) ===")
print(classification_report(le.inverse_transform(y_true),
                            le.inverse_transform(y_pred),
                            digits=3, zero_division=0))

summary = pd.DataFrame({
    "metric": [
        "accuracy", "balanced_accuracy",
        "macro_precision", "macro_recall", "macro_f1",
        "micro_precision", "micro_recall", "micro_f1",
        "mcc", "cohen_kappa",
        "log_loss", "roc_auc_macro", "roc_auc_micro", "pr_auc_macro", "pr_auc_micro"
    ],
    "value": [
        acc, bal_acc,
        macro_p, macro_r, macro_f1,
        micro_p, micro_r, micro_f1,
        mcc, kappa,
        lloss, roc_auc_macro, roc_auc_micro, pr_auc_macro, pr_auc_micro
    ]
})
print("\n=== Summary metrics (multi-class) ===")
print(summary)

per_class_tbl = pd.DataFrame({
    "class": class_names,
    "precision": per_p,
    "recall": per_r,
    "f1": per_f1,
    "support": per_support
})
print("\n=== Per-class metrics ===")
print(per_class_tbl)

summary.to_csv(os.path.join(OUTDIR, "summary_metrics.csv"), index=False)
per_class_tbl.to_csv(os.path.join(OUTDIR, "per_class_metrics.csv"), index=False)

# ===== Plots =====

# 1) Confusion matrix (normalized by true class)
cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig = plt.figure(figsize=(6,5))
plt.imshow(cm_norm, interpolation='nearest')
plt.title("Confusion Matrix (Normalized)")
plt.colorbar()
ticks = np.arange(num_classes)
plt.xticks(ticks, class_names, rotation=45, ha='right')
plt.yticks(ticks, class_names)
thr = cm_norm.max() / 2
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, f"{cm_norm[i, j]*100:.1f}%",
                 ha='center', va='center',
                 color="white" if cm_norm[i, j] > thr else "black")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "confusion_matrix_normalized.png"), dpi=180)
plt.show()

# 2) ROC curves (OvR)
try:
    fig = plt.figure(figsize=(6,5))
    for k, cname in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_oh[:, k], y_proba_safe[:, k])
        plt.plot(fpr, tpr, label=f"{cname}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "roc_ovr.png"), dpi=180)
    plt.show()
except Exception as e:
    print("Skipping ROC curves:", e)

# 3) Precision–Recall curves (OvR)
try:
    fig = plt.figure(figsize=(6,5))
    for k, cname in enumerate(class_names):
        p, r, _ = precision_recall_curve(y_test_oh[:, k], y_proba_safe[:, k])
        plt.plot(r, p, label=f"{cname}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curves (One-vs-Rest)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "pr_ovr.png"), dpi=180)
    plt.show()
except Exception as e:
    print("Skipping PR curves:", e)

# ===== Statistical tests =====

# A) Bootstrap 95% CI for Accuracy and Macro-F1
def bootstrap_ci(y_true_vec, y_pred_vec, B=1000, seed=1, metric="accuracy"):
    rng = np.random.default_rng(seed)
    n = len(y_true_vec)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if metric == "accuracy":
            vals.append(accuracy_score(y_true_vec[idx], y_pred_vec[idx]))
        elif metric == "macro_f1":
            vals.append(precision_recall_fscore_support(
                y_true_vec[idx], y_pred_vec[idx], average='macro', zero_division=0
            )[2])
        else:
            raise ValueError("metric must be 'accuracy' or 'macro_f1'")
    vals = np.array(vals)
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

boot_acc_mean, boot_acc_lo, boot_acc_hi = bootstrap_ci(y_true, y_pred, B=1000, seed=1, metric="accuracy")
boot_f1_mean,  boot_f1_lo,  boot_f1_hi  = bootstrap_ci(y_true, y_pred, B=1000, seed=1, metric="macro_f1")

print("\n=== Bootstrap 95% CIs (1000 resamples) ===")
print(f"Accuracy: mean={boot_acc_mean:.4f}, 95% CI=({boot_acc_lo:.4f}, {boot_acc_hi:.4f})")
print(f"Macro-F1: mean={boot_f1_mean:.4f}, 95% CI=({boot_f1_lo:.4f}, {boot_f1_hi:.4f})")

# ===== McNemar test vs Majority-class baseline (exact, log-space; version-safe) =====


# Majority baseline = always predict the majority TRAIN class
majority_label_name = y_train_5.value_counts().idxmax()
majority_label_int  = le.transform([majority_label_name])[0]
baseline_pred = np.full_like(y_true, fill_value=majority_label_int)

# Discordant pairs
model_correct    = (y_pred == y_true)
baseline_correct = (baseline_pred == y_true)
b = int((~model_correct &  baseline_correct).sum())  # model wrong, baseline right
c = int(( model_correct & ~baseline_correct).sum())  # model right, baseline wrong
n = b + c

def mcnemar_exact_pvalue_logspace(b: int, c: int) -> float:
    """
    Exact two-sided McNemar p-value using Binomial(n=b+c, p=0.5),
    computed in log-space to avoid overflow/underflow.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)

    # log PMF for Binom(n, 0.5): log C(n,i) - n*log(2)
    # sum_{i=0..k} exp(log_pmf) with log-sum-exp trick
    log_vals = []
    ln2n = n * math.log(2.0)
    ln_fact_n = math.lgamma(n + 1.0)
    for i in range(k + 1):
        log_pmf = ln_fact_n - math.lgamma(i + 1.0) - math.lgamma(n - i + 1.0) - ln2n
        log_vals.append(log_pmf)

    m = max(log_vals)
    log_tail = m + math.log(sum(math.exp(v - m) for v in log_vals))
    tail = math.exp(log_tail)

    p_two_sided = min(1.0, 2.0 * tail)
    return float(p_two_sided)

pval = mcnemar_exact_pvalue_logspace(b, c)

print("\n=== McNemar Test vs Majority Baseline ===")
print(f"b (model wrong, baseline right) = {b}")
print(f"c (model right, baseline wrong) = {c}")
print(f"n = b + c                        = {n}")
print(f"Two-sided exact p-value          = {pval:.6f}")

