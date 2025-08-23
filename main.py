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

# ----------------------------
# 6) Train a simple baseline model
# ----------------------------
#print("\n>>> Training LogisticRegression (multinomial, saga) ...")
#clf = LogisticRegression(
#    multi_class='multinomial',
#    solver='saga',
#    penalty='l2',
#    C=1.0,
#    max_iter=300,
#    n_jobs=-1
#)
#clf.fit(X_train, y_train)

# ----------------------------
# 7) Evaluate on official test set (KDDTest+)
# ----------------------------
#print("\n=== EVALUATION ON OFFICIAL TEST SET (KDDTest+) ===")
#y_pred = clf.predict(X_test)
#print(classification_report(y_test, y_pred, digits=3, zero_division=0))

#cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
#cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
#cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.csv")
#cm_df.to_csv(cm_path)
#print(f"Confusion matrix saved to: {cm_path}")
#print("Done.")

#------------MODEL TRAINING-------------------



#---------------- Evaluation Metrics-----------------
