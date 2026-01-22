import os
import glob
import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Simple ML pipeline that trains LR and RF for baseline and anonymized CSVs
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
ANON_DIR = os.path.join(BASE_DIR, "data", "anonymized_clean")
CLEAN_FILES = {
    "adult": os.path.abspath(os.path.join(BASE_DIR, "..", "project", "data","clean", "adult_clean.csv")),
    "bank": os.path.abspath(os.path.join(BASE_DIR, "..", "project", "data","clean", "bank_clean.csv")),
    "german": os.path.abspath(os.path.join(BASE_DIR, "..", "project", "data","clean", "german_clean.csv")),
}
TARGETS = {"adult": "income", "bank": "y", "german": "class"}
DATASETS = ["adult", "bank", "german"]
K_VALUES = [2, 5, 10, 20]
STRATEGIES = ["public", "all"]

results = []

print("Running the pipeline for all 3 datasets took about 6 minutes. Go grab a coffee for yourself")


def load_csv_try(path):
    # try semicolon first (most cleaned files use ';'), fallback to default delimiter
    try:
        df = pd.read_csv(path, sep=';', keep_default_na=False)
        # if this produced more than one column, accept it
        if df.shape[1] > 1:
            return df
    except Exception:
        pass
    # fallback to standard read (comma-delimited)
    return pd.read_csv(path, keep_default_na=False)


def fit_eval(X, y):
    # simple transform + train/test split with stratify fallback
    X = X.astype(str)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except Exception:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=None)

    feat_cols = list(X.columns)
    preproc = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=True), feat_cols)
    ], remainder='drop')

    # Logistic Regression
    lr_pipe = Pipeline([
        ("pre", preproc),
        ("clf", LogisticRegression(solver='saga', max_iter=2000, random_state=42))
    ])
    lr_pipe.fit(X_tr, y_tr)
    y_pred = lr_pipe.predict(X_te)
    lr_acc = accuracy_score(y_te, y_pred)
    lr_f1 = f1_score(y_te, y_pred, average='macro')

    # Random Forest (requires dense)
    preproc_rf = preproc
    Xtr_t = preproc_rf.fit_transform(X_tr)
    Xte_t = preproc_rf.transform(X_te)
    to_dense = FunctionTransformer(lambda x: x.toarray() if hasattr(x, 'toarray') else x, accept_sparse=True)
    Xtr_d = to_dense.transform(Xtr_t)
    Xte_d = to_dense.transform(Xte_t)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(Xtr_d, y_tr)
    y_pred_rf = rf.predict(Xte_d)
    rf_acc = accuracy_score(y_te, y_pred_rf)
    rf_f1 = f1_score(y_te, y_pred_rf, average='macro')

    return (len(feat_cols), (lr_acc, lr_f1), (rf_acc, rf_f1))


if __name__ == '__main__':
    start_time = time.time()
    for dataset in DATASETS:
        clean_path = CLEAN_FILES[dataset]
        print(f"Now: baseline {dataset}")
        if not os.path.exists(clean_path):
            print(f"  -> missing baseline file: {clean_path}")
            continue
        df_clean = load_csv_try(clean_path)
        target = TARGETS[dataset]
        if target not in df_clean.columns:
            print(f"  -> target {target} missing in baseline file")
            continue

        Xc = df_clean.drop(columns=[target])
        yc = df_clean[target]
        n_feat, lr_res, rf_res = fit_eval(Xc, yc)
        results.append({
            'dataset': dataset,
            'variant': 'baseline',
            'strategy': 'NA',
            'k': 'NA',
            'model': 'LogisticRegression',
            'n_rows': df_clean.shape[0],
            'n_features': n_feat,
            'accuracy': lr_res[0],
            'f1_macro': lr_res[1]
        })
        results.append({
            'dataset': dataset,
            'variant': 'baseline',
            'strategy': 'NA',
            'k': 'NA',
            'model': 'RandomForest',
            'n_rows': df_clean.shape[0],
            'n_features': n_feat,
            'accuracy': rf_res[0],
            'f1_macro': rf_res[1]
        })

        # anonymized variants
        for strategy in STRATEGIES:
            for k in K_VALUES:
                fname = f"{dataset}_k{k}_{strategy}.csv"
                path = os.path.join(ANON_DIR, fname)
                print(f"Now: {fname}")
                if not os.path.exists(path):
                    print(f"  -> file not found, skipping")
                    continue
                df_anon = load_csv_try(path)
                if target not in df_anon.columns:
                    print(f"  -> target missing in anonymized file, skipping")
                    continue
                Xa = df_anon.drop(columns=[target])
                ya = df_anon[target]
                try:
                    n_feat_a, lr_a, rf_a = fit_eval(Xa, ya)
                    results.append({
                        'dataset': dataset,
                        'variant': 'anonymized',
                        'strategy': strategy,
                        'k': k,
                        'model': 'LogisticRegression',
                        'n_rows': df_anon.shape[0],
                        'n_features': n_feat_a,
                        'accuracy': lr_a[0],
                        'f1_macro': lr_a[1]
                    })
                    results.append({
                        'dataset': dataset,
                        'variant': 'anonymized',
                        'strategy': strategy,
                        'k': k,
                        'model': 'RandomForest',
                        'n_rows': df_anon.shape[0],
                        'n_features': n_feat_a,
                        'accuracy': rf_a[0],
                        'f1_macro': rf_a[1]
                    })
                except Exception as e:
                    print(f"  -> exception during training: {e}")

    # Print results neatly
    if not results:
        print("No results collected.")
    else:
        res_df = pd.DataFrame(results)
        pd.set_option('display.max_rows', 200)
        print("\n=== Detailed results ===")
        print(res_df[['dataset','variant','strategy','k','model','n_rows','n_features','accuracy','f1_macro']].to_string(index=False))

        print("\n=== Summary per dataset/model ===")
        for dataset in DATASETS:
            sub = res_df[res_df['dataset']==dataset]
            if sub.empty:
                continue
            print(f"\n-- {dataset.upper()} --")
            for model in ['LogisticRegression','RandomForest']:
                base_row = sub[(sub['variant']=='baseline') & (sub['model']==model)]
                if not base_row.empty:
                    base_acc = base_row['accuracy'].values[0]
                    print(f"{model:<16} baseline acc = {base_acc:.3f}")
                anon = sub[(sub['variant']=='anonymized') & (sub['model']==model)]
                if anon.empty:
                    print(f"  no anonymized results for {model}")
                    continue
                best = anon.loc[anon['accuracy'].idxmax()]
                print(f"  best anonymized -> strategy={best['strategy']}, k={best['k']}, acc={best['accuracy']:.3f}")

        elapsed = time.time() - start_time
        print(f'\nDone. Total time: {elapsed:.1f} seconds (~{elapsed/60:.1f} minutes).')
        
        # Save results to CSV
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(BASE_DIR, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ml_results_{ts}.csv")
        res_df.to_csv(out_path, index=False)
        print(f"Results saved to: {out_path}")
