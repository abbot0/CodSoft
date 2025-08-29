# src/imdb_genre_ml.py
# End-to-end multi-label genre classifier.
# - reads data (train/test/test_solution)
# - trains TF-IDF -> One-vs-Rest classifier (LogReg/SVM/NB)
# - prints validation + official test metrics (if solution present)
# - saves model bundle (.joblib)
# - optional batch prediction for test_data.txt -> CSV/TXT

import os, argparse, joblib, numpy as np, pandas as pd
from typing import List, Tuple

from data_io import load_train_data, load_test_data

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, jaccard_score, hamming_loss

SEED = 42

def texts_from(df: pd.DataFrame) -> np.ndarray:
    return (df["title"].fillna("") + " " + df["plot"].fillna("")).values

def build_pipeline(clf_name: str = "logreg") -> Pipeline:
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=120_000,
        lowercase=True,
        strip_accents="unicode"
    )
    if clf_name == "logreg":
        base = LogisticRegression(max_iter=1000, solver="liblinear",
                                  class_weight="balanced", random_state=SEED)
        clf = OneVsRestClassifier(base, n_jobs=-1)
    elif clf_name == "svm":
        base = CalibratedClassifierCV(LinearSVC(), cv=3)  # adds predict_proba
        clf = OneVsRestClassifier(base, n_jobs=-1)
    elif clf_name == "nb":
        base = MultinomialNB()
        clf = OneVsRestClassifier(base, n_jobs=-1)
    else:
        raise ValueError("Unknown --clf (use: logreg | svm | nb)")
    return Pipeline([("tfidf", tfidf), ("clf", clf)])

def proba_from_pipeline(pipeline: Pipeline, X_text: np.ndarray) -> np.ndarray:
    # Pipeline exposes predict_proba/decision_function if final step has it
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X_text)
    scores = pipeline.decision_function(X_text)
    return 1.0 / (1.0 + np.exp(-scores))

def evaluate_arrays(Y_true, Y_hat, title="Evaluation"):
    print(f"\n=== {title} ===")
    print(f"F1-micro : {f1_score(Y_true, Y_hat, average='micro', zero_division=0):.4f}")
    print(f"F1-macro : {f1_score(Y_true, Y_hat, average='macro', zero_division=0):.4f}")
    print(f"Jaccard (samples): {jaccard_score(Y_true, Y_hat, average='samples', zero_division=0):.4f}")
    print(f"Hamming loss: {hamming_loss(Y_true, Y_hat):.4f}")

def join_labels(pairs: List[Tuple[str, float]]) -> str:
    return ", ".join([g for g, _ in pairs])

def ranked_from_proba(proba: np.ndarray, classes: np.ndarray,
                      threshold: float, topk: int = 3) -> List[List[Tuple[str,float]]]:
    out = []
    for i in range(proba.shape[0]):
        idx = np.where(proba[i] >= threshold)[0]
        if len(idx) == 0:
            idx = np.argsort(-proba[i])[:topk]
        ranked = sorted([(classes[j], float(proba[i, j])) for j in idx],
                        key=lambda x: -x[1])[:topk]
        out.append(ranked)
    return out

def main():
    ap = argparse.ArgumentParser(description="IMDB Genre Classification (multi-label)")
    ap.add_argument("--data_dir", type=str, default="data",
                    help="Folder with train_data.txt, test_data.txt, test_data_solution.txt")
    ap.add_argument("--out", type=str, default="genre_model.joblib",
                    help="Path to save trained model bundle")
    ap.add_argument("--clf", type=str, default="logreg", choices=["logreg", "svm", "nb"])
    ap.add_argument("--threshold", type=float, default=0.30,
                    help="Probability threshold for multi-label selection")
    ap.add_argument("--predict_file", type=str, default=None,
                    help="If set, write predictions for test_data.txt to this CSV")
    args = ap.parse_args()

    train_path = os.path.join(args.data_dir, "train_data.txt")
    test_path = os.path.join(args.data_dir, "test_data.txt")
    test_sol_path = os.path.join(args.data_dir, "test_data_solution.txt")

    # ---- Load train
    print("Loading training data...")
    train_df = load_train_data(train_path)
    print(f"Train rows: {len(train_df)}")

    # ---- Labels
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(train_df["genre_list"])
    X = texts_from(train_df)

    # ---- Split, fit
    X_tr, X_va, Y_tr, Y_va = train_test_split(X, Y, test_size=0.15, random_state=SEED)
    model = build_pipeline(args.clf)
    print(f"Training classifier = {args.clf} ...")
    model.fit(X_tr, Y_tr)

    # ---- Validation metrics
    proba_va = proba_from_pipeline(model, X_va)
    Y_hat_va = (proba_va >= args.threshold).astype(int)
    evaluate_arrays(Y_va, Y_hat_va, title="Validation")

    # ---- Save bundle
    bundle = {"model": model, "mlb": mlb, "threshold": float(args.threshold), "clf": args.clf}
    joblib.dump(bundle, args.out)
    print(f"\nSaved model → {args.out}")

    # ---- Evaluate on official test (if available)
    if os.path.exists(test_sol_path):
        test_sol_df = load_test_data(test_sol_path, has_genre=True)
        # drop labels unseen during training to keep shapes consistent
        allowed = set(mlb.classes_)
        test_sol_df["genre_list"] = test_sol_df["genre_list"].apply(lambda xs: [x for x in xs if x in allowed])
        X_te = texts_from(test_sol_df)
        Y_te = mlb.transform(test_sol_df["genre_list"])
        proba_te = proba_from_pipeline(model, X_te)
        Y_hat_te = (proba_te >= args.threshold).astype(int)
        evaluate_arrays(Y_te, Y_hat_te, title="Official Test (with solution)")

    # ---- Optional: batch predict for test_data.txt
    if args.predict_file:
        if not os.path.exists(test_path):
            print(f"Warning: {test_path} not found; skipping batch predictions.")
            return
        test_df = load_test_data(test_path, has_genre=False)
        X_pred = texts_from(test_df)
        proba = proba_from_pipeline(model, X_pred)
        ranked = ranked_from_proba(proba, mlb.classes_, threshold=args.threshold, topk=3)

        out_csv = args.predict_file
        pd.DataFrame({
            "id": test_df["id"].values,
            "title": test_df["title"].values,
            "predicted_genres": [join_labels(r) for r in ranked],
        }).to_csv(out_csv, index=False, encoding="utf-8")
        base, _ = os.path.splitext(out_csv)
        with open(base + ".txt", "w", encoding="utf-8") as f:
            for _id, title, r in zip(test_df["id"].values, test_df["title"].values, ranked):
                f.write(f"{_id} ::: {title} ::: {join_labels(r)}\n")
        print(f"Saved predictions → {out_csv} and {base + '.txt'}")

if __name__ == "__main__":
    main()
