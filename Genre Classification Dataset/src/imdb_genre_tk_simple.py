# src/imdb_genre_tk_simple.py
# Super-simple Tkinter UI for IMDB genre prediction.
# • Auto-loads genre_model.joblib from project root (asks if missing)
# • Paste plot text → Predict
# • Optional: Batch from test_data.txt → save CSV/TXT

import os
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# ------------------ Minimal helpers ------------------

def _read_test_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = [p.strip() for p in line.split(":::", maxsplit=3)]
            if len(parts) != 3:
                parts = (parts + ["", "", ""])[:3]
            _id, title, plot = parts
            rows.append({"id": str(_id), "title": title, "plot": plot})
    df = pd.DataFrame(rows)
    for c in ("title", "plot"):
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    return df

def _texts_from(df: pd.DataFrame) -> np.ndarray:
    return (df["title"].fillna("") + " " + df["plot"].fillna("")).values

def _pipeline_proba(pipeline, texts_np: np.ndarray) -> np.ndarray:
    # Call on the Pipeline so TF-IDF is applied automatically
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(texts_np)
    scores = pipeline.decision_function(texts_np)
    return 1.0 / (1.0 + np.exp(-scores))

def predict_ranked(bundle_path: str, texts, threshold: float = 0.30, topk: int = 3):
    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    mlb = bundle["mlb"]
    # fall back to stored threshold if caller passes None
    if threshold is None and "threshold" in bundle:
        threshold = float(bundle["threshold"])
    texts_np = np.array(texts)
    proba = _pipeline_proba(model, texts_np)
    classes = mlb.classes_
    out = []
    for i in range(len(texts_np)):
        idx = np.where(proba[i] >= threshold)[0]
        if len(idx) == 0:
            idx = np.argsort(-proba[i])[:topk]
        ranked = sorted([(classes[j], float(proba[i][j])) for j in idx],
                        key=lambda x: -x[1])[:topk]
        out.append(ranked)
    return out

# ------------------ Super-simple UI ------------------

class SimpleApp:
    def __init__(self, root):
        self.root = root
        root.title("IMDB Genre Predictor (Simple)")
        root.geometry("780x520")

        self.model_path = tk.StringVar(value=os.path.abspath("genre_model.joblib"))
        self.topk = tk.IntVar(value=3)
        self.threshold = tk.DoubleVar(value=0.30)

        # Row 1: model picker (tiny)
        r1 = ttk.Frame(root, padding=8); r1.pack(fill="x")
        ttk.Label(r1, text="Model:").pack(side="left")
        self.ent_model = ttk.Entry(r1, textvariable=self.model_path, width=55)
        self.ent_model.pack(side="left", padx=6)
        ttk.Button(r1, text="Change…", command=self.pick_model).pack(side="left")
        ttk.Button(r1, text="Load", command=self.load_model).pack(side="left", padx=6)

        # Row 2: small controls
        r2 = ttk.Frame(root, padding=(8,0)); r2.pack(fill="x")
        ttk.Label(r2, text="Top-K").pack(side="left")
        ttk.Spinbox(r2, from_=1, to=10, textvariable=self.topk, width=4).pack(side="left", padx=6)

        ttk.Label(r2, text="Threshold").pack(side="left", padx=(12,0))
        self.ent_th = ttk.Entry(r2, width=6, textvariable=self.threshold)
        self.ent_th.pack(side="left", padx=6)

        ttk.Button(r2, text="Predict", command=self.predict_text).pack(side="right")
        ttk.Button(r2, text="Batch from test_data.txt…", command=self.batch_from_file).pack(side="right", padx=8)

        # Row 3: big input box
        inp = ttk.LabelFrame(root, text="Plot summary", padding=8)
        inp.pack(fill="both", expand=True, padx=8, pady=8)
        self.txt = ScrolledText(inp, height=10, wrap="word")
        self.txt.pack(fill="both", expand=True)

        # Row 4: simple output label
        out = ttk.Frame(root, padding=8); out.pack(fill="x")
        ttk.Label(out, text="Predicted genres:").pack(side="left")
        self.res = ttk.Label(out, text="—", font=("Segoe UI", 11, "bold"))
        self.res.pack(side="left", padx=8)

        self.bundle_loaded = False
        self._auto_load_model()

    # ---------- model handling ----------
    def _auto_load_model(self):
        if os.path.exists(self.model_path.get()):
            self.load_model()
        else:
            messagebox.showinfo("Model not found",
                                "Pick your saved model (genre_model.joblib).")

    def pick_model(self):
        path = filedialog.askopenfilename(
            title="Select model (.joblib)",
            filetypes=[("Joblib bundle", "*.joblib"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)

    def load_model(self):
        try:
            joblib.load(self.model_path.get())
            self.bundle_loaded = True
            messagebox.showinfo("Ready", "Model loaded.")
        except Exception as e:
            self.bundle_loaded = False
            messagebox.showerror("Load error", f"Could not load model:\n{e}")

    # ---------- actions ----------
    def predict_text(self):
        if not self.bundle_loaded:
            messagebox.showwarning("No model", "Load a model first."); return
        text = self.txt.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty", "Paste a plot summary."); return
        try:
            th = float(self.threshold.get())
            k = int(self.topk.get())
            ranked = predict_ranked(self.model_path.get(), [text], threshold=th, topk=k)[0]
            genres = ", ".join([g for g, _ in ranked])
            self.res.configure(text=genres if genres else "—")
        except Exception as e:
            messagebox.showerror("Predict error", f"{e}")

    def batch_from_file(self):
        if not self.bundle_loaded:
            messagebox.showwarning("No model", "Load a model first."); return
        test_path = filedialog.askopenfilename(
            title="Choose test_data.txt",
            filetypes=[("Text files","*.txt"),("All files","*.*")]
        )
        if not test_path:
            return
        try:
            df = _read_test_file(test_path)
            texts = _texts_from(df)
            th = float(self.threshold.get()); k = int(self.topk.get())
            ranked_all = predict_ranked(self.model_path.get(), texts, threshold=th, topk=k)
        except Exception as e:
            messagebox.showerror("Batch error", f"{e}"); return

        save_csv = filedialog.asksaveasfilename(
            title="Save predictions CSV", defaultextension=".csv",
            filetypes=[("CSV","*.csv")]
        )
        if not save_csv:
            return
        out = pd.DataFrame({
            "id": df["id"].values,
            "title": df["title"].values,
            "predicted_genres": [", ".join([g for g, _ in ranked]) for ranked in ranked_all]
        })
        try:
            out.to_csv(save_csv, index=False, encoding="utf-8")
            base, _ = os.path.splitext(save_csv)
            with open(base + ".txt", "w", encoding="utf-8") as f:
                for (_id, title, ranked) in zip(df["id"].values, df["title"].values, ranked_all):
                    f.write(f"{_id} ::: {title} ::: {', '.join([g for g, _ in ranked])}\n")
            messagebox.showinfo("Done", f"Saved:\n{save_csv}\n{base + '.txt'}")
        except Exception as e:
            messagebox.showerror("Save error", f"{e}")

def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    SimpleApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
