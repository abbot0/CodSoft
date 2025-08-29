# src/data_io.py
# Robust readers for the Kaggle IMDB Genre dataset (":::"-separated)

import os
import pandas as pd

def _read_kaggle_imdb(path: str, has_genre: bool) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = [p.strip() for p in line.split(":::", maxsplit=3)]
            if has_genre:
                if len(parts) != 4:
                    parts = (parts + ["", "", "", ""])[:4]
                _id, title, genre, plot = parts
                rows.append({
                    "id": str(_id),
                    "title": title,
                    "genre": genre,
                    "genre_list": [g.strip() for g in genre.split(",") if g.strip()],
                    "plot": plot
                })
            else:
                if len(parts) != 3:
                    parts = (parts + ["", "", ""])[:3]
                _id, title, plot = parts
                rows.append({"id": str(_id), "title": title, "plot": plot})

    df = pd.DataFrame(rows)
    for c in {"title", "plot"} & set(df.columns):
        df[c] = df[c].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if has_genre:
        df["genre"] = df["genre"].fillna("")
        df["genre_list"] = df["genre_list"].apply(lambda xs: [x for x in xs if x])
    return df

def load_train_data(path: str) -> pd.DataFrame:
    return _read_kaggle_imdb(path, has_genre=True)

def load_test_data(path: str, has_genre: bool = False) -> pd.DataFrame:
    return _read_kaggle_imdb(path, has_genre=has_genre)
