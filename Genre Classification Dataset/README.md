IMDB Genre Classification (Multi-Label)

1) Place the Kaggle dataset files in ./data:
   - train_data.txt
   - test_data.txt
   - test_data_solution.txt (optional)

2) Create venv & install:
   py -m venv .venv
   . .venv\Scripts\Activate.ps1
   pip install -r requirements.txt

3) Train & evaluate:
   python src/imdb_genre_ml.py --data_dir data --out genre_model.joblib --predict_file predictions.csv

4) GUI for predictions:
   python src/imdb_genre_tk.py
   • Browse… to select genre_model.joblib (if not in project root)
   • Paste plot text and click "Predict Text"
   • Batch → select test_data.txt → export CSV/TXT
