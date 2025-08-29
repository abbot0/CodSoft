import tkinter as tk
from tkinter import messagebox
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
import joblib

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert_genre_model")
tokenizer = BertTokenizer.from_pretrained("bert_genre_model")
mlb = joblib.load("bert_label_binarizer.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Inference function
def predict_genres(plot):
    encoding = tokenizer(plot, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    top_indices = np.argsort(probs)[-3:][::-1]
    top_genres = [mlb.classes_[i] for i in top_indices]
    return top_genres

# GUI setup
def on_predict():
    plot = input_box.get("1.0", tk.END).strip()
    if not plot:
        messagebox.showwarning("Input Error", "Please enter a movie plot.")
        return

    genres = predict_genres(plot)
    result_label.config(text=f"Top 3 Predicted Genres:\n{', '.join(genres)}")

# Main window
root = tk.Tk()
root.title("Movie Genre Predictor")

tk.Label(root, text="Enter Movie Plot:", font=("Arial", 14)).pack(pady=5)
input_box = tk.Text(root, height=10, width=60, font=("Arial", 12))
input_box.pack(padx=10, pady=5)

tk.Button(root, text="Predict Genres", command=on_predict, font=("Arial", 12), bg="lightblue").pack(pady=10)
result_label = tk.Label(root, text="", font=("Arial", 14), fg="green")
result_label.pack(pady=5)

root.mainloop()
