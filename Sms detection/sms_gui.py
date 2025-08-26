import tkinter as tk
from tkinter import messagebox
import joblib
import re
import os


if not os.path.exists("spam_model.pkl") or not os.path.exists("spam_tfidf.pkl"):
    raise FileNotFoundError("Trained model not found. Please run sms_spam_classifier.py first.")

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("spam_tfidf.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_spam():
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter a message.")
        return

    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result_label.config(text=" SPAM Message", fg="red")
    else:
        result_label.config(text="Legitimate Message (HAM)", fg="green")


window = tk.Tk()
window.title("SMS Spam Detector")
window.geometry("600x400")
window.config(padx=20, pady=20, bg="white")

title = tk.Label(window, text="📩 SMS Spam Detector", font=("Helvetica", 18, "bold"), bg="white")
title.pack(pady=10)

text_input = tk.Text(window, height=8, width=70, font=("Arial", 12))
text_input.pack(pady=10)

check_button = tk.Button(window, text="Check Message", command=predict_spam,
                         font=("Arial", 12), bg="black", fg="white")
check_button.pack(pady=10)

result_label = tk.Label(window, text="", font=("Helvetica", 16, "bold"), bg="white")
result_label.pack(pady=10)

window.mainloop()
