import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import joblib

model = joblib.load("XGBoost_churn_model.pkl")
data_sample = pd.read_csv("Churn_Modelling.csv")
columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
           'HasCrCard', 'IsActiveMember', 'EstimatedSalary']

label_encoder_cols = {}
for col in ['Geography', 'Gender']:
    label_encoder_cols[col] = {k: i for i, k in enumerate(data_sample[col].unique())}

root = tk.Tk()
root.title("Customer Churn Prediction")

entries = {}

def predict():
    try:
        values = []
        for col in columns:
            if col == "Gender":
                val = gender_dropdown.get().strip()
                val = label_encoder_cols["Gender"].get(val, -1)
                if val == -1:
                    raise ValueError("Invalid gender selected")
            else:
                val = entries[col].get().strip()
                if col in label_encoder_cols and col != "Gender":
                    val = label_encoder_cols[col].get(val, -1)
                    if val == -1:
                        raise ValueError(f"Invalid input for {col}")
                else:
                    val = float(val)
            values.append(val)

        x_input = np.array(values).reshape(1, -1)
        prediction = model.predict(x_input)[0]
        result = "Churn" if prediction == 1 else "Not Churn"
        result_label.config(text=f"Prediction: {result}", fg="green")
    except Exception as e:
        result_label.config(text=f"Error: {str(e)}", fg="orange")

# GUI layout
for i, col in enumerate(columns):
    label = tk.Label(root, text=col)
    label.grid(row=i, column=0, padx=5, pady=5)

    if col == "Gender":
        gender_dropdown = ttk.Combobox(root, values=list(label_encoder_cols["Gender"].keys()), state="readonly")
        gender_dropdown.grid(row=i, column=1, padx=5, pady=5)
        gender_dropdown.set(list(label_encoder_cols["Gender"].keys())[0])
    else:
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[col] = entry

submit_btn = tk.Button(root, text="Predict Churn", command=predict)
submit_btn.grid(row=len(columns), column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="", fg="blue")
result_label.grid(row=len(columns)+1, column=0, columnspan=2)

root.mainloop()
