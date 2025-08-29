import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib
import os

# Load dataset
df = pd.read_csv("train_data.txt", sep=":::", engine="python", header=None, names=["ID", "Title", "Genre", "Plot"])
df.dropna(subset=["Genre", "Plot"], inplace=True)

# Keep top 15 genres for better performance
df["Genre"] = df["Genre"].apply(lambda x: [g.strip().lower() for g in x.split(",")])
top_genres = df["Genre"].explode().value_counts().nlargest(15).index
df["Genre"] = df["Genre"].apply(lambda g_list: [g for g in g_list if g in top_genres])
df = df[df["Genre"].map(len) > 0]  # Keep rows with at least one valid genre

# Binarize labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["Genre"])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df["Plot"], y, test_size=0.1, random_state=42)

# Save label encoder
joblib.dump(mlb, "bert_label_binarizer.pkl")

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataset class
class MovieDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        item = {key: val.squeeze() for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# Create dataloaders
train_dataset = MovieDataset(X_train, y_train)
test_dataset = MovieDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=y.shape[1], problem_type="multi_label_classification")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Training on: {device}")

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train loop
for epoch in range(3):
    print(f"\nEpoch {epoch + 1}")
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

# Save model + tokenizer
os.makedirs("bert_genre_model", exist_ok=True)
model.save_pretrained("bert_genre_model")
tokenizer.save_pretrained("bert_genre_model")

# Evaluation
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        labels = batch["labels"].cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        outputs = model(**batch)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()

        all_preds.append(probs)
        all_labels.append(labels)

y_pred = np.vstack(all_preds)
y_true = np.vstack(all_labels)

# Threshold-based prediction
y_bin = (y_pred >= 0.3).astype(int)

print("\nEvaluation Report:")
print(classification_report(y_true, y_bin, target_names=mlb.classes_))
