import gc
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import  compute_class_weight

from db_utils import get_data, get_canadian_data


class RecordingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.long).to(device)

    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, n_features: int, n_genres: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_genres)
        )
    def forward(self, x: torch.Tensor):
        return self.network(x)

def _run_epoch(
        model: MLP,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.CrossEntropyLoss,
        optimizer: Optional[Adam] = None
        ) -> tuple[float, float]:
    
    if optimizer:
        model.train()

    else:
        model.eval()

    total_loss = 0
    correct = 0

    with torch.set_grad_enabled(optimizer is not None):
        for X_batch, y_batch in loader:
            y_batch = y_batch.to(device)
            X_batch = X_batch.to(device)

            if optimizer:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if optimizer:
                loss.backward()
                optimizer.step() #type: ignore

            total_loss += loss.item() * len(y_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
    
    return total_loss / len(loader.dataset), correct / len(loader.dataset) #type: ignore

def main(canadian:bool = False, plot: bool = False):
    if canadian:
        df = get_canadian_data()
    else:
        df = get_data()

    le = LabelEncoder()
    y = le.fit_transform(df["genre_tzanetakis"])
    X = StandardScaler().fit_transform(df.drop(columns=["genre_tzanetakis"])) 

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    del df, X, y
    gc.collect()

    loader_train = DataLoader(RecordingDataset(X_train, y_train), batch_size = 64, shuffle = True)
    loader_val = DataLoader(RecordingDataset(X_val, y_val), batch_size = 64)
    loader_test = DataLoader(RecordingDataset(X_test, y_test), batch_size = 64)

    model = MLP(X_train.shape[1], len(le.classes_)).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = Adam(model.parameters(), lr = 0.001)

    best_val_loss = float("inf")
    patience = 5
    counter = 0

    if plot:
        train_losses, val_losses = [], []
        train_accs,   val_accs   = [], []


    for epoch in range(1, 51):
        train_loss, train_acc = _run_epoch(model, loader_train, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, loader_val, criterion)

        if plot:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

        print(f"Epoch {epoch}:\n\tTrain Loss - {train_loss:4f}\n\tVal Accuracy - {val_acc:4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0

        else:
            counter += 1
            if counter >= patience:
                print("Early stopping...")
                break
    
    model.load_state_dict(torch.load("best_model.pth"))
    _, test_acc = _run_epoch(model, loader_test, criterion)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")

    all_preds, all_labels = [], []

    model.eval()
    model.cpu()
    with torch.no_grad():
        for X_batch, y_batch in loader_test:
            preds = model(X_batch.cpu()).argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y_batch.cpu().numpy())

    print(f"Weighted F1 Score: {f1_score(all_labels, all_preds, average='weighted')}")
    print(f"Test F1 Per Category: {f1_score(all_labels, all_preds, average=None)}")

    if plot:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label="Train Loss", color="steelblue")
        plt.plot(epochs, val_losses,   label="Val Loss",   color="tomato")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./images/{'canadian_' if canadian else ''}mlp_training_curves.png", bbox_inches="tight", dpi=150)
        plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(True, True)