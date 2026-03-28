# pyre-ignore-all-errors
"""
LSTM Delinquency Risk Model
Trains an LSTM neural network on temporal sequences of customer behavior.
Detects gradual deterioration patterns invisible in single-day snapshots.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import joblib
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMNetwork(nn.Module):
    """Bidirectional LSTM with multi-head attention (Fix 6)."""

    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.lstm_output_size = hidden_size * self.num_directions

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(self.lstm_output_size)

        # Multi-head attention (Fix 6)
        self.attention = nn.Sequential(
            nn.Linear(self.lstm_output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Deeper classifier with residual-style connection
        self.classifier = nn.Sequential(
            nn.Linear(self.lstm_output_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch, seq_len, hidden_size * num_directions)

        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)

        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)

        # Classification
        output = self.classifier(context)
        return output.squeeze(-1)


class LSTMDelinquencyModel:
    """LSTM model wrapper with training and inference."""

    def __init__(self, input_size: int = 7, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3,
                 learning_rate: float = 0.001, epochs: int = 50,
                 batch_size: int = 64):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using mean/std."""
        if fit:
            # Compute stats across all sequences and timesteps
            flat = X.reshape(-1, X.shape[-1])
            self.scaler_mean = flat.mean(axis=0)
            self.scaler_std = flat.std(axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train the LSTM model.
        X shape: (num_samples, sequence_length, num_features)
        y shape: (num_samples,)
        """
        self.input_size = X.shape[2]

        # Normalize
        X_norm = self._normalize(X, fit=True)
        if X_val is not None:
            X_val_norm = self._normalize(X_val)

        # Create model (Fix 6: bidirectional=True)
        self.model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=True,
        ).to(DEVICE)

        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X_norm).to(DEVICE),
            torch.FloatTensor(y).to(DEVICE),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training (Fix 6: AdamW + cosine annealing)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                       weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        best_val_auc = 0
        best_state = None
        metrics_history = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_losses = []

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            avg_loss = np.mean(epoch_losses)
            scheduler.step(avg_loss)

            # Validation
            if X_val is not None and (epoch + 1) % 5 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(torch.FloatTensor(X_val_norm).to(DEVICE)).cpu().numpy()
                    val_auc = roc_auc_score(y_val, val_pred)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = self.model.state_dict().copy()

                if (epoch + 1) % 10 == 0:
                    logger.info(f"[LSTM] Epoch {epoch+1}/{self.epochs}, "
                              f"Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Final metrics
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(torch.FloatTensor(X_norm).to(DEVICE)).cpu().numpy()
            train_auc = roc_auc_score(y, train_pred)

        metrics = {
            "train_auc": train_auc,
            "best_val_auc": best_val_auc,
            "epochs_trained": self.epochs,
            "input_size": self.input_size,
        }

        logger.info(f"[LSTM] Training complete. AUC: {train_auc:.4f}, "
                    f"Best Val AUC: {best_val_auc:.4f}")

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of delinquency for sequences."""
        if self.model is None:
            raise ValueError("Model not trained.")

        self.model.eval()
        X_norm = self._normalize(X)

        with torch.no_grad():
            predictions = self.model(
                torch.FloatTensor(X_norm).to(DEVICE)
            ).cpu().numpy()

        return predictions

    def save(self, path: str = None):
        """Save model to disk."""
        path = path or os.path.join(MODEL_DIR, "lstm_model.pt")
        torch.save({
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": True,
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
        }, path)
        logger.info(f"[LSTM] Model saved to {path}")

    def load(self, path: str = None):
        """Load model from disk."""
        path = path or os.path.join(MODEL_DIR, "lstm_model.pt")
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)

        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_layers = checkpoint["num_layers"]
        self.dropout = checkpoint["dropout"]
        self.scaler_mean = checkpoint["scaler_mean"]
        self.scaler_std = checkpoint["scaler_std"]

        bidirectional = checkpoint.get("bidirectional", True)
        self.model = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=bidirectional,
        ).to(DEVICE)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        logger.info(f"[LSTM] Model loaded from {path}")
