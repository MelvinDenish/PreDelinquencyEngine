# pyre-ignore-all-errors
"""
Temporal Fusion Transformer (TFT) for Delinquency Risk Scoring
Uses pytorch-forecasting's TFT architecture for handling mixed
static + temporal covariates with interpretable attention weights.
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 60, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) — the core building block of TFT.
    Applies ELU activation, gating via GLU, and residual connections.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None
        self.fc2 = nn.Linear(hidden_size, output_size * 2)  # *2 for GLU
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

        # Skip connection projection if sizes differ
        if input_size != output_size:
            self.skip_proj = nn.Linear(input_size, output_size)
        else:
            self.skip_proj = None

    def forward(self, x, context=None):
        residual = self.skip_proj(x) if self.skip_proj else x
        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            h = h + self.context_fc(context)
        h = torch.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        # GLU (Gated Linear Unit)
        h1, h2 = h.chunk(2, dim=-1)
        h = h1 * torch.sigmoid(h2)
        return self.layer_norm(h + residual)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network — learns which features matter most
    at each timestep. Key TFT component for interpretability.
    """

    def __init__(self, input_size: int, num_features: int, hidden_size: int,
                 dropout: float = 0.1, context_size: int = None):
        super().__init__()
        self.num_features = num_features

        # Feature-level GRNs
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(input_size, hidden_size, hidden_size, dropout)
            for _ in range(num_features)
        ])

        # Softmax weights for variable selection
        self.weight_grn = GatedResidualNetwork(
            input_size * num_features, hidden_size, num_features,
            dropout, context_size
        )

    def forward(self, x, context=None):
        # x: (batch, seq, num_features, feature_dim) or (batch, num_features, feature_dim)
        # Flatten for weight computation
        batch_size = x.shape[0]
        if x.dim() == 4:
            seq_len = x.shape[1]
            flat = x.reshape(batch_size * seq_len, -1)
        else:
            flat = x.reshape(batch_size, -1)

        # Compute selection weights
        weights = self.weight_grn(flat, context)
        weights = torch.softmax(weights, dim=-1)  # (batch*seq, num_features)

        # Apply per-feature GRNs
        if x.dim() == 4:
            processed = []
            for i in range(self.num_features):
                feat = x[:, :, i, :]  # (batch, seq, feature_dim)
                processed.append(
                    self.feature_grns[i](feat.reshape(-1, feat.shape[-1]))
                )
            processed = torch.stack(processed, dim=-2)  # (batch*seq, num_features, hidden)
            weights_exp = weights.unsqueeze(-1)
            selected = (processed * weights_exp).sum(dim=-2)
            return selected.reshape(batch_size, seq_len, -1), weights.reshape(batch_size, seq_len, -1)
        else:
            processed = []
            for i in range(self.num_features):
                processed.append(self.feature_grns[i](x[:, i, :]))
            processed = torch.stack(processed, dim=-2)
            weights_exp = weights.unsqueeze(-1)
            selected = (processed * weights_exp).sum(dim=-2)
            return selected, weights


class TFTDelinquencyModel:
    """
    Temporal Fusion Transformer for delinquency risk prediction.

    Architecture:
        Static covariates → GRN → context vectors
        Temporal features → Positional Encoding → Variable Selection
        → LSTM Encoder → Multi-Head Attention → GRN → Sigmoid

    Input:
        static_features: (batch, n_static) — age, income_bracket, segment_type, etc.
        temporal_features: (batch, seq_len, n_temporal) — 30-day feature sequences

    Output:
        risk_probability: (batch,) — default risk probability
        attention_weights: (batch, seq_len) — per-timestep interpretable attention
    """

    def __init__(self, n_temporal_features: int, n_static_features: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2,
                 dropout: float = 0.2, seq_len: int = 30,
                 epochs: int = 50, batch_size: int = 64,
                 learning_rate: float = 1e-3, label_smoothing: float = 0.05):
        self.n_temporal = n_temporal_features
        self.n_static = n_static_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.label_smoothing = label_smoothing

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def _build_model(self):
        """Build the TFT architecture."""
        model = _TFTNetwork(
            n_temporal=self.n_temporal,
            n_static=self.n_static,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
        )
        return model.to(self.device)

    def train(self, X_temporal_train: np.ndarray, X_static_train: np.ndarray,
              y_train: np.ndarray, X_temporal_val: np.ndarray = None,
              X_static_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train the TFT model.

        Args:
            X_temporal_train: (N, seq_len, n_temporal)
            X_static_train: (N, n_static)
            y_train: (N,)
        """
        self.model = self._build_model()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([2.0]).to(self.device)  # Class imbalance
        )

        # Convert to tensors
        X_temp = torch.FloatTensor(X_temporal_train).to(self.device)
        X_stat = torch.FloatTensor(X_static_train).to(self.device)
        y = torch.FloatTensor(y_train).to(self.device)

        # Label smoothing
        if self.label_smoothing > 0:
            y = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        dataset = torch.utils.data.TensorDataset(X_temp, X_stat, y)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        best_val_auc = 0.0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            for batch_temp, batch_stat, batch_y in loader:
                optimizer.zero_grad()
                logits, _ = self.model(batch_temp, batch_stat)
                loss = criterion(logits.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()

            # Validation
            if X_temporal_val is not None and (epoch + 1) % 5 == 0:
                val_auc = self._evaluate(X_temporal_val, X_static_val, y_val)
                avg_loss = total_loss / len(loader)
                logger.info(
                    f"[TFT] Epoch {epoch+1}/{self.epochs} | "
                    f"Loss: {avg_loss:.4f} | Val AUC: {val_auc:.4f}"
                )
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # Restore best model
        if best_state:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        metrics = {"best_val_auc": best_val_auc}
        logger.info(f"[TFT] Training complete | Best Val AUC: {best_val_auc:.4f}")
        return metrics

    def _evaluate(self, X_temporal: np.ndarray, X_static: np.ndarray,
                  y: np.ndarray) -> float:
        """Evaluate AUC on a dataset."""
        from sklearn.metrics import roc_auc_score
        probs = self.predict_proba(X_temporal, X_static)
        try:
            return roc_auc_score(y, probs)
        except ValueError:
            return 0.5

    def predict_proba(self, X_temporal: np.ndarray,
                      X_static: np.ndarray) -> np.ndarray:
        """Predict default probabilities."""
        self.model.eval()
        with torch.no_grad():
            X_temp = torch.FloatTensor(X_temporal).to(self.device)
            X_stat = torch.FloatTensor(X_static).to(self.device)
            logits, _ = self.model(X_temp, X_stat)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
        return probs

    def get_attention_weights(self, X_temporal: np.ndarray,
                              X_static: np.ndarray) -> np.ndarray:
        """
        Get per-timestep attention weights for interpretability.
        Returns: (batch, seq_len) — which days drove the risk prediction.
        """
        self.model.eval()
        with torch.no_grad():
            X_temp = torch.FloatTensor(X_temporal).to(self.device)
            X_stat = torch.FloatTensor(X_static).to(self.device)
            _, attention = self.model(X_temp, X_stat)
        return attention.cpu().numpy()

    def save(self, path: str):
        """Save model state."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": {
                "n_temporal": self.n_temporal,
                "n_static": self.n_static,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "seq_len": self.seq_len,
            }
        }, path)
        logger.info(f"[TFT] Model saved to {path}")

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        config = checkpoint["config"]
        self.n_temporal = config["n_temporal"]
        self.n_static = config["n_static"]
        self.d_model = config["d_model"]
        self.nhead = config["nhead"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.seq_len = config["seq_len"]
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        logger.info(f"[TFT] Model loaded from {path}")


class _TFTNetwork(nn.Module):
    """Internal TFT neural network architecture."""

    def __init__(self, n_temporal: int, n_static: int, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model

        # Static covariate encoder
        self.static_encoder = GatedResidualNetwork(n_static, d_model, d_model, dropout)
        self.static_context_enrichment = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.static_context_state_h = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.static_context_state_c = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # Temporal input projection
        self.temporal_proj = nn.Linear(n_temporal, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # LSTM encoder (local temporal processing)
        self.lstm_encoder = nn.LSTM(
            input_size=d_model, hidden_size=d_model,
            num_layers=1, batch_first=True, dropout=dropout
        )

        # Gated skip connection after LSTM
        self.post_lstm_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # Self-attention (interpretable multi-head)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )
        self.post_attn_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # Final prediction head
        self.output_grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, temporal: torch.Tensor, static: torch.Tensor):
        """
        Args:
            temporal: (batch, seq_len, n_temporal)
            static: (batch, n_static)

        Returns:
            logits: (batch, 1)
            attention_weights: (batch, seq_len)
        """
        batch_size = temporal.shape[0]
        seq_len = temporal.shape[1]

        # 1. Encode static covariates → context vectors
        static_enc = self.static_encoder(static)  # (batch, d_model)
        context_enrichment = self.static_context_enrichment(static_enc)
        h0 = self.static_context_state_h(static_enc).unsqueeze(0)  # (1, batch, d_model)
        c0 = self.static_context_state_c(static_enc).unsqueeze(0)

        # 2. Project temporal features and add positional encoding
        temporal_proj = self.temporal_proj(temporal)  # (batch, seq, d_model)
        temporal_proj = self.pos_encoder(temporal_proj)

        # 3. LSTM encoder with static context as initial state
        lstm_out, _ = self.lstm_encoder(temporal_proj, (h0, c0))

        # 4. Gated skip connection after LSTM
        lstm_gate = self.post_lstm_grn(lstm_out.reshape(-1, self.d_model))
        lstm_gate = lstm_gate.reshape(batch_size, seq_len, self.d_model)
        lstm_out = self.post_lstm_norm(lstm_gate + temporal_proj)

        # 5. Static enrichment: add context to each timestep
        enrichment = context_enrichment.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_out = lstm_out + enrichment

        # 6. Interpretable multi-head self-attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            need_weights=True, average_attn_weights=True
        )
        # attn_weights: (batch, seq_len, seq_len)

        # 7. Gated skip connection after attention
        attn_gate = self.post_attn_grn(attn_out.reshape(-1, self.d_model))
        attn_gate = attn_gate.reshape(batch_size, seq_len, self.d_model)
        attn_out = self.post_attn_norm(attn_gate + lstm_out)

        # 8. Take last timestep output for classification
        final_state = attn_out[:, -1, :]  # (batch, d_model)

        # 9. Final GRN + linear output
        output = self.output_grn(final_state)
        logits = self.output_fc(output)  # (batch, 1)

        # 10. Aggregate attention: average over heads and query dimension
        # → per-timestep importance (which days drove the prediction)
        per_timestep_attn = attn_weights[:, -1, :]  # (batch, seq_len) — last query's attention

        return logits, per_timestep_attn
