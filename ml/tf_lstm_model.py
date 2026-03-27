# pyre-ignore-all-errors
"""
TensorFlow LSTM Delinquency Risk Model
Alternative deep learning model using TensorFlow/Keras for temporal sequence analysis.
Detects gradual deterioration patterns alongside the PyTorch LSTM model.

This provides framework diversity as required by the problem statement
(PyTorch + TensorFlow for deep learning).
"""
import os
import sys
import numpy as np
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


class TFLSTMDelinquencyModel:
    """TensorFlow/Keras LSTM model for delinquency risk prediction.

    Uses the same interface as the PyTorch LSTMDelinquencyModel to allow
    seamless swapping in the ensemble scorer.
    """

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

    def _build_model(self):
        """Build TF/Keras LSTM model with attention."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            logger.warning("[TF-LSTM] TensorFlow not installed. Install with: pip install tensorflow")
            raise

        # Input layer
        inputs = keras.Input(shape=(None, self.input_size))

        # Stacked LSTM layers
        x = inputs
        for i in range(self.num_layers):
            return_seq = (i < self.num_layers - 1)  # return sequences for all but last
            x = layers.LSTM(
                self.hidden_size,
                return_sequences=True,  # Always return sequences for attention
                dropout=self.dropout,
                recurrent_dropout=0.1,
                name=f"lstm_{i}"
            )(x)

        # Self-Attention mechanism
        attention_scores = layers.Dense(1, activation='tanh', name='attention_score')(x)
        attention_weights = layers.Softmax(axis=1, name='attention_weights')(attention_scores)
        context = layers.Multiply()([x, attention_weights])
        context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name='context_sum')(context)

        # Classifier head
        x = layers.Dense(32, activation='relu', name='dense_1')(context)
        x = layers.Dropout(self.dropout)(x)
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='tf_lstm_delinquency')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['AUC']
        )

        return model

    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize features using mean/std."""
        if fit:
            flat = X.reshape(-1, X.shape[-1])
            self.scaler_mean = flat.mean(axis=0)
            self.scaler_std = flat.std(axis=0) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std

    def train(self, X: np.ndarray, y: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        Train the TensorFlow LSTM model.
        X shape: (num_samples, sequence_length, num_features)
        y shape: (num_samples,)
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            logger.warning("[TF-LSTM] TensorFlow not available. Skipping training.")
            return {"status": "skipped", "reason": "tensorflow not installed"}

        self.input_size = X.shape[2]
        X_norm = self._normalize(X, fit=True)

        # Build model
        self.model = self._build_model()

        # Callbacks
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5, patience=5, min_lr=1e-6
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10, restore_best_weights=True
            ),
        ]

        # Prepare validation data
        validation_data = None
        if X_val is not None:
            X_val_norm = self._normalize(X_val)
            validation_data = (X_val_norm, y_val)

        # Train
        history = self.model.fit(
            X_norm, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0,
        )

        # Compute metrics
        from sklearn.metrics import roc_auc_score
        train_pred = self.model.predict(X_norm, verbose=0).flatten()
        train_auc = roc_auc_score(y, train_pred)

        val_auc = 0
        if X_val is not None:
            val_pred = self.model.predict(X_val_norm, verbose=0).flatten()
            val_auc = roc_auc_score(y_val, val_pred)

        metrics = {
            "framework": "tensorflow",
            "train_auc": float(train_auc),
            "best_val_auc": float(val_auc),
            "epochs_trained": len(history.history['loss']),
            "input_size": self.input_size,
        }

        logger.info(f"[TF-LSTM] Training complete. AUC: {train_auc:.4f}, "
                    f"Val AUC: {val_auc:.4f}")
        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of delinquency for sequences."""
        if self.model is None:
            raise ValueError("Model not trained or loaded.")

        X_norm = self._normalize(X)
        predictions = self.model.predict(X_norm, verbose=0).flatten()
        return predictions

    def save(self, path: str = None):
        """Save model to disk."""
        path = path or os.path.join(MODEL_DIR, "tf_lstm_model")
        self.model.save(path)
        # Save normalization params separately
        np.savez(
            path + "_scaler.npz",
            mean=self.scaler_mean,
            std=self.scaler_std,
            input_size=np.array([self.input_size]),
        )
        logger.info(f"[TF-LSTM] Model saved to {path}")

    def load(self, path: str = None):
        """Load model from disk."""
        try:
            from tensorflow import keras
        except ImportError:
            logger.warning("[TF-LSTM] TensorFlow not installed.")
            return

        path = path or os.path.join(MODEL_DIR, "tf_lstm_model")
        self.model = keras.models.load_model(path)
        scaler = np.load(path + "_scaler.npz")
        self.scaler_mean = scaler["mean"]
        self.scaler_std = scaler["std"]
        self.input_size = int(scaler["input_size"][0])
        logger.info(f"[TF-LSTM] Model loaded from {path}")
