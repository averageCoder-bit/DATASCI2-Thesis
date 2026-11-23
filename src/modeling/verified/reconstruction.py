"""Verified Exp04 checkpoint restoration and 60x7 MAE reconstruction.

Adapted from `reproduce_lgmma_reconstruction.py`. TensorFlow is imported only
when restoration/inference is called, so calibration can be tested separately.
The protected architecture source is imported unchanged from src/modeling/architecture.
"""
from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path

import h5py
import numpy as np

from .config import (BATCH_SIZE, FEATURE_COUNT, HIDDEN_UNITS, LATENT_DIM,
                     DROPOUT, SEQUENCE_LENGTH)

_TENSOR_PATHS = {
    "enc_lstm_kernel": "encoder/lstm/cell/vars/0",
    "enc_lstm_recurrent": "encoder/lstm/cell/vars/1",
    "enc_lstm_bias": "encoder/lstm/cell/vars/2",
    "enc_latent_kernel": "encoder/latent/vars/0",
    "enc_latent_bias": "encoder/latent/vars/1",
    "dec_lstm_kernel": "decoder/lstm/cell/vars/0",
    "dec_lstm_recurrent": "decoder/lstm/cell/vars/1",
    "dec_lstm_bias": "decoder/lstm/cell/vars/2",
    "dec_output_kernel": "decoder/output_layer/layer/vars/0",
    "dec_output_bias": "decoder/output_layer/layer/vars/1",
}


def checkpoint_tensors(weights_bytes: bytes) -> dict[str, np.ndarray]:
    with h5py.File(io.BytesIO(weights_bytes), "r") as h5:
        return {name: h5[path][()] for name, path in _TENSOR_PATHS.items()}


def reconstruction_mae(sequences, reconstructed) -> np.ndarray:
    """Exact legacy MAE axes and float32 storage for (N,60,7) sequences."""
    x = np.asarray(sequences)
    y = np.asarray(reconstructed)
    if x.ndim != 3 or x.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(f"Expected (N,60,7) input; got {x.shape}")
    if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Input and reconstructed data differ in shape or contain nonfinite values")
    return np.mean(np.abs(x - y), axis=(1, 2)).astype(np.float32)


def restored_exp04_model(checkpoint_path: str | Path, input_sample: np.ndarray):
    """Restore .keras checkpoint weights and verify all 10 variables exactly.

    No checkpoint/model files are written or modified. Temporary HDF5 bytes
    are copied into a disposable temp directory as required by Keras.
    """
    import tensorflow as tf
    from src.modeling.architecture.autoencoder import LSTMAutoencoder

    path = Path(checkpoint_path)
    with zipfile.ZipFile(path, "r") as archive:
        weight_bytes = archive.read("model.weights.h5")
    saved = checkpoint_tensors(weight_bytes)
    sample = np.asarray(input_sample)
    if sample.ndim != 3 or sample.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT) or len(sample) == 0:
        raise ValueError(f"Invalid sample for model initialization: {sample.shape}")
    model = LSTMAutoencoder(
        hidden_units=HIDDEN_UNITS, latent_dim=LATENT_DIM,
        dropout=DROPOUT, sequence_length=SEQUENCE_LENGTH,
        feature_count=FEATURE_COUNT,
    )
    _ = model(tf.convert_to_tensor(sample[:1]), training=False)
    with tempfile.TemporaryDirectory() as folder:
        weights_file = Path(folder)/"model.weights.h5"
        weights_file.write_bytes(weight_bytes)
        model.load_weights(str(weights_file))
    actual = {
        "enc_lstm_kernel": model.encoder.lstm.cell.kernel.numpy(),
        "enc_lstm_recurrent": model.encoder.lstm.cell.recurrent_kernel.numpy(),
        "enc_lstm_bias": model.encoder.lstm.cell.bias.numpy(),
        "enc_latent_kernel": model.encoder.latent.kernel.numpy(),
        "enc_latent_bias": model.encoder.latent.bias.numpy(),
        "dec_lstm_kernel": model.decoder.lstm.cell.kernel.numpy(),
        "dec_lstm_recurrent": model.decoder.lstm.cell.recurrent_kernel.numpy(),
        "dec_lstm_bias": model.decoder.lstm.cell.bias.numpy(),
        "dec_output_kernel": model.decoder.output_layer.layer.kernel.numpy(),
        "dec_output_bias": model.decoder.output_layer.layer.bias.numpy(),
    }
    if len(model.weights) != 10 or set(actual) != set(saved) or not all(
        np.array_equal(actual[k], saved[k]) for k in saved
    ):
        raise RuntimeError("Exp04 checkpoint tensor restoration failed")
    return model


def infer_reconstruction_mae(model, sequences, batch_size: int = BATCH_SIZE):
    """Batch inference without model fitting; result shape (N,) float32."""
    import tensorflow as tf
    x = np.asarray(sequences)
    if x.ndim != 3 or x.shape[1:] != (SEQUENCE_LENGTH, FEATURE_COUNT):
        raise ValueError(f"Expected (N,60,7) inputs; got {x.shape}")
    results = []
    for offset in range(0, len(x), batch_size):
        batch = np.asarray(x[offset:offset+batch_size])
        predicted = model(tf.convert_to_tensor(batch), training=False).numpy()
        results.append(reconstruction_mae(batch, predicted))
    return np.concatenate(results) if results else np.empty(0, dtype=np.float32)
