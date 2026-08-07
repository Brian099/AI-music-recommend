# -*- coding: utf-8 -*-
# Written by GD Studio
# Date: 2026-04-16

import numpy as np
import os
import sys
import torch
import torch.nn.functional as functional
from typing import Optional

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from train.dataset import build_dense_vector, key_to_idx, mode_to_idx, tempo_to_idx, time_signature_to_idx
from train.model import EmbeatMLP, EmbeatMLPConfig


# Get model default device
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


# Build model and load state dict from checkpoint
def load_model(checkpoint_path: str = "train/checkpoints/model.pt", device: Optional[str] = None, strict: bool = True):
    checkpoint_path = os.path.abspath(str(checkpoint_path).replace("\\", "/").strip())
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint_dict, dict):
        raise ValueError(f"Checkpoint must be a dict, got: {type(checkpoint_dict)}")
    model_config_dict = ((checkpoint_dict.get("config") or {}).get("model_config")) or {}
    model_config = EmbeatMLPConfig(**model_config_dict)
    model = EmbeatMLP(model_config)
    model_state_dict = checkpoint_dict.get("model_state_dict", None)
    if model_state_dict is None or not isinstance(model_state_dict, dict):
        raise KeyError(f"`model_state_dict` is missing in checkpoint file: {checkpoint_path}")
    model.load_state_dict(model_state_dict, strict=strict)
    if device is None:
        target_device = get_device()
    else:
        target_device = torch.device(str(device))
    model = model.to(target_device)
    model.eval()
    return model


# Build model input features from HuggingFace Dataset samples
def build_features(samples: list[dict], torch_device: Optional[torch.device] = None):
    if not samples:
        raise ValueError("HuggingFace Dataset samples should not be empty.")
    key_index = [int(key_to_idx(sample.get("key"))) for sample in samples]
    mode_index = [int(mode_to_idx(sample.get("mode"))) for sample in samples]
    ts_index = [int(time_signature_to_idx(sample.get("time_signature"))) for sample in samples]
    tempo_index = [int(tempo_to_idx(sample.get("tempo"))) for sample in samples]
    dense_list = [build_dense_vector(sample) for sample in samples]
    dense_matrix = np.stack(dense_list, axis=0).astype(np.float32, copy=False)
    features = {
        "key_idx": torch.tensor(key_index, dtype=torch.long, device=torch_device),
        "mode_idx": torch.tensor(mode_index, dtype=torch.long, device=torch_device),
        "ts_idx": torch.tensor(ts_index, dtype=torch.long, device=torch_device),
        "tempo_idx": torch.tensor(tempo_index, dtype=torch.long, device=torch_device),
        "dense": torch.from_numpy(dense_matrix).to(device=torch_device)
    }
    return features


# Raw cosine similarity in range [-1.0, 1.0]
def cosine_similarity(embedding_a: torch.Tensor, embedding_b: torch.Tensor):
    embedding_a = embedding_a.float()
    embedding_b = embedding_b.float()
    if embedding_a.ndim == 1:
        embedding_a = embedding_a.unsqueeze(0)
    if embedding_b.ndim == 1:
        embedding_b = embedding_b.unsqueeze(0)
    if embedding_a.shape != embedding_b.shape:
        raise ValueError(f"shape mismatch: embedding_a={tuple(embedding_a.shape)} embedding_b={tuple(embedding_b.shape)}")
    embedding_a = functional.normalize(embedding_a, p=2.0, dim=1, eps=1e-12)
    embedding_b = functional.normalize(embedding_b, p=2.0, dim=1, eps=1e-12)
    raw_cosine = (embedding_a * embedding_b).sum(dim=1)
    similarity = float(raw_cosine.squeeze(0).detach().cpu().item())
    return similarity


# Cosine similarity in range [0.0, 1.0]
def cosine_similarity_01(embedding_a: torch.Tensor, embedding_b: torch.Tensor):
    raw_cosine = cosine_similarity(embedding_a, embedding_b)
    similarity_01 = 0.5 * (float(raw_cosine) + 1.0)
    similarity_01 = min(max(0.0, similarity_01), 1.0)
    return similarity_01


# Main inference entry
def infer(sample_a: dict, sample_b: dict, checkpoint_path: str = "train/checkpoints/model.pt", device: Optional[str] = None):
    model = load_model(checkpoint_path=checkpoint_path, device=device, strict=True)
    device = next(model.parameters()).device
    features = build_features(samples=[sample_a, sample_b], torch_device=device)
    with torch.no_grad():
        embedding = model(features)
    embedding = embedding.detach()
    similarity_01 = cosine_similarity_01(embedding[0], embedding[1])
    return similarity_01
