# src/utils.py
import os
from typing import Optional, List, Any, Tuple

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
import flwr as fl
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters, Parameters

# --------------------------------------------------------------------
# Model checkpoint name (change if you want another pretrained model)
# --------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased"  # good balance speed/accuracy
# If you have more RAM/GPU, try "bert-base-uncased" or "roberta-base"

# --------------------------------------------------------------------
# Simple dataset + collate_fn for text classification CSVs
# Expects CSV with columns: 'text' and 'label' (label = integer)
# --------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        assert len(texts) == len(labels)
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "labels": int(self.labels[idx])}


def collate_fn_builder(tokenizer: AutoTokenizer, max_length: Optional[int] = 128):
    def collate_fn(batch: List[dict]):
        texts = [example["text"] for example in batch]
        labels = torch.tensor([int(example["labels"]) for example in batch], dtype=torch.long)

        enc = tokenizer(
            texts,
            padding=True,       # pad to longest in batch
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc["labels"] = labels
        return enc

    return collate_fn


def load_client_data(client_csv_path: str, batch_size: int = 8, max_length: int = 128, shuffle: bool = True) -> DataLoader:
    if not os.path.exists(client_csv_path):
        raise FileNotFoundError(f"Client CSV not found: {client_csv_path}")

    df = pd.read_csv(client_csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    dataset = TextDataset(texts, labels)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    collate_fn = collate_fn_builder(tokenizer, max_length=max_length)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader


# --------------------------------------------------------------------
# Parameter conversion helpers for Flower compatibility
# --------------------------------------------------------------------
def get_numpy_parameters(model: torch.nn.Module):
    ndarrays = [val.detach().cpu().numpy() for _, val in model.state_dict().items()]
    return ndarrays


def set_model_params_from_numpy(model: torch.nn.Module, params: Any):
    """
    Load parameters into a PyTorch model robustly.
    Accepts flwr.common.Parameters or list/iterable of ndarrays / tensors.
    Skips keys whose shapes don't match and prints warnings.
    """
    if isinstance(params, Parameters):
        ndarrays = parameters_to_ndarrays(params)
    else:
        ndarrays = params

    if not isinstance(ndarrays, (list, tuple)):
        ndarrays = list(ndarrays)

    state_dict = model.state_dict()
    if len(ndarrays) != len(state_dict):
        print(f"[WARN] Received {len(ndarrays)} arrays but model.state_dict has {len(state_dict)} entries.")
        print("We'll try mapping in-order and skip mismatches.")

    new_state_dict = {}
    skipped = []
    for (key, cur_tensor), arr in zip(state_dict.items(), ndarrays):
        # Convert incoming to torch tensor
        if isinstance(arr, torch.Tensor):
            tensor = arr.to(dtype=cur_tensor.dtype)
        else:
            try:
                tensor = torch.tensor(arr, dtype=cur_tensor.dtype)
            except Exception as e:
                skipped.append((key, f"conversion_failed: {e}"))
                continue

        if tuple(tensor.shape) == tuple(cur_tensor.shape):
            new_state_dict[key] = tensor.to(cur_tensor.device)
        else:
            skipped.append((key, tuple(tensor.shape), tuple(cur_tensor.shape)))

    if skipped:
        print("[WARN] Skipped loading parameters due to shape mismatch or conversion issues:")
        for info in skipped:
            if len(info) == 2 and isinstance(info[1], str):
                key, reason = info
                print(f"  - {key}: {reason}")
            else:
                key, from_shape, to_shape = info
                print(f"  - {key}: incoming shape {from_shape} != model shape {to_shape}")

    model.load_state_dict(new_state_dict, strict=False)


# --------------------------------------------------------------------
# Helper: detect number of labels from client CSV files
# Scans data/clients folder for all client_*.csv and returns max_label+1
# --------------------------------------------------------------------
def detect_num_labels(clients_dir: str = "data/clients") -> int:
    """
    Scan client CSVs to find unique labels and return the number of distinct labels.
    Falls back to 2 if nothing found.
    """
    if not os.path.isdir(clients_dir):
        return 2
    labels_set = set()
    for fname in os.listdir(clients_dir):
        if not fname.lower().endswith(".csv"):
            continue
        path = os.path.join(clients_dir, fname)
        try:
            df = pd.read_csv(path)
            if "label" in df.columns:
                labels = df["label"].dropna().astype(int).unique().tolist()
                labels_set.update(labels)
        except Exception:
            continue
    if len(labels_set) == 0:
        return 2
    unique = sorted(labels_set)
    # number of distinct classes
    return len(unique)
