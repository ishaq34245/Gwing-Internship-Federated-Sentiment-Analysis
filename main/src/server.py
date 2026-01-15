# src/server.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Parameters
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import csv

from src.utils import MODEL_NAME, get_numpy_parameters, set_model_params_from_numpy, detect_num_labels

# -----------------------------
# Server hyperparameters
# -----------------------------
NUM_ROUNDS = 75  # more rounds for better learning
MIN_AVAILABLE_CLIENTS = 3
MIN_FIT_CLIENTS = 3
MIN_EVAL_CLIENTS = 3

SERVER_ADDRESS = "127.0.0.1:8080"

SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "checkpoints"))
os.makedirs(SAVE_DIR, exist_ok=True)

METRICS_CSV = os.path.join(SAVE_DIR, "training_metrics.csv")


def weighted_metrics_agg(results):
    """Aggregate metrics from clients weighted by number of examples."""
    total_examples = sum(n for n, _ in results)
    if total_examples == 0:
        return {}
    agg = {}
    keys = set().union(*[set(m.keys()) for _, m in results])
    for key in keys:
        agg[key] = sum(n * m.get(key, 0.0) for n, m in results) / total_examples
    return agg


class SavingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg strategy that saves aggregated model + training metrics each round."""

    def __init__(self, server_model: torch.nn.Module, save_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_model = server_model
        self.save_dir = save_dir
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def aggregate_fit(self, server_round, results, failures):
        # Let FedAvg do the normal aggregation
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is None:
            return None

        # -----------------------------
        # Handle (parameters, metrics) tuple from Flower
        # -----------------------------
        if isinstance(aggregated, tuple):
            aggregated_params, aggregated_metrics = aggregated
        else:
            aggregated_params, aggregated_metrics = aggregated, {}

        # -----------------------------
        # Convert aggregated_params to ndarrays
        # -----------------------------
        try:
            if isinstance(aggregated_params, Parameters):
                ndarrays = parameters_to_ndarrays(aggregated_params)
            else:
                ndarrays = aggregated_params
        except Exception as e:
            print(f"[WARN] Could not convert aggregated params: {e}")
            ndarrays = aggregated_params

        if not isinstance(ndarrays, (list, tuple)):
            ndarrays = list(ndarrays)

        # -----------------------------
        # Apply aggregated weights to the server model and save
        # -----------------------------
        try:
            set_model_params_from_numpy(self.server_model, ndarrays)
            path = os.path.join(self.save_dir, f"aggregated_round_{server_round}.pt")
            torch.save(self.server_model.state_dict(), path)
            print(f"✅ Saved aggregated model at round {server_round} -> {path}")

            # Save "final_model" in HF format after every round (overwrite)
            final_dir = os.path.join(self.save_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            try:
                self.server_model.save_pretrained(final_dir)
                self.tokenizer.save_pretrained(final_dir)
                print(f"✅ Saved HF-style final_model to {final_dir}")
            except Exception as e:
                print(f"[WARN] Could not save HF-style model: {e}")
        except Exception as e:
            print(f"[WARN] Could not set/save aggregated parameters: {e}")

        # -----------------------------
        # Aggregate metrics across clients and write CSV
        # -----------------------------
        try:
            metrics_list = []
            for r in results:
                try:
                    # In most Flower versions: r = (parameters, num_examples, metrics)
                    if isinstance(r, tuple) and len(r) >= 3:
                        _, num_examples, metrics = r
                        metrics_list.append((num_examples, metrics or {}))
                except Exception:
                    pass

            agg_metrics = weighted_metrics_agg(metrics_list)
            if agg_metrics:
                fieldnames = ["round"] + sorted(agg_metrics.keys())
                file_exists = os.path.exists(METRICS_CSV)
                with open(METRICS_CSV, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    row = {"round": int(server_round)}
                    row.update({k: float(v) for k, v in agg_metrics.items()})
                    writer.writerow(row)
                print(f"[SERVER] Appended metrics for round {server_round} -> {METRICS_CSV}")
            else:
                print(f"[SERVER] No metrics to aggregate for round {server_round}")
        except Exception as e:
            print(f"[WARN] Could not write metrics CSV: {e}")

        # Return aggregated in same structure FedAvg expects
        return aggregated


def main():
    logging.info("🚀 Starting server...")

    # Detect number of labels from data/clients
    num_labels = detect_num_labels(
        clients_dir=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "clients"))
    )
    print(f"[SERVER] Detected num_labels={num_labels} from data/clients (if available).")

    # 1) Load server model with detected num_labels
    print(f"[SERVER] Loading base model '{MODEL_NAME}' with num_labels={num_labels}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    server_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    ).to(device)

    # 2) Initial parameters -> Parameters proto
    ndarrays = get_numpy_parameters(server_model)
    initial_parameters = ndarrays_to_parameters(ndarrays)

    strategy = SavingFedAvg(
        server_model=server_model,
        save_dir=SAVE_DIR,
        min_available_clients=MIN_AVAILABLE_CLIENTS,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=MIN_EVAL_CLIENTS,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=weighted_metrics_agg,
        evaluate_metrics_aggregation_fn=weighted_metrics_agg,
    )

    print(f"[SERVER] Starting Flower server at {SERVER_ADDRESS} for {NUM_ROUNDS} rounds...")
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
