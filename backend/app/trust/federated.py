import numpy as np
from typing import List, Dict, Any

class FederatedSimulation:
    @staticmethod
    def run_federated_avg(client_models_transmats: List[np.ndarray]) -> np.ndarray:
        """
        Implements Federated Averaging (FedAvg) on HMM transition matrices.
        Combines decentralized weight states.
        """
        if not client_models_transmats:
            return np.zeros((4, 4))
        
        # Simple coordinate-wise average of transition probabilities
        global_transmat = np.mean(client_models_transmats, axis=0)
        
        # Ensure row probabilities sum up to exactly 1.0 (normalization)
        for i in range(len(global_transmat)):
            row_sum = np.sum(global_transmat[i])
            if row_sum > 0:
                global_transmat[i] = global_transmat[i] / row_sum
                
        return global_transmat

    @staticmethod
    def simulate_training_curves(epochs: int = 10, num_clients: int = 5) -> Dict[str, Any]:
        """
        Generates simulated training metrics comparing:
        1. Centralized Learning (All data pooled in database)
        2. Federated Learning (Decentralized clients, aggregated model)
        3. Local-Only Learning (No cooperation, high variance)
        
        Returns:
            chart_data: Lists of epochs, losses, and accuracies for Plotly plotting.
        """
        epochs_list = list(range(1, epochs + 1))
        
        # Centralized starts high and converges fastest (full data visibility)
        centralized_acc = [0.72 + 0.22 * (1 - np.exp(-0.6 * e)) for e in epochs_list]
        centralized_loss = [0.85 * np.exp(-0.5 * e) for e in epochs_list]
        
        # Federated converges slightly slower due to averaging rounds, but reaches similar final accuracy
        # (while preserving raw metadata privacy locally!)
        federated_acc = [0.65 + 0.28 * (1 - np.exp(-0.4 * e)) for e in epochs_list]
        federated_loss = [0.95 * np.exp(-0.35 * e) for e in epochs_list]
        
        # Local-only fluctuates, gets stuck on local minima, and has lowest overall generalization
        local_acc = [0.55 + 0.25 * (1 - np.exp(-0.25 * e)) + np.random.normal(0, 0.02) for e in epochs_list]
        local_loss = [1.2 * np.exp(-0.2 * e) + np.random.normal(0, 0.03) for e in epochs_list]
        
        # Clamp scores
        local_acc = [max(0.5, min(0.99, a)) for a in local_acc]
        
        return {
            "epochs": epochs_list,
            "centralized": {
                "accuracy": centralized_acc,
                "loss": centralized_loss
            },
            "federated": {
                "accuracy": federated_acc,
                "loss": federated_loss
            },
            "local_only": {
                "accuracy": local_acc,
                "loss": local_loss
            }
        }
