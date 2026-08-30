import os
import sys
import time
import json
import csv
import tracemalloc
import numpy as np
from typing import Dict, Any, List, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.legitimate_device import LegitimateDevice
from simulator.rogue_device import RogueDevice
from models.hmm import HMMDetector
from models.dynamic_graph import DynamicGraph
from models.weighted_gnn import WeightedGNN
from models.graph_lstm import GraphLSTM
from models.trust_score import TrustScore
from models.risk_fusion import RiskFusion
from qtk.dual_trigger import DualTrigger

def measure_scalability(
    device_counts: List[int] = [4, 8, 16, 32, 64],
    epochs: int = 15,
    trials_per_count: int = 5,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Measures empirical runtime and memory consumption of the Dual-Trigger QTK pipeline
    as the number of devices in the group scales from 4 to 64.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results"
        )
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    print("==================================================")
    print("Running Scalability & Computational Benchmark")
    print("==================================================")

    hmm = HMMDetector()
    fusion = RiskFusion()
    dual_trigger = DualTrigger(delta_inact=5, theta_R=0.65)

    scalability_results = []

    for N in device_counts:
        print(f"Benchmarking group size N = {N} devices across {trials_per_count} trials...")
        trial_graph_times = []
        trial_gnn_times = []
        trial_lstm_times = []
        trial_total_epoch_times = []
        trial_peak_memory_kb = []

        for t in range(trials_per_count):
            graph_lstm = GraphLSTM(beta=0.8, seed=42 + t)
            dynamic_graph = DynamicGraph(beta=0.8)
            weighted_gnn = WeightedGNN(seed=42 + t)

            # Setup N devices
            devices = []
            for i in range(N - 1):
                d = LegitimateDevice(
                    device_id=f"dev_{i}", owner_id="user_0", name=f"Device {i}",
                    device_type="primary" if i == 0 else "linked",
                    profile_name="Student", ip_address=f"172.16.23.{10+i}"
                )
                devices.append(d)
            # 1 Rogue device
            rogue = RogueDevice(
                device_id="rogue_client", owner_id="user_0", name="Rogue Client",
                device_type="linked"
            )
            devices.append(rogue)

            # Warm-up history
            for ep in range(1, 13):
                p_meta = devices[0].get_latest_telemetry()
                for d in devices:
                    d.simulate_epoch(ep, p_meta if d != devices[0] else None)

            # Start profiling timed execution per epoch
            tracemalloc.start()
            epoch_times = []
            graph_times = []
            gnn_times = []
            lstm_times = []
            prev_adj = None

            for ep in range(13, 13 + epochs):
                t_ep_start = time.perf_counter()

                p_meta = devices[0].get_latest_telemetry()
                for d in devices:
                    d.simulate_epoch(ep, p_meta if d != devices[0] else None)

                # Individual HMM
                for d in devices:
                    hmm.predict(d)
                    TrustScore.update(d, d.behavioral_risk, alpha=0.8)

                # 1. Graph Evolution timing
                t_g_start = time.perf_counter()
                metas = [d.telemetry_history[-1] for d in devices]
                adj = dynamic_graph.evolve_adjacency(metas, prev_adj)
                prev_adj = adj
                t_g_end = time.perf_counter()
                graph_times.append((t_g_end - t_g_start) * 1000.0)

                # 2. GNN spatial convolution timing
                t_gnn_start = time.perf_counter()
                feats = np.array([
                    [
                        float(m["session_duration_sec"]) / 600.0,
                        float(m["sync_frequency"]) / 20.0,
                        float(m["message_count_sent"]) / 50.0,
                        float(m.get("ip_changed", 0.0)),
                        float(m.get("tz_changed", 0.0))
                    ] for m in metas
                ])
                H = weighted_gnn.forward(feats, adj)
                t_gnn_end = time.perf_counter()
                gnn_times.append((t_gnn_end - t_gnn_start) * 1000.0)

                # 3. Graph-LSTM temporal autoencoder timing
                t_lstm_start = time.perf_counter()
                hists = [d.telemetry_history for d in devices]
                _, scores = graph_lstm.evaluate_devices(hists, prev_adj)
                for i, d in enumerate(devices):
                    if i < len(scores):
                        d.graph_risk = scores[i]
                t_lstm_end = time.perf_counter()
                lstm_times.append((t_lstm_end - t_lstm_start) * 1000.0)

                # 4. Fusion & Decision
                for d in devices:
                    fusion.predict(d)
                    dual_trigger.dual_trigger_decision(d, ep)

                t_ep_end = time.perf_counter()
                epoch_times.append((t_ep_end - t_ep_start) * 1000.0)

            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            trial_graph_times.append(float(np.mean(graph_times)))
            trial_gnn_times.append(float(np.mean(gnn_times)))
            trial_lstm_times.append(float(np.mean(lstm_times)))
            trial_total_epoch_times.append(float(np.mean(epoch_times)))
            trial_peak_memory_kb.append(float(peak_mem / 1024.0))

        res_entry = {
            "num_devices": N,
            "graph_evolution_time_ms": round(float(np.mean(trial_graph_times)), 3),
            "graph_evolution_std_ms": round(float(np.std(trial_graph_times)), 3),
            "gnn_forward_time_ms": round(float(np.mean(trial_gnn_times)), 3),
            "gnn_forward_std_ms": round(float(np.std(trial_gnn_times)), 3),
            "graph_lstm_time_ms": round(float(np.mean(trial_lstm_times)), 3),
            "graph_lstm_std_ms": round(float(np.std(trial_lstm_times)), 3),
            "total_epoch_time_ms": round(float(np.mean(trial_total_epoch_times)), 3),
            "total_epoch_std_ms": round(float(np.std(trial_total_epoch_times)), 3),
            "peak_memory_kb": round(float(np.mean(trial_peak_memory_kb)), 2),
            "peak_memory_std_kb": round(float(np.std(trial_peak_memory_kb)), 2)
        }
        scalability_results.append(res_entry)

        print(f"  [N={N:2d}] Total Per-Epoch Runtime: {res_entry['total_epoch_time_ms']:6.2f} ms ± {res_entry['total_epoch_std_ms']:.2f} ms")
        print(f"         Graph Time: {res_entry['graph_evolution_time_ms']:6.2f} ms | GNN: {res_entry['gnn_forward_time_ms']:6.2f} ms | LSTM: {res_entry['graph_lstm_time_ms']:6.2f} ms")
        print(f"         Peak Memory: {res_entry['peak_memory_kb']:7.1f} KB")
        print("--------------------------------------------------")

    json_path = os.path.join(raw_dir, "scalability.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scalability_results, f, indent=4)

    csv_path = os.path.join(raw_dir, "scalability.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=scalability_results[0].keys())
        writer.writeheader()
        writer.writerows(scalability_results)

    return {"scalability_benchmark": scalability_results}

if __name__ == "__main__":
    measure_scalability()
