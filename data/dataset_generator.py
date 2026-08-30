import os
import sys
import json
import csv
import random
import yaml
from typing import Dict, Any, List, Tuple

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.legitimate_device import LegitimateDevice
from simulator.silent_device import SilentDevice
from simulator.rogue_device import RogueDevice
from simulator.mimicry_attacker import MimicryAttacker
from simulator.irregular_legitimate import IrregularLegitimateDevice
from simulator.telemetry_generator import TelemetryGenerator

def generate_simulation_dataset(
    num_runs: int = 20,
    epochs_per_run: int = 30,
    base_seed: int = 42,
    split_ratios: Tuple[float, float, float] = (0.70, 0.15, 0.15),
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generates deterministic, run-level split simulation datasets.
    Ensures complete train / validation / test isolation.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "generated"
        )
    os.makedirs(output_dir, exist_ok=True)

    train_ratio, val_ratio, test_ratio = split_ratios
    num_train = int(num_runs * train_ratio)
    num_val = int(num_runs * val_ratio)
    num_test = num_runs - num_train - num_val

    all_records = []
    split_records = {"train": [], "val": [], "test": []}

    print(f"Generating {num_runs} simulation runs (Train: {num_train}, Val: {num_val}, Test: {num_test})...")

    for run_id in range(num_runs):
        seed = base_seed + run_id
        rng = random.Random(seed)

        if run_id < num_train:
            split_name = "train"
        elif run_id < num_train + num_val:
            split_name = "val"
        else:
            split_name = "test"

        # Create devices for this run
        devices = []

        # 3 Legitimate devices (Phone primary, Laptop linked, Tablet linked)
        phone = LegitimateDevice(
            device_id=f"user0_phone",
            owner_id="user_0",
            name="Pixel Phone",
            device_type="primary",
            profile_name="Student",
            ip_address="172.16.23.10"
        )
        laptop = LegitimateDevice(
            device_id=f"user0_laptop",
            owner_id="user_0",
            name="MacBook Pro",
            device_type="linked",
            profile_name="Student",
            ip_address="172.16.23.20"
        )
        devices.extend([phone, laptop])

        # Include an irregular or silent device depending on run variety
        if run_id % 3 == 0:
            tablet = SilentDevice(
                device_id="user0_silent_tablet",
                owner_id="user_0",
                name="Standby Tablet",
                device_type="linked"
            )
        else:
            tablet = IrregularLegitimateDevice(
                device_id="user0_travel_tablet",
                owner_id="user_0",
                name="Travel Tablet",
                device_type="linked",
                profile_name="Student"
            )
        devices.append(tablet)

        # 1 Rogue device injected at epoch 10
        injection_epoch = 10
        if run_id % 2 == 0:
            rogue = RogueDevice(
                device_id="user0_rogue_terminal",
                owner_id="user_0",
                name="Rogue Terminal",
                device_type="linked",
                attack_mode="stealth_burst"
            )
        else:
            rogue = MimicryAttacker(
                device_id="user0_mimic_client",
                owner_id="user_0",
                name="Mimic Client",
                device_type="linked",
                mimicry_strength="moderate_mimicry"
            )

        active_devices = [phone, laptop, tablet]

        # Simulation timeline loop
        for epoch in range(1, epochs_per_run + 1):
            if epoch == injection_epoch:
                active_devices.append(rogue)
                rogue.update_key(epoch)

            # Get peer metadata for relational correlation
            primary_peer_meta = phone.get_latest_telemetry()

            for dev in active_devices:
                obs = dev.simulate_epoch(
                    current_epoch=epoch,
                    peer_telemetry=primary_peer_meta if dev != phone else None,
                    rng=rng
                )
                obs["run_id"] = run_id
                obs["seed"] = seed
                obs["split"] = split_name

                all_records.append(obs)
                split_records[split_name].append(obs)

    # Save to disk as JSONL and flattened CSV
    metadata = {
        "num_runs": num_runs,
        "epochs_per_run": epochs_per_run,
        "base_seed": base_seed,
        "splits": {
            "train_runs": num_train,
            "val_runs": num_val,
            "test_runs": num_test,
            "train_observations": len(split_records["train"]),
            "val_observations": len(split_records["val"]),
            "test_observations": len(split_records["test"]),
            "total_observations": len(all_records)
        },
        "class_distribution": {
            "legitimate_observations": sum(1 for r in all_records if r["ground_truth_label"] == 0),
            "rogue_observations": sum(1 for r in all_records if r["ground_truth_label"] == 1)
        }
    }

    with open(os.path.join(output_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    for split_name, recs in split_records.items():
        jsonl_path = os.path.join(output_dir, f"{split_name}.jsonl")
        csv_path = os.path.join(output_dir, f"{split_name}.csv")

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        if recs:
            csv_rows = []
            for r in recs:
                c = r["context_telemetry"]
                p = r["protocol_telemetry"]
                row = {
                    "run_id": r["run_id"],
                    "seed": r["seed"],
                    "split": r["split"],
                    "epoch": r["epoch"],
                    "device_id": r["device_id"],
                    "device_type": r["device_type"],
                    "key_update_age": p.get("key_update_age", 0),
                    "performed_key_update": p.get("performed_key_update", False),
                    "session_duration_sec": c.get("session_duration_sec", 0.0),
                    "sync_frequency": c.get("sync_frequency", 0.0),
                    "message_count_sent": c.get("message_count_sent", 0),
                    "network_type": c.get("network_type", ""),
                    "network_ip": c.get("network_ip", ""),
                    "location_country": c.get("location_country", ""),
                    "active_timezone": c.get("active_timezone", ""),
                    "is_vpn": c.get("is_vpn", 0.0),
                    "ip_changed": c.get("ip_changed", 0.0),
                    "tz_changed": c.get("tz_changed", 0.0),
                    "ground_truth_label": r["ground_truth_label"]
                }
                csv_rows.append(row)

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

    print(f"Dataset generated successfully in {output_dir}")
    print(f"Total Observations: {len(all_records)} | Legit: {metadata['class_distribution']['legitimate_observations']} | Rogue: {metadata['class_distribution']['rogue_observations']}")
    return metadata

if __name__ == "__main__":
    generate_simulation_dataset()
