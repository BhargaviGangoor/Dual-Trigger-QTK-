from typing import Dict, Any, List, Optional

def calculate_classification_metrics(
    tp: int,
    fp: int,
    fn: int,
    tn: int
) -> Dict[str, float]:
    """
    Computes standard classification metrics from confusion matrix counts:
    - Precision
    - Recall (Detection Rate)
    - F1-Score
    - False Positive Rate (FPR)
    - Accuracy
    """
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    accuracy = float((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "detection_rate": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

def calculate_qtk_system_metrics(
    quarantined_rogue_epochs: List[int],
    injection_epoch: int,
    total_epochs: int,
    false_quarantined_legit_count: int,
    total_legitimate_devices: int
) -> Dict[str, float]:
    """
    Computes protocol-specific operational metrics:
    - Detection Latency: (quarantine_epoch - injection_epoch) in epochs for caught rogues (0 if caught immediately).
    - QTK Evasion Duration: number of epochs the rogue device remained active in the MLS group before quarantine.
    - False Quarantine Rate (FQR):
        FQR = false_quarantined_legit_count / total_legitimate_devices
        Measures the expected number of false quarantine actions per legitimate device slot.
        In binary evaluation runs without recovery, FQR equals the standard False Positive Rate (FPR = FP / (FP + TN)).
        In multi-epoch lifecycle experiments with key recovery, devices may be quarantined, recovered, and re-quarantined,
        so FQR can exceed 1.0 (e.g., 1.43 indicates an average of 1.43 quarantine interruptions per device over the run).
    """
    latencies = [max(0, ep - injection_epoch) for ep in quarantined_rogue_epochs if ep is not None]
    avg_latency = float(sum(latencies) / len(latencies)) if latencies else float(total_epochs - injection_epoch)

    evasion_durations = []
    for ep in quarantined_rogue_epochs:
        if ep is not None:
            evasion_durations.append(max(0, ep - injection_epoch))
        else:
            evasion_durations.append(total_epochs - injection_epoch)

    avg_evasion = float(sum(evasion_durations) / len(evasion_durations)) if evasion_durations else float(total_epochs - injection_epoch)
    fqr = float(false_quarantined_legit_count / max(1, total_legitimate_devices))

    return {
        "avg_detection_latency": round(avg_latency, 2),
        "avg_evasion_duration": round(avg_evasion, 2),
        "false_quarantine_rate": round(fqr, 4)
    }
