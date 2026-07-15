def calculate_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * (precision * recall) / (precision + recall), 4)

def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """Computes standard classification evaluation metrics."""
    accuracy = round((tp + tn) / (tp + tn + fp + fn), 4) if (tp + tn + fp + fn) > 0 else 0.0
    precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = calculate_f1(precision, recall)
    
    fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr
    }
