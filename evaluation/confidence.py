import numpy as np
from typing import List, Dict, Any

def compute_confidence_intervals(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Computes statistical aggregations: Mean, Standard Deviation, and Confidence Interval error margin:
    CI_margin = z * (std / sqrt(N))
    """
    clean_vals = [v for v in values if v is not None and not np.isnan(v)]
    if not clean_vals:
        return {
            "mean": 0.0,
            "std": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_margin": 0.0,
            "n": 0
        }

    arr = np.array(clean_vals, dtype=np.float64)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    # Exact Student-t critical value for two-tailed 95% CI (df = n - 1)
    # For n=20 (df=19): t = 2.093 (replaces asymptotic normal z=1.96)
    t_crit_table = {
        5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
        15: 2.145, 20: 2.093, 25: 2.064, 30: 2.045, 50: 2.010, 100: 1.984
    }
    t_crit = t_crit_table.get(n, 2.093 if n < 30 else 1.960)
    ci_margin = float(t_crit * (std / np.sqrt(n))) if n > 1 else 0.0

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_lower": round(max(0.0, mean - ci_margin), 4),
        "ci_upper": round(mean + ci_margin, 4),
        "ci_margin": round(ci_margin, 4),
        "n": n
    }

def aggregate_run_metrics(run_metrics_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregates metric dictionaries across multiple independent simulation runs.
    Returns: Dict mapping each metric key to its {mean, std, ci_lower, ci_upper, ci_margin} stats.
    """
    if not run_metrics_list:
        return {}

    all_keys = set()
    for m in run_metrics_list:
        for k, v in m.items():
            if isinstance(v, (int, float)):
                all_keys.add(k)

    aggregated = {}
    for k in sorted(all_keys):
        vals = [float(m[k]) for m in run_metrics_list if k in m and m[k] is not None]
        aggregated[k] = compute_confidence_intervals(vals)

    return aggregated
