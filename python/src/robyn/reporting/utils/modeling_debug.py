import os
import json
from datetime import datetime


def debug_model_metrics(self, X, y, lambda_info, metrics, iteration=None):
    """Log debug metrics to JSON file every 10 iterations"""
    # if iteration is None or iteration == 1 or iteration % 10 == 0:
    debug_info = {
        "iteration": metrics.get("iterNG", iteration),
        "nrmse_train": metrics.get("nrmse_train"),
        "nrmse_val": metrics.get("nrmse_val"),
        "nrmse_test": metrics.get("nrmse_test"),
        "nrmse": metrics.get("nrmse"),
        "rsq_train": metrics.get("rsq_train"),
        "rsq_val": metrics.get("rsq_val"),
        "rsq_test": metrics.get("rsq_test"),
        "lambda": metrics.get("lambda"),
        "lambda_max": metrics.get("lambda_max"),
        "lambda_hp": metrics.get("lambda_hp"),
        "decomp_rssd": metrics.get("decomp_rssd"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Read existing logs if file exists
    json_path = os.path.join(
        os.getcwd(),
        "/Users/yijuilee/robynpy_release_reviews/Robyn/python/src/robyn/debug/March4_2025/python_debug_model_data.json",
    )
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            existing_logs = json.load(f)
    else:
        existing_logs = {}

    # Use iteration number as key if available, otherwise use sequential number
    key = str(iteration) if iteration is not None else str(len(existing_logs) + 1)
    existing_logs[key] = debug_info

    # Write updated logs
    with open(json_path, "w") as f:
        json.dump(existing_logs, f, indent=2)
