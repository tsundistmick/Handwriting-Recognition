import json
import os
from datetime import datetime
from typing import Dict, Any

def save_results_to_json(results: Dict[str, Any], filename: str = "experiment_results.json"):
    
    os.makedirs("results", exist_ok=True)
    filepath = os.path.join("results", filename)
    if "timestamp" not in results:
        results["timestamp"] = datetime.now().isoformat()
    experiments = []
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    experiments = data
                else:
                    experiments = [data]
        except (json.JSONDecodeError, FileNotFoundError):
            experiments = []
    experiments.append(results)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(experiments, f, ensure_ascii=False, indent=2)
    return filepath