import pytest
import subprocess
import re
import matplotlib.pyplot as plt

THRESHOLD = 0.96

def extract_metric(output, metric_name):
    
    match = re.search(fr"{metric_name}: ([0-9.]+)", output)
    if match:
        return float(match.group(1))
    return None

@pytest.mark.parametrize("dataset_path, target_column", [("test_data.csv", "diabetes")])
def test_model_performance(mocker, dataset_path, target_column):
        
    mocker.patch.object(plt, "show")

    
    result = subprocess.run(
        ["python", "ex3_classification.py"],  
        text=True,
        capture_output=True,
        input=f"{dataset_path}\n{target_column}\n"  
    )
    
    output = result.stdout

    
    best_accuracy = extract_metric(output, "Best Accuracy Score")
    best_auc = extract_metric(output, "Best AUC Score")

    
    assert best_accuracy is not None, "Failed to extract Accuracy Score."
    assert best_auc is not None, "Failed to extract AUC Score."

    
    assert best_accuracy >= THRESHOLD, f"Accuracy {best_accuracy:.4f} is below {THRESHOLD}"
    assert best_auc >= THRESHOLD, f"AUC {best_auc:.4f} is below {THRESHOLD}"
