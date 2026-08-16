# Model Baseline Comparison Experiment

## 1. Training Cohort Details
- **Clean Training Patients**: 102
- **Clean Training Windows**: 14688
- **Scenario Distribution**: {'STABLE': 55, 'IMPROVING': 47}
- **Ground-Truth State Distribution**: {'normal': 7920, 'improving': 4512, 'improved': 2256}

## 2. Baseline vs Cleaned-Training Comparison
| Configuration | Precision | Recall | F1 | FPR | Sudden Det. Recall | Adverse Event Recall |
|---------------|----------:|-------:|---:|----:|-------------------:|---------------------:|
| Current baseline (All scenarios, 0.10) | 0.5578 | 0.2447 | 0.3402 | 0.0539 | 0.4802 | 0.9977 |
| Stable/Improving training, 0.10 | 0.5044 | 0.6741 | 0.5771 | 0.1841 | 0.9991 | 1.0000 |
| Stable/Improving training, 0.15 | 0.4564 | 0.7189 | 0.5584 | 0.2380 | 1.0000 | 1.0000 |
| Stable/Improving training, 0.20 | 0.4194 | 0.7522 | 0.5386 | 0.2894 | 1.0000 | 1.0000 |

## 3. Detailed Results
### Stable/Improving training, 0.10
- **Predicted Anomalous Rows**: 10376 (28.82%)
- **Detected Events**: 66 detected, 0 missed
- **Mean Delay**: 0.0 minutes (Worst: 0 mins)

**Stable/Improving False Positive Inspection:**
- STABLE FPR: 0.1209
- IMPROVING FPR: 0.0840

### Stable/Improving training, 0.15
- **Predicted Anomalous Rows**: 12245 (34.01%)
- **Detected Events**: 66 detected, 0 missed
- **Mean Delay**: 0.0 minutes (Worst: 0 mins)

**Stable/Improving False Positive Inspection:**
- STABLE FPR: 0.1734
- IMPROVING FPR: 0.1196

### Stable/Improving training, 0.20
- **Predicted Anomalous Rows**: 13950 (38.75%)
- **Detected Events**: 66 detected, 0 missed
- **Mean Delay**: 0.0 minutes (Worst: 0 mins)

**Stable/Improving False Positive Inspection:**
- STABLE FPR: 0.2300
- IMPROVING FPR: 0.1503
