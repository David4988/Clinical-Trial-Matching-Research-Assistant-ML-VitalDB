# Synthetic Evaluation Report

## Setup
- **Train Patients**: 250
- **Eval Patients**: 250
- **Train Windows**: 36000
- **Eval Windows**: 36000

## Overall Physiological Metrics (Excluding Data Gaps)
| Metric | Value |
|--------|-------|
| Total Windows | 35607 |
| Precision | 0.5578 |
| Recall | 0.2447 |
| F1 Score | 0.3402 |
| False Positive Rate | 0.0539 |

## Detection Delay (Sudden/Adverse Events)
- **Detected Cases**: 66
- **Missed Cases**: 0
- **Mean Delay (mins)**: 0.0 | **Worst Delay (mins)**: 0

## Per-Scenario Metrics
| Scenario | Total Windows | Precision | Recall | F1 | FPR |
|----------|---------------|-----------|--------|----|-----|
| RECOVERY | 4608 | 0.3613 | 0.1289 | 0.19 | 0.1139 |
| STABLE | 9648 | 0.0 | 0.0 | 0.0 | 0.0613 |
| GRADUAL_DETERIORATION | 4608 | 1.0 | 0.1521 | 0.2641 | 0.0 |
| ADVERSE_EVENT | 5616 | 0.5786 | 0.9977 | 0.7324 | 0.0616 |
| DATA_QUALITY_FAILURE | 2775 | 0.0 | 0.0 | 0.0 | 0.04 |
| SUDDEN_DETERIORATION | 3888 | 0.8455 | 0.4802 | 0.6125 | 0.0374 |
| IMPROVING | 4464 | 0.0 | 0.0 | 0.0 | 0.0065 |

## Data Quality Monitoring
- **Total Gap Windows**: 393
- **Flagged Gap Windows**: 14
- **Flag Rate**: 0.0356
