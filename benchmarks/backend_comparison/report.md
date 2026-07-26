# C++ Inference Backend Final Comparison Report

## 1. Purpose
Compare ONNX Runtime CUDA, Native TensorRT FP32, and Native TensorRT FP16 using existing benchmark result files only.

## 2. Compared Backends

| Backend | Precision | Images | Detections | Inference mean ms | End-to-end mean ms | Failed images | Validation mismatches |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CUDA | FP32 | 21 | 79 | 8.00144 | 18.8403 | 0 | 0 |
| Native TensorRT FP32 | FP32 | 21 | 80 | 3.14165 | 16.7106 | 0 | 0 |
| Native TensorRT FP16 | FP16 | 21 | 80 | 2.05774 | 15.6929 | 0 | 0 |

## 3. Performance Speedups

| Comparison | Inference speedup | Inference reduction | End-to-end speedup | End-to-end reduction |
| --- | ---: | ---: | ---: | ---: |
| ORT CUDA vs TensorRT FP32 | 2.54689x | 60.7364% | 1.12744x | 11.3038% |
| ORT CUDA vs TensorRT FP16 | 3.88845x | 74.2828% | 1.20056x | 16.7055% |
| TensorRT FP32 vs FP16 | 1.52675x | 34.5012% | 1.06485x | 6.09021% |

## 4. Detection Validation

Confidence threshold boundary margin: threshold=0.15, margin=0.001.

| Comparison | Status | PASS | Warning | Boundary warning | FAIL | Structural failures | Max confidence diff | Max bbox diff | Min IoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ONNX Runtime CUDA vs Native TensorRT FP32 | NUMERICAL_WARNING | 13 | 8 | 1 | 0 | 0 | 0.001613 | 0.046386 | 0.998564 |
| ONNX Runtime CUDA vs Native TensorRT FP16 | NUMERICAL_WARNING | 4 | 17 | 1 | 0 | 0 | 0.00454 | 0.115478 | 0.995353 |
| Native TensorRT FP32 vs Native TensorRT FP16 | NUMERICAL_WARNING | 4 | 17 | 0 | 0 | 0 | 0.004622 | 0.099975 | 0.995534 |

## 5. Recommendation

Recommended backend: Native TensorRT FP16
Recommendation status: recommended_with_numerical_warnings

Native TensorRT FP16 had the lowest mean end-to-end latency, with no structural detection failures. Remaining differences are limited to confidence-threshold boundary detections or practical numerical confidence tolerance.

Threshold-boundary unmatched detections are treated as `NUMERICAL_WARNING` only when every unmatched detection is within the configured confidence boundary. Class mismatches, bbox mismatches, and unmatched detections outside that boundary remain `FAIL`.

## 6. Limitations

- This report uses previously generated benchmark files and does not rerun inference.
- ONNX Runtime CUDA end-to-end p95 is computed from timing_runs.csv because summary.json does not store it directly.
