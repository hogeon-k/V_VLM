# Backend Comparison Report

## Execution Environment

- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`

## Models And Engines

- PyTorch: `models\best.pt`
- ONNX: `models\best.onnx`
- TensorRT FP32: `benchmarks\tensorrt\best_fp32.engine`
- TensorRT FP16: `benchmarks\tensorrt\best_fp16.engine`

## Common Conditions

- Images: `datasets\pcb\images\test` (21)
- imgsz/conf/iou/match_iou: `960` / `0.15` / `0.5` / `0.5`
- Warmup/repeat: `5` / `20`
- Device/provider: `0` / `CUDAExecutionProvider`
- Warmup policy: Warm up each backend on the first selected image only.

Backend stage timings are backend-reported values. End-to-end is the Python wall-clock duration. TensorRT startup is reported separately and excluded from steady-state statistics.

## Backend Summary

| Backend | Provider | Precision | Mismatch | Inference mean | End-to-end mean | p95 | Result |
|---|---|---:|---:|---:|---:|---:|---|
| pytorch | cuda:0 | FP32 | 0 | 8.414 ms | 43.340 ms | 52.656 ms | BASELINE |
| onnx_cuda | CUDAExecutionProvider | FP32 | 0 | 8.325 ms | 47.805 ms | 55.188 ms | PASS |
| tensorrt_fp32 | Native TensorRT | FP32 | 1 | 3.384 ms | 73.973 ms | 88.786 ms | FAIL |
| tensorrt_fp16 | Native TensorRT | FP16 | 1 | 2.269 ms | 72.686 ms | 92.623 ms | FAIL |

## Accuracy Summary

| Reference | Target | Images | Mismatch images | FP | FN | Class mismatch | Result |
|---|---|---:|---:|---:|---:|---:|---|
| pytorch | onnx_cuda | 21 | 0 | 0 | 0 | 0 | PASS |
| pytorch | tensorrt_fp32 | 21 | 1 | 1 | 0 | 0 | FAIL |
| pytorch | tensorrt_fp16 | 21 | 1 | 1 | 0 | 0 | FAIL |
| tensorrt_fp32 | tensorrt_fp16 | 21 | 0 | 0 | 0 | 0 | WARNING |

## TensorRT Worker

- tensorrt_fp32: startup `377.644 ms`, first request `67.392 ms`, PID reused `True`, fallbacks `0`.
- tensorrt_fp16: startup `189.188 ms`, first request `63.815 ms`, PID reused `True`, fallbacks `0`.

## Fallbacks And Errors

- No backend initialization errors were recorded.

## Final Conclusion

Final status: **FAIL**
