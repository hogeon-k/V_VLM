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

- Images: `datasets\pcb\images\test` (3)
- imgsz/conf/iou/match_iou: `960` / `0.15` / `0.5` / `0.5`
- Warmup/repeat: `2` / `3`
- Device/provider: `0` / `CUDAExecutionProvider`
- Warmup policy: Warm up each backend on the first selected image only.

Backend stage timings are backend-reported values. End-to-end is the Python wall-clock duration. TensorRT startup is reported separately and excluded from steady-state statistics.

## Backend Summary

| Backend | Provider | Precision | Mismatch | Inference mean | End-to-end mean | p95 | Result |
|---|---|---:|---:|---:|---:|---:|---|
| pytorch | cuda:0 | FP32 | 0 | 9.292 ms | 45.221 ms | 49.151 ms | BASELINE |
| onnx_cuda | CUDAExecutionProvider | FP32 | 0 | 9.345 ms | 56.448 ms | 65.889 ms | PASS |
| tensorrt_fp32 | Native TensorRT | FP32 | 0 | 3.406 ms | 68.423 ms | 70.258 ms | PASS |
| tensorrt_fp16 | Native TensorRT | FP16 | 0 | 2.480 ms | 65.915 ms | 67.939 ms | WARNING |

## Accuracy Summary

| Reference | Target | Images | Mismatch images | FP | FN | Class mismatch | Result |
|---|---|---:|---:|---:|---:|---:|---|
| pytorch | onnx_cuda | 3 | 0 | 0 | 0 | 0 | PASS |
| pytorch | tensorrt_fp32 | 3 | 0 | 0 | 0 | 0 | PASS |
| pytorch | tensorrt_fp16 | 3 | 0 | 0 | 0 | 0 | WARNING |
| tensorrt_fp32 | tensorrt_fp16 | 3 | 0 | 0 | 0 | 0 | WARNING |

## TensorRT Worker

- tensorrt_fp32: startup `422.258 ms`, first request `67.874 ms`, PID reused `True`, fallbacks `0`.
- tensorrt_fp16: startup `198.559 ms`, first request `66.214 ms`, PID reused `True`, fallbacks `0`.

## Fallbacks And Errors

- No backend initialization errors were recorded.

## Final Conclusion

Final status: **WARNING**
