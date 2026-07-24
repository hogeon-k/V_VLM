# PyTorch CUDA vs ONNX Runtime CUDA Benchmark

## 1. Execution Environment
- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- GPU: `NVIDIA GeForce RTX 4060`
- PyTorch: `2.5.1+cu121`
- PyTorch CUDA/cuDNN: `12.1` / `90100`
- ONNX Runtime: `1.20.1`

## 2. Actual Devices
- PyTorch actual device: `cuda:0`
- ONNX actual providers: `['CUDAExecutionProvider']`

## 3. Run Conditions
- Images: `datasets\pcb\images\test`
- Image count: `21`
- imgsz/conf/iou/match_iou: `960` / `0.15` / `0.7` / `0.5`
- warmup/repeat/batch: `10` / `50` / `1`

## 4. Accuracy Equivalence
- Valid: `True`
- PyTorch/ONNX detections: `79` / `79`
- Matched/PT only/ONNX only: `79` / `0` / `0`
- Confidence diff mean/max: `0.0002` / `0.0013`
- BBox IoU mean/min: `0.9999` / `0.9900`

## 5. PyTorch Stage Timing
- preprocess_ms: mean `8.8325` ms, median `8.5053` ms, P95 `14.2202` ms, FPS `113.2187`
- inference_ms: mean `12.3339` ms, median `11.9445` ms, P95 `17.8333` ms, FPS `81.0771`
- postprocess_ms: mean `2.9090` ms, median `2.7639` ms, P95 `4.3205` ms, FPS `343.7603`
- total_ms: mean `64.6950` ms, median `64.1682` ms, P95 `87.1965` ms, FPS `15.4571`

## 6. ONNX Stage Timing
- preprocess_ms: mean `21.5450` ms, median `20.2908` ms, P95 `32.7365` ms, FPS `46.4145`
- inference_ms: mean `10.9450` ms, median `10.0627` ms, P95 `15.0770` ms, FPS `91.3655`
- postprocess_ms: mean `1.1699` ms, median `1.0676` ms, P95 `1.9022` ms, FPS `854.7882`
- total_ms: mean `33.6699` ms, median `31.7596` ms, P95 `48.2651` ms, FPS `29.7002`

## 7. Speedup And FPS
- Total mean speedup: `1.9215x`
- Total median speedup: `2.0204x`
- Total P95 speedup: `1.8066x`
- Inference mean speedup: `1.1269x`
- PyTorch total FPS: `15.4571`
- ONNX total FPS: `29.7002`

## 8. Measurement Validity
- speed_comparison_valid: `True`
- reason: `Both backends used CUDA with the same image set, batch size, thresholds, warm-up, and repeat count.`

## 9. Notes
- Model load, ONNX session creation, DLL registration, PyTorch CUDA preload, file writes, and warm-up runs are excluded from timing statistics.
- PyTorch stage timing uses Ultralytics `result.speed`; total timing is externally measured with CUDA synchronization.
- ONNX Runtime timing measures preprocessing, `session.run`, postprocessing, and total around those three stages.

## 10. Final Conclusion
- Final status: `PASS`
