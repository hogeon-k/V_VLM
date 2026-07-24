# G1 ONNX Diagnostic Report

## 1. New FN Target Image
No GT-level new FN was found relative to PyTorch.

## 5. Stage Trace
- Debug summary: `not generated`

## 6. FN Cause
- The diagnostic classification is based on raw decode, confidence filter, class-aware NMS, and final GT matching when debug output is generated.

## 7. Accuracy Impact
- Precision diff: `0.0001657903462649879`
- Recall diff: `0.0`
- mAP50 diff: `4.721452204314858e-06`
- mAP50-95 diff: `1.805521454550929e-05`

## 8. Fix Required
- No threshold or model-output correction was applied in this diagnostic run.
- Recommended evaluation fix: keep restored ONNX boxes as float coordinates for metric matching, then round only for GUI drawing/storage.

## 9. CUDA Provider Failure Cause
- See `cuda_diagnostics.json` for package, provider, PATH, DLL, and session creation details.

## 10. Recommended Environment Fix
- Align ONNX Runtime GPU requirements with the installed CUDA/cuDNN/MSVC runtime, or run both PyTorch and ONNX on CPU for fair speed checks.

## 11. Speed Comparison Validity
- Valid: `False`
- Reason: `PyTorch used cuda while ONNX Runtime used CPUExecutionProvider.`

## 12. G1 Final Judgement
- Result: `PASS`
- Original comparison result: `PASS`

## Metric Snapshot
- PyTorch overall: `{'precision': 0.11290322580645161, 'recall': 0.9746835443037974, 'f1': 0.20236530880420497, 'mAP50': 0.9439066667169586, 'mAP50-95': 0.4485430132021417, 'tp': 77, 'fp': 605, 'fn': 2, 'gt': 79, 'pred': 682}`
- ONNX overall: `{'precision': 0.1130690161527166, 'recall': 0.9746835443037974, 'f1': 0.2026315789473684, 'mAP50': 0.9439113881691629, 'mAP50-95': 0.4485610684166872, 'tp': 77, 'fp': 604, 'fn': 2, 'gt': 79, 'pred': 681}`
