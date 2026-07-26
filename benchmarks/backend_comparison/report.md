# 4-Backend 비교 보고서

## 실행 환경

- 측정일: `2026-07-26`
- Python: `3.11.9`
- Platform: `Windows-10-10.0.26200-SP0`
- GPU: `NVIDIA GeForce RTX 4060` (`8188 MiB`)
- 측정 당시 NVIDIA driver: `591.86`
- Device/provider: `0` / `CUDAExecutionProvider`

이 보고서는 실제 저장된 `summary.csv`, `summary.json`, `detection_comparisons.csv`의 결과를 요약한 저장 증거입니다. 성능 수치는 하드웨어, 드라이버, CUDA/TensorRT 버전, 이미지와 실행 설정에 따라 달라질 수 있습니다.

## 비교 대상

- PyTorch CUDA: `models\best.pt`
- ONNX Runtime CUDA: `models\best.onnx`
- TensorRT FP32 persistent worker: `benchmarks\tensorrt\best_fp32.engine`
- TensorRT FP16 persistent worker: `benchmarks\tensorrt\best_fp16.engine`

## 공통 조건

- Images: `datasets\pcb\images\test` (21장)
- imgsz/conf/iou/match_iou: `960` / `0.15` / `0.5` / `0.5`
- Warmup/repeat: `5` / `20`
- Backend sequential execution
- 각 backend의 첫 선택 이미지에서만 warmup하고 통계에서 제외
- TensorRT startup은 steady-state 통계에서 분리
- TensorRT one-shot fallback: `0`

Backend stage timing은 각 backend가 보고한 값이고, host end-to-end는 Python wall-clock 기준입니다.

## 정확도 비교

| Comparison | Reference detections | Target detections | FP | FN | Class mismatch | Result |
|---|---:|---:|---:|---:|---:|---|
| PyTorch vs ONNX Runtime CUDA | 78 | 78 | 0 | 0 | 0 | PASS |
| PyTorch vs TensorRT FP32 | 78 | 79 | 1 | 0 | 0 | FAIL |
| PyTorch vs TensorRT FP16 | 78 | 79 | 1 | 0 | 0 | FAIL |
| TensorRT FP32 vs FP16 | 79 | 79 | 0 | 0 | 0 | WARNING |

PyTorch와 ONNX Runtime CUDA는 21장 모두에서 FP, FN, class mismatch가 없었습니다. TensorRT FP32와 FP16은 `05_short_06.jpg`에서 `short` detection을 각각 하나 더 반환했습니다.

- TensorRT FP32 confidence: `0.150476`
- TensorRT FP16 confidence: `0.150527`
- 적용 threshold: `0.15`

두 confidence가 임계값을 아주 조금 초과했으며, backend별 부동소수점 계산과 전처리 경계 차이가 threshold 부근에서 드러난 사례입니다. 추가 detection을 숨기거나 threshold를 backend별로 변경하지 않았고, 현재 정책에 따라 FP 1건과 FAIL을 그대로 기록했습니다. TensorRT FP32와 FP16끼리는 detection 구성이 같지만 일부 bbox가 strict IoU `0.99` 기준을 벗어나 WARNING입니다.

## 성능 비교

| Backend | Pure inference mean | Host end-to-end mean | Median | p95 |
|---|---:|---:|---:|---:|
| PyTorch CUDA | 8.98 ms | 43.57 ms | 41.08 ms | 52.09 ms |
| ONNX Runtime CUDA | 8.67 ms | 46.84 ms | 44.50 ms | 56.11 ms |
| TensorRT FP32 | 3.40 ms | 71.99 ms | 67.78 ms | 88.85 ms |
| TensorRT FP16 | 2.27 ms | 69.94 ms | 65.55 ms | 84.52 ms |

TensorRT FP16은 FP32보다 순수 inference가 빨랐고, TensorRT의 모델 계산 시간은 PyTorch와 ONNX Runtime보다 짧았습니다. 그러나 현재 통합 구조의 host end-to-end에는 Python-C++ JSONL IPC, 파일 기반 이미지 로딩, 결과 직렬화, 전처리와 후처리가 포함되어 PyTorch와 ONNX Runtime보다 길었습니다. TensorRT가 전체 프로그램에서 항상 빠르다는 의미는 아니며, 실제 적용 판단에는 전체 파이프라인 지연시간을 함께 사용해야 합니다.

## Persistent Worker

- Worker 시작 시 engine을 한 번 deserialize
- stdin/stdout UTF-8 JSON Lines 통신
- TensorRT FP32 startup `231.555 ms`, first request `67.645 ms`, PID `[1124]`
- TensorRT FP16 startup `185.498 ms`, first request `63.567 ms`, PID `[19980]`
- 각 backend의 전체 반복에서 같은 PID 재사용: `True`
- one-shot fallback: `0`
- backend 전환 및 실행 종료 후 worker 정리
- 측정 종료 후 orphan `pcb_onnx_infer.exe` 프로세스: 없음
- backend initialization error: 없음

## 검증

- 전체 테스트: `443 passed`
- 비교 CLI 관련 테스트: `21 passed`
- Python 문법 검사: 통과
- `git diff --check`: 통과

## 최종 결론

최종 상태는 **FAIL**입니다. PyTorch와 ONNX Runtime CUDA의 검출 결과는 일치했지만, TensorRT FP32와 FP16이 threshold 경계에서 각각 FP 1건을 추가 생성했습니다. TensorRT의 순수 inference 개선은 확인됐으나 현재 파일/IPC 기반 통합에서는 host end-to-end 개선으로 이어지지 않았습니다. GUI 비교 기능은 제공하지 않으며 이 결과는 운영 GUI와 분리된 독립 benchmark CLI에서 측정했습니다.
