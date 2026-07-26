# V_VLM

PySide6 기반 PCB Vision Inspection 데스크톱 프로젝트입니다. PCB 이미지를 입력받아 YOLO로 불량 위치와 유형을 탐지하고, NG 결과에 대해서는 Ollama 기반 VLM 분석을 수행한 뒤 검사 이력과 통계를 SQLite에 저장합니다.

YOLO + Ollama VLM 터미널 검사 가이드는 [docs/yolo_vlm_terminal.md](C:/workspace/V_VLM/docs/yolo_vlm_terminal.md)를 참고하세요.

## 주요 기능

- PCB 이미지 검사 화면
- YOLO Bounding Box 결과 이미지 생성
- Ollama VLM 기반 NG 이미지 분석
- SQLite 검사 이력 저장 및 상세 조회
- 검사 이력 삭제
- 검사 통계 화면
- 시스템 상태 화면
- 실행 로그 표시

## 기술 스택

- Python
- PySide6
- OpenCV
- Pillow
- Ultralytics YOLO
- Torch / torchvision
- SQLite
- pandas / numpy
- pytest
- PyInstaller
- Ollama VLM

## 폴더 구조

- `config/`: 프로젝트 경로와 공통 설정
- `view/`: PySide6 화면 구성
- `viewmodel/`: 화면 상태와 Service 연결
- `model/`: 검사 결과 dataclass
- `service/`: 검사, YOLO, VLM, 통계, 상태 확인 로직
- `repository/`: SQLite DB 연결과 저장소
- `yolo/`: YOLO 모델 로딩과 탐지
- `vlm/`: Ollama VLM 클라이언트, 프롬프트, 응답 파서
- `image_processing/`: 이미지 로딩, 전처리, Bounding Box 렌더링
- `data/input_images/`: 입력 이미지 보관 위치
- `data/result_images/`: 검사 결과 이미지 보관 위치
- `models/`: 로컬 YOLO 모델 파일 위치
- `database/`: SQLite DB 파일 위치
- `logs/`: 실행 로그 위치
- `tests/`: pytest 테스트
- `tools/`: 데이터 변환 및 보조 스크립트

## 개발 환경 설정

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 실행

```powershell
.\.venv\Scripts\python.exe main.py
```

## Ollama VLM 설정

기본 Ollama Host는 `vlm.vlm_client.VlmClient` 설정을 사용합니다.

예:

```text
http://127.0.0.1:11434
```

시스템 상태 화면의 VLM 상태는 설정값이나 모델명 문자열만으로 판단하지 않습니다. 상태 새로고침 시 실제 Ollama 서버에 다음 요청을 보내 확인합니다.

```http
GET /api/tags
```

판정 기준:

- `연결됨`: Ollama 서버 연결 성공, HTTP 정상 응답, 설정된 VLM 모델이 설치 목록에 있음
- `모델 없음`: Ollama 서버는 연결되지만 설정된 VLM 모델이 설치 목록에 없음
- `연결 실패`: connection refused, timeout, 네트워크 오류 등으로 서버 확인 실패
- `응답 오류`: HTTP 오류, JSON 파싱 실패, 잘못된 응답 구조

Ollama를 종료한 뒤 시스템 화면에서 상태 새로고침을 누르면 VLM 상태가 `연결 실패`로 표시되어야 합니다.

## 검사 이력 DB 구조

검사 이력은 SQLite에 저장하며 기본 DB 파일은 `database/inspection_results.sqlite3`입니다. 스키마는 [repository/schema.sql](C:/workspace/V_VLM/repository/schema.sql)에 정의되어 있습니다.

주요 테이블:

- `inspections`: 검사 이력 메인 테이블
- `defects`: 검사별 YOLO 탐지 결과와 불량 상세 테이블

`inspections` 주요 컬럼:

| 컬럼 | 설명 |
| --- | --- |
| `id` | DB 내부 기본키 |
| `image_name` | 원본 이미지 파일명 |
| `original_image_path` | 원본 이미지 경로 |
| `result_image_path` | Bounding Box 결과 이미지 경로 |
| `status` | 검사 상태 |
| `defect_count` | 탐지된 불량 개수 |
| `vlm_description` | 검사 단위 VLM 분석 결과 |
| `inspected_at` | 검사 시각 |

`defects` 주요 컬럼:

| 컬럼 | 설명 |
| --- | --- |
| `inspection_id` | `inspections.id` 참조 |
| `defect_type` | 불량 유형 |
| `confidence` | YOLO 신뢰도 |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | Bounding Box 좌표 |
| `vlm_description` | 불량 단위 VLM 분석 결과 |

`defects.inspection_id`는 `inspections.id`를 참조하며 `ON DELETE CASCADE`가 적용됩니다. 따라서 검사 이력 1건을 삭제하면 연결된 불량 상세 데이터도 함께 삭제됩니다.

이미지는 DB BLOB로 저장하지 않고 파일 경로만 저장합니다. 실제 이미지 파일은 `data/input_images/`, `data/result_images/` 같은 프로젝트 관리 폴더에 저장합니다.

## 검사 이력 번호 정책

검사 이력 화면의 첫 번째 컬럼은 DB 내부 `id`가 아니라 사용자 표시용 `번호`입니다.

- 가장 먼저 검사한 기록이 `1번`입니다.
- 이후 검사는 검사 시각(`inspected_at`) 순서대로 `2번`, `3번`처럼 표시됩니다.
- 같은 검사 시각이면 DB 내부 `id`가 작은 기록이 먼저입니다.
- 삭제된 기록이 있으면 화면 번호는 남아 있는 기록 기준으로 다시 연속 표시됩니다.
- 상세 조회와 삭제는 화면 번호가 아니라 숨겨 둔 실제 DB `id`로 처리합니다.

즉, 화면 번호는 사용자가 보기 쉬운 순번이고, DB `id`는 내부 식별자입니다.

## Pascal VOC XML 라벨을 YOLO TXT로 변환

Pascal VOC XML 파일을 `data/annotations/` 아래에 두거나 [tools/convert_voc_to_yolo.py](C:/workspace/V_VLM/tools/convert_voc_to_yolo.py) 상단의 `XML_DIR` 값을 수정합니다. 변환된 YOLO TXT 라벨은 기본적으로 `labels/` 폴더에 저장됩니다.

현재 변환 대상 클래스:

| 클래스 번호 | 불량 유형 | XML 이름 |
| --- | --- | --- |
| 0 | Open Circuit | open_circuit |
| 1 | Short | short |
| 2 | Missing Hole | missing_hole |

```powershell
.\.venv\Scripts\python.exe tools\convert_voc_to_yolo.py
```

## YOLO 데이터셋 분할

이미지는 `data/images/`, YOLO TXT 라벨은 `labels/`에 둡니다. 다른 경로를 사용하는 경우 [tools/split_yolo_dataset.py](C:/workspace/V_VLM/tools/split_yolo_dataset.py)의 `IMAGE_DIR`, `LABEL_DIR`, `OUTPUT_DIR` 값을 수정합니다.

기본 분할 비율은 train/val/test = 8:1:1입니다.

```powershell
.\.venv\Scripts\python.exe tools\split_yolo_dataset.py
```

결과는 `datasets/pcb/` 아래에 생성됩니다.

- `datasets/pcb/images/train`
- `datasets/pcb/images/val`
- `datasets/pcb/images/test`
- `datasets/pcb/labels/train`
- `datasets/pcb/labels/val`
- `datasets/pcb/labels/test`
- `datasets/pcb/data.yaml`

## 테스트

전체 테스트:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

문법 검사:

```powershell
.\.venv\Scripts\python.exe -m compileall .
```

이번 시스템 상태 화면 변경과 관련된 주요 테스트:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_ollama_status_service.py tests\test_status_view.py tests\test_app_smoke.py
```

검사 이력 삭제와 번호 표시 관련 테스트:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_inspection_history_deletion.py
```

## 모델 비교

`compare_models.py`는 두 YOLO 모델을 같은 검증 데이터셋에서 평가하고 precision, recall, mAP, confusion matrix 정보를 비교합니다.

```powershell
.\.venv\Scripts\python.exe compare_models.py `
  --model-a runs\detect\pcb_default\weights\best.pt `
  --model-b runs\detect\pcb_custom\weights\best.pt `
  --name-a default `
  --name-b custom `
  --data datasets\pcb\data.yaml `
  --imgsz 960 `
  --conf 0.001 `
  --iou 0.7 `
  --device 0 `
  --split val
```

기본 출력 위치는 `runs/compare/`입니다.

## ONNX 변환 검증 및 평가

현재 ONNX 기준 모델은 `models/best.onnx`이며, PyTorch 원본 모델은 `models/best.pt`입니다. 변환 조건은 고정 입력 `1 x 3 x 960 x 960`, batch size `1`, dynamic shape 미사용, 기본 opset `12`를 기준으로 관리합니다. ONNX simplify 적용 여부는 변환 실행 조건에 따라 별도 기록해야 하며, 이 저장소의 검증 스크립트는 실제 ONNX 파일의 input/output shape, opset, producer, SHA256을 읽어 결과에 남깁니다.

ONNX 모델 유효성 검사와 메타데이터 생성:

```powershell
.\.venv\Scripts\python.exe scripts\validate_onnx_model.py `
  --model models\best.onnx `
  --source-model models\best.pt `
  --data datasets\pcb\data.yaml `
  --output benchmarks\onnx\onnx_validation.json `
  --metadata-output models\model_metadata.json
```

생성되는 `models/model_metadata.json`에는 모델명, 원본 모델, task, input/output 이름과 shape, batch size, dynamic 여부, opset, 클래스 순서, ONNX 파일 크기, SHA256, 생성 시각이 저장됩니다. 클래스 이름은 `datasets/pcb/data.yaml`의 `names` 순서를 우선 사용합니다.

ONNX 단독 평가 및 PyTorch 비교:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_onnx.py `
  --model models\best.onnx `
  --pytorch-model models\best.pt `
  --data datasets\pcb\data.yaml `
  --split test `
  --imgsz 960 `
  --conf 0.001 `
  --iou 0.7 `
  --match-iou 0.5 `
  --device 0 `
  --output benchmarks\onnx
```

평가에서 `--iou`는 NMS IoU이고, `--match-iou`는 GT와 prediction의 정답 매칭 IoU입니다. 매칭은 같은 클래스끼리만 confidence 내림차순의 one-to-one greedy 방식으로 수행합니다. mAP는 IoU `0.50:0.05:0.95` 구간에서 101-point interpolated precision envelope 방식으로 계산합니다. Ultralytics 내부 metric 객체를 그대로 쓰는 방식은 아니므로, 이 제한은 결과 JSON의 `matching.method`에도 기록됩니다.

기본 PASS/WARNING 기준:

| 항목 | 기준 |
| --- | --- |
| `abs(mAP50 difference)` | `<= 0.01` |
| `abs(mAP50-95 difference)` | `<= 0.01` |
| `abs(precision difference)` | `<= 0.02` |
| `abs(recall difference)` | `<= 0.02` |
| `class mismatch count` | `0` |
| `new FP count` | `0` |
| `new FN count` | `0` |
| `average matched box IoU` | `>= 0.99` |

치명적인 실행 오류나 모델 오류는 `FAIL`, 기준 초과는 `WARNING`, 기준 만족은 `PASS`로 기록됩니다. 기준값은 `scripts\evaluate_onnx.py`의 CLI 인자로 조정할 수 있습니다.

주요 결과 파일:

- `benchmarks/onnx/onnx_validation.json`: ONNX checker 결과와 모델 입출력 정보
- `benchmarks/onnx/onnx_metrics.json`: ONNX 전체 및 클래스별 Precision, Recall, F1, mAP, TP/FP/FN
- `benchmarks/onnx/onnx_predictions.json`: 이미지별 ONNX 예측 결과
- `benchmarks/onnx/pytorch_metrics.json`: 동일 조건의 PyTorch 평가 지표
- `benchmarks/onnx/pytorch_predictions.json`: 이미지별 PyTorch 예측 결과
- `benchmarks/onnx/pytorch_vs_onnx.csv`: 이미지별 PyTorch/ONNX 탐지 수와 매칭 요약
- `benchmarks/onnx/final_comparison.json`: 최종 PASS/WARNING/FAIL 판정과 차이 요약
- `benchmarks/onnx/failure_cases/failure_cases.json`: ONNX FP/FN 감사 기록

ONNX Runtime은 `CUDAExecutionProvider`를 우선 사용하고 사용 불가하면 `CPUExecutionProvider`로 fallback합니다. 이 정보는 평가 결과의 `runtime.providers`에 기록됩니다.

## C++ ONNX 단일 이미지 추론

`cpp_inference/`에는 `models/best.onnx`를 C++ ONNX Runtime으로 실행하는 단일 이미지 추론 CLI가 있습니다. Python 기준 구현(`service/onnx_detector.py`)과 동일하게 letterbox, BGR to RGB, HWC to CHW, `float32` 정규화, `[1, 7, 18900]` decode, class-aware NMS, 원본 좌표 복원을 적용합니다.

빌드에는 Python wheel이 아니라 ONNX Runtime C/C++ 배포 패키지가 필요합니다. `ONNXRUNTIME_ROOT`는 `include/onnxruntime_cxx_api.h`, `lib/onnxruntime.lib`, Windows 기준 `bin/onnxruntime.dll`을 포함한 경로여야 합니다.

```powershell
cmake -S cpp_inference -B cpp_inference\build `
  -DCMAKE_BUILD_TYPE=Release `
  -DOpenCV_DIR="C:\path\to\opencv\build" `
  -DONNXRUNTIME_ROOT="C:\path\to\onnxruntime"

cmake --build cpp_inference\build --config Release
```

단일 이미지 실행 예:

```powershell
.\cpp_inference\build\Release\pcb_onnx_infer.exe `
  --model models\best.onnx `
  --metadata models\model_metadata.json `
  --image datasets\pcb\images\test\01_missing_hole_03.jpg `
  --output benchmarks\cpp_onnx\single `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7
```

Python ONNX 기준 결과와 C++ 결과 비교:

```powershell
.\.venv\Scripts\python.exe scripts\write_python_onnx_reference.py `
  --image datasets\pcb\images\test\01_missing_hole_03.jpg

.\.venv\Scripts\python.exe scripts\compare_python_cpp_onnx.py
```

자세한 환경 준비와 산출물 설명은 `cpp_inference/README.md`를 참고하세요.

## 예측 오류 분석

`compare_predictions.py`는 PCB 테스트 이미지와 YOLO TXT 정답 라벨을 직접 매칭해 TP/FP/FN을 계산합니다.

```powershell
.\.venv\Scripts\python.exe compare_predictions.py `
  --model-a models\best.pt `
  --model-b runs\detect\pcb_ablation_scale05\weights\best.pt `
  --name-a existing_best `
  --name-b scale05 `
  --images datasets\pcb\images\test `
  --labels datasets\pcb\labels\test `
  --data datasets\pcb\data.yaml `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --match-iou 0.5 `
  --device 0 `
  --run-name open_circuit_error_analysis
```

결과는 `runs/prediction_compare/<run-name>/` 아래에 생성됩니다.

## 런타임 데이터 주의

다음 파일과 폴더는 사용자 데이터 또는 생성물입니다. Git에 포함하지 않습니다.

- `data/input_images/`
- `data/result_images/`
- `models/`
- `database/`
- `logs/`
- `runs/`
- `build/`
- `dist/`
