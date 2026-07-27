# V_VLM

PySide6 기반 PCB Vision Inspection 데스크톱 프로젝트입니다. PCB 이미지를 입력받아 YOLO로 불량 위치와 유형을 탐지한 뒤 검사 결과를 먼저 SQLite에 저장합니다. 저장된 NG 검사 이력을 사용자가 선택해 VLM 분석을 요청하면 Ollama 기반 설명을 생성하고 같은 검사 이력에 결과를 갱신합니다.

YOLO + Ollama VLM 터미널 검사 가이드는 [docs/yolo_vlm_terminal.md](C:/workspace/V_VLM/docs/yolo_vlm_terminal.md)를 참고하세요.

## 달성 성과

- PyTorch 모델을 ONNX로 변환하고 ONNX 모델 유효성, 입출력 shape, opset, SHA256 메타데이터 검증
- Python ONNX Runtime과 C++ ONNX Runtime 추론 결과 비교
- Native C++ TensorRT FP32/FP16 추론 backend 구현
- TensorRT engine 재사용을 위한 persistent subprocess worker 구현
- PyTorch CUDA, ONNX Runtime CUDA, TensorRT FP32, TensorRT FP16 4-backend 비교
- TensorRT FP16 순수 추론 평균 2.27 ms 확인
- 파일 기반 Python-C++ 통합 구조에서 host end-to-end 병목 확인
- 2026-07-26 기준 전체 테스트 443 passed 기록

## 주요 기능

- PCB 이미지 검사 화면
- PyTorch와 TensorRT backend의 YOLO Bounding Box 결과 이미지 생성
- Python ONNX Runtime backend의 detection 기반 OK/NG 판정 및 저장
- Ollama VLM 기반 NG 이미지 분석
- SQLite 검사 이력 저장 및 상세 조회
- 검사 이력 삭제
- 검사 통계 화면
- 시스템 상태 화면
- 실행 로그 표시

## 시스템 구조

```text
[PySide6 GUI 기본 흐름]

이미지 폴더 선택
    |
하위 폴더를 포함한 이미지 목록 수집
    |
선택된 YOLO Backend
    +-- YoloDetector / PyTorch
    +-- OnnxDetector / Python ONNX Runtime
    +-- TensorRtDetectorAdapter
             |
             | UTF-8 JSON Lines
             v
        C++ Persistent Worker
             |
        TensorRT Engine
    |
Detection 기반 OK/NG 판정
    |
inspections와 defects를 SQLite에 저장
    |
메인 검사 GUI 갱신
    |
이력 화면에서 저장된 NG 검사 선택 및 VLM 생성
    |
전체 이미지 + 앞 2개 Detection Crop Montage
    |
Ollama VLM /api/chat
    |
JSON 파싱과 구조·Detection 개수·ID 검증
    |
정상 설명 또는 YOLO 기반 Fallback 설명을 inspections에 저장
    |
이력 GUI 갱신

[통합 서비스 경로]

InspectionService.inspect()
    |
YOLO 탐지 -> NG이면 VLM 즉시 실행 -> inspections와 defects 저장
```

현재 GUI의 운영 backend는 PyTorch, Python ONNX Runtime, C++ TensorRT입니다. C++ ONNX Runtime 실행 경로는 단일 이미지 검증과 Python ONNX Runtime 결과 비교를 위한 CLI이며 GUI backend에는 연결되어 있지 않습니다. 4-backend 성능 비교도 운영 GUI가 아니라 독립 benchmark 스크립트에서 수행합니다.

현재 PySide6 메인 GUI는 `InspectionService.inspect()`가 아니라 `inspect_image()`를 호출합니다. `inspect_image()`는 `inspect_yolo_only()`로 위임하므로 메인 검사 중에는 VLM을 자동 호출하지 않고, YOLO 판정과 SQLite 저장을 마친 뒤 GUI를 갱신합니다. `InspectionService.inspect()`에는 YOLO와 VLM을 연속 실행하는 통합 경로가 유지되어 있으며, `scripts/test_yolo_vlm.py`의 CLI 통합 실행과 `tests/test_inspection_service.py`에서 사용됩니다.

이력 화면은 앱 시작 시 검사 기록을 조회하지만 탭 전환만으로 자동 reload하지 않습니다. 새 검사 결과는 메인 검사 완료 후 DB에 저장되며, 이력 화면에서는 검색 또는 reload를 수행해 최신 기록을 조회합니다.

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

### VLM 실행 및 JSON 검증

메인 검사 과정에서는 VLM을 자동 실행하지 않습니다. 이력 화면에서 저장된 NG 검사를 선택하고 `VLM 설명 생성` 버튼을 누르면 별도 작업 스레드가 해당 inspection과 defects를 조회하고 `vlm_status`를 `PROCESSING`으로 갱신한 뒤 Ollama `/api/chat`을 호출합니다.

기본 `full_montage` 모드에서는 전체 이미지와 YOLO detection 순서 기준 앞 2개 영역으로 만든 crop montage를 함께 VLM에 전달합니다. PyTorch와 TensorRT backend에서는 Bounding Box가 표시된 annotated 결과 이미지를 전체 이미지와 montage 생성 기준으로 사용합니다. Python ONNX Runtime backend에서는 annotated 결과 이미지가 없으므로 원본 이미지와 원본 기반 montage를 사용합니다.

Ollama 요청에는 `vlm/response_schema.py`의 JSON Schema를 `format`으로 전달합니다. 최상위 필수 필드는 `final_judgment`, `detections`, `summary`이며, 각 detection에는 `detection_id`, `visual_feature`, `visibility`, `review_required`가 필요합니다. 응답 parser는 JSON 구조, 허용되지 않은 추가 필드, detection 개수, 필수 필드와 타입, detection ID 순서 `1..N`을 다시 검증합니다. 여기서 `N`은 전체 YOLO detection이 아니라 VLM에 전달하도록 제한한 detection 개수이며 기본 최대값은 2입니다.

JSON 파싱 또는 구조 검증에 실패하면 YOLO 결과를 기준으로 fallback 설명을 생성합니다. Ollama 빈 응답, content 오류, `ValueError` 또는 `RuntimeError`도 `VlmService`의 재시도 대상이며, 재시도 후 복구되지 않으면 fallback 설명으로 변환됩니다. 정상 JSON 설명과 fallback 설명은 모두 현재 `COMPLETED` 상태로 저장됩니다. VLM 입력 이미지 누락이나 이미지 준비 오류처럼 `VlmService` 밖으로 예외가 전파된 경우에는 `FAILED` 상태와 오류 메시지를 저장합니다.

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
| `vlm_status` | VLM 상태: `NOT_REQUESTED`, `PROCESSING`, `COMPLETED`, `FAILED` |
| `vlm_description` | 검사 단위 VLM 분석 결과 |
| `vlm_error_message` | VLM 실행 경로 밖으로 전파되어 실패 처리된 예외 메시지 |
| `vlm_updated_at` | VLM 상태 또는 결과를 마지막으로 갱신한 시각 |
| `inspected_at` | 검사 시각 |

`defects` 주요 컬럼:

| 컬럼 | 설명 |
| --- | --- |
| `inspection_id` | `inspections.id` 참조 |
| `defect_type` | 불량 유형 |
| `confidence` | YOLO 신뢰도 |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | Bounding Box 좌표 |
| `vlm_description` | 불량 단위 VLM 분석 결과 |

현재 기본 화면 흐름에서는 NG 검사 단위 VLM 결과를 `inspections.vlm_description`에 저장합니다. `defects.vlm_description`은 스키마와 저장소 코드에 존재하지만, 기본 VLM 생성 흐름에서는 불량별 개별 분석 결과가 아니라 검사 단위 설명을 사용합니다. 이 컬럼은 향후 불량별 분석 저장을 위한 확장 지점입니다.

이력 화면의 VLM 실행은 `inspections.vlm_status`, `vlm_description`, `vlm_error_message`, `vlm_updated_at`을 갱신하며 `defects.vlm_description`은 개별 갱신하지 않습니다.

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

이미지는 `data/images/`, YOLO TXT 라벨은 `labels/`에 둡니다. 현재 저장소의 분할 스크립트는 [tools/stratified_split_yolo_dataset.py](C:/workspace/V_VLM/tools/stratified_split_yolo_dataset.py)이며, 클래스별 대표 라벨을 기준으로 train/val/test를 층화 분할합니다.

현재 데이터셋은 train/val/test = 70:15:15 비율로 계층화 분할합니다.
클래스별 객체 분포를 고려해 각 split에 특정 클래스가 누락되지 않도록 구성했습니다.

```powershell
.\.venv\Scripts\python.exe tools\stratified_split_yolo_dataset.py
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

현재 ONNX 기준 모델은 `models/best.onnx`이며, PyTorch 원본 모델은 `models/best.pt`입니다. `benchmarks/onnx/onnx_validation.json` 기준 실제 ONNX 모델은 opset `17`, 고정 입력 `1 x 3 x 960 x 960`, 출력 `1 x 7 x 18900`, batch size `1`, dynamic shape 미사용입니다. ONNX simplify 적용 여부는 변환 실행 조건에 따라 별도 기록해야 하며, 이 저장소의 검증 스크립트는 실제 ONNX 파일의 input/output shape, opset, producer, SHA256을 읽어 결과에 남깁니다.

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

개별 모델 검증과 backend 비교는 실행 목적에 따라 NMS IoU 설정이 다를 수 있습니다. 예를 들어 ONNX 단독 평가와 prediction 비교 예시는 NMS IoU `0.7`을 사용하고, 4-backend 비교는 공통 조건으로 NMS IoU `0.5`를 사용합니다. backend 간 결과는 같은 명령 내의 같은 공통 조건에서만 비교해야 하며, NMS IoU와 GT/prediction match IoU는 서로 다른 개념입니다.

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

GUI에서 ONNX Runtime을 선택하면 `service/detector_backend_factory.py`가 Python `OnnxDetector`를 생성하므로 운영 검사 경로에 연결됩니다. 현재 `OnnxDetector.detect()`는 원본 이미지 경로와 detection 정보만 반환하고 annotated 결과 이미지를 생성하거나 `annotated_image_path`를 설정하지 않습니다. 따라서 메인 결과 이미지 패널과 이력의 YOLO 결과 이미지 영역은 비어 있으며, 원본 이미지는 메인 화면의 별도 현재 검사 이미지 영역에 표시됩니다. 이는 예외가 아니므로 detection 저장과 OK/NG 판정은 정상 동작합니다. 이력에서 VLM을 실행할 때 `result_image_path`가 없으면 원본 이미지로 fallback하며, backend 공통 annotation 후처리는 현재 없습니다.

## C++ ONNX 단일 이미지 추론

`cpp_inference/`에는 `models/best.onnx`를 C++ ONNX Runtime으로 실행하는 단일 이미지 추론 CLI가 있습니다. Python 기준 구현(`service/onnx_detector.py`)과 동일하게 letterbox, BGR to RGB, HWC to CHW, `float32` 정규화, `[1, 7, 18900]` decode, class-aware NMS, 원본 좌표 복원을 적용합니다.

이 C++ ONNX Runtime CLI는 현재 GUI의 backend factory에는 등록되어 있지 않습니다. standalone 단일 이미지 추론, Python/C++ 결과 비교와 정확도 검증, 배치 benchmark, ONNX Runtime CUDA 성능 측정에 사용합니다.

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

## TensorRT 서비스 연결

Python 검사 서비스는 검증된 C++ Native TensorRT CLI를 persistent subprocess worker로 호출합니다. GUI에서 TensorRT backend를 처음 사용할 때 engine을 한 번 deserialize하고, 이후 검사는 같은 프로세스의 stdin/stdout UTF-8 JSON Lines 프로토콜로 처리합니다. backend, engine, metadata, precision 또는 device ID가 바뀌거나 앱이 종료되면 기존 worker를 정상 종료합니다.

필수 파일:

- `cpp_inference/build_gpu/Release/pcb_onnx_infer.exe`
- `benchmarks/tensorrt/best_fp16.engine`
- `models/model_metadata.json`

위 경로는 `DetectorSettings`의 기본값이며 Qt `QSettings`와 GUI 추론 설정 화면에서 변경할 수 있습니다. TensorRT backend 실행에는 이 파일들뿐 아니라 CUDA runtime, TensorRT, ONNX Runtime, OpenCV 관련 runtime DLL이 필요합니다. 필요한 DLL은 실행 파일 인접 경로나 시스템 `PATH`에서 로드 가능해야 합니다.

단일 검사 CLI 예:

```powershell
.\.venv\Scripts\python.exe scripts\test_yolo_vlm.py `
  --backend tensorrt `
  --image datasets\pcb\images\test\01_missing_hole_03.jpg `
  --tensorrt-executable cpp_inference\build_gpu\Release\pcb_onnx_infer.exe `
  --tensorrt-engine benchmarks\tensorrt\best_fp16.engine `
  --tensorrt-engine-label fp16 `
  --tensorrt-metadata models\model_metadata.json `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --device 0
```

Worker는 시작 시 `ready` handshake를 반환하고, 각 `infer` 요청에 기존 result JSON과 같은 detection/timing schema를 한 줄 JSON으로 반환합니다. stdout은 프로토콜 전용이며 C++ 로그는 stderr로 분리됩니다. 탐지가 0개이면 OK, 탐지가 있으면 NG로 판정하며 YOLO 결과와 annotated image 경로를 SQLite에 먼저 저장합니다. 이후 사용자가 저장된 NG 검사 이력에서 VLM 실행을 요청하면 crop montage와 VLM 분석을 수행하고 같은 검사 이력의 VLM 상태와 설명을 갱신합니다. annotated image 임시 산출물은 기본적으로 정리하며 GUI/DB에서 참조할 결과 이미지만 `data/result_images/`로 복사합니다.

Worker startup, timeout, crash 또는 JSONL protocol 오류는 warning 로그를 남기고 기존 one-shot CLI로 fallback합니다. 이미지 decode 같은 요청 단위 오류는 worker를 종료하거나 one-shot으로 재실행하지 않고 해당 요청의 명시적 오류로 반환합니다. fallback은 `DetectorSettings.tensorrt_fallback_to_oneshot`으로 끌 수 있고, persistent mode 자체는 `tensorrt_use_persistent_worker`로 제어합니다.

2026-07-26 로컬 FP16 검증에서 `01_missing_hole_03.jpg`를 기준으로 one-shot 20회 평균은 467.91 ms, persistent 첫 JSONL 요청은 100.62 ms, 이후 protocol/inference 20회 평균/중앙값/p95는 63.14/62.86/64.99 ms였습니다. worker startup은 188.78 ms였고 요청 통계에서 분리했습니다. GUI adapter와 같은 annotated image 생성/복사까지 포함한 steady-state 20회 평균/중앙값/p95는 139.27/138.33/144.31 ms였습니다. 두 방식 모두 `missing_hole` 3개를 검출했으며 class, confidence(0.001), bbox(1 px) 허용 오차 내 mismatch는 0이었습니다. 이 수치는 현재 개발 PC의 참고값이며 다른 GPU와 시스템에서는 달라질 수 있습니다.

## 4-Backend 정확도 및 성능 비교

운영 GUI에는 benchmark 기능을 넣지 않고 `scripts/compare_all_backends.py`에서 PyTorch CUDA, Python ONNX Runtime CUDA, TensorRT FP32 persistent worker, TensorRT FP16 persistent worker를 순차 비교합니다. 이 분리는 운영 검사 화면의 수명주기와 GPU benchmark의 반복 실행 및 대용량 산출물 생성을 분리하기 위한 것입니다.

비교 대상:

- PyTorch CUDA
- ONNX Runtime CUDA
- TensorRT FP32 persistent worker
- TensorRT FP16 persistent worker

```powershell
.\.venv\Scripts\python.exe scripts\compare_all_backends.py `
  --images datasets\pcb\images\test `
  --pytorch-model models\best.pt `
  --onnx-model models\best.onnx `
  --tensorrt-fp32-engine benchmarks\tensorrt\best_fp32.engine `
  --tensorrt-fp16-engine benchmarks\tensorrt\best_fp16.engine `
  --metadata models\model_metadata.json `
  --output benchmarks\backend_comparison `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.5 `
  --match-iou 0.5 `
  --warmup 5 `
  --repeat 20 `
  --device 0 `
  --provider CUDAExecutionProvider
```

모든 backend에 같은 이미지와 설정을 적용하고 PyTorch 결과를 기준으로 class-aware IoU matching을 수행합니다. 2026-07-26 RTX 4060 8GB Windows/CUDA 환경에서 사용한 공통 조건은 테스트 이미지 21장, `imgsz=960`, `conf=0.15`, NMS `iou=0.5`, `match_iou=0.5`, warmup 5회, 측정 20회입니다. backend는 순차 실행하며 warmup은 각 backend의 첫 선택 이미지에서만 수행하고 통계에서 제외합니다. TensorRT startup은 steady-state와 분리했습니다.

생성 파일은 `summary.json`, `summary.csv`, `per_image_results.csv`, `detection_comparisons.csv`, `timing_samples.csv`, `report.md`이며 mismatch가 있으면 `mismatch_cases/`에 backend별 detection JSON을 기록합니다. 생성 CSV/JSON과 mismatch 산출물은 git에서 제외하고 최종 `report.md`와 `.gitkeep`만 추적합니다.

정확도 비교:

| Comparison | Reference detections | Target detections | FP | FN | Class mismatch | Result |
|---|---:|---:|---:|---:|---:|---|
| PyTorch vs ONNX Runtime CUDA | 78 | 78 | 0 | 0 | 0 | PASS |
| PyTorch vs TensorRT FP32 | 78 | 79 | 1 | 0 | 0 | FAIL |
| PyTorch vs TensorRT FP16 | 78 | 79 | 1 | 0 | 0 | FAIL |
| TensorRT FP32 vs FP16 | 79 | 79 | 0 | 0 | 0 | WARNING |

TensorRT FP32와 FP16은 `05_short_06.jpg`에서 confidence가 임계값 `0.15`를 아주 조금 초과한 `short` detection을 각각 하나 더 생성했습니다. FP32 confidence는 `0.150476`, FP16 confidence는 `0.150527`입니다. 이는 backend별 부동소수점 계산과 전처리 경계 차이가 threshold 부근에서 드러난 사례이며, 현재 판정 정책대로 FP 1건으로 유지해 FAIL로 기록했습니다. FP32와 FP16끼리는 detection 구성이 같지만 strict bbox IoU `0.99` 기준을 일부 벗어나 WARNING입니다.

성능 비교:

| Backend | Pure inference mean | Host end-to-end mean |
|---|---:|---:|
| PyTorch CUDA | 8.98 ms | 43.57 ms |
| ONNX Runtime CUDA | 8.67 ms | 46.84 ms |
| TensorRT FP32 | 3.40 ms | 71.99 ms |
| TensorRT FP16 | 2.27 ms | 69.94 ms |

TensorRT FP16의 순수 inference는 PyTorch CUDA 대비 약 74.7%, ONNX Runtime CUDA 대비 약 73.8% 짧았고, 두 TensorRT engine 모두 모델 계산 자체에서는 PyTorch와 ONNX Runtime보다 빨랐습니다. 반면 현재 통합 구조의 host end-to-end에는 Python에서 C++ worker로 이미지 경로 전달, C++ 이미지 파일 로딩과 디코딩, Python-C++ JSONL IPC, JSON 직렬화와 역직렬화, letterbox 전처리, decode/NMS/좌표 복원 후처리, 결과 이미지 처리 비용이 포함되어 PyTorch와 ONNX Runtime보다 길었습니다. 따라서 TensorRT 적용 여부는 순수 GPU inference뿐 아니라 전체 파이프라인 지연시간을 기준으로 판단해야 합니다. JSONL IPC만을 단독 병목으로 단정하지 않고, 파일 기반 Python-C++ 분리 구조 전체 비용으로 해석합니다. Persistent worker는 engine을 한 번만 deserialize하고 backend별 같은 PID를 재사용했으며, 실제 비교에서 one-shot fallback은 0회였고 종료 후 orphan `pcb_onnx_infer.exe` 프로세스는 없었습니다.

이 결과는 RTX 4060 8GB와 당시 Windows/CUDA 환경의 측정값입니다. 하드웨어, 드라이버, CUDA/TensorRT 버전, 이미지와 설정에 따라 달라질 수 있습니다. GUI는 비교 기능을 제공하지 않으며 운영 검사 흐름과 개발 benchmark를 분리하기 위해 독립 CLI로 실행합니다. 전체 테스트 결과는 2026-07-26 기준 `443 passed`였고, 상세 측정 근거는 `benchmarks/backend_comparison/report.md`에 보존합니다.

기본 backend는 기존 동작과 동일하게 `pytorch`입니다. CLI에서 선택 가능한 backend는 다음과 같습니다.

```powershell
--backend pytorch
--backend onnx
--backend tensorrt
```

GUI에서는 상단 메뉴의 `추론 설정` 화면에서 backend를 선택하고 저장합니다. 설정은 Qt `QSettings`에 저장되므로 개인 로컬 경로가 저장소 파일로 생성되지 않습니다.

예시 설정:

```text
Backend: TensorRT
Executable: cpp_inference/build_gpu/Release/pcb_onnx_infer.exe
Engine: benchmarks/tensorrt/best_fp16.engine
Precision: FP16
Metadata: models/model_metadata.json
Device ID: 0
```

TensorRT를 선택하면 실행 파일, engine, precision, metadata, CUDA device ID 입력이 활성화됩니다. PyTorch 또는 ONNX Runtime을 선택하면 TensorRT 전용 입력은 비활성화됩니다. 저장 시 TensorRT 설정은 다음 조건을 검사합니다.

- 실행 파일 존재 및 `.exe` 확장자
- engine 존재 및 `.engine` 또는 `.plan` 확장자
- metadata 존재 및 `.json` 확장자
- device ID 0 이상
- precision `FP16` 또는 `FP32`

Python 설정 단계의 검증 범위는 위 파일 존재, 확장자, device ID와 engine label입니다. 실제 C++ 실행 단계에서는 CUDA device 설정, engine deserialize, 입력·출력 tensor 개수, 이름, shape와 dtype을 추가로 검증합니다. metadata의 ONNX SHA256과 engine의 직접 대응 관계, 실제 engine 정밀도와 `engine_label`의 완전한 일치는 현재 검증하지 않습니다.

저장한 설정은 다음 검사 시작 때 적용됩니다. 검사 화면에는 현재 적용될 backend가 `추론 Backend: ...` 형태로 표시됩니다. 기본값은 기존 호환성을 위해 `PyTorch`입니다.

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
