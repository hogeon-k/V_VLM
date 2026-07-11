# V_VLM

PCB 검사용 Python 데스크톱 Vision AI 프로젝트입니다. 목표 흐름은 PCB 이미지를 입력받고, YOLO로 불량 위치와 종류를 탐지한 뒤, NG 이미지에 대해서만 VLM 설명을 요청하고, 검색 가능한 검사 결과를 SQLite에 저장하는 것입니다.

현재 저장소에는 초기 프로젝트 구조와 가벼운 클래스 골격만 포함되어 있습니다. YOLO 추론, VLM 호출, 전체 PySide6 화면, 데이터베이스 CRUD 흐름은 TODO로 남겨두었습니다.

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

VLM 관련 의존성은 아직 확정하지 않았습니다. 외부 provider 또는 로컬 모델 선택이 명확해진 뒤 추가합니다.

## 폴더 구조

- `config/`: 프로젝트 루트 기준 경로와 공통 설정
- `view/`: PySide6 위젯과 화면
- `viewmodel/`: MVVM 구조의 UI 상태와 서비스 오케스트레이션
- `model/`: dataclass 기반 검사 결과 구조
- `service/`: 검사, 이미지 처리, YOLO, VLM, 결과, 통계 관련 애플리케이션 흐름
- `repository/`: SQLite 연결 관리, 저장소, 스키마
- `yolo/`: YOLO 설정, 모델 로딩, 탐지기 경계
- `vlm/`: provider 중립 VLM 클라이언트, 프롬프트 생성, 응답 파서 경계
- `image_processing/`: 이미지 로딩, 전처리, 바운딩 박스 그리기 헬퍼
- `data/input_images/`: 사용자 입력 이미지, `.gitkeep`만 추적
- `data/result_images/`: 생성된 결과 이미지, `.gitkeep`만 추적
- `models/`: `best.pt` 같은 로컬 모델 파일, `.gitkeep`만 추적
- `database/`: 로컬 SQLite 데이터베이스 파일, `.gitkeep`만 추적
- `logs/`: 런타임 로그, `.gitkeep`만 추적
- `tests/`: 집중 테스트용 스캐폴드

## 개발 환경 설정

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 실행

```powershell
.\.venv\Scripts\python.exe main.py
```

## Pascal VOC XML 라벨을 YOLO TXT로 변환

Pascal VOC XML 파일을 `data/annotations/` 아래에 넣거나, [tools/convert_voc_to_yolo.py](C:/workspace/V_VLM/tools/convert_voc_to_yolo.py) 상단의 `XML_DIR` 값을 수정합니다.
변환된 YOLO TXT 라벨은 기본적으로 `labels/` 폴더에 저장됩니다.
현재 변환 대상은 YOLO 학습에 사용할 3개 불량 유형만 포함합니다. `mouse_bite`, `spur`, `spurious_copper` 등 매핑에 없는 XML object는 경고 메시지를 출력하고 건너뜁니다.

| 클래스 번호 | 불량 유형 | XML 이름 |
| --- | --- | --- |
| 0 | Open Circuit | open_circuit |
| 1 | Short | short |
| 2 | Missing Hole | missing_hole |

```powershell
python tools/convert_voc_to_yolo.py
```

## YOLO 학습용 데이터셋 분할

이미지는 `data/images/`, YOLO TXT 라벨은 `labels/`에 둡니다. 다른 경로를 사용할 경우 [tools/split_yolo_dataset.py](C:/workspace/V_VLM/tools/split_yolo_dataset.py) 상단의 `IMAGE_DIR`, `LABEL_DIR`, `OUTPUT_DIR` 값을 수정합니다.
스크립트는 이미지와 같은 이름의 `.txt` 라벨을 찾아 함께 복사하고, 라벨이 없는 이미지는 경고 후 건너뜁니다. 지원 이미지 확장자는 `jpg`, `jpeg`, `png`입니다. 기본 비율은 train/val/test = 8:1:1이며, 전체 데이터가 60개라면 대략 train 48개, val 6개, test 6개로 나뉩니다.

```powershell
python tools/split_yolo_dataset.py
```

결과는 `datasets/pcb/` 아래에 생성됩니다.

- `datasets/pcb/images/train`
- `datasets/pcb/images/val`
- `datasets/pcb/images/test`
- `datasets/pcb/labels/train`
- `datasets/pcb/labels/val`
- `datasets/pcb/labels/test`
- `datasets/pcb/data.yaml`

생성되는 `data.yaml`은 3개 클래스 기준입니다.

```yaml
path: datasets/pcb
train: images/train
val: images/val
test: images/test
nc: 3
names:
  0: open_circuit
  1: short
  2: missing_hole
```

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest
```

테스트를 실행할 수 없는 경우에는 최소한 문법 검사를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m compileall .
```

## YOLO 모델 검증 비교

`compare_models.py`는 두 YOLO 모델을 동일한 검증 데이터와 평가 옵션으로 `val()` 실행한 뒤, 전체/클래스별 Precision, Recall, mAP, 속도와 confusion matrix 정보를 비교합니다. 실행 전 `runs/detect/*/weights/best.pt` 후보를 출력하며, 비교할 두 가중치는 명시적으로 지정해야 합니다.

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

기본 출력 위치는 `runs/compare/`입니다. 각 모델의 검증 결과와 함께 `model_comparison.csv`, `class_comparison.csv`, `comparison_summary.json`이 생성됩니다. 두 실행의 실제 정답 객체 수가 다르면 스크립트가 비교 무효 경고를 출력합니다.

## 테스트 예측 및 오류 분석 비교

`compare_predictions.py`는 동일한 PCB 테스트 이미지와 YOLO TXT 정답 라벨을 직접 매칭해 TP/FP/FN을 계산합니다. 기본 비교 대상은 `models/best.pt`와 `runs/detect/pcb_ablation_scale05/weights/best.pt`이며, CLI 옵션으로 변경할 수 있습니다.

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

동일 클래스이고 `match_iou` 이상인 한 쌍은 TP입니다. 매칭되지 않은 예측은 FP, 매칭되지 않은 정답은 FN으로 집계합니다. 서로 다른 클래스가 같은 위치에서 `match_iou` 이상으로 겹치면 해당 객체를 FP와 FN으로 각각 집계하고 class confusion으로 기록합니다. 한 GT와 한 prediction은 greedy matching에서 한 번만 연결됩니다.

결과는 기존 실행을 덮어쓰지 않도록 `runs/prediction_compare/<run-name>/` 아래에 생성됩니다.

- `<model>/images/`: 예측 및 TP/FP/FN 주석 이미지
- `<model>/labels/`: confidence를 포함한 YOLO 형식 예측 라벨
- `side_by_side/`: 정답, 모델 A, 모델 B를 나란히 비교하는 이미지
- `image_details.csv`, `class_summary.csv`, `confusion_details.csv`: 기본 비교 결과
- `error_details.csv`: TP/FP/FN별 클래스, confidence, GT/예측 좌표, 매칭 IoU
- `open_circuit_errors.csv`: open_circuit 관련 FP/FN과 검토 보조용 reason hint
- `image_error_summary.csv`: FN, FP, class confusion 우선순위로 정렬된 이미지별 오류 요약
- `error_analysis/<model>/`: `open_circuit_fp`, `open_circuit_fn`, `class_confusion`, `all_errors` 오류 전용 주석 이미지

오류 전용 이미지의 색상은 GT 초록색, prediction 파란색, FP 강조 빨간색, FN 강조 노란색, class confusion 주황색(BGR 기준)입니다. `reason_hint`는 라벨 검토 우선순위를 돕기 위한 자동 분류이며 최종 결함 원인 판정은 아닙니다.

## 예정 작업

- PySide6 검사 화면을 메인 스레드를 막지 않는 구조로 구현
- `models/best.pt`에서 YOLO 모델을 로드하고 캐싱
- YOLO 탐지 결과를 `DefectInfo` 레코드로 변환
- provider 선택 후 NG 탐지 결과 기반 VLM 프롬프트 생성
- VLM 응답을 정규화된 설명으로 파싱
- 바운딩 박스를 그리고 결과 이미지를 `data/result_images/`에 저장
- SQLite 초기화와 repository CRUD 흐름 구현
- 이력, 통계, 런타임 상태 화면 추가
- 진입점이 안정화된 뒤 PyInstaller 패키징 준비
