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

```powershell
python tools/convert_voc_to_yolo.py
```

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest
```

테스트를 실행할 수 없는 경우에는 최소한 문법 검사를 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m compileall .
```

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
