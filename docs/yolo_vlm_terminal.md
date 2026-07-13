# YOLO + Ollama VLM Terminal Inspection

`models/best.pt`를 사용해 YOLO 탐지를 실행하고, 탐지가 1개 이상인 NG 이미지에 대해서만 Ollama의 `qwen2.5vl:3b`로 한국어 불량 설명을 생성합니다.

YOLO 결과 이미지는 `data/result_images/`에만 직접 저장합니다. Ultralytics의 `runs/detect/predict*` 자동 저장은 `save=False`로 사용하지 않습니다.

## 사전 준비

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5vl:3b
ollama list
```

Ollama 프로그램 자체는 pip 패키지가 아니므로 별도로 설치 및 실행되어 있어야 합니다.

## YOLO 단독 테스트

```powershell
.\.venv\Scripts\python.exe scripts\test_yolo_vlm.py `
  --image data\input_images\test.jpg `
  --model models\best.pt `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --device 0 `
  --skip-vlm
```

## YOLO + VLM 통합 테스트

```powershell
.\.venv\Scripts\python.exe scripts\test_yolo_vlm.py `
  --image data\input_images\test.jpg `
  --model models\best.pt `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --device 0 `
  --vlm-model qwen2.5vl:3b `
  --ollama-host http://127.0.0.1:11434 `
  --vlm-num-ctx 8192 `
  --vlm-num-predict 512
```

## PySide6 연결 위치

PySide6 연결 시에는 `service.inspection_service.InspectionService.inspect()`를 worker thread 안에서 호출하고, 반환된 `InspectionResult.status`, `detections`, `result_image_path`, `vlm_explanation`을 UI 상태로 전달하면 됩니다.
