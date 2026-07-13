# YOLO + Ollama VLM Terminal Inspection

`models/best.pt`로 YOLO 탐지를 실행하고, 탐지가 1개 이상인 NG 이미지에 대해서만 Ollama `qwen2.5vl:3b`로 한국어 불량 설명을 생성합니다.

## Image Flow

- YOLO inference uses `imgsz=960`.
- The annotated YOLO result image is saved under `data/result_images/` at the original result resolution.
- Bounding Box locations are calculated in Python, not inferred by the VLM.
- The original image size and each box center are classified into a 3x3 region: top/middle/bottom and left/center/right.
- The VLM is instructed to use the calculated location as-is.
- Immediately before the VLM request, the result image is resized in memory so its longest side is at most 960 px by default.
- Detection crops are created in memory from the original-resolution result image around each YOLO box.
- The crops are combined into one RGB JPEG crop montage, also capped at 960 px by default.
- The VLM receives exactly two RGB JPEG byte images for NG results: the full YOLO result image first, then the crop montage second.
- No temporary image files, crop files, montage files, or extra log images are created for VLM input.
- Detection order is preserved in the crop montage: Detection 1, Detection 2, Detection 3 match the printed YOLO list.
- The full image is used for board-level context, while the crop montage is used to inspect fine defect shape.
- The two-image VLM request can improve defect detail but may increase response time.
- VLM image preparation time and inference time are printed separately.
- If the VLM images are too small to confirm a fine defect shape, the VLM should say that the detail is difficult to confirm from the image.

## Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
ollama pull qwen2.5vl:3b
ollama list
```

Ollama itself is a separate desktop/server program, not a pip package.

## YOLO-Only Test

```powershell
.\.venv\Scripts\python.exe scripts\test_yolo_vlm.py `
  --image data\input_images\test.jpg `
  --model models\best.pt `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.5 `
  --device 0 `
  --skip-vlm
```

## YOLO + VLM Test

```powershell
.\.venv\Scripts\python.exe scripts\test_yolo_vlm.py `
  --image data\input_images\test.jpg `
  --model models\best.pt `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.5 `
  --device 0 `
  --vlm-model qwen2.5vl:3b `
  --ollama-host http://127.0.0.1:11434 `
  --vlm-num-ctx 8192 `
  --vlm-num-predict 512 `
  --vlm-image-size 960 `
  --vlm-crop-montage-size 960 `
  --vlm-crop-padding 192 `
  --vlm-crop-min-size 256 `
  --vlm-crop-max-size 512
```

Use `--vlm-image-size` to lower or raise the maximum side length of the full image sent to the VLM. Use `--vlm-crop-montage-size` for the crop montage maximum side length. The default is `960` for both.

## PySide6 Connection Point

Call `service.inspection_service.InspectionService.inspect()` inside a worker thread. Pass `InspectionResult.status`, `detections`, `result_image_path`, and `vlm_explanation` back to the UI state.
