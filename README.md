# V_VLM

Python desktop Vision AI project for PCB inspection. The planned workflow is to receive PCB images, detect defect locations and types with YOLO, request VLM explanations only for NG images, and store searchable inspection results in SQLite.

This repository currently contains the initial project structure and lightweight class skeletons only. YOLO inference, VLM calls, full PySide6 screens, and database CRUD flows are intentionally left as TODOs.

## Tech Stack

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

VLM-specific dependencies are not finalized yet and should be added only after the provider or local model choice is clear.

## Folder Structure

- `config/`: project-root-relative paths and shared settings
- `view/`: PySide6 widgets and windows
- `viewmodel/`: UI state and service orchestration for MVVM
- `model/`: dataclass-based inspection result structures
- `service/`: application workflows for inspection, image handling, YOLO, VLM, results, and statistics
- `repository/`: SQLite connection management, repositories, and schema
- `yolo/`: YOLO configuration, model loading, and detector boundary
- `vlm/`: provider-neutral VLM client, prompt builder, and response parser boundary
- `image_processing/`: image loading, preprocessing, and bounding box drawing helpers
- `data/input_images/`: user-provided images, ignored except `.gitkeep`
- `data/result_images/`: generated result images, ignored except `.gitkeep`
- `models/`: local model assets such as `best.pt`, ignored except `.gitkeep`
- `database/`: local SQLite database files, ignored except `.gitkeep`
- `logs/`: runtime logs, ignored except `.gitkeep`
- `tests/`: focused scaffold tests

## Development Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe main.py
```

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest
```

If tests are not available, run a syntax check instead:

```powershell
.\.venv\Scripts\python.exe -m compileall .
```

## Planned TODOs

- Implement non-blocking PySide6 inspection screens.
- Load and cache the YOLO model from `models/best.pt`.
- Convert YOLO detections into `DefectInfo` records.
- Generate VLM prompts from NG detections after a provider is selected.
- Parse VLM responses into normalized descriptions.
- Draw bounding boxes and save result images under `data/result_images/`.
- Create SQLite initialization and repository CRUD flows.
- Add history, statistics, and runtime status views.
- Prepare PyInstaller packaging once the entry point stabilizes.
