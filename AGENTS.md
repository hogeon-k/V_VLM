# AGENTS.md

## Project Overview

This repository is an early-stage Python desktop vision project. The dependency set points to a PySide6 GUI, OpenCV/Pillow image handling, Ultralytics/Torch model inference, pandas/numpy data handling, and PyInstaller packaging.

Expected runtime directories:

- `data/input_images/`: user-provided input images, ignored by git except `.gitkeep`
- `data/result_images/`: generated visual results, ignored by git except `.gitkeep`
- `models/`: trained or downloaded model files, ignored by git except `.gitkeep`
- `database/`: local database files, ignored by git except `.gitkeep`
- `logs/`: runtime logs, ignored by git except `.gitkeep`

## Development Environment

- Use Python with the local virtual environment when available: `.venv`.
- Install dependencies from `requirements.txt`.
- Keep heavyweight runtime assets out of git. Do not commit model weights, generated images, local databases, logs, PyInstaller output, or virtual environments.
- VLM-specific dependencies are not finalized yet. Add them only after the model/provider choice is clear.

Typical setup on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Coding Guidelines

- Prefer small, focused Python modules with clear boundaries between GUI, image processing, model inference, persistence, and packaging code.
- Keep GUI code responsive. Long-running image/model work should not block the PySide6 main thread.
- Keep model loading explicit and reusable. Avoid reloading large models for every image.
- Use `pathlib.Path` for filesystem paths.
- Store generated outputs under `data/result_images/`, logs under `logs/`, and local model assets under `models/`.
- Do not hard-code absolute local paths. Resolve paths relative to the project root or through configuration.
- Treat input images and local database contents as user data. Avoid destructive operations unless the user explicitly asks for them.
- Prefer structured parsing and serialization libraries over ad hoc text manipulation.

## Testing and Validation

There is no test suite yet. When adding non-trivial logic, add focused tests where practical.

Before handing off changes, run the strongest applicable checks available:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

If tests do not exist or pytest is not installed, at minimum run syntax checks on changed Python files:

```powershell
.\.venv\Scripts\python.exe -m compileall .
```

For GUI or vision changes, also do a manual smoke test when possible:

- Launch the app entry point, if one exists.
- Load a representative image from `data/input_images/`.
- Confirm inference or processing completes without freezing the UI.
- Confirm result files are written to `data/result_images/`.
- Confirm no large generated files were staged accidentally.

## Packaging Notes

- PyInstaller output belongs in `build/` and `dist/`; both are ignored.
- Do not commit generated `.spec` files unless the project intentionally standardizes one.
- Packaging should not depend on machine-specific absolute paths.

## Git Hygiene

- Check `git status --short` before and after edits.
- Preserve user changes. Do not revert, delete, or overwrite unrelated work.
- Keep commits scoped to the requested task.
- Do not stage ignored runtime artifacts from `data/`, `models/`, `database/`, or `logs/`.

## Agent Notes

- This project is currently skeletal. Prefer adding lightweight structure only when it directly supports the requested feature.
- When introducing an app entry point, document the command in `README.md`.
- When adding model or VLM integration, document required model files, expected locations, and any external provider configuration without committing secrets.
