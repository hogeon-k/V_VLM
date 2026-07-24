# PCB C++ ONNX Runtime Single Image Inference

This folder contains the C++17 single-image ONNX Runtime inference path for `models/best.onnx`.

It mirrors the Python ONNX implementation in `service/onnx_detector.py`:

- OpenCV BGR image load
- Ultralytics-style fixed `960 x 960` letterbox with padding value `114`
- BGR to RGB
- HWC to CHW
- `float32` normalization to `0..1`
- ONNX Runtime inference
- `[1, 7, 18900]` output decode as `[channel][candidate]`
- class-score confidence selection without extra objectness or sigmoid
- class-aware NMS
- float bbox restoration to the original image coordinates
- JSON, CSV, and annotated image output

## Folder Structure

```text
cpp_inference/
|-- CMakeLists.txt
|-- README.md
|-- include/
|   |-- detector.hpp
|   |-- image_preprocessor.hpp
|   |-- inference_result.hpp
|   `-- postprocessor.hpp
|-- src/
|   |-- main.cpp
|   |-- detector.cpp
|   |-- image_preprocessor.cpp
|   `-- postprocessor.cpp
|-- config/
|   `-- classes.txt
|-- models/
|   `-- .gitkeep
|-- results/
|   `-- .gitkeep
`-- tests/
    `-- .gitkeep
```

## Class Order

The class order must match Python YOLO, exported ONNX, TensorRT engines, and C++ postprocessing.

```text
0: open_circuit
1: short
2: missing_hole
```

The same order is recorded in `cpp_inference/config/classes.txt` and matches `datasets/pcb/data.yaml`.

## Model, Data, And Result Paths

| Category | Actual path | Status |
| --- | --- | --- |
| PyTorch model | `models/best.pt` | Confirmed |
| ONNX model | `models/best.onnx` | Confirmed |
| Metadata | `models/model_metadata.json` | Confirmed |
| data.yaml | `datasets/pcb/data.yaml` | Confirmed |
| Test images | `datasets/pcb/images/test/*.jpg` | Confirmed |
| C++ inference output | `benchmarks/cpp_onnx/single/` | Expected |
| Python/C++ comparison output | `benchmarks/cpp_onnx/comparison/` | Expected |

Do not copy large model files into this folder. Keep shared model assets under the existing project `models/` directory.

## Comparison Conditions For Future Work

Use identical settings when comparing Python, ONNX Runtime, TensorRT, and C++ results:

| Item | Value |
| --- | --- |
| PyTorch weights | `models/best.pt`, unless a specific run weight is selected |
| Test images | Same files from `data/images/` or `datasets/pcb/images/test/` |
| imgsz | Match the Python YOLO configuration or CLI argument |
| confidence threshold | Match the Python YOLO configuration or CLI argument |
| NMS IoU threshold | Match the Python YOLO configuration or CLI argument |
| Class order | `open_circuit`, `short`, `missing_hole` |
| Preprocessing | Letterbox resize, normalization, CHW conversion |

## Development Environment

The C++ app needs the C++ development packages, not only Python wheels:

- C++17 compiler, such as MSVC Build Tools
- CMake
- OpenCV C++ package
- ONNX Runtime C/C++ package containing:
  - `include/onnxruntime_cxx_api.h`
  - `lib/onnxruntime.lib` on Windows, or `lib/libonnxruntime.so` on Linux
  - `bin/onnxruntime.dll` on Windows, or runtime shared library equivalent

The Python package under `.venv/Lib/site-packages/onnxruntime` provides the Python binding and runtime DLL, but it does not necessarily provide the C++ header and import library required for a native build.

Current Windows PowerShell PATH check in this environment:

- `cmake`: not found
- `g++`: not found
- `ninja`: not found
- `onnxruntime.dll`: found in the Python wheel
- `onnxruntime_cxx_api.h` and `onnxruntime.lib`: not found in the project

## Build

Windows PowerShell, after CMake, MSVC, OpenCV C++, and ONNX Runtime C/C++ are available:

```powershell
cmake -S cpp_inference -B cpp_inference\build `
  -DCMAKE_BUILD_TYPE=Release `
  -DOpenCV_DIR="C:\path\to\opencv\build" `
  -DONNXRUNTIME_ROOT="C:\path\to\onnxruntime"

cmake --build cpp_inference\build --config Release
```

Expected `ONNXRUNTIME_ROOT` layout:

```text
onnxruntime/
|-- include/
|   `-- onnxruntime_cxx_api.h
|-- lib/
|   `-- onnxruntime.lib
`-- bin/
    `-- onnxruntime.dll
```

## Single Image Inference

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

Outputs:

```text
benchmarks/cpp_onnx/single/
|-- result.json
|-- detections.csv
`-- result.jpg
```

## Python Reference And Comparison

Create the Python ONNX reference for the same image:

```powershell
.\.venv\Scripts\python.exe scripts\write_python_onnx_reference.py `
  --model models\best.onnx `
  --metadata models\model_metadata.json `
  --image datasets\pcb\images\test\01_missing_hole_03.jpg `
  --output benchmarks\cpp_onnx\reference\python_onnx_result.json `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --provider CPUExecutionProvider
```

Compare Python and C++ result JSON files:

```powershell
.\.venv\Scripts\python.exe scripts\compare_python_cpp_onnx.py `
  --python-result benchmarks\cpp_onnx\reference\python_onnx_result.json `
  --cpp-result benchmarks\cpp_onnx\single\result.json `
  --output benchmarks\cpp_onnx\comparison
```
