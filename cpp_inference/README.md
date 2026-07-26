# PCB C++ ONNX Runtime Single Image Inference

This folder contains the C++17 single-image ONNX Runtime inference path for `models/best.onnx`.

It mirrors the Python ONNX implementation in `service/onnx_detector.py`:

- OpenCV BGR image load
- Ultralytics-style fixed `960 x 960` letterbox with padding value `114`
- BGR to RGB
- HWC to CHW
- `float32` normalization to `0..1`
- ONNX Runtime inference
- Native TensorRT engine inference for a single image
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
| Native TensorRT engines | `benchmarks/tensorrt/best_fp32.engine`, `benchmarks/tensorrt/best_fp16.engine` | Expected |
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
  - `bin/onnxruntime_providers_cuda.dll` and `bin/onnxruntime_providers_shared.dll` for CUDA execution
- TensorRT package containing:
  - `include/NvInfer.h`
  - `lib/nvinfer_10.lib`
  - runtime DLLs such as `nvinfer_10.dll` under `lib/`

The Python package under `.venv/Lib/site-packages/onnxruntime` provides the Python binding and runtime DLL, but it does not necessarily provide the C++ header and import library required for a native build.

Current Windows PowerShell PATH check in this environment:

- `cmake`: not found
- `g++`: not found
- `ninja`: not found
- `onnxruntime.dll`: found in the Python wheel
- `onnxruntime_cxx_api.h` and `onnxruntime.lib`: not found in the project

## Build

Windows PowerShell, after CMake, MSVC, OpenCV C++, and ONNX Runtime GPU C/C++ are available:

```powershell
cmake -S cpp_inference -B cpp_inference\build `
  -DCMAKE_BUILD_TYPE=Release `
  -DOpenCV_DIR="C:\path\to\opencv\build" `
  -DOpenCV_ARCH=x64 `
  -DOpenCV_RUNTIME=vc16 `
  -DONNXRUNTIME_ROOT="C:\libs\onnxruntime-win-x64-gpu-1.20.1" `
  -DTENSORRT_ROOT="C:\libs\TensorRT-10.4.0.26"

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
    |-- onnxruntime.dll
    |-- onnxruntime_providers_cuda.dll
    `-- onnxruntime_providers_shared.dll
```

The CMake build copies those three ONNX Runtime DLLs next to `pcb_onnx_infer.exe` after a successful build.
It also links TensorRT with `nvinfer_10.lib` and `CUDA::cudart`, and copies available TensorRT runtime DLLs next to `pcb_onnx_infer.exe`.

## Single Image Inference

CUDA is the default provider. Use `--provider cpu` to keep the previous CPU-only behavior.
The app creates one ONNX Runtime session per process, runs warmup iterations that are excluded from statistics, then records repeated `Session.Run()` timings and separate end-to-end timings.

If CUDA/cuDNN DLLs are not already on PATH, add them before the CUDA run:

```powershell
$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;C:\Program Files\NVIDIA\CUDNN\v9.25\bin\x64;$env:Path"
```

```powershell
.\cpp_inference\build\Release\pcb_onnx_infer.exe `
  --model models\best.onnx `
  --metadata models\model_metadata.json `
  --image datasets\pcb\images\test\01_missing_hole_03.jpg `
  --output benchmarks\cpp_onnx\single `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --provider cuda `
  --warmup 10 `
  --repeat 50
```

CPU-only run:

```powershell
.\cpp_inference\build\Release\pcb_onnx_infer.exe `
  --model models\best.onnx `
  --metadata models\model_metadata.json `
  --image datasets\pcb\images\test\01_missing_hole_03.jpg `
  --output benchmarks\cpp_onnx\single_cpu `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --provider cpu `
  --warmup 10 `
  --repeat 50
```

Outputs:

```text
benchmarks/cpp_onnx/single/
|-- result.json
|-- detections.csv
|-- result.jpg
|-- benchmark.json
`-- benchmark.csv
```

Expected provider lines for a CUDA run:

```text
Available providers: [TensorrtExecutionProvider, CUDAExecutionProvider, CPUExecutionProvider]
Requested provider: CUDAExecutionProvider
CUDA registration: success
CUDA device id: 0
cuDNN convolution algorithm search: HEURISTIC
CPU fallback: enabled by ONNX Runtime after CUDA provider
Provider: CUDAExecutionProvider
Session.Run stats (ms): first=..., min=..., mean=..., median=..., p95=..., max=..., stddev=...
End-to-end total stats (ms): first=..., min=..., mean=..., median=..., p95=..., max=..., stddev=...
Validation mismatches: session_run=0, end_to_end=0
```

If CUDA initialization fails, the program exits with a detailed `Error:` message instead of silently falling back to CPU mode.

For fair CPU/CUDA comparison, compare the `session_run_ms.stats.mean`, `median`, and `p95` values in each `benchmark.json`. The first measured value is reported separately because it may still show residual one-time setup effects even after warmup. `end_to_end_ms` shows the practical full pipeline cost including preprocess and postprocess.

## Native TensorRT Single Image Inference

TensorRT uses the same C++ preprocessing, YOLO output decoding, class-aware NMS,
and bbox restoration as the ONNX path. The native backend only replaces
`Session.Run()` with TensorRT 10 engine execution through
`setTensorAddress()` and `enqueueV3()`.

```powershell
$env:Path += ";C:\libs\TensorRT-10.4.0.26\lib"

.\cpp_inference\build_gpu\Release\pcb_onnx_infer.exe `
  --backend tensorrt `
  --engine benchmarks\tensorrt\best_fp32.engine `
  --engine-label fp32 `
  --image datasets\pcb\images\test\01_missing_hole_03.jpg `
  --conf 0.15 `
  --iou 0.7 `
  --warmup 10 `
  --repeat 50
```

FP16 engine run:

```powershell
.\cpp_inference\build_gpu\Release\pcb_onnx_infer.exe `
  --backend tensorrt `
  --engine benchmarks\tensorrt\best_fp16.engine `
  --engine-label fp16 `
  --image datasets\pcb\images\test\01_missing_hole_03.jpg `
  --conf 0.15 `
  --iou 0.7 `
  --warmup 10 `
  --repeat 50
```

TensorRT output files use the same folder contract as ONNX single-image runs:
`result.json`, `detections.csv`, `result.jpg`, `benchmark.json`, and
`benchmark.csv`. The benchmark JSON records input/output tensor metadata,
H2D mean, D2H mean, TensorRT total mean, and GPU execution mean/median/p95.

## Native TensorRT Batch Benchmark

The TensorRT batch executable reuses the same `TensorRtDetector`,
`ImagePreprocessor`, YOLO postprocessing, detection matching, and timing stats
helpers. One TensorRT runtime, engine, execution context, stream, and buffer set
is created per process and reused for every image.

FP32:

```powershell
.\cpp_inference\build_gpu\Release\pcb_tensorrt_batch_benchmark.exe `
  --engine benchmarks\tensorrt\best_fp32.engine `
  --engine-label fp32 `
  --images datasets\pcb\images\test `
  --output benchmarks\tensorrt\batch_fp32 `
  --metadata models\model_metadata.json `
  --device-id 0 `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --match-iou 0.5 `
  --warmup 10 `
  --repeat 30
```

FP16:

```powershell
.\cpp_inference\build_gpu\Release\pcb_tensorrt_batch_benchmark.exe `
  --engine benchmarks\tensorrt\best_fp16.engine `
  --engine-label fp16 `
  --images datasets\pcb\images\test `
  --output benchmarks\tensorrt\batch_fp16 `
  --metadata models\model_metadata.json `
  --device-id 0 `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --match-iou 0.5 `
  --warmup 10 `
  --repeat 30
```

Outputs:

```text
summary.json
per_image.csv
detections.json
failure_cases/.gitkeep
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

## CPU vs CUDA Batch Benchmark

Run CPU and CUDA over every image in a folder with one CPU session and one CUDA session per process:

```powershell
$env:Path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;C:\Program Files\NVIDIA\CUDNN\v9.25\bin\x64;$env:Path"

.\cpp_inference\build_gpu\Release\pcb_onnx_batch_benchmark.exe `
  --model models\best.onnx `
  --metadata models\model_metadata.json `
  --images datasets\pcb\images\test `
  --output benchmarks\cpp_onnx\cpu_cuda_batch `
  --imgsz 960 `
  --conf 0.15 `
  --iou 0.7 `
  --match-iou 0.5 `
  --warmup 10 `
  --repeat 30 `
  --cudnn-conv-algo-search heuristic `
  --provider-order alternate
```

Main outputs:

```text
summary.json
image_results.csv
timing_runs.csv
environment.json
cpu/predictions/
cuda/predictions/
comparisons/
failure_cases/
```

## CUDA Algorithm Search Baseline

`HEURISTIC` is the default cuDNN convolution algorithm search mode for both
single-image inference and the CPU/CUDA batch benchmark. Omitting
`--cudnn-conv-algo-search` therefore applies `HEURISTIC`.

The CLI accepts `heuristic`, `exhaustive`, and `default` case-insensitively and
records the normalized uppercase value in JSON output.

The modes were compared on an RTX 4060 with ONNX Runtime GPU 1.20.1, CUDA 12.6,
and cuDNN 9.25:

| Mode | CUDA Mean | Median | P95 | Structural Mismatch | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| HEURISTIC | 8.26085 ms | 8.12172 ms | 9.3214 ms | 0 | Final baseline |
| EXHAUSTIVE | 8.43391 ms | 8.21594 ms | 9.73795 ms | 0 | Correct, slightly slower |
| DEFAULT | Not selected | - | - | - | cuDNN fallback mode observed |

HEURISTIC and EXHAUSTIVE produced the same detection counts, classes, bounding
boxes, confidence comparison status, and repeat stability. HEURISTIC had the
lower mean and p95 latency, so it is the final CUDA baseline.

With `DEFAULT`, cuDNN fallback mode was observed for several convolution
operations inside the CUDA Execution Provider, indicating a potential
performance degradation. This does not mean CUDA provider initialization
failed or that execution switched to the CPU provider.
