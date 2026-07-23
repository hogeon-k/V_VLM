# PCB C++ Inference Environment

This folder is the G-1 preparation area for C++ inference work. It currently provides a CMake/OpenCV environment check executable and minimal interfaces for future ONNX Runtime and TensorRT implementations. It does not run real YOLO, ONNX, or TensorRT inference yet.

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
| PyTorch model candidates | `yolo11n.pt`, `yolo26n.pt`, `runs/detect/pcb_ablation_scale05_translate05_img640/weights/best.pt` | Confirmed |
| ONNX model | `cpp_inference/models/*.onnx` | Not generated |
| TensorRT engine | `cpp_inference/models/*.engine`, `cpp_inference/models/*.plan` | Not generated |
| data.yaml | `datasets/pcb/data.yaml` | Confirmed |
| Test images | `data/images/*.jpg` | Confirmed |
| Python inference results | `data/result_images/*_yolo_*.jpg` | Confirmed |
| C++ inference results | `cpp_inference/results/` | Created |
| Benchmark results | Undecided | Not confirmed |

Do not copy large model files into this folder. Keep shared model assets in the existing project model locations or generate future ONNX/TensorRT files under `cpp_inference/models/`, which is ignored except for `.gitkeep`.

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

## Development Environment Snapshot

These values were checked from the current Windows PowerShell environment on 2026-07-23.

| Item | Version or status | Check method |
| --- | --- | --- |
| Operating system | Windows 10.0.26200.8875 | `wsl --version` output |
| WSL distribution | Current environment cannot list a distro | `wsl -l -v` failed |
| WSL version | WSL 2.6.2.0, default version output indicates 2 | `wsl --version`, `wsl --status` |
| Python | 3.11.9 | `.\.venv\Scripts\python.exe --version` |
| PyTorch | 2.5.1+cu121 | Python import check |
| PyTorch CUDA | 12.1 | `torch.version.cuda` |
| Ultralytics | 8.4.90 | Python import check |
| ONNX | Not installed | Python import check |
| ONNX Runtime | Not installed | Python import check |
| ONNX Runtime providers | None, package not installed | Python import check |
| C++ compiler | Not installed on Windows PATH | `cl`, `g++ --version` |
| CMake | Not installed on Windows PATH | `cmake --version`, `where.exe cmake` |
| Ninja | Not checked because CMake/compiler are unavailable on PATH | Not run |
| OpenCV C++ | Not confirmed | CMake configure unavailable |
| NVIDIA GPU | NVIDIA GeForce RTX 4060 | `nvidia-smi`, PyTorch |
| NVIDIA driver | 591.86 | `nvidia-smi` |
| CUDA Driver API displayed version | 13.1 | `nvidia-smi` |
| CUDA Toolkit | Not installed on Windows PATH | `nvcc --version` |
| TensorRT Python | Not installed | Python import check |
| TensorRT C++ | Not confirmed; common-path search did not find libraries and broad header search timed out | PowerShell file search |
| trtexec | Not installed on Windows PATH | `trtexec --version`, `where.exe trtexec` |

Notes:

- The CUDA version shown by `nvidia-smi` is the maximum CUDA version supported by the installed driver, not proof that the CUDA Toolkit is installed.
- `nvcc` is not required for PyTorch GPU inference, but CUDA Toolkit headers/libraries are typically needed for C++ TensorRT builds.
- TensorRT Python package availability does not prove C++ headers and libraries are installed. Check `NvInfer.h`, `libnvinfer`, `nvinfer.lib`, and ONNX parser libraries separately.
- Official CUDA/TensorRT compatibility should be verified against NVIDIA's TensorRT support matrix before locking versions.

## Build

Windows PowerShell, when CMake, a C++ compiler, and OpenCV C++ are installed and visible on PATH:

```powershell
cmake -S cpp_inference -B cpp_inference/build
cmake --build cpp_inference/build
.\cpp_inference\build\pcb_inference_app.exe
.\cpp_inference\build\pcb_inference_app.exe --image data\images\01_open_circuit_01.jpg
```

WSL2 Ubuntu, when an Ubuntu distribution and build tools are installed:

```bash
cd /mnt/c/workspace/V_VLM
cmake -S cpp_inference -B cpp_inference/build -G Ninja
cmake --build cpp_inference/build
./cpp_inference/build/pcb_inference_app
./cpp_inference/build/pcb_inference_app --image data/images/01_open_circuit_01.jpg
```

If Ninja is not installed:

```bash
cmake -S cpp_inference -B cpp_inference/build
cmake --build cpp_inference/build
```

Suggested WSL2 Ubuntu setup command, for the user to run only when installation is intended:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build pkg-config libopencv-dev
```

## Future Implementation TODO

- Export YOLO `.pt` weights to ONNX.
- Verify ONNX model input/output shapes.
- Implement Python ONNX Runtime inference and compare with PyTorch.
- Implement C++ ONNX Runtime inference.
- Convert ONNX to TensorRT engine.
- Implement C++ TensorRT inference.
- Benchmark accuracy, preprocessing, inference, postprocessing, and end-to-end latency.
