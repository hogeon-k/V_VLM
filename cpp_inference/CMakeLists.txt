cmake_minimum_required(VERSION 3.16)

project(pcb_onnx_infer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OpenCV_ARCH "x64" CACHE STRING "OpenCV binary architecture folder, e.g. x64")
set(OpenCV_RUNTIME "vc16" CACHE STRING "OpenCV runtime folder, e.g. vc16")
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")

set(ONNXRUNTIME_ROOT "C:/libs/onnxruntime-win-x64-gpu-1.20.1" CACHE PATH "Path to ONNX Runtime C/C++ package root")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNX Runtime C++ SDK root was not provided.\n"
        "Download or extract the Windows x64 ONNX Runtime C/C++ package and pass:\n"
        "  -DONNXRUNTIME_ROOT=<path>\n"
        "Expected layout:\n"
        "  <path>/include/onnxruntime_cxx_api.h\n"
        "  <path>/lib/onnxruntime.lib\n"
        "  <path>/bin/onnxruntime.dll\n"
        "  <path>/bin/onnxruntime_providers_cuda.dll\n"
        "  <path>/bin/onnxruntime_providers_shared.dll"
    )
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR
        "ONNX Runtime C++ header was not found:\n"
        "  ${ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h\n"
        "Pass the ONNX Runtime C/C++ SDK root with:\n"
        "  -DONNXRUNTIME_ROOT=<path>"
    )
endif()
if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime import library was not found under:\n"
        "  ${ONNXRUNTIME_ROOT}/lib\n"
        "Expected Windows x64 file:\n"
        "  onnxruntime.lib"
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

set(TENSORRT_ROOT "C:/libs/TensorRT-10.4.0.26" CACHE PATH "Path to TensorRT package root")
if(NOT TENSORRT_ROOT)
    message(FATAL_ERROR
        "TensorRT root was not provided.\n"
        "Pass the TensorRT SDK root with:\n"
        "  -DTENSORRT_ROOT=<path>\n"
        "Expected layout:\n"
        "  <path>/include/NvInfer.h\n"
        "  <path>/lib/nvinfer_10.lib"
    )
endif()

set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer_10 nvinfer
    PATHS "${TENSORRT_ROOT}/lib"
    NO_DEFAULT_PATH
)
find_library(TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin_10 nvinfer_plugin
    PATHS "${TENSORRT_ROOT}/lib"
    NO_DEFAULT_PATH
)

if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR
        "TensorRT header was not found:\n"
        "  ${TENSORRT_INCLUDE_DIR}/NvInfer.h\n"
        "Pass the TensorRT SDK root with:\n"
        "  -DTENSORRT_ROOT=<path>"
    )
endif()
if(NOT TENSORRT_NVINFER_LIBRARY)
    message(FATAL_ERROR
        "TensorRT nvinfer import library was not found under:\n"
        "  ${TENSORRT_ROOT}/lib\n"
        "Expected Windows x64 file:\n"
        "  nvinfer_10.lib"
    )
endif()

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer library: ${TENSORRT_NVINFER_LIBRARY}")
if(TENSORRT_PLUGIN_LIBRARY)
    message(STATUS "TensorRT plugin library: ${TENSORRT_PLUGIN_LIBRARY}")
else()
    message(WARNING "TensorRT plugin library was not found; continuing because native engine deserialization does not require the ONNX parser.")
endif()

add_executable(pcb_onnx_infer
    src/main.cpp
    src/detector.cpp
    src/tensorrt_detector.cpp
    src/image_preprocessor.cpp
    src/postprocessor.cpp
)

add_executable(pcb_onnx_batch_benchmark
    src/batch_benchmark_main.cpp
    src/batch_benchmark.cpp
    src/detector.cpp
    src/image_preprocessor.cpp
    src/postprocessor.cpp
)

target_include_directories(pcb_onnx_infer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${TENSORRT_INCLUDE_DIR}
)

target_include_directories(pcb_onnx_batch_benchmark
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(pcb_onnx_infer PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY} ${TENSORRT_NVINFER_LIBRARY} CUDA::cudart)
if(TENSORRT_PLUGIN_LIBRARY)
    target_link_libraries(pcb_onnx_infer PRIVATE ${TENSORRT_PLUGIN_LIBRARY})
endif()
target_link_libraries(pcb_onnx_batch_benchmark PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

if(MSVC)
    target_compile_options(pcb_onnx_infer PRIVATE /W4 /permissive-)
    target_compile_options(pcb_onnx_batch_benchmark PRIVATE /W4 /permissive-)
else()
    target_compile_options(pcb_onnx_infer PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(pcb_onnx_batch_benchmark PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    set(ONNXRUNTIME_RUNTIME_DLLS
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_cuda.dll"
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime_providers_shared.dll"
    )
    set(ONNXRUNTIME_FOUND_RUNTIME_DLLS "")
    foreach(ort_dll IN LISTS ONNXRUNTIME_RUNTIME_DLLS)
        if(EXISTS "${ort_dll}")
            list(APPEND ONNXRUNTIME_FOUND_RUNTIME_DLLS "${ort_dll}")
            add_custom_command(TARGET pcb_onnx_infer POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${ort_dll}"
                        "$<TARGET_FILE_DIR:pcb_onnx_infer>/"
            )
            add_custom_command(TARGET pcb_onnx_batch_benchmark POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${ort_dll}"
                        "$<TARGET_FILE_DIR:pcb_onnx_batch_benchmark>/"
            )
        else()
            message(WARNING "Required ONNX Runtime runtime DLL was not found: ${ort_dll}")
        endif()
    endforeach()
    message(STATUS "ONNX Runtime runtime DLLs: ${ONNXRUNTIME_FOUND_RUNTIME_DLLS}")

    set(TENSORRT_RUNTIME_DLLS
        "${TENSORRT_ROOT}/lib/nvinfer_10.dll"
        "${TENSORRT_ROOT}/lib/nvinfer_plugin_10.dll"
    )
    set(TENSORRT_FOUND_RUNTIME_DLLS "")
    foreach(trt_dll IN LISTS TENSORRT_RUNTIME_DLLS)
        if(EXISTS "${trt_dll}")
            list(APPEND TENSORRT_FOUND_RUNTIME_DLLS "${trt_dll}")
            add_custom_command(TARGET pcb_onnx_infer POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${trt_dll}"
                        "$<TARGET_FILE_DIR:pcb_onnx_infer>/"
            )
        else()
            message(STATUS "TensorRT runtime DLL was not found or not needed: ${trt_dll}")
        endif()
    endforeach()
    message(STATUS "TensorRT runtime DLLs: ${TENSORRT_FOUND_RUNTIME_DLLS}")

    if(OpenCV_ARCH AND OpenCV_RUNTIME)
        set(OpenCV_BIN_DIR "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin")
        file(GLOB OpenCV_RUNTIME_DLLS "${OpenCV_BIN_DIR}/opencv_world*.dll")
    else()
        file(GLOB OpenCV_RUNTIME_DLLS "${OpenCV_DIR}/x64/vc*/bin/opencv_world*.dll")
        set(OpenCV_BIN_DIR "${OpenCV_DIR}/x64/vc*/bin")
    endif()

    if(OpenCV_RUNTIME_DLLS)
        list(FILTER OpenCV_RUNTIME_DLLS EXCLUDE REGEX "d\\.dll$")
        if(OpenCV_RUNTIME_DLLS)
            foreach(opencv_dll IN LISTS OpenCV_RUNTIME_DLLS)
                add_custom_command(TARGET pcb_onnx_infer POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${opencv_dll}"
                            "$<TARGET_FILE_DIR:pcb_onnx_infer>/"
                )
                add_custom_command(TARGET pcb_onnx_batch_benchmark POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${opencv_dll}"
                            "$<TARGET_FILE_DIR:pcb_onnx_batch_benchmark>/"
                )
            endforeach()
            message(STATUS "OpenCV runtime DLLs: ${OpenCV_RUNTIME_DLLS}")
        else()
            message(WARNING "OpenCV runtime DLLs were not found under ${OpenCV_BIN_DIR}; add that directory to PATH or copy DLLs next to the executables.")
        endif()
    else()
        message(STATUS "OpenCV runtime DLLs were not found automatically; if OpenCV DLLs are not on PATH, copy the required OpenCV runtime DLLs next to the executables.")
    endif()
endif()

enable_testing()

add_executable(pcb_cpp_unit_tests
    tests/test_preprocessing.cpp
    tests/test_postprocessing.cpp
    src/image_preprocessor.cpp
    src/postprocessor.cpp
)

add_executable(pcb_batch_benchmark_tests
    tests/test_batch_benchmark.cpp
    src/batch_benchmark.cpp
)

target_include_directories(pcb_cpp_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(pcb_batch_benchmark_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pcb_cpp_unit_tests PRIVATE ${OpenCV_LIBS})
target_link_libraries(pcb_batch_benchmark_tests PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(pcb_cpp_unit_tests PRIVATE /W4 /permissive-)
    target_compile_options(pcb_batch_benchmark_tests PRIVATE /W4 /permissive-)
else()
    target_compile_options(pcb_cpp_unit_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(pcb_batch_benchmark_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME pcb_cpp_unit_tests COMMAND pcb_cpp_unit_tests)
add_test(NAME pcb_batch_benchmark_tests COMMAND pcb_batch_benchmark_tests)

if(WIN32 AND OpenCV_RUNTIME_DLLS)
    foreach(opencv_dll IN LISTS OpenCV_RUNTIME_DLLS)
        add_custom_command(TARGET pcb_cpp_unit_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${opencv_dll}"
                    "$<TARGET_FILE_DIR:pcb_cpp_unit_tests>/"
        )
        add_custom_command(TARGET pcb_batch_benchmark_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${opencv_dll}"
                    "$<TARGET_FILE_DIR:pcb_batch_benchmark_tests>/"
        )
    endforeach()
endif()
