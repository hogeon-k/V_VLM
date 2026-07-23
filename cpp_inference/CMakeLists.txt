cmake_minimum_required(VERSION 3.16)

project(pcb_inference_app VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(pcb_inference_app
    src/main.cpp
    src/detector.cpp
    src/image_preprocessor.cpp
    src/postprocessor.cpp
)

target_include_directories(pcb_inference_app
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pcb_inference_app PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(pcb_inference_app PRIVATE /W4 /permissive-)
else()
    target_compile_options(pcb_inference_app PRIVATE -Wall -Wextra -Wpedantic)
endif()

# TODO: Add ONNX Runtime include and library paths.
# TODO: Add TensorRT and CUDA include and library paths.
