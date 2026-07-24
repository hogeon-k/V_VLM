cmake_minimum_required(VERSION 3.16)

project(pcb_onnx_infer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
set(OpenCV_ARCH "" CACHE STRING "OpenCV binary architecture folder, e.g. x64")
set(OpenCV_RUNTIME "" CACHE STRING "OpenCV runtime folder, e.g. vc16")

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime C/C++ package root")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR
        "ONNX Runtime C++ SDK root was not provided.\n"
        "Download or extract the Windows x64 ONNX Runtime C/C++ package and pass:\n"
        "  -DONNXRUNTIME_ROOT=<path>\n"
        "Expected layout:\n"
        "  <path>/include/onnxruntime_cxx_api.h\n"
        "  <path>/lib/onnxruntime.lib\n"
        "  <path>/bin/onnxruntime.dll"
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

add_executable(pcb_onnx_infer
    src/main.cpp
    src/detector.cpp
    src/image_preprocessor.cpp
    src/postprocessor.cpp
)

target_include_directories(pcb_onnx_infer
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(pcb_onnx_infer PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

if(MSVC)
    target_compile_options(pcb_onnx_infer PRIVATE /W4 /permissive-)
else()
    target_compile_options(pcb_onnx_infer PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    set(ONNXRUNTIME_DLL_CANDIDATES
        "${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll"
        "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
    )
    set(ONNXRUNTIME_DLL "")
    foreach(candidate IN LISTS ONNXRUNTIME_DLL_CANDIDATES)
        if(EXISTS "${candidate}")
            set(ONNXRUNTIME_DLL "${candidate}")
            break()
        endif()
    endforeach()
    if(ONNXRUNTIME_DLL)
        add_custom_command(TARGET pcb_onnx_infer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${ONNXRUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:pcb_onnx_infer>/onnxruntime.dll"
        )
        message(STATUS "ONNX Runtime DLL: ${ONNXRUNTIME_DLL}")
    else()
        message(WARNING "onnxruntime.dll not found under ${ONNXRUNTIME_ROOT}/bin or ${ONNXRUNTIME_ROOT}/lib; copy it next to the executable before running.")
    endif()

    if(OpenCV_ARCH AND OpenCV_RUNTIME)
        set(OpenCV_BIN_DIR "${OpenCV_DIR}/${OpenCV_ARCH}/${OpenCV_RUNTIME}/bin")
        file(GLOB OpenCV_RUNTIME_DLLS "${OpenCV_BIN_DIR}/opencv_world*.dll")
        list(FILTER OpenCV_RUNTIME_DLLS EXCLUDE REGEX "d\\.dll$")
        if(OpenCV_RUNTIME_DLLS)
            foreach(opencv_dll IN LISTS OpenCV_RUNTIME_DLLS)
                add_custom_command(TARGET pcb_onnx_infer POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${opencv_dll}"
                            "$<TARGET_FILE_DIR:pcb_onnx_infer>/"
                )
            endforeach()
            message(STATUS "OpenCV runtime DLLs: ${OpenCV_RUNTIME_DLLS}")
        else()
            message(WARNING "OpenCV runtime DLLs were not found under ${OpenCV_BIN_DIR}; add that directory to PATH or copy DLLs next to the executables.")
        endif()
    else()
        message(STATUS "OpenCV_ARCH/OpenCV_RUNTIME not set; if OpenCV DLLs are not on PATH, copy the required OpenCV runtime DLLs next to the executables.")
    endif()
endif()

enable_testing()

add_executable(pcb_cpp_unit_tests
    tests/test_preprocessing.cpp
    tests/test_postprocessing.cpp
    src/image_preprocessor.cpp
    src/postprocessor.cpp
)

target_include_directories(pcb_cpp_unit_tests
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pcb_cpp_unit_tests PRIVATE ${OpenCV_LIBS})

if(MSVC)
    target_compile_options(pcb_cpp_unit_tests PRIVATE /W4 /permissive-)
else()
    target_compile_options(pcb_cpp_unit_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_test(NAME pcb_cpp_unit_tests COMMAND pcb_cpp_unit_tests)

if(WIN32 AND OpenCV_RUNTIME_DLLS)
    foreach(opencv_dll IN LISTS OpenCV_RUNTIME_DLLS)
        add_custom_command(TARGET pcb_cpp_unit_tests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${opencv_dll}"
                    "$<TARGET_FILE_DIR:pcb_cpp_unit_tests>/"
        )
    endforeach()
endif()
