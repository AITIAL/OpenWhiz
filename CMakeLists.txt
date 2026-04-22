cmake_minimum_required(VERSION 3.19)

# GPU Support Option
option(USE_GPU "Enable CUDA GPU Acceleration" OFF)


if(NOT USE_GPU)
    # CPU Mode - Clang
    set(CMAKE_C_COMPILER "C:/dev/glist/zbin/glistzbin-win64/clang64/bin/clang.exe")
    set(CMAKE_CXX_COMPILER "C:/dev/glist/zbin/glistzbin-win64/clang64/bin/clang++.exe")
    
    project(OpenWhiz LANGUAGES CXX)
    include_directories(${CMAKE_SOURCE_DIR}/include)
else()
    # Define Toolchain Paths
    set(CUDA_PATH "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.2")
    set(MSVC_PATH "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64")
    set(MSVC_INCLUDE "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/include")
    set(MSVC_LIB "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/lib/x64")
    set(WIN_SDK_BIN "C:/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64")
    set(WIN_SDK_INCLUDE "C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0")
    set(WIN_SDK_LIB "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.26100.0")

    # Setup Environment Variables for MSVC (CRITICAL for Linker)
    set(ENV{INCLUDE} "${MSVC_INCLUDE};${WIN_SDK_INCLUDE}/ucrt;${WIN_SDK_INCLUDE}/shared;${WIN_SDK_INCLUDE}/um;${WIN_SDK_INCLUDE}/winrt;$ENV{INCLUDE}")
    set(ENV{LIB} "${MSVC_LIB};${WIN_SDK_LIB}/ucrt/x64;${WIN_SDK_LIB}/um/x64;$ENV{LIB}")
    set(ENV{PATH} "${MSVC_PATH};${WIN_SDK_BIN};$ENV{PATH}")

    # Explicitly set compilers BEFORE project()
    set(CMAKE_C_COMPILER "${MSVC_PATH}/cl.exe")
    set(CMAKE_CXX_COMPILER "${MSVC_PATH}/cl.exe")
    set(CMAKE_RC_COMPILER "${WIN_SDK_BIN}/rc.exe")
    set(CMAKE_MT "${WIN_SDK_BIN}/mt.exe")
    set(CMAKE_CUDA_COMPILER "${CUDA_PATH}/bin/nvcc.exe")
    set(CMAKE_CUDA_HOST_COMPILER "${MSVC_PATH}/cl.exe")
    
    # Initialize Project with CUDA support
    project(OpenWhiz LANGUAGES CXX CUDA)

    # Explicitly add library directories for the linker
    link_directories("${MSVC_LIB}"
                    "${WIN_SDK_LIB}/ucrt/x64"
                    "${WIN_SDK_LIB}/um/x64"
                    "${CUDA_PATH}/lib/x64")
    
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
    add_compile_definitions(OW_USE_GPU)
    set(CUDA_PROPAGATE_HOST_FLAGS ON)

    # CUDA Library for GPU kernels
    add_library(OpenWhizCUDA STATIC include/OpenWhiz/core/owCuda.cu)
    target_include_directories(OpenWhizCUDA PUBLIC ${CMAKE_SOURCE_DIR}/include)
    
    include_directories(${CMAKE_SOURCE_DIR}/include
                        "${CUDA_PATH}/include"
                        "${MSVC_INCLUDE}"
                        "${WIN_SDK_INCLUDE}/ucrt"
                        "${WIN_SDK_INCLUDE}/shared"
                        "${WIN_SDK_INCLUDE}/um"
                        "${WIN_SDK_INCLUDE}/winrt")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Examples
add_subdirectory(examples/cac40NSForecastExample)
#add_subdirectory(examples/cac40ForecastExample)
#add_subdirectory(examples/cac40ConcatenateForecastExample)
#add_subdirectory(examples/tryusdForecastExample)
#add_subdirectory(examples/tryusdConcatenateForecastingExample)
#add_subdirectory(examples/forecastExample)
#add_subdirectory(examples/concatenateForecastingExample)
#add_subdirectory(examples/airfoilExample)
#add_subdirectory(examples/bankChurnExample)
#add_subdirectory(examples/breastCancerExample)
#add_subdirectory(examples/clusteringExample)
#add_subdirectory(examples/anomalyDetectionExample)
#add_subdirectory(examples/concatenateForecastingExample)
#add_subdirectory(examples/approximationExample)
#add_subdirectory(examples/classificationExample)
#add_subdirectory(examples/tensorExample)
#add_subdirectory(examples/datasetExample)
#add_subdirectory(examples/forecastExample)
#add_subdirectory(examples/attentionExample)
#add_subdirectory(examples/clippingExample)
#add_subdirectory(examples/concatenatedForecastingExample)
#add_subdirectory(examples/multiheadAttentionExample)
#add_subdirectory(examples/incrementalLearningExample)
#add_subdirectory(examples/smoothingExample)
#add_subdirectory(examples/clusterExample)
#add_subdirectory(examples/pcaExample)
#add_subdirectory(examples/rankingExample)
