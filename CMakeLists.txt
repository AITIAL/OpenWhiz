cmake_minimum_required(VERSION 3.19)

# Compiler paths (Absolute paths are more reliable)
# Change this as you need, or delete if the variables is managed by your IDE
set(CMAKE_C_COMPILER "C:/dev/glist/zbin/glistzbin-win64/clang64/bin/clang.exe")
set(CMAKE_CXX_COMPILER "C:/dev/glist/zbin/glistzbin-win64/clang64/bin/clang++.exe")

project(OpenWhiz LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the main include directory so examples can access library headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add examples
add_subdirectory(examples/tryusdForecastExample)
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
