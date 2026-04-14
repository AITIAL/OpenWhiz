# Minimum CMake version required
cmake_minimum_required(VERSION 3.19)

# Define the project name and set the language to C++
project(tryusdForecastExample LANGUAGES CXX)

# Set the C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Create an executable from the source files
add_executable(tryusdForecastExample main.cpp)

# Specify the include directories for the target
target_include_directories(tryusdForecastExample PRIVATE ../../include)
