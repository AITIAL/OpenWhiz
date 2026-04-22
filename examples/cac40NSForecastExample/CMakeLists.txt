add_executable(cac40NSForecastExample main.cpp)
target_include_directories(cac40NSForecastExample PRIVATE ${CMAKE_SOURCE_DIR}/include)

if(USE_GPU)
    target_link_libraries(cac40NSForecastExample PRIVATE OpenWhizCUDA cudart cuda)
endif()
