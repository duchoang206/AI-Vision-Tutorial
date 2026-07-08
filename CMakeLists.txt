cmake_minimum_required(VERSION 3.10)
project(DetectRackProject VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add custom module path for Findlibmodbus.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find libmodbus, Threads and ixwebsocket
find_package(Threads REQUIRED)
find_package(libmodbus REQUIRED)
find_package(ixwebsocket REQUIRED)

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${libmodbus_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include
)

# Source files
set(SOURCES
    main.cpp
    src/CameraStream.cpp
    src/YOLOv8Detector.cpp
    src/RegionMonitor.cpp
)

# Executable
add_executable(${PROJECT_NAME} ${SOURCES})

# ONNXRuntime Library
find_library(ONNXRUNTIME_LIB onnxruntime HINTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib REQUIRED)

# Link OpenCV, libmodbus, ixwebsocket, ONNXRuntime, and Windows socket dependencies
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    libmodbus::libmodbus
    ixwebsocket::ixwebsocket
    ${ONNXRUNTIME_LIB}
)

if(WIN32)
    target_link_libraries(${PROJECT_NAME} PRIVATE
        ws2_32
        crypt32
    )
endif()
