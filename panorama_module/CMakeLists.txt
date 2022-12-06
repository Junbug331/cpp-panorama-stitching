cmake_minimum_required(VERSION 3.10)
project(panorama)

set(CMAKE_CXX_STANDARD 17)
set(TARGET ${PROJECT_NAME})

set(SRC_FILES
    src/panorama.cpp
)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

add_library(${TARGET} ${SRC_FILES})

target_include_directories(${TARGET} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${TARGET} PUBLIC
    ${OpenCV_LIBS}
    spdlog::spdlog
    stdc++fs
)
