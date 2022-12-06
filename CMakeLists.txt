cmake_minimum_required(VERSION 3.10)
project(Panorama)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DRES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res")

add_subdirectory(panorama_module)

add_executable(demo example/main.cpp)

target_link_libraries(demo
    panorama
)
