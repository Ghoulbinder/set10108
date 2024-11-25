project(set10108-cw2 LANGUAGES CXX CUDA) # Enable CUDA language
cmake_minimum_required(VERSION 3.18) # CUDA support is more robust in CMake 3.18+

# Set the C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Include directories
include_directories(../contrib)
if (WIN32)
    include_directories(../contrib/sfml/include)

    link_directories(../contrib/sfml/lib/Debug)
    link_directories(../contrib/sfml/lib/Release)
else()
    find_package(SFML 2.5 COMPONENTS window graphics system REQUIRED)
endif()

# Add the main executable
add_executable(cw2 main.cpp)

# Add the CUDA executable
add_executable(cw2-cuda CW2.cu)
set_target_properties(cw2-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON) # Enable CUDA separable compilation

# Link SFML libraries to the main executable
target_link_libraries(cw2
    optimized sfml-system optimized sfml-window optimized sfml-graphics
    debug sfml-system-d debug sfml-window-d debug sfml-graphics-d
)

# Link CUDA runtime library to the CUDA target
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cw2-cuda PRIVATE CUDA::cudart)

# Add post-build commands to copy SFML DLLs for the main executable
add_custom_command(TARGET cw2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Debug"
    "$<TARGET_FILE_DIR:cw2>"
)
add_custom_command(TARGET cw2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_SOURCE_DIR}/../contrib/sfml/lib/Release"
    "$<TARGET_FILE_DIR:cw2>"
)
