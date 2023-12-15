cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

set(OpenCV_PATH C:/projects/opencv/build)

find_package(Torch REQUIRED PATHS C:/projects/libtorch-win)

find_package(OpenCV REQUIRED PATHS ${OpenCV_PATH} )
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)

target_link_libraries(example-app ${OpenCV_LIBS} ${TORCH_LIBRARIES})
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)

  file(GLOB OpenCV_DLLS "C:/projects/opencv/build/x64/vc16/bin/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OpenCV_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)