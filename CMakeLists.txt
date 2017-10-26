cmake_minimum_required(VERSION 3.1.0)

project(tinyrenderer)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories("include")

set(HEADERS
  include/geometry.h
  include/tgaimage.h
  include/our_gl.h
  include/model.h)

set(SOURCES
src/main.cpp
src/geometry.cpp
src/model.cpp
src/our_gl.cpp
src/tgaimage.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
