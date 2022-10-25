cmake_minimum_required(VERSION 3.23)
project(3d_mapping)

set(CMAKE_CXX_STANDARD 14)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(Eigen3 REQUIRED)
include_directories(EIGEN_INCLUDE_DIR)

add_executable(3d_mapping main.cpp)
target_link_libraries(3d_mapping PRIVATE ${OpenCV_LIBS})

