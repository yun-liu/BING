#
find_package(OpenCV REQUIRED)

#OPENCV_DIR should be set as an environment variable in the shell if it is not
#present in the standard directories

set(LIB_DIRS ${BING_BINARY_DIR}/LibLinear)
link_directories(${LIB_DIRS})

if (${CMAKE_COMPILER_IS_GNUCXX})
  set(OPENMP_FLAG "-fopenmp -std=c++11")
endif (${CMAKE_COMPILER_IS_GNUCXX})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPENMP_FLAG} -O3")

set(SRC_FILES CmFile.cpp DataSetVOC.cpp FilterBING.cpp BING.cpp stdafx.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES} Main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} liblinear)
