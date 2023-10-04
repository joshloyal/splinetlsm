cmake_minimum_required (VERSION 3.5)
project(splinetlsm)


## ======================================================================================##
## Check for C++17. For GCC this is >=4.7
## ======================================================================================##
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++17" COMPILER_SUPPORTS_CXX17)
if(COMPILER_SUPPORTS_CXX17)
    message("Compiler with C++17 support found.")
else()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++17 support. Please use a different C++ compiler, i.e. gcc >= 4.7 or Clang >= 3.0.")
endif()

## ======================================================================================##
## Compiler flags
## ======================================================================================##
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++17 -O2")

## ======================================================================================##
## External libraries
## ======================================================================================##
#find_package(Armadillo REQUIRED)
#message("Armadillo Include: ${ARMADILLO_INCLUDE_DIRS}")
#include_directories(${ARMADILLO_INCLUDE_DIRS})
include_directories(/gpfs/home/jdl22h/workspace/code/splinetlsm/third-party/)

#find_package(Boost 1.66 REQUIRED COMPONENTS math)
find_package(Boost REQUIRED)
message("Boost Include: ${Boost_INCLUDE_DIR}")
message("Boost Libraries: ${Boost_LIBRARY_DIRS}")
message("Boost Libraries: ${Boost_LIBRARIES}")
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})

# NOTE: apple needs to use the clang-omp compilers
#find_package(OpenMP REQUIRED)

# GSL
find_package(GSL REQUIRED)
message("GSL Include: ${GSL_INCLUDE_DIRS}")
message("GSL Libraries: ${GSL_LIBRARIES}")
include_directories(${GSL_INCLUDE_DIRS})
#link_directories(${GSL_LIBRARIES})

find_package(BLAS REQUIRED)
message("BLAS Include: ${BLAS_LIBRARIES}")
link_directories(${BLAS_LIBRARIES})
find_package(LAPACK REQUIRED)
message("LAPACK Include: ${LAPACK_LIBRARIES}")
link_directories(${LAPACK_LIBRARIES})

# testing with Catch2
add_subdirectory(/gpfs/home/jdl22h/lib/Catch2 build)
#find_package(Catch2 3 REQUIRED)

## ======================================================================================##
## Subdirectories and source files
## ======================================================================================##
include_directories(src tests)
file(GLOB_RECURSE SOURCES src/*.cpp tests/*.cpp)

## ======================================================================================##
## Debug and release targets
## ======================================================================================##
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

ADD_CUSTOM_TARGET(debug
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
    COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
    )
ADD_CUSTOM_TARGET(release
    COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
    COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

## ======================================================================================##
## Executable
## ======================================================================================##
add_executable(splinetlsm ${SOURCES})
#target_link_libraries(splinetlsm PRIVATE ${ARMADILLO_LIBRARIES} Boost::filesystem OpenMP::OpenMP_CXX Catch2::Catch2WithMain)
#target_link_libraries(splinetlsm PRIVATE ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${ARMADILLO_LIBRARIES} GSL::gsl Catch2::Catch2WithMain)
target_link_libraries(splinetlsm PRIVATE /opt/rcc/gnu/lib64/libblas.so /opt/rcc/gnu/lib64/liblapack.so ${ARMADILLO_LIBRARIES} GSL::gsl Catch2::Catch2WithMain)
