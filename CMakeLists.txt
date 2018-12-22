cmake_minimum_required(VERSION 3.0.2)

project(cluster-bigartm)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

find_package(Boost REQUIRED) 
include_directories(${Boost_INCLUDE_DIRS}) 

find_package(glog 0.3.5 REQUIRED)

set(CMAKE_CXX_COMPILER "g++")

include_directories(./include)
include_directories(./3rdparty)

set(SOURCE_LIB
  src/blas.cc
  src/helpers.cc
  src/messages.pb.cc
  src/processor_helpers.cc
  src/redis_phi_matrix.cc
  src/token.cc
  src/redis_client.cc
)

set(CMAKE_CXX_STANDARD 11)

set(CXX_STANDARD_REQUIRED)

set(CMAKE_CXX_FLAGS
  "-lboost_filesystem -lboost_system -lboost_program_options"
)

add_library(cluster_bigartm_lib STATIC ${SOURCE_LIB})

add_executable(executor_main executor_main.cc)
add_executable(master_main master_main.cc)

target_link_libraries(
  executor_main
  cluster_bigartm_lib
  ${Boost_LIBRARIES}
  ${PROTOBUF_LIBRARY}
  glog::glog
  -lhiredis
)

target_link_libraries(
  master_main
  cluster_bigartm_lib
  glog::glog
  ${Boost_LIBRARIES}
  ${PROTOBUF_LIBRARY}
  -lhiredis
)
