cmake_minimum_required(VERSION 3.0.2)

project(cluster-bigartm)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

add_custom_command(
  OUTPUT messages.pb.cc messages.pb.h
  DEPENDS messages.proto
  COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} messages.proto --cpp_out=./
)

set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

find_package(Boost COMPONENTS
  filesystem
  thread
  system
  program_options
REQUIRED)

include_directories(${Boost_INCLUDE_DIRS}) 

find_package(glog 0.3.5 REQUIRED)

set(CMAKE_CXX_COMPILER "g++")

include_directories(./include)
include_directories(./3rdparty)
include_directories(.)

set(SOURCE_LIB
  messages.pb.cc
  src/blas.cc
  src/helpers.cc
  src/processor_helpers.cc
  src/redis_phi_matrix.cc
  src/token.cc
  src/redis_client.cc
  src/executor_thread.cc
)

set(CMAKE_CXX_STANDARD 11)

set(CXX_STANDARD_REQUIRED)

add_library(cluster_bigartm_lib STATIC ${SOURCE_LIB})

add_executable(executor_main src/executor_main.cc)
add_executable(master_main src/master_main.cc)

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
