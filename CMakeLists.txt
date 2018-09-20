cmake_minimum_required(VERSION 2.8)

project(large_artm)

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})

set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 

find_package(Boost REQUIRED) 
include_directories(${Boost_INCLUDE_DIRS}) 

set(CMAKE_CXX_COMPILER "g++")

set(SOURCE_LIB
  blas.cc
  dense_phi_matrix.cc
  helpers.cc
  messages.pb.cc
  processor_helpers.cc
  redis_phi_matrix.cc
  token.cc
  redis_client.cc
)

set(CMAKE_CXX_STANDARD 11)

set(CXX_STANDARD_REQUIRED)

set(CMAKE_CXX_FLAGS "-lboost_filesystem -lboost_system -lboost_program_options")

set(SOURCE_EXE srcmain.cc)

add_library(help_lib STATIC ${SOURCE_LIB})

add_executable(srcmain ${SOURCE_EXE})

target_link_libraries(srcmain help_lib ${Boost_LIBRARIES} ${PROTOBUF_LIBRARY} -lhiredis)
