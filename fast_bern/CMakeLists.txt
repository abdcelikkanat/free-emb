cmake_minimum_required(VERSION 3.12)
project(fast_bern)

set(CMAKE_CXX_STANDARD 14)

add_executable(fast_bern main.cpp Graph.cpp Graph.h Unigram.cpp Unigram.h Model.cpp Model.h)