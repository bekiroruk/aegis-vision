cmake_minimum_required(VERSION 3.24)
project(AegisVision VERSION 0.1.0 LANGUAGES CXX)

option(AEGISVISION_BUILD_TESTS "Build unit tests" ON)

add_library(aegisvision
    src/adapters.cpp
    src/domain.cpp
    src/geometry.cpp
    src/pipeline.cpp
    src/tracking.cpp
    src/vector_store.cpp
)
target_include_directories(aegisvision PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(aegisvision PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(aegisvision PRIVATE /W4 /permissive-)
else()
    target_compile_options(aegisvision PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

add_executable(aegisvision_demo apps/demo.cpp)
target_link_libraries(aegisvision_demo PRIVATE aegisvision)

if(AEGISVISION_BUILD_TESTS)
    enable_testing()
    add_executable(aegisvision_tests tests/test_main.cpp)
    target_link_libraries(aegisvision_tests PRIVATE aegisvision)
    add_test(NAME aegisvision_tests COMMAND aegisvision_tests)
endif()

install(TARGETS aegisvision aegisvision_demo)
install(DIRECTORY include/ DESTINATION include)

