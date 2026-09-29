cmake_minimum_required(VERSION 3.24)
project(AegisVision VERSION 0.1.0 LANGUAGES CXX)

option(AEGISVISION_BUILD_TESTS "Build unit tests" ON)
option(AEGISVISION_WITH_OPENCV "Build image tools and geometric registration" OFF)

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
    target_compile_options(aegisvision PUBLIC /utf-8)
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

if(AEGISVISION_WITH_OPENCV)
    find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d)
    add_library(aegisvision_opencv src/vision.cpp)
    target_link_libraries(aegisvision_opencv PUBLIC aegisvision ${OpenCV_LIBS})
    target_include_directories(aegisvision_opencv SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
    if(MSVC)
        target_compile_options(aegisvision_opencv PRIVATE /W4 /permissive-)
    else()
        target_compile_options(aegisvision_opencv PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_executable(aegisvision_image apps/image.cpp)
    target_link_libraries(aegisvision_image PRIVATE aegisvision_opencv)
    # Copy imported OpenCV DLLs next to executables on Windows; no global PATH edits.
    function(aegisvision_runtime target)
        if(WIN32)
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
                COMMAND_EXPAND_LISTS)
        endif()
    endfunction()
    aegisvision_runtime(aegisvision_image)
    if(AEGISVISION_BUILD_TESTS)
        add_executable(aegisvision_vision_tests tests/test_vision.cpp)
        target_link_libraries(aegisvision_vision_tests PRIVATE aegisvision_opencv)
        aegisvision_runtime(aegisvision_vision_tests)
        add_test(NAME aegisvision_vision_tests COMMAND aegisvision_vision_tests)
        add_test(NAME aegisvision_image_help COMMAND aegisvision_image --help)
        add_test(NAME aegisvision_image_demo COMMAND aegisvision_image demo
            ${CMAKE_CURRENT_BINARY_DIR}/image-demo)
    endif()
    install(TARGETS aegisvision_opencv aegisvision_image)
endif()

