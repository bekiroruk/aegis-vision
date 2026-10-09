cmake_minimum_required(VERSION 3.24)
project(AegisVision VERSION 0.1.0 LANGUAGES CXX)

option(AEGISVISION_BUILD_TESTS "Build unit tests" ON)
option(AEGISVISION_WITH_OPENCV "Build image tools and geometric registration" OFF)
option(AEGISVISION_WITH_SEARCH "Build CLIP/ONNX Runtime and local Qdrant search" OFF)

add_library(aegisvision
    src/adapters.cpp
    src/assignment.cpp
    src/domain.cpp
    src/evaluation.cpp
    src/geometry.cpp
    src/kalman_tracker.cpp
    src/pipeline.cpp
    src/tracking.cpp
    src/tracking_evaluation.cpp
    src/two_stage_tracker.cpp
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
    add_executable(aegisvision_tracking_tests tests/test_two_stage.cpp)
    target_link_libraries(aegisvision_tracking_tests PRIVATE aegisvision)
    add_test(NAME aegisvision_tracking_tests COMMAND aegisvision_tracking_tests)
    add_executable(aegisvision_kalman_tests tests/test_kalman_tracker.cpp)
    target_link_libraries(aegisvision_kalman_tests PRIVATE aegisvision)
    add_test(NAME aegisvision_kalman COMMAND aegisvision_kalman_tests)
    add_executable(aegisvision_evaluation_tests tests/test_evaluation.cpp)
    target_link_libraries(aegisvision_evaluation_tests PRIVATE aegisvision)
    add_test(NAME aegisvision_evaluation COMMAND aegisvision_evaluation_tests)
    add_executable(aegisvision_tracking_evaluation_tests tests/test_tracking_evaluation.cpp)
    target_link_libraries(aegisvision_tracking_evaluation_tests PRIVATE aegisvision)
    add_test(NAME aegisvision_tracking_evaluation COMMAND aegisvision_tracking_evaluation_tests)
endif()

install(TARGETS aegisvision aegisvision_demo)
install(DIRECTORY include/ DESTINATION include)

if(AEGISVISION_WITH_OPENCV)
    find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs features2d calib3d dnn videoio)
    find_package(Threads REQUIRED)
    add_library(aegisvision_opencv src/vision.cpp src/yolo.cpp src/video.cpp src/live.cpp src/segmentation.cpp)
    target_link_libraries(aegisvision_opencv PUBLIC aegisvision ${OpenCV_LIBS} Threads::Threads)
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
    include(cmake/Config.cmake)
    aegisvision_runtime(aegisvision_image)
    add_executable(aegisvision_detect apps/detect.cpp)
    target_link_libraries(aegisvision_detect PRIVATE aegisvision_configuration)
    aegisvision_runtime(aegisvision_detect)
    add_executable(aegisvision_video apps/video.cpp)
    target_link_libraries(aegisvision_video PRIVATE aegisvision_configuration)
    aegisvision_runtime(aegisvision_video)
    add_executable(aegisvision_segment apps/segment.cpp)
    target_link_libraries(aegisvision_segment PRIVATE aegisvision_opencv)
    aegisvision_runtime(aegisvision_segment)
    install(TARGETS aegisvision_segment)
    if(AEGISVISION_BUILD_TESTS)
        add_executable(aegisvision_segmentation_tests tests/test_segmentation.cpp)
        target_link_libraries(aegisvision_segmentation_tests PRIVATE aegisvision_opencv)
        aegisvision_runtime(aegisvision_segmentation_tests)
        add_test(NAME aegisvision_segmentation COMMAND aegisvision_segmentation_tests)
        add_test(NAME aegisvision_segment_help COMMAND aegisvision_segment --help)
    endif()
    add_executable(aegisvision_stream apps/stream.cpp)
    target_link_libraries(aegisvision_stream PRIVATE aegisvision_configuration)
    aegisvision_runtime(aegisvision_stream)
    # Official Windows OpenCV distributions load their FFmpeg plugin dynamically.
    if(WIN32 AND TARGET opencv_world)
        get_target_property(_opencv_dll opencv_world IMPORTED_LOCATION_RELEASE)
        get_filename_component(_opencv_bin "${_opencv_dll}" DIRECTORY)
        file(GLOB _opencv_ffmpeg "${_opencv_bin}/opencv_videoio_ffmpeg*.dll")
        if(_opencv_ffmpeg)
            add_custom_command(TARGET aegisvision_segment POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_opencv_ffmpeg}
                    $<TARGET_FILE_DIR:aegisvision_segment> COMMAND_EXPAND_LISTS)
            add_custom_command(TARGET aegisvision_video POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_opencv_ffmpeg}
                    $<TARGET_FILE_DIR:aegisvision_video> COMMAND_EXPAND_LISTS)
            add_custom_command(TARGET aegisvision_stream POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_opencv_ffmpeg}
                    $<TARGET_FILE_DIR:aegisvision_stream> COMMAND_EXPAND_LISTS)
        endif()
    endif()
    if(AEGISVISION_BUILD_TESTS)
        add_executable(aegisvision_vision_tests tests/test_vision.cpp)
        target_link_libraries(aegisvision_vision_tests PRIVATE aegisvision_opencv)
        aegisvision_runtime(aegisvision_vision_tests)
        add_test(NAME aegisvision_vision_tests COMMAND aegisvision_vision_tests)
        add_test(NAME aegisvision_image_help COMMAND aegisvision_image --help)
        add_test(NAME aegisvision_image_demo COMMAND aegisvision_image demo
            ${CMAKE_CURRENT_BINARY_DIR}/image-demo)
        add_executable(aegisvision_yolo_tests tests/test_yolo.cpp)
        target_link_libraries(aegisvision_yolo_tests PRIVATE aegisvision_opencv)
        aegisvision_runtime(aegisvision_yolo_tests)
        add_test(NAME aegisvision_yolo_tests COMMAND aegisvision_yolo_tests)
        add_test(NAME aegisvision_detect_help COMMAND aegisvision_detect --help)
        add_executable(aegisvision_video_tests tests/test_video.cpp)
        target_link_libraries(aegisvision_video_tests PRIVATE aegisvision_opencv)
        aegisvision_runtime(aegisvision_video_tests)
        add_test(NAME aegisvision_video_tests COMMAND aegisvision_video_tests)
        add_test(NAME aegisvision_video_help COMMAND aegisvision_video --help)
        add_executable(aegisvision_live_tests tests/test_live.cpp)
        target_link_libraries(aegisvision_live_tests PRIVATE aegisvision_configuration)
        aegisvision_runtime(aegisvision_live_tests)
        add_test(NAME aegisvision_live_tests COMMAND aegisvision_live_tests)
        set_tests_properties(aegisvision_live_tests PROPERTIES TIMEOUT 30)
        add_test(NAME aegisvision_stream_help COMMAND aegisvision_stream --help)
        add_executable(aegisvision_video_fixture apps/video_fixture.cpp)
        target_link_libraries(aegisvision_video_fixture PRIVATE aegisvision_opencv)
        aegisvision_runtime(aegisvision_video_fixture)
        set(AEGISVISION_YOLO_MODEL "" CACHE FILEPATH "Optional local YOLOv8 ONNX model for smoke test")
        set(AEGISVISION_YOLO_IMAGE "" CACHE FILEPATH "Optional local bus image for smoke test")
        if(AEGISVISION_YOLO_MODEL AND AEGISVISION_YOLO_IMAGE)
            add_test(NAME aegisvision_yolo_inference COMMAND aegisvision_yolo_tests
                "${AEGISVISION_YOLO_MODEL}" "${AEGISVISION_YOLO_IMAGE}")
        endif()
    endif()
    install(TARGETS aegisvision_opencv aegisvision_image aegisvision_detect aegisvision_video aegisvision_stream)
endif()

if(AEGISVISION_WITH_SEARCH)
    if(NOT AEGISVISION_WITH_OPENCV)
        message(FATAL_ERROR "Search requires AEGISVISION_WITH_OPENCV=ON")
    endif()
    include(cmake/Search.cmake)
endif()

