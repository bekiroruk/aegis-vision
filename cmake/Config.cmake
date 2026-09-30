include(FetchContent)
FetchContent_Declare(tomlplusplus
    URL https://codeload.github.com/marzer/tomlplusplus/tar.gz/30172438cee64926dc41fdd9c11fb3ba5b2ba9de
    URL_HASH SHA256=291254ffe7f2433f90deef878d0d9335534a350a958ea23ecf511b7b2277bf7f
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(tomlplusplus)
add_library(aegisvision_configuration src/application_config.cpp)
target_link_libraries(aegisvision_configuration PUBLIC aegisvision_opencv PRIVATE tomlplusplus::tomlplusplus)
add_executable(aegisvision_config apps/config.cpp)
target_link_libraries(aegisvision_config PRIVATE aegisvision_configuration)
aegisvision_runtime(aegisvision_config)
install(TARGETS aegisvision_configuration aegisvision_config)
if(AEGISVISION_BUILD_TESTS)
    add_executable(aegisvision_config_tests tests/test_application_config.cpp)
    target_link_libraries(aegisvision_config_tests PRIVATE aegisvision_configuration)
    aegisvision_runtime(aegisvision_config_tests)
    add_test(NAME aegisvision_config_tests COMMAND aegisvision_config_tests)
endif()
