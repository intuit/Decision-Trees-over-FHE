cmake_minimum_required(VERSION 3.10.2)
project(fhe_random_forest)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(SEAL 4.0.0 REQUIRED
        PATHS /$ENV{HOME}/SEAL/native/${CMAKE_BUILD_TYPE} /$ENV{HOME}/SEAL/build/cmake)

add_executable(fhe_tests
        tests/fhe_tests.cpp
        src/decision_tree_node.h
        src/enc_tree_node.h
        src/soft_if.cpp
        src/soft_if.h
        src/tree_eval_server.cpp
        src/tree_eval_server.h
        src/fhe_client.cpp
        src/fhe_client.h
        src/tree_train_server.cpp
        src/tree_train_server.h
        src/utils.h
        src/data_types.h)

add_executable(fhe_random_forest
        src/main.cpp
        src/decision_tree_node.h
        src/enc_tree_node.h
        src/soft_if.cpp
        src/soft_if.h
        src/tree_eval_server.cpp
        src/tree_eval_server.h
        src/fhe_client.cpp
        src/fhe_client.h
        src/tree_train_server.cpp
        src/tree_train_server.h
        src/utils.h
        src/data_types.h)

if(TARGET SEAL::seal)
    target_link_libraries(fhe_tests PRIVATE SEAL::seal gtest gtest_main)
    target_link_libraries(fhe_random_forest PRIVATE SEAL::seal gtest gtest_main)
elseif(TARGET SEAL::seal_shared)
    target_link_libraries(fhe_tests PRIVATE SEAL::seal_shared gtest gtest_main)
    target_link_libraries(fhe_random_forest PRIVATE SEAL::seal_shared gtest gtest_main)
else()
    message(FATAL_ERROR "Cannot find target SEAL::seal or SEAL::seal_shared")
endif()

enable_testing()

include(GoogleTest)
gtest_discover_tests(fhe_tests PROPERTIES DISCOVERY_TIMEOUT 70000)