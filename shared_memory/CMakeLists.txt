
cmake_minimum_required(VERSION 3.15)

# Project name and language
project(shared_memory LANGUAGES C CXX)

# Enforce C23
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# --- Configuration ---


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -std=c23 -fPIC -pthread")
set(ENABLE_TESTING ON)
set(ENABLE_BENCH ON)

# --- Directories ---


set(SRC_DIR src)
set(INCLUDE_DIR include)
set(TEST_DIR tests)
set(BENCH_DIR bench)


# --- Library Definition ---


include_directories(${INCLUDE_DIR})
include_directories(PUBLIC TracyClient)
include_directories(${SRC_DIR})

file(GLOB LIBRARY_SRCS CONFIGURE_DEPENDS "${SRC_DIR}/*.c")

add_library(${PROJECT_NAME} SHARED ${LIBRARY_SRCS})

# -- Compile options
message("Using:")
macro(set_option option help value)
    option(${option} ${help} ${value})
    set(${option} ${value})
    if(${option})
        message(STATUS "${option}: ON")
        target_compile_definitions(${PROJECT_NAME} PUBLIC ${option})
    else()
        message(STATUS "${option}: OFF")
    endif()
endmacro()
# ON -> emits logs directly from the library
set_option(ENABLE_LOGS "Enable log emit from within the library" ON)
# ON -> assert statements do nothing
set_option(NDEBUG "Disable assert statements" OFF)
# ON -> enables all mprotect calls
set_option(ENABLE_MPROTECT "Enable mprotect syscalls" ON)
# ON -> enable tracy instrumentation
set_option(TRACY_ENABLE "Enable Tracy integration" OFF)
# - Release build should set
# NDEBUG ON
# ENABLE_MPROTECT ON
# ENABLE_LOGS OFF
# ENABLE_TRACY OFF


# --- Third Party ---


if(TRACY_ENABLE)
    set(TRACY_NO_EXIT ON) #Used to keep connection open until client connects (does not work on MacOS)
    FetchContent_Declare(
        tracy
        GIT_REPOSITORY https://github.com/wolfpld/tracy
        GIT_TAG master
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(tracy)


    target_link_libraries(${PROJECT_NAME}
        PUBLIC TracyClient
    )
    target_include_directories(${PROJECT_NAME}
        PUBLIC
        ${INCLUDE_DIR}
        ${tracy_SOURCE_DIR}/public/tracy
        PRIVATE
        ${SRC_DIR}
    )
endif()

# Tracy requires this so we always use it to avoid inconsistencies
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)


# --- Benches ---


if(ENABLE_BENCH)
    enable_testing()
    message(STATUS "Building benches...")

    file(GLOB_RECURSE BENCH_SRCS CONFIGURE_DEPENDS "${BENCH_DIR}/*.c*")

    foreach(BENCH_SRC ${BENCH_SRCS})
        message(${BENCH_SRC})

        get_filename_component(BENCH_NAME ${BENCH_SRC} NAME_WE)

        add_executable(${BENCH_NAME} ${BENCH_SRC})
        target_compile_options(${BENCH_NAME} PRIVATE $<$<CONFIG:Debug>:-g>)
        target_link_libraries(${BENCH_NAME} ${PROJECT_NAME})

        # This is not the correct func...
        add_test(NAME ${BENCH_NAME} COMMAND ${BENCH_NAME})
    endforeach()
endif()


# --- Tests ---


if(ENABLE_TESTING)

    set(NDEBUG OFF)

    enable_testing()
    message(STATUS "Building tests...")

    file(GLOB_RECURSE TEST_SRCS CONFIGURE_DEPENDS "${TEST_DIR}/*.c")

    foreach(TEST_SRC ${TEST_SRCS})
        message(${TEST_SRC})

        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)

        add_executable(${TEST_NAME} ${TEST_SRC})
        target_compile_options(${TEST_NAME} PRIVATE $<$<CONFIG:Debug>:-g>)
        target_link_libraries(${TEST_NAME} ${PROJECT_NAME})
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()
endif()
