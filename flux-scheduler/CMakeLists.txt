cmake_minimum_required(VERSION 3.22)

project(flux_scheduler LANGUAGES CXX)

option(FLUX_SCHEDULER_BUILD_PYTHON "Build Python bindings" ON)
option(FLUX_SCHEDULER_BUILD_TESTS "Build C++ tests" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenSSL REQUIRED)

set(FLUX_SCHEDULER_SOURCES
    csrc/core/token_container.cpp
    csrc/fsm/cache_events.cpp
    csrc/fsm/forward_events.cpp
    csrc/resource/allocator/kv_allocator.cpp
    csrc/resource/allocator/owned_pages.cpp
    csrc/resource/allocator/page_allocator.cpp
    csrc/resource/allocator/paged_cache_group.cpp
    csrc/resource/allocator/req_pool_allocator.cpp
    csrc/resource/hybrid_prefix_cache/hybrid_prefix_cache.cpp
    csrc/resource/kv_prefix_cache/cache_coordinator.cpp
    csrc/resource/kv_prefix_cache/kv_prefix_cache.cpp
    csrc/resource/page_container.cpp
    csrc/resource/radix_tree/radix_tree.cpp
    csrc/resource/radix_tree/tree_node.cpp
    csrc/resource/types.cpp
    csrc/scheduler/kv_cache_events.cpp
    csrc/scheduler/operations/cache.cpp
    csrc/scheduler/operations/forward.cpp
    csrc/scheduler/outside_event_handler.cpp
    csrc/scheduler/request.cpp
    csrc/scheduler/scheduler.cpp
)

add_library(flux_scheduler_core STATIC ${FLUX_SCHEDULER_SOURCES})
target_include_directories(flux_scheduler_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc)
target_link_libraries(flux_scheduler_core PUBLIC OpenSSL::Crypto)
set_target_properties(flux_scheduler_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(FLUX_SCHEDULER_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(flux_scheduler_ext bindings/python_module.cpp)
    target_link_libraries(flux_scheduler_ext PRIVATE flux_scheduler_core)
    install(TARGETS flux_scheduler_ext LIBRARY DESTINATION flux_scheduler)
endif()
