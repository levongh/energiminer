cmake_minimum_required(VERSION 3.5)

include(cmake/cable/bootstrap.cmake)

include(CableBuildInfo)
include(CableBuildType)
include(CableToolchains)
include(HunterGate)

include(defaults/HunterCacheServers)

cable_configure_toolchain(DEFAULT cxx11)

set(HUNTER_CONFIGURATION_TYPES Release CACHE STRING "Build type of Hunter packages")
set(HUNTER_JOBS_NUMBER 6 CACHE STRING "Number of parallel builds used by Hunter")
HunterGate(
    URL "https://github.com/ruslo/hunter/archive/v0.23.6.tar.gz"
    SHA1 "951e8daf57a51708b0e6a00cab342a042db57a2f"
    LOCAL
)

project(energiminer)
set(PROJECT_VERSION 2.0.0)

cable_set_build_type(DEFAULT Release CONFIGURATION_TYPES Release RelWithDebInfo)
option(ETHASHCL "Build with OpenCL mining" ON)
option(ETHASHCUDA "Build with CUDA mining" OFF)
option(BINKERN "Install AMD binary kernels" ON)

# propagates CMake configuration options to the compiler
function(configureProject)
    if (ETHASHCL)
        add_definitions(-DETH_ETHASHCL)
    endif()
	if (ETHASHCUDA)
        add_definitions(-DETH_ETHASHCUDA)
	endif()
    if (BINKERN)
        add_definitions(-DBIN_KERN)
    endif()
endfunction()

hunter_add_package(Boost COMPONENTS system filesystem)
find_package(Boost CONFIG REQUIRED COMPONENTS system filesystem)

hunter_add_package(jsoncpp)
find_package(jsoncpp CONFIG REQUIRED)

hunter_add_package(libjson-rpc-cpp)
find_package(libjson-rpc-cpp CONFIG REQUIRED)

hunter_add_package(CppNetlibUri)
find_package(CppNetlibUri CONFIG REQUIRED)

configureProject()

message("------------------------------------------------------------------------")
message("-- CMake ${CMAKE_VERSION}")
message("-- Build ${CMAKE_BUILD_TYPE} / ${CMAKE_SYSTEM_NAME}")
message("------------------------------------------------------------- components")
message("-- ETHASHCL         Build OpenCL components                  ${ETHASHCL}")
message("-- ETHASHCUDA       Build CUDA components                    ${ETHASHCUDA}")
message("-- BINKERN          Install AMD binary kernels               ${BINKERN}")
message("------------------------------------------------------------------------")
message("")

include(EthCompilerSettings)
#TODO
if(UNIX AND NOT APPLE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libstdc++")
endif()

cable_add_buildinfo_library(PROJECT_NAME ${PROJECT_NAME})

add_subdirectory(protocol)
add_subdirectory(primitives)
add_subdirectory(common)
add_subdirectory(libhwmon)
add_subdirectory(nrghash)
add_subdirectory(nrgcore)

if (ETHASHCL)
    add_subdirectory(libegihash-cl)
    if (BINKERN)
        add_subdirectory(libegihash-cl/kernels)
    endif()
endif()
if (ETHASHCUDA)
    add_subdirectory(libnrghash-cuda)
endif()
add_subdirectory(energiminer)


if(WIN32)
	set(CPACK_GENERATOR ZIP)
else()
	set(CPACK_GENERATOR TGZ)
endif()
set(CPACK_PACKAGE_FILE_NAME ${PROJECT_NAME})
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY FALSE)
include(CPack)
