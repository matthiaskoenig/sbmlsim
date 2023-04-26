cmake_minimum_required(VERSION 3.15)

project(icg_sd)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckCXXCompilerFlag)
set(MY_CXX_FLAGS -Wall -Wno-unused-function -Wno-unused-variable)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND MY_CXX_FLAGS  -Wno-unused-but-set-variable)
endif()
foreach(FLAG ${MY_CXX_FLAGS})
    unset(CUR_FLAG_SUPPORTED CACHE)
    CHECK_CXX_COMPILER_FLAG(-Werror ${FLAG} CUR_FLAG_SUPPORTED)
    if(${CUR_FLAG_SUPPORTED})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
    endif()
endforeach(FLAG)

find_package(Amici 0.16.1 REQUIRED HINTS ${CMAKE_CURRENT_LIST_DIR}/../../build)
message(STATUS "Found AMICI ${Amici_DIR}")

set(MODEL_DIR ${CMAKE_CURRENT_LIST_DIR})

set(SRC_LIST_LIB dJydy.cpp 
dxdotdw_colptrs.cpp 
dJydsigma.cpp 
dxdotdw_rowvals.cpp 
dJydy_colptrs.cpp 
wrapfunctions.cpp 
dx_rdatadx_solver_colptrs.cpp 
dx_rdatadx_solver.cpp 
sigmay.cpp 
dwdx_rowvals.cpp 
w.cpp 
dx_rdatadx_solver_rowvals.cpp 
dxdotdp_explicit.cpp 
Jy.cpp 
dwdw.cpp 
dwdx_colptrs.cpp 
dwdp_rowvals.cpp 
dwdw_rowvals.cpp 
dwdp_colptrs.cpp 
dwdx.cpp 
dtotal_cldx_rdata_colptrs.cpp 
dJydy_rowvals.cpp 
dxdotdx_explicit_rowvals.cpp 
dxdotdx_explicit_colptrs.cpp 
dx_rdatadtcl.cpp 
dtotal_cldx_rdata.cpp 
dydx.cpp 
y.cpp 
dx_rdatadtcl_colptrs.cpp 
x_solver.cpp 
x_rdata.cpp 
dxdotdp_explicit_colptrs.cpp 
xdot.cpp 
dxdotdp_explicit_rowvals.cpp 
dxdotdx_explicit.cpp 
dwdp.cpp 
dtotal_cldx_rdata_rowvals.cpp 
dwdw_colptrs.cpp 
icg_sd.cpp 
dx_rdatadtcl_rowvals.cpp 
x0.cpp 
dxdotdw.cpp 
dydp.cpp 
total_cl.cpp 
${MODEL_DIR}/wrapfunctions.cpp
)

add_library(${PROJECT_NAME} ${SRC_LIST_LIB})
add_library(model ALIAS ${PROJECT_NAME})

# This option can be helpful when using the Intel compiler and compilation of
# wrapfunctions.cpp fails due to insufficient memory.
option(ENABLE_WRAPFUNCTIONS_OPTIMIZATIONS "Enable compiler optimizations for wrapfunctions.cpp?" ON)
if(NOT ENABLE_WRAPFUNCTIONS_OPTIMIZATIONS)
    set_source_files_properties(wrapfunctions.cpp PROPERTIES COMPILE_FLAGS -O0)
endif()

target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(${PROJECT_NAME}
    PUBLIC Upstream::amici
)

set(SRC_LIST_EXE main.cpp)

add_executable(simulate_${PROJECT_NAME} ${SRC_LIST_EXE})

target_link_libraries(simulate_${PROJECT_NAME} ${PROJECT_NAME})

if($ENV{ENABLE_GCOV_COVERAGE})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage")
endif()

## SWIG
option(ENABLE_SWIG "Build swig/python library?" ON)
if(ENABLE_SWIG)
    if(NOT(${CMAKE_VERSION} VERSION_LESS 3.8))
        add_subdirectory(swig)
    else()
        message(WARNING "Unable to build SWIG interface, upgrade CMake to >=3.8.")
    endif()
endif()


# <Export cmake configuration>
include(GNUInstallDirs)
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
    ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
export(EXPORT ${PROJECT_NAME}Targets FILE ${PROJECT_NAME}Config.cmake
    NAMESPACE Upstream::
    )
# </Export cmake configuration>

