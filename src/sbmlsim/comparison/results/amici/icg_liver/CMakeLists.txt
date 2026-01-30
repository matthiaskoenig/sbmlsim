# Build AMICI model
cmake_minimum_required(VERSION 3.15)

project(icg_liver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckCXXCompilerFlag)
set(MY_CXX_FLAGS -Wall -Wno-unused-function -Wno-unused-variable)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND MY_CXX_FLAGS -Wno-unused-but-set-variable)
endif()
foreach(flag ${MY_CXX_FLAGS})
  unset(CUR_FLAG_SUPPORTED CACHE)
  check_cxx_compiler_flag(-Werror ${flag} CUR_FLAG_SUPPORTED)
  if(${CUR_FLAG_SUPPORTED})
    string(APPEND CMAKE_CXX_FLAGS " ${flag}")
  endif()
endforeach()

if(DEFINED ENV{AMICI_CXXFLAGS})
  message(STATUS "Appending flags from AMICI_CXXFLAGS: $ENV{AMICI_CXXFLAGS}")
  add_compile_options("$ENV{AMICI_CXXFLAGS}")
endif()
if(DEFINED ENV{AMICI_LDFLAGS})
  message(STATUS "Appending flags from AMICI_LDFLAGS: $ENV{AMICI_LDFLAGS}")
  link_libraries("$ENV{AMICI_LDFLAGS}")
endif()

find_package(Amici 0.16.1 REQUIRED HINTS
             ${CMAKE_CURRENT_LIST_DIR}/../../build)
message(STATUS "Found AMICI ${Amici_DIR}")

set(MODEL_DIR ${CMAKE_CURRENT_LIST_DIR})

set(SRC_LIST_LIB dwdx.h 
dJydy.cpp 
total_cl.h 
p.h 
x_solver.h 
dtcldp.h 
dxdotdw_colptrs.cpp 
dJydsigma.cpp 
dxdotdw_rowvals.cpp 
w.h 
sigmay.h 
dJydy_colptrs.cpp 
icg_liver.h 
wrapfunctions.cpp 
dx_rdatadx_solver_colptrs.cpp 
dx_rdatadx_solver.cpp 
sigmay.cpp 
dwdx_rowvals.cpp 
w.cpp 
dx_rdatadx_solver_rowvals.cpp 
Jy.cpp 
dwdw.cpp 
dwdx_colptrs.cpp 
dwdp_rowvals.cpp 
dwdw_rowvals.cpp 
dwdp_colptrs.cpp 
Jy.h 
dwdx.cpp 
dtotal_cldx_rdata_colptrs.cpp 
dJydy_rowvals.cpp 
y.h 
dx_rdatadtcl.cpp 
dx_rdatadtcl.h 
dtotal_cldx_rdata.cpp 
dx_rdatadx_solver.h 
dydx.cpp 
x.h 
y.cpp 
dx_rdatadtcl_colptrs.cpp 
x_solver.cpp 
x_rdata.h 
dxdotdw.h 
icg_liver.cpp 
x_rdata.cpp 
xdot.cpp 
dwdw.h 
dwdp.cpp 
dtotal_cldx_rdata_rowvals.cpp 
dwdw_colptrs.cpp 
dx_rdatadtcl_rowvals.cpp 
x0.cpp 
xdot.h 
dtotal_cldx_rdata.h 
dxdotdw.cpp 
dJydy.h 
my.h 
wrapfunctions.h 
tcl.h 
dwdp.h 
total_cl.cpp  ${MODEL_DIR}/wrapfunctions.cpp)

add_library(${PROJECT_NAME} ${SRC_LIST_LIB})
add_library(model ALIAS ${PROJECT_NAME})

# Some special functions require boost
#
# TODO: set some flag during code generation whether the given model requires
# boost. for now, try to find it, add include directories and link against it.
# let the compiler/linker error if it is required but not found
find_package(Boost)

target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC Upstream::amici
  PRIVATE $<$<BOOL:${Boost_FOUND}>:Boost::boost>)

if(NOT "${AMICI_PYTHON_BUILD_EXT_ONLY}")
  set(SRC_LIST_EXE main.cpp)
  add_executable(simulate_${PROJECT_NAME} ${SRC_LIST_EXE})
  target_link_libraries(simulate_${PROJECT_NAME} ${PROJECT_NAME})
endif()

# Debug build?
if("$ENV{ENABLE_AMICI_DEBUGGING}" OR "$ENV{ENABLE_GCOV_COVERAGE}")
  add_compile_options(-UNDEBUG -O0 -g)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

# coverage options
if($ENV{ENABLE_GCOV_COVERAGE})
  string(APPEND CMAKE_CXX_FLAGS_DEBUG " --coverage")
  string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG " --coverage")
endif()

# SWIG
option(ENABLE_SWIG "Build swig/python library?" ON)
if(ENABLE_SWIG)
  add_subdirectory(swig)
endif()

# <Export cmake configuration>
include(GNUInstallDirs)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
export(
  EXPORT ${PROJECT_NAME}Targets
  FILE ${PROJECT_NAME}Config.cmake
  NAMESPACE Upstream::)
# </Export cmake configuration>
