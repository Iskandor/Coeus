cmake_minimum_required(VERSION 3.3)
project(Coeus VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(CLAB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Coeus/CLAB")
set(CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Coeus/Core")
set(RL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Coeus/RL")
set(EXAMPLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Example")

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(BUILD_EXAMPLES TRUE CACHE BOOL "Build examples")
set(BUILD_RL_MODULE TRUE CACHE BOOL "Build reinforcement learning module")
set(BUILD_SOM_MODULE TRUE CACHE BOOL "Build self-organizing maps module")

include(GNUInstallDirs)

if(MSVC)
  set(pdb_output_dir "${CMAKE_CURRENT_BINARY_DIR}/pdb-files")

  set(CMAKE_PDB_OUTPUT_DIRECTORY "${pdb_output_dir}")
  set(CMAKE_COMPILE_PDB_OUTPUT_DIRECTORY "${pdb_output_dir}")

  get_cmake_property(is_multi GENERATOR_IS_MULTI_CONFIG)
  if(is_multi)
    set(config_suffix "$<CONFIG>")
  else()
    set(config_suffix "")
  endif()

  set(pdb_dst ${CMAKE_INSTALL_BINDIR})

  install(
      DIRECTORY "${pdb_output_dir}/${config_suffix}/"
      DESTINATION ${pdb_dst}
  )
endif()

find_package(MKL REQUIRED)

add_subdirectory(${CLAB_DIR})
add_subdirectory(${CORE_DIR})

set(COEUS_TARGETS $<TARGET_OBJECTS:CLAB> $<TARGET_OBJECTS:CoeusCore>)
set(COEUS_INCLUDE_DIRS ${CLAB_DIR}/ ${CORE_DIR}/)

if(BUILD_RL_MODULE)
	add_subdirectory(${RL_DIR})
	set(COEUS_TARGETS ${COEUS_TARGETS} $<TARGET_OBJECTS:CoeusRL>)
	set(COEUS_INCLUDE_DIRS ${COEUS_INCLUDE_DIRS} ${RL_DIR}/)
endif(BUILD_RL_MODULE)

if(BUILD_EXAMPLES)
	add_subdirectory(${EXAMPLE_DIR})
endif(BUILD_EXAMPLES)


add_library(Coeus SHARED ${COEUS_TARGETS})

target_link_libraries(Coeus PRIVATE
	${MKL_LIBRARIES}
	)

set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_RELWITHDEBINFO_POSTFIX "rd")
set(CMAKE_DEBUG_POSTFIX "d")

set_target_properties(Coeus
    PROPERTIES
	VERSION ${PROJECT_VERSION}
    DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
    )

set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(config_install_dir "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "cmake/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
)

install(
    TARGETS Coeus
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

install(
    DIRECTORY ${COEUS_INCLUDE_DIRS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h"
)

install(
    DIRECTORY ${COEUS_INCLUDE_DIRS}
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.hpp"
)

install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)

install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    DESTINATION "${config_install_dir}"
)

if(WIN32 OR CYGWIN)
  set(new_path "$ENV{PATH};${CMAKE_INSTALL_BINDIR}")

  if(WIN32)
    string(REPLACE ";" "\;" new_path "${CMAKE_INSTALL_BINDIR}")
  elseif(CYGWIN)
    string(REPLACE ";" ":" new_path "${CMAKE_INSTALL_BINDIR}")
  else()
    message(FATAL_ERROR "Unreachable")
  endif()
endif()
