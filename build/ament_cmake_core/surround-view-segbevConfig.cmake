# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_surround-view-segbev_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED surround-view-segbev_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(surround-view-segbev_FOUND FALSE)
  elseif(NOT surround-view-segbev_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(surround-view-segbev_FOUND FALSE)
  endif()
  return()
endif()
set(_surround-view-segbev_CONFIG_INCLUDED TRUE)

# output package information
if(NOT surround-view-segbev_FIND_QUIETLY)
  message(STATUS "Found surround-view-segbev: 0.0.0 (${surround-view-segbev_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'surround-view-segbev' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${surround-view-segbev_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(surround-view-segbev_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${surround-view-segbev_DIR}/${_extra}")
endforeach()
