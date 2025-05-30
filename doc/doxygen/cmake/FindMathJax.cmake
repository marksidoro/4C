# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

# Sets DOXYGEN_MATHJAX_RELPATH variable and provides target setup_mathjax

if(FOUR_C_DOXYGEN_USE_LOCAL_MATHJAX)
  set(DOXYGEN_MATHJAX_RELPATH ./mathjax)

  # find local mathjax: heuristic check that the main js file exists
  find_file(
    _mathjax_main_file MathJax.js
    PATHS ${FOUR_C_DOXYGEN_LOCAL_MATHJAX_BASEPATH}
    NO_DEFAULT_PATH
    )

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(MathJax DEFAULT_MSG _mathjax_main_file)

  # add a target that copies the local installation to the build directory
  add_custom_target(
    setup_mathjax
    COMMAND
      ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_OUT_DIRECTORY}/html/${DOXYGEN_MATHJAX_RELPATH}
    COMMAND
      ${CMAKE_COMMAND} -E copy_directory ${FOUR_C_DOXYGEN_LOCAL_MATHJAX_BASEPATH}
      ${DOXYGEN_OUT_DIRECTORY}/html/${DOXYGEN_MATHJAX_RELPATH}
    COMMENT "Copy local MathJax to documentation build folder"
    )

else()
  set(DOXYGEN_MATHJAX_RELPATH https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.9/)
  message(STATUS "Doxygen: Use Mathjax from CDN")

  add_custom_target(setup_mathjax COMMENT "Use Mathjax from CDN, no setup necessary")
endif()
