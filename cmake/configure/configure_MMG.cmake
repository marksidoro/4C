# This file is part of 4C multiphysics licensed under the
# GNU Lesser General Public License v3.0 or later.
#
# See the LICENSE.md file in the top-level for license information.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

if (NOT FOUR_C_WITH_DEAL_II)
    message(FATAL_ERROR "MMG requires FOUR_C_WITH_DEAL_II to be ON")
endif ()

find_package(MMG REQUIRED)

target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE MMG::MMG)

if (MMG_FLUID_FOUND)
    target_link_libraries(four_c_all_enabled_external_dependencies INTERFACE MMG::FLUID_LIB)
endif ()

configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/templates/MMG.cmake.in"
        "${PROJECT_BINARY_DIR}/cmake/templates/MMG.cmake"
        @ONLY
)
