// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_deal_ii_mimic_fe_values.hpp"

#include "4C_deal_ii_fe_values_context.hpp"


FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{
  template class MimicFEValuesFunction<3, 3>;
}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE
