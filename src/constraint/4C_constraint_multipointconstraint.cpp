// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_constraint_multipointconstraint.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"
#include "4C_linalg_utils_sparse_algebra_manipulation.hpp"

#include <iostream>

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
Constraints::MPConstraint::MPConstraint(std::shared_ptr<Core::FE::Discretization> discr,
    const std::string& conditionname, int& minID, int& maxID)
    : Constraints::Constraint(discr, conditionname, minID, maxID)
{
  return;
}

/*----------------------------------------------------------------------*
 |  ctor (public)                                               tk 07/08|
 *----------------------------------------------------------------------*/
Constraints::MPConstraint::MPConstraint(
    std::shared_ptr<Core::FE::Discretization> discr, const std::string& conditionname)
    : Constraints::Constraint(discr, conditionname)
{
  return;
}

/// Set state of the underlying constraint discretization
void Constraints::MPConstraint::set_constr_state(
    const std::string& state,              ///< name of state to set
    const Core::LinAlg::Vector<double>& V  ///< values to set
)
{
  if (constrtype_ != none)
  {
    std::map<int, std::shared_ptr<Core::FE::Discretization>>::iterator discrit;
    for (discrit = constraintdis_.begin(); discrit != constraintdis_.end(); ++discrit)
    {
      std::shared_ptr<Core::LinAlg::Vector<double>> tmp =
          Core::LinAlg::create_vector(*(discrit->second)->dof_col_map(), false);
      Core::LinAlg::export_to(V, *tmp);
      (discrit->second)->clear_state();
      (discrit->second)->set_state(state, *tmp);
    }
  }
}

FOUR_C_NAMESPACE_CLOSE
