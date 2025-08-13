// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_deal_ii_fsi_deal_ii_assembler.hpp"

FOUR_C_NAMESPACE_OPEN

namespace DealiiFSI
{

  void FluidProblem::NavierStokesTerm::add_divergence_term(dealii::FullMatrix<double>& local_matrix,
      const std::vector<double>& div_u, const std::vector<double>& p, double JxW,
      unsigned int dofs_per_cell)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        local_matrix(i, j) -= div_u[i] * p[j] * JxW -
                              p[i] * div_u[j] * JxW;  // mass term in the navier stokes equation
  }
  void FluidProblem::NavierStokesTerm::add_divergence_rhs(dealii::Vector<double>& local_rhs,
      double div_u, double p, const std::vector<double>& phi_div_u,
      const std::vector<double>& phi_p, double JxW, unsigned int dofs_per_cell)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      local_rhs(i) -= div_u * phi_p[i] * JxW - p * phi_div_u[i] * JxW;
  }
}  // namespace DealiiFSI
FOUR_C_NAMESPACE_CLOSE