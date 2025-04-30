// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_deal_ii_create_discretization_helper_test.hpp"
#include "4C_deal_ii_mimic_fe_values.hpp"
#include "4C_deal_ii_triangulation.hpp"
#include "4C_fem_discretization.hpp"

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools_interpolate.templates.h>
#include <Epetra_MpiComm.h>
#include <Epetra_SerialComm.h>
#include <Teuchos_ParameterList.hpp>

namespace
{
  using namespace FourC;

  constexpr int dim = 3;

  TEST(AssembleCoupling, SerialTriaOneCell)
  {
    dealii::parallel::fullydistributed::Triangulation<dim> tria{MPI_COMM_WORLD};
    dealii::DoFHandler<dim> dof_handler{tria};
    const dealii::FE_Q<dim> deal_fe(1);
    dof_handler.distribute_dofs(deal_fe);

    Core::FE::Discretization discret{"empty", MPI_COMM_WORLD, dim};
    DealiiWrappers::Context<dim> context;

    DealiiWrappers::MimicFEValuesFunction<dim>(context, discret);

    // make sparsity pattern for the matrix
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());

    std::vector<dealii::types::global_dof_index> global_dofs_on_cell_deal_ii;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      const unsigned int dofs_per_cell_range = cell->get_fe().dofs_per_cell;
      global_dofs_on_cell_deal_ii.resize(dofs_per_cell_range);
      cell->get_dof_indices(global_dofs_on_cell_deal_ii);

      const auto* four_c_element = DealiiWrappers::Internal::to_element(context, discret, cell);
      Core::Elements::LocationArray location_array(discret.num_dof_sets());
      four_c_element->location_vector(discret, location_array, false);
      const auto& global_dofs_on_cell_four_c = location_array[0].lm_;
      for (const auto& dof : global_dofs_on_cell_deal_ii)
      {
        dsp.add_entries(dof, global_dofs_on_cell_four_c.begin(), global_dofs_on_cell_four_c.end());
      }
    }
    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    dealii::SparseMatrix<double> matrix;
    matrix.reinit(sparsity_pattern);

    // Assembly loop:
    dealii::FEValues<dim> fe_values(
        deal_fe, dealii::QGauss<dim>(1), dealii::update_values | dealii::update_JxW_values);

    DealiiWrappers::MimicFEValuesFunction<dim> mimic_fe_values_function(context, discret);

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      mimic_fe_values_function.reinit(cell, fe_values.get_quadrature());

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        for (const unsigned int i : fe_values.dof_indices())
        {
        }
      }


      const unsigned int dofs_per_cell_range = cell->get_fe().dofs_per_cell;
      global_dofs_on_cell_deal_ii.resize(dofs_per_cell_range);
      cell->get_dof_indices(global_dofs_on_cell_deal_ii);

      const auto* four_c_element = DealiiWrappers::Internal::to_element(context, discret, cell);
      Core::Elements::LocationArray location_array(discret.num_dof_sets());
      four_c_element->location_vector(discret, location_array, false);
      const auto& global_dofs_on_cell_four_c = location_array[0].lm_;



      for (const auto& dof : global_dofs_on_cell_deal_ii)
      {
        dsp.add_entries(dof, global_dofs_on_cell_four_c.begin(), global_dofs_on_cell_four_c.end());
      }
    }


    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    VectorType dealii_vector;
    dealii::IndexSet locally_relevant_dofs =
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    dealii_vector.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, MPI_COMM_WORLD);
    dealii_vector = 1.0;
  }


}  // namespace