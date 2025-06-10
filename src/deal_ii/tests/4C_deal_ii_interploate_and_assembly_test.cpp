// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_deal_ii_create_discretization_helper_test.hpp"
#include "4C_deal_ii_element_conversion.hpp"
#include "4C_deal_ii_fe_values_context.hpp"
#include "4C_deal_ii_mimic_fe_values.hpp"
#include "4C_deal_ii_mimic_mapping.hpp"
#include "4C_deal_ii_quadrature_transform.hpp"
#include "4C_deal_ii_triangulation.hpp"
#include "4C_deal_ii_vector_conversion.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_linalg_utils_sparse_algebra_create.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/vector_tools_interpolate.templates.h>
#include <Epetra_SerialComm.h>

namespace
{
  using namespace FourC;

  template <int dim, typename DenseMatrixType,
      typename DofIndicesTypeTest = std::vector<unsigned int>,
      typename DofIndicesTypeTrial = std::vector<unsigned int>>
  void assemble_mass_contrib_local(const dealii::FEValues<dim>& test_values,
      const dealii::FEValues<dim>& trial_values, DenseMatrixType& local_matrix,
      const dealii::Quadrature<dim>& quadrature, const DofIndicesTypeTest& dof_indices_test,
      const DofIndicesTypeTrial& dof_indices_trial)
  {
    for (auto q : std::ranges::iota_view<unsigned>{0U, quadrature.size()})
    {
      // Loop over quadrature points
      for (const auto i : dof_indices_test)
      {
        for (const auto j : dof_indices_trial)
        {
          local_matrix(i, j) +=
              test_values.shape_value(i, q) * trial_values.shape_value(j, q) * test_values.JxW(q);
        }
      }
    }
  }



  constexpr int dim = 3;

  TEST(AssembleVolumeInterpolation, SerialTria)
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;


    dealii::Triangulation<dim> tria;
    const auto comm = MPI_COMM_WORLD;

    Core::FE::Discretization discret{"one_cell", comm, 3};
    TESTING::fill_discretization_hyper_cube(discret, 1, comm);
    DealiiWrappers::Context<dim> context = DealiiWrappers::create_triangulation(tria, discret);

    context.pimpl_->mapping_collection =
        DealiiWrappers::ElementConversion::create_linear_mapping_collection(
            context.pimpl_->finite_elements);



    const auto four_c_vector = Core::LinAlg::create_vector(*discret.dof_row_map());
    // Set random values of the vector
    for (int i = 0; i < four_c_vector->local_length(); ++i)
    {
      four_c_vector->operator[](i) = static_cast<double>(std::rand()) / RAND_MAX;
    }

    const auto four_c_vector_result = Core::LinAlg::create_vector(*discret.dof_row_map());

    dealii::DoFHandler<dim> dof_handler{tria};
    const dealii::FE_Q<dim> deal_fe(1);
    dof_handler.distribute_dofs(deal_fe);


    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DealiiWrappers::make_context_sparsity_pattern(context, dof_handler, discret, dsp);
    dealii::SparsityPattern sparsity_pattern;



    sparsity_pattern.copy_from(dsp);
    dealii::SparseMatrix<double> matrix;
    matrix.reinit(sparsity_pattern);

    dealii::Vector<double> global_vector, solution;
    dealii::Vector<double> local_vector;
    dealii::FullMatrix<double> local_matrix;

    // Assembly loop:
    dealii::QGauss<dim> quadrature(1);

    dealii::FEValues<dim> fe_values_test(
        deal_fe, quadrature, dealii::update_values | dealii::update_JxW_values);

    DealiiWrappers::FEValuesContext<dim> fe_values_context(
        context, discret, dealii::update_values | dealii::update_JxW_values);

    std::vector<double> evaluated_values;

    std::vector<dealii::types::global_dof_index> global_dofs_on_cell_deal_ii;
    std::vector<dealii::types::global_dof_index> global_dofs_on_cell_four_c;


    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      fe_values_test.reinit(cell);
      fe_values_context.reinit(cell);

      const unsigned int dofs_per_cell_range = cell->get_fe().dofs_per_cell;
      global_dofs_on_cell_deal_ii.resize(dofs_per_cell_range);
      cell->get_dof_indices(global_dofs_on_cell_deal_ii);

      const unsigned int dofs_per_cell_domain =
          fe_values_context.get_present_fe_values().dofs_per_cell;
      global_dofs_on_cell_four_c.resize(dofs_per_cell_domain);
      fe_values_context.get_dof_indices_four_c_ordering(global_dofs_on_cell_four_c);

      FOUR_C_ASSERT(dofs_per_cell_range == global_dofs_on_cell_deal_ii.size(),
          "The number of dofs per cell in the range discretization does not match the size of the "
          "global dofs vector.");
      local_matrix.reinit(dofs_per_cell_range, dofs_per_cell_domain);
      local_vector.reinit(dofs_per_cell_range);

      const auto& fe_values_trial = fe_values_context.get_present_fe_values();

      for (unsigned int q_index : fe_values_test.quadrature_point_indices())
      {
        for (auto i : fe_values_test.dof_indices())
        {
          for (auto j : fe_values_context.shape_indices_four_c())
          {
            local_matrix(i, j) += fe_values_test.shape_value(i, q_index) *
                                  fe_values_trial.shape_value(j, q_index) *
                                  fe_values_test.JxW(q_index);
          }
          // assemble the rhs contribution only on the test space
          local_vector(i) = fe_values_test.quadrature_point(q_index).square();
        }
      }  // local assembly
      matrix.add(global_dofs_on_cell_deal_ii, global_dofs_on_cell_four_c, local_matrix);
      global_vector.add(global_dofs_on_cell_deal_ii, local_vector);
    }

    // Now solve the linear system
    dealii::SparseDirectUMFPACK direct_solver;
    direct_solver.initialize(matrix);
    solution.reinit(global_vector);
    direct_solver.vmult(solution, global_vector);

    // Copy into parallel vector to compare
    VectorType parallel_vec_helper;
    parallel_vec_helper.reinit(solution.size());
    for (unsigned int i = 0; i < solution.size(); ++i)
    {
      parallel_vec_helper[i] = solution[i];
    }


    DealiiWrappers::VectorConverter<VectorType, dim> vector_mapping{dof_handler, discret, context};
    vector_mapping.to_four_c(*four_c_vector_result, parallel_vec_helper);

    for (int i = 0; i < four_c_vector->local_length(); ++i)
    {
      EXPECT_DOUBLE_EQ(four_c_vector->operator[](i), four_c_vector_result->operator[](i))
          << "i=" << i;
    }
  }



  /*
  TEST(AssembleCoupling, SerialTria)
  {
    dealii::parallel::fullydistributed::Triangulation<dim> tria{MPI_COMM_WORLD};
    dealii::DoFHandler<dim> dof_handler{tria};
    const dealii::FE_Q<dim> deal_fe(1);
    dof_handler.distribute_dofs(deal_fe);

    Core::FE::Discretization discret{"empty", MPI_COMM_WORLD, dim};
    DealiiWrappers::Context<dim> context;


    const auto four_c_vector = Core::LinAlg::create_vector(*discret.dof_row_map());

    DealiiWrappers::MimicFEValuesFunction<dim>(context, discret);

    // make sparsity pattern for the matrix
    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());

    std::vector<dealii::types::global_dof_index> global_dofs_on_cell_deal_ii;
    std::vector<dealii::types::global_dof_index> global_dofs_on_cell_four_c;


    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      const unsigned int dofs_per_cell_range = cell->get_fe().dofs_per_cell;
      global_dofs_on_cell_deal_ii.resize(dofs_per_cell_range);
      cell->get_dof_indices(global_dofs_on_cell_deal_ii);

      DealiiWrappers::TOOLS::get_dof_indices(context, discret, cell, global_dofs_on_cell_four_c);
      for (const auto& dof : global_dofs_on_cell_deal_ii)
      {
        dsp.add_entries(dof, global_dofs_on_cell_four_c.begin(), global_dofs_on_cell_four_c.end());
      }
    }
    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    dealii::SparseMatrix<double> matrix;
    matrix.reinit(sparsity_pattern);

    dealii::FullMatrix<double> full_matrix;
    dealii::Vector<double> vector;

    // Assembly loop:
    dealii::FEValues<dim> fe_values(
        deal_fe, dealii::QGauss<dim>(1), dealii::update_values | dealii::update_JxW_values);

    DealiiWrappers::MimicFEValuesFunction<dim> mimic_fe_values_function(context, discret);
    std::vector<double> evaluated_values;

    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      mimic_fe_values_function.reinit(cell, fe_values.get_quadrature());
      full_matrix.reinit(
          fe_values.dofs_per_cell, mimic_fe_values_function.n_local_shape_functions());


      mimic_fe_values_function.evaluate_from_dof_vector(four_c_vector, evaluated_values);

      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {
        for (const unsigned int i : fe_values.dof_indices())
        {
          vector(i) += fe_values.shape_value(i, q_index) * evaluated_values[q_index] *
                       fe_values.JxW(q_index) * mimic_fe_values_function.get_jacobian_scaling();

          for (const unsigned int j : mimic_fe_values_function.local_shape_indices())
          {
            full_matrix(i, j) += fe_values.shape_value(i, q_index) *
                                 mimic_fe_values_function.shape_value(j, q_index) *
                                 fe_values.JxW(q_index) *
                                 mimic_fe_values_function.get_jacobian_scaling();
          }
        }
      }  // local assembly

      const unsigned int dofs_per_cell_range = cell->get_fe().dofs_per_cell;
      global_dofs_on_cell_deal_ii.resize(dofs_per_cell_range);
      cell->get_dof_indices(global_dofs_on_cell_deal_ii);

      DealiiWrappers::TOOLS::get_dof_indices(context, discret, cell, global_dofs_on_cell_four_c);
      matrix.add(global_dofs_on_cell_deal_ii, global_dofs_on_cell_four_c, full_matrix);
    }
  }*/
}  // namespace