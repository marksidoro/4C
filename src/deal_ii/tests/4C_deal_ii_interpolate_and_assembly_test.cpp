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
#include "4C_deal_ii_linalg_sparsity.hpp"
#include "4C_deal_ii_mapping.hpp"
#include "4C_deal_ii_triangulation.hpp"
#include "4C_deal_ii_vector_conversion.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_dofset.hpp"

#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.templates.h>
#include <Epetra_SerialComm.h>


namespace
{
  using namespace FourC;

  template <int dim>
  struct Transformations
  {
    static dealii::Point<dim> no_transform(const dealii::Point<dim>& coords) { return coords; }
    static dealii::Point<dim> linear_transform(const dealii::Point<dim>& coords)
    {
      dealii::Point transformed_coords = coords;
      transformed_coords[0] = coords[0] + 0.5 * (coords[1] + 1.0);
      if constexpr (dim > 1) transformed_coords[1] = coords[1] + 0.5 * (coords[0] + 1.0);
      if constexpr (dim > 2) transformed_coords[2] = coords[2] * 0.5;
      return transformed_coords;
    }
    static dealii::Point<dim> quadratic_transform(const dealii::Point<dim>& coords)
    {
      dealii::Point transformed_coords = coords;
      transformed_coords[0] = coords[0] + 0.25 * (coords[1] + 1.0) * (coords[1] + 1.0);
      if constexpr (dim > 1) transformed_coords[1] = coords[1];
      if constexpr (dim > 2) transformed_coords[2] = coords[2];
      return transformed_coords;
    }

    static double linear_function(const dealii::Point<dim>& p, const unsigned int component)
    {
      double value = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
      {
        value += (component + 1) * (p[d]);
      }
      return value;
    }

    static double quadratic_function(const dealii::Point<dim>& p, const unsigned int component)
    {
      double value = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
      {
        value += ((component * component) + 1) * (d + 1) * (p[d] * p[d]);
      }
      return value;
    }

    static void fill_hex_27(Core::FE::Discretization& dis, bool vector_valued,
        const std::function<dealii::Point<dim>(const dealii::Point<dim>&)>& transform)
    {
      TESTING::fill_undeformed_hex27(dis, MPI_COMM_WORLD, vector_valued, transform);
    }

    template <int subdivisions>
    static void fill_hex_8(Core::FE::Discretization& dis, bool vector_valued,
        const std::function<dealii::Point<dim>(const dealii::Point<dim>&)>& transform)
    {
      TESTING::fill_discretization_hyper_cube(
          dis, subdivisions, MPI_COMM_WORLD, vector_valued, transform);
    }
  };


  /**
   * Helper struct for easier setup of the test cases.
   *
   * The main test is executed in the `run` method, which assembles a matrix and a right-hand side
   * and solves the linear system.
   * The problem that's considered is an L2 projection on a possibly distorted but simple mesh.
   * The matrix is assembled using both deal.II (domain/source)  and 4C (range/dst) ordered finite
   * elements (the spaces are the same though).
   * In the tests the functions/deformations and fe spaces are chosen such that the approximation
   * is exact and thus the result should be just the function evaluated at the nodes of the
   * 4C discretization.
   * @tparam dim
   */
  template <int dim>
  struct AssembleAndSolve
  {
    // using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    using VectorType = dealii::TrilinosWrappers::MPI::Vector;

    std::function<dealii::Point<dim>(const dealii::Point<dim>&)> transform;
    std::function<double(const dealii::Point<dim>&, unsigned int)> function;
    std::function<void(Core::FE::Discretization&, bool,
        std::function<dealii::Point<dim>(const dealii::Point<dim>&)>)>
        fill_function;


    double run(bool vector_valued = false)
    {
      const auto comm = MPI_COMM_WORLD;
      const unsigned int n_components = vector_valued ? dim : 1;

      dealii::parallel::fullydistributed::Triangulation<dim> tria(comm);

      Core::FE::Discretization discret{"one_cell", comm, dim};
      fill_function(discret, vector_valued, transform);

      DealiiWrappers::Context<dim> context = DealiiWrappers::create_triangulation(tria, discret);
      auto local_four_c_dofs = context.locally_owned_dofs();


      auto isogeometric_mapping =
          // DealiiWrappers::MappingContext<dim>::create_linear_mapping(context);
          DealiiWrappers::MappingContext<dim>::create_isoparametric_mapping(context);
      dealii::DoFHandler<dim> dof_handler{tria};
      const auto& deal_fe = context.get_finite_elements()[0];
      dof_handler.distribute_dofs(deal_fe);


      EXPECT_EQ(dof_handler.n_dofs(), discret.dof_row_map()->num_global_elements())
          << "The number of dofs in the deal.II DoFHandler does not match the number of dofs in "
             "the ";

      auto local_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
      dealii::DynamicSparsityPattern dsp(
          dof_handler.n_dofs(), dof_handler.n_dofs(), local_relevant_dofs);
      DealiiWrappers::make_context_sparsity_pattern<false>(context, dof_handler, dsp);
      dealii::SparsityTools::distribute_sparsity_pattern(
          dsp, dof_handler.locally_owned_dofs(), comm, local_relevant_dofs);


      // dealii::TrilinosWrappers::SparsityPattern sparsity_pattern;
      // sparsity_pattern.reinit(dof_handler.locally_owned_dofs(), local_four_c_dofs, dsp, comm,
      // true); sparsity_pattern.compress();


      dealii::TrilinosWrappers::SparseMatrix matrix;
      matrix.reinit(dof_handler.locally_owned_dofs(), local_four_c_dofs, dsp, comm, true);



      auto locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
      VectorType rhs, solution;
      rhs.reinit(dof_handler.locally_owned_dofs(), comm);
      // rhs.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, comm);
      solution.reinit(local_four_c_dofs, comm);


      // =======================================================================================
      // Assembly of the matrix and rhs vector
      {
        dealii::Vector<double> local_vector;
        dealii::FullMatrix<double> local_matrix;


        auto max_degree = std::max(deal_fe.degree, context.get_finite_elements().max_degree());
        // Assembly loop:
        dealii::QGauss<dim> quadrature_gauss(max_degree + 1);
        const auto& quadrature = quadrature_gauss;

        dealii::hp::QCollection<dim> quadrature_collection(quadrature);
        dealii::FEValues<dim> fe_values_test(isogeometric_mapping.get_mapping_collection()[0],
            deal_fe, quadrature,
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

        DealiiWrappers::FEValuesContext<dim> fe_values_context(
            isogeometric_mapping.get_mapping_collection(), context, quadrature_collection,
            dealii::update_values | dealii::update_JxW_values);

        std::vector<double> evaluated_values;

        std::vector<dealii::types::global_dof_index> global_dofs_on_cell_deal_ii;
        std::vector<dealii::types::global_dof_index> global_dofs_on_cell_four_c;


        for (const auto& cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned()) continue;

          fe_values_test.reinit(cell);
          fe_values_context.reinit(cell);


          const unsigned int dofs_per_cell_range = cell->get_fe().dofs_per_cell;
          global_dofs_on_cell_deal_ii.resize(dofs_per_cell_range);
          cell->get_dof_indices(global_dofs_on_cell_deal_ii);

          const unsigned int dofs_per_cell_domain =
              fe_values_context.get_present_fe_values().dofs_per_cell;
          global_dofs_on_cell_four_c.resize(dofs_per_cell_domain);


          // fe_values_context.get_dof_indices_dealii_ordering(global_dofs_on_cell_four_c);
          // const auto& local_indexing = fe_values_context.local_dealii_indexing();

          fe_values_context.get_dof_indices_four_c_ordering(global_dofs_on_cell_four_c);
          const auto& local_indexing = fe_values_context.local_four_c_indexing();



          FOUR_C_ASSERT(dofs_per_cell_range == global_dofs_on_cell_deal_ii.size(),
              "The number of dofs per cell in the range discretization does not match the size of "
              "the "
              "global dofs vector.");


          local_matrix.reinit(dofs_per_cell_range, dofs_per_cell_domain);
          local_vector.reinit(dofs_per_cell_range);


          const auto& fe_values_trial = fe_values_context.get_present_fe_values();
          for (unsigned int q_index : fe_values_test.quadrature_point_indices())
          {
            auto quad_point = fe_values_test.quadrature_point(q_index);
            for (auto i : fe_values_test.dof_indices())
            {
              for (auto j : fe_values_trial.dof_indices())
              {
                /**
                 * collapses to the following for n_components = 1:
                 *
                 * local_matrix(i, j) += fe_values_test.shape_value(i, q_index) *
                                      fe_values_trial.shape_value(local_indexing[j], q_index) *
                                      fe_values_test.JxW(q_index);
                 */
                for (unsigned int c = 0; c < n_components; ++c)
                {
                  local_matrix(i, j) +=
                      fe_values_test.shape_value_component(i, q_index, c) *
                      fe_values_trial.shape_value_component(local_indexing[j], q_index, c) *
                      fe_values_test.JxW(q_index);
                }
              }
              // assemble the rhs contribution only on the test space
              for (unsigned int c = 0; c < n_components; ++c)
              {
                // add the contribution to the local vector
                local_vector(i) += function(quad_point, c) *
                                   fe_values_test.shape_value_component(i, q_index, c) *
                                   fe_values_test.JxW(q_index);
              }
            }
          }  // local assembly
          matrix.add(global_dofs_on_cell_deal_ii, global_dofs_on_cell_four_c, local_matrix);
          rhs.add(global_dofs_on_cell_deal_ii, local_vector);
        }
      }
      matrix.compress(dealii::VectorOperation::add);
      rhs.compress(dealii::VectorOperation::add);
      // =======================================================================================
      // solve
      {
        dealii::TrilinosWrappers::SolverDirect direct_solver;
        direct_solver.initialize(matrix);
        direct_solver.vmult(solution, rhs);
      }

      // =======================================================================================
      // Compare to solution:

      // true solution for 4C is given by the function evaluated on all the transformed nodes i.e.
      // we use a dealii::Vector to store the solution for simplicity of comparison
      VectorType four_c_vector_result;
      four_c_vector_result.reinit(local_four_c_dofs, comm);
      for (int i = 0; i < discret.num_my_row_elements(); ++i)
      {
        const auto* element = discret.l_row_element(i);
        Core::Elements::LocationArray location_array(context.get_discretization().num_dof_sets());
        element->location_vector(context.get_discretization(), location_array);
        for (int node = 0; node < element->num_node(); ++node)
        {
          dealii::Point<dim> coords;
          for (unsigned int d = 0; d < dim; ++d) coords[d] = element->nodes()[node]->x()[d];
          for (unsigned int c = 0; c < n_components; ++c)
          {
            four_c_vector_result[location_array[0].lm_.at(c + (node * n_components))] =
                function(coords, c);
          }
        }
      }
      four_c_vector_result.compress(dealii::VectorOperation::insert);
      // now we can compare the solution with the four_c_vector_result
      solution -= four_c_vector_result;
      return solution.l2_norm();  // return the error norm
    }
    double run_four_c_range(bool vector_valued = false)
    {
      const auto comm = MPI_COMM_WORLD;
      const unsigned int n_components = vector_valued ? dim : 1;

      dealii::parallel::fullydistributed::Triangulation<dim> tria(comm);

      Core::FE::Discretization discret{"one_cell", comm, dim};
      fill_function(discret, vector_valued, transform);

      DealiiWrappers::Context<dim> context = DealiiWrappers::create_triangulation(tria, discret);
      auto local_four_c_dofs = context.locally_owned_dofs();


      auto isogeometric_mapping =
          // DealiiWrappers::MappingContext<dim>::create_linear_mapping(context);
          DealiiWrappers::MappingContext<dim>::create_isoparametric_mapping(context);
      dealii::DoFHandler<dim> dof_handler{tria};
      const auto& deal_fe = context.get_finite_elements()[0];
      dof_handler.distribute_dofs(deal_fe);


      EXPECT_EQ(dof_handler.n_dofs(), discret.dof_row_map()->num_global_elements())
          << "The number of dofs in the deal.II DoFHandler does not match the number of dofs in "
             "the ";

      auto locally_relevant_dofs_four_c = context.dofs_on_local_cells();
      dealii::DynamicSparsityPattern dsp(
          context.n_dofs(), dof_handler.n_dofs(), locally_relevant_dofs_four_c);
      DealiiWrappers::make_context_sparsity_pattern<true>(context, dof_handler, dsp);

      dealii::SparsityTools::distribute_sparsity_pattern(
          dsp, local_four_c_dofs, comm, locally_relevant_dofs_four_c);

      dealii::TrilinosWrappers::SparseMatrix matrix;
      matrix.reinit(local_four_c_dofs, dof_handler.locally_owned_dofs(), dsp, comm, true);



      auto locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
      VectorType rhs, solution;
      solution.reinit(dof_handler.locally_owned_dofs(), comm);
      rhs.reinit(local_four_c_dofs, comm);


      // =======================================================================================
      // Assembly of the matrix and rhs vector
      {
        dealii::Vector<double> local_vector;
        dealii::FullMatrix<double> local_matrix;


        auto max_degree = std::max(deal_fe.degree, context.get_finite_elements().max_degree());
        // Assembly loop:
        dealii::QGauss<dim> quadrature_gauss(max_degree + 1);
        const auto& quadrature = quadrature_gauss;

        dealii::hp::QCollection<dim> quadrature_collection(quadrature);
        dealii::FEValues<dim> fe_values_deal(isogeometric_mapping.get_mapping_collection()[0],
            deal_fe, quadrature,
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

        DealiiWrappers::FEValuesContext<dim> fe_values_four_c(
            isogeometric_mapping.get_mapping_collection(), context, quadrature_collection,
            dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

        std::vector<double> evaluated_values;

        std::vector<dealii::types::global_dof_index> global_dofs_on_cell_deal_ii;
        std::vector<dealii::types::global_dof_index> global_dofs_on_cell_four_c;


        for (const auto& cell : dof_handler.active_cell_iterators())
        {
          if (not cell->is_locally_owned()) continue;

          fe_values_deal.reinit(cell);
          fe_values_four_c.reinit(cell);


          const unsigned int dofs_per_cell_deal = cell->get_fe().dofs_per_cell;
          global_dofs_on_cell_deal_ii.resize(dofs_per_cell_deal);
          cell->get_dof_indices(global_dofs_on_cell_deal_ii);

          const unsigned int dofs_per_cell_four_c =
              fe_values_four_c.get_present_fe_values().dofs_per_cell;
          global_dofs_on_cell_four_c.resize(dofs_per_cell_four_c);


          // fe_values_context.get_dof_indices_dealii_ordering(global_dofs_on_cell_four_c);
          // const auto& local_indexing = fe_values_context.local_dealii_indexing();

          fe_values_four_c.get_dof_indices_four_c_ordering(global_dofs_on_cell_four_c);
          const auto& local_indexing_four_c = fe_values_four_c.local_four_c_indexing();

          const auto& local_indexing_deal =
              std::ranges::iota_view<unsigned, unsigned>(0, dofs_per_cell_deal);



          const auto& dofs_per_cell_range = dofs_per_cell_four_c;
          const auto& dofs_per_cell_domain = dofs_per_cell_deal;


          local_matrix.reinit(dofs_per_cell_range, dofs_per_cell_domain);
          local_vector.reinit(dofs_per_cell_range);

          const auto& fe_trial = fe_values_deal.get_present_fe_values();
          const auto& fe_test = fe_values_four_c.get_present_fe_values();

          const auto& global_dofs_range = global_dofs_on_cell_four_c;
          const auto& global_dofs_domain = global_dofs_on_cell_deal_ii;

          const auto& local_indexing_range = local_indexing_four_c;
          const auto& local_indexing_domain = local_indexing_deal;



          for (unsigned int q_index : fe_test.quadrature_point_indices())
          {
            auto quad_point = fe_test.quadrature_point(q_index);
            for (auto i : fe_test.dof_indices())
            {
              for (auto j : fe_trial.dof_indices())
              {
                /**
                 * collapses to the following for n_components = 1:
                 *
                 * local_matrix(i, j) += fe_values_test.shape_value(i, q_index) *
                                      fe_values_trial.shape_value(local_indexing[j], q_index) *
                                      fe_values_test.JxW(q_index);
                 */
                for (unsigned int c = 0; c < n_components; ++c)
                {
                  local_matrix(i, j) +=
                      fe_test.shape_value_component(local_indexing_range[i], q_index, c) *
                      fe_trial.shape_value_component(local_indexing_domain[j], q_index, c) *
                      fe_test.JxW(q_index);
                }
              }
              // assemble the rhs contribution only on the test space
              for (unsigned int c = 0; c < n_components; ++c)
              {
                // add the contribution to the local vector
                local_vector(i) +=
                    function(quad_point, c) *
                    fe_test.shape_value_component(local_indexing_range[i], q_index, c) *
                    fe_test.JxW(q_index);
              }
            }
          }  // local assembly
          matrix.add(global_dofs_range, global_dofs_domain, local_matrix);
          rhs.add(global_dofs_range, local_vector);
        }
      }
      matrix.compress(dealii::VectorOperation::add);
      rhs.compress(dealii::VectorOperation::add);
      // =======================================================================================
      // solve
      {
        dealii::TrilinosWrappers::SolverDirect direct_solver;
        direct_solver.initialize(matrix);
        direct_solver.vmult(solution, rhs);
      }

      // =======================================================================================
      // Compare to solution:

      // true solution for 4C is given by the function evaluated on all the transformed nodes i.e.
      // we use a dealii::Vector to store the solution for simplicity of comparison
      VectorType deal_ii_vector_result;
      deal_ii_vector_result.reinit(dof_handler.locally_owned_dofs(), comm);
      dealii::FunctionFromFunctionObjects<dim> result_function(function, 1);
      dealii::VectorTools::interpolate(isogeometric_mapping.get_mapping_collection()[0],
          dof_handler, result_function, deal_ii_vector_result);
      // deal_ii_vector_result.compress(dealii::VectorOperation::insert);
      // now we can compare the solution with the four_c_vector_result
      solution -= deal_ii_vector_result;
      return solution.l2_norm();  // return the error norm
    }
  };



  TEST(AssembleVolumeInterpolationScalar, LinearLinearHex27)
  {
    constexpr int dim = 3;
    AssembleAndSolve<dim> runner{.transform = Transformations<dim>::linear_transform,
        .function = Transformations<dim>::linear_function,
        .fill_function = Transformations<dim>::fill_hex_27};
    const double error = runner.run(false);
    EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with linear "
                                      "transformation and linear function.";
  }



  TEST(AssembleVolumeInterpolationScalar, LinearLinearHex8)
  {
    constexpr int dim = 3;
    AssembleAndSolve<dim> runner{.transform = Transformations<dim>::linear_transform,
        .function = Transformations<dim>::linear_function,
        .fill_function = Transformations<dim>::fill_hex_8<3>};
    {
      const double error = runner.run(false);
      EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with linear "
                                        "transformation and linear function.";
    }
    {
      const double error = runner.run_four_c_range(false);
      EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with linear "
                                        "transformation and linear function.";
    }
  }


  TEST(AssembleVolumeInterpolationScalar, LinearQuadraticHex27)
  {
    constexpr int dim = 3;
    AssembleAndSolve<dim> runner{.transform = Transformations<dim>::linear_transform,
        .function = Transformations<dim>::quadratic_function,
        .fill_function = Transformations<dim>::fill_hex_27};
    const double error = runner.run(false);
    EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with linear "
                                      "transformation and quadratic function.";
  }

  TEST(AssembleVolumeInterpolationScalar, QuadraticLinearHex27)
  {
    constexpr int dim = 3;
    AssembleAndSolve<dim> runner{.transform = Transformations<dim>::quadratic_transform,
        .function = Transformations<dim>::linear_function,
        .fill_function = Transformations<dim>::fill_hex_27};
    const double error = runner.run(false);
    EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with quadratic "
                                      "transformation and linear function.";
  }



  TEST(AssembleVolumeInterpolationVector, LinearLinearHex27)
  {
    constexpr int dim = 3;
    AssembleAndSolve<dim> runner{.transform = Transformations<dim>::linear_transform,
        .function = Transformations<dim>::linear_function,
        .fill_function = Transformations<dim>::fill_hex_27};
    const double error = runner.run(true);
    EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with linear "
                                      "transformation and linear function.";
  }



  TEST(AssembleVolumeInterpolationVector, LinearLinearHex8)
  {
    constexpr int dim = 3;
    AssembleAndSolve<dim> runner{.transform = Transformations<dim>::linear_transform,
        .function = Transformations<dim>::linear_function,
        .fill_function = Transformations<dim>::fill_hex_8<2>};
    const double error = runner.run(true);
    EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with linear "
                                      "transformation and linear function.";
  }


  TEST(AssembleVolumeInterpolationVector, LinearQuadraticHex27)
  {
    constexpr int dim = 3;
    AssembleAndSolve<dim> runner{.transform = Transformations<dim>::linear_transform,
        .function = Transformations<dim>::quadratic_function,
        .fill_function = Transformations<dim>::fill_hex_27};
    const double error = runner.run(true);
    EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with linear "
                                      "transformation and quadratic function.";
  }

  TEST(AssembleVolumeInterpolationVector, QuadraticLinearHex27)
  {
    constexpr int dim = 3;
    AssembleAndSolve<dim> runner{.transform = Transformations<dim>::quadratic_transform,
        .function = Transformations<dim>::linear_function,
        .fill_function = Transformations<dim>::fill_hex_27};
    const double error = runner.run(true);
    EXPECT_NEAR(error, 0.0, 1e-10) << "Error in assembly of volume interpolation with quadratic "
                                      "transformation and linear function.";
  }



}  // namespace