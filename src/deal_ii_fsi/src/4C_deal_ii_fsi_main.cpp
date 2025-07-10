// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later


#include "4C_deal_ii_fsi_main.hpp"

#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_deal_ii_fsi_deal_ii_assembler.hpp"
#include "4C_deal_ii_fsi_tools.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include <4C_deal_ii_context.hpp>
#include <4C_deal_ii_triangulation.hpp>
#include <4C_linalg_sparsematrix.hpp>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "deal.II/numerics/data_out.h"

FOUR_C_NAMESPACE_OPEN

namespace DealiiFSI
{
  constexpr unsigned int refine = 3;
  template <int dim>
  void write_tria(const dealii::Triangulation<dim>& tria, std::string name)
  {
    dealii::DataOut<dim> data_out;
    data_out.attach_triangulation(tria);
    dealii::Vector<float> subdomain(tria.n_active_cells());
    for (const auto& cell : tria.active_cell_iterators())
    {
      subdomain[cell->active_cell_index()] = cell->subdomain_id();
    }
    data_out.add_data_vector(subdomain, "subdomain");
    data_out.build_patches();
    data_out.write_vtu_with_pvtu_record("./", name, 0, MPI_COMM_WORLD, 0, 8);
  };

  using MatrixType = Epetra_CrsMatrix;
  using InputType = std::vector<std::vector<const MatrixType*>>;

  std::shared_ptr<MatrixType> build_monolithic(const InputType& matrices)
  {
    using MapT = Epetra_Map;
    const int full_row_size = std::invoke(
        [&]()
        {
          unsigned int rows = 0;
          for (unsigned int r = 0; r < matrices.size(); ++r)
          {
            FOUR_C_ASSERT(
                matrices[r].size() == matrices.size(), " Only square matrices are supported.");
            rows += matrices[r][0]->NumGlobalRows();
          }
          return rows;
        });

    MapT full_map(full_row_size, 0, matrices[0][0]->Comm());
    auto full_matrix_ptr = std::make_shared<MatrixType>(Copy, full_map, 0);
    auto& full_matrix = *full_matrix_ptr;

    // insert block
    auto insert_block = [&](const MatrixType& block, int row_offset, int col_offset)
    {
      for (int row = 0; row < block.NumMyRows(); ++row)
      {
        int global_row = row + row_offset;
        int num_entries;
        double* values;
        int* indices;
        block.ExtractMyRowView(row, num_entries, values, indices);
        std::vector<int> global_indices(num_entries);
        for (int j = 0; j < num_entries; ++j) global_indices[j] = indices[j] + col_offset;

        full_matrix.InsertGlobalValues(global_row, num_entries, values, global_indices.data());
      }
    };
    auto compute_ofsets = [&](unsigned int r, unsigned int c)
    {
      int row_offset = 0;
      for (unsigned int i = 0; i < r; ++i)
      {
        row_offset += matrices[i][0]->NumGlobalRows();
      }
      int col_offset = 0;
      for (unsigned int j = 0; j < c; ++j)
      {
        col_offset += matrices[0][j]->NumGlobalCols();
      }
      return std::make_pair(row_offset, col_offset);
    };

    for (unsigned int r = 0; r < matrices.size(); ++r)
    {
      for (unsigned int c = 0; c < matrices[r].size(); ++c)
      {
        const auto& block = *matrices[r][c];
        if (block.NumMyRows() > 0 && block.NumMyCols() > 0)
        {
          auto offsets = compute_ofsets(r, c);
          insert_block(block, offsets.first, offsets.second);
        }
      }
    }
    return full_matrix_ptr;
  }



  void run()
  {
    constexpr int dim = 2;

    // Adapt structure discretization to extract the structure matrices and other data so that I
    // can use it my own FSI code.

    // get input lists
    const Teuchos::ParameterList& sdyn = Global::Problem::instance()->structural_dynamic_params();
    // access the structural discretization
    std::shared_ptr<Core::FE::Discretization> structdis =
        Global::Problem::instance()->get_dis("structure");

    // connect degrees of freedom for periodic boundary conditions
    {
      Core::Conditions::PeriodicBoundaryConditions pbc_struct(structdis);

      if (pbc_struct.has_pbc())
      {
        pbc_struct.update_dofs_for_periodic_boundary_conditions();
      }
    }

    // create an adapterbase and adapter
    std::shared_ptr<Adapter::Structure> structadapter = nullptr;

    // setup the system matrix and other data
    {
      std::shared_ptr<Adapter::StructureBaseAlgorithmNew> adapterbase_ptr =
          Adapter::build_structure_algorithm(sdyn);
      adapterbase_ptr->init(sdyn, const_cast<Teuchos::ParameterList&>(sdyn), structdis);
      adapterbase_ptr->setup();
      structadapter = adapterbase_ptr->structure_field();
      structadapter->post_setup();

      structadapter->pre_predict();
      structadapter->prepare_time_step();
    }


    structadapter->integrate();

    auto matrix = structadapter->system_matrix();



    auto rhs_original = structadapter->rhs();
    auto rhs = std::make_shared<Core::LinAlg::Vector<double>>(*rhs_original);


    dealii::Vector<double> solid_rhs(rhs->global_length());
    for (unsigned int i = 0; i < solid_rhs.size(); ++i)
    {
      solid_rhs(i) = (*rhs)[i];
    }



    dealii::Triangulation<dim> fluid_tria;
    dealii::SparseMatrix<double> fluid_system_matrix;
    dealii::Vector<double> fluid_rhs;
    DealiiFSI::FluidProblem::StokesProblem<dim> stokes_problem;
    stokes_problem.build_problem(fluid_tria, fluid_system_matrix, fluid_rhs, refine);

    std::cout << "Fluid rhs: " << fluid_rhs.l2_norm() << std::endl;


    // build up the transfer objects:
    dealii::Triangulation<dim> solid_tria;
    auto context = DealiiWrappers::create_triangulation(solid_tria, *structdis);

    dealii::DoFHandler<dim> iso_dof_handler;
    dealii::LinearAlgebra::distributed::Vector<double> iso_vector;


    // auto isogeometric_mapping = DealiiWrappers::create_isoparametric_mapping(
    //       context, *structdis, iso_vector, iso_dof_handler);


    // context.pimpl_->mapping_collection.push_back(isogeometric_mapping);
    context.pimpl_->mapping_collection.push_back(dealii::MappingQ1<dim>());



    InterfaceMatcher<dim> matcher =
        DealiiFSI::make_interface_matcher_across_boundary(solid_tria, fluid_tria, 1e-14);

    dealii::DynamicSparsityPattern dsp(
        rhs->global_length(), stokes_problem.get_dof_handler().n_dofs());
    make_interface_sparsity_pattern(matcher, context, stokes_problem.get_dof_handler(), dsp);

    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);

    dealii::SparseMatrix<double> interface_matrix(sparsity_pattern);

    {
      dealii::QGauss<dim - 1> quadrature_gauss(5);
      const auto& quadrature = quadrature_gauss;
      dealii::hp::QCollection<dim - 1> quadrature_collection(quadrature);


      dealii::FEFaceValues<dim> fe_values_stokes(stokes_problem.get_stokes_fe(), quadrature,
          dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_quadrature_points | dealii::update_normal_vectors);

      DealiiWrappers::FEFaceValuesContext<dim> fe_values_solid(context, *structdis,
          quadrature_collection,
          dealii::update_values | dealii::update_JxW_values | dealii::update_normal_vectors |
              dealii::update_mapping);


      std::vector<dealii::types::global_dof_index> global_dofs_on_cell_stokes;
      std::vector<dealii::types::global_dof_index> global_dofs_on_cell_solid;

      dealii::FullMatrix<double> local_interface_matrix(
          context.pimpl_->finite_elements.max_dofs_per_cell(),
          stokes_problem.get_dof_handler().get_fe_collection().max_dofs_per_cell());

      for (const auto& interface_pair : matcher.interface_range())
      {
        const auto& cell_range = (*interface_pair).first.cell;
        const auto& cell_domain =
            (*interface_pair)
                .second.cell->as_dof_handler_iterator(stokes_problem.get_dof_handler());

        FOUR_C_ASSERT(cell_range->face((*interface_pair).first.face)->center() ==
                          cell_domain->face((*interface_pair).second.face)->center(),
            "Centers must match");

        fe_values_solid.reinit(cell_range, (*interface_pair).first.face);
        fe_values_stokes.reinit(cell_domain, (*interface_pair).second.face);

        const auto& phi_stokes = fe_values_stokes.get_present_fe_values();
        const auto& phi_solid = fe_values_solid.get_present_fe_values();

        global_dofs_on_cell_solid.resize(phi_solid.dofs_per_cell);
        global_dofs_on_cell_stokes.resize(phi_stokes.dofs_per_cell);
        local_interface_matrix.reinit(phi_solid.dofs_per_cell, phi_stokes.dofs_per_cell);

        const dealii::FEValuesExtractors::Vector velocities(0);
        const dealii::FEValuesExtractors::Scalar pressure(dim);

        fe_values_solid.get_dof_indices_four_c_ordering(global_dofs_on_cell_solid);
        const auto& local_indexing = fe_values_solid.local_four_c_indexing();

        cell_domain->get_dof_indices(global_dofs_on_cell_stokes);

        for (auto q_index : phi_stokes.quadrature_point_indices())
        {
          const dealii::Tensor<1, dim> normal_vector = phi_solid.normal_vector(q_index);
          // local assembly loop
          for (unsigned int i = 0; i < phi_solid.dofs_per_cell; ++i)
          {
            for (const auto j : phi_stokes.dof_indices())
            {
              const auto stokes_sym_grad = phi_stokes[velocities].symmetric_gradient(j, q_index);
              const auto stokes_pressure = phi_stokes[pressure].value(j, q_index);
              dealii::Tensor<1, dim> solid_value_vector;
              for (unsigned int c = 0; c < dim; ++c)
              {
                solid_value_vector[c] =
                    phi_solid.shape_value_component(local_indexing[i], q_index, c);
              }
              local_interface_matrix(i, j) -=
                  (2 * stokes_problem.get_viscosity() * (stokes_sym_grad * normal_vector) -
                      stokes_pressure * normal_vector) *
                  solid_value_vector * phi_stokes.JxW(q_index);
            }
          }
        }
        interface_matrix.add(
            global_dofs_on_cell_solid, global_dofs_on_cell_stokes, local_interface_matrix);
      }
    }


    /*// update rhs
    {
      dealii::QGauss<dim - 1> quadrature_gauss(5);
      const auto& quadrature = quadrature_gauss;
      dealii::hp::QCollection<dim - 1> quadrature_collection(quadrature);

      DealiiWrappers::FEFaceValuesContext<dim> fe_values_solid(context, *structdis,
          quadrature_collection,
          dealii::update_values | dealii::update_JxW_values | dealii::update_normal_vectors);


      std::vector<dealii::types::global_dof_index> global_dofs_on_cell_solid;
      dealii::Vector<double> solid_rhs_vector(context.pimpl_->finite_elements.max_dofs_per_cell());
      for (const auto& interface_pair : matcher.interface_range())
      {
        const auto& cell_range = (*interface_pair).first.cell;

        fe_values_solid.reinit(cell_range, (*interface_pair).first.face);
        const auto& phi_solid = fe_values_solid.get_present_fe_values();

        global_dofs_on_cell_solid.resize(phi_solid.dofs_per_cell);
        const dealii::FEValuesExtractors::Vector solid_displacement(0);

        fe_values_solid.get_dof_indices_four_c_ordering(global_dofs_on_cell_solid);
        const auto& local_indexing = fe_values_solid.local_four_c_indexing();

        dealii::Tensor<1, 2> constant_force_vector(
            {0.01, 0.0});  // constant force from left to right
        for (auto q_index : phi_solid.quadrature_point_indices())
        {
          // local assembly loop
          for (const auto i : phi_solid.dof_indices())
          {
            dealii::Tensor<1, dim> solid_value_vector;
            for (unsigned int c = 0; c < dim; ++c)
            {
              solid_value_vector[c] =
                  phi_solid.shape_value_component(local_indexing[i], q_index, c);
            }
            solid_rhs_vector[i] +=
                constant_force_vector * solid_value_vector * phi_solid.JxW(q_index);
          }
        }
        solid_rhs.add(global_dofs_on_cell_solid, solid_rhs_vector);
      }
    }*/

    dealii::TrilinosWrappers::SparseMatrix fluid_block, solid_block, interface_block, zero_block;
    fluid_block.reinit(fluid_system_matrix);
    solid_block.reinit(*matrix->epetra_matrix());
    interface_block.reinit(interface_matrix);
    auto zero_matrix = std::make_shared<Epetra_CrsMatrix>(Copy,
        fluid_block.trilinos_matrix().RowMap(), solid_block.trilinos_matrix().RowMap(), 0, true);
    zero_matrix->FillComplete(
        solid_block.trilinos_matrix().RangeMap(), fluid_block.trilinos_matrix().RangeMap());


    InputType blocks(2);
    blocks[0].resize(2);
    blocks[0][0] = &fluid_block.trilinos_matrix();
    blocks[0][1] = &zero_block.trilinos_matrix();
    blocks[1].resize(2);
    blocks[1][0] = &interface_block.trilinos_matrix();
    blocks[1][1] = &solid_block.trilinos_matrix();


    auto full_matrix = build_monolithic(blocks);
    full_matrix->FillComplete();

    dealii::TrilinosWrappers::SparseMatrix system_matrix;
    system_matrix.reinit(*full_matrix);

    dealii::TrilinosWrappers::SolverDirect solver;
    solver.initialize(system_matrix);

    dealii::LinearAlgebra::distributed::Vector<double> rhs_full, solution_full;
    rhs_full.reinit(fluid_rhs.size() + solid_rhs.size());
    solution_full.reinit(fluid_rhs.size() + solid_rhs.size());
    for (unsigned int i = 0; i < fluid_rhs.size(); ++i)
    {
      rhs_full(i) = fluid_rhs(i);
    }
    for (unsigned int i = 0; i < solid_rhs.size(); ++i)
    {
      rhs_full(i + fluid_rhs.size()) = solid_rhs(i);
    }

    solver.vmult(solution_full, rhs_full);
    // copy the solution back to the output vector
    dealii::Vector<double> fluid_output(fluid_rhs.size());
    Core::LinAlg::Vector<double> solid_output(*rhs);

    for (unsigned int i = 0; i < fluid_rhs.size(); ++i)
    {
      fluid_output(i) = solution_full(i);
    }
    stokes_problem.get_constraints().distribute(fluid_output);
    stokes_problem.output_results(fluid_output);

    for (unsigned int i = 0; i < solid_rhs.size(); ++i)
    {
      solid_output[i] = solution_full(i + fluid_rhs.size());
    }



    const auto visualization_writer =
        std::make_unique<Core::IO::DiscretizationVisualizationWriterMesh>(
            structdis, Core::IO::visualization_parameters_factory(
                           Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                           *Global::Problem::instance()->output_control_file(), 0));


    visualization_writer->append_result_data_vector_with_context(
        solid_output, Core::IO::OutputEntity::dof, {"solid_rhs"});

    visualization_writer->append_result_data_vector_with_context(solid_output,
        Core::IO::OutputEntity::dof, {"displacement_x", "displacement_y", std::nullopt});

    visualization_writer->append_result_data_vector_with_context(
        solid_output, Core::IO::OutputEntity::dof, {"displacement", "displacement", std::nullopt});
    visualization_writer->write_to_disk(0.0, 0);
  }
}  // namespace DealiiFSI
FOUR_C_NAMESPACE_CLOSE
