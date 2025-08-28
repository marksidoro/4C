// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later


#include "4C_config.hpp"

#include "4C_deal_ii_fsi_main.hpp"

#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_fsi_deal_ii_assembler.hpp"
#include "4C_deal_ii_fsi_tools.hpp"
#include "4C_deal_ii_mapping.hpp"
#include "4C_deal_ii_triangulation.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_discretization_nullspace.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include "4C_io_discretization_visualization_writer_mesh.hpp"
#include "4C_linalg_sparsematrix.hpp"
#include "4C_utils_exceptions.hpp"

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/multigrid.templates.h>
#include <deal.II/numerics/data_out.h>
#include <MMG/Fluid/Details/dg_fluid_monolithic.hpp>
#include <MMG/Fluid/geometry/base_geometry_setups.hpp>
#include <MMG/Fluid/solver/block_schwarz_pc.hpp>
#include <MMG/Fluid/solver/mg_hierarchy.hpp>
#include <MMG/Multigrid/Block/coarse_grid/direct_solver.hpp>
#include <MMG/Multigrid/Block/coarse_grid/no_solver.hpp>
#include <MMG/Multigrid/Block/operator/operator_base.hpp>
#include <MMG/Multigrid/Block/smoother/block_gauss_seidel.hpp>
#include <MMG/Multigrid/Block/smoother/block_jacobi.hpp>
#include <MMG/Multigrid/Block/transfer/transfer_base.hpp>
#include <MMG/Multigrid/SingleField/muelu_wrapper.hpp>


FOUR_C_NAMESPACE_OPEN

namespace DealiiFSI
{


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



  template <int dim, typename number>
  void output_results(const std::vector<const dealii::DoFHandler<dim>*>& dof_handler,
      const MMG::BlockVectorType<number>& solution, unsigned int refinement_cycle,
      std::string filename = "solution")
  {
    using namespace dealii;
    {
      DataOut<dim> data_out;
      std::vector<std::string> solution_names(dim, "velocity");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation(
              dim, DataComponentInterpretation::component_is_part_of_vector);
      data_out.attach_dof_handler(*dof_handler[0]);
      data_out.add_data_vector(solution.block(0), solution_names, DataOut<dim>::type_dof_data,
          data_component_interpretation);
      data_out.build_patches(1);
      data_out.write_vtu_with_pvtu_record(
          "./", "velocity_" + filename, refinement_cycle, MPI_COMM_WORLD);
    }

    {
      DataOut<dim> data_out;
      std::vector<std::string> solution_names(1, "pressure");
      std::vector<DataComponentInterpretation::DataComponentInterpretation>
          data_component_interpretation;
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);
      data_out.attach_dof_handler(*dof_handler[1]);
      data_out.add_data_vector(solution.block(1), solution_names, DataOut<dim>::type_dof_data,
          data_component_interpretation);
      data_out.build_patches(1);
      data_out.write_vtu_with_pvtu_record(
          "./", "pressure_" + filename, refinement_cycle, MPI_COMM_WORLD);
    }
  }


  void output_results(const Core::LinAlg::Vector<double>& solution_solid,
      Core::IO::DiscretizationVisualizationWriterMesh& writer, unsigned int index,
      std::string field_name = "solution")
  {
    writer.append_result_data_vector_with_context(solution_solid, Core::IO::OutputEntity::dof,
        {field_name + "_x", field_name + "_y", std::nullopt});
    writer.write_to_disk(0.0, index);
  }



  void run()
  {
    constexpr int dim = 2;
    using number = double;
    const number viscosity = 1.0;

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

    dealii::Triangulation<dim> solid_tria;
    auto context = DealiiWrappers::create_triangulation(solid_tria, *structdis);

    auto nullspace = Core::FE::compute_null_space(*structdis, 2, 3, *structdis->dof_col_map());
    auto ns_eptra_rcp = Teuchos::rcpFromRef(nullspace->get_epetra_multi_vector());
    Teuchos::RCP<MMG::TRILINOS::TYPES::MultiVector> nullspace_mv =
        Teuchos::RCP(new MMG::TRILINOS::TYPES::internal::Xpetra_EpertaMultiVector(ns_eptra_rcp));

    auto matrix = structadapter->system_matrix();

    MMG::TRILINOS::SparseMatrixInterface mmg_solid_matrix;
    dealii::TrilinosWrappers::SparseMatrix deal_solid_matrix;
    deal_solid_matrix.reinit(*matrix->epetra_matrix());
    mmg_solid_matrix.copy_from(deal_solid_matrix);

    std::cout << mmg_solid_matrix.n_col_blocks() << " " << mmg_solid_matrix.n_row_blocks()
              << std::endl;

    std::cout << "Solid matrix size: " << mmg_solid_matrix.trilinos_ref().getGlobalNumRows()
              << " x " << mmg_solid_matrix.trilinos_ref().getGlobalNumCols() << std::endl;

    MMG::MB::MueLuMultigrid<number> mue_lu_multigrid;
    MMG::MB::MueLuMultigrid<number>::AdditionalData mue_lu_data;
    mue_lu_data.max_coarse_size = std::min(
        static_cast<unsigned long>(2000), mmg_solid_matrix.trilinos_ref().getGlobalNumRows() / 10);
    mue_lu_data.block_parameters[0].transfer_type =
        MMG::MB::MueLuMultigrid<number>::AdditionalData::TransferType::EnergyMinimization;
    mue_lu_data.block_parameters[0].pre_smoother =
        MMG::MB::MueLuMultigrid<number>::AdditionalData::SmootherType::ILU;
    mue_lu_data.block_parameters[0].post_smoother =
        MMG::MB::MueLuMultigrid<number>::AdditionalData::SmootherType::ILU;
    dealii::IndexSet local_dofs(structdis->dof_row_map()->get_epetra_block_map());
    mue_lu_data.block_parameters[0].block_indices = local_dofs;
    mue_lu_data.block_parameters[0].nullspace = nullspace_mv;
    mue_lu_data.block_parameters[0].n_equations = dim;


    mue_lu_multigrid.reinit(mmg_solid_matrix, mue_lu_data);

    auto rhs_original = structadapter->rhs();
    auto rhs = std::make_shared<Core::LinAlg::Vector<double>>(*rhs_original);
    MMG::VectorType<number> solid_rhs(rhs->global_length());
    for (unsigned int i = 0; i < solid_rhs.size(); ++i)
    {
      solid_rhs(i) = (*rhs)[i];
    }

    /// ---------------------------------------------------------------------------------------
    // Fluid Problem Setup:

    dealii::Triangulation<dim> fluid_domain_tria;
    MMG::Fluid::GeometrySetups::step_46_fluid(fluid_domain_tria);

    {
      dealii::Triangulation<dim> solid_domain_tria;
      MMG::Fluid::GeometrySetups::step_46_fluid(solid_domain_tria, false);
      unsigned int max_refine = 5;
      for (unsigned int level = 0; level < max_refine; ++level)
      {
        MMG::Fluid::GeometrySetups::set_boundary_ids_solid(solid_domain_tria);
        MMG::Fluid::GeometrySetups::write_to_four_c_input_file(
            "geometry_solid_" + std::to_string(level) + ".inp", solid_domain_tria);
        if (level < max_refine - 1) solid_domain_tria.refine_global(1);  // refine 4 times
      }
    }

    unsigned int n_elements_solid = structdis->num_global_elements();
    std::cout << "Number of structure elements: " << n_elements_solid << std::endl;
    unsigned int refine = -1;  // default
    switch (n_elements_solid)
    {
      case 24:
        refine = 0;
        break;
      case 96:
        refine = 1;
        break;
      case 384:
        refine = 2;
        break;
      case 1536:
        refine = 3;
        break;
      case 6144:
        refine = 4;
        break;
      default:
        FOUR_C_THROW("NOT IMPLEMENTED FOR THIS NUMBER OF STRUCTURE ELEMENTS");
    }
    std::vector<std::shared_ptr<dealii::Triangulation<dim>>> triangulations;
    MMG::Fluid::GeometrySetups::setup_geometry(
        triangulations, fluid_domain_tria, refine, MPI_COMM_WORLD);
    for (auto& tria : triangulations)
    {
      MMG::Fluid::GeometrySetups::set_boundary_ids(*tria);
    }
    const auto& fine_fluid_tria = *triangulations.back();



    const int max_degree = 3;
    MMG::Fluid::MGHierarchy<dim, number, MMG::Fluid::IncNSMonolithicDG> mg_hierarchy(
        max_degree, triangulations);
    using Details =
        typename MMG::Fluid::MGHierarchy<dim, number, MMG::Fluid::IncNSMonolithicDG>::Details;
    const unsigned int min_level = mg_hierarchy.min_level();
    const unsigned int max_level = mg_hierarchy.max_level();


    auto setup_boundary_conditions =
        [](std::vector<MMG::Fluid::BoundaryCondition<dim, number>>& boundary_conditions)
    {
      boundary_conditions.clear();
      boundary_conditions.resize(3);
      boundary_conditions[0] = {std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim + 1),
          MMG::Fluid::BoundaryType::Dirichlet};
      boundary_conditions[1] = {
          std::make_shared<MMG::Fluid::GeometrySetups::Step46TopBoundary<dim>>(),
          MMG::Fluid::BoundaryType::Dirichlet};
      boundary_conditions[2] = {std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim + 1),
          MMG::Fluid::BoundaryType::None};
    };


    mg_hierarchy.setup_operators(setup_boundary_conditions);
    mg_hierarchy.setup_transfer();
    mg_hierarchy.assemble_coarse_matrix();


    dealii::MGLevelObject<MMG::Fluid::UniformBlockSmoother<dim, number, Details::CellIntegrator>>
        mg_block_pc(min_level, max_level);
    for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_block_pc[level].initialize(mg_hierarchy.get_matrix_free(level), &Details::cell_integral,
          &Details::flux_integral_local_cell, &mg_hierarchy.get_operator(level).get_details());
    }


    /*dealii::MGLevelObject<MMG::Fluid::IndividualBlockSmoother<dim, number,
    Details::CellIntegrator>> mg_block_pc(min_level, max_level); for (unsigned int level =
    min_level; level <= max_level; ++level)
    {
      mg_block_pc[level].initialize(mg_hierarchy.get_matrix_free(level), &Details::cell_integral,
          &Details::flux_integral_local_cell, &Details::boundary_integral_local_cell,
          &mg_hierarchy.get_operator(level).get_details());
    }
    */


    using pc_chebychev_full = dealii::PreconditionChebyshev<MMG::MF::OperatorMF<Details>,
        MMG::BlockVectorType<number>, dealii::LinearOperator<MMG::BlockVectorType<number>>>;
    dealii::MGLevelObject<pc_chebychev_full::AdditionalData> mg_smoother_data(min_level, max_level);
    for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_smoother_data[level].degree = 4;
      mg_smoother_data[level].smoothing_range = 15.0;
      mg_smoother_data[level].eig_cg_n_iterations = 10;
      mg_smoother_data[level].eigenvalue_algorithm =
          dealii::internal::EigenvalueAlgorithm::power_iteration;
      mg_smoother_data[level].preconditioner =
          std::make_shared<dealii::LinearOperator<MMG::BlockVectorType<number>>>();
      mg_smoother_data[level].preconditioner->vmult =
          [&mg_block_pc, level](auto& dst, const auto& src) { mg_block_pc[level].vmult(dst, src); };
    }
    using MGSmoother = dealii::MGSmootherPrecondition<MMG::MF::OperatorMF<Details>,
        pc_chebychev_full, MMG::BlockVectorType<number>>;


    /*
    using pc_relaxation = dealii::PreconditionRelaxation<MMG::MF::OperatorMF<Details>,
        dealii::LinearOperator<MMG::BlockVectorType<number>>>;
    dealii::MGLevelObject<pc_relaxation::AdditionalData> mg_smoother_data(min_level, max_level);
    for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_smoother_data[level].relaxation = 0.8;
      mg_smoother_data[level].eig_cg_n_iterations = 0;
      mg_smoother_data[level].preconditioner =
          std::make_shared<dealii::LinearOperator<MMG::BlockVectorType<number>>>();
      mg_smoother_data[level].preconditioner->vmult =
          [&mg_block_pc, level](auto& dst, const auto& src) { mg_block_pc[level].vmult(dst, src); };
    }
    using MGSmoother = dealii::MGSmootherPrecondition<MMG::MF::OperatorMF<Details>, pc_relaxation,
        MMG::BlockVectorType<number>>;*/


    auto mg_smoother = std::make_shared<MGSmoother>();
    mg_smoother->initialize(mg_hierarchy.get_operators(), mg_smoother_data);
    mg_hierarchy.pre_smoother = mg_smoother;
    mg_hierarchy.post_smoother = mg_smoother;


    /// ---------------------------------------------------------------------------------------
    // build up the transfer objects:

    auto mapping = DealiiWrappers::MappingContext<dim>::create_linear_mapping(context);
    InterfaceMatcher<dim> matcher =
        DealiiFSI::make_interface_matcher_across_boundary(solid_tria, fine_fluid_tria, 1e-14);

    const auto n_dofs_vel = mg_hierarchy.get_dof_handler_velocity(max_level, true).n_dofs();
    const auto n_dofs_pres = mg_hierarchy.get_dof_handler_velocity(max_level, false).n_dofs();

    dealii::DynamicSparsityPattern dsp_vel(rhs->global_length(), n_dofs_vel);
    make_interface_sparsity_pattern(
        matcher, context, mg_hierarchy.get_dof_handler_velocity(max_level, true), dsp_vel);
    dealii::DynamicSparsityPattern dsp_pres(rhs->global_length(), n_dofs_pres);
    make_interface_sparsity_pattern(
        matcher, context, mg_hierarchy.get_dof_handler_velocity(max_level, false), dsp_pres);
    dealii::TrilinosWrappers::SparseMatrix interface_matrix_vel, interface_matrix_pres;
    interface_matrix_vel.reinit(dsp_vel);
    interface_matrix_pres.reinit(dsp_pres);

    auto assemble_interface = [&](dealii::TrilinosWrappers::SparseMatrix& interface_matrix,
                                  const dealii::DoFHandler<dim>& dof_handler)
    {
      dealii::QGauss<dim - 1> quadrature_gauss(5);
      const auto& quadrature = quadrature_gauss;
      dealii::hp::QCollection<dim - 1> quadrature_collection(quadrature);


      dealii::FEFaceValues<dim> fe_values_stokes(dof_handler.get_fe(), quadrature,
          dealii::update_values | dealii::update_gradients | dealii::update_JxW_values |
              dealii::update_quadrature_points | dealii::update_normal_vectors);

      DealiiWrappers::FEFaceValuesContext<dim> fe_values_solid(mapping.get_mapping_collection(),
          context, quadrature_collection,
          dealii::update_values | dealii::update_JxW_values | dealii::update_normal_vectors |
              dealii::update_mapping);


      std::vector<dealii::types::global_dof_index> global_dofs_on_cell_stokes;
      std::vector<dealii::types::global_dof_index> global_dofs_on_cell_solid;

      dealii::FullMatrix<double> local_interface_matrix(
          context.get_finite_elements().max_dofs_per_cell(),
          dof_handler.get_fe_collection().max_dofs_per_cell());

      for (const auto& interface_pair : matcher.interface_range())
      {
        const auto& cell_range = (*interface_pair).first.cell;
        const auto& cell_domain =
            (*interface_pair).second.cell->as_dof_handler_iterator(dof_handler);

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

        const bool velocity_part = phi_stokes.get_fe().n_components() > 1;

        const dealii::FEValuesExtractors::Vector velocities(0);
        const dealii::FEValuesExtractors::Scalar pressure(0);

        fe_values_solid.get_dof_indices_four_c_ordering(global_dofs_on_cell_solid);
        const auto& local_indexing = fe_values_solid.local_four_c_indexing();

        cell_domain->get_dof_indices(global_dofs_on_cell_stokes);
        dealii::SymmetricTensor<dim, dim, number> stokes_sym_grad;
        stokes_sym_grad = 0;
        number stokes_pressure = 0;

        for (auto q_index : phi_stokes.quadrature_point_indices())
        {
          const dealii::Tensor<1, dim> normal_vector = phi_solid.normal_vector(q_index);
          // local assembly loop
          for (unsigned int i = 0; i < phi_solid.dofs_per_cell; ++i)
          {
            for (const auto j : phi_stokes.dof_indices())
            {
              if (velocity_part)
                stokes_sym_grad = phi_stokes[velocities].symmetric_gradient(j, q_index);
              else
                stokes_pressure = phi_stokes[pressure].value(j, q_index);

              dealii::Tensor<1, dim> solid_value_vector;
              for (unsigned int c = 0; c < dim; ++c)
              {
                solid_value_vector[c] =
                    phi_solid.shape_value_component(local_indexing[i], q_index, c);
              }
              local_interface_matrix(i, j) -= (2.0 * viscosity * (stokes_sym_grad * normal_vector) +
                                                  stokes_pressure * normal_vector) *
                                              solid_value_vector * phi_stokes.JxW(q_index);
            }
          }
        }
        interface_matrix.add(
            global_dofs_on_cell_solid, global_dofs_on_cell_stokes, local_interface_matrix);
      }
    };

    assemble_interface(
        interface_matrix_vel, mg_hierarchy.get_dof_handler_velocity(max_level, true));

    assemble_interface(
        interface_matrix_pres, mg_hierarchy.get_dof_handler_velocity(max_level, false));

    MMG::TRILINOS::SparseMatrixInterface single_matrix_vel, single_matrix_pres;
    single_matrix_vel.copy_from(interface_matrix_vel);
    single_matrix_pres.copy_from(interface_matrix_pres);
    MMG::TRILINOS::SparseMatrixInterface interface_matrix(1, 2);
    interface_matrix.trilinos_rcp(0, 0) = single_matrix_vel.trilinos_rcp();
    interface_matrix.trilinos_rcp(0, 1) = single_matrix_pres.trilinos_rcp();


    /// ---------------------------------------------------------------------------------------
    auto mg_solid_operator = mue_lu_multigrid.get_mg_operator();
    MMG::TOOLS::LevelMap level_map(
        {{mg_hierarchy.min_level(), mg_hierarchy.max_level()},
            {mue_lu_multigrid.min_level(), mue_lu_multigrid.max_level()}},
        MMG::TOOLS::LevelMap::LevelMapType::AppendFine);

    MMG::BlockMGMatrix<MMG::BlockVectorType<number>, MMG::VectorType<number>> mg_block_matrix(
        mg_hierarchy.get_mg_operator(), mg_solid_operator, level_map);

    auto mg_solid_fluid_coupling = std::make_shared<
        MMG::MGCouplingMatrix<MMG::VectorType<number>, MMG::BlockVectorType<number>>>(
        *mue_lu_multigrid.get_mg_transfer(), level_map.get_block(1), interface_matrix,
        level_map.get_block(0), *mg_hierarchy.get_mg_transfer());

    auto mg_fluid_solid_coupling = std::make_shared<
        MMG::MGCouplingZeroBlock<MMG::BlockVectorType<number>, MMG::VectorType<number>>>(
        *mg_hierarchy.get_mg_transfer(), level_map.get_block(0), level_map.get_block(1),
        *mue_lu_multigrid.get_mg_transfer());

    mg_block_matrix.set_coupling_operator<1, 0>(mg_solid_fluid_coupling);
    mg_block_matrix.set_coupling_operator<0, 1>(mg_fluid_solid_coupling);

    MMG::BlockMGTransfer<MMG::BlockVectorType<number>, MMG::VectorType<number>> mg_block_transfer(
        level_map);
    mg_block_transfer.reinit_block<0>(0, mg_hierarchy.get_mg_transfer());
    mg_block_transfer.reinit_block<1>(1, mue_lu_multigrid.get_mg_transfer());


    number smoother_dumping = 1.0;
    // MMG::BlockGaussSeidelSmoother<MMG::BlockVectorType<number>, MMG::VectorType<number>>
    MMG::BlockJacobiSmoother<MMG::BlockVectorType<number>, MMG::VectorType<number>>
        mg_block_smoother_pre(mg_block_matrix, level_map);
    mg_block_smoother_pre.reinit_block(0, mg_hierarchy.get_mg_pre_smoother());
    mg_block_smoother_pre.reinit_block(1, mue_lu_multigrid.get_mg_pre_smoother());

    const auto& mg_block_smoother_post = mg_block_smoother_pre;

    dealii::Table<2, MMG::TRILINOS::SparseMatrixInterface> coarse_matrices(2, 2);
    coarse_matrices(0, 0) = mg_hierarchy.get_coarse_matrix();
    coarse_matrices(0, 1) = mg_fluid_solid_coupling->get_coarse_matrix(min_level);
    coarse_matrices(1, 0) = mg_solid_fluid_coupling->get_coarse_matrix(min_level);
    coarse_matrices(1, 1) = mue_lu_multigrid.get_coarse_matrix();
    auto coarse_block_matrix = MMG::TRILINOS::BlockMatrixInterface<MMG::BlockVectorType<number>,
        MMG::VectorType<number>>::combine_matrices(coarse_matrices);

    {
      MMG::MultiBlockVector<MMG::BlockVectorType<number>, MMG::VectorType<number>> rhs_coarse,
          solution_coarse_op, solution_coarse;

      MMG::TOOLS::initialize_dof_vector(
          min_level, level_map, rhs_coarse, mg_hierarchy, mue_lu_multigrid);
      MMG::TOOLS::initialize_dof_vector(
          min_level, level_map, solution_coarse_op, mg_hierarchy, mue_lu_multigrid);
      MMG::TOOLS::initialize_dof_vector(
          min_level, level_map, solution_coarse, mg_hierarchy, mue_lu_multigrid);

      MMG::TOOLS::set_random_entries(rhs_coarse);
      coarse_block_matrix.vmult(solution_coarse, rhs_coarse);

      mg_block_matrix.vmult(min_level, solution_coarse_op, rhs_coarse);
      solution_coarse -= solution_coarse_op;
      std::cout << "Coarse solution norm: " << solution_coarse.l2_norm() << std::endl;
      std::cout << "Coarse solution operator norm: " << solution_coarse_op.l2_norm() << std::endl;
    }


    // build the coarse matrix...
    MMG::BlockCoarseGridDirectSolve<MMG::BlockVectorType<number>, MMG::VectorType<number>>
        mg_block_coarse_grid(coarse_block_matrix);

    MMG::MultiBlockVector<MMG::BlockVectorType<number>, MMG::VectorType<number>> rhs_coupled,
        solution_coupled;

    MMG::TOOLS::initialize_dof_vector(
        max_level, level_map, rhs_coupled, mg_hierarchy, mue_lu_multigrid);
    MMG::TOOLS::initialize_dof_vector(
        max_level, level_map, solution_coupled, mg_hierarchy, mue_lu_multigrid);

    dealii::Multigrid multigrid(mg_block_matrix, mg_block_coarse_grid, mg_block_transfer,
        mg_block_smoother_pre, mg_block_smoother_post);


    for (unsigned int level = level_map.min_level(); level <= level_map.max_level(); ++level)
    {
      MMG::TOOLS::initialize_dof_vector(
          level, level_map, multigrid.defect[level], mg_hierarchy, mue_lu_multigrid);
    }



    dealii::LinearOperator<
        MMG::MultiBlockVector<MMG::BlockVectorType<number>, MMG::VectorType<number>>>
        mg_pc;
    mg_pc.vmult = [&](auto& dst, const auto& src)
    {
      multigrid.defect[max_level] = src;
      for (unsigned int level = level_map.min_level(); level < level_map.max_level(); ++level)
      {
        multigrid.defect[level] = 0;
      }
      multigrid.cycle();
      dst = multigrid.solution[max_level];
    };

    dealii::LinearOperator<
        MMG::MultiBlockVector<MMG::BlockVectorType<number>, MMG::VectorType<number>>>
        fine_operator;
    fine_operator.vmult = [&](auto& dst, const auto& src)
    { mg_block_matrix.vmult(max_level, dst, src); };

    MMG::vector_assembly_loop(mg_hierarchy.get_matrix_free(max_level),
        rhs_coupled.template block<0>(), &Details::assemble_cell, &Details::assemble_flux,
        &Details::assemble_boundary, &mg_hierarchy.get_operator(max_level).get_details());
    rhs_coupled.template block<1>() = solid_rhs;
    std::cout << "RHS norm: " << rhs_coupled.l2_norm() << std::endl;


    mg_block_matrix.vmult(max_level, solution_coupled, rhs_coupled);
    std::cout << "Solution norm: " << solution_coupled.l2_norm() << std::endl;

    mg_pc.vmult(solution_coupled, rhs_coupled);
    std::cout << "Solution norm: " << solution_coupled.l2_norm() << std::endl;

    std::vector<const dealii::DoFHandler<dim>*> dof_handlers = {
        &mg_hierarchy.get_dof_handler_velocity(max_level, true),
        &mg_hierarchy.get_dof_handler_velocity(max_level, false)};
    Core::LinAlg::Vector<double> solid_output(*rhs);


    /*
    {
      auto dst_solid = solution_coupled.template block<1>();
      auto src_solid = rhs_coupled.template block<1>();

      MMG::MultiBlockVector<MMG::BlockVectorType<number>> dst_fluid, src_fluid;
      dst_fluid.template block<0>() = solution_coupled.template block<0>();
      src_fluid.template block<0>() = rhs_coupled.template block<0>();


      mg_hierarchy.assemble_coarse_matrix(max_level);
      dealii::Table<2, MMG::TRILINOS::SparseMatrixInterface> fine_matrices(1, 1);
      fine_matrices[0][0] = mg_hierarchy.get_coarse_matrix();
      auto fine_matrix_fluid =
          MMG::TRILINOS::BlockMatrixInterface<MMG::BlockVectorType<number>>::combine_matrices(
              fine_matrices);


      MMG::BlockCoarseGridDirectSolve fluid_direct_solve(fine_matrix_fluid);
      fluid_direct_solve(0, dst_fluid, src_fluid);
      interface_matrix.vmult_add(src_solid, dst_fluid.template block<0>());
      auto solver_connector =
          [&](const unsigned int iteration, const double check_value, const auto& current_iterate)
      {
        (void)current_iterate;
        std::cout << "Iteration: " << iteration << " Error: " << check_value << std::endl;
        return dealii::SolverControl::success;
      };

      dealii::SolverControl solver_control(200, 1e-6);
      dealii::SolverBicgstab<MMG::VectorType<number>> solver(solver_control);
      solver.connect(solver_connector);

      try
      {
        solution_coupled = 0;
        solver.solve(mmg_solid_matrix, dst_solid, src_solid, mue_lu_multigrid);
        std::cout << "Solver converged after: " << solver_control.last_step()
                  << " steps with residual: " << solver_control.last_value() << std::endl;
      }
      catch (const dealii::SolverControl::NoConvergence& e)
      {
        std::cout << "Solver failed after: " << solver_control.last_step()
                  << " steps with residual: " << solver_control.last_value() << std::endl;
      }
    }
    */



    /*
    for (unsigned int i = 0; i < solid_rhs.size(); ++i)
    {
      solid_output[i] = solution_coupled.template block<1>()[i];
    }

    */



    const auto visualization_writer =
        std::make_unique<Core::IO::DiscretizationVisualizationWriterMesh>(
            structdis, Core::IO::visualization_parameters_factory(
                           Global::Problem::instance()->io_params().sublist("RUNTIME VTK OUTPUT"),
                           *Global::Problem::instance()->output_control_file(), 0));



    auto solver_connector =
        [&](const unsigned int iteration, const double check_value, const auto& current_iterate)
    {
      (void)current_iterate;
      std::cout << "Iteration: " << iteration << " Error: " << check_value << std::endl;


      /*
      for (unsigned int i = 0; i < solid_rhs.size(); ++i)
      {
        solid_output[i] = current_iterate.template block<1>()[i];
      }
      output_results(solid_output, *visualization_writer, iteration, "solver_step_solid");
      output_results(
          dof_handlers, current_iterate.template block<0>(), iteration, "solver_step_fluid");*/
      return dealii::SolverControl::success;
    };



    dealii::ReductionControl solver_control(100, 1e-14, 1e-8);
    dealii::SolverGMRES<
        MMG::MultiBlockVector<MMG::BlockVectorType<number>, MMG::VectorType<number>>>
        solver(solver_control);
    solver.connect(solver_connector);

    mg_pc.vmult(solution_coupled, rhs_coupled);
    for (unsigned int i = 0; i < solid_rhs.size(); ++i)
    {
      solid_output[i] = solution_coupled.template block<1>()[i];
    }
    output_results(solid_output, *visualization_writer, 0, "mg_step_solid");
    output_results(dof_handlers, solution_coupled.template block<0>(), 0, "mg_step_fluid");

    dealii::Timer timer;
    timer.start();
    try
    {
      solution_coupled = 0;
      solver.solve(fine_operator, solution_coupled, rhs_coupled, mg_pc);
      std::cout << "Solver converged after: " << solver_control.last_step()
                << " steps with residual: " << solver_control.last_value() << std::endl;
    }
    catch (const dealii::SolverControl::NoConvergence& e)
    {
      std::cout << "Solver failed after: " << solver_control.last_step()
                << " steps with residual: " << solver_control.last_value() << std::endl;
    }
    timer.stop();
    for (unsigned int i = 0; i < solid_rhs.size(); ++i)
    {
      solid_output[i] = solution_coupled.template block<1>()[i];
    }
    output_results(solid_output, *visualization_writer, 0, "solution_solid");
    output_results(dof_handlers, solution_coupled.template block<0>(), 0, "solution_fluid");
    std::cout << "Time for solve: " << timer.wall_time() << " seconds." << std::endl;
    std::cout << "Problem size: " << solution_coupled.size() << std::endl;
    std::cout << "  Fluid dofs: " << solution_coupled.block<0>().size() << " --- "
              << mg_hierarchy.get_dof_handler_velocity(max_level, true).n_dofs() << " / "
              << mg_hierarchy.get_dof_handler_velocity(max_level, false).n_dofs() << std::endl;
    std::cout << "  Solid dofs: " << solution_coupled.block<1>().size() << std::endl;
  }
}  // namespace DealiiFSI
FOUR_C_NAMESPACE_CLOSE
