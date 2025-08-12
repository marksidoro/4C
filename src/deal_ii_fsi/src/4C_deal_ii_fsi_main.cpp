// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later


#include "4C_config.hpp"

#include "4C_deal_ii_fsi_main.hpp>

#include "4C_adapter_str_factory.hpp>
#include "4C_adapter_str_structure.hpp>
#include "4C_adapter_str_structure_new.hpp>
#include "4C_deal_ii_context.hpp>
#include "4C_deal_ii_fsi_deal_ii_assembler.hpp>
#include "4C_deal_ii_fsi_tools.hpp>
#include "4C_deal_ii_mapping.hpp>
#include "4C_deal_ii_triangulation.hpp>
#include "4C_fem_condition_periodic.hpp>
#include "4C_fem_discretization.hpp>
#include "4C_global_data.hpp>
#include "4C_io.hpp>
#include "4C_io_discretization_visualization_writer_mesh.hpp>
#include "4C_linalg_sparsematrix.hpp>
#include "4C_utils_exceptions.hpp"

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
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/numerics/data_out.h>
#include <MMG/Fluid/Details/dg_fluid_monolithic.hpp>
#include <MMG/Fluid/geometry/base_geometry_setups.hpp>
#include <MMG/Fluid/solver/block_schwarz_pc.hpp>
#include <MMG/Fluid/solver/mg_hierarchy.hpp>
#include <MMG/Multigrid/Block/coarse_grid/direct_solver.hpp>
#include <MMG/Multigrid/Block/operator/operator_base.hpp>
#include <MMG/Multigrid/Block/smoother/block_jacobi.hpp>
#include <MMG/Multigrid/Block/transfer/transfer_base.hpp>
#include <MMG/Multigrid/SingleField/muelu_wrapper.hpp>


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

    auto matrix = structadapter->system_matrix();

    MMG::TRILINOS::SparseMatrixInterface mmg_solid_matrix;
    dealii::TrilinosWrappers::SparseMatrix deal_solid_matrix;
    deal_solid_matrix.reinit(*matrix->epetra_matrix());
    mmg_solid_matrix.copy_from(deal_solid_matrix);

    MMG::MB::MueLuMultigrid<number> mue_lu_multigrid;
    mue_lu_multigrid.reinit(mmg_solid_matrix);



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

    /*using pc_chebychev_full = dealii::PreconditionChebyshev< MMG::MF::OperatorMF< Details >,
    MMG::BlockVectorType< number >, dealii::LinearOperator< MMG::BlockVectorType< number > > >;
    dealii::MGLevelObject< pc_chebychev_full::AdditionalData > mg_smoother_data(min_level,
    max_level); for (unsigned int level = min_level; level <= max_level; ++level)
    {
      mg_smoother_data[level].degree                = 4;
      mg_smoother_data[level].smoothing_range       = 15.0;
      mg_smoother_data[level].eig_cg_n_iterations   = 10;
      mg_smoother_data[level].preconditioner        = std::make_shared< dealii::LinearOperator<
    MMG::BlockVectorType< number > > >(); mg_smoother_data[level].preconditioner->vmult =
    [&mg_block_pc, level](auto &dst, const auto &src) { mg_block_pc[level].vmult(dst, src);
      };
    }
    using MGSmoother = dealii::MGSmootherPrecondition< MMG::MF::OperatorMF< Details >,
    pc_chebychev_full, MMG::BlockVectorType< number > >;
    */


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
        MMG::BlockVectorType<number>>;


    auto mg_smoother = std::shared_ptr<MGSmoother>();
    mg_smoother->initialize(mg_hierarchy.get_operators(), mg_smoother_data);
    mg_hierarchy.pre_smoother = mg_smoother;
    mg_hierarchy.post_smoother = mg_smoother;


    /// ---------------------------------------------------------------------------------------
    // build up the transfer objects:
    dealii::Triangulation<dim> solid_tria;
    auto context = DealiiWrappers::create_triangulation(solid_tria, *structdis);
    auto mapping = DealiiWrappers::MappingContext<dim>::create_linear_mapping(context);
    InterfaceMatcher<dim> matcher =
        DealiiFSI::make_interface_matcher_across_boundary(solid_tria, fine_fluid_tria, 1e-14);

    const auto n_dofs_vel = mg_hierarchy.get_dof_handler_velocity(min_level, true).n_dofs();
    const auto n_dofs_pres = mg_hierarchy.get_dof_handler_velocity(min_level, false).n_dofs();

    dealii::DynamicSparsityPattern dsp_vel(rhs->global_length(), n_dofs_vel);
    make_interface_sparsity_pattern(
        matcher, context, mg_hierarchy.get_dof_handler_velocity(min_level, true), dsp_vel);
    dealii::DynamicSparsityPattern dsp_pres(rhs->global_length(), n_dofs_pres);
    make_interface_sparsity_pattern(
        matcher, context, mg_hierarchy.get_dof_handler_velocity(min_level, false), dsp_pres);
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
              local_interface_matrix(i, j) -= (2 * viscosity * (stokes_sym_grad * normal_vector) -
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
        interface_matrix_vel, mg_hierarchy.get_dof_handler_velocity(min_level, true));

    assemble_interface(
        interface_matrix_pres, mg_hierarchy.get_dof_handler_velocity(min_level, false));

    MMG::TRILINOS::SparseMatrixInterface single_matrix_vel, single_matrix_pres;
    single_matrix_vel.copy_from(interface_matrix_vel);
    single_matrix_pres.copy_from(interface_matrix_pres);
    MMG::TRILINOS::SparseMatrixInterface interface_matrix(2, 1);
    interface_matrix.trilinos_rcp(0, 0) = single_matrix_vel.trilinos_rcp();
    interface_matrix.trilinos_rcp(1, 0) = single_matrix_pres.trilinos_rcp();


    /// ---------------------------------------------------------------------------------------
    auto mg_solid_operator = mue_lu_multigrid.get_mg_operator();
    MMG::TOOLS::LevelMap level_map({{mg_hierarchy.min_level(), mg_hierarchy.min_level()},
        {mue_lu_multigrid.min_level(), mue_lu_multigrid.max_level()}});

    MMG::BlockMGMatrix<MMG::BlockVectorType<number>, MMG::VectorType<number>> mg_block_matrix(
        mg_hierarchy.get_mg_operator(), mg_solid_operator, level_map);

    auto mg_fluid_solid_coupling = std::make_shared<
        MMG::MGCouplingMatrix<MMG::BlockVectorType<number>, MMG::VectorType<number>>>(
        *mg_hierarchy.get_mg_transfer(), level_map.get_block(0), interface_matrix,
        level_map.get_block(1), *mue_lu_multigrid.get_mg_transfer());


    mg_block_matrix.set_coupling_operator<0, 1>(mg_fluid_solid_coupling);
    MMG::BlockMGTransfer<MMG::BlockVectorType<number>, MMG::VectorType<number>> mg_block_transfer(
        level_map);
    mg_block_transfer.reinit_block<0>(0, mg_hierarchy.get_mg_transfer());
    mg_block_transfer.reinit_block<1>(1, mue_lu_multigrid.get_mg_transfer());

    number smoother_dumping = 0.8;
    MMG::BlockJacobiSmoother<MMG::BlockVectorType<number>, MMG::VectorType<number>>
        mg_block_smoother_pre(mg_block_matrix, level_map, {smoother_dumping});

    const auto& mg_block_smoother_post = mg_block_smoother_pre;


    // build the coarse matrix...
    MMG::BlockCoarseGridDirectSolve<MMG::BlockVectorType<number>, MMG::VectorType<number>>*
        mg_block_coarse_grid;


    dealii::Multigrid multigrid(mg_block_matrix, *mg_block_coarse_grid, mg_block_transfer,
        mg_block_smoother_pre, mg_block_smoother_post);

    /*
    for (unsigned int level = level_map.min_level(); level <= level_map.max_level(); ++level)
    {
      mg_block_matrix.initialize_dof_vector(level, multigrid.defect[level]);
    }
    */

    /*
    dealii::LinearOperator<
        MMG::MultiBlockVector<MMG::BlockVectorType<number>, MMG::VectorType<number>>>
        mg_pc;
    mg_pc.vmult = [&multigrid](auto& dst, const auto& src)
    {
      multigrid.defect = src;
      multigrid.cycle();
      dst = multigrid.solution;
    };

    dealii::LinearOperator<
        MMG::MultiBlockVector<MMG::BlockVectorType<number>, MMG::VectorType<number>>>
        fine_operator;
    fine_operator.vmult = [&mg_block_matrix](auto& dst, const auto& src)
    { mg_block_matrix.vmult(max_level, dst, src); };

    dealii::SolverControl solver_control(200, 1e-6);
    dealii::SolverGMRES<
        MMG::MultiBlockVector<MMG::BlockVectorType<number>, MMG::VectorType<number>>>
        solver(solver_control);
        */

    // solver.solve(fine_operator, )



    /*
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
    visualization_writer->write_to_disk(0.0, 0);*/
  }
}  // namespace DealiiFSI
FOUR_C_NAMESPACE_CLOSE
