#ifndef INC_4C_DEAL_II_FSI_DEAL_II_ASSEMBLER_HPP
#define INC_4C_DEAL_II_FSI_DEAL_II_ASSEMBLER_HPP



#include "4C_config.hpp"

#include "4C_adapter_str_factory.hpp"
#include "4C_adapter_str_structure.hpp"
#include "4C_adapter_str_structure_new.hpp"
#include "4C_fem_condition_periodic.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_global_data.hpp"
#include "4C_io.hpp"
#include <4C_deal_ii_context.hpp>
#include <4C_deal_ii_triangulation.hpp>
#include <4C_linalg_sparsematrix.hpp>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_boundary.h>

FOUR_C_NAMESPACE_OPEN

namespace DealiiFSI
{
  namespace FluidProblem
  {
    template <int dim>
    class StokesBoundaryValues : public dealii::Function<dim>
    {
     public:
      StokesBoundaryValues() : dealii::Function<dim>(dim + 1) {}

      virtual double value(
          const dealii::Point<dim>& p, const unsigned int component = 0) const override;

      virtual void vector_value(
          const dealii::Point<dim>& p, dealii::Vector<double>& value) const override;
    };


    template <int dim>
    double StokesBoundaryValues<dim>::value(
        const dealii::Point<dim>& p, const unsigned int component) const
    {
      Assert(
          component < this->n_components, dealii::ExcIndexRange(component, 0, this->n_components));

      if (component == dim - 1) switch (dim)
        {
          case 2:
            return std::sin(dealii::numbers::PI * p[0]);
          case 3:
            return std::sin(dealii::numbers::PI * p[0]) * std::sin(dealii::numbers::PI * p[1]);
          default:
            DEAL_II_NOT_IMPLEMENTED();
        }

      return 0;
    }


    template <int dim>
    void StokesBoundaryValues<dim>::vector_value(
        const dealii::Point<dim>& p, dealii::Vector<double>& values) const
    {
      for (unsigned int c = 0; c < this->n_components; ++c)
        values(c) = StokesBoundaryValues<dim>::value(p, c);
    }

    template <int dim>
    class StokesProblem
    {
     public:
      static constexpr unsigned int interface_boundary_id = 42;
      static constexpr unsigned int top_boundary_id = 1;
      static constexpr unsigned int sides_boundary_id = 2;

      const double viscosity = 2.0;                     // viscosity of the fluid
      static constexpr unsigned int stokes_degree = 1;  // degree of the stokes finite element


      StokesProblem() = default;

      void build_problem(dealii::Triangulation<dim>& triangulation,
          dealii::SparseMatrix<double>& system_matrix, dealii::Vector<double>& rhs,
          unsigned int refine = 0)
      {
        make_grid(triangulation, refine);
        setup_dofs();
        setup_sparsity();
        assemble_system(system_matrix, rhs);
      }

      const dealii::AffineConstraints<double>& get_constraints() const { return constraints; }
      const dealii::DoFHandler<dim>& get_dof_handler() const { return dof_handler; }
      const dealii::FiniteElement<dim>& get_stokes_fe() const { return stokes_fe; }
      double get_viscosity() const { return viscosity; }


      void output_results(dealii::Vector<double>& solution_vector) const
      {
        using namespace dealii;
        std::vector<std::string> solution_names(dim, "velocity");
        solution_names.emplace_back("pressure");

        std::vector data_component_interpretation(
            dim, DataComponentInterpretation::component_is_part_of_vector);
        data_component_interpretation.push_back(DataComponentInterpretation::component_is_scalar);

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);

        data_out.add_data_vector(solution_vector, solution_names, DataOut<dim>::type_dof_data,
            data_component_interpretation);
        data_out.build_patches();
        data_out.write_vtu_with_pvtu_record(
            "./", "fluid_solution", 0, dof_handler.get_communicator(), 0, 8);
      }


     private:
      dealii::FESystem<dim> stokes_fe{
          dealii::FE_Q<dim>(stokes_degree + 1), dim, dealii::FE_Q<dim>(stokes_degree), 1};

      dealii::DoFHandler<dim> dof_handler;
      dealii::AffineConstraints<double> constraints;
      dealii::SparsityPattern sparsity_pattern;

      void make_grid(dealii::Triangulation<dim>& triangulation_in, unsigned int refine)
      {
        dealii::Triangulation<dim> helper_triangulation;
        dealii::GridGenerator::hyper_cube(helper_triangulation, -1.0, 1.0);
        helper_triangulation.refine_global(3);

        std::set<typename dealii::Triangulation<dim>::active_cell_iterator> cells_to_remove;
        for (const auto& cell : helper_triangulation.active_cell_iterators())
        {
          if (((std::fabs(cell->center()[0]) < 0.25) && (cell->center()[dim - 1] > 0.5)) ||
              ((std::fabs(cell->center()[0]) >= 0.25) && (cell->center()[dim - 1] > -0.5)))
          {
            continue;
          }
          cells_to_remove.insert(cell);
        }
        dealii::GridGenerator::create_triangulation_with_removed_cells(
            helper_triangulation, cells_to_remove, triangulation_in);

        // refine the grid if requested (zero means no refinement)
        triangulation_in.refine_global(refine);

        // set the boundary ids

        for (const auto& cell : triangulation_in.active_cell_iterators())
          for (const auto& face : cell->face_iterators())
          {
            // top boundary
            if (face->at_boundary() && (face->center()[dim - 1] == 1))
            {
              face->set_all_boundary_ids(top_boundary_id);
            }
            else if (face->at_boundary() and
                     (face->center()[0] == -1.0 or face->center()[0] == 1.0))
            {
              face->set_all_boundary_ids(sides_boundary_id);
            }
            else if (face->at_boundary())
            {
              face->set_all_boundary_ids(interface_boundary_id);
            }
          }
        dof_handler.reinit(triangulation_in);
      }

      void setup_dofs()
      {
        dof_handler.distribute_dofs(stokes_fe);

        // setup constraints
        // Dirichlet at the top boundary
        // Zero Dirichlet on the interface
        // Nothing on the sides
        {
          constraints.clear();
          dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

          const dealii::FEValuesExtractors::Vector velocities(0);
          dealii::VectorTools::interpolate_boundary_values(dof_handler, top_boundary_id,
              StokesBoundaryValues<dim>(), constraints, stokes_fe.component_mask(velocities));

          // Zero Dirichlet on the interface
          dealii::VectorTools::interpolate_boundary_values(dof_handler, interface_boundary_id,
              dealii::Functions::ZeroFunction<dim>(dim + 1), constraints,
              stokes_fe.component_mask(velocities));
          constraints.close();
        }
      }

      void setup_sparsity()
      {
        dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);
        constraints.condense(dsp);
        sparsity_pattern.copy_from(dsp);
      }

      void assemble_system(
          dealii::SparseMatrix<double>& system_matrix, dealii::Vector<double>& system_rhs)
      {
        system_matrix.reinit(sparsity_pattern);
        system_rhs.reinit(dof_handler.n_dofs());

        using namespace dealii;
        const QGauss<dim> stokes_quadrature(stokes_degree + 2);

        FEValues<dim> stokes_fe_values(
            stokes_fe, stokes_quadrature, update_values | update_gradients | update_JxW_values);

        // ...to objects that are needed to describe the local contributions to
        // the global linear system...
        const unsigned int stokes_dofs_per_cell = stokes_fe.n_dofs_per_cell();

        FullMatrix<double> local_matrix;
        Vector<double> local_rhs;

        std::vector<types::global_dof_index> local_dof_indices;
        const Functions::ZeroFunction<dim> right_hand_side(dim + 1);

        // ...to variables that allow us to extract certain components of the
        // shape functions and cache their values rather than having to recompute
        // them at every quadrature point:
        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        std::vector<SymmetricTensor<2, dim>> stokes_symgrad_phi_u(stokes_dofs_per_cell);
        std::vector<double> stokes_div_phi_u(stokes_dofs_per_cell);
        std::vector<double> stokes_phi_p(stokes_dofs_per_cell);

        // Then comes the main loop over all cells and, as in step-27, the
        // initialization of the hp::FEValues object for the current cell and the
        // extraction of a FEValues object that is appropriate for the current
        // cell:
        for (const auto& cell : dof_handler.active_cell_iterators())
        {
          stokes_fe_values.reinit(cell);
          local_matrix.reinit(cell->get_fe().n_dofs_per_cell(), cell->get_fe().n_dofs_per_cell());
          local_rhs.reinit(cell->get_fe().n_dofs_per_cell());


          const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
          Assert(dofs_per_cell == stokes_dofs_per_cell, ExcInternalError());

          for (unsigned int q = 0; q < stokes_fe_values.n_quadrature_points; ++q)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              stokes_symgrad_phi_u[k] = stokes_fe_values[velocities].symmetric_gradient(k, q);
              stokes_div_phi_u[k] = stokes_fe_values[velocities].divergence(k, q);
              stokes_phi_p[k] = stokes_fe_values[pressure].value(k, q);
            }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                local_matrix(i, j) +=
                    (2 * viscosity * stokes_symgrad_phi_u[i] * stokes_symgrad_phi_u[j] -
                        stokes_div_phi_u[i] * stokes_phi_p[j] -
                        stokes_phi_p[i] * stokes_div_phi_u[j]) *
                    stokes_fe_values.JxW(q);
          }

          local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(
              local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
        }
      }
    };  // StokesProblem



  }  // namespace FluidProblem

}  // namespace DealiiFSI
FOUR_C_NAMESPACE_CLOSE



#endif  // INC_4C_DEAL_II_FSI_DEAL_II_ASSEMBLER_HPP
