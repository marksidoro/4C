#ifndef FOUR_C_DEAL_II_FSI_ALE_HELPER_HPP
#define FOUR_C_DEAL_II_FSI_ALE_HELPER_HPP


#include "4C_config.hpp"

#include "4C_deal_ii_fe_values_context.hpp"
#include "4C_deal_ii_fsi_tools.hpp"

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/numerics/data_out.h>

FOUR_C_NAMESPACE_OPEN

template <int dim>
void reset_affine_constraints_from_displacement(dealii::AffineConstraints<double>& constraints,
    const dealii::DoFHandler<dim>& constraints_dof_handler,
    const dealii::Vector<double>& displacement_vector,
    const DealiiFSI::InterfaceMatcher<dim>& interface)
{
  const unsigned int dealii_tria_index =
      &interface.get_triangulation(0) ==
              constraints_dof_handler.get_triangulation().get_triangulation_index()
          ? 0
          : 1;  // for clarity
  FOUR_C_ASSERT(&interface.get_triangulation(dealii_tria_index) ==
                    constraints_dof_handler.get_triangulation(),
      "The constraints dof handler must be on the same triangulation as the interface matcher");


  constraints.clear();
  // Set the constraints to zero for all dofs at the boundary faces
  for (const auto& boundary_face : constraints_dof_handler.active_cell_iterators())
  {
    for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if (boundary_face->face(f)->at_boundary())
      {
        const auto& face = boundary_face->face(f);
        std::vector<dealii::types::global_dof_index> dof_indices(face->get_fe().dofs_per_cell);
        face->get_dof_indices(dof_indices);
        for (auto dof : dof_indices) constraints.add_constraint(dof, {}, 0);
      }
    }
  }

  // Now loop over the interface and set the constraints for the dofs, from the
  // displacement vector
  for (const auto& interface_pair : interface.interface_range())
  {
    auto dealii_entity =
        dealii_tria_index == 0 ? (*interface_pair).first : (*interface_pair).second;
    auto four_c_entity =
        dealii_tria_index == 0 ? (*interface_pair).second : (*interface_pair).first;






  }
}



template <int dim, int fe_degree = -1, typename number = double>
class AleHelper
{
 public:
  AleHelper(const dealii::Triangulation<dim>& triangulation)
    requires(fe_degree != 1)
      : fe{fe_degree, dim}, quad(fe_degree + 1), dof_handler(triangulation)
  {
    dof_handler.distribute_dofs(fe);
  }

  AleHelper(unsigned int degree, const dealii::Triangulation<dim>& triangulation)
    requires fe_degree == -1
      : fe{degree, dim},
  quad{degree + 1}, dof_handler(triangulation)
  {
    dof_handler.distribute_dofs(fe);
  }

  void compute_mesh_displacement(const dealii::AffineConstraints<number>& movement_constraints)
  {
    setup_system(movement_constraints);
    assemble_rhs();
    solve();
  }

  dealii::LinearAlgebra::distributed::Vector<number>& get_solution() { return solution; }



 private:
  class LaplaceOperator : public dealii::MatrixFreeOperators::Base<dim,
                              dealii::LinearAlgebra::distributed::Vector<number>>
  {
   public:
    using Integrator = dealii::FEEvaluation<dim, fe_degree, fe_degree + 1, dim, number>;


    using value_type = number;
    LaplaceOperator();

    void clear() override
    {
      dealii::MatrixFreeOperators::Base<dim,
          dealii::LinearAlgebra::distributed::Vector<number>>::clear();
    }

    void compute_diagonal() override;

   private:
    void apply_add(dealii::LinearAlgebra::distributed::Vector<number>& dst,
        const dealii::LinearAlgebra::distributed::Vector<number>& src) const override
    {
      this->data->cell_loop(&LaplaceOperator::local_apply, this, dst, src);
    }

    void local_apply(const dealii::MatrixFree<dim, number>& data,
        dealii::LinearAlgebra::distributed::Vector<number>& dst,
        const dealii::LinearAlgebra::distributed::Vector<number>& src,
        const std::pair<unsigned int, unsigned int>& cell_range) const

    {
      Integrator phi(data);
      for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(dealii::EvaluationFlags::gradients);
        for (const unsigned int q : phi.quadrature_point_indices())
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(dealii::EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
    }

    void local_compute_diagonal(Integrator& integrator) const
    {
      this->inverse_diagonal_entries.reset(
          new dealii::DiagonalMatrix<dealii::LinearAlgebra::distributed::Vector<number>>());
      dealii::LinearAlgebra::distributed::Vector<number>& inverse_diagonal =
          this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal);

      dealii::MatrixFreeTools::compute_diagonal(
          *this->data, inverse_diagonal, &LaplaceOperator::local_compute_diagonal, this);

      this->set_constrained_entries_to_one(inverse_diagonal);

      for (unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
            dealii::ExcMessage("No diagonal entry in a positive definite operator "
                               "should be zero"));
        inverse_diagonal.local_element(i) = 1. / inverse_diagonal.local_element(i);
      }
    }
  };


  const dealii::FESystem<dim> fe;
  dealii::DoFHandler<dim> dof_handler;
  const dealii::MappingQ1<dim> mapping;
  dealii::QGauss<dim> quad;
  dealii::AffineConstraints<double> constraints;

  using SystemMatrixType = LaplaceOperator;
  SystemMatrixType system_matrix;
  typename dealii::MatrixFree<dim, double>::AdditionalData additional_data;

  using PreconditionerType = dealii::PreconditionChebyshev<SystemMatrixType,
      dealii::LinearAlgebra::distributed::Vector<double>>;



  dealii::LinearAlgebra::distributed::Vector<double> solution;
  dealii::LinearAlgebra::distributed::Vector<double> system_rhs;

  void setup_system(const dealii::AffineConstraints<double>& constraints)
  {
    system_matrix.clear();
    additional_data.tasks_parallel_scheme = dealii::MatrixFree<dim, double>::AdditionalData::none;
    additional_data.mapping_update_flags =
        (dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);
    std::shared_ptr<dealii::MatrixFree<dim, double>> system_mf_storage(
        new dealii::MatrixFree<dim, double>());
    system_mf_storage->reinit(
        mapping, dof_handler, constraints, dealii::QGauss<1>(fe.degree + 1), additional_data);
    system_matrix.initialize(system_mf_storage);

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(system_rhs);
    system_matrix.compute_diagonal();
  }


  void assemble_rhs()
  {
    system_rhs = 0;
    dealii::AffineConstraints<number> constraints_without_dbc(dof_handler.locally_owned_dofs(),
        dealii::DoFTools::extract_locally_relevant_dofs(dof_handler));
    constraints_without_dbc.close();

    dealii::LinearAlgebra::distributed::Vector<double> b, x;
    dealii::MatrixFree<dim, number> matrix_free;
    matrix_free.reinit(mapping, dof_handler, constraints_without_dbc, quad, additional_data);

    matrix_free.initialize_dof_vector(b);
    matrix_free.initialize_dof_vector(x);

    constraints.distribute(x);
    matrix_free.cell_loop(&LaplaceOperator::local_apply, system_matrix, b, x);
    constraints.set_zero(b);
    system_rhs -= b;
    system_rhs.compress(dealii::VectorOperation::add);
  }


  void solve()
  {
    typename PreconditionerType::AdditionalData preconditioner_data;
    preconditioner_data.smoothing_range = 15.;
    preconditioner_data.degree = 5;
    preconditioner_data.eig_cg_n_iterations = 10;
    preconditioner_data.preconditioner = system_matrix.get_matrix_diagonal_inverse();

    PreconditionerType preconditioner;
    preconditioner.initialize(system_matrix, preconditioner_data);

    dealii::SolverControl solver_control(100, 1e-12 * system_rhs.l2_norm());
    dealii::SolverCG<dealii::LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    constraints.set_zero(solution);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);
  }
};



FOUR_C_NAMESPACE_CLOSE


#endif  // INC_4C_DEAL_II_FSI_ALE_HELPER_HPP
