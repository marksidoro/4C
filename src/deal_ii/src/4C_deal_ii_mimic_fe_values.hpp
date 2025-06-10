#ifndef INC_4C_DEAL_II_MIMIC_FE_VALUES_HPP
#define INC_4C_DEAL_II_MIMIC_FE_VALUES_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_mimic_mapping.hpp"
#include "4C_deal_ii_quadrature_transform.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_shape_function_type.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include <4C_deal_ii_context_implementation.hpp>
#include <4C_fem_geometry_position_array.hpp>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/hp/fe_collection.h>



FOUR_C_NAMESPACE_OPEN


namespace DealiiWrappers
{
  namespace Internal::Evaluation
  {
    template <int dim>
    void evaluate(const Core::Elements::Element* element, const dealii::Point<dim>& point,
        Core::LinAlg::SerialDenseVector& shape_function_values)
    {
      // resize if necessary
      if (shape_function_values.num_rows() != Core::FE::num_nodes(element->shape()))
        shape_function_values.resize(Core::FE::num_nodes(element->shape()));

      // evaluate the shape functions at the required points
      Core::FE::shape_function_dim<dealii::Point<dim>, Core::LinAlg::SerialDenseVector, dim>(
          point, shape_function_values, element->shape());
    }

    template <int dim>
    void evaluate_gradient(const Core::Elements::Element* element, const dealii::Point<dim>& point,
        const Core::LinAlg::SerialDenseMatrix& inverse_jacobian,
        Core::LinAlg::SerialDenseMatrix& function_gradient)
    {
      // FOUR_C_ASSERT(is_in_four_c_unit_cell(point), "The point is not in the 4C unit cell
      // [-1,1]."); get size of the required shape function
      Core::LinAlg::SerialDenseMatrix shape_function_gradient(
          dim, Core::FE::num_nodes(element->shape()));

      // evaluate the shape functions at the required points
      Core::FE::shape_function_deriv1_dim<dealii::Point<dim>, Core::LinAlg::SerialDenseMatrix, dim>(
          point, shape_function_gradient, element->shape());

      function_gradient.shape(dim, Core::FE::num_nodes(element->shape()));
      function_gradient.multiply(
          Teuchos::TRANS, Teuchos::NO_TRANS, 1.0, inverse_jacobian, shape_function_gradient, 1.0);
    }



    template <int dim, int spacedim = dim>
    void get_dof_indices(const Context<dim, spacedim>& context,
        const Core::FE::Discretization& discretization,
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
        std::vector<unsigned int>& dof_indices)
    {
      const auto* four_c_element =
          DealiiWrappers::Internal::to_element(context, discretization, cell);
      Core::Elements::LocationArray location_array(discretization.num_dof_sets());
      four_c_element->location_vector(discretization, location_array, false);
      const auto& global_dofs_on_cell_four_c = location_array[0].lm_;
      dof_indices.resize(global_dofs_on_cell_four_c.size());
      for (unsigned int i = 0; i < global_dofs_on_cell_four_c.size(); ++i)
      {
        dof_indices[i] = global_dofs_on_cell_four_c[i];
      }
    }
  }  // namespace Internal::Evaluation



  template <int dim, int spacedim = dim>
  class MimicFEValuesFunction
  {
    using scalar_value = double;
    using vector_value = dealii::Tensor<1, dim, scalar_value>;

    const Context<dim, spacedim>& context_;
    const HexElementContext<dim>& quadrature_context_;
    const MappingBase<dim, spacedim>& mapping_;
    const Core::FE::Discretization& discretization_;

    unsigned int num_dofs_per_cell_;

    struct CellData
    {
      /**
       * For each quadrature point, save a vector of shape function values evaluated at the point.
       * I.e. for the q-th quadrature point shape_function_values_[q][j] = N_j(x_q)
       * Where N_j is the j-th shape function and x_q is the q-th quadrature point.
       */
      std::vector<Core::LinAlg::SerialDenseVector> shape_function_values_;

      /**
       * For each quadrature point, save the shape function gradients evaluated at the point.
       * I.e. for the q-th quadrature point shape_function_gradient_[q][j] = dN_j(x_q)/dx
       * Where N_j is the j-th shape function and x_q is the q-th quadrature point.
       * The matrix is thus of the size num_dofs_per_cell_ x dim
       */
      std::vector<Core::LinAlg::SerialDenseMatrix> shape_function_gradient_;

      typename MappingBase<dim>::MappingCellData mapping_cell_data_;
    } cell_data_;


    const Core::Elements::Element* current_four_c_element_;

   public:
    MimicFEValuesFunction(const Context<dim, spacedim>& context,
        const HexElementContext<dim>& quad_context, const MappingBase<dim, spacedim>& mapping,
        const Core::FE::Discretization& discretization)
        : context_(context),
          quadrature_context_(quad_context),
          mapping_(mapping),
          discretization_(discretization)
    {
    }

    /**
     * Reinit the object on a given new cell, this means that we compute all the values
     * that can later be requested and accessed via the respective functions.
     * All the relevant data is stored internally.
     * The bool init_for_deal_ii_ref_cell is used for which context this stored data should be
     * initialized. Due to the fact that 4C and deal_ii might use different reference cells for the
     * quadrature rules functions might need to be scaled
     * @param cell
     * @param init_for_deal_ii_ref_cell
     */
    void reinit(const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
        bool init_for_deal_ii_ref_cell = true)
    {
      // transform the cell to the 4C element
      // (we for now assume that such a 1-to-1 mapping exists)
      current_four_c_element_ = Internal::to_element(context_, discretization_, cell);

      // get local dof number
      num_dofs_per_cell_ = Core::FE::num_nodes(current_four_c_element_->shape());
      // evaluate the shape functions of the element at the quadrature points
      // and store the values in the values_ vector
      const unsigned int n_quad_points = quadrature_context_.n_quad_points();
      const auto& four_c_quadrature = quadrature_context_.get_four_c_quadrature();


      std::fill_n(cell_data_.shape_function_values_.begin(), n_quad_points,
          Core::LinAlg::SerialDenseVector(num_dofs_per_cell_));


      // Initialize the shape function values
      for (unsigned int i = 0; i < n_quad_points; ++i)
      {
        Internal::Evaluation::evaluate(current_four_c_element_, four_c_quadrature.point(i),
            cell_data_.shape_function_values_[i]);

        // Scale the shape function values to the deal_ii reference cell
        if (init_for_deal_ii_ref_cell)
        {
          // multiply the shape function values with the scaling factor
          cell_data_.shape_function_values_[i].scale(quadrature_context_.deal_ii_scaling());
        }
      }

      cell_data_.shape_function_gradient_.resize(n_quad_points);
      auto& mapping_data = cell_data_.mapping_cell_data_;

      mapping_.compute_mapping_data(
          mapping_data, current_four_c_element_, init_for_deal_ii_ref_cell);

      // Initialize the shape function gradient
      for (unsigned int i = 0; i < n_quad_points; ++i)
      {
        Internal::Evaluation::evaluate_gradient<dim>(current_four_c_element_,
            four_c_quadrature.point(i), mapping_data.inverse_jacobians[i],
            cell_data_.shape_function_gradient_[i]);

        // Scale the shape function values to the deal_ii reference cell
        if (init_for_deal_ii_ref_cell)
        {
          cell_data_.shape_function_gradient_[i].scale(quadrature_context_.deal_ii_scaling());
        }
      }
    }

    /**
     * Compute the value of the requested shape function at the requested quadrature point
     * @param shape_index index of the shape function to be evaluated
     * @param evaluation_index index of the quadrature point at which the shape function is
     * @return
     */
    [[nodiscard]] double shape_value(unsigned int shape_index, unsigned int evaluation_index) const
    {
      // TODO check if the shape_index is valid
      // TODO check if the evaluation_index is valid
      return cell_data_.shape_function_values_[evaluation_index][shape_index];
    }

    /**
     * Compute the gradient of the requested shape function at the requested quadrature point
     * with respect to real cell coordinates.
     * @param shape_index index of the shape function to be evaluated
     * @param evaluation_index index of the quadrature point at which the shape function is
     * evaluated
     * @return
     */
    dealii::Tensor<1, spacedim> shape_grad(
        const unsigned int shape_index, const unsigned int evaluation_index) const
    {
      return dealii::Tensor<1,
          spacedim>();  // dealii::ArrayView<double>(
                        //  cell_data_.shape_function_gradient_[evaluation_index][shape_index],
                        //  spacedim));
    }


    /**
     * Get the range of the local shape function indices
     */
    [[nodiscard]] std::ranges::iota_view<unsigned int, unsigned int> local_shape_indices() const
    {
      return std::ranges::iota_view<unsigned int, unsigned int>(
          0U, cell_data_.shape_function_values_[0].num_cols());
    }

    /**
     * Get the number of local shape functions
     * @return
     */
    [[nodiscard]] unsigned int n_local_shape_functions() const
    {
      return cell_data_.shape_function_values_[0].num_rows();
    }

    void get_function_values(std::shared_ptr<Core::LinAlg::Vector<double>> state_vector_,
        std::vector<double>& function_values_at_quad_points) const
    {
      // create location array needed for the extraction from the global vector
      Core::Elements::LocationArray location_array(discretization_.num_dof_sets());
      current_four_c_element_->location_vector(this->discretization_, location_array);

      // extract the local values here:
      auto local_values = Core::FE::extract_values(*state_vector_, location_array[0].lm_);

      function_values_at_quad_points.resize(cell_data_.shape_function_values_.size());
      for (unsigned int q = 0; q < cell_data_.shape_function_values_.size(); ++q)
      {
        for (int i = 0; i < cell_data_.shape_function_values_[q].num_rows(); ++i)
        {
          function_values_at_quad_points[q] +=
              cell_data_.shape_function_values_[q][i] * local_values[i];
        }
      }
    }
  };



}  // namespace DealiiWrappers


FOUR_C_NAMESPACE_CLOSE

#endif  // INC_4C_DEAL_II_MIMIC_FE_VALUES_HPP
