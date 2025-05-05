#ifndef FOUR_C_DEAL_II_MIMIC_FE_VALUES_HPP
#define FOUR_C_DEAL_II_MIMIC_FE_VALUES_HPP



#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_mimic_mapping.hpp"
#include "4C_deal_ii_quadrature_transform.hpp"
#include "4C_deal_ii_triangulation.hpp"
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
  enum class UpdateFlags
  {
    update_values = 0x0001,
    update_gradients = 0x0002,
    update_JxW_values = 0x0004,
    update_jacobians = 0x0008,
    update_inverse_jacobians = 0x0010,
  };


  inline UpdateFlags operator|(const UpdateFlags f1, const UpdateFlags f2)
  {
    return static_cast<UpdateFlags>(static_cast<unsigned int>(f1) | static_cast<unsigned int>(f2));
  }
  inline UpdateFlags& operator|=(UpdateFlags& f1, const UpdateFlags f2)
  {
    f1 = f1 | f2;
    return f1;
  }
  inline UpdateFlags operator&(const UpdateFlags f1, const UpdateFlags f2)
  {
    return static_cast<UpdateFlags>(static_cast<unsigned int>(f1) & static_cast<unsigned int>(f2));
  }
  inline UpdateFlags& operator&=(UpdateFlags& f1, const UpdateFlags f2)
  {
    f1 = f1 & f2;
    return f1;
  }



  namespace Internal::Evaluation
  {
    template <int dim>
    void evaluate(const Core::Elements::Element* element, const dealii::Point<dim>& point,
        std::vector<double>& shape_function_values)
    {
      FOUR_C_ASSERT(
          is_in_four_c_unit_cell(point), "The point {} is not in the 4C unit cell [-1,1].");
      // get size of the required shape function
      shape_function_values.resize(Core::FE::num_nodes(element->shape()));
      // evaluate the shape functions at the required points
      Core::FE::shape_function_dim<dealii::Point<dim>, std::vector<double>, dim>(
          point, shape_function_values, element->shape());
    }

    template <int dim>
    void evaluate_gradient(const Core::Elements::Element* element, const dealii::Point<dim>& point,
        Core::LinAlg::Matrix<dim, dim>& inverse_jacobian,
        Core::LinAlg::SerialDenseMatrix& function_gradient)
    {
      FOUR_C_ASSERT(
          is_in_four_c_unit_cell(point), "The point {} is not in the 4C unit cell [-1,1].");


      // get size of the required shape function
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
    const MappingBase<dim>& mapping_;
    const Core::FE::Discretization& discretization_;

    unsigned int num_dofs_per_cell_;

    struct CellData
    {
      std::vector<std::vector<scalar_value>> shape_function_values_;
      std::vector<Core::LinAlg::SerialDenseMatrix> shape_function_gradient_;
      typename MappingBase<dim>::MappingCellData mapping_cell_data_;
    } cell_data_;


    Core::Elements::Element* current_four_c_element_;

   public:
    MimicFEValuesFunction(const Context<dim, spacedim>& context,
        const HexElementContext<dim>& quad_context, const MappingBase<dim>& mapping,
        const Core::FE::Discretization& discretization)
        : context_(context),
          quadrature_context_(quad_context),
          mapping_(mapping),
          discretization_(discretization)
    {
    }

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


      cell_data_.shape_function_values_.resize(n_quad_points);
      // Initialize the shape function values
      for (unsigned int i = 0; i < n_quad_points; ++i)
      {
        Internal::Evaluation::evaluate(*current_four_c_element_, four_c_quadrature.point(i),
            cell_data_.shape_function_values_[i]);

        // Scale the shape function values to the deal_ii reference cell
        if (init_for_deal_ii_ref_cell)
        {
          cell_data_.shape_function_values_[i] *= quadrature_context_.deal_ii_scaling();
        }
      }

      cell_data_.shape_function_gradient_.resize(n_quad_points);
      auto& mapping_data = cell_data_.mapping_cell_data_;

      mapping_.compute_mapping_data(mapping_data, cell, init_for_deal_ii_ref_cell);

      // Initialize the shape function gradient
      for (unsigned int i = 0; i < n_quad_points; ++i)
      {
        Internal::Evaluation::evaluate_gradient(*current_four_c_element_,
            four_c_quadrature.point(i), mapping_data.inverse_jacobian_[i],
            cell_data_.shape_function_gradient_[i]);

        // Scale the shape function values to the deal_ii reference cell
        if (init_for_deal_ii_ref_cell)
        {
          cell_data_.shape_function_gradient_[i] *= quadrature_context_.deal_ii_scaling();
        }
      }
    }

    [[nodiscard]] double shape_value(unsigned int shape_index, unsigned int evaluation_index) const
    {
      // TODO check if the shape_index is valid
      // TODO check if the evaluation_index is valid
      return cell_data_.shape_function_values_[evaluation_index][shape_index];
    }

    [[nodiscard]] std::ranges::iota_view<unsigned int, unsigned int> local_shape_indices() const
    {
      return std::ranges::iota_view<unsigned int, unsigned int>(
          0U, cell_data_.shape_function_values_[0].size());
    }
    [[nodiscard]] unsigned int n_local_shape_functions() const
    {
      return cell_data_.shape_function_values_[0].size();
    }

    void evaluate_from_dof_vector(std::shared_ptr<Core::LinAlg::Vector<double>> state_vector_,
        std::vector<double>& function_values_at_quad_points)
    {
      // create location array needed for the extraction from the global vector
      Core::Elements::LocationArray location_array(discretization_.num_dof_sets());
      current_four_c_element_->location_vector(this->discretization_, location_array, false);

      // extract the local values here:
      auto local_values = Core::FE::extract_values(*state_vector_, location_array[0].lm_);

      function_values_at_quad_points.resize(cell_data_.shape_function_values_.size());
      for (unsigned int q = 0; q < cell_data_.shape_function_values_.size(); ++q)
      {
        for (unsigned int i = 0; i < cell_data_.shape_function_values_[q].size(); ++i)
        {
          function_values_at_quad_points[q] +=
              cell_data_.shape_function_values_[q][i] * local_values[i];
        }
      }
    }
  };
};  // namespace DealiiWrappers



FOUR_C_NAMESPACE_CLOSE



#endif  // INC_4C_DEAL_II_MIMIC_FE_VALUES_HPP
