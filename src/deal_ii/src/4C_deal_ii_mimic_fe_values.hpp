#ifndef FOUR_C_DEAL_II_MIMIC_FE_VALUES_HPP
#define FOUR_C_DEAL_II_MIMIC_FE_VALUES_HPP



#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_triangulation.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_shape_function_type.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include <4C_deal_ii_context_implementation.hpp>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/hp/fe_collection.h>



FOUR_C_NAMESPACE_OPEN


namespace DealiiWrappers
{


  namespace TOOLS
  {
    /**
     * Function to transform a deal.II point to a 4C unit cell point.
     * In 4C the unit cell is defined as [-1, 1]^dim
     * while in deal.II the unit cell is defined as [0, 1]^dim
     *
     * @tparam dim
     * @param deal_ii_unit_cell_point
     * @return
     */
    template <int dim>
    dealii::Point<dim> deal_to_four_c_unit_cell_transform(
        const dealii::Point<dim>& deal_ii_unit_cell_point)
    {
      return deal_ii_unit_cell_point * 2 - 1;
    }

    template <int dim>
    constexpr double deal_to_four_c_unit_cell_jacobian_scaling()
    {
      return static_cast<double>(std::pow(2, dim));
    }


    template <int dim>
    bool is_in_four_c_unit_cell(const dealii::Point<dim>& point)
    {
      for (int d = 0; d < dim; ++d)
      {
        if (point[d] < -1.0 || point[d] > 1.0) return false;
      }
      return true;
    }

    template <int dim>
    bool is_in_deal_ii_unit_cell(const dealii::Point<dim>& point)
    {
      for (int d = 0; d < dim; ++d)
      {
        if (point[d] < 0.0 || point[d] > 1.0) return false;
      }
      return true;
    }

    template <int dim>
    void evaluate(const Core::Elements::Element* element, const dealii::Point<dim>& point,
        std::vector<double>& shape_function_values)
    {
      FOUR_C_ASSERT(
          is_in_four_c_unit_cell(point), "The point {} is not in the 4C unit cell [-1,1].");
      // get size of the required shape function
      shape_function_values.resize(Core::FE::Internal::num_nodes(element->shape()));
      // evaluate the shape functions at the required points
      Core::FE::shape_function_dim<dealii::Point<dim>, std::vector<double>, dim>(
          point, shape_function_values, element->shape());
    }


  }  // namespace TOOLS



  template <int dim, int spacedim = dim>
  class MimicFEValuesFunction
  {
    const Context<dim, spacedim>& context_;
    const Core::FE::Discretization& discretization_;

    std::vector<std::vector<double>> shape_function_values_;
    Core::Elements::Element* current_four_c_element_;

   public:
    MimicFEValuesFunction(
        const Context<dim, spacedim>& context, const Core::FE::Discretization& discretization)
        : context_(context), discretization_(discretization)
    {
    }

    void reinit(const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
        const dealii::Quadrature<dim>& quadrature_rule)
    {
      // transform the cell to the 4C element
      // (we for now assume that such a 1-to-1 mapping exists)
      current_four_c_element_ = Internal::to_element(context_, discretization_, cell);

      // evaluate the shape functions of the element at the quadrature points
      // and store the values in the values_ vector
      const auto& dealii_quadrature_points = quadrature_rule.get_points();
      std::vector<dealii::Point<dim>> transformed_quadrature_points(
          dealii_quadrature_points.size());

      // Transform the quadrature points to the 4C unit cell
      for (unsigned int i = 0; i < dealii_quadrature_points.size(); ++i)
      {
        transformed_quadrature_points[i] =
            TOOLS::deal_to_four_c_unit_cell_transform<dim>(dealii_quadrature_points[i]);
      }
      shape_function_values_.resize(transformed_quadrature_points.size());
      // Initialize the shape function values
      for (unsigned int i = 0; i < transformed_quadrature_points.size(); ++i)
      {
        TOOLS::evaluate(
            *current_four_c_element_, transformed_quadrature_points[i], shape_function_values_[i]);
      }
    }

    double get_jacobian_scaling() const
    {
      return TOOLS::deal_to_four_c_unit_cell_jacobian_scaling<dim>();
    }

    /*void reinit(const typename dealii::Triangulation<dim, spacedim>::cell_iterator &cell,
        const std::vector<dealii::Point<dim>> &evaluation_points)
    {
       // transform the cell to the 4C element
       // (we for now assume that such a 1-to-1 mapping exists)
       const auto element_lid = context_.pimpl_->cell_index_to_element_lid[cell->index()];


      // TODO make sure this is a local_id
       const auto* four_c_element = discretization_.l_row_element(element_lid);

      // evaluate the shape functions of the element at the quadrature points
        // and store the values in the values_ vector

       // create location array needed for the extraction from the global vector
       Core::Elements::LocationArray location_array(discretization_.num_dof_sets());
       four_c_element->location_vector(this->discretization_, location_array, false);

       // either do the element action here or do get the shape functions:

       // extract the local values here:
       std::vector <double> local_values = Core::FE::extract_values(*state_vector_,
    location_array[0].lm_);
    }*/

    double shape_value(unsigned int shape_index, unsigned int evaluation_index) const
    {
      // TODO check if the shape_index is valid
      // TODO check if the evaluation_index is valid
      return shape_function_values_[evaluation_index][shape_index];
    }



    void evaluate_from_dof_vector(std::shared_ptr<Core::LinAlg::Vector<double>> state_vector_,
        std::vector<double>& function_values_at_quad_points)
    {
      // create location array needed for the extraction from the global vector
      Core::Elements::LocationArray location_array(discretization_.num_dof_sets());
      current_four_c_element_->location_vector(this->discretization_, location_array, false);

      // extract the local values here:
      auto local_values = Core::FE::extract_values(*state_vector_, location_array[0].lm_);

      function_values_at_quad_points.resize(shape_function_values_.size());
      for (unsigned int q = 0; q < shape_function_values_.size(); ++q)
      {
        for (unsigned int i = 0; i < shape_function_values_[q].size(); ++i)
        {
          function_values_at_quad_points[q] += shape_function_values_[q][i] * local_values[i];
        }
      }
    }
  };


};  // namespace DealiiWrappers



FOUR_C_NAMESPACE_CLOSE



#endif  // INC_4C_DEAL_II_MIMIC_FE_VALUES_HPP
