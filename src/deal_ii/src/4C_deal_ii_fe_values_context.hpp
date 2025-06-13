#ifndef INC_4C_DEAL_II_FE_VALUES_CONTEXT_HPP
#define INC_4C_DEAL_II_FE_VALUES_CONTEXT_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_element_conversion.hpp"
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
#include <mat/4C_mat_inelastic_defgrad_factors_service.hpp>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>

FOUR_C_NAMESPACE_OPEN
namespace DealiiWrappers
{

  /**
   * Class to provide the relevante context for the FEValues class to transfer a 4C discretization
   * to a deal.II FEValues object. This class is used to store the data that is needed for the
   */
  template <int dim, int spacedim = dim>
  class FEValuesContext
  {
    const Context<dim, spacedim>& context_;
    const Core::FE::Discretization& discretization_;

    dealii::hp::FEValues<dim, spacedim> fe_values_;

    struct CellData
    {
      const Core::Elements::Element* element;
      std::span<const int> four_c_shape_indices;
      std::span<const int> dealii_shape_indices;
      unsigned int active_index;
    };

    CellData cell_data_;

   public:
    FEValuesContext(const Context<dim, spacedim>& context,
        const Core::FE::Discretization& discretization,
        const dealii::hp::QCollection<dim>& quadrature_collection, dealii::UpdateFlags update_flags)
        : context_(context),
          discretization_(discretization),
          fe_values_(context.pimpl_->mapping_collection, context.pimpl_->finite_elements,
              quadrature_collection, update_flags)
    {
    }

    void reinit(const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell)
    {
      cell_data_.element = Internal::to_element(context_, discretization_, cell);

      cell_data_.four_c_shape_indices =
          ElementConversion::reindex_dealii_to_four_c(cell_data_.element->shape());

      cell_data_.dealii_shape_indices =
          ElementConversion::reindex_four_c_to_dealii(cell_data_.element->shape());

      auto pos = std::find(context_.pimpl_->finite_element_names.begin(),
          context_.pimpl_->finite_element_names.end(),
          ElementConversion::dealii_fe_name(cell_data_.element->shape()));

      std::cout << "FE name size:" << context_.pimpl_->finite_element_names.size() << std::endl;
      std::cout << "FE size" << context_.pimpl_->finite_elements.size() << std::endl;
      for (auto name : context_.pimpl_->finite_element_names)
      {
        std::cout << "Finite element name: " << name << std::endl;
      }

      FOUR_C_ASSERT(pos != context_.pimpl_->finite_element_names.end(),
          "The finite element name '{}' is not in the finite element collection.",
          Core::FE::cell_type_to_string(cell_data_.element->shape()));

      cell_data_.active_index = std::distance(context_.pimpl_->finite_element_names.begin(), pos);

      fe_values_.reinit(
          cell, cell_data_.active_index, cell_data_.active_index, cell_data_.active_index);
    }


    /**
     * On the local cell/elelement on which this object was last reinitialized, return the
     * transformation from the deal.II shape index to the four c shape index.
     * I.e. for the j-th shape function in the deal.II ordering we can access its
     * four c index via four_c_shape_indices[j].
     * @return
     */
    const std::span<const int>& shape_indices_four_c() const
    {
      return cell_data_.four_c_shape_indices;
    }


    const std::span<const int>& shape_indices_dealii() const
    {
      return cell_data_.four_c_shape_indices;
    }


    const dealii::FEValues<dim>& get_present_fe_values() const
    {
      return fe_values_.get_present_fe_values();
    }


    /**
     * Fill the dof_indices vector with the global dof indices of dofs corresponding to the
     * local shape function indices on the cell/element on which this object was last
     * reinitialized. The dof_indices are ordered in the four_c ordering. I.e. in the ordering
     * in which the four c shapefunctions are defined. That means that for the j-th shape
     * function dof_indices[j] contains its corresponding global dof index.
     * @param dof_indices
     */
    void get_dof_indices_four_c_ordering(
        std::vector<dealii::types::global_dof_index>& dof_indices) const
    {
      Core::Elements::LocationArray location_array(discretization_.num_dof_sets());
      cell_data_.element->location_vector(this->discretization_, location_array);
      dof_indices.resize(location_array[0].lm_.size());
      std::copy(location_array[0].lm_.begin(), location_array[0].lm_.end(), dof_indices.begin());
    }



    /**
     * Fill the dof_indices vector with the global dof indices of dofs corresponding to the
     * local shape function indices on the cell/element on which this object was last
     * reinitialized. The dof_indices are ordered in the dealii ordering. I.e. in the ordering
     * in which the dealii shapefunctions are defined. That means that for the j-th shape
     * function dof_indices[j] contains its corresponding global dof index.
     * @param dof_indices
     */
    void get_dof_indices_dealii_ordering(
        std::vector<dealii::types::global_dof_index>& dof_indices) const
    {
      Core::Elements::LocationArray location_array(discretization_.num_dof_sets());
      cell_data_.element->location_vector(this->discretization_, location_array);
      dof_indices.resize(location_array[0].lm_.size());
      const auto& local_reorder = cell_data_.four_c_shape_indices;

      for (unsigned int i = 0; i < location_array[0].lm_.size(); ++i)
      {
        dof_indices[i] = location_array[0].lm_[local_reorder[i]];
      }
    }
  };

}  // namespace DealiiWrappers
FOUR_C_NAMESPACE_CLOSE



#endif  // INC_4C_DEAL_II_FE_VALUES_CONTEXT_HPP
