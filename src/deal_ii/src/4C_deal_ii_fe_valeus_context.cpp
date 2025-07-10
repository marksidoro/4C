#include "4C_deal_ii_fe_values_context.hpp"
FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{

  namespace Internal
  {
    void get_dof_indices_dealii_ordering(const Core::FE::Discretization& discretization,
        const Core::Elements::Element* element, const std::span<const int>& local_reorder,
        std::vector<dealii::types::global_dof_index>& dof_indices)
    {
      Core::Elements::LocationArray location_array(discretization.num_dof_sets());
      element->location_vector(discretization, location_array);
      dof_indices.resize(location_array[0].lm_.size());
      for (unsigned int i = 0; i < location_array[0].lm_.size(); ++i)
      {
        dof_indices[i] = location_array[0].lm_[local_reorder[i]];
      }
    }

    void get_dof_indices_four_c_ordering(const Core::FE::Discretization& discretization,
        const Core::Elements::Element* element,
        std::vector<dealii::types::global_dof_index>& dof_indices)
    {
      Core::Elements::LocationArray location_array(discretization.num_dof_sets());
      element->location_vector(discretization, location_array);
      dof_indices.resize(location_array[0].lm_.size());
      std::copy(location_array[0].lm_.begin(), location_array[0].lm_.end(), dof_indices.begin());
    }
  }  // namespace Internal
}  // namespace DealiiWrappers



FOUR_C_NAMESPACE_CLOSE