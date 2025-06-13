// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_CONTEXT_HPP
#define FOUR_C_DEAL_II_CONTEXT_HPP

#include "4C_config.hpp"

#include <4C_deal_ii_context_implementation.hpp>
#include <4C_fem_discretization.hpp>
#include <4C_fem_general_element.hpp>

#include <deal.II/base/std_cxx20/iota_view.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{

  // forward declaration
  namespace Internal
  {
    template <int dim, int spacedim>
    struct ContextImplementation;
  }

  /**
   * This class holds data which helps other classes and functions in this namespace to understand
   * the link between 4C and deal.II data structures. There is no documented and supported
   * functionality for this class. In fact, you should not try to do anything with the internals of
   * this class.
   */
  template <int dim, int spacedim = dim>
  struct Context
  {
    std::shared_ptr<Internal::ContextImplementation<dim, spacedim>> pimpl_;
  };


  /**
   * Make a sparsity pattern for the given context. It is a sparsity pattern for the coupling
   * from the domain discretization (4C) to the range discretization (deal.II).
   * It builds the sparsity pattern for coupling all dofs on the equivalent cells given in the
   * context object. This means that the underlying triangulation in the context must be the
   * same as in the provided dof_handler.
   * @tparam dim
   * @tparam spacedim
   * @param context
   * @param sparsity_pattern
   */
  template <int dim, int spacedim>
  void make_context_sparsity_pattern(const Context<dim, spacedim>& context,
      const dealii::DoFHandler<dim, spacedim>& range_dof_handler,
      const Core::FE::Discretization& domain_discretization,
      dealii::SparsityPatternBase& sparsity_pattern)
  {
    using namespace dealii;

    // Assert that the sparsity pattern is allready sized correctly
    FOUR_C_ASSERT(sparsity_pattern.n_rows() == range_dof_handler.n_dofs(),
        "The sparsity pattern must be sized to the number of dofs in the range discretization.");
    FOUR_C_ASSERT(sparsity_pattern.n_cols() ==
                      static_cast<unsigned int>(domain_discretization.num_global_nodes()),
        "The sparsity pattern must be sized to the number of dofs in the domain discretization.");

    // TODO check that we have the same triangulation in the context and the dof_handler
    // For this the context should hold a pointer to the triangulation

    std::vector<types::global_dof_index> dofs_range;
    std::vector<types::global_dof_index> dofs_domain;

    std::cout << domain_discretization.num_dof_sets() << " dof sets in domain discretization."
              << std::endl;
    Core::Elements::LocationArray location_array(domain_discretization.num_dof_sets());



    dofs_range.reserve(range_dof_handler.get_fe_collection().max_dofs_per_cell());
    dofs_domain.reserve(30);  // TODO: fetch an actual value from the discretization

    for (const auto& cell : range_dof_handler.active_cell_iterators())
    {
      // skip ghost cells
      if (!cell->is_locally_owned()) continue;

      const unsigned int dofs_on_cell_range = cell->get_fe().dofs_per_cell;
      dofs_range.resize(dofs_on_cell_range);
      cell->get_dof_indices(dofs_range);

      auto element = Internal::to_element<dim, spacedim>(context, domain_discretization, cell);
      element->location_vector(domain_discretization, location_array);

      dofs_domain.clear();
      dofs_domain.resize(location_array[0].lm_.size());
      std::copy(location_array[0].lm_.begin(), location_array[0].lm_.end(), dofs_domain.begin());



      // speed up insertions by sorting the domain dofs, otherwise this is done
      // for eact row entry
      std::sort(dofs_domain.begin(), dofs_domain.end());

      for (auto dof : dofs_domain)
      {
        std::cout << "Column: " << dof << std::endl;
      }


      for (auto dof : dofs_range)
      {
        std::cout << "Row: " << dof << std::endl;
        // Add the coupling from the range dof to the domain dofs
        sparsity_pattern.add_row_entries(dof, dofs_domain);
      }
    }
  }

}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif
