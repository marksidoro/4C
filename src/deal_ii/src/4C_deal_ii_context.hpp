// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_CONTEXT_HPP
#define FOUR_C_DEAL_II_CONTEXT_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_element_conversion.hpp"
#include <4C_deal_ii_context_implementation.hpp>
#include <4C_fem_discretization.hpp>
#include <4C_fem_general_element.hpp>

#include <deal.II/base/std_cxx20/iota_view.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/lac/la_parallel_vector.h>
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



  template <int dim, int spacedim>
  dealii::MappingFEField<dim, spacedim> create_isoparametric_mapping(
      DealiiWrappers::Context<dim, spacedim>& context,
      const Core::FE::Discretization& discretization)
  {
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

    FOUR_C_ASSERT(context.pimpl_->finite_elements.size() == 1,
        "Currently only supported for the case that there is only one finite element in the "
        "context, since the underlying dealii::MappingFEField does not support multiple finite "
        "elements.");

    // create an internal dofhandler using the finite element that is provided
    const auto& fe = context.pimpl_->finite_elements[0];
    FOUR_C_ASSERT(fe.n_components() == 1, "TODO : support multiple components in the FE.");

    // create an FE System object that has the right dimension
    dealii::FESystem<dim, spacedim> isoparametric_fe(fe, spacedim);

    // create a DofHandler for the isoparametric mapping
    dealii::DoFHandler<dim, spacedim> dof_handler(*context.pimpl_->triangulation);
    dof_handler.distribute_dofs(isoparametric_fe);

    // create ghosted vector for the postions of the nodes
    auto locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(dof_handler);
    VectorType position_vector;
    position_vector.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs,
        context.pimpl_->triangulation->get_mpi_communicator());

    // Now fill the position vector with the positions of the nodes
    for (const auto& cell : dof_handler.active_cell_iterators())
    {
      // skip ghost cells
      if (!cell->is_locally_owned()) continue;

      // get the equivalent element in four_c
      const auto* element = Internal::to_element(context, discretization, cell);
      const unsigned int n_nodes = element->num_node();
      const auto* nodes = element->nodes();


      // get the dof indices for the cell
      const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
      cell->get_dof_indices(dof_indices);

      FOUR_C_ASSERT(n_nodes * spacedim == dofs_per_cell,
          "Since this is an isoparametric mapping, the number of nodes x {} must be equal to the "
          "number of dofs per cell.",
          spacedim);

      // we now have to assign the postion of the nodes to the dof indices
      dealii::Vector<double> local_position_vector(dofs_per_cell);
      auto reordering = ElementConversion::reindex_four_c_to_dealii(element->shape());
      for (unsigned int n = 0; n < n_nodes; ++n)
      {
        const auto local_dealii_index = reordering[n];
        for (unsigned int d = 0; d < spacedim; ++d)
        {
          const auto local_vector_index =
              isoparametric_fe.component_to_system_index(d, local_dealii_index);
          local_position_vector[local_vector_index] = nodes[n].x()[d];
        }
      }
      // now we can add the local position vector to the global position vector
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        // only add local entries
        if (dof_handler.locally_owned_dofs().is_element(dof_indices[i]))
        {
          position_vector[dof_indices[i]] = local_position_vector[i];
        }
      }
    }
    return dealii::MappingFEField<dim, spacedim>(isoparametric_fe, position_vector);
  }

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

      for (auto dof : dofs_range)
      {
        // Add the coupling from the range dof to the domain dofs
        sparsity_pattern.add_row_entries(dof, dofs_domain, true);
      }
    }
  }

}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif
