// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_CONTEXT_IMPLEMENTATION_HPP
#define FOUR_C_DEAL_II_CONTEXT_IMPLEMENTATION_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"

#include <deal.II/hp/mapping_collection.h>



FOUR_C_NAMESPACE_OPEN



namespace DealiiWrappers
{
  template <int dim, int spacedim>
  class Context;

  namespace Internal
  {
    template <int dim, int spacedim>
    const Core::Elements::Element* to_element(const DealiiWrappers::Context<dim, spacedim>& context,
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell)
    {
      return context.get_discretization().l_row_element(
          context.pimpl_->cell_index_to_element_lid[cell->index()]);
    }


    /**
     * Generate a Gauss quadrature collection that is sufficient to integrate polynomials of degree
     * 2 * (deg_shape + deg_mapping) + 1 on all elements.
     * the degree is inferred from the finite element and the mapping.
     * @req The mapping must be of type MappingQ or MappingQEulerian.
     * @tparam dim
     * @tparam spacedim
     * @param context
     * @return
     */
    template <int dim, int spacedim>
    dealii::hp::QCollection<dim> fill_required_quadrature_gauss(Context<dim, spacedim>& context)
    {
      dealii::hp::QCollection<dim> quadrature_collection;
      const auto& fe_collection = context.pimpl_->finite_elements;
      const auto& mapping_collection = context.pimpl_->mapping_collection;

      FOUR_C_ASSERT(fe_collection.size() == mapping_collection.size(),
          "The number of finite elements and the number of mappings do not match.");

      for (unsigned int i = 0; i < fe_collection.size(); ++i)
      {
        // We restrict to the case where the mapping is either MappingQ or MappingQEulerian
        dealii::MappingQ<dim>* mapping =
            dynamic_cast<dealii::MappingQ<dim>>(&mapping_collection[i]);
        FOUR_C_ASSERT(
            mapping != nullptr, "The mapping is not of type MappingQ or MappingQEulerian.");
        // Exact for degree (2 * (deg_shape + deg_mapping) + 1)
        quadrature_collection.push_back(
            dealii::QGauss<dim>(fe_collection[i].degree + mapping->get_degree() + 1));
      }
      return quadrature_collection;
    }
  }  // namespace Internal
}  // namespace DealiiWrappers

FOUR_C_NAMESPACE_CLOSE

#endif
