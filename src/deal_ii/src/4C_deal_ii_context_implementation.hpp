// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_CONTEXT_IMPLEMENTATION_HPP
#define FOUR_C_DEAL_II_CONTEXT_IMPLEMENTATION_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_fem_discretization.hpp"

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>

#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers::Internal
{
  template <int dim, int spacedim = dim>
  struct ContextImplementation
  {
    //! Store the local mapping between deal.II cells and 4C elements
    std::unordered_map<int, int> cell_index_to_element_lid;

    //! All dealii::FiniteElement objects that are required for the original 4C discretization.
    dealii::hp::FECollection<dim, spacedim> finite_elements;

    //! The names of the FiniteElements in #finite_elements.
    // These are in the same order as the finite elements within the finite_elements collection.
    std::vector<std::string> finite_element_names;

    //! The mapping collection that is used to map the quadrature points from the reference cell to
    //! the real cell.
    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;
  };

  template <int dim, int spacedim>
  const Core::Elements::Element* to_element(const Context<dim, spacedim>& context,
      const Core::FE::Discretization& discretization,
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell)
  {
    return discretization.l_row_element(context.pimpl_->cell_index_to_element_lid[cell->index()]);
  }


  /**
   * Fill mapping data in the context.
   *
   * @pre FiniteElements have already been set in context.
   */
  template <int dim, int spacedim>
  void fill_mapping(Context<dim, spacedim>& context, const Core::FE::Discretization& discretization)
  {
    FOUR_C_ASSERT(context.pimpl_->finite_elements.size() == 1, "Internal error.");

    const auto& fe = context.pimpl_->finite_elements[0];

    if (fe.degree == 1)
    {
      // simple linear mapping
      context.pimpl_->mapping.push_back(dealii::MappingQ<dim, spacedim>(1));
    }
    else if (fe.degree == 2)
    {
      // create a MappingQEulerian that takes the shift into account
    }
    else
      FOUR_C_THROW("Only finite elements up to degree 2 are supported.");
  }


  /**
   * Generate a Gauss quadrature collection that is sufficient to integrate polynomials of degree
   * 2 * (deg_shape + deg_mapping) + 1 on all elements.
   * the degree is infered from the finite element and the mapping.
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
      dealii::MappingQ<dim>* mapping = dynamic_cast<dealii::MappingQ<dim>>(&mapping_collection[i]);
      FOUR_C_ASSERT(mapping != nullptr, "The mapping is not of type MappingQ or MappingQEulerian.");
      // Exact for degree (2 * (deg_shape + deg_mapping) + 1)
      quadrature_collection.push_back(
          dealii::QGauss<dim>(fe_collection[i].degree + mapping->get_degree() + 1));
    }
    return quadrature_collection;
  }

}  // namespace DealiiWrappers::Internal

FOUR_C_NAMESPACE_CLOSE

#endif
