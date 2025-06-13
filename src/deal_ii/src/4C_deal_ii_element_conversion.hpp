// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_DEAL_II_ELEMENT_CONVERSION_HPP
#define FOUR_C_DEAL_II_ELEMENT_CONVERSION_HPP

#include "4C_config.hpp"

#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_node.hpp"

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/hp/fe_collection.h>

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers::ElementConversion
{
  /**
   * Helper namespace containing function to allow for compile time function definitions
   * of the deal.II fe-name to 4C cell type conversion.
   * This is necessary since the
   */
  namespace Internal
  {
    constexpr std::uint64_t hash(std::string_view s)
    {
      std::uint64_t h = 1469598103934665603ULL;
      for (char c : s)
      {
        h ^= static_cast<std::uint8_t>(c);
        h *= 1099511628211ULL;
      }
      return h;
    }

    template <int dim>
    constexpr std::array<char, 1> dim_to_str()
    {
      static_assert(dim == 1 || dim == 2 || dim == 3, "dim must be 1, 2 or 3");
      return {'0' + dim};
    }


    template <size_t N>
    struct fixed_string
    {
      char value[N];
      constexpr fixed_string(const char (&str)[N])
      {
        for (size_t i = 0; i < N; ++i) value[i] = str[i];
      }
      constexpr std::string_view view() const { return {value, N - 1}; }  // Exclude null terminator
    };
    template <size_t N>
    fixed_string(const char (&)[N]) -> fixed_string<N>;


    template <fixed_string Prefix, int dim, fixed_string Suffix>
    constexpr auto concatenate()
    {
      constexpr auto prefix = Prefix.value;
      constexpr auto suffix = Suffix.value;
      constexpr auto numarr = dim_to_str<dim>();
      constexpr size_t prefix_len = Prefix.view().size();
      constexpr size_t num_len = numarr.size();
      constexpr size_t suffix_len = Suffix.view().size();

      std::array<char, prefix_len + num_len + suffix_len> buf{};
      size_t i = 0;
      for (size_t j = 0; j < prefix_len; ++j) buf[i++] = prefix[j];
      for (size_t j = 0; j < num_len; ++j) buf[i++] = numarr[j];
      for (size_t j = 0; j < suffix_len; ++j) buf[i++] = suffix[j];
      return buf;
    }

    template <fixed_string Prefix, int Value, fixed_string Suffix>
    constexpr std::string_view concat_view()
    {
      constexpr auto arr = concatenate<Prefix, Value, Suffix>();
      return {arr.data(), arr.size()};
    }
  }  // namespace Internal



  /**
   * Returns the reindexing of deal.II vertices to 4C vertices for a given cell type. This means
   * that the i-th vertex of a deal.II cell corresponds to the reindex[i]-th vertex of the
   * corresponding 4C cell.
   */
  inline std::span<const int> reindex_dealii_to_four_c(Core::FE::CellType cell_type)
  {
    switch (cell_type)
    {
      case Core::FE::CellType::line2:
      {
        static constexpr std::array reindex{0, 1};
        return reindex;
      }
      case Core::FE::CellType::tet4:
      {
        static constexpr std::array reindex{0, 1, 2, 3};
        return reindex;
      }
      case Core::FE::CellType::hex8:
      {
        static constexpr std::array reindex{0, 1, 3, 2, 4, 5, 7, 6};
        return reindex;
      }
      case Core::FE::CellType::hex27:
      {
        static constexpr std::array reindex{// vertices
            0, 1, 3, 2, 4, 5, 7, 6,
            // lines
            11, 9, 8, 10, 19, 17, 16, 18, 12, 13, 15, 14,
            // faces
            24, 22, 21, 23, 20, 25,
            // center
            26};
        return reindex;
      }
      default:
      {
        FOUR_C_THROW(
            "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(cell_type).c_str());
      }
    }
  }

  /**
   * Same function as above but takes a deal.II finite element as input.
   * @tparam dim
   * @param fe
   * @return
   */
  template <int dim>
  inline std::span<const int> reindex_dealii_to_four_c(dealii::FiniteElement<dim> fe)
  {
    return reindex_dealii_to_four_c(four_c_cell_type<dim>(fe.get_name().c_str()));
  }

  /**
   * Returns the reindexing of 4C  vertices to deal.II vertices for a given cell type. This means
   * that the i-th vertex of a 4C cell corresponds to the reindex[i]-th vertex of the
   * corresponding deal.II cell.
   */
  inline std::span<const int> reindex_four_c_to_dealii(Core::FE::CellType cell_type)
  {
    switch (cell_type)
    {
      case Core::FE::CellType::line2:
      {
        static constexpr std::array reindex{0, 1};
        return reindex;
      }
      case Core::FE::CellType::tet4:
      {
        static constexpr std::array reindex{0, 1, 2, 3};
        return reindex;
      }
      case Core::FE::CellType::hex8:
      {
        static constexpr std::array reindex{0, 1, 3, 2, 4, 5, 7, 6};
        return reindex;
      }
      case Core::FE::CellType::hex27:
      {
        static constexpr std::array reindex{0, 1, 3, 2, 4, 5, 7, 6, 10, 9, 11, 8, 16, 17, 19, 18,
            14, 13, 15, 12, 24, 22, 21, 23, 20, 25, 26};
        return reindex;
      }
      default:
      {
        FOUR_C_THROW(
            "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(cell_type).c_str());
      }
    }
  }

  /**
   * Same function as above but takes a deal.II finite element as input.
   * @tparam dim
   * @param fe
   * @return
   */
  template <int dim>
  inline std::span<const int> reindex_four_c_to_dealii(dealii::FiniteElement<dim> fe)
  {
    return reindex_four_c_to_dealii(four_c_cell_type<dim>(fe.get_name().c_str()));
  }


  /**
   * Given a 4C element, extract the GIDs of its nodes and rearrange them to be compatible with
   * deal.II. Also, return the element center which we assume uniquely identifies the element.
   */
  template <int spacedim>
  dealii::Point<spacedim> vertices_to_dealii(
      const Core::Elements::Element* element, std::vector<unsigned>& vertex_gids)
  {
    auto reindexing = reindex_dealii_to_four_c(element->shape());

    switch (element->shape())
    {
      case Core::FE::CellType::line2:
      case Core::FE::CellType::tet4:
      case Core::FE::CellType::hex8:
      {
        dealii::Point<spacedim> element_center;
        vertex_gids.resize(element->num_node());

        for (int lid = 0; lid < element->num_node(); ++lid)
        {
          const auto& node = element->nodes()[reindexing[lid]];
          vertex_gids[lid] = node->id();
          for (unsigned d = 0; d < spacedim; ++d) element_center[d] += node->x()[d];
        }

        // Normalize the center
        element_center /= element->num_node();
        return element_center;
      }
      case Core::FE::CellType::hex27:
      {
        dealii::Point<spacedim> element_center;
        vertex_gids.resize(8);

        // Only require the first 8 nodes for deal.II
        for (int lid = 0; lid < 8; ++lid)
        {
          const auto& node = element->nodes()[reindexing[lid]];
          vertex_gids[lid] = node->id();
          for (unsigned d = 0; d < spacedim; ++d) element_center[d] += node->x()[d];
        }
        // Normalize the center
        element_center *= 0.125;
        return element_center;
      }
      default:
        FOUR_C_THROW(
            "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(element->shape()).c_str());
    }
  }


  /**
   * The name of the deal.II FiniteElement that corresponds to the given 4C cell type.
   */
  constexpr std::string dealii_fe_name(Core::FE::CellType cell_type)
  {
    switch (cell_type)
    {
      case Core::FE::CellType::line2:
        return "FE_Q<1>(1)";
      case Core::FE::CellType::tet4:
        return "FE_SimplexP(1)";
      case Core::FE::CellType::hex8:
        return "FE_Q<3>(1)";
      case Core::FE::CellType::hex27:
        return "FE_Q<3>(2)";
      default:
        FOUR_C_THROW(
            "Unsupported cell type '{}'.", Core::FE::cell_type_to_string(cell_type).c_str());
    }
  }


  /**
   * Helper to get the four_c_cell_type from a deal.II finite element name.
   * the const char* type is used so that this function can be used as constexpr
   * @tparam dim
   * @tparam spacedim
   * @param finite_element_name
   * @return
   */
  template <int dim, int spacedim = dim>
  constexpr Core::FE::CellType four_c_cell_type(const std::string_view finite_element_name)
  {
    (void)finite_element_name;  // suppress unused variable warning
    FOUR_C_THROW("Not implemented dimension {} and spacedim {}.", dim, spacedim);
    return Core::FE::CellType::dis_none;
  }

  /**
   * Not constexpr version of the above function. This is used to get the cell type from a
   * dealii::FiniteElement object
   * @tparam dim
   * @tparam spacedim
   * @param finite_element
   * @return
   */
  template <int dim, int spacedim = dim>
  Core::FE::CellType four_c_cell_type(const dealii::FiniteElement<dim, spacedim>& finite_element)
  {
    return four_c_cell_type<dim, spacedim>(finite_element.get_name().c_str());
  }


  /**
   * Specializations of the above functions for 1D, 2D and 3D elements.
   * @param finite_element_name
   * @return
   */
  template <>
  constexpr Core::FE::CellType four_c_cell_type<1, 1>(const std::string_view finite_element_name)
  {
    // grab function pointer to get rid of namespace
    using namespace Internal;

    switch (hash(finite_element_name))
    {
      case hash("FE_Q(1)"):
      case hash("FE_Q<1>(1)"):
        return Core::FE::CellType::line2;
      default:
        FOUR_C_THROW("Unsupported finite element type '{}' for dim = {} and spacedim = {}.",
            finite_element_name, 1, 1);
    }
  }

  template <>
  constexpr Core::FE::CellType four_c_cell_type<2, 2>(const std::string_view finite_element_name)
  {
    // grab function pointer to get rid of namespace
    using namespace Internal;

    switch (hash(finite_element_name))
    {
      case hash("FE_Q(1)"):
      case hash("FE_Q<2>(1)"):
        return Core::FE::CellType::quad4;
      default:
        FOUR_C_THROW("Unsupported finite element type '{}' for dim = {} and spacedim = {}.",
            finite_element_name, 2, 2);
    }
  }
  template <>
  constexpr Core::FE::CellType four_c_cell_type<3, 3>(const std::string_view finite_element_name)
  {
    // grab function pointer to get rid of namespace
    using namespace Internal;

    switch (hash(finite_element_name))
    {
      case hash("FE_Q(1)"):
      case hash("FE_Q<3>(1)"):
        return Core::FE::CellType::hex8;
      case hash("FE_Q(2)"):
      case hash("FE_Q<3>(2)"):
        return Core::FE::CellType::hex27;
      case hash("FE_SimplexP(1)"):
      case hash("FE_SimplexP<3>(1)"):
        return Core::FE::CellType::tet4;
      default:
        FOUR_C_THROW("Unsupported finite element type '{}' for dim = {} and spacedim = {}.",
            finite_element_name, 3, 3);
    }
  }


  /**
   * Create the dealii::hp::FECollection with all FE types that appear in the given @p
   * discretization. This also included FEs that appear only on other MPI ranks. In addition,
   * this function returns the names of these elements in the same order as in the FECollection.
   *
   * @note The ordering of the FEs in the collection is the same on all ranks.
   */
  template <int dim, int spacedim>
  std::pair<dealii::hp::FECollection<dim, spacedim>, std::vector<std::string>>
  create_required_finite_element_collection(const Core::FE::Discretization& discretization)
  {
    // First, determine all FEs we require locally
    int max_num_dof_per_node{};
    std::set<std::string> local_dealii_fes;

    const MPI_Comm comm = discretization.get_comm();

    for (int i = 0; i < discretization.num_my_row_elements(); ++i)
    {
      const auto* four_c_element = discretization.l_row_element(i);
      max_num_dof_per_node = std::max(
          max_num_dof_per_node, four_c_element->num_dof_per_node(*four_c_element->nodes()[0]));
      local_dealii_fes.emplace(dealii_fe_name(four_c_element->shape()));
    }

    max_num_dof_per_node = dealii::Utilities::MPI::max(max_num_dof_per_node, comm);

    // Communicate the required deal.II FEs
    const auto all_dealii_fe_names = std::invoke(
        [&]()
        {
          std::vector<std::string> local_dealii_fes_vector(
              local_dealii_fes.begin(), local_dealii_fes.end());
          std::vector<std::vector<std::string>> all_dealii_fes_vector =
              dealii::Utilities::MPI::all_gather(comm, local_dealii_fes_vector);

          std::set<std::string> all_dealii_fes;
          for (const auto& my : all_dealii_fes_vector)
          {
            for (const auto& fe : my)
            {
              all_dealii_fes.emplace(fe);
            }
          }
          return std::vector<std::string>(all_dealii_fes.begin(), all_dealii_fes.end());
        });

    // create the deal.II FiniteElement as a collection
    dealii::hp::FECollection<dim, spacedim> fe_collection;

    for (const auto& fe_string : all_dealii_fe_names)
    {
      const auto fe = std::invoke(
          [&]() -> std::unique_ptr<dealii::FiniteElement<dim, spacedim>>
          {
            // NOTE: work around a limitation in deal.II: the convenience getter is not
            // implemented for simplex
            if (fe_string == "FE_SimplexP(1)")
            {
              return std::make_unique<dealii::FE_SimplexP<dim, spacedim>>(1);
            }
            else
            {
              return dealii::FETools::get_fe_by_name<dim, spacedim>(fe_string);
            }
          });

      if (max_num_dof_per_node == 1)
        fe_collection.push_back(*fe);
      else
        fe_collection.push_back(dealii::FESystem<dim, spacedim>(*fe, max_num_dof_per_node));
    }

    return {fe_collection, all_dealii_fe_names};
  }

  /**
   * Function to create a mapping collection for every finite element in the given
   * fe_collection, The mapping is for all elements linear and uses the deal.II
   * MappingQ class. (currently this work only for linear and quadratic elements and for hex
   * meshes)
   * @tparam dim
   * @tparam spacedim
   * @param fe_collection
   * @return
   */
  template <int dim, int spacedim>
  dealii::hp::MappingCollection<dim, spacedim> create_linear_mapping_collection(
      const dealii::hp::FECollection<dim, spacedim>& fe_collection)
  {
    using namespace Internal;
    // grab function pointer to get rid of namespace

    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;
    for (unsigned int i = 0; i < fe_collection.size(); ++i)
    {
      switch (hash(fe_collection[i].get_name()))
      {
        case hash("FE_Q(1)"):
        case hash(concat_view<"FE_Q<", dim, ">(1)">()):
          mapping_collection.push_back(dealii::MappingQ<dim, spacedim>(1));
          break;
        default:
          FOUR_C_THROW("Unsupported finite element type '{}'.", fe_collection[i].get_name());
      }
    }
    return mapping_collection;
  }

}  // namespace DealiiWrappers::ElementConversion

FOUR_C_NAMESPACE_CLOSE

#endif
