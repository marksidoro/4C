// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef INC_4C_DEAL_II_FSI_TOOLS_HPP
#define INC_4C_DEAL_II_FSI_TOOLS_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_fe_values_context.hpp"

#include <deal.II/grid/tria.h>

#include <set>

FOUR_C_NAMESPACE_OPEN

namespace DealiiFSI
{

  template <int dim>
  class InterfaceMatcher : public dealii::EnableObserverPointer
  {
   public:
    template <typename Iterator>
    using IteratorRange = dealii::IteratorRange<Iterator>;
    using CellIterator = typename dealii::Triangulation<dim>::active_cell_iterator;
    using FaceIterator = typename dealii::Triangulation<dim>::active_face_iterator;

    /**
     * InterfaceEntities describe a geometric entity of one triangulation on the interface.
     * These can either be
     *		- a cell (volume interface)
     *		- a cell and a face (surface interface)
     *		- a cell, a face and a subface (in case one side is refined)
     * Note that the cell is always filled with the iterator of the triangulation. The other entries
     * may not be needed. In this case they are set to dealii::numbers::invalid_unsigned_int.
     */
    struct InterfaceEntity
    {
      InterfaceEntity(const CellIterator& cell,
          unsigned int face = dealii::numbers::invalid_unsigned_int,
          unsigned int subface = dealii::numbers::invalid_unsigned_int);


      CellIterator cell;
      unsigned int face = dealii::numbers::invalid_unsigned_int;
      unsigned int subface = dealii::numbers::invalid_unsigned_int;

      bool operator<(const InterfaceEntity& other) const;
    };

    using InterfacePair = std::pair<InterfaceEntity, InterfaceEntity>;


    InterfaceMatcher(const dealii::Triangulation<dim>& tria1,
        const dealii::Triangulation<dim>& tria2, std::vector<InterfacePair>&& matches);


    unsigned int n_matches() const;


    const dealii::Triangulation<dim>& get_triangulation(unsigned int tria_index) const;


    const dealii::Triangulation<dim>& get_first_triangulation() const;
    const dealii::Triangulation<dim>& get_second_triangulation() const;

    unsigned int get_tria_index(const dealii::Triangulation<dim>& tria) const;


    /**
     * compute and return the reversed interface matcher, i.e. where the first triangulation
     * is now the second triangulation and vice versa (with upadted indexing also in the
     * InterfacePair).
     * @return
     */
    InterfaceMatcher compute_reversed() const;

    // ==========================================================================
    // ==========================================================================
    // ==========================================================================

    class interface_iterator
    {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = InterfacePair;
      using difference_type = std::ptrdiff_t;
      using pointer = const value_type*;
      using reference = const value_type&;

      interface_iterator(typename std::vector<InterfacePair>::const_iterator it);

      value_type operator*() const;
      interface_iterator& operator++();
      interface_iterator operator++(int);
      bool operator==(const interface_iterator& other) const;
      bool operator!=(const interface_iterator& other) const;

     private:
      typename std::vector<InterfacePair>::const_iterator it_;
    };
    interface_iterator interface_begin() const;
    interface_iterator interface_end() const;
    IteratorRange<interface_iterator> interface_range() const;

    class interface_cell_iterator
    {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = CellIterator;
      using difference_type = std::ptrdiff_t;
      using pointer = const value_type*;
      using reference = const value_type&;

      interface_cell_iterator(
          const std::vector<InterfacePair>* matches, int tria_index, size_t pos = 0);
      value_type operator*() const;
      interface_cell_iterator& operator++();
      interface_cell_iterator operator++(int);
      bool operator==(const interface_cell_iterator& other) const;
      bool operator!=(const interface_cell_iterator& other) const;

     private:
      void advance_to_next_unique();
      const std::vector<InterfacePair>* matches_;
      int tria_index_;
      size_t pos_;
      mutable std::set<CellIterator> seen_;
    };
    interface_cell_iterator interface_cell_begin(unsigned int tria_index) const;
    interface_cell_iterator interface_cell_end(unsigned int tria_index) const;
    IteratorRange<interface_cell_iterator> interface_cell_range(unsigned int tria_index) const;


    class interface_face_iterator
    {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = FaceIterator;
      using difference_type = std::ptrdiff_t;
      using pointer = const value_type*;
      using reference = const value_type&;

      interface_face_iterator(
          const std::vector<InterfacePair>* matches, int tria_index, size_t pos = 0);

      value_type operator*() const;

      interface_face_iterator& operator++();

      interface_face_iterator operator++(int);

      bool operator==(const interface_face_iterator& other) const;

      bool operator!=(const interface_face_iterator& other) const;

     private:
      void advance_to_next_unique();
      const std::vector<InterfacePair>* matches_;
      int tria_index_;
      size_t pos_;
      mutable std::set<value_type> seen_;
    };
    interface_face_iterator interface_face_begin(unsigned int tria_index) const;
    interface_face_iterator interface_face_end(unsigned int tria_index) const;
    IteratorRange<interface_face_iterator> interface_face_range(unsigned int tria_index) const;


    class interface_entity_iterator
    {
     public:
      using iterator_category = std::forward_iterator_tag;
      using value_type = InterfaceEntity;
      using difference_type = std::ptrdiff_t;
      using pointer = const value_type*;
      using reference = const value_type&;

      interface_entity_iterator(
          const std::vector<InterfacePair>* matches, int tria_index, size_t pos = 0);

      value_type operator*() const;

      interface_entity_iterator& operator++();


      interface_entity_iterator operator++(int);


      bool operator==(const interface_entity_iterator& other) const;

      bool operator!=(const interface_entity_iterator& other) const;

     private:
      void advance_to_next_unique();
      const std::vector<InterfacePair>* matches_;
      int tria_index_;
      size_t pos_;
      mutable std::set<value_type> seen_;
    };
    interface_entity_iterator interface_entity_begin(int tria_index) const;
    interface_entity_iterator interface_entity_end(int tria_index) const;
    IteratorRange<interface_entity_iterator> interface_entity_range(unsigned int tria_index) const;

   private:
    const dealii::Triangulation<dim>& tria_first;
    const dealii::Triangulation<dim>& tria_second;
    std::vector<InterfacePair> matches;
  };

  // Free function to build an InterfaceMatcher for matching faces (default: boundary faces)
  template <int dim>
  InterfaceMatcher<dim> make_interface_matcher_across_boundary(
      const dealii::Triangulation<dim>& tria_first, const dealii::Triangulation<dim>& tria_second,
      double tolerance = std::numeric_limits<double>::epsilon());

  template <int dim>
  void make_interface_sparsity_pattern(const InterfaceMatcher<dim>& matcher,
      const dealii::DoFHandler<dim>& deal_ii_discretization,
      const DealiiWrappers::Context<dim>& four_c_discretization,
      dealii::DynamicSparsityPattern& sparsity_pattern)
  {
    using namespace dealii;
    std::vector<types::global_dof_index> dof_indices_range(
        deal_ii_discretization.get_fe_collection().max_dofs_per_cell());

    std::vector<types::global_dof_index> dof_indices_domain(
        four_c_discretization.pimpl_->finite_elements.max_dofs_per_cell());


    for (const auto& interface_pair : matcher.interface_range())
    {
      const auto& cell_range =
          (*interface_pair).first.cell->as_dof_handler_iterator(deal_ii_discretization);
      const auto& cell_domain = (*interface_pair).second.cell;

      dof_indices_range.resize(cell_range->get_fe().dofs_per_cell);
      dof_indices_domain.resize(four_c_discretization.fe(cell_domain).dofs_per_cell);

      cell_range->get_dof_indices(dof_indices_range);
      four_c_discretization.get_dof_indices_four_c_ordering(cell_domain, dof_indices_domain);
      std::sort(dof_indices_domain.begin(), dof_indices_domain.end());

      for (const auto& dof_range : dof_indices_range)
      {
        // Add the coupling from the range dof to the domain dofs
        sparsity_pattern.add_row_entries(dof_range, dof_indices_domain, true);
      }
    }
  }

  template <int dim>
  void make_interface_sparsity_pattern(const InterfaceMatcher<dim>& matcher,
      const DealiiWrappers::Context<dim>& four_c_discretization,
      const dealii::DoFHandler<dim>& deal_ii_discretization,
      dealii::DynamicSparsityPattern& sparsity_pattern)
  {
    using namespace dealii;
    std::vector<types::global_dof_index> dof_indices_range(
        deal_ii_discretization.get_fe_collection().max_dofs_per_cell());

    std::vector<types::global_dof_index> dof_indices_domain(
        four_c_discretization.pimpl_->finite_elements.max_dofs_per_cell());


    for (const auto& interface_pair : matcher.interface_range())
    {
      const auto& cell_range = (*interface_pair).first.cell;
      const auto& cell_domain =
          (*interface_pair).second.cell->as_dof_handler_iterator(deal_ii_discretization);

      dof_indices_range.resize(four_c_discretization.fe(cell_range).dofs_per_cell);
      dof_indices_domain.resize(cell_domain->get_fe().dofs_per_cell);


      four_c_discretization.get_dof_indices_four_c_ordering(cell_range, dof_indices_range);
      cell_domain->get_dof_indices(dof_indices_domain);

      // sort for faster insertion
      std::sort(dof_indices_domain.begin(), dof_indices_domain.end());
      for (const auto& dof_range : dof_indices_range)
      {
        // Add the coupling from the range dof to the domain dofs
        sparsity_pattern.add_row_entries(dof_range, dof_indices_domain, true);
      }
    }
  }


}  // namespace DealiiFSI
FOUR_C_NAMESPACE_CLOSE


// ===========================================================================
// ===========================================================================
// Template implementations

FOUR_C_NAMESPACE_OPEN

namespace DealiiFSI
{

  template <int dim>
  InterfaceMatcher<dim>::InterfaceEntity::InterfaceEntity(
      const CellIterator& cell, unsigned int face, unsigned int subface)
      : cell(cell), face(face), subface(subface)
  {
  }
  template <int dim>
  bool InterfaceMatcher<dim>::InterfaceEntity::operator<(const InterfaceEntity& other) const
  {
    if (cell != other.cell) return cell < other.cell;
    if (face != other.face) return face < other.face;
    if (subface != other.subface) return subface < other.subface;
    return false;  // all fields are equal
  }
  template <int dim>
  InterfaceMatcher<dim>::InterfaceMatcher(const dealii::Triangulation<dim>& tria1,
      const dealii::Triangulation<dim>& tria2, std::vector<InterfacePair>&& matches)
      : tria_first(tria1), tria_second(tria2), matches(std::move(matches))
  {
  }
  template <int dim>
  const dealii::Triangulation<dim>& InterfaceMatcher<dim>::get_triangulation(
      unsigned int tria_index) const
  {
    if (tria_index == 0)
      return tria_first;
    else if (tria_index == 1)
      return tria_second;

    Assert(false, dealii::ExcMessage("Invalid triangulation index."));
  }
  template <int dim>
  const dealii::Triangulation<dim>& InterfaceMatcher<dim>::get_first_triangulation() const
  {
    return tria_first;
  }
  template <int dim>
  const dealii::Triangulation<dim>& InterfaceMatcher<dim>::get_second_triangulation() const
  {
    return tria_second;
  }
  template <int dim>
  unsigned int InterfaceMatcher<dim>::get_tria_index(const dealii::Triangulation<dim>& tria) const
  {
    if (&tria == &tria_first)
      return 0;
    else if (&tria == &tria_second)
      return 1;

    Assert(false, dealii::ExcMessage("Triangulation not part of the interface matcher."));
    return dealii::numbers::invalid_unsigned_int;
  }
  template <int dim>
  InterfaceMatcher<dim>::interface_iterator::interface_iterator(
      typename std::vector<InterfacePair>::const_iterator it)
  {
    it_ = it;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_iterator::value_type
  InterfaceMatcher<dim>::interface_iterator::operator*() const
  {
    return *it_;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_iterator&
  InterfaceMatcher<dim>::interface_iterator::operator++()
  {
    ++it_;
    return *this;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_iterator
  InterfaceMatcher<dim>::interface_iterator::operator++(int)
  {
    interface_iterator tmp = *this;
    ++it_;
    return tmp;
  }
  template <int dim>
  bool InterfaceMatcher<dim>::interface_iterator::operator==(const interface_iterator& other) const
  {
    return it_ == other.it_;
  }
  template <int dim>
  bool InterfaceMatcher<dim>::interface_iterator::operator!=(const interface_iterator& other) const
  {
    return it_ != other.it_;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::IteratorRange<typename InterfaceMatcher<dim>::interface_iterator>
  InterfaceMatcher<dim>::interface_range() const
  {
    return {interface_begin(), interface_end()};
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_iterator InterfaceMatcher<dim>::interface_begin() const
  {
    return interface_iterator(matches.begin());
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_iterator InterfaceMatcher<dim>::interface_end() const
  {
    return interface_iterator(matches.end());
  }

  template <int dim>
  InterfaceMatcher<dim>::interface_cell_iterator::interface_cell_iterator(
      const std::vector<InterfacePair>* matches, int tria_index, size_t pos)
      : matches_(matches), tria_index_(tria_index), pos_(pos)
  {
  }

  template <int dim>
  typename InterfaceMatcher<dim>::interface_cell_iterator::value_type
  InterfaceMatcher<dim>::interface_cell_iterator::operator*() const
  {
    return (tria_index_ == 0) ? (*matches_)[pos_].first.cell : (*matches_)[pos_].second.cell;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_cell_iterator&
  InterfaceMatcher<dim>::interface_cell_iterator::operator++()
  {
    advance_to_next_unique();
    return *this;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_cell_iterator
  InterfaceMatcher<dim>::interface_cell_iterator::operator++(int)
  {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }
  template <int dim>
  bool InterfaceMatcher<dim>::interface_cell_iterator::operator==(
      const interface_cell_iterator& other) const
  {
    Assert(matches_ == other.matches_,
        dealii::ExcMessage("Comparing iterators from different matchers."));
    Assert(tria_index_ == other.tria_index_,
        dealii::ExcMessage("Comparing iterators from different triangulations."));
    return pos_ == other.pos_;
  }
  template <int dim>
  bool InterfaceMatcher<dim>::interface_cell_iterator::operator!=(
      const interface_cell_iterator& other) const
  {
    return !(*this == other);
  }
  template <int dim>
  void InterfaceMatcher<dim>::interface_cell_iterator::advance_to_next_unique()
  {
    while (pos_ < matches_->size())
    {
      const auto& cell =
          (tria_index_ == 0) ? (*matches_)[pos_].first.cell : (*matches_)[pos_].second.cell;
      if (seen_.insert(cell).second) break;  // found a unique entity
      ++pos_;
    }
  }

  template <int dim>
  typename InterfaceMatcher<dim>::interface_cell_iterator
  InterfaceMatcher<dim>::interface_cell_begin(unsigned int tria_index) const
  {
    return interface_cell_iterator(&matches, tria_index, 0);
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_cell_iterator InterfaceMatcher<dim>::interface_cell_end(
      unsigned int tria_index) const
  {
    return interface_cell_iterator(&matches, tria_index, matches.size());
  }
  template <int dim>
  InterfaceMatcher<dim>::IteratorRange<typename InterfaceMatcher<dim>::interface_cell_iterator>
  InterfaceMatcher<dim>::interface_cell_range(unsigned int tria_index) const
  {
    return {interface_cell_begin(tria_index), interface_cell_end(tria_index)};
  }
  template <int dim>
  InterfaceMatcher<dim>::interface_face_iterator::interface_face_iterator(
      const std::vector<InterfacePair>* matches, int tria_index, size_t pos)
      : matches_(matches), tria_index_(tria_index), pos_(pos)
  {
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_face_iterator::value_type
  InterfaceMatcher<dim>::interface_face_iterator::operator*() const
  {
    return (tria_index_ == 0) ? (*matches_)[pos_].first.cell->face((*matches_)[pos_].first.face)
                              : (*matches_)[pos_].second.cell->face((*matches_)[pos_].second.face);
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_face_iterator&
  InterfaceMatcher<dim>::interface_face_iterator::operator++()
  {
    advance_to_next_unique();
    return *this;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_face_iterator
  InterfaceMatcher<dim>::interface_face_iterator::operator++(int)
  {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }
  template <int dim>
  bool InterfaceMatcher<dim>::interface_face_iterator::operator==(
      const interface_face_iterator& other) const
  {
    Assert(matches_ == other.matches_,
        dealii::ExcMessage("Comparing iterators from different matchers."));
    Assert(tria_index_ == other.tria_index_,
        dealii::ExcMessage("Comparing iterators from different triangulations."));
    return pos_ == other.pos_;
  }
  template <int dim>
  bool InterfaceMatcher<dim>::interface_face_iterator::operator!=(
      const interface_face_iterator& other) const
  {
    return !(*this == other);
  }
  template <int dim>
  void InterfaceMatcher<dim>::interface_face_iterator::advance_to_next_unique()
  {
    while (pos_ < matches_->size())
    {
      const auto& entity = (tria_index_ == 0) ? (*matches_)[pos_].first : (*matches_)[pos_].second;
      if (seen_.insert(entity.cell->face(entity.face)).second) break;  // found a unique entity
      ++pos_;
    }
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_face_iterator
  InterfaceMatcher<dim>::interface_face_begin(unsigned int tria_index) const
  {
    return interface_face_iterator(&matches, tria_index, 0);
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_face_iterator InterfaceMatcher<dim>::interface_face_end(
      unsigned int tria_index) const
  {
    return interface_face_iterator(&matches, tria_index, matches.size());
  }
  template <int dim>
  InterfaceMatcher<dim>::IteratorRange<typename InterfaceMatcher<dim>::interface_face_iterator>
  InterfaceMatcher<dim>::interface_face_range(unsigned int tria_index) const
  {
    return {interface_face_begin(tria_index), interface_face_end(tria_index)};
  }
  template <int dim>
  InterfaceMatcher<dim>::interface_entity_iterator::interface_entity_iterator(
      const std::vector<InterfacePair>* matches, int tria_index, size_t pos)
      : matches_(matches), tria_index_(tria_index), pos_(pos)
  {
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_entity_iterator::value_type
  InterfaceMatcher<dim>::interface_entity_iterator::operator*() const
  {
    return (tria_index_ == 0) ? (*matches_)[pos_].first : (*matches_)[pos_].second;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_entity_iterator&
  InterfaceMatcher<dim>::interface_entity_iterator::operator++()
  {
    advance_to_next_unique();
    return *this;
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_entity_iterator
  InterfaceMatcher<dim>::interface_entity_iterator::operator++(int)
  {
    auto tmp = *this;
    ++(*this);
    return tmp;
  }
  template <int dim>
  bool InterfaceMatcher<dim>::interface_entity_iterator::operator==(
      const interface_entity_iterator& other) const
  {
    Assert(matches_ == other.matches_,
        dealii::ExcMessage("Comparing iterators from different matchers."));
    Assert(tria_index_ == other.tria_index_,
        dealii::ExcMessage("Comparing iterators from different triangulations."));
    return pos_ == other.pos_;
  }
  template <int dim>
  bool InterfaceMatcher<dim>::interface_entity_iterator::operator!=(
      const interface_entity_iterator& other) const
  {
    return !(*this == other);
  }
  template <int dim>
  void InterfaceMatcher<dim>::interface_entity_iterator::advance_to_next_unique()
  {
    while (pos_ < matches_->size())
    {
      const auto& entity = (tria_index_ == 0) ? (*matches_)[pos_].first : (*matches_)[pos_].second;
      if (seen_.insert(entity).second) break;  // found a unique entity
      ++pos_;
    }
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_entity_iterator
  InterfaceMatcher<dim>::interface_entity_begin(int tria_index) const
  {
    return interface_entity_iterator(&matches, tria_index, 0);
  }
  template <int dim>
  typename InterfaceMatcher<dim>::interface_entity_iterator
  InterfaceMatcher<dim>::interface_entity_end(int tria_index) const
  {
    return interface_entity_iterator(&matches, tria_index, matches.size());
  }

  template <int dim>
  InterfaceMatcher<dim>::IteratorRange<typename InterfaceMatcher<dim>::interface_entity_iterator>
  InterfaceMatcher<dim>::interface_entity_range(unsigned int tria_index) const
  {
    return {interface_entity_begin(tria_index), interface_entity_begin(tria_index)};
  }

  namespace internal
  {
    template <int dim>
    std::vector<dealii::Point<dim>> get_sorted_face_vertices(
        const typename InterfaceMatcher<dim>::CellIterator& cell, unsigned int face_idx)
    {
      std::vector<dealii::Point<dim>> vertices;
      const auto& face = cell->face(face_idx);
      for (unsigned int v = 0; v < face->n_vertices(); ++v) vertices.push_back(face->vertex(v));
      std::sort(vertices.begin(), vertices.end(),
          [](const dealii::Point<dim>& a, const dealii::Point<dim>& b)
          {
            for (unsigned int d = 0; d < dim; ++d)
              if (std::abs(a[d] - b[d]) > 1e-14) return a[d] < b[d];
            return false;
          });
      return vertices;
    }
    template <int dim>
    bool compare_vertices(const std::vector<dealii::Point<dim>>& v1,
        const std::vector<dealii::Point<dim>>& v2, double tolerance)
    {
      if (v1.size() != v2.size()) return false;
      for (std::size_t i = 0; i < v1.size(); ++i)
        if ((v1[i] - v2[i]).norm() > tolerance) return false;
      return true;
    }
  }  // namespace internal



  template <int dim>
  InterfaceMatcher<dim> make_interface_matcher_across_boundary(
      const dealii::Triangulation<dim>& tria1, const dealii::Triangulation<dim>& tria2,
      double tolerance)
  {
    using Matcher = InterfaceMatcher<dim>;
    using CellIterator = typename Matcher::CellIterator;
    using InterfacePair = typename Matcher::InterfacePair;
    using Entity = typename Matcher::InterfaceEntity;

    std::vector<InterfacePair> matches;

    // Store all faces from tria2, indexed by sorted vertex coordinates
    std::vector<std::pair<std::vector<dealii::Point<dim>>, std::pair<CellIterator, unsigned int>>>
        tria2_faces;


    for (const auto& cell2 : tria2.active_cell_iterators())
    {
      if (not cell2->is_locally_owned()) continue;
      for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (cell2->face(f)->at_boundary())
        {
          auto vertices = internal::get_sorted_face_vertices<dim>(cell2, f);
          tria2_faces.emplace_back(vertices, std::make_pair(cell2, f));
        }
      }
    }

    for (const auto& cell1 : tria1.active_cell_iterators())
    {
      if (not cell1->is_locally_owned()) continue;
      for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if (cell1->face(f)->at_boundary())
        {
          auto v1 = internal::get_sorted_face_vertices<dim>(cell1, f);
          for (auto& face2 : tria2_faces)
          {
            if (internal::compare_vertices(v1, face2.first, tolerance))
            {
              matches.emplace_back(
                  InterfacePair(Entity(cell1, f, dealii::numbers::invalid_unsigned_int),
                      Entity(face2.second.first, face2.second.second,
                          dealii::numbers::invalid_unsigned_int)));
              break;
            }
          }
        }
      }
    }
    return InterfaceMatcher<dim>(tria1, tria2, std::move(matches));
  }

}  // namespace DealiiFSI



FOUR_C_NAMESPACE_CLOSE


#endif  // INC_4C_DEAL_II_FSI_TOOLS_HPP
