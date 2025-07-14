#ifndef INC_4C_DEAL_II_MAPPING_HPP
#define INC_4C_DEAL_II_MAPPING_HPP

#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q1.h>
FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{
  template <int dim, int spacedim>
  class Context;

  /**
   * Class holding a  mapping collection that is used to handle a collection of
   * dealii::Mapping objects describing the mapping from the reference cell to the real cell.
   * Since the Isoparametric mapping (using MappingFEField) requires additional data to be stored,
   * and kept alive for the lifetime of the MappingFEField, we use a pimpl structure to hide the
   * lifetime of the additional data.
   */
  template <int dim, int spacedim = dim>
  struct MappingContext
  {
    static MappingContext create_isoparametric_mapping(const Context<dim, spacedim>& context);
    static MappingContext create_linear_mapping(const Context<dim, spacedim>& context);
    const dealii::hp::MappingCollection<dim, spacedim>& get_mapping_collection() const;

   private:
    dealii::hp::MappingCollection<dim, spacedim> mapping_collection;

    struct ImplementationDetails
    {
      // =============================================================
      // Isoparametric mapping data
      dealii::DoFHandler<dim, spacedim> iso_dof_handler;
      dealii::LinearAlgebra::distributed::Vector<double> position_vector;
    };

    std::vector<std::unique_ptr<ImplementationDetails>> pimpl_;
  };


  namespace Internal
  {
    template <int dim, int spacedim = dim,
        typename VectorType = dealii::LinearAlgebra::distributed::Vector<double>>
    dealii::MappingFEField<dim, spacedim, VectorType> create_isoparametric_mapping(
        const Context<dim, spacedim>& context, VectorType& position_vector,
        dealii::DoFHandler<dim>& iso_dof_handler);
  }  // namespace Internal


  // ===========================================================================================
  // Implementation of the MappingContext methods

  template <int dim, int spacedim>
  MappingContext<dim, spacedim> MappingContext<dim, spacedim>::create_isoparametric_mapping(
      const Context<dim, spacedim>& context)
  {
    FOUR_C_ASSERT(context.n_finite_elements() == 1,
        "Currently only supported for the case that there is only one finite element in the "
        "context, since the underlying dealii::MappingFEField does not support multiple finite "
        "elements.");

    auto data_holder = std::make_unique<ImplementationDetails>();
    auto mapping = Internal::create_isoparametric_mapping<dim, spacedim>(
        context, data_holder->position_vector, data_holder->iso_dof_handler);
    MappingContext mapping_context;
    mapping_context.mapping_collection.push_back(mapping);
    mapping_context.pimpl_.push_back(std::move(data_holder));
    return mapping_context;
  }
  template <int dim, int spacedim>
  MappingContext<dim, spacedim> MappingContext<dim, spacedim>::create_linear_mapping(
      const Context<dim, spacedim>& context)
  {
    MappingContext mapping_context;
    for (unsigned int i = 0; i < context.n_finite_elements(); ++i)
    {
      mapping_context.mapping_collection.push_back(dealii::MappingQ1<dim, spacedim>());
      mapping_context.pimpl_.push_back(
          std::make_unique<ImplementationDetails>());  // empty implementation details
    }


    return mapping_context;
  }
  template <int dim, int spacedim>
  const dealii::hp::MappingCollection<dim, spacedim>&
  MappingContext<dim, spacedim>::get_mapping_collection() const
  {
    return mapping_collection;
  }

  namespace Internal
  {
    template <int dim, int spacedim, typename VectorType>
    dealii::MappingFEField<dim, spacedim, VectorType> create_isoparametric_mapping(
        const Context<dim, spacedim>& context, VectorType& position_vector,
        dealii::DoFHandler<dim>& iso_dof_handler)
    {
      FOUR_C_ASSERT(context.n_finite_elements() == 1,
          "Currently only supported for the case that there is only one finite element in the "
          "context, since the underlying dealii::MappingFEField does not support multiple finite "
          "elements.");

      // create an internal dofhandler using the finite element that is provided
      // this is potentially a multicomponent system, so we have to get
      // the scalar valued finite element
      const auto& fe_system = context.get_finite_elements()[0];
      FOUR_C_ASSERT(fe_system.n_base_elements() == 1,
          "The finite element must have exactly one base element, since we are creating an "
          "isoparametric mapping. This is not the case for the finite element '{}'.",
          fe_system.get_name());
      const auto& fe = fe_system.base_element(0);

      // create an FE System object that has the right dimension
      dealii::FESystem<dim, spacedim> isoparametric_fe(fe, spacedim);

      // create a DofHandler for the isoparametric mapping
      iso_dof_handler.reinit(context.get_triangulation());
      iso_dof_handler.distribute_dofs(isoparametric_fe);

      // create ghosted vector for the postions of the nodes
      auto locally_relevant_dofs = dealii::DoFTools::extract_locally_relevant_dofs(iso_dof_handler);
      position_vector.reinit(iso_dof_handler.locally_owned_dofs(), locally_relevant_dofs,
          iso_dof_handler.get_communicator());

      // Now fill the position vector with the positions of the nodes
      for (const auto& cell : iso_dof_handler.active_cell_iterators())
      {
        // skip ghost cells
        if (not cell->is_locally_owned()) continue;

        // get the equivalent element in four_c
        const auto* element = context.to_element(cell);
        const unsigned int n_nodes = element->num_node();
        const auto* nodes = element->nodes();


        // get the dof indices for the cell
        const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        std::vector<dealii::types::global_dof_index> dof_indices(dofs_per_cell);
        cell->get_dof_indices(dof_indices);

        FOUR_C_ASSERT(n_nodes * spacedim == dofs_per_cell,
            "Since this is an isoparametric mapping, the number of nodes x {} must be equal to "
            "the number of dofs per cell.",
            spacedim);


        // we now have to assign the postion of the nodes to the dof indices
        dealii::Vector<double> local_position_vector(dofs_per_cell);
        auto reordering =
            ConversionTools::FourCToDeal::reindex_shape_functions_scalar(element->shape());
        for (unsigned int n = 0; n < n_nodes; ++n)
        {
          const auto local_dealii_index = reordering[n];
          for (unsigned int d = 0; d < spacedim; ++d)
          {
            const auto local_vector_index =
                isoparametric_fe.component_to_system_index(d, local_dealii_index);
            local_position_vector[local_vector_index] = nodes[n]->x()[d];
          }
        }
        // now we can add the local position vector to the global position vector
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // only add local entries
          if (iso_dof_handler.locally_owned_dofs().is_element(dof_indices[i]))
          {
            position_vector[dof_indices[i]] = local_position_vector[i];
          }
        }
      }
      const dealii::ComponentMask mask(spacedim, true);
      return dealii::MappingFEField<dim, spacedim, VectorType>(
          iso_dof_handler, position_vector, mask);
    }
  }  // namespace Internal

}  // namespace DealiiWrappers
FOUR_C_NAMESPACE_CLOSE

#endif  // INC_4C_DEAL_II_MAPPING_HPP
