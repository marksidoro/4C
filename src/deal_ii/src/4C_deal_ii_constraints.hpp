#ifndef INC_4C_DEAL_II_CONSTRAINTS_HPP
#define INC_4C_DEAL_II_CONSTRAINTS_HPP

#include "4C_config.hpp"

FOUR_C_NAMESPACE_OPEN
/**
 * Function to build and fill an AffineConstraints object from the dirichlet boundary conditions
 * given in the discretization object.
 * @tparam dim
 * @tparam spacedim
 * @tparam number
 * @param constraints
 * @param context
 * @param discretization
 */
/* template <int dim, int spacedim = dim, typename number = double>
 void build_equivalent_dirichlet_constraints(dealii::AffineConstraints<number>& constraints,
     const Core::FE::Discretization& discretization,
     const Teuchos::ParameterList& params = Teuchos::ParameterList())
 {
   auto dirichlet_values =
       std::make_shared<Core::LinAlg::Vector<double>>(*discretization.dof_row_map());
   auto dirichlet_index_extractor = std::make_shared<Core::LinAlg::MapExtractor>();


   Core::FE::Utils::evaluate_dirichlet(discretization, params, dirichlet_values, nullptr, nullptr,
       nullptr, dirichlet_index_extractor);


   const auto dirichlet_map = dirichlet_index_extractor->cond_map();
   const auto full_map = dirichlet_index_extractor->full_map();

   dealii::IndexSet dirichlet_indices(dirichlet_map->get_epetra_block_map());
   dealii::IndexSet full_indices(full_map->get_epetra_block_map());


   constraints.clear();
   constraints.reinit(full_indices, dirichlet_indices);

   for (const auto global_index : dirichlet_indices)
   {
     auto local_index = dirichlet_map->lid(global_index);
     FOUR_C_ASSERT(local_index != -1,
         "The local index for the global index {} is -1. This means that the global index was not"
         "found which should not happen",
         global_index);

     const auto value = dirichlet_values->operator[](local_index);
     // test if the value is not zero, since then we have to add an inhomogenity
     if (std::abs(value) > std::numeric_limits<double>::epsilon())
     {
       constraints.set_inhomogeneity(global_index, value);
     }
     else
     {
       constraints.constrain_dof_to_zero(global_index);
     }
   }
   constraints.close();
 }*/


FOUR_C_NAMESPACE_CLOSE


#endif  // INC_4C_DEAL_II_CONSTRAINTS_HPP
