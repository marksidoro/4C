#ifndef INC_4C_DEAL_II_MIMIC_MAPPING_HPP
#define INC_4C_DEAL_II_MIMIC_MAPPING_HPP


#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_quadrature_transform.hpp"
#include "4C_deal_ii_triangulation.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_node.hpp"
#include "4C_fem_general_shape_function_type.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include <4C_deal_ii_context_implementation.hpp>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/hp/fe_collection.h>



FOUR_C_NAMESPACE_OPEN


namespace DealiiWrappers
{

  template <int dim, int spacedim = dim>
  class MappingBase
  {
    const Context<dim, spacedim>& context_;
    const Core::FE::Discretization& discretization_;
    const HexElementContext<dim>& quadrature_context_;

   public:
    MappingBase(const HexElementContext<dim>& quadrature_contex,
        const Context<dim, spacedim>& context, const Core::FE::Discretization& discretization)
        : context_(context), discretization_(discretization), quadrature_context_(quadrature_contex)
    {
    }
    const HexElementContext<dim>& quadrature_context() const { return quadrature_context_; }
    const Context<dim, spacedim>& context() const { return context_; }
    const Core::FE::Discretization& discretization() const { return discretization_; }


    struct MappingCellData
    {
      std::vector<dealii::Point<dim>> quadrature_points;
      std::vector<Core::LinAlg::Matrix<dim, spacedim>> jacobians;
      std::vector<Core::LinAlg::Matrix<dim, spacedim>> inverse_jacobians;
      std::vector<double> jacobian_determinants;
    };

    virtual void compute_mapping_data(MappingCellData& cell_data,
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
        const bool init_for_deal_ii_ref_cell = true) const = 0;
  };


  namespace Internal::Evaluation
  {
    template <int dim, typename NodeMatrixType>
    void evaluate_isoparametric_jaccobian(const Core::Elements::Element* element,
        const dealii::Point<dim>& ref_point, const NodeMatrixType& node_coordinates,
        Core::LinAlg::Matrix<dim, dim>& jaccobian)
    {
      Core::LinAlg::SerialDenseMatrix gradient =
          Core::LinAlg::SerialDenseMatrix(dim, element->num_node());
      Core::FE::shape_function_deriv1_dim<dealii::Point<dim>, Core::LinAlg::SerialDenseMatrix, dim>(
          ref_point, gradient, element->shape());
      jaccobian.multiply_nt(gradient, node_coordinates);
    }
  }  // namespace Internal::Evaluation

  template <int dim, int spacedim = dim>
  class IsoparametricMapping : MappingBase<dim, spacedim>
  {
   public:
    void compute_mapping_data(typename MappingBase<dim, spacedim>::MappingCellData& cell_data,
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
        const bool init_for_deal_ii_ref_cell = true) const override
    {
      Core::Elements::Element* current_four_c_element_ =
          Internal::to_element(this->context(), this->discretization(), cell);

      const auto node_coords = Core::Geo::initial_position_array(current_four_c_element_);

      unsigned n_quad_points = this->quadrature_context().n_quad_points();
      const auto& four_c_quadrature = this->quadrature_context().get_four_c_quadrature();

      cell_data.jacobians.resize(n_quad_points);
      cell_data.inverse_jacobians.resize(n_quad_points);
      cell_data.jacobian_determinants.resize(n_quad_points);

      // Transform the quadrature points to the 4C unit cell
      for (unsigned int q = 0; q < n_quad_points; ++q)
      {
        // evaluate the jacobian at the quadrature points, i.e. the jaccobian of
        // the transformation from the unit cell to the real cell

        Internal::Evaluation::evaluate_isoparametric_jaccobian(current_four_c_element_,
            four_c_quadrature.get_quadrature_point(q), node_coords, cell_data.jacobians[q]);

        if (init_for_deal_ii_ref_cell)
        {
          cell_data.jacobians *= this->quadrature_context().deal_ii_scaling();
        }

        // compute its inverse and determinant and cache the values
        cell_data.jacobian_determinants[q] =
            cell_data.jacobians[q].invert(cell_data.inverse_jacobians[q]);
      }
    }
  };

}  // namespace DealiiWrappers
FOUR_C_NAMESPACE_CLOSE

#endif
