#ifndef INC_4C_DEAL_II_MIMIC_MAPPING_HPP
#define INC_4C_DEAL_II_MIMIC_MAPPING_HPP


#include "4C_config.hpp"

#include "4C_deal_ii_context.hpp"
#include "4C_deal_ii_quadrature_transform.hpp"
#include "4C_fem_discretization.hpp"
#include "4C_fem_general_element.hpp"
#include "4C_fem_general_extract_values.hpp"
#include "4C_fem_general_utils_fem_shapefunctions.hpp"
#include "4C_fem_geometry_position_array.hpp"
#include <4C_deal_ii_context_implementation.hpp>

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <Teuchos_SerialDenseSolver.hpp>



FOUR_C_NAMESPACE_OPEN


namespace DealiiWrappers
{

  /**
   * Base Interface for the mapping classes that are used to transform the
   * reference cell to the real cell.
   * @tparam dim
   * @tparam spacedim
   */
  template <int dim, int spacedim = dim>
  class MappingBase
  {
    const HexElementContext<dim>& quadrature_context_;



   public:
    static constexpr unsigned u_dim = dim;
    static constexpr unsigned u_spacedim = spacedim;


    virtual ~MappingBase();
    MappingBase(const HexElementContext<dim>& quadrature_contex);

    const HexElementContext<dim>& quadrature_context() const;

    struct MappingCellData
    {
      /**
       * Physical coordinates of the quadrature points in the real cell.
       */
      std::vector<dealii::Point<dim>> quadrature_points;

      /**
       * Jacobian of the mapping at the quadrature points in the reference cell.
       */
      std::vector<Core::LinAlg::SerialDenseMatrix> jacobians;

      /**
       * Inverse Jacobian of the mapping at the quadrature points in the reference cell.
       */
      std::vector<Core::LinAlg::SerialDenseMatrix> inverse_jacobians;

      /**
       * Determinant of the Jacobian at the quadrature points in the reference cell.
       */
      std::vector<double> jacobian_determinants;
    };


    /**
     * This function computes and fills the mapping data for a given element of the mesh.
     * This is the main function that needs to be implemented in the derived classes.
     * @param cell_data
     * @param element
     * @param init_for_deal_ii_ref_cell parameter which indicates if the mapping data should be
     * computed for the deal.II reference cell or for the 4C reference cell. This is specifically
     * important for the Jacobian as it might be necessary to scale it by a factor.
     * This factor is defined through the quadrature context.
     * By default this is set to true, meaning that the jacobians can be used together with
     * an FEValues object that is initialized with the deal.II reference cell.
     */
    virtual void compute_mapping_data(MappingCellData& cell_data,
        const Core::Elements::Element* element,
        const bool init_for_deal_ii_ref_cell = true) const = 0;


    /**
     * Compute the mapping data for a given cell of the triangulation. By default this function
     * extrats the element from the cell and calls the compute_mapping_data function.
     * @param cell_data
     * @param cell
     * @param context
     * @param discretization
     * @param init_for_deal_ii_ref_cell
     */
    virtual void compute_mapping_data(MappingCellData& cell_data,
        const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
        const Context<dim, spacedim>& context, const Core::FE::Discretization& discretization,
        const bool init_for_deal_ii_ref_cell = true) const;
  };


  /**
   * Class describing an Isoparametric mapping from the reference cell to the real cell.
   * This is the standard case in the 4C code and can be computed solely from the data provided
   * by the element class.
   * Mathematically this means that the mapping is defined by the shape functions of the element.
   * I.e. the mapping uses the same FEM approximation for the geometry as for the
   * actual computation.
   * @tparam dim
   * @tparam spacedim
   */
  template <int dim, int spacedim = dim>
  class IsoparametricMapping : public MappingBase<dim, spacedim>
  {
   public:
    IsoparametricMapping(const HexElementContext<dim>& quadrature_context);


    void compute_mapping_data(typename MappingBase<dim, spacedim>::MappingCellData& cell_data,
        const Core::Elements::Element* element,
        const bool init_for_deal_ii_ref_cell = true) const override;
  };



  namespace Internal::Evaluation
  {
    template <int dim>
    void evaluate_isoparametric_jaccobian(const Core::Elements::Element* element,
        const dealii::Point<dim>& ref_point,
        const Core::LinAlg::SerialDenseMatrix& node_coordinates,
        Core::LinAlg::SerialDenseMatrix& jaccobian)
    {
      FOUR_C_ASSERT(jaccobian.num_cols() == dim and jaccobian.num_rows() == dim,
          "Jacobian is not of size {} x {}", dim, dim);

      Core::LinAlg::SerialDenseMatrix gradient =
          Core::LinAlg::SerialDenseMatrix(dim, element->num_node());
      Core::FE::shape_function_deriv1_dim<dealii::Point<dim>, Core::LinAlg::SerialDenseMatrix, dim>(
          ref_point, gradient, element->shape());
      jaccobian.multiply(Teuchos::NO_TRANS, Teuchos::TRANS, 1.0, gradient, node_coordinates, 0.0);
    }
  }  // namespace Internal::Evaluation



  template <int dim, int spacedim>
  MappingBase<dim, spacedim>::~MappingBase() = default;
  template <int dim, int spacedim>
  MappingBase<dim, spacedim>::MappingBase(const HexElementContext<dim>& quadrature_contex)
      : quadrature_context_(quadrature_contex)
  {
  }
  template <int dim, int spacedim>
  const HexElementContext<dim>& MappingBase<dim, spacedim>::quadrature_context() const
  {
    return quadrature_context_;
  }
  template <int dim, int spacedim>
  void MappingBase<dim, spacedim>::compute_mapping_data(MappingCellData& cell_data,
      const typename dealii::Triangulation<dim, spacedim>::cell_iterator& cell,
      const Context<dim, spacedim>& context, const Core::FE::Discretization& discretization,
      const bool init_for_deal_ii_ref_cell) const
  {
    const Core::Elements::Element* current_four_c_element_ =
        Internal::to_element(context, discretization, cell);
    compute_mapping_data(cell_data, current_four_c_element_, init_for_deal_ii_ref_cell);
  }


  template <int dim, int spacedim>
  IsoparametricMapping<dim, spacedim>::IsoparametricMapping(
      const HexElementContext<dim>& quadrature_contex)
      : MappingBase<dim, spacedim>(quadrature_contex)
  {
  }

  template <int dim, int spacedim>
  void IsoparametricMapping<dim, spacedim>::compute_mapping_data(
      typename MappingBase<dim, spacedim>::MappingCellData& cell_data,
      const Core::Elements::Element* element, const bool init_for_deal_ii_ref_cell) const
  {
    const auto node_coords = Core::Geo::initial_position_array(element);

    unsigned n_quad_points = this->quadrature_context().n_quad_points();
    const auto& four_c_quadrature = this->quadrature_context().get_four_c_quadrature();


    std::fill_n(
        cell_data.jacobians.begin(), n_quad_points, Core::LinAlg::SerialDenseMatrix(dim, dim));
    std::fill_n(cell_data.inverse_jacobians.begin(), n_quad_points,
        Core::LinAlg::SerialDenseMatrix(dim, dim));


    cell_data.jacobian_determinants.resize(n_quad_points);

    Teuchos::SerialDenseSolver<int, double> matrix_inverter;

    // Transform the quadrature points to the 4C unit cell
    for (unsigned int q = 0; q < n_quad_points; ++q)
    {
      // evaluate the jacobian at the quadrature points, i.e. the jaccobian of
      // the transformation from the unit cell to the real cell

      Internal::Evaluation::evaluate_isoparametric_jaccobian(
          element, four_c_quadrature.point(q), node_coords, cell_data.jacobians[q]);

      if (init_for_deal_ii_ref_cell)
      {
        cell_data.jacobians[q].scale(this->quadrature_context().deal_ii_scaling());
      }

      Core::LinAlg::Matrix<dim, dim> inv_helper(
          cell_data.jacobians[q]);  // copy the jacobian to a helper matrix

      cell_data.jacobian_determinants[q] = inv_helper.invert();
      // tarnsfer the data by hand
      for (int r = 0; r < dim; ++r)
      {
        for (int c = 0; c < dim; ++c)
        {
          cell_data.inverse_jacobians[q](r, c) = inv_helper(r, c);
        }
      }
    }
  }
}  // namespace DealiiWrappers
FOUR_C_NAMESPACE_CLOSE

#endif
