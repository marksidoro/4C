#ifndef INC_4C_DEAL_II_QUADRATURE_TRANSFORM_HPP
#define INC_4C_DEAL_II_QUADRATURE_TRANSFORM_HPP
#include "4C_config.hpp"

#include "4C_fem_general_utils_integration.hpp"

#include "deal.II/base/quadrature.h"

FOUR_C_NAMESPACE_OPEN

namespace DealiiWrappers
{

  namespace QuadTools
  {
    namespace deal_to_four_c
    {
      template <int dim>
      dealii::Point<dim> transform_unit_cell_point(const dealii::Point<dim>& deal_unit_cell_point)
      {
        auto helper = deal_unit_cell_point * 2;
        for (int d = 0; d < dim; ++d)
        {
          helper[d] -= 1;
        }
        return helper;
      }

      template <int dim>
      double transform_weight(const double deal_weight)
      {
        return deal_weight * std::pow(2, dim);
      }

      template <int dim>
      constexpr double transformation_scaling()
      {
        return std::pow(2, dim);
      }


      /**
       * Transform a deal.II quadrature defined on its unit cell to a deal.II quadrature object
       * defined on the 4C unit cell
       * @tparam dim       * @param quadrature       * @return
       */
      template <int dim>
      dealii::Quadrature<dim> transform_quadrature(const dealii::Quadrature<dim>& quadrature)
      {
        auto n_quad_points_ = quadrature.size();
        // Transform the quadrature points to the 4C unit cell
        std::vector<dealii::Point<dim>> transformed_quadrature_points(n_quad_points_);
        std::vector<double> transformed_weights(n_quad_points_);

        for (unsigned int q = 0; q < n_quad_points_; ++q)
        {
          transformed_quadrature_points[q] =
              deal_to_four_c::transform_unit_cell_point(quadrature.point(q));
          transformed_weights[q] = deal_to_four_c::transform_weight<dim>(quadrature.weight(q));
        }
        // Create the 4C quadrature object
        // std::move here since we are not keeping helper variables
        return dealii::Quadrature<dim>(
            std::move(transformed_quadrature_points), std::move(transformed_weights));
      }


      /**
       * Transform a deal.II quadrature object to a 4C quadrature object
       * WITHOUT transforming the underlying points and weights
       * @tparam dim       * @param quadrature
       * @return
       */
      template <int dim>
      Core::FE::IntPointsAndWeights<dim> to_integration_points(
          const dealii::Quadrature<dim>& quadrature)
      {
        FOUR_C_ASSERT(quadrature.size() <= Core::FE::IntegrationPoints<dim>::max_nquad,
            "Due to the 4C internal implementation only quadrature rules with at most {} are "
            "allowed",
            Core::FE::IntegrationPoints<dim>::max_nquad);

        Core::FE::IntegrationPoints<dim> integration_points(
            Core::FE::QuadratureRule<dim>::undefined);

        integration_points.nquad = quadrature.size();
        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          for (int d = 0; d < dim; ++d)
          {
            integration_points.points[q][d] = quadrature.get_point(q)[d];
          }
          integration_points.weights[q] = transform_weight<dim>(quadrature.weight(q));
        }
        return integration_points;
      }

    }  // namespace deal_to_four_c

    namespace four_c_to_deal
    {
      template <int dim>
      dealii::Point<dim> transform_unit_cell_point(const dealii::Point<dim>& four_c_unit_cell_point)
      {
        return (four_c_unit_cell_point + 1) / 2;
      }

      template <int dim>
      double transform_weight(const double deal_weight)
      {
        return deal_weight * std::pow(0.5, dim);
      }

      template <int dim>
      constexpr double transformation_scaling()
      {
        return std::pow(0.5, dim);
      }

      /**
       * Transform a 4C quadrature object to a deal.II quadrature object
       * WITHOUT transforming the underlying points and weights
       * @tparam dim       * @param quadrature       * @return
       */
      template <int dim>
      dealii::Quadrature<dim> from_integration_points(
          const Core::FE::IntPointsAndWeights<dim>& quadrature)
      {
        // extract the points and weights from the quadrature object
        std::vector<dealii::Point<dim>> quadrature_points(quadrature.num_points());
        std::vector<double> weights(quadrature.num_points());

        for (unsigned int q = 0; q < quadrature.num_points(); ++q)
        {
          dealii::Point<dim> point;
          for (int d = 0; d < dim; ++d)
          {
            point[d] = quadrature.points[q][d];
          }
          quadrature_points[q] = std::move(point);
          weights[q] = quadrature.weights[q];
        }
        return dealii::Quadrature<dim>(std::move(quadrature_points), std::move(weights));
      }


      /**
       * Transform a deal.II quadrature defined on a 4C unit cell to a deal.II quadrature object
       * now defined on the deal.II unit cell
       */
      template <int dim>
      dealii::Quadrature<dim> transform_quadrature(const dealii::Quadrature<dim>& quadrature)
      {
        auto n_quad_points_ = quadrature.size();
        // Transform the quadrature points to the 4C unit cell
        std::vector<dealii::Point<dim>> transformed_quadrature_points(n_quad_points_);
        std::vector<double> transformed_weights(n_quad_points_);

        for (unsigned int q = 0; q < n_quad_points_; ++q)
        {
          transformed_quadrature_points[q] =
              four_c_to_deal::transform_unit_cell_point(quadrature.point(q));
          transformed_weights[q] = four_c_to_deal::transform_weight<dim>(quadrature.weight(q));
        }
        // Create the 4C quadrature object
        // std::move here since we are not keeping helper variables
        return dealii::Quadrature<dim>(
            std::move(transformed_quadrature_points), std::move(transformed_weights));
      }

    }  // namespace four_c_to_deal



    template <int dim>
    bool is_in_four_c_unit_cell(const dealii::Point<dim>& point)
    {
      for (int d = 0; d < dim; ++d)
      {
        if (point[d] < -1.0 || point[d] > 1.0) return false;
      }
      return true;
    }

    template <int dim>
    bool is_in_deal_ii_unit_cell(const dealii::Point<dim>& point)
    {
      for (int d = 0; d < dim; ++d)
      {
        if (point[d] < 0.0 || point[d] > 1.0) return false;
      }
      return true;
    }
  }  // namespace QuadTools


  /**
   * Class to handle the Transfer of quadrature points and weights from deal.II to 4C.
   * The issue is that deal.II uses a different reference cell than 4C for Hex elements
   * 4C uses the [-1, 1]^dim cell while deal.II uses [0, 1]^dim
   * This class is constructed by either providing a deal.II quadrature object or a
   * Core::Fe::IntPointsAndWeights object with initialized quadrature points and weights.
   *
   *
   */
  template <int dim>
  class HexElementContext
  {
    dealii::Quadrature<dim> deal_type_quadrature_;
    dealii::Quadrature<dim> four_c_type_quadrature_;

    unsigned int n_quad_points_ = 0;

   public:
    explicit HexElementContext(const dealii::Quadrature<dim>& dealii_quadrature)
    {
      deal_type_quadrature_ = dealii_quadrature;
      n_quad_points_ = deal_type_quadrature_.size();
      four_c_type_quadrature_ =
          QuadTools::deal_to_four_c::transform_quadrature(deal_type_quadrature_);
    }

    explicit HexElementContext(const Core::FE::IntPointsAndWeights<dim>& four_c_quadrature)
    {
      four_c_type_quadrature_ =
          QuadTools::four_c_to_deal::from_integration_points(four_c_quadrature);
      n_quad_points_ = four_c_type_quadrature_.size();
      deal_type_quadrature_ =
          QuadTools::four_c_to_deal::transform_quadrature(four_c_type_quadrature_);
    }


    unsigned int n_quad_points() const { return n_quad_points_; }

    const std::vector<dealii::Point<dim>>& get_deal_ii_points() const
    {
      return deal_type_quadrature_.get_points();
    }
    const std::vector<double>& get_deal_ii_weights() const
    {
      return deal_type_quadrature_.get_weights();
    }

    double deal_ii_scaling() const
    {
      return QuadTools::deal_to_four_c::transformation_scaling<dim>();
    }

    const std::vector<dealii::Point<dim>>& get_four_c_points() const
    {
      return four_c_type_quadrature_.get_points();
    }
    const std::vector<double>& get_four_c_weights() const
    {
      return four_c_type_quadrature_.get_weights();
    }

    double four_c_scaling() const
    {
      return QuadTools::four_c_to_deal::transformation_scaling<dim>();
    }

    const dealii::Quadrature<dim>& get_deal_ii_quadrature() const { return deal_type_quadrature_; }
    const dealii::Quadrature<dim>& get_four_c_quadrature() const { return four_c_type_quadrature_; }
  };



}  // namespace DealiiWrappers


FOUR_C_NAMESPACE_CLOSE


#endif  // INC_4C_DEAL_II_QUADRATURE_TRANSFORM_HPP
