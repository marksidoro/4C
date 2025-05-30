// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PAIR_MORTAR_HPP
#define FOUR_C_FBI_BEAM_TO_FLUID_MESHTYING_PAIR_MORTAR_HPP


#include "4C_config.hpp"

#include "4C_fbi_beam_to_fluid_meshtying_pair_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace BeamInteraction
{
  /**
   * \brief Class for beam to fluid meshtying using mortar shape functions for the contact
   * tractions.
   * @param beam Type from GeometryPair::ElementDiscretization... representing the beam.
   * @param fluid Type from GeometryPair::ElementDiscretization... representing the fluid.
   * @param mortar Type from BeamInteraction::ElementDiscretization... representing the mortar shape
   * functions.
   */
  template <typename Beam, typename Fluid, typename Mortar>
  class BeamToFluidMeshtyingPairMortar : public BeamToFluidMeshtyingPairBase<Beam, Fluid>
  {
   private:
    //! Shortcut to base class.
    using base_class = BeamToFluidMeshtyingPairBase<Beam, Fluid>;

    //! Scalar type for FAD variables.
    using scalar_type = typename base_class::scalar_type;

   public:
    /**
     * \brief Standard Constructor
     */
    BeamToFluidMeshtyingPairMortar();



    /**
     * \brief Evaluate the mortar matrices $D$ and $M$ for this meshtying element pair.
     * @param local_D (out) Local mortar matrix $D$.
     * @param local_M (out) Local mortar matrix $M$.
     * @param local_kappa (out) Local scaling vector.
     * @param local_constraint_offset (outl) Local constraint offset vector.
     * @return True if pair is in contact.
     */
    bool evaluate_dm(Core::LinAlg::SerialDenseMatrix& local_D,
        Core::LinAlg::SerialDenseMatrix& local_M, Core::LinAlg::SerialDenseVector& local_kappa,
        Core::LinAlg::SerialDenseVector& local_constraint_offset) override;

    /**
     * \brief This pair enforces constraints via a mortar-type method, which requires an own
     * assembly method (provided by the mortar manager).
     */
    inline bool is_assembly_direct() const override { return false; };

    /**
     * \brief Add the visualization of this pair to the vtu output writer. This will
     * add mortar specific data to the output.
     * @param visualization_writer Object that manages all visualization related data for beam
     * to fluid pairs
     * @param visualization_params Parameter list
     */
    void get_pair_visualization(
        std::shared_ptr<BeamToSolidVisualizationOutputWriterBase> visualization_writer,
        Teuchos::ParameterList& visualization_params) const override;

   protected:
    virtual void evaluate_penalty_force(Core::LinAlg::Matrix<3, 1, scalar_type>& force,
        const GeometryPair::ProjectionPoint1DTo3D<double>& projected_gauss_point,
        Core::LinAlg::Matrix<3, 1, scalar_type> v_beam) const;
  };
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif
