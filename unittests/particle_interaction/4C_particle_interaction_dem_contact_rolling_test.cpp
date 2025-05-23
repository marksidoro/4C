// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_particle_interaction_dem_contact_rolling.hpp"

#include "4C_particle_interaction_utils.hpp"
#include "4C_unittest_utils_assertions_test.hpp"

#include <Teuchos_ParameterList.hpp>

namespace
{
  using namespace FourC;

  class DEMContactRollingViscousTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<ParticleInteraction::DEMContactRollingViscous> contactrolling_;

    const double e_ = 0.8;
    const double nue_ = 0.4;
    const double mu_rolling_ = 0.2;

    const double young_ = 200.0e3;
    const double v_max_ = 0.025;

    const double k_normal_ = 4.0;

    DEMContactRollingViscousTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_dem;
      params_dem.set("COEFF_RESTITUTION", e_);
      params_dem.set("POISSON_RATIO", nue_);
      params_dem.set("FRICT_COEFF_ROLL", mu_rolling_);

      params_dem.set("YOUNG_MODULUS", young_);
      params_dem.set("MAX_VELOCITY", v_max_);

      // create rolling contact handler
      contactrolling_ = std::make_unique<ParticleInteraction::DEMContactRollingViscous>(params_dem);

      // init rolling contact handler
      contactrolling_->init();

      // setup rolling contact handler
      contactrolling_->setup(k_normal_);
    }
    // note: the public functions init() and setup() of class DEMContactRollingViscous are
    // called in the constructor and thus implicitly tested by all following unittests
  };

  TEST_F(DEMContactRollingViscousTest, effective_radius_particle)
  {
    const double rad_i = 1.2;
    const double rad_j = 0.8;
    const double gap = -0.3;

    double r_eff = 0.0;
    contactrolling_->effective_radius_particle(&rad_i, &rad_j, gap, r_eff);

    const double r_eff_ref = rad_i * rad_j / (rad_i + rad_j);

    EXPECT_NEAR(r_eff, r_eff_ref, 1.0e-12);
  }

  TEST_F(DEMContactRollingViscousTest, EffectiveRadiusParticleNullptrRadJ)
  {
    const double rad_i = 1.2;
    const double gap = -0.3;

    double r_eff = 0.0;
    contactrolling_->effective_radius_particle(&rad_i, nullptr, gap, r_eff);

    const double r_eff_ref = rad_i;

    EXPECT_NEAR(r_eff, r_eff_ref, 1.0e-12);
  }

  TEST_F(DEMContactRollingViscousTest, relative_rolling_velocity)
  {
    const double r_eff = 0.5;

    double e_ji[3] = {0.0};
    e_ji[0] = 1.0 / std::sqrt(21);
    e_ji[1] = 2.0 / std::sqrt(21);
    e_ji[2] = 4.0 / std::sqrt(21);

    double angvel_i[3] = {0.0};
    angvel_i[0] = -0.03;
    angvel_i[1] = 0.1;
    angvel_i[2] = 0.12;

    double angvel_j[3] = {0.0};
    angvel_j[0] = 0.15;
    angvel_j[1] = -0.2;
    angvel_j[2] = 0.0;

    double v_rel_rolling[3] = {0.0};
    contactrolling_->relative_rolling_velocity(r_eff, e_ji, angvel_i, angvel_j, v_rel_rolling);

    double v_rel_rolling_ref[3] = {0.0};
    ParticleInteraction::Utils::vec_set_cross(v_rel_rolling_ref, angvel_i, e_ji);
    ParticleInteraction::Utils::vec_add_cross(v_rel_rolling_ref, e_ji, angvel_j);

    FOUR_C_EXPECT_ITERABLE_NEAR(v_rel_rolling, v_rel_rolling_ref, 3, 1.0e-12);
  }

  TEST_F(DEMContactRollingViscousTest, RelativeRollingVelocityNullptrAngvelJ)
  {
    const double r_eff = 0.5;

    double e_ji[3] = {0.0};
    e_ji[0] = 1.0 / std::sqrt(21);
    e_ji[1] = 2.0 / std::sqrt(21);
    e_ji[2] = 4.0 / std::sqrt(21);

    double angvel_i[3] = {0.0};
    angvel_i[0] = -0.03;
    angvel_i[1] = 0.1;
    angvel_i[2] = 0.12;

    double v_rel_rolling[3] = {0.0};
    contactrolling_->relative_rolling_velocity(r_eff, e_ji, angvel_i, nullptr, v_rel_rolling);

    double v_rel_rolling_ref[3] = {0.0};
    ParticleInteraction::Utils::vec_set_cross(v_rel_rolling_ref, angvel_i, e_ji);

    FOUR_C_EXPECT_ITERABLE_NEAR(v_rel_rolling, v_rel_rolling_ref, 3, 1.0e-12);
  }

  TEST_F(DEMContactRollingViscousTest, rolling_contact_moment)
  {
    double gap_rolling[3] = {0.0};
    gap_rolling[0] = 0.1;
    gap_rolling[1] = 0.05;
    gap_rolling[2] = -0.25;

    bool stick_rolling = false;

    double e_ji[3] = {0.0};
    e_ji[0] = 1.0 / std::sqrt(21);
    e_ji[1] = 2.0 / std::sqrt(21);
    e_ji[2] = 4.0 / std::sqrt(21);

    double v_rel_rolling[3] = {0.0};
    v_rel_rolling[0] = -0.03;
    v_rel_rolling[1] = 0.1;
    v_rel_rolling[2] = 0.12;

    const double m_eff = 2.5;
    const double r_eff = 0.5;
    const double normalcontactforce = 1.5;

    double rollingcontactmoment[3] = {0.0};
    contactrolling_->rolling_contact_moment(gap_rolling, stick_rolling, e_ji, v_rel_rolling, m_eff,
        r_eff, mu_rolling_, normalcontactforce, rollingcontactmoment);

    double rollingcontactmoment_ref[3] = {0.0};
    const double fac = young_ / (1.0 - ParticleInteraction::Utils::pow<2>(nue_));
    const double c_1 = 1.15344;
    const double d_rolling_fac =
        mu_rolling_ * (1.0 - e_) / (c_1 * std::pow(fac, 0.4) * std::pow(v_max_, 0.2));
    const double d_rolling = d_rolling_fac * std::pow(0.5 * r_eff, -0.2);

    double rollingcontactforce[3];
    ParticleInteraction::Utils::vec_set_scale(
        rollingcontactforce, -(d_rolling * normalcontactforce), v_rel_rolling);

    ParticleInteraction::Utils::vec_set_cross(rollingcontactmoment_ref, rollingcontactforce, e_ji);
    ParticleInteraction::Utils::vec_scale(rollingcontactmoment_ref, r_eff);

    for (int i = 0; i < 3; ++i)
      EXPECT_NEAR(rollingcontactmoment[i], rollingcontactmoment_ref[i], 1.0e-12);
  }

  TEST_F(DEMContactRollingViscousTest, rolling_potential_energy)
  {
    double gap_rolling[3] = {0.0};

    const double rollingpotentialenergy_ref = 0.0;

    double rollingpotentialenergy = 0.0;
    contactrolling_->rolling_potential_energy(gap_rolling, rollingpotentialenergy);

    EXPECT_NEAR(rollingpotentialenergy, rollingpotentialenergy_ref, 1.0e-12);
  }

  class DEMContactRollingCoulombTest : public ::testing::Test
  {
   protected:
    std::unique_ptr<ParticleInteraction::DEMContactRollingCoulomb> contactrolling_;

    const double e_ = 0.8;
    const double nue_ = 0.4;
    const double mu_rolling_ = 0.2;

    const double k_normal_ = 4.0;

    DEMContactRollingCoulombTest()
    {
      // create a parameter list
      Teuchos::ParameterList params_dem;
      params_dem.set("COEFF_RESTITUTION", e_);
      params_dem.set("POISSON_RATIO", nue_);
      params_dem.set("FRICT_COEFF_ROLL", mu_rolling_);

      // create rolling contact handler
      contactrolling_ = std::make_unique<ParticleInteraction::DEMContactRollingCoulomb>(params_dem);

      // init rolling contact handler
      contactrolling_->init();

      // setup rolling contact handler
      contactrolling_->setup(k_normal_);
    }
    // note: the public functions init() and setup() of class DEMContactRollingViscous are
    // called in the constructor and thus implicitly tested by all following unittests
  };

  TEST_F(DEMContactRollingCoulombTest, effective_radius_particle)
  {
    const double rad_i = 1.2;
    const double rad_j = 0.8;
    const double gap = -0.3;

    double r_eff = 0.0;
    contactrolling_->effective_radius_particle(&rad_i, &rad_j, gap, r_eff);

    const double r_eff_ref = (rad_i + 0.5 * gap) * (rad_j + 0.5 * gap) / (rad_i + rad_j + gap);

    EXPECT_NEAR(r_eff, r_eff_ref, 1.0e-12);
  }

  TEST_F(DEMContactRollingCoulombTest, EffectiveRadiusParticleNullptrRadJ)
  {
    const double rad_i = 1.2;
    const double gap = -0.3;

    double r_eff = 0.0;
    contactrolling_->effective_radius_particle(&rad_i, nullptr, gap, r_eff);

    const double r_eff_ref = rad_i + gap;

    EXPECT_NEAR(r_eff, r_eff_ref, 1.0e-12);
  }

  TEST_F(DEMContactRollingCoulombTest, relative_rolling_velocity)
  {
    const double r_eff = 0.5;

    double e_ji[3] = {0.0};
    e_ji[0] = 1.0 / std::sqrt(21);
    e_ji[1] = 2.0 / std::sqrt(21);
    e_ji[2] = 4.0 / std::sqrt(21);

    double angvel_i[3] = {0.0};
    angvel_i[0] = -0.03;
    angvel_i[1] = 0.1;
    angvel_i[2] = 0.12;

    double angvel_j[3] = {0.0};
    angvel_j[0] = 0.15;
    angvel_j[1] = -0.2;
    angvel_j[2] = 0.0;

    double v_rel_rolling[3] = {0.0};
    contactrolling_->relative_rolling_velocity(r_eff, e_ji, angvel_i, angvel_j, v_rel_rolling);

    double v_rel_rolling_ref[3] = {0.0};
    ParticleInteraction::Utils::vec_set_cross(v_rel_rolling_ref, e_ji, angvel_i);
    ParticleInteraction::Utils::vec_add_cross(v_rel_rolling_ref, angvel_j, e_ji);
    ParticleInteraction::Utils::vec_scale(v_rel_rolling_ref, r_eff);

    FOUR_C_EXPECT_ITERABLE_NEAR(v_rel_rolling, v_rel_rolling_ref, 3, 1.0e-12);
  }

  TEST_F(DEMContactRollingCoulombTest, RelativeRollingVelocityNullptrAngvelJ)
  {
    const double r_eff = 0.5;

    double e_ji[3] = {0.0};
    e_ji[0] = 1.0 / std::sqrt(21);
    e_ji[1] = 2.0 / std::sqrt(21);
    e_ji[2] = 4.0 / std::sqrt(21);

    double angvel_i[3] = {0.0};
    angvel_i[0] = -0.03;
    angvel_i[1] = 0.1;
    angvel_i[2] = 0.12;

    double v_rel_rolling[3] = {0.0};
    contactrolling_->relative_rolling_velocity(r_eff, e_ji, angvel_i, nullptr, v_rel_rolling);

    double v_rel_rolling_ref[3] = {0.0};
    ParticleInteraction::Utils::vec_set_cross(v_rel_rolling_ref, e_ji, angvel_i);
    ParticleInteraction::Utils::vec_scale(v_rel_rolling_ref, r_eff);

    FOUR_C_EXPECT_ITERABLE_NEAR(v_rel_rolling, v_rel_rolling_ref, 3, 1.0e-12);
  }

  TEST_F(DEMContactRollingCoulombTest, RollingContactMomentStick)
  {
    double gap_rolling[3] = {0.0};
    gap_rolling[0] = 0.1;
    gap_rolling[1] = 0.05;
    gap_rolling[2] = -0.25;

    bool stick_rolling = true;

    double e_ji[3] = {0.0};
    e_ji[0] = 1.0 / std::sqrt(21);
    e_ji[1] = 2.0 / std::sqrt(21);
    e_ji[2] = 4.0 / std::sqrt(21);

    double v_rel_rolling[3] = {0.0};
    v_rel_rolling[0] = -0.03;
    v_rel_rolling[1] = 0.1;
    v_rel_rolling[2] = 0.12;

    const double m_eff = 2.5;
    const double r_eff = 0.5;
    const double normalcontactforce = 1.5;

    double rollingcontactmoment[3] = {0.0};
    contactrolling_->rolling_contact_moment(gap_rolling, stick_rolling, e_ji, v_rel_rolling, m_eff,
        r_eff, mu_rolling_, normalcontactforce, rollingcontactmoment);

    double gap_rolling_ref[3] = {0.06858666958093279, 0.05062425428544091, -0.05782673704478762};
    double rollingcontactmoment_ref[3] = {
        -0.1119618097864935, 0.09699534835316952, -0.02050722172996138};

    FOUR_C_EXPECT_ITERABLE_NEAR(gap_rolling, gap_rolling_ref, 3, 1.0e-12);

    for (int i = 0; i < 3; ++i)
      EXPECT_NEAR(rollingcontactmoment[i], rollingcontactmoment_ref[i], 1.0e-12);

    EXPECT_FALSE(stick_rolling);
  }

  TEST_F(DEMContactRollingCoulombTest, RollingContactMomentSlip)
  {
    double gap_rolling[3] = {0.0};
    gap_rolling[0] = 0.1;
    gap_rolling[1] = 0.05;
    gap_rolling[2] = -0.25;

    bool stick_rolling = false;

    double e_ji[3] = {0.0};
    e_ji[0] = 1.0 / std::sqrt(21);
    e_ji[1] = 2.0 / std::sqrt(21);
    e_ji[2] = 4.0 / std::sqrt(21);

    double v_rel_rolling[3] = {0.0};
    v_rel_rolling[0] = -0.03;
    v_rel_rolling[1] = 0.1;
    v_rel_rolling[2] = 0.12;

    const double m_eff = 2.5;
    const double r_eff = 0.5;
    const double normalcontactforce = 1.5;

    double rollingcontactmoment[3] = {0.0};
    contactrolling_->rolling_contact_moment(gap_rolling, stick_rolling, e_ji, v_rel_rolling, m_eff,
        r_eff, mu_rolling_, normalcontactforce, rollingcontactmoment);

    double gap_rolling_ref[3] = {0.06858666958093279, 0.05062425428544091, -0.05782673704478762};
    double rollingcontactmoment_ref[3] = {
        -0.1119618097864935, 0.09699534835316952, -0.02050722172996138};

    FOUR_C_EXPECT_ITERABLE_NEAR(gap_rolling, gap_rolling_ref, 3, 1.0e-12);

    for (int i = 0; i < 3; ++i)
      EXPECT_NEAR(rollingcontactmoment[i], rollingcontactmoment_ref[i], 1.0e-12);

    EXPECT_FALSE(stick_rolling);
  }

  TEST_F(DEMContactRollingCoulombTest, rolling_potential_energy)
  {
    double gap_rolling[3] = {0.0};
    gap_rolling[0] = 0.1;
    gap_rolling[1] = 0.05;
    gap_rolling[2] = -0.25;

    const double k_rolling = (1.0 - nue_) / (1.0 - 0.5 * nue_) * k_normal_;
    const double rollingpotentialenergy_ref =
        0.5 * k_rolling * ParticleInteraction::Utils::vec_dot(gap_rolling, gap_rolling);

    double rollingpotentialenergy = 0.0;
    contactrolling_->rolling_potential_energy(gap_rolling, rollingpotentialenergy);

    EXPECT_NEAR(rollingpotentialenergy, rollingpotentialenergy_ref, 1.0e-12);
  }
}  // namespace
