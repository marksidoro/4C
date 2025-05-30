// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_config.hpp"

#include "4C_comm_utils.hpp"
#include "4C_fem_condition_definition.hpp"
#include "4C_fem_general_element_definition.hpp"
#include "4C_fem_general_utils_createdis.hpp"
#include "4C_global_data.hpp"
#include "4C_global_legacy_module.hpp"
#include "4C_global_legacy_module_validconditions.hpp"
#include "4C_io_input_file_utils.hpp"
#include "4C_io_input_spec_builders.hpp"
#include "4C_pre_exodus_readbc.hpp"
#include "4C_pre_exodus_validate.hpp"
#include "4C_pre_exodus_writedat.hpp"
#include "4C_utils_singleton_owner.hpp"

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <memory>


using namespace FourC;

namespace
{
  /*----------------------------------------------------------------------*/
  /* create default bc file                                               */
  /*----------------------------------------------------------------------*/
  int create_default_bc_file(Core::IO::Exodus::Mesh& mymesh)
  {
    using namespace FourC;

    std::string defaultbcfilename = "default.bc";
    std::cout << "found no BC specification file --> creating " << defaultbcfilename << std::endl;

    // open default bc specification file
    std::ofstream defaultbc(defaultbcfilename.c_str());
    if (!defaultbc) FOUR_C_THROW("failed to open file: {}", defaultbcfilename);

    // write mesh verbosely
    defaultbc << "----------- Mesh contents -----------" << std::endl << std::endl;
    mymesh.print(defaultbc, false);

    // give examples for element and boundary condition syntax
    defaultbc << "---------- Syntax examples ----------" << std::endl
              << std::endl
              << "Element Block, named: " << std::endl
              << "of Shape: TET4" << std::endl
              << "has 9417816 Elements" << std::endl
              << "'*eb0=\"ELEMENT\"'" << std::endl
              << "sectionname=\"FLUID\"" << std::endl
              << "description=\"MAT 1 NA Euler\"" << std::endl
              << "elementname=\"FLUID\" \n"
              << std::endl
              << "Element Block, named: " << std::endl
              << "of Shape: HEX8" << std::endl
              << "has 9417816 Elements" << std::endl
              << "'*eb0=\"ELEMENT\"'" << std::endl
              << "sectionname=\"STRUCTURE\"" << std::endl
              << "description=\"MAT 1 KINEM nonlinear\"" << std::endl
              << "elementname=\"SOLID\" \n"
              << std::endl
              << "Node Set, named:" << std::endl
              << "Property Name: INFLOW" << std::endl
              << "has 45107 Nodes" << std::endl
              << "'*ns0=\"CONDITION\"'" << std::endl
              << "sectionname=\"DESIGN SURF DIRICH CONDITIONS\"" << std::endl
              << "description=\"NUMDOF 6 ONOFF 1 1 1 0 0 0 VAL 2.0 0.0 0.0 0.0 0.0 0.0 FUNCT 1 0 0 "
                 "0 0 0\""
              << std::endl
              << std::endl;

    defaultbc << "MIND that you can specify a condition also on an ElementBlock, just replace "
                 "'ELEMENT' with 'CONDITION'"
              << std::endl;
    defaultbc << "The 'E num' in the dat-file depends on the order of the specification below"
              << std::endl;
    defaultbc << "------------------------------------------------BCSPECS" << std::endl
              << std::endl;

    // write ElementBlocks with specification proposal
    for (const auto& [eb_id, eb] : mymesh.get_element_blocks())
    {
      eb.print(defaultbc);
      defaultbc << "*eb" << eb_id << "=\"ELEMENT\"" << std::endl
                << "sectionname=\"\"" << std::endl
                << "description=\"\"" << std::endl
                << "elementname=\"\"" << std::endl
                << std::endl;
    }

    // write NodeSets with specification proposal
    const std::map<int, Core::IO::Exodus::NodeSet> mynodesets = mymesh.get_node_sets();
    std::map<int, Core::IO::Exodus::NodeSet>::const_iterator ins;
    for (ins = mynodesets.begin(); ins != mynodesets.end(); ++ins)
    {
      ins->second.print(defaultbc);
      defaultbc << "*ns" << ins->first << "=\"CONDITION\"" << std::endl
                << "sectionname=\"\"" << std::endl
                << "description=\"\"" << std::endl
                << std::endl;
    }

    // write SideSets with specification proposal
    const std::map<int, Core::IO::Exodus::SideSet> mysidesets = mymesh.get_side_sets();
    std::map<int, Core::IO::Exodus::SideSet>::const_iterator iss;
    for (iss = mysidesets.begin(); iss != mysidesets.end(); ++iss)
    {
      iss->second.print(defaultbc);
      defaultbc << "*ss" << iss->first << "=\"CONDITION\"" << std::endl
                << "sectionname=\"\"" << std::endl
                << "description=\"\"" << std::endl
                << std::endl;
    }

    // print validconditions as proposal
    defaultbc << "-----------------------------------------VALIDCONDITIONS" << std::endl;
    std::vector<Core::Conditions::ConditionDefinition> condlist = Global::valid_conditions();
    Global::print_empty_condition_definitions(defaultbc, condlist);

    // print valid element lines as proposal (parobjects have to be registered for doing this!)
    defaultbc << std::endl << std::endl;
    Core::Elements::ElementDefinition ed;
    ed.print_element_dat_header_to_stream(defaultbc);

    // close default bc specification file
    if (defaultbc.is_open()) defaultbc.close();

    return 0;
  }
}  // namespace

/**
 *
 * Pre_exodus contains classes to open and preprocess exodusII files into the
 * discretization of 4C. It uses the "valid-parameters"-list defined in 4C for preparing
 * a up-to-date 4C header and another file specifying element and boundary
 * specifications. As result either a preliminary input file set is suggestioned,
 * or the well-known .dat file is created.
 *
 */
int main(int argc, char** argv)
{
  using namespace FourC;
  Core::Utils::SingletonOwnerRegistry::ScopeGuard guard;

  // communication
  MPI_Init(&argc, &argv);

  global_legacy_module_callbacks().RegisterParObjectTypes();

  // create default communicators
  std::shared_ptr<Core::Communication::Communicators> communicators =
      Core::Communication::create_comm({});
  Global::Problem::instance()->set_communicators(communicators);
  MPI_Comm comm = communicators->global_comm();

  try
  {
    if ((Core::Communication::num_mpi_ranks(comm) > 1))
      FOUR_C_THROW("Using more than one processor is not supported.");

    std::string exofile;
    std::string bcfile;
    std::string headfile;
    std::string outfile;
    std::string cline;

    bool twodim = false;


    Teuchos::CommandLineProcessor My_CLP;
    My_CLP.setDocString("This preprocessor converts Exodus2 files into 4C input files\n");
    My_CLP.throwExceptions(false);
    My_CLP.setOption("exo", &exofile, "exodus file to open");
    My_CLP.setOption("bc", &bcfile, "bc's and ele's file to open (custom format)");
    My_CLP.setOption("head", &headfile, "4C header file to open (yaml format)");
    My_CLP.setOption("out", &outfile, "output file name, defaults to exodus file name");

    // switch for generating a 2d file
    My_CLP.setOption("d2", "d3", &twodim, "space dimensions in .dat-file: d2: 2D, d3: 3D");

    Teuchos::CommandLineProcessor::EParseCommandLineReturn parseReturn = My_CLP.parse(argc, argv);

    if (parseReturn == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED)
    {
      return 0;
    }
    if (parseReturn != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
      FOUR_C_THROW("CommandLineProcessor reported an error");
    }

    if (outfile != "")
    {
      const std::string basename = outfile.substr(0, outfile.find_last_of(".")) + "_pre";
      Core::IO::cout.setup(true, false, false, Core::IO::standard, comm, 0, 0,
          basename);  // necessary setup of Core::IO::cout
    }
    else
    {
      Core::IO::cout.setup(true, false, false, Core::IO::standard, comm, 0, 0,
          "xxx_pre");  // necessary setup of Core::IO::cout
    }


    /**************************************************************************
     * Start with the preprocessing
     **************************************************************************/
    if (exofile == "")
    {
      if (outfile != "")
      {
        // just validate a given 4C input file
        EXODUS::validate_input_file(comm, outfile);
        return 0;
      }
      else
      {
        My_CLP.printHelpMessage(argv[0], std::cout);
        FOUR_C_THROW("No Exodus II file was found");
      }
    }

    // create mesh object based on given exodus II file
    Core::IO::Exodus::Mesh mymesh(exofile);
    // print infos to std::cout
    mymesh.print(std::cout);

    /**************************************************************************
     * Read ControlFile for Boundary and Element descriptions
     **************************************************************************/

    // declare empty vectors for holding "boundary" conditions
    std::vector<EXODUS::ElemDef> eledefs;
    std::vector<EXODUS::CondDef> condefs;

    if (bcfile == "")
    {
      int error = create_default_bc_file(mymesh);
      if (error != 0) FOUR_C_THROW("Creation of default bc-file not successful.");
    }
    else
    {
      // read provided bc-file
      EXODUS::read_bc_file(bcfile, eledefs, condefs);

      int sum =
          mymesh.get_num_element_blocks() + mymesh.get_num_node_sets() + mymesh.get_num_side_sets();
      int test = eledefs.size() + condefs.size();
      if (test != sum)
        std::cout
            << "Your " << test << " definitions do not match the " << sum
            << " entities in your mesh!" << std::endl
            << "(This is OK, if more than one BC is applied to an entity, e.g in FSI simulations)"
            << std::endl;
    }

    /**************************************************************************
     * Finally, create and validate the 4C input file
     **************************************************************************/
    if ((headfile != "") && (bcfile != "") && (exofile != ""))
    {
      // set default dat-file name if needed
      if (outfile == "")
      {
        const std::string exofilebasename = exofile.substr(0, exofile.find_last_of("."));
        outfile = exofilebasename + ".4C.yaml";
      }

      // screen info
      std::cout << "creating and checking 4C input file       --> " << outfile << std::endl;
      auto timer = Teuchos::TimeMonitor::getNewTimer("pre-exodus timer");

      // check for positive Element-Center-Jacobians and otherwise rewind them
      {
        std::cout << "...Ensure positive element jacobians";
        timer->start();
        EXODUS::validate_mesh_element_jacobians(mymesh);
        timer->stop();
        std::cout << "        in...." << timer->totalElapsedTime(true) << " secs" << std::endl;
        timer->reset();
      }

      // in case of periodic boundary conditions :
      // ensure that the two coordinates of two matching nodes,
      // which should be the same are exactly the same
      // in order to keep the Krylov norm below 1e-6 :-)
      // only supported for angle 0.0
      {
        if (periodic_boundary_conditions_found(condefs))
        {
          std::cout << "...Ensure high quality p.b.c.";
          timer->start();
          correct_nodal_coordinates_for_periodic_boundary_conditions(mymesh, condefs);
          timer->stop();
          std::cout << "               in...." << timer->totalElapsedTime(true) << " secs"
                    << std::endl;
          timer->reset();
        }
      }

      // write the 4C input file
      {
        if (twodim) mymesh.set_nsd(2);
        std::cout << "...Writing file " << outfile;
        timer->start();
        EXODUS::write_dat_file(outfile, mymesh, headfile, eledefs, condefs);
        timer->stop();
        std::cout << "                         in...." << timer->totalElapsedTime(true) << " secs"
                  << std::endl;
        timer->reset();
      }

      // validate the generated 4C input file
      EXODUS::validate_input_file(comm, outfile);
    }
  }
  catch (Core::Exception& err)
  {
    char line[] = "=========================================================================\n";
    std::cout << "\n\n" << line << err.what_with_stacktrace() << "\n" << line << "\n" << std::endl;

#ifdef FOUR_C_ENABLE_CORE_DUMP
    abort();
#endif

    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
  }

  MPI_Finalize();
  return 0;
}
