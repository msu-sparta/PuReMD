/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of 
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#include "cuda_forces.h"

#include "box.h"
#include "forces.h"
#include "index_utils.h"
#include "list.h"
#include "print_utils.h"
#include "system_props.h"
#include "vector.h"

#include "cuda_utils.h"
#include "cuda_init.h"
#include "cuda_bond_orders.h"
#include "cuda_single_body_interactions.h"
#include "cuda_two_body_interactions.h"
#include "cuda_three_body_interactions.h"
#include "cuda_four_body_interactions.h"
#include "cuda_list.h"
#include "cuda_QEq.h"
#include "cuda_reduction.h"
#include "cuda_system_props.h"
#include "validation.h"

#include "cudaProfiler.h"


void Cuda_Compute_Bonded_Forces( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace,
        list **lists, output_controls *out_control )
{
    real t_start, t_elapsed;
    real *spad = (real *)scratch;
    rvec *rvec_spad;

    //Compute the bonded for interaction here. 
    //Step 1.
#ifdef __DEBUG_CUDA__
    t_start = Get_Time( );
    fprintf (stderr, " Begin Bonded Forces ... %d x %d\n", BLOCKS, BLOCK_SIZE);
#endif

    Cuda_Calculate_Bond_Orders_Init<<< BLOCKS, BLOCK_SIZE >>>
        (  system->d_atoms, system->reaxprm.d_gp, system->reaxprm.d_sbp,
           *dev_workspace, system->reaxprm.num_atom_types, system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );

    Cuda_Calculate_Bond_Orders<<< BLOCKS, BLOCK_SIZE >>>
        ( system->d_atoms, system->reaxprm.d_gp, system->reaxprm.d_sbp, 
          system->reaxprm.d_tbp, *dev_workspace, 
          *(dev_lists + BONDS), *(dev_lists + DDELTA), *(dev_lists + DBO), 
          system->reaxprm.num_atom_types, system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );

    Cuda_Update_Uncorrected_BO<<<BLOCKS, BLOCK_SIZE>>>
        (*dev_workspace, *(dev_lists + BONDS), system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );

    Cuda_Update_Workspace_After_Bond_Orders<<<BLOCKS, BLOCK_SIZE>>>
        (system->d_atoms, system->reaxprm.d_gp, system->reaxprm.d_sbp, 
         *dev_workspace, system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );

#ifdef __DEBUG_CUDA__
    t_elapsed = Get_Timing_Info( t_start );
    fprintf( stderr, "Bond Orders... return value --> %d --- Timing %lf \n", cudaGetLastError (), t_elapsed );
    fprintf( stderr, "Cuda_Calculate_Bond_Orders Done... \n" );
#endif

    //Step 2.
#ifdef __DEBUG_CUDA__
    t_start = Get_Time( );
#endif
    //cuda_memset (spad, 0, system->N * ( 2 * REAL_SIZE + system->N * REAL_SIZE + 16 * REAL_SIZE), RES_SCRATCH );
    cuda_memset (spad, 0, system->N * ( 2 * REAL_SIZE ) , RES_SCRATCH );

    Cuda_Bond_Energy <<< BLOCKS, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>>
        ( system->d_atoms, system->reaxprm.d_gp, system->reaxprm.d_sbp, system->reaxprm.d_tbp,
          (simulation_data *)data->d_simulation_data, *dev_workspace, *(dev_lists + BONDS), 
          system->N, system->reaxprm.num_atom_types, spad );
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Reduction for E_BE
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad, spad + system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> 
        //(spad + system->N, spad + system->N + 16, 16);
        (spad + system->N, &((simulation_data *)data->d_simulation_data)->E_BE, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

#ifdef __DEBUG_CUDA__
    t_elapsed = Get_Timing_Info( t_start );
    fprintf (stderr, "Cuda_Bond_Energy ... return value --> %d --- Timing %lf \n", cudaGetLastError (), t_elapsed );
    fprintf (stderr, "Cuda_Bond_Energy Done... \n");
#endif

    //Step 3.
#ifdef __DEBUG_CUDA__
    t_start = Get_Time( );
#endif
    cuda_memset (spad, 0, ( 6 * REAL_SIZE * system->N ), RES_SCRATCH );

    test_LonePair_OverUnder_Coordination_Energy_LP <<<BLOCKS, BLOCK_SIZE>>>( system->d_atoms, system->reaxprm.d_gp, 
            system->reaxprm.d_sbp, system->reaxprm.d_tbp, 
            *dev_workspace, (simulation_data *)data->d_simulation_data,
            *(dev_lists + BONDS), system->N, system->reaxprm.num_atom_types, 
            spad, spad + 2 * system->N, spad + 4*system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    test_LonePair_OverUnder_Coordination_Energy <<<BLOCKS, BLOCK_SIZE>>>( system->d_atoms, system->reaxprm.d_gp, 
            system->reaxprm.d_sbp, system->reaxprm.d_tbp, 
            *dev_workspace, (simulation_data *)data->d_simulation_data,
            *(dev_lists + BONDS), system->N, system->reaxprm.num_atom_types, 
            spad, spad + 2 * system->N, spad + 4*system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    test_LonePair_Postprocess        <<<BLOCKS, BLOCK_SIZE, 0>>>( system->d_atoms, system->reaxprm.d_gp, 
            system->reaxprm.d_sbp, system->reaxprm.d_tbp, 
            *dev_workspace, (simulation_data *)data->d_simulation_data,
            *(dev_lists + BONDS), system->N, system->reaxprm.num_atom_types);
    cudaThreadSynchronize ();
    cudaCheckError ();


    //Reduction for E_Lp
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad, spad + system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
        (spad + system->N, &((simulation_data *)data->d_simulation_data)->E_Lp, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Reduction for E_Ov
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad + 2*system->N, spad + 3*system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
        (spad + 3*system->N, &((simulation_data *)data->d_simulation_data)->E_Ov, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Reduction for E_Un
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad + 4*system->N, spad + 5*system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
        (spad + 5*system->N, &((simulation_data *)data->d_simulation_data)->E_Un, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

#ifdef __DEBUG_CUDA__
    t_elapsed = Get_Timing_Info( t_start );
    fprintf (stderr, "test_LonePair_postprocess ... return value --> %d --- Timing %lf \n", cudaGetLastError (), t_elapsed );
    fprintf (stderr, "test_LonePair_postprocess Done... \n");
#endif

    //Step 4.
#ifdef __DEBUG_CUDA__
    t_start = Get_Time( );
#endif

    cuda_memset(spad, 0, (dev_lists + BONDS)->num_intrs * sizeof (int), RES_SCRATCH);
    k_Three_Body_Estimate<<<BLOCKS, BLOCK_SIZE>>>
        (system->d_atoms, 
         (control_params *)control->d_control, 
         *(dev_lists + BONDS),
         system->N, (int *)spad);
    cudaThreadSynchronize ();
    cudaCheckError ();

#ifdef __DEBUG_CUDA__
    t_elapsed = Get_Timing_Info( t_start );
    fprintf (stderr, "Three_Body_Estimate... return value --> %d --- Timing %lf \n", cudaGetLastError (), t_elapsed );
#endif

    int *thbody = (int *) malloc (sizeof (int) * (dev_lists + BONDS)->num_intrs);
    memset (thbody, 0, sizeof (int) * (dev_lists + BONDS)->num_intrs);
    copy_host_device (thbody, spad, (dev_lists + BONDS)->num_intrs * sizeof (int), cudaMemcpyDeviceToHost, RES_SCRATCH);

    int total_3body = thbody [0] * SAFE_ZONE;
    for (int x = 1; x < (dev_lists + BONDS)->num_intrs; x++) {
        total_3body += thbody [x]*SAFE_ZONE;
        thbody [x] += thbody [x-1];
    }
    system->num_thbodies = thbody [(dev_lists+BONDS)->num_intrs-1];

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Total Three body estimate is %d (bonds: %d) \n", total_3body, (dev_lists+BONDS)->num_intrs);
#endif

    if (!system->init_thblist) 
    {
        system->init_thblist = TRUE;
        if(!Cuda_Make_List( (dev_lists+BONDS)->num_intrs, total_3body, TYP_THREE_BODY, dev_lists + THREE_BODIES )) {
            fprintf( stderr, "Problem in initializing three-body list. Terminating!\n" );
            exit( INIT_ERR );
        }
#ifdef __CUDA_MEM__
        fprintf (stderr, "Device memory allocated: three body list = %d MB\n", 
                sizeof (three_body_interaction_data) * total_3body / (1024*1024));
#endif
    } else {
        if ((dev_workspace->realloc.bonds > 0) || (system->num_thbodies > (dev_lists+THREE_BODIES)->num_intrs )) {
            int size = MAX (dev_workspace->realloc.num_bonds, (dev_lists+BONDS)->num_intrs);

            /*Delete Three-body list*/
            Cuda_Delete_List( dev_lists + THREE_BODIES );

#ifdef __CUDA_MEM__
            fprintf (stderr, "Reallocating Three-body list: step: %d n - %d num_intrs - %d used: %d \n", 
                    data->step, dev_workspace->realloc.num_bonds, total_3body, system->num_thbodies);
#endif
            /*Recreate Three-body list */
            if(!Cuda_Make_List( size, total_3body, TYP_THREE_BODY, dev_lists + THREE_BODIES )) {
                fprintf( stderr, "Problem in initializing three-body list. Terminating!\n" );
                exit( INIT_ERR );
            }
        }
    }

    //copy the indexes into the thb list;
    copy_host_device (thbody, ((dev_lists + THREE_BODIES)->index + 1), sizeof (int) * ((dev_lists+BONDS)->num_intrs - 1), 
            cudaMemcpyHostToDevice, LIST_INDEX);
    copy_host_device (thbody, ((dev_lists + THREE_BODIES)->end_index + 1), sizeof (int) * ((dev_lists+BONDS)->num_intrs - 1), 
            cudaMemcpyHostToDevice, LIST_END_INDEX);

    free (thbody );

#ifdef __DEBUG_CUDA__
    t_start = Get_Time( );
#endif

    cuda_memset (spad, 0, ( 6 * REAL_SIZE * system->N + RVEC_SIZE * system->N * 2), RES_SCRATCH );

    k_Three_Body_Interactions <<< BLOCKS, BLOCK_SIZE >>>
        ( system->d_atoms,
          system->reaxprm.d_sbp, system->reaxprm.d_thbp, system->reaxprm.d_gp, 
          (control_params *)control->d_control,
          (simulation_data *)data->d_simulation_data,
          *dev_workspace, 
          *(dev_lists + BONDS), *(dev_lists + THREE_BODIES),
          system->N, system->reaxprm.num_atom_types, 
          spad, spad + 2*system->N, spad + 4*system->N, (rvec *)(spad + 6*system->N));
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Not necessary to validate three-body list anymore, 
    // Estimate is already done at the beginning which makes sure that 
    // we have sufficient size for this list
    //Cuda_Threebody_List( system, workspace, dev_lists + THREE_BODIES, data->step );

    //Reduction for E_Ang
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad, spad + system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
        (spad + system->N, &((simulation_data *)data->d_simulation_data)->E_Ang, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Reduction for E_Pen
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad + 2*system->N, spad + 3*system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
        (spad + 3*system->N, &((simulation_data *)data->d_simulation_data)->E_Pen, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Reduction for E_Coa
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad + 4*system->N, spad + 5*system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction<<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
        (spad + 5*system->N, &((simulation_data *)data->d_simulation_data)->E_Coa, BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );

    //Reduction for ext_pres
    rvec_spad = (rvec *) (spad + 6*system->N);
    Cuda_reduction_rvec<<<BLOCKS_POW_2, BLOCK_SIZE, RVEC_SIZE * BLOCK_SIZE >>> 
        (rvec_spad, rvec_spad + system->N,  system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );

    Cuda_reduction_rvec<<<1, BLOCKS_POW_2, RVEC_SIZE * BLOCKS_POW_2 >>> 
        (rvec_spad + system->N, &((simulation_data *)data->d_simulation_data)->ext_press, BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );

    real t_1, t_2;
    t_1 = Get_Time( );
    //Sum up the f vector for each atom and collect the CdDelta from all the bonds
    k_Three_Body_Interactions_results <<< BLOCKS, BLOCK_SIZE >>>
        (system->d_atoms, (control_params *)control->d_control,
            *dev_workspace, *(dev_lists + BONDS), system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );
    t_2 = Get_Timing_Info( t_1 );

#ifdef __DEBUG_CUDA__
    t_elapsed = Get_Timing_Info( t_start );
    fprintf( stderr, "Three_Body_Interactions post process Timing %lf \n", t_2 );
    fprintf( stderr, "Three_Body_Interactions ...  Timing %lf \n", t_elapsed );
    fprintf( stderr, "Three_Body_Interactions Done... \n" );
#endif

    //Step 5.
#ifdef __DEBUG_CUDA__
    t_start = Get_Time( );
#endif

    cuda_memset( spad, 0, ( 4 * REAL_SIZE * system->N + RVEC_SIZE * system->N * 2), RES_SCRATCH );
    //k_Four_Body_Interactions<<< system->N, 32, 32*( 2*REAL_SIZE + RVEC_SIZE)>>>
    k_Four_Body_Interactions<<< BLOCKS, BLOCK_SIZE >>>
        ( system->d_atoms, system->reaxprm.d_gp, system->reaxprm.d_fbp,
          (control_params *)control->d_control, *(dev_lists + BONDS), *(dev_lists + THREE_BODIES),
          (simulation_box *)system->d_box, (simulation_data *)data->d_simulation_data,
          *dev_workspace, system->N, system->reaxprm.num_atom_types, 
          spad, spad + 2*system->N, (rvec *) (spad + 4*system->N) );
    cudaThreadSynchronize( );
    cudaCheckError( );

    //Reduction for E_Tor
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad, spad + system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
        (spad + system->N, &((simulation_data *)data->d_simulation_data)->E_Tor, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Reduction for E_Con
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (spad + 2*system->N, spad + 3*system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
        (spad + 3*system->N, &((simulation_data *)data->d_simulation_data)->E_Con, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Reduction for ext_pres
    rvec_spad = (rvec *) (spad + 4*system->N);
    Cuda_reduction_rvec <<<BLOCKS_POW_2, BLOCK_SIZE, RVEC_SIZE * BLOCK_SIZE >>> 
        (rvec_spad, rvec_spad + system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction_rvec <<<1, BLOCKS_POW_2, RVEC_SIZE * BLOCKS_POW_2 >>> 
        (rvec_spad + system->N, &((simulation_data *)data->d_simulation_data)->ext_press, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //Post process here
    k_Four_Body_Postprocess<<< BLOCKS, BLOCK_SIZE >>>
        ( system->d_atoms, *dev_workspace, *(dev_lists + BONDS),
            system->N );
    cudaThreadSynchronize( );
    cudaCheckError( );

#ifdef __DEBUG_CUDA__
    t_elapsed = Get_Timing_Info( t_start );
    fprintf (stderr, "Four_Body_post process return value --> %d --- Four body Timing %lf \n", cudaGetLastError (), t_elapsed );
    fprintf (stderr, " Four_Body_ Done... \n");
#endif

    //Step 6.
    if (control->hb_cut > 0) {
#ifdef __DEBUG_CUDA__
        t_start = Get_Time( );
#endif
        cuda_memset (spad, 0, ( 2 * REAL_SIZE * system->N + RVEC_SIZE * system->N * 2 ), RES_SCRATCH );

        /*
           k_Hydrogen_Bonds <<< BLOCKS, BLOCK_SIZE, BLOCK_SIZE *( REAL_SIZE + RVEC_SIZE) >>>
           (  system->d_atoms, 
           system->reaxprm.d_sbp,
           system->reaxprm.d_hbp,
           (control_params *)control->d_control,
           (simulation_data *)data->d_simulation_data,
         *dev_workspace, 
         *(dev_lists + BONDS), *(dev_lists + HBONDS),
         system->N, system->reaxprm.num_atom_types, 
         spad, (rvec *) (spad + 2*system->N), NULL);
         cudaThreadSynchronize ();
         cudaCheckError ();
         */

#ifdef __DEBUG_CUDA__
        real test1,test2;
        test1 = Get_Time ();
#endif

        int hbs = (system->N * HBONDS_THREADS_PER_ATOM/ HBONDS_BLOCK_SIZE) + 
            (((system->N * HBONDS_THREADS_PER_ATOM) % HBONDS_BLOCK_SIZE) == 0 ? 0 : 1);
        k_Hydrogen_Bonds_HB <<< hbs, HBONDS_BLOCK_SIZE, HBONDS_BLOCK_SIZE * ( 2 * REAL_SIZE + 2 * RVEC_SIZE )  >>>
            (  system->d_atoms, 
               system->reaxprm.d_sbp,
               system->reaxprm.d_hbp,
               (control_params *)control->d_control,
               (simulation_data *)data->d_simulation_data,
               *dev_workspace, 
               *(dev_lists + BONDS), *(dev_lists + HBONDS),
               system->N, system->reaxprm.num_atom_types, 
               spad, (rvec *) (spad + 2*system->N), NULL);
        cudaThreadSynchronize ();
        cudaCheckError ();

#ifdef __DEBUG_CUDA__
        test2 = Get_Timing_Info (test1);
        fprintf (stderr, "Timing for the hb and forces ---> %f \n", test2);
#endif

        //Reduction for E_HB
        Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
            (spad, spad + system->N,  system->N);
        cudaThreadSynchronize ();
        cudaCheckError ();

        Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>> 
            (spad + system->N, &((simulation_data *)data->d_simulation_data)->E_HB, BLOCKS_POW_2);
        cudaThreadSynchronize ();
        cudaCheckError ();


        //Reduction for ext_pres
        rvec_spad = (rvec *) (spad + 2*system->N);
        Cuda_reduction_rvec <<<BLOCKS_POW_2, BLOCK_SIZE, RVEC_SIZE * BLOCK_SIZE >>> 
            (rvec_spad, rvec_spad + system->N,  system->N);
        cudaThreadSynchronize ();
        cudaCheckError ();

        Cuda_reduction_rvec <<<1, BLOCKS_POW_2, RVEC_SIZE * BLOCKS_POW_2 >>> 
            (rvec_spad + system->N, &((simulation_data *)data->d_simulation_data)->ext_press, BLOCKS_POW_2);
        cudaThreadSynchronize ();
        cudaCheckError ();

        //Post process here
#ifdef __DEBUG_CUDA__
        real t_1, t_2;
        t_1 = Get_Time ();
#endif

        k_Hydrogen_Bonds_Postprocess <<< BLOCKS, BLOCK_SIZE, BLOCK_SIZE * RVEC_SIZE >>>
            (     system->d_atoms, 
                system->reaxprm.d_sbp, 
                *dev_workspace, 
                *(dev_lists + BONDS),
                *(dev_lists + HBONDS), 
                *(dev_lists + FAR_NBRS),
                system->N, 
                spad); //this is for the fix to use the shared memory
        cudaThreadSynchronize ();
        cudaCheckError ();

#ifdef __DEBUG_CUDA__
        t_2 = Get_Timing_Info ( t_1 );
        fprintf (stderr, " Hydrogen Bonds post process -----%f \n", t_2);
        t_1 = Get_Time ();
#endif

        //k_Hydrogen_Bonds_Far_Nbrs <<< system->N, 32, 32 * RVEC_SIZE>>>
        k_Hydrogen_Bonds_HNbrs <<< system->N, 32, 32 * RVEC_SIZE>>>
            (     system->d_atoms, 
                system->reaxprm.d_sbp, 
                *dev_workspace, 
                *(dev_lists + BONDS),
                *(dev_lists + HBONDS), 
                *(dev_lists + FAR_NBRS),
                system->N );
        cudaThreadSynchronize ();
        cudaCheckError ();
        t_2 = Get_Timing_Info ( t_1 );

#ifdef __DEBUG_CUDA__
        fprintf (stderr, " Hydrogen Bonds post process -----%f \n", t_2);
        t_elapsed = Get_Timing_Info( t_start );
        fprintf (stderr, "Hydrogen bonds post process return value --> %d --- HydrogenBonds Timing %lf \n", cudaGetLastError (), t_elapsed );
        fprintf (stderr, "Hydrogen_Bond Done... \n");
#endif
    }
    return; 
}


void Cuda_Compute_NonBonded_Forces( reax_system *system, control_params *control, 
        simulation_data *data,static_storage *workspace,
        list** lists, output_controls *out_control )
{
    real t_start, t_elapsed;
    real t1 = 0, t2 = 0;
    real *spad = (real *) scratch;
    rvec *rvec_spad;
    int cblks;

    t_start = Get_Time( );
    Cuda_QEq( system, control, data, workspace, lists[FAR_NBRS], out_control );
    t_elapsed = Get_Timing_Info( t_start );
    d_timing.QEq += t_elapsed;

#ifdef __DEBUG_CUDA__
    fprintf (stderr, " Cuda_QEq done with timing %lf \n", t_elapsed );
#endif

    cuda_memset (spad, 0, system->N * ( 4 * REAL_SIZE + 2 * RVEC_SIZE), RES_SCRATCH );

    t_start = Get_Time ();
    if ( control->tabulate == 0)
    {
        cblks = (system->N * VDW_THREADS_PER_ATOM / VDW_BLOCK_SIZE) + 
            ((system->N * VDW_THREADS_PER_ATOM/VDW_BLOCK_SIZE) == 0 ? 0 : 1);
        Cuda_vdW_Coulomb_Energy <<< cblks, VDW_BLOCK_SIZE, VDW_BLOCK_SIZE * ( 2*REAL_SIZE + RVEC_SIZE) >>>
            ( system->d_atoms,   
              system->reaxprm.d_tbp,
              system->reaxprm.d_gp, 
              (control_params *)control->d_control, 
              (simulation_data *)data->d_simulation_data,  
              *(dev_lists + FAR_NBRS), 
              spad , spad + 2 * system->N, (rvec *) (spad + system->N * 4), 
              system->reaxprm.num_atom_types,
              system->N ) ;
        cudaThreadSynchronize ();
        cudaCheckError ();
    }
    else
    {
        cblks = (system->N * VDW_THREADS_PER_ATOM / VDW_BLOCK_SIZE) + 
            ((system->N * VDW_THREADS_PER_ATOM/VDW_BLOCK_SIZE) == 0 ? 0 : 1);
        Cuda_Tabulated_vdW_Coulomb_Energy <<< cblks, VDW_BLOCK_SIZE, VDW_BLOCK_SIZE* (2*REAL_SIZE + RVEC_SIZE)>>>
            (   (reax_atom *)system->d_atoms, 
                (control_params *)control->d_control,
                (simulation_data *)data->d_simulation_data, 
                *(dev_lists + FAR_NBRS), 
                spad , spad + 2 * system->N, (rvec *) (spad + system->N * 4), 
                d_LR,
                system->reaxprm.num_atom_types,
                out_control->energy_update_freq,
                system->N ) ;

        cudaThreadSynchronize ();
        cudaCheckError ();
    }

    t_elapsed = Get_Timing_Info (t_start );

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Cuda_Tabulated_vdW_Coulomb_Energy done... %lf \n", (t_elapsed - t2));
    fprintf (stderr, "Cuda_Tabulated_vdW_Coulomb_Energy done... %lf \n", (t_elapsed));
#endif

    //Reduction on E_vdW
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>> 
        (spad, spad + system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> 
        (spad + system->N, &((simulation_data *)data->d_simulation_data)->E_vdW, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //reduction on E_Ele
    Cuda_reduction <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>> 
        (spad + 2*system->N, spad + 3*system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> 
        (spad + 3*system->N, &((simulation_data *)data->d_simulation_data)->E_Ele, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();
    rvec_spad = (rvec *) (spad + 4*system->N);

    //reduction on ext_press
    Cuda_reduction_rvec <<<BLOCKS_POW_2, BLOCK_SIZE, RVEC_SIZE * BLOCK_SIZE>>> 
        (rvec_spad, rvec_spad + system->N,  system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction_rvec <<<1, BLOCKS_POW_2, RVEC_SIZE * BLOCKS_POW_2>>> 
        (rvec_spad + system->N, &((simulation_data *)data->d_simulation_data)->ext_press, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();
}


void Cuda_Validate_Lists( reax_system *system, static_storage *workspace, list **lists, int step, int n,
        int num_bonds, int num_hbonds )
{
    int i, flag;
    list *bonds, *hbonds, *thblist;
    int *bonds_start, *bonds_end;
    int *hbonds_start, *hbonds_end;
    int *mat_start, *mat_end;
    int max_sparse_entries = 0;

    bonds = *lists + BONDS;
    hbonds = *lists + HBONDS;

    bonds_start = (int *) calloc (bonds->n, INT_SIZE);
    bonds_end = (int *) calloc (bonds->n, INT_SIZE);

    hbonds_start = (int *) calloc (hbonds->n, INT_SIZE );
    hbonds_end = (int *) calloc (hbonds->n, INT_SIZE );

    mat_start = (int *) calloc (workspace->H.n, INT_SIZE );
    mat_end = (int *) calloc (workspace->H.n, INT_SIZE );

    copy_host_device (bonds_start, bonds->index, bonds->n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );
    copy_host_device (bonds_end, bonds->end_index, bonds->n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );

    copy_host_device (hbonds_start, hbonds->index, hbonds->n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );
    copy_host_device (hbonds_end, hbonds->end_index, hbonds->n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );

    copy_host_device (mat_start, workspace->H.start, workspace->H.n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );
    copy_host_device (mat_end, workspace->H.end, workspace->H.n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );

    /* Sparse Matrix entries */

#ifdef __CUDA_TEST__
    /*
       workspace->realloc.Htop = 0;
       for (i = 0; i < workspace->H.n-1; i++) {
       if (workspace->realloc.Htop <= (mat_end[i] - mat_start[i])){
       workspace->realloc.Htop = mat_end[i] - mat_start[i];
       }
       }
     */
#endif

    flag = -1;
    workspace->realloc.Htop = 0;
    for ( i = 0; i < n-1; i ++){

        if( (mat_end[i] - mat_start[i]) > 
                (system->max_sparse_matrix_entries * DANGER_ZONE )) {
            //fprintf (stderr, "step %d, Reached the water mark for sparse matrix for index: %d (%d %d) \n", 
            //                                step, i, mat_start[i], mat_end[i]);
            if (workspace->realloc.Htop <= (mat_end[i] - mat_start[i]))
                workspace->realloc.Htop = (mat_end[i] - mat_start[i]) ;
        }

        if ( (mat_end[i] > mat_start[i+1]) ){
            fprintf( stderr, "step%d-matcheck failed: i=%d end(i)=%d start(i+1)=%d\n",
                    step, flag, mat_end[i], mat_start[i+1]);
            exit(INSUFFICIENT_SPACE);
        }
    }

    if( (mat_end[i] - mat_start[i]) > system->max_sparse_matrix_entries * DANGER_ZONE ) {
        if (workspace->realloc.Htop <= (mat_end[i] - mat_start[i]))
            workspace->realloc.Htop = (mat_end[i] - mat_start[i]) ;
        //fprintf (stderr, "step %d, Reached the water mark for sparse matrix for index %d (%d %d)  -- %d \n", 
        //                                step, i, mat_start[i], mat_end[i], 
        //                                (int) (system->max_sparse_matrix_entries * DANGER_ZONE));

        if( mat_end[i] > system->N * system->max_sparse_matrix_entries ) {
            fprintf( stderr, "step%d-matchk failed: i=%d end(i)=%d mat_end=%d\n",
                    step, flag, mat_end[i], system->N * system->max_sparse_matrix_entries);
            exit(INSUFFICIENT_SPACE);
        }
    }

    /* bond list */
#ifdef __CUDA_TEST__
    //workspace->realloc.bonds = 1;
#endif
    flag = -1;
    workspace->realloc.num_bonds = 0;
    for( i = 0; i < n-1; ++i ) {
        workspace->realloc.num_bonds += MAX((bonds_end [i] - bonds_start[i]) * 2, MIN_BONDS );
        if( bonds_end[i] >= bonds_start[i+1]-2 ) {
            workspace->realloc.bonds = 1;
            //fprintf (stderr, "step: %d, reached the water mark for bonds for atom: %d (%d %d) \n", 
            //                        step, i, bonds_start [i], bonds_end[i]);
            if( bonds_end[i] > bonds_start[i+1] )
                flag = i;
        }
    }

    if( flag > -1 ) {
        fprintf( stderr, "step%d-bondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                step, flag, bonds_end[flag], bonds_start[flag+1] );
        exit(INSUFFICIENT_SPACE);
    }    

    workspace->realloc.num_bonds += MAX((bonds_end [i] - bonds_start[i]) * 2, MIN_BONDS );
    if( bonds_end[i] >= bonds->num_intrs-2 ) {
        workspace->realloc.bonds = 1;
        //fprintf (stderr, "step: %d, reached the water mark for bonds for atom: %d (%d %d) \n", 
        //                        step, i, bonds_start [i], bonds_end[i]);

        if( bonds_end[i] > bonds->num_intrs ) {
            fprintf( stderr, "step%d-bondchk failed: i=%d end(i)=%d bond_end=%d\n",
                    step, flag, bonds_end[i], bonds->num_intrs );
            exit(INSUFFICIENT_SPACE);
        }
    }

    //fprintf (stderr, "step:%d Total bonds: %d \n", step, workspace->realloc.num_bonds);

    /* hbonds list */
    if( workspace->num_H > 0 ) {
#ifdef __CUDA_TEST__
        //workspace->realloc.hbonds = 1;
#endif
        flag = -1;
        workspace->realloc.num_hbonds = 0;
        for( i = 0; i < workspace->num_H-1; ++i ) {
            workspace->realloc.num_hbonds += MAX( (hbonds_end[i] - hbonds_start[i]) * SAFE_HBONDS, MIN_HBONDS );

            if( (hbonds_end[i] - hbonds_start[i]) >= 
                    (hbonds_start[i+1] - hbonds_start[i]) * DANGER_ZONE ) {
                workspace->realloc.hbonds = 1;
                //fprintf (stderr, "step: %d, reached the water mark for hbonds for atom: %d (%d %d) \n", 
                //                        step, i, hbonds_start [i], hbonds_end[i]);
                if( hbonds_end[i] > hbonds_start[i+1] )
                    flag = i;
            }
        }

        if( flag > -1 ) {
            fprintf( stderr, "step%d-hbondchk failed: i=%d start(i)=%d,end(i)=%d str(i+1)=%d\n",
                    step, flag, hbonds_start[(flag)],hbonds_end[(flag)], hbonds_start[(flag+1)] );
            exit(INSUFFICIENT_SPACE);
        }

        workspace->realloc.num_hbonds += MAX( (hbonds_end[i] - hbonds_start[i]) * SAFE_HBONDS, MIN_HBONDS );
        if( (hbonds_end[i] - hbonds_start[i]) >= 
                (hbonds->num_intrs - hbonds_start[i]) * DANGER_ZONE ) {
            workspace->realloc.hbonds = 1;
            //fprintf (stderr, "step: %d, reached the water mark for hbonds for atom: %d (%d %d) \n", 
            //                        step, i, hbonds_start [i], hbonds_end[i]);

            if( hbonds_end[i] > hbonds->num_intrs ) {
                fprintf( stderr, "step%d-hbondchk failed: i=%d end(i)=%d hbondend=%d\n",
                        step, flag, hbonds_end[i], hbonds->num_intrs );
                exit(INSUFFICIENT_SPACE);
            }
        }
    }

    //fprintf (stderr, "step:%d Total Hbonds: %d \n", step, workspace->realloc.num_hbonds);

    free (bonds_start);
    free (bonds_end );

    free (hbonds_start );
    free (hbonds_end  );

    free (mat_start );
    free (mat_end );
}


void Cuda_Threebody_List( reax_system *system, static_storage *workspace, list *thblist, int step )
{
    int *thb_start, *thb_end;
    int i, flag;

    thb_start = (int *) calloc (thblist->n, INT_SIZE);
    thb_end = (int *) calloc (thblist->n, INT_SIZE );

    copy_host_device (thb_start, thblist->index, thblist->n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );
    copy_host_device (thb_end, thblist->end_index, thblist->n * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );

    /*three_body list*/
    flag = -1;
    workspace->realloc.num_3body = 0;
    for( i = 0; i < thblist->n-1; ++i ){
        if( (thb_end[i] - thb_start[i]) >= (thb_start[i+1] - thb_start[i])*DANGER_ZONE ) {
            workspace->realloc.thbody = 1;
            if( thb_end[i] > thb_end[i+1] || thb_end[i] > thblist->num_intrs ) {
                flag = i;
                break;
            }
        }
    }

    if( flag > -1 ) {
        //fprintf( stderr, "step%d-thbchk failed: i=%d end(i)=%d str(i+1)=%d\n",
        //   step, flag, thb_end[flag], thb_start[flag+1] );
        fprintf( stderr, "step%d-thbchk failed: i=%d start(i)=%d end(i)=%d thb_end=%d\n",
                step, flag-1, thb_start[flag-1], thb_end[flag-1], thblist->num_intrs );
        fprintf( stderr, "step%d-thbchk failed: i=%d start(i)=%d end(i)=%d thb_end=%d\n",
                step, flag, thb_start[flag], thb_end[flag], thblist->num_intrs );
        exit(INSUFFICIENT_SPACE);
    }    

    if( (thb_end[i]-thb_start[i]) >= (thblist->num_intrs - thb_start[i])*DANGER_ZONE ) {
        workspace->realloc.thbody = 1;

        if( thb_end[i] > thblist->num_intrs ) {
            fprintf( stderr, "step%d-thbchk failed: i=%d start(i)=%d end(i)=%d thb_end=%d\n",
                    step, i-1, thb_start[i-1], thb_end[i-1], thblist->num_intrs );
            fprintf( stderr, "step%d-thbchk failed: i=%d start(i)=%d end(i)=%d thb_end=%d\n",
                    step, i, thb_start[i], thb_end[i], thblist->num_intrs );
            exit(INSUFFICIENT_SPACE);
        }
    }

    free (thb_start);
    free (thb_end);
}


GLOBAL void k_Estimate_Sparse_Matrix_Entries ( reax_atom *atoms, control_params *control, 
        simulation_data *data, simulation_box *box, list far_nbrs, int N, int *indices ) {

    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop;
    int flag;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    int temp;

    Htop = 0;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    atom_i = &(atoms[i]);
    type_i  = atom_i->type;
    start_i = Start_Index(i, &far_nbrs);
    end_i   = End_Index(i, &far_nbrs);
    indices[i] = Htop;

    for( pj = start_i; pj < end_i; ++pj ) {
        nbr_pj = &( far_nbrs.select.far_nbr_list[pj] );
        j = nbr_pj->nbr;
        atom_j = &(atoms[j]);

        //CHANGE ORIGINAL
        //if (i < j) continue;
        //CHANGE ORIGINAL

        flag = 0;
        if((data->step-data->prev_steps) % control->reneighbor == 0) { 
            if( nbr_pj->d <= control->r_cut)
                flag = 1;
            else flag = 0;
        }
        else if((nbr_pj->d=Sq_Distance_on_T3(atom_i->x,atom_j->x,box,nbr_pj->dvec)) <=     
                SQR(control->r_cut)){
            nbr_pj->d = sqrt(nbr_pj->d);
            flag = 1;
        }

        if( flag ){    
            ++Htop;
        }
    }

    ++Htop;

    // mark the end of j list
    indices[i] = Htop;
}


GLOBAL void k_Init_Forces( reax_atom *atoms,         global_parameters g_params, control_params *control, 
        single_body_parameters *sbp, two_body_parameters *tbp, 
        simulation_data *data, simulation_box *box,    static_storage workspace,
        list far_nbrs,             list bonds,                list hbonds, 
        int N,                         int max_sparse_entries, int num_atom_types ) 
{

    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop, btop_i, btop_j, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top, jhb_top;
    int flag;
    real r_ij, r2, self_coef;
    real dr3gamij_1, dr3gamij_3, Tap;
    //real val, dif, base;
    real C12, C34, C56;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
    real BO, BO_s, BO_pi, BO_pi2;
    real p_boc1, p_boc2;   
    sparse_matrix *H;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    //LR_lookup_table *t;
    reax_atom *atom_i, *atom_j;
    bond_data *ibond, *jbond;
    bond_order_data *bo_ij, *bo_ji;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    H = &( workspace.H );
    //CHANGE ORIGINAL
    //Htop = 0;
    Htop = i * max_sparse_entries;
    //CHANGE ORIGINAL
    num_bonds = 0;
    num_hbonds = 0;
    btop_i = btop_j = 0;
    p_boc1 = g_params.l[0];
    p_boc2 = g_params.l[1];

    //for( i = 0; i < system->N; ++i ) 
    atom_i = &(atoms[i]);
    type_i  = atom_i->type;
    start_i = Start_Index(i, &far_nbrs);
    end_i   = End_Index(i, &far_nbrs);

    H->start[i] = Htop;
    H->end[i] = Htop;

    btop_i = End_Index( i, &bonds );
    sbp_i = &(sbp[type_i]);
    ihb = ihb_top = -1;

    ihb = sbp_i->p_hbond;

    if( control->hb_cut > 0 && (ihb==1 || ihb == 2))
        ihb_top = End_Index( workspace.hbond_index[i], &hbonds );

    for( pj = start_i; pj < end_i; ++pj ) {
        nbr_pj = &( far_nbrs.select.far_nbr_list[pj] );
        j = nbr_pj->nbr;
        atom_j = &(atoms[j]);

        flag = 0;
        if((data->step-data->prev_steps) % control->reneighbor == 0) { 
            if( nbr_pj->d <= control->r_cut)
                flag = 1;
            else flag = 0;
        }
        else if (i > j) {
            if((nbr_pj->d=Sq_Distance_on_T3(atom_i->x,atom_j->x,box,nbr_pj->dvec))<=SQR(control->r_cut)){
                nbr_pj->d = sqrt(nbr_pj->d);
                flag = 1;
            }
        } else if (i < j) {
            if((nbr_pj->d=Sq_Distance_on_T3(atom_j->x,atom_i->x,box,nbr_pj->dvec))<=SQR(control->r_cut)){
                nbr_pj->d = sqrt(nbr_pj->d);
                flag = 1;
            }
        }

        if( flag ){    

            type_j = atoms[j].type;
            r_ij = nbr_pj->d;
            sbp_j = &(sbp[type_j]);
            twbp = &(tbp[ index_tbp (type_i,type_j, num_atom_types) ]);
            self_coef = (i == j) ? 0.5 : 1.0;

            /* H matrix entry */

            //CHANGE ORIGINAL
            //if (i > j) {
            Tap = control->Tap7 * r_ij + control->Tap6;
            Tap = Tap * r_ij + control->Tap5;
            Tap = Tap * r_ij + control->Tap4;
            Tap = Tap * r_ij + control->Tap3;
            Tap = Tap * r_ij + control->Tap2;
            Tap = Tap * r_ij + control->Tap1;
            Tap = Tap * r_ij + control->Tap0;          

            dr3gamij_1 = ( r_ij * r_ij * r_ij + twbp->gamma );
            dr3gamij_3 = POW( dr3gamij_1 , 0.33333333333333 );

            H->entries[Htop].j = j;
            H->entries[Htop].val = self_coef * Tap * EV_to_KCALpMOL / dr3gamij_3;

            ++Htop;
            //}
            //CHANGE ORIGINAL

            /* hydrogen bond lists */ 
            if( control->hb_cut > 0 && (ihb==1 || ihb == 2) && 
                    nbr_pj->d <= control->hb_cut ) {
                // fprintf( stderr, "%d %d\n", atom1, atom2 );
                jhb = sbp_j->p_hbond;

                if (ihb == 1 && jhb == 2) {
                    if (i > j) {
                        hbonds.select.hbond_list[ihb_top].nbr = j;
                        hbonds.select.hbond_list[ihb_top].scl = 1;
                        hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;

                        //Auxilary data structures
                        rvec_MakeZero (hbonds.select.hbond_list[ihb_top].h_f);
                        hbonds.select.hbond_list[ihb_top].sym_index= -1;
                        ++ihb_top;
                        ++num_hbonds;
                    } else {
                        hbonds.select.hbond_list[ihb_top].nbr = j;
                        hbonds.select.hbond_list[ihb_top].scl = -1;
                        hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;

                        //Auxilary data structures
                        rvec_MakeZero (hbonds.select.hbond_list[ihb_top].h_f);
                        hbonds.select.hbond_list[ihb_top].sym_index= -1;
                        ++ihb_top;
                        ++num_hbonds;
                    }
                } else if (ihb == 2 && jhb == 1) { 
                    hbonds.select.hbond_list[ihb_top].nbr = j; 
                    hbonds.select.hbond_list[ihb_top].scl = 1; 
                    hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;
                    //TODO
                    rvec_MakeZero (hbonds.select.hbond_list[ihb_top].h_f);
                    hbonds.select.hbond_list[ihb_top].sym_index= -1;
                    ++ihb_top;
                    ++num_hbonds;
                } 
            }

            /* uncorrected bond orders */
            if( far_nbrs.select.far_nbr_list[pj].d <= control->nbr_cut ) {
                r2 = SQR(r_ij);

                if( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0) {
                    C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                    BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                }
                else BO_s = C12 = 0.0;

                if( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0) {
                    C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                    BO_pi = EXP( C34 );
                }
                else BO_pi = C34 = 0.0;

                if( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0) {
                    C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );    
                    BO_pi2= EXP( C56 );
                }
                else BO_pi2 = C56 = 0.0;

                /* Initially BO values are the uncorrected ones, page 1 */
                BO = BO_s + BO_pi + BO_pi2;


                if( BO >= control->bo_cut ) {
                    //CHANGE ORIGINAL
                    num_bonds += 1;
                    //CHANGE ORIGINAL

                    /****** bonds i-j and j-i ******/

                    /* Bond Order page2-3, derivative of total bond order prime */
                    Cln_BOp_s = twbp->p_bo2 * C12 / r2;
                    Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
                    Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;


                    if (i > j) 
                    {
                        ibond = &( bonds.select.bond_list[btop_i] );
                        ibond->nbr = j;
                        ibond->d = r_ij;
                        rvec_Copy( ibond->dvec, nbr_pj->dvec );
                        ivec_Copy( ibond->rel_box, nbr_pj->rel_box );

                        //ibond->dbond_index = btop_i;
                        //ibond->sym_index = btop_j;
                        ++btop_i;

                        bo_ij = &( ibond->bo_data );
                        bo_ij->BO = BO;
                        bo_ij->BO_s = BO_s;
                        bo_ij->BO_pi = BO_pi;
                        bo_ij->BO_pi2 = BO_pi2;

                        //Auxilary data structures
                        ibond->scratch = 0;
                        ibond->CdDelta_ij = 0;
                        rvec_MakeZero (ibond->f);

                        ibond->l = -1;
                        ibond->CdDelta_jk = 0;
                        ibond->Cdbo_kl = 0;
                        rvec_MakeZero (ibond->i_f);
                        rvec_MakeZero (ibond->k_f);

                        rvec_MakeZero (ibond->h_f);

                        rvec_MakeZero (ibond->t_f);

                        // Only dln_BOp_xx wrt. dr_i is stored here, note that 
                        //     dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 
                        rvec_Scale(bo_ij->dln_BOp_s,-bo_ij->BO_s*Cln_BOp_s,ibond->dvec);
                        rvec_Scale(bo_ij->dln_BOp_pi,-bo_ij->BO_pi*Cln_BOp_pi,ibond->dvec);
                        rvec_Scale(bo_ij->dln_BOp_pi2,
                                -bo_ij->BO_pi2*Cln_BOp_pi2,ibond->dvec);

                        // Only dBOp wrt. dr_i is stored here, note that 
                        //    dBOp/dr_i = -dBOp/dr_j and all others are 0 
                        rvec_Scale( bo_ij->dBOp, 
                                -(bo_ij->BO_s * Cln_BOp_s + 
                                    bo_ij->BO_pi * Cln_BOp_pi + 
                                    bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );

                        rvec_Add( workspace.dDeltap_self[i], bo_ij->dBOp );

                        bo_ij->BO_s -= control->bo_cut;
                        bo_ij->BO -= control->bo_cut;
                        workspace.total_bond_order[i] += bo_ij->BO; //currently total_BOp

                        bo_ij->Cdbo = bo_ij->Cdbopi = bo_ij->Cdbopi2 = 0.0;


                    } else if ( i < j )
                    {
                        rvec dln_BOp_s, dln_BOp_pi, dln_BOp_pi2;
                        rvec dBOp;

                        btop_j = btop_i;

                        jbond = &(bonds.select.bond_list[btop_j]);
                        jbond->nbr = j;
                        jbond->d = r_ij;
                        rvec_Scale( jbond->dvec, -1, nbr_pj->dvec );
                        ivec_Scale( jbond->rel_box, -1, nbr_pj->rel_box );

                        btop_i ++;
                        //jbond->dbond_index = btop_i;
                        //jbond->sym_index = btop_i;

                        bo_ji = &( jbond->bo_data );
                        bo_ji->BO = BO;
                        bo_ji->BO_s = BO_s;
                        bo_ji->BO_pi = BO_pi;
                        bo_ji->BO_pi2 = BO_pi2;

                        //Auxilary data structures
                        jbond->scratch = 0;
                        jbond->CdDelta_ij = 0;
                        rvec_MakeZero (jbond->f);

                        jbond->l = -1;
                        jbond->CdDelta_jk = 0;
                        jbond->Cdbo_kl = 0;
                        rvec_MakeZero (jbond->i_f);
                        rvec_MakeZero (jbond->k_f);

                        rvec_MakeZero (jbond->h_f);

                        rvec_MakeZero (jbond->t_f);

                        // Only dln_BOp_xx wrt. dr_i is stored here, note that 
                        // dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0
                        rvec_Scale(dln_BOp_s,-BO_s*Cln_BOp_s,nbr_pj->dvec);
                        rvec_Scale(dln_BOp_pi,-BO_pi*Cln_BOp_pi,nbr_pj->dvec);
                        rvec_Scale(dln_BOp_pi2,
                                -BO_pi2*Cln_BOp_pi2,nbr_pj->dvec);

                        rvec_Scale(bo_ji->dln_BOp_s, -1., dln_BOp_s);
                        rvec_Scale(bo_ji->dln_BOp_pi, -1., dln_BOp_pi );
                        rvec_Scale(bo_ji->dln_BOp_pi2, -1., dln_BOp_pi2 );

                        // Only dBOp wrt. dr_i is stored here, note that 
                        //    dBOp/dr_i = -dBOp/dr_j and all others are 0 
                        rvec_Scale( dBOp, 
                                -(BO_s * Cln_BOp_s + 
                                    BO_pi * Cln_BOp_pi + 
                                    BO_pi2 * Cln_BOp_pi2), nbr_pj->dvec );
                        rvec_Scale( bo_ji->dBOp, -1., dBOp );

                        rvec_Add( workspace.dDeltap_self[i] , bo_ji->dBOp );

                        bo_ji->BO_s -= control->bo_cut;
                        bo_ji->BO -= control->bo_cut;
                        workspace.total_bond_order[i] += bo_ji->BO; //currently total_BOp

                        bo_ji->Cdbo = bo_ji->Cdbopi = bo_ji->Cdbopi2 = 0.0;

                    }
                } 
            }
        }
    }

    H->entries[Htop].j = i;
    H->entries[Htop].val = sbp[type_i].eta;
    ++Htop;

    H->end[i] = Htop;

    Set_End_Index( i, btop_i, &bonds );
    if( ihb == 1 || ihb == 2)
        Set_End_Index( workspace.hbond_index[i], ihb_top, &hbonds );

    //fprintf( stderr, "%d bonds start: %d, end: %d\n", 
    //     i, Start_Index( i, bonds ), End_Index( i, bonds ) );
    //}

    // mark the end of j list
    //H->start[i] = Htop; 
    /* validate lists - decide if reallocation is required! */
    //Validate_Lists( workspace, lists, 
    //      data->step, system->N, H->m, Htop, num_bonds, num_hbonds ); 
}


GLOBAL void k_Init_Forces_Tab ( reax_atom *atoms, global_parameters g_params, control_params *control, 
        single_body_parameters *sbp, two_body_parameters *tbp, 
        simulation_data *data, simulation_box *box, static_storage workspace,
        list far_nbrs, list bonds, list hbonds, 
        int N, int max_sparse_entries, int num_atom_types, 
        LR_lookup_table *d_LR) 
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop, btop_i, btop_j, num_bonds, num_hbonds;
    int tmin, tmax, r;
    int ihb, jhb, ihb_top, jhb_top;
    int flag;
    real r_ij, r2, self_coef;
    real val, dif, base;
    real C12, C34, C56;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
    real BO, BO_s, BO_pi, BO_pi2;
    real p_boc1, p_boc2;   
    sparse_matrix *H;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    LR_lookup_table *t;
    reax_atom *atom_i, *atom_j;
    bond_data *ibond, *jbond;
    bond_order_data *bo_ij, *bo_ji;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    H = &(workspace.H);
    //CHANGE ORIGINAL
    Htop = i * max_sparse_entries;
    //CHANGE ORIGINAL
    num_bonds = 0;
    num_hbonds = 0;
    btop_i = btop_j = 0;
    p_boc1 = g_params.l[0];
    p_boc2 = g_params.l[1];

    //for( i = 0; i < system->N; ++i )
    atom_i = &(atoms[i]);
    type_i  = atom_i->type;
    start_i = Start_Index(i, &far_nbrs);
    end_i   = End_Index(i, &far_nbrs);
    H->start[i] = Htop;
    H->end[i] = Htop;
    btop_i = End_Index( i, &bonds );
    sbp_i = &(sbp[type_i]);
    ihb = ihb_top = -1;

    ihb = sbp_i->p_hbond;

    if( control->hb_cut > 0 && (ihb==1 || ihb == 2))
        ihb_top = End_Index( workspace.hbond_index[i], &hbonds );

    for( pj = start_i; pj < end_i; ++pj ) {
        nbr_pj = &( far_nbrs.select.far_nbr_list[pj] );
        j = nbr_pj->nbr;
        atom_j = &(atoms[j]);

        flag = 0;
        if((data->step-data->prev_steps) % control->reneighbor == 0) { 
            if(nbr_pj->d <= control->r_cut)
                flag = 1;
            else flag = 0;
        }
        else if (i > j) {
            if((nbr_pj->d=Sq_Distance_on_T3(atom_i->x,atom_j->x,box,nbr_pj->dvec))<=SQR(control->r_cut)){
                nbr_pj->d = sqrt(nbr_pj->d);
                flag = 1;
            }
        }
        else if ( i < j) {
            if((nbr_pj->d=Sq_Distance_on_T3(atom_j->x,atom_i->x,box,nbr_pj->dvec))<=SQR(control->r_cut)){
                nbr_pj->d = sqrt(nbr_pj->d);
                flag = 1;
            }
        }

        if( flag ){    
            type_j = atoms[j].type;
            r_ij = nbr_pj->d;
            sbp_j = &(sbp[type_j]);
            twbp = &(tbp[ index_tbp (type_i,type_j,num_atom_types) ]);
            self_coef = (i == j) ? 0.5 : 1.0;
            tmin  = MIN( type_i, type_j );
            tmax  = MAX( type_i, type_j );
            t = &( d_LR[ index_lr (tmin, tmax, num_atom_types) ]);      

            /* cubic spline interpolation */
            //CHANGE ORIGINAL
            //if (i > j) {
            r = (int)(r_ij * t->inv_dx);
            if( r == 0 )  ++r;
            base = (real)(r+1) * t->dx;
            dif = r_ij - base;
            val = ((t->ele[r].d*dif + t->ele[r].c)*dif + t->ele[r].b)*dif + 
                t->ele[r].a;
            val *= EV_to_KCALpMOL / C_ele;

            H->entries[Htop].j = j;
            H->entries[Htop].val = self_coef * val;
            //H->j [Htop] = j;
            //H->val [Htop] = self_coef * val;
            ++Htop;
            //}
            //CHANGE ORIGINAL

            /* hydrogen bond lists */ 
            if( control->hb_cut > 0 && (ihb==1 || ihb==2) && 
                    nbr_pj->d <= control->hb_cut ) {
                // fprintf( stderr, "%d %d\n", atom1, atom2 );
                jhb = sbp_j->p_hbond;

                if ( ihb == 1 && jhb == 2 ) {
                    if (i > j) {
                        hbonds.select.hbond_list[ihb_top].nbr = j;
                        hbonds.select.hbond_list[ihb_top].scl = 1;
                        hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;

                        //Auxilary data structures
                        rvec_MakeZero (hbonds.select.hbond_list[ihb_top].h_f);
                        hbonds.select.hbond_list[ihb_top].sym_index= -1;
                        ++ihb_top;
                        ++num_hbonds;
                    } else {
                        hbonds.select.hbond_list[ihb_top].nbr = j;
                        hbonds.select.hbond_list[ihb_top].scl = -1;
                        hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;

                        //Auxilary data structures
                        rvec_MakeZero (hbonds.select.hbond_list[ihb_top].h_f);
                        hbonds.select.hbond_list[ihb_top].sym_index= -1;
                        ++ihb_top;
                        ++num_hbonds;
                    }
                } else if (ihb == 2 && jhb == 1) {
                    hbonds.select.hbond_list[ihb_top].nbr = j;
                    hbonds.select.hbond_list[ihb_top].scl = 1;
                    hbonds.select.hbond_list[ihb_top].ptr = nbr_pj;

                    //Auxilary data structures
                    rvec_MakeZero (hbonds.select.hbond_list[ihb_top].h_f);
                    hbonds.select.hbond_list[ihb_top].sym_index= -1;
                    ++ihb_top;
                    ++num_hbonds;
                }
            }

            /* uncorrected bond orders */
            if( far_nbrs.select.far_nbr_list[pj].d <= control->nbr_cut ) {
                r2 = SQR(r_ij);

                if( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0) {
                    C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                    BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                }
                else BO_s = C12 = 0.0;

                if( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0) {
                    C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                    BO_pi = EXP( C34 );
                }
                else BO_pi = C34 = 0.0;

                if( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0) {
                    C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );    
                    BO_pi2= EXP( C56 );
                }
                else BO_pi2 = C56 = 0.0;

                /* Initially BO values are the uncorrected ones, page 1 */
                BO = BO_s + BO_pi + BO_pi2;

                if( BO >= control->bo_cut ) {

                    //CHANGE ORIGINAL
                    num_bonds += 1;
                    //CHANGE ORIGINAL

                    /****** bonds i-j and j-i ******/
                    if ( i > j )
                    {
                        ibond = &( bonds.select.bond_list[btop_i] );
                        ibond->nbr = j;
                        ibond->d = r_ij;

                        rvec_Copy( ibond->dvec, nbr_pj->dvec );
                        ivec_Copy( ibond->rel_box, nbr_pj->rel_box );

                        //ibond->dbond_index = btop_i;
                        //ibond->sym_index = btop_j;

                        ++btop_i;

                        bo_ij = &( ibond->bo_data );
                        bo_ij->BO = BO;
                        bo_ij->BO_s = BO_s;
                        bo_ij->BO_pi = BO_pi;
                        bo_ij->BO_pi2 = BO_pi2;

                        //Auxilary data strucutres to resolve dependencies
                        ibond->scratch = 0;
                        ibond->CdDelta_ij = 0;
                        rvec_MakeZero (ibond->f);

                        ibond->l = -1;
                        ibond->CdDelta_jk = 0;
                        ibond->Cdbo_kl = 0;
                        rvec_MakeZero (ibond->i_f);
                        rvec_MakeZero (ibond->k_f);

                        rvec_MakeZero (ibond->h_f);

                        rvec_MakeZero (ibond->t_f);

                        /* Bond Order page2-3, derivative of total bond order prime */
                        Cln_BOp_s = twbp->p_bo2 * C12 / r2;
                        Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
                        Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

                        /* Only dln_BOp_xx wrt. dr_i is stored here, note that 
                           dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
                        rvec_Scale(bo_ij->dln_BOp_s,-bo_ij->BO_s*Cln_BOp_s,ibond->dvec);
                        rvec_Scale(bo_ij->dln_BOp_pi,-bo_ij->BO_pi*Cln_BOp_pi,ibond->dvec);
                        rvec_Scale(bo_ij->dln_BOp_pi2,
                                -bo_ij->BO_pi2*Cln_BOp_pi2,ibond->dvec);

                        /* Only dBOp wrt. dr_i is stored here, note that 
                           dBOp/dr_i = -dBOp/dr_j and all others are 0 */
                        rvec_Scale( bo_ij->dBOp, 
                                -(bo_ij->BO_s * Cln_BOp_s + 
                                    bo_ij->BO_pi * Cln_BOp_pi + 
                                    bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );

                        rvec_Add( workspace.dDeltap_self[i], bo_ij->dBOp );

                        bo_ij->BO_s -= control->bo_cut;
                        bo_ij->BO -= control->bo_cut;

                        workspace.total_bond_order[i] += bo_ij->BO; //currently total_BOp

                        bo_ij->Cdbo = bo_ij->Cdbopi = bo_ij->Cdbopi2 = 0.0;
                    } 
                    else {
                        rvec dln_BOp_s, dln_BOp_pi, dln_BOp_pi2;
                        rvec dBOp;

                        btop_j = btop_i;

                        jbond = &( bonds.select.bond_list[btop_j] );
                        jbond->nbr = j; 
                        jbond->d = r_ij;

                        rvec_Scale( jbond->dvec, -1, nbr_pj->dvec );
                        ivec_Scale( jbond->rel_box, -1, nbr_pj->rel_box );

                        //jbond->dbond_index = btop_i;
                        //jbond->sym_index = btop_i;

                        ++btop_i;

                        bo_ji = &( jbond->bo_data );

                        bo_ji->BO = BO;
                        bo_ji->BO_s = BO_s;
                        bo_ji->BO_pi = BO_pi;
                        bo_ji->BO_pi2 = BO_pi2;

                        // Auxilary data structures to resolve dependencies
                        jbond->scratch = 0;
                        jbond->CdDelta_ij = 0;
                        rvec_MakeZero (jbond->f);

                        jbond->l = -1;
                        jbond->CdDelta_jk = 0;
                        jbond->Cdbo_kl = 0;
                        rvec_MakeZero (jbond->i_f);
                        rvec_MakeZero (jbond->k_f);

                        rvec_MakeZero (jbond->h_f);

                        rvec_MakeZero (jbond->t_f);

                        // Bond Order page2-3, derivative of total bond order prime
                        Cln_BOp_s = twbp->p_bo2 * C12 / r2;
                        Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
                        Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

                        // Only dln_BOp_xx wrt. dr_i is stored here, note that 
                        //   dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 

                        rvec_Scale(dln_BOp_s,-BO_s*Cln_BOp_s,nbr_pj->dvec);
                        rvec_Scale(dln_BOp_pi,-BO_pi*Cln_BOp_pi,nbr_pj->dvec);
                        rvec_Scale(dln_BOp_pi2, -BO_pi2*Cln_BOp_pi2,nbr_pj->dvec);

                        rvec_Scale(bo_ji->dln_BOp_s, -1., dln_BOp_s);
                        rvec_Scale(bo_ji->dln_BOp_pi, -1., dln_BOp_pi );
                        rvec_Scale(bo_ji->dln_BOp_pi2, -1., dln_BOp_pi2 );

                        // Only dBOp wrt. dr_i is stored here, note that 
                        //   dBOp/dr_i = -dBOp/dr_j and all others are 0
                        //CHANGE ORIGINAL
                        rvec_Scale( dBOp, 
                                -(BO_s * Cln_BOp_s + 
                                    BO_pi * Cln_BOp_pi + 
                                    BO_pi2 * Cln_BOp_pi2), nbr_pj->dvec);
                        rvec_Scale( bo_ji->dBOp, -1., dBOp);
                        //CHANGE ORIGINAL

                        rvec_Add( workspace.dDeltap_self[i], bo_ji->dBOp );

                        bo_ji->BO_s -= control->bo_cut;
                        bo_ji->BO -= control->bo_cut;

                        workspace.total_bond_order[i] += bo_ji->BO; //currently total_BOp

                        bo_ji->Cdbo = bo_ji->Cdbopi = bo_ji->Cdbopi2 = 0.0;
                    }
                }
            }
        }
    }

    H->entries[Htop].j = i;
    H->entries[Htop].val = sbp[type_i].eta;

    //H->j [Htop] = i;
    //H->val [Htop] = sbp[type_i].eta;

    ++Htop;

    H->end[i] = Htop;
    Set_End_Index( i, btop_i, &bonds );
    if( ihb == 1  || ihb == 2)
        Set_End_Index( workspace.hbond_index[i], ihb_top, &hbonds );
}


GLOBAL void k_fix_sym_dbond_indices (list pbonds, int N)
{
    int i, nbr;
    bond_data *ibond, *jbond;
    int atom_j;

    list *bonds = &pbonds;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int j = Start_Index (i, bonds); j < End_Index (i, bonds); j++)
    {
        ibond = &( bonds->select.bond_list [j] );    
        nbr = ibond->nbr;

        for (int k = Start_Index (nbr, bonds); k < End_Index (nbr, bonds); k ++)
        {
            jbond = &( bonds->select.bond_list[ k ] );
            atom_j = jbond->nbr;

            if ( (atom_j == i) )
            {
                if (i > nbr) {
                    ibond->dbond_index = j; 
                    jbond->dbond_index = j;

                    ibond->sym_index = k;
                    jbond->sym_index = j;
                }
            }
        }
    }
}


GLOBAL void k_fix_sym_hbond_indices (static_storage p_workspace, list hbonds, int N)
{
    static_storage *workspace = &p_workspace;
    hbond_data *ihbond, *jhbond;
    int nbr;

    //int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 4;
    int i = (blockIdx.x);
    int start = Start_Index (workspace->hbond_index[i], &hbonds);
    int end = End_Index (workspace->hbond_index[i], &hbonds);
    //int j = start + threadIdx.x;
    //int j = start + (threadIdx.x % 16);

    //for (int j = Start_Index (workspace->hbond_index[i], &hbonds); 
    //        j < End_Index (workspace->hbond_index[i], &hbonds); j++)
    int j = start + threadIdx.x;
    while (j < end)
        //for (int j = start; j < end; j++)
    {
        ihbond = &( hbonds.select.hbond_list [j] );
        nbr = ihbond->nbr;

        int nbrstart = Start_Index (workspace->hbond_index[nbr], &hbonds);
        int nbrend = End_Index (workspace->hbond_index[nbr], &hbonds);

        for (int k = nbrstart; k < nbrend; k++)
            //k = nbrstart + threadIdx.x;
            //while (k < nbrend)
        {
            jhbond = &( hbonds.select.hbond_list [k] );

            if (jhbond->nbr == i){
                ihbond->sym_index = k;
                jhbond->sym_index = j;
                break;
            }

            //k += blockDim.x;
        }

        j += 32;
    }
}


GLOBAL void k_New_fix_sym_hbond_indices (static_storage p_workspace, list hbonds, int N )
{

    static_storage *workspace = &p_workspace;
    hbond_data *ihbond, *jhbond;

    int __THREADS_PER_ATOM__ = HBONDS_SYM_THREADS_PER_ATOM;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / __THREADS_PER_ATOM__;
    int lane_id = thread_id & (__THREADS_PER_ATOM__ - 1);
    int my_bucket = threadIdx.x / __THREADS_PER_ATOM__;

    if (warp_id >= N) return;

    int i = warp_id;
    int nbr;
    int k;
    int start = Start_Index (workspace->hbond_index[i], &hbonds);
    int end = End_Index (workspace->hbond_index[i], &hbonds);
    int j = start + lane_id;
    //for (int j = start; j < end; j++)
    while (j < end)
    {
        ihbond = &( hbonds.select.hbond_list [j] );
        nbr = ihbond->nbr;

        int nbrstart = Start_Index (workspace->hbond_index[nbr], &hbonds);
        int nbrend = End_Index (workspace->hbond_index[nbr], &hbonds);

        //k = nbrstart + lane_id;
        //if (lane_id == 0) found [my_bucket] = 0;
        //while (k < nbrend)
        for (k = nbrstart; k < nbrend; k++)
        {
            jhbond = &( hbonds.select.hbond_list [k] );

            if (jhbond->nbr == i){
                ihbond->sym_index = k;
                jhbond->sym_index = j;
                break;
            }
        }

        j += __THREADS_PER_ATOM__;
    }
}


GLOBAL void k_Estimate_Storage_Sizes(reax_atom *atoms, 
    int N, single_body_parameters *sbp,
    two_body_parameters *tbp,
    global_parameters gp, 
    control_params *control, 
    list far_nbrs,
    int num_atom_types, int *results)
{
    int *Htop = &results[0];
    int *num_3body  = &results[1];
    int *hb_top = &results [ 2 ];
    int *bond_top = &results [ 2 + N ];

    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    real r_ij, r2;
    real C12, C34, C56;
    real BO, BO_s, BO_pi, BO_pi2;
    real p_boc1, p_boc2; 
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    p_boc1 = gp.l[0];
    p_boc2 = gp.l[1];

    //for( i = 0; i < N; ++i ) {
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N ) return ;

    atom_i = &(atoms[i]);
    type_i  = atom_i->type;
    start_i = Start_Index(i, &far_nbrs);
    end_i   = End_Index(i, &far_nbrs);
    sbp_i = &(sbp[type_i]);
    ihb = sbp_i->p_hbond;

    for( pj = start_i; pj < end_i; ++pj ) {
        nbr_pj = &( far_nbrs.select.far_nbr_list[pj] );
        j = nbr_pj->nbr;
        atom_j = &( atoms[j] );
        type_j = atom_j->type;
        sbp_j = &( sbp[type_j] );
        twbp = &( tbp[ index_tbp (type_i,type_j,num_atom_types) ] );


        if( nbr_pj->d <= control->r_cut ) {
            //++(*Htop);
            atomicAdd(Htop, 1);

            /* hydrogen bond lists */ 
            //TODO - CHANGE ORIGINAL
            if( control->hb_cut > 0 && (ihb==1 || ihb==2) && 
                    nbr_pj->d <= control->hb_cut ) {
                jhb = sbp_j->p_hbond;
                if( ihb == 1 && jhb == 2 )
                    //++hb_top[i];
                    atomicAdd(&hb_top[i], 1);
                else if( ihb == 2 && jhb == 1 )
                    //++hb_top[j];
                    //atomicAdd(&hb_top[j], 1);
                    atomicAdd(&hb_top[i], 1);
            }
            //TODO -- CHANGE ORIGINAL

            //CHANGE ORIGINAL
            if (i < j) continue;
            //CHANGE ORIGINAL


            /* uncorrected bond orders */
            if( nbr_pj->d <= control->nbr_cut ) {
                r_ij = nbr_pj->d;
                r2 = SQR(r_ij);

                if( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0) {
                    C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                    BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                }
                else BO_s = C12 = 0.0;

                if( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0) {
                    C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                    BO_pi = EXP( C34 );
                }
                else BO_pi = C34 = 0.0;

                if( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0) {
                    C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );    
                    BO_pi2= EXP( C56 );
                }
                else BO_pi2 = C56 = 0.0;

                /* Initially BO values are the uncorrected ones, page 1 */
                BO = BO_s + BO_pi + BO_pi2;

                if( BO >= control->bo_cut ) {
                    //++bond_top[i];
                    //++bond_top[j];
                    atomicAdd(&bond_top[i], 1);
                    atomicAdd(&bond_top[j], 1);
                }
            }
        }
    }
    //}
}


void Cuda_Estimate_Storage_Sizes (reax_system *system, control_params *control, int *output)
{
    int *Htop, *num_3body, input_size;
    int *hb_top, *bond_top;
    int *input = (int *) scratch;
    int max_3body = 0;

    Htop = 0;
    num_3body = 0;
    input_size = INT_SIZE * (2 * system->N + 1 + 1);

    //cuda_malloc ((void **) &input, input_size, 1, __LINE__);
    cuda_memset (input, 0, input_size, RES_SCRATCH );

    k_Estimate_Storage_Sizes <<<BLOCKS_POW_2, BLOCK_SIZE>>>
        (system->d_atoms, system->N, system->reaxprm.d_sbp, system->reaxprm.d_tbp, 
         system->reaxprm.d_gp, (control_params *)control->d_control, *(dev_lists + FAR_NBRS), 
         system->reaxprm.num_atom_types, input);
    cudaThreadSynchronize();
    cudaCheckError();

    copy_host_device (output, input, input_size, cudaMemcpyDeviceToHost, __LINE__ );

    Htop = &output[0];
    num_3body  = &output[1];
    hb_top = &output[ 2 ];
    bond_top = &output[ 2 + system->N ];

    *Htop += system->N;
    *Htop *= SAFE_ZONE;

    for( int i = 0; i < system->N; ++i ) {
        hb_top[i] = MAX( hb_top[i] * SAFE_HBONDS, MIN_HBONDS );

        if (max_3body <= SQR (bond_top[i]))
            max_3body = SQR (bond_top[i]);

        *num_3body += SQR(bond_top[i]);
        bond_top[i] = MAX( bond_top[i] * 2, MIN_BONDS );
    }

    *num_3body = max_3body * SAFE_ZONE;
}


void Cuda_Compute_Forces( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        list** lists, output_controls *out_control )
{
    real t_start, t_elapsed;
    real t_1, t_2;
    int *indices;
    int *Htop;
    int max_sparse_entries = 0;
    list *far_nbrs = dev_lists + FAR_NBRS;
    int hblocks;

    t_start = Get_Time ();
    if ( !control->tabulate ) {
        k_Init_Forces <<<BLOCKS, BLOCK_SIZE>>>
            (system->d_atoms,         system->reaxprm.d_gp, (control_params *)control->d_control, 
             system->reaxprm.d_sbp, system->reaxprm.d_tbp, 
             (simulation_data *)data->d_simulation_data, (simulation_box *)system->d_box, *dev_workspace,
             *(dev_lists + FAR_NBRS), *(dev_lists + BONDS), *(dev_lists + HBONDS), 
             system->N, system->max_sparse_matrix_entries, system->reaxprm.num_atom_types ); 
        cudaThreadSynchronize ();
        cudaCheckError ();
    }
    else 
    {
        k_Init_Forces_Tab <<< BLOCKS, BLOCK_SIZE >>>
            ( system->d_atoms,         system->reaxprm.d_gp, (control_params *)control->d_control, 
              system->reaxprm.d_sbp, system->reaxprm.d_tbp, 
              (simulation_data *)data->d_simulation_data, (simulation_box *)system->d_box,  *dev_workspace,
              *(dev_lists + FAR_NBRS),     *(dev_lists + BONDS), *(dev_lists + HBONDS), 
              system->N, system->max_sparse_matrix_entries, system->reaxprm.num_atom_types, 
              d_LR );
        cudaThreadSynchronize ();
        cudaCheckError ();
    }

    /*This is for bonds processing to fix dbond and sym_indexes */
    t_1 = Get_Time ();
    k_fix_sym_dbond_indices <<<BLOCKS, BLOCK_SIZE>>> (*(dev_lists + BONDS), system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();
    t_2 = Get_Timing_Info ( t_1 );

    //FIX -1 HYDROGEN BOND fix for cases where there are no hbonds.
    if ((control->hb_cut > 0) && (dev_workspace->num_H > 0))
    {

        hblocks = (system->N * HBONDS_SYM_THREADS_PER_ATOM / HBONDS_SYM_BLOCK_SIZE) + 
            ((system->N * HBONDS_SYM_THREADS_PER_ATOM % HBONDS_SYM_BLOCK_SIZE) == 0 ? 0 : 1);
        t_1 = Get_Time ();
        /*
           int bs = system->N;
           int ss = 32;
           fix_sym_hbond_indices <<<bs, ss>>> (*dev_workspace, *(dev_lists + HBONDS), system->N);
         */
        k_New_fix_sym_hbond_indices <<<hblocks, HBONDS_SYM_BLOCK_SIZE>>> (*dev_workspace, *(dev_lists + HBONDS), system->N);
        cudaThreadSynchronize ();
        cudaCheckError ();
    }
    t_2 = Get_Timing_Info ( t_1 );

    t_elapsed = Get_Timing_Info (t_start);
    d_timing.init_forces+= t_elapsed;

    Cuda_Validate_Lists( system, dev_workspace, &dev_lists, data->step, system->N,
            system->num_bonds, system->num_hbonds );
#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Done with Cuda List Validation \n");
#endif

    //Bonded Force Calculations here.
    t_start = Get_Time ();
    Cuda_Compute_Bonded_Forces( system, control, data, workspace, lists, out_control );
    t_elapsed = Get_Timing_Info (t_start);
    d_timing.bonded += t_elapsed;

    //Compute the Non Bonded Forces here. 
    t_start = Get_Time ();
    Cuda_Compute_NonBonded_Forces( system, control, data, workspace, lists, out_control );
    t_elapsed = Get_Timing_Info (t_start);
    d_timing.nonb += t_elapsed;

    //Compute Total Forces here
    Cuda_Compute_Total_Force<<< BLOCKS, BLOCK_SIZE >>>
        (system->d_atoms, (simulation_data *)data->d_simulation_data, *dev_workspace, 
         *(dev_lists + BONDS), control->ensemble, system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_Compute_Total_Force_PostProcess<<< BLOCKS, BLOCK_SIZE >>>
        (system->d_atoms, (simulation_data *)data->d_simulation_data, *dev_workspace, 
         *(dev_lists + BONDS), control->ensemble, system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();
}


int validate_device (reax_system *system, simulation_data *data, static_storage *workspace, list **lists )
{
    int retval = FALSE;

#ifdef __BUILD_DEBUG__

    retval |= validate_neighbors (system, lists);
    retval |= validate_sym_dbond_indices (system, workspace, lists);
    retval |= validate_bonds (system, workspace, lists);
    retval |= validate_sparse_matrix (system, workspace);
    retval |= validate_three_bodies (system, workspace, lists );
    retval |= validate_hbonds (system, workspace, lists);
    retval |= validate_workspace (system, workspace, lists);
    retval |= validate_data (system, data);
    retval |= validate_atoms (system, lists);
    //analyze_hbonds (system, workspace, lists);

    if (!retval) {
        fprintf (stderr, "Results *DOES NOT* mattch between device and host \n");
    }
#endif

    return retval;
}
