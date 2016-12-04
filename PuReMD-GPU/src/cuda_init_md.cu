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

#include "cuda_init_md.h"

#include "allocate.h"
#include "box.h"
#include "forces.h"
#include "grid.h"
#include "index_utils.h"
#include "init_md.h"
#include "integrate.h"
#include "lookup.h"
#include "print_utils.h"
#include "reset_utils.h"
#include "system_props.h"
#include "traj.h"
#include "vector.h"

#include "cuda_allocate.h"
#include "cuda_utils.h"
#include "cuda_init.h"
#include "cuda_copy.h"
#include "cuda_box.h"
#include "cuda_forces.h"
#include "cuda_grid.h"
#include "cuda_integrate.h"
#include "cuda_lin_alg.h"
#include "cuda_list.h"
#include "cuda_lookup.h"
#include "cuda_neighbors.h"
#include "cuda_reduction.h"
#include "cuda_reset_utils.h"
#include "cuda_system_props.h"
#include "validation.h"


void Cuda_Init_System( reax_system *system, control_params *control, 
        simulation_data *data )
{
    int i;
    rvec dx;

    if( !control->restart )
    {
        Cuda_Reset_Atoms( system );
    }

    Cuda_Compute_Total_Mass( system, data );

    Cuda_Compute_Center_of_Mass( system, data, stderr );

    /* reposition atoms */
    // just fit the atoms to the periodic box
    if( control->reposition_atoms == 0 )
    {
        rvec_MakeZero( dx );
    }
    // put the center of mass to the center of the box
    else if( control->reposition_atoms == 1 )
    {
        rvec_Scale( dx, 0.5, system->box.box_norms );
        rvec_ScaledAdd( dx, -1., data->xcm );
    }
    // put the center of mass to the origin
    else if( control->reposition_atoms == 2 )
    {
        rvec_Scale( dx, -1., data->xcm );
    }
    else
    {
        fprintf( stderr, "UNKNOWN OPTION: reposition_atoms. Terminating...\n" );
        exit( UNKNOWN_OPTION );
    }

    k_compute_Inc_on_T3<<<BLOCKS_POW_2, BLOCK_SIZE>>>
        (system->d_atoms, system->N, system->d_box, dx[0], dx[1], dx[2]);
    cudaThreadSynchronize( );
    cudaCheckError( );

    //copy back the atoms from device to the host
    copy_host_device( system->atoms, system->d_atoms, REAX_ATOM_SIZE * system->N , 
            cudaMemcpyDeviceToHost, RES_SYSTEM_ATOMS );

    /* Initialize velocities so that desired init T can be attained */
    if( !control->restart || (control->restart && control->random_vel) )  {
        Generate_Initial_Velocities( system, control->T_init );
    }

    Setup_Grid( system );
}


void Cuda_Init_Simulation_Data( reax_system *system, control_params *control, 
        simulation_data *data, output_controls *out_control, 
        evolve_function *Evolve )
{

    Reset_Simulation_Data( data );

    if( !control->restart )  
        data->step = data->prev_steps = 0;

    switch( control->ensemble ) {
        case NVE:
            data->N_f = 3 * system->N;
            *Evolve = Cuda_Velocity_Verlet_NVE;
            break;


        case NVT:
            data->N_f = 3 * system->N + 1;
            //control->Tau_T = 100 * data->N_f * K_B * control->T_final;
            if( !control->restart || (control->restart && control->random_vel) ) {
                data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin - 
                        data->N_f * K_B * control->T );
                data->therm.v_xi = data->therm.G_xi * control->dt;
                data->therm.v_xi_old = 0;
                data->therm.xi = 0;
#if defined(DEBUG_FOCUS)
                fprintf( stderr, "init_md: G_xi=%f Tau_T=%f E_kin=%f N_f=%f v_xi=%f\n",
                        data->therm.G_xi, control->Tau_T, data->E_Kin, 
                        data->N_f, data->therm.v_xi );
#endif
            }

            *Evolve = Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein;
            break;


        case NPT: // Anisotropic NPT
            fprintf( stderr, "THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
            exit( UNKNOWN_OPTION );
            data->N_f = 3 * system->N + 9;
            if( !control->restart ) {
                data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin - 
                        data->N_f * K_B * control->T );
                data->therm.v_xi = data->therm.G_xi * control->dt;
                data->iso_bar.eps = 0.33333 * log(system->box.volume);
                //data->inv_W = 1. / (data->N_f*K_B*control->T*SQR(control->Tau_P));
                //Compute_Pressure( system, data, workspace );
            }
            *Evolve = Velocity_Verlet_Berendsen_Isotropic_NPT;
            break;


        case sNPT: // Semi-Isotropic NPT
            fprintf( stderr, "THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
            exit( UNKNOWN_OPTION );
            data->N_f = 3 * system->N + 4;
            *Evolve = Velocity_Verlet_Berendsen_SemiIsotropic_NPT;
            break;


        case iNPT: // Isotropic NPT
            fprintf( stderr, "THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
            exit( UNKNOWN_OPTION );
            data->N_f = 3 * system->N + 2;
            *Evolve = Velocity_Verlet_Berendsen_Isotropic_NPT;
            break;

        case bNVT: //berendensen NVT
            data->N_f = 3 * system->N + 1; 
            *Evolve = Cuda_Velocity_Verlet_Berendsen_NVT;
            break;

        default:
            break;
    }

    Cuda_Compute_Kinetic_Energy( system, data );

#ifdef __BUILD_DEBUG__
    real t_E_Kin = 0;
    t_E_Kin = data->E_Kin;
#endif

    copy_host_device( &data->E_Kin, &((simulation_data *)data->d_simulation_data)->E_Kin, 
            REAL_SIZE, cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );
    data->therm.T = (2. * data->E_Kin) / (data->N_f * K_B);
    if( fabs(data->therm.T) < ALMOST_ZERO ) // avoid T being an absolute zero! 
        data->therm.T = ALMOST_ZERO;

#ifdef __BUILD_DEBUG__
    if (check_zero( t_E_Kin, data->E_Kin)){
        fprintf( stderr, "SimulationData:E_Kin does not match between host and device (%f %f) \n", t_E_Kin, data->E_Kin );
        exit( 1 );
    }
    //validate_data ( system, data );
#endif

    /* init timing info for the host*/
    data->timing.start = Get_Time( );
    data->timing.total = data->timing.start;
    data->timing.nbrs = 0;
    data->timing.init_forces = 0;
    data->timing.bonded = 0;
    data->timing.nonb = 0;
    data->timing.QEq = 0;
    data->timing.matvecs = 0;

    /* init timing info for the device */
    d_timing.start = Get_Time( );
    d_timing.total = data->timing.start;
    d_timing.nbrs = 0;
    d_timing.init_forces = 0;
    d_timing.bonded = 0;
    d_timing.nonb = 0;
    d_timing.QEq = 0;
    d_timing.matvecs = 0;
}


int Estimate_Device_Matrix( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        list **lists, output_controls *out_control )
{
    int *indices, *Htop;
    list *far_nbrs = dev_lists + FAR_NBRS;
    int max_sparse_entries = 0;
    real t1, t2;

    indices = (int *) scratch;
    cuda_memset( indices, 0, INT_SIZE * system->N, RES_SCRATCH );

    t1 = Get_Time( );

    k_Estimate_Sparse_Matrix_Entries<<<BLOCKS, BLOCK_SIZE>>>
        ( system->d_atoms, (control_params *)control->d_control, 
          (simulation_data *)data->d_simulation_data, (simulation_box *)system->d_box, 
          *far_nbrs, system->N, indices );
    cudaThreadSynchronize( );
    cudaCheckError( );

    t2 = Get_Timing_Info( t1 );

    //fprintf (stderr, " Time to estimate sparse matrix entries --- > %f \n", t2 );

    Htop = (int *) malloc( INT_SIZE * (system->N + 1) );
    memset( Htop, 0, INT_SIZE * (system->N + 1) );
    copy_host_device( Htop, indices, system->N * INT_SIZE, cudaMemcpyDeviceToHost, __LINE__ );

    for (int i = 0; i < system->N; i++) 
    {
        if (max_sparse_entries < Htop[i]) {
            max_sparse_entries = Htop[i];
        }    
    }

#ifdef __DEBUG_CUDA__
    fprintf( stderr,
        " Max sparse entries for this run are ---> %d \n", max_sparse_entries );
#endif

    return max_sparse_entries * SAFE_ZONE;
    //return max_sparse_entries;
}


void Allocate_Device_Matrix (reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        list **lists, output_controls *out_control )
{

    //Allocate space for the sparse Matrix entries here. 
    system->max_sparse_matrix_entries = 
        Estimate_Device_Matrix( system, control, data, workspace, lists, out_control );
    dev_workspace->H.n = system->N ;
    dev_workspace->H.m = system->N * system->max_sparse_matrix_entries;
    Cuda_Init_Sparse_Matrix( &dev_workspace->H, system->max_sparse_matrix_entries * system->N, system->N );

#ifdef __CUDA_MEM__
    fprintf( stderr, "Device memory allocated: sparse matrix= %ld (MB)\n", 
            system->max_sparse_matrix_entries * system->N * sizeof(sparse_matrix_entry) / (1024*1024) );
#endif
}


void Cuda_Init_Lists( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        list **lists, output_controls *out_control )
{
    int i, num_nbrs, num_hbonds, num_bonds, num_3body, Htop;
    int *hb_top, *bond_top;

    real t_start, t_elapsed;

    grid *g = &( system->g );
    int *d_indices = (int *) scratch;
    int total = g->ncell[0] * g->ncell[1] * g->ncell[2];

    cuda_memset( d_indices, 0, INT_SIZE * system->N, RES_SCRATCH );

#ifdef __BUILD_DEBUG__
    for (int i = 0; i < g->max_nbrs; i ++)
    {
        if ((g->nbrs[i][0] >= g->ncell[0]) ||
                (g->nbrs[i][1] >= g->ncell[1]) ||
                (g->nbrs[i][2] >= g->ncell[2]) )
        {
            fprintf( stderr, " Grid Incorrectly built.... \n" );
            exit( 1 );
        }

    }
#endif

    dim3 blockspergrid( system->g.ncell[0], system->g.ncell[1], system->g.ncell[2] );
    dim3 threadsperblock( system->g.max_atoms );

#ifdef __BUILD_DEBUG__
    fprintf( stderr, "Blocks per grid (%d %d %d)\n", system->g.ncell[0], system->g.ncell[1], system->g.ncell[2] );
    fprintf( stderr, "Estimate Num  Neighbors with threads per block as %d \n", system->d_g.max_atoms );
    fprintf( stderr, "Max nbrs %d \n", system->d_g.max_nbrs );
#endif 

    //First Bin atoms and they sync the host and the device for the grid.
    //This will copy the atoms from host to device.
    Cuda_Bin_Atoms( system, workspace );
    Sync_Host_Device( &system->g, &system->d_g, cudaMemcpyHostToDevice );

    k_Estimate_NumNeighbors<<<blockspergrid, threadsperblock >>>
        (system->d_atoms, system->d_g, system->d_box, 
         (control_params *)control->d_control, d_indices);
    cudaThreadSynchronize( );
    cudaCheckError( );

    int *nbrs_indices = (int *) malloc( INT_SIZE * (system->N+1) );
    memset( nbrs_indices , 0, INT_SIZE * (system->N + 1) );

    nbrs_indices [0] = 0;
    copy_host_device( &nbrs_indices [1], d_indices, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__ ); 

    for (int i = 1; i <= system->N; i++)
    {
        nbrs_indices [i] += nbrs_indices [i-1];
    }

    num_nbrs = nbrs_indices [system->N] ;
    system->num_nbrs = num_nbrs;

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "Total neighbors %d \n", nbrs_indices[system->N]);
    fprintf (stderr, "Corrected Total neighbors %d \n", num_nbrs);
#endif

    list *far_nbrs = (dev_lists + FAR_NBRS);
    if( !Cuda_Make_List(system->N, num_nbrs, TYP_FAR_NEIGHBOR, far_nbrs) ) {
        fprintf(stderr, "Problem in initializing far nbrs list. Terminating!\n");
        exit( INIT_ERR );
    }

#ifdef __CUDA_MEM__
    fprintf( stderr, "Device memory allocated: far_nbrs = %ld (MB)\n", 
            num_nbrs * sizeof(far_neighbor_data) / (1024*1024) );
#endif

    copy_host_device( nbrs_indices, far_nbrs->index, INT_SIZE * system->N, cudaMemcpyHostToDevice, __LINE__  );
    copy_host_device( nbrs_indices, far_nbrs->end_index, INT_SIZE * system->N, cudaMemcpyHostToDevice, __LINE__  );
    Cuda_Generate_Neighbor_Lists( system, workspace, control, FALSE );

#ifdef __BUILD_DEBUG__

    int *end = (int *)malloc( sizeof (int) * system->N );
    int *start = (int *) malloc( sizeof (int) * system->N );

    copy_host_device( start, far_nbrs->index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, 0 );
    copy_host_device( end, far_nbrs->end_index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, 0 );

    far_neighbor_data *far_data = (far_neighbor_data *) 
        malloc( FAR_NEIGHBOR_SIZE * num_nbrs );
    copy_host_device( far_data, far_nbrs->select.far_nbr_list, 
            FAR_NEIGHBOR_SIZE * num_nbrs, cudaMemcpyDeviceToHost, 0 );

    compare_far_neighbors( nbrs_indices, start, end, far_data, *lists + FAR_NBRS, system->N );

    free( start );
    free( end );
#endif

    int *output, size;
    size = INT_SIZE * 2 * system->N + 2;
    output = (int *) malloc (size);
    Cuda_Estimate_Storage_Sizes( system, control, output );

    Htop = output[0];
    num_3body  = output[1];
    hb_top = &output[ 2 ]; 
    bond_top = &output[ 2 + system->N ];

#ifdef __DEBUG_CUDA__
    int max_hbonds = 0;
    int min_hbonds = 1000;
    int max_bonds = 0;
    int min_bonds = 1000;

    for (int i = 0; i < system->N; i++)
    {
        if ( max_hbonds < hb_top[i])
        {
            max_hbonds = hb_top[i];
        }
        if (min_hbonds > hb_top[i])
        {
            min_hbonds = hb_top[i];
        }

        if (max_bonds < bond_top [i])
        {
            max_bonds = bond_top[i];
        }
        if (min_bonds > bond_top[i])
        {
            min_bonds = bond_top[i];
        }
    }

    fprintf( stderr, "Max Hbonds %d min Hbonds %d \n", max_hbonds, min_hbonds );
    fprintf( stderr, "Max bonds %d min bonds %d \n", max_bonds, min_bonds );
    fprintf( stderr, "Device HTop --> %d and num_3body --> %d \n", Htop, num_3body );
#endif

    Allocate_Device_Matrix( system, control, data, workspace, lists, out_control );

    dev_workspace->num_H = 0;

    if( control->hb_cut > 0 )
    {

        int *hbond_index = (int *) malloc ( INT_SIZE * system->N );
        // init H indexes 
        num_hbonds = 0;
        for( i = 0; i < system->N; ++i )
        {
            if( system->reaxprm.sbp[ system->atoms[i].type ].p_hbond == 1 || 
                    system->reaxprm.sbp[ system->atoms[i].type ].p_hbond == 2  ) // H atom
            {
                //hbond_index[i] = workspace->num_H++;
                hbond_index[i] = num_hbonds ++;
            }
            else 
            {
                hbond_index[i] = -1;
            }
        }

        copy_host_device( hbond_index, dev_workspace->hbond_index, 
                system->N * INT_SIZE, cudaMemcpyHostToDevice, RES_STORAGE_HBOND_INDEX );
        dev_workspace->num_H = num_hbonds;

#ifdef __DEBUG_CUDA__
        fprintf( stderr, "Device num_H --> %d \n", dev_workspace->num_H );
#endif

        Cuda_Allocate_HBond_List( system->N, dev_workspace->num_H, dev_workspace->hbond_index, 
                hb_top, (dev_lists+HBONDS) );
        num_hbonds = hb_top[system->N-1];
        system->num_hbonds = num_hbonds;

#ifdef __CUDA_MEM__
        fprintf( stderr, "Device memory allocated: Hydrogen Bonds list: %ld (MB) \n", 
                sizeof (hbond_data) * num_hbonds / (1024*1024) );
#endif

#ifdef __DEBUG_CUDA__
        fprintf( stderr, "Device Total number of HBonds --> %d \n", num_hbonds );
#endif

        free( hbond_index );
    }

    // bonds list 
    Cuda_Allocate_Bond_List( system->N, bond_top, dev_lists+BONDS );
    num_bonds = bond_top[system->N-1];
    system->num_bonds = num_bonds;

#ifdef __CUDA_MEM__
    fprintf( stderr, "Device memory allocated: Bonds list: %ld (MB) \n", 
            sizeof (bond_data) * num_bonds / (1024*1024));
#endif

#ifdef __DEBUG_CUDA__
   fprintf( stderr, "Device Total Bonds --> %d \n", num_bonds );
#endif

    //    system->max_thb_intrs = num_3body;
    // 3bodies list 
    //if(!Cuda_Make_List(num_bonds, num_bonds * MAX_THREE_BODIES, TYP_THREE_BODY, dev_lists + THREE_BODIES)) {
    //  fprintf( stderr, "Problem in initializing angles list. Terminating!\n" );
    //  exit( INIT_ERR );
    //}

    //fprintf( stderr, "***memory allocated: three_body = %ldMB\n", 
    //   num_bonds * MAX_THREE_BODIES *sizeof(three_body_interaction_data) / (1024*1024) );
    //fprintf (stderr, "size of (three_body_interaction_data) : %d \n", sizeof (three_body_interaction_data));

    free( output );
    free( nbrs_indices );
}


void Cuda_Initialize( reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, list **lists, 
        output_controls *out_control, evolve_function *Evolve )
{
    compute_blocks( &BLOCKS, &BLOCK_SIZE, system->N );
    compute_nearest_pow_2( BLOCKS, &BLOCKS_POW_2 );

    //MATVEC_BLOCKS = system.N;
    //MATVEC_BLOCK_SIZE = 32;

    MATVEC_BLOCKS = (system->N * MATVEC_THREADS_PER_ROW / MATVEC_BLOCK_SIZE) + 
        ((system->N * MATVEC_THREADS_PER_ROW / MATVEC_BLOCK_SIZE) == 0 ? 0 : 1);

#ifdef __DEBUG_CUDA__
    fprintf( stderr, " MATVEC Blocks : %d, Block_Size : %d \n", MATVEC_BLOCKS, MATVEC_BLOCK_SIZE );
    fprintf( stderr, " Blocks : %d, Blocks_Pow_2 : %d, Block_Size : %d \n", BLOCKS, BLOCKS_POW_2, BLOCK_SIZE );
    fprintf( stderr, " Size of far neighbor data %d \n", sizeof (far_neighbor_data) );
    fprintf( stderr, " Size of reax_atom %d \n", sizeof (reax_atom) );
    fprintf( stderr, " size of sparse matrix entry %d \n", sizeof (sparse_matrix_entry) );
    fprintf( stderr, " TOTAL NUMBER OF ATOMS IN THE SYSTEM --> %d \n", system.N );
#endif

    Randomize( );

    Cuda_Init_Scratch( );

    //System
    Cuda_Init_System( system );
    Sync_Host_Device( system, cudaMemcpyHostToDevice );
    Cuda_Init_System( system, control, data );

    //Simulation Data
    copy_host_device( system->atoms, system->d_atoms, REAX_ATOM_SIZE * system->N , 
            cudaMemcpyHostToDevice, RES_SYSTEM_ATOMS );
    Cuda_Init_Simulation_Data( data );
    //Sync_Host_Device (data, (simulation_data *)data->d_simulation_data, cudaMemcpyHostToDevice);
    Cuda_Init_Simulation_Data( system, control, data, out_control, Evolve );
    Sync_Host_Device( data, (simulation_data *)data->d_simulation_data, cudaMemcpyHostToDevice );

    //static storage
    Cuda_Init_Workspace_System( system, dev_workspace );
    Cuda_Init_Workspace( system, control, dev_workspace );
    Cuda_Init_Workspace_Device( workspace );

    //control
    Cuda_Init_Control( control );

    //Grid
    Cuda_Init_Grid( &system->g, &system->d_g );

    //lists
    Cuda_Init_Lists( system, control, data, workspace, lists, out_control );

    Init_Out_Controls( system, control, workspace, out_control );

    if( control->tabulate )
    {
        real start, end;
        start = Get_Time( );
        Make_LR_Lookup_Table( system, control );
        copy_LR_table_to_device( system, control );
        end = Get_Timing_Info( start );

#ifdef __DEBUG_CUDA__
        fprintf( stderr, "Done copying the LR table to the device ---> %f \n", end );
#endif
    }
}
