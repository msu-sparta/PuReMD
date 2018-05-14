/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
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

#include "reax_types.h"

#include "allocate.h"
#include "analyze.h"
#include "comm_tools.h"
#include "control.h"
#include "ffield.h"
#include "forces.h"
#include "geo_tools.h"
#include "init_md.h"
#include "integrate.h"
#include "io_tools.h"
#include "neighbors.h"
#include "reset_tools.h"
#include "restart.h"
#include "system_props.h"
#include "tool_box.h"
#include "traj.h"
#include "vector.h"

#ifdef HAVE_CUDA
  #include "cuda/cuda_copy.h"
  #include "cuda/cuda_environment.h"
  #include "cuda/cuda_forces.h"
  #include "cuda/cuda_init_md.h"
  #include "cuda/cuda_neighbors.h"
  #include "cuda/cuda_post_evolve.h"
  #include "cuda/cuda_reset_tools.h"
  #include "cuda/cuda_system_props.h"
  #include "cuda/cuda_utils.h"
  #if defined(DEBUG)
    #include "cuda/cuda_validation.h"
  #endif
#endif


evolve_function Evolve;
evolve_function Cuda_Evolve;
LR_lookup_table *LR;
LR_lookup_table *d_LR;

/* CUDA SPECIFIC DECLARATIONS */
reax_list **dev_lists;
storage *dev_workspace;
void *scratch;
void *host_scratch;

int BLOCKS, BLOCKS_POW_2, BLOCK_SIZE;
int BLOCKS_N, BLOCKS_POW_2_N;
int MATVEC_BLOCKS;


void Read_Control_Files( char *geo_file, char *ffield_file, char *control_file,
        reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, output_controls *out_control, mpi_datatypes *mpi_data )
{
    Read_Force_Field( ffield_file, &(system->reax_param), system, control );

    Read_Control_File( control_file, control, out_control );

    if ( control->geo_format == CUSTOM )
    {
        Read_Geo( geo_file, system, control, data, workspace, mpi_data );
    }
    else if ( control->geo_format == PDB )
    {
        Read_PDB( geo_file, system, control, data, workspace, mpi_data );
    }
    else if ( control->geo_format == ASCII_RESTART )
    {
        Read_Restart( geo_file, system, control, data, workspace, mpi_data );
        control->restart = 1;
    }
    else if ( control->geo_format == BINARY_RESTART )
    {
        Read_Binary_Restart( geo_file, system, control, data, workspace, mpi_data );
        control->restart = 1;
    }
    else
    {
        fprintf( stderr, "unknown geo file format. terminating!\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_GEO );
    }
}


void Post_Evolve( reax_system* system, control_params* control,
        simulation_data* data, storage* workspace, reax_list** lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i;
    rvec diff, cross;

    /* remove translational and rotational velocity of the center of mass from system */
    if ( control->ensemble != NVE && control->remove_CoM_vel &&
            data->step % control->remove_CoM_vel == 0 )
    {
        /* compute velocity of the center of mass */
        Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );

        for ( i = 0; i < system->n; i++ )
        {
            /* remove translational term */
            rvec_ScaledAdd( system->my_atoms[i].v, -1.0, data->vcm );

            /* remove rotational term */
            rvec_ScaledSum( diff, 1.0, system->my_atoms[i].x, -1.0, data->xcm );
            rvec_Cross( cross, data->avcm, diff );
            rvec_ScaledAdd( system->my_atoms[i].v, -1.0, cross );
        }
    }

    /* compute kinetic energy of system */
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

    if ( (out_control->energy_update_freq > 0
                && data->step % out_control->energy_update_freq == 0)
            || (out_control->write_steps > 0
                && data->step % out_control->write_steps == 0) )
    {
        Compute_Total_Energy( system, data, MPI_COMM_WORLD );
    }
}


#ifdef HAVE_CUDA
int Cuda_Post_Evolve( reax_system* system, control_params* control,
        simulation_data* data, storage* workspace, reax_list** lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    /* remove trans & rot velocity of the center of mass from system */
    if ( control->ensemble != NVE && control->remove_CoM_vel &&
            data->step % control->remove_CoM_vel == 0 )
    {
        /* compute velocity of the center of mass */
        Cuda_Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );

        post_evolve_velocities( system, data );
    }

    /* compute kinetic energy of the system */
    Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

    return SUCCESS;
}
#endif


static void usage( char* argv[] )
{
    fprintf( stderr, "usage: ./%s geometry ffield control\n", argv[0] );
}


int main( int argc, char* argv[] )
{
    reax_system *system;
    control_params *control;
    simulation_data *data;
    storage *workspace;
    reax_list **lists;
    output_controls *out_control;
    mpi_datatypes *mpi_data;
    int i, ret, retries;
    real t_start = 0, t_elapsed;
#if defined(DEBUG)
    real t_begin, t_end;
#endif

    MPI_Init( &argc, &argv );

    if ( argc != 4 )
    {
        usage( argv );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }

#ifdef HAVE_CUDA
    system = smalloc( sizeof(reax_system), "main::system" );
    control = smalloc( sizeof(control_params), "main::control" );
    data = smalloc( sizeof(simulation_data), "main::data" );
    workspace = smalloc( sizeof(storage), "main::workspace" );
    lists = smalloc( sizeof(reax_list *) * LIST_N, "main::lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        lists[i] = smalloc( sizeof(reax_list), "main::lists[i]" );
        lists[i]->allocated = FALSE;
    }
    out_control = smalloc( sizeof(output_controls), "main::out_control" );
    mpi_data = smalloc( sizeof(mpi_datatypes), "main::mpi_data" );

    /* allocate auxiliary data structures (GPU) */
    dev_workspace = smalloc( sizeof(storage), "main::dev_workspace" );
    dev_lists = smalloc ( sizeof(reax_list *) * LIST_N, "main::dev_lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        dev_lists[i] = smalloc( sizeof(reax_list), "main::dev_lists[i]" );
        dev_lists[i]->allocated = FALSE;
    }

    /* setup MPI environment */
    MPI_Comm_size( MPI_COMM_WORLD, &(control->nprocs) );
    MPI_Comm_rank( MPI_COMM_WORLD, &(system->my_rank) );

    /* read system config files */
    Read_Control_Files( argv[1], argv[2], argv[3], system, control,
            data, workspace, out_control, mpi_data );

    /* setup the CUDA Device for this process */
    Setup_Cuda_Environment( system->my_rank, control->nprocs, control->gpus_per_node );

#if defined(DEBUG)
    print_device_mem_usage( );
#endif

    /* init blocks sizes */
    init_blocks( system );

    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }

    Cuda_Initialize( system, control, data, workspace, lists, out_control, mpi_data );

#if defined(__CUDA_DEBUG__)
    Pure_Initialize( system, control, data, workspace, lists, out_control, mpi_data );
#endif

#if defined(DEBUG)
    print_device_mem_usage( );
#endif

    /* init the blocks sizes for cuda kernels */
    init_blocks( system );

    /* compute f_0 */
    Comm_Atoms( system, control, data, workspace, mpi_data, TRUE );
    Sync_Atoms( system );
    Sync_Grid( &system->my_grid, &system->d_my_grid );
    init_blocks( system );

    Cuda_Reset( system, control, data, workspace, lists );

#if defined(__CUDA_DEBUG__)
    Reset( system, control, data, workspace, lists );
#endif

    Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );

#if defined(__CUDA_DEBUG__)
    Generate_Neighbor_Lists( system, data, workspace, lists );
#endif

#if defined(__CUDA_DEBUG__)
    Compute_Forces( system, control, data, workspace,
            lists, out_control, mpi_data );
#endif

    Cuda_Compute_Forces( system, control, data, workspace, lists,
            out_control, mpi_data );

#if defined (__CUDA_DEBUG__)
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
#endif

    Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

#if defined(__CUDA_DEBUG__)
    validate_device( system, data, workspace, lists );
#endif

#if !defined(__CUDA_DEBUG__)
    Output_Results( system, control, data, lists, out_control, mpi_data );
#endif

#if defined(DEBUG)
    fprintf( stderr, "p%d: step%d completed\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    ++data->step;
    retries = 0;
    while ( data->step <= control->nsteps && retries < MAX_RETRIES )
    {
        ret = SUCCESS;

        if ( control->T_mode && retries == 0 )
        {
            Temperature_Control( control, data );
        }

#if defined(DEBUG)
        t_begin = Get_Time();
#endif

#if defined(__CUDA_DEBUG__)
        ret = Evolve( system, control, data, workspace, lists, out_control, mpi_data );
#endif
    
        ret = Cuda_Evolve( system, control, data, workspace, lists, out_control, mpi_data );
    
#if defined(DEBUG)
        t_end = Get_Timing_Info( t_begin );
        fprintf( stderr, " Evolve time: %f \n", t_end );
#endif

#if defined(DEBUG)
        t_begin = Get_Time();
#endif

        if ( ret == SUCCESS )
        {
            ret = Cuda_Post_Evolve( system, control, data, workspace, lists,
                    out_control, mpi_data );
        }

#if defined(__CUDA_DEBUG__)
        Post_Evolve(system, control, data, workspace, lists, out_control, mpi_data);
#endif

#if defined(DEBUG)
        t_end = Get_Timing_Info( t_begin );
        fprintf( stderr, " Post Evolve time: %f \n", t_end );
#endif

        if ( ret == SUCCESS )
        {
            data->timing.num_retries = retries;

#if !defined(__CUDA_DEBUG__)
            Output_Results( system, control, data, lists, out_control, mpi_data );
#endif

//        Analysis(system, control, data, workspace, lists, out_control, mpi_data);

        /* dump restart info */
//        if ( out_control->restart_freq &&
//                (data->step-data->prev_steps) % out_control->restart_freq == 0 )
//        {
//            if( out_control->restart_format == WRITE_ASCII )
//            {
//                Write_Restart( system, control, data, out_control, mpi_data );
//            }
//            else if( out_control->restart_format == WRITE_BINARY )
//            {
//                Write_Binary_Restart( system, control, data, out_control, mpi_data );
//            }
//        }

#if defined(DEBUG)
            fprintf( stderr, "p%d: step%d completed\n", system->my_rank, data->step );
            MPI_Barrier( MPI_COMM_WORLD );
#endif

            ++data->step;
            retries = 0;
        }
        else
        {
            ++retries;

#if defined(DEBUG)
            fprintf( stderr, "[INFO] p%d: retrying step %d...\n", system->my_rank, data->step );
#endif
        }
    }

    if ( retries >= MAX_RETRIES )
    {
        fprintf( stderr, "[ERROR] Maximum retries reached for this step (%d). Terminating...\n",
              retries );
        MPI_Abort( MPI_COMM_WORLD, MAX_RETRIES_REACHED );
    }

#if defined(__CUDA_DEBUG__)
    /* vaildate the results in debug mode */
    validate_device( system, data, workspace, lists );
#endif

#else 
    system = smalloc( sizeof(reax_system), "main::system" );
    control = smalloc( sizeof(control_params), "main::control" );
    data = smalloc( sizeof(simulation_data), "main::data" );
    workspace = smalloc( sizeof(storage), "main::workspace" );
    lists = smalloc( sizeof(reax_list *) * LIST_N, "main::lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
	lists[i] = smalloc( sizeof(reax_list), "main::lists[i]" );
        lists[i]->allocated = FALSE;
    }
    out_control = smalloc( sizeof(output_controls), "main::out_control" );
    mpi_data = smalloc( sizeof(mpi_datatypes), "main::mpi_data" );

    /* setup MPI environment */
    MPI_Comm_size( MPI_COMM_WORLD, &control->nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &system->my_rank );

    /* read config files */
    Read_Control_Files( argv[1], argv[2], argv[3], system, control,
            data, workspace, out_control, mpi_data );

    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }

    Initialize( system, control, data, workspace, lists, out_control, mpi_data );
   
    /* compute f_0 */
    Comm_Atoms( system, control, data, workspace, mpi_data, TRUE );

    Reset( system, control, data, workspace, lists );

    if ( ret == FAILURE )
    {
        ReAllocate( system, control, data, workspace, lists, mpi_data );
    }

    ret = Generate_Neighbor_Lists( system, data, workspace, lists );

    if ( ret != SUCCESS )
    {
        fprintf( stderr, "[ERROR] cannot generate initial neighbor lists. Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

    ret = Compute_Forces( system, control, data, workspace, lists, out_control, mpi_data );

    if ( ret != SUCCESS )
    {
        fprintf( stderr, "[ERROR] cannot compute initial forces. Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

    Compute_Total_Energy( system, data, MPI_COMM_WORLD );

    Output_Results( system, control, data, lists, out_control, mpi_data );

    Check_Energy( data );

    retries = 0;
    ++data->step;
    while ( data->step <= control->nsteps && retries < MAX_RETRIES )
    {
        ret = SUCCESS;

        if ( control->T_mode && retries == 0 )
        {
            Temperature_Control( control, data );
        }

        ret = Evolve( system, control, data, workspace, lists, out_control, mpi_data );

        if ( ret == SUCCESS )
        {
            Post_Evolve(system, control, data, workspace, lists, out_control, mpi_data);
        }

        if ( ret == SUCCESS )
        {
            data->timing.num_retries = retries;

            Output_Results( system, control, data, lists, out_control, mpi_data );

//            Analysis(system, control, data, workspace, lists, out_control, mpi_data);

            /* dump restart info */
            if ( out_control->restart_freq &&
                    (data->step - data->prev_steps) % out_control->restart_freq == 0 )
            {
                if ( out_control->restart_format == WRITE_ASCII )
                {
                    Write_Restart( system, control, data, out_control, mpi_data );
                }
                else if ( out_control->restart_format == WRITE_BINARY )
                {
                    Write_Binary_Restart( system, control, data, out_control, mpi_data );
                }
            }

            Check_Energy( data );

            ++data->step;
            retries = 0;
        }
        else
        {
            ++retries;

#if defined(DEBUG)
            fprintf( stderr, "[INFO] p%d: retrying step %d...\n", system->my_rank, data->step );
#endif
        }
    }

    if ( retries >= MAX_RETRIES )
    {
        fprintf( stderr, "[ERROR] Maximum retries reached for this step (%d). Terminating...\n",
              retries );
        MPI_Abort( MPI_COMM_WORLD, MAX_RETRIES_REACHED );
    }
    
#endif

    /* end of the simulation, write total simulation time */
    if ( system->my_rank == MASTER_NODE )
    {
        t_elapsed = Get_Timing_Info( t_start );
        fprintf( out_control->out, "Total Simulation Time: %.2f secs\n", t_elapsed );
    }

//    Write_PDB( &system, &lists[BONDS], &out_control );
    Close_Output_Files( system, control, out_control, mpi_data );

#if defined(TEST_ENERGY) || defined(TEST_FORCES)
//    Integrate_Results(control);
#endif

#if defined(DEBUG)
    fprintf( stderr, "p%d has reached the END\n", system->my_rank );
#endif

    MPI_Finalized( &ret );
    if ( !ret )
    { 
        MPI_Finalize( );
    }

    sfree( mpi_data, "main::mpi_data" );
    sfree( out_control, "main::out_control" );
    for ( i = 0; i < LIST_N; ++i )
    {
        sfree( lists[i], "main::lists[i]" );
    }
    sfree( lists, "main::lists" );
    sfree( workspace, "main::workspace" );
    sfree( data, "main::data" );
    sfree( control, "main::control" );
    sfree( system, "main::system" );

    return 0;
}
