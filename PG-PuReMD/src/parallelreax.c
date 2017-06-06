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
  #include "cuda_copy.h"
  #include "cuda_environment.h"
  #include "cuda_neighbors.h"
  #include "cuda_post_evolve.h"
  #include "cuda_utils.h"
  #include "validation.h"
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


void Read_System( char *geo_file, char *ffield_file, char *control_file,
        reax_system *system, control_params *control, simulation_data *data,
        storage *workspace, output_controls *out_control, mpi_datatypes *mpi_data )
{
    /* ffield file */
    Read_Force_Field( ffield_file, &(system->reax_param), control );

    /* control file */
    Read_Control_File( control_file, control, out_control );

    /* geo file */
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

    /* remove trans & rot velocity of the center of mass from system */
    if ( control->ensemble != NVE && control->remove_CoM_vel &&
            data->step % control->remove_CoM_vel == 0 )
    {
        /* compute velocity of the center of mass */
        Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );

        for ( i = 0; i < system->n; i++ )
        {
            /* remove translational vel */
            rvec_ScaledAdd( system->my_atoms[i].v, -1., data->vcm );

            /* remove rotational */
            rvec_ScaledSum( diff, 1., system->my_atoms[i].x, -1., data->xcm );
            rvec_Cross( cross, data->avcm, diff );
            rvec_ScaledAdd( system->my_atoms[i].v, -1., cross );
        }
    }

    /* compute kinetic energy of the system */
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
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


#ifdef HAVE_CUDA
void init_blocks(reax_system *system)
{
    compute_blocks( &BLOCKS, &BLOCK_SIZE, system->n );
    compute_nearest_pow_2( BLOCKS, &BLOCKS_POW_2 );

    compute_blocks( &BLOCKS_N, &BLOCK_SIZE, system->N );
    compute_nearest_pow_2( BLOCKS_N, &BLOCKS_POW_2_N );

    compute_matvec_blocks( &MATVEC_BLOCKS, system->N );

#if defined(__CUDA_DEBUG_LOG__)
    fprintf( stderr, " MATVEC_BLOCKS: %d BLOCKSIZE: %d  - N:%d \n",
            MATVEC_BLOCKS, MATVEC_BLOCK_SIZE, system->N );
#endif
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
    int i, ret;
    real t_start = 0, t_elapsed;
    real t_begin, t_end;

    if ( argc != 4 )
    {
        usage( argv );
        exit( INVALID_INPUT );
    }

#ifdef HAVE_CUDA

    /* Remove this debug information later */
#if defined(__CUDA_DEBUG_LOG__)
    fprintf( stderr, " Size of LR Lookup table %d \n", sizeof(LR_lookup_table) );
#endif

#if defined( __SM_35__)
    fprintf( stderr, " nbrs block size: %d \n", NBRS_BLOCK_SIZE );
    fprintf( stderr, " nbrs threads per atom: %d \n", NB_KER_THREADS_PER_ATOM );

    fprintf( stderr, " hbonds block size: %d \n", HB_BLOCK_SIZE );
    fprintf( stderr, " hbonds threads per atom: %d \n", HB_KER_THREADS_PER_ATOM );

    fprintf( stderr, " vdw block size: %d \n", VDW_BLOCK_SIZE );
    fprintf( stderr, " vdw threads per atom: %d \n", VDW_KER_THREADS_PER_ATOM );

    fprintf( stderr, " matvec block size: %d \n", MATVEC_BLOCK_SIZE );
    fprintf( stderr, " matvec threads per atom: %d \n", MATVEC_KER_THREADS_PER_ROW);

    fprintf( stderr, " General block size: %d \n", DEF_BLOCK_SIZE );
#endif

    /* allocate main data structures */
    system = (reax_system *) smalloc( sizeof(reax_system), "system" );
    control = (control_params *) smalloc( sizeof(control_params), "control" );
    data = (simulation_data *) smalloc( sizeof(simulation_data), "data" );
    workspace = (storage *) smalloc( sizeof(storage), "workspace" );
    lists = (reax_list **) smalloc( LIST_N * sizeof(reax_list*), "lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        lists[i] = (reax_list *) smalloc( sizeof(reax_list), "lists[i]" );
        lists[i]->allocated = FALSE;

        lists[i]->n = 0;
        lists[i]->num_intrs = 0;
        lists[i]->index = NULL;
        lists[i]->end_index = NULL;
        lists[i]->select.v = NULL;
    }
    out_control = (output_controls *) smalloc( sizeof(output_controls), "out_control" );
    mpi_data = (mpi_datatypes *) smalloc( sizeof(mpi_datatypes), "mpi_data" );

    /* allocate auxiliary data structures (GPU) */
    dev_workspace = (storage *) smalloc( sizeof(storage), "dev_workspace" );
    dev_lists = (reax_list **) smalloc ( LIST_N * sizeof (reax_list *), "dev_lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        dev_lists[i] = (reax_list *) smalloc( sizeof(reax_list), "lists[i]" );
        dev_lists[i]->allocated = FALSE;
        lists[i]->n = 0; 
        lists[i]->num_intrs = 0;
        lists[i]->index = NULL;
        lists[i]->end_index = NULL;
        lists[i]->select.v = NULL;
    }

    /* setup MPI environment */
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &(control->nprocs) );
    MPI_Comm_rank( MPI_COMM_WORLD, &(system->my_rank) );

    /* read system config files */
    Read_System( argv[1], argv[2], argv[3], system, control,
            data, workspace, out_control, mpi_data );

#if defined(DEBUG)
    fprintf( stderr, "p%d: read simulation info\n", system->my_rank );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    /* setup the CUDA Device for this process */
    Setup_Cuda_Environment( system->my_rank, control->nprocs, control->gpus_per_node );

#if defined(DEBUG)
    print_device_mem_usage( );
    fprintf( stderr, "p%d: Total number of GPUs on this node -- %d\n", system->my_rank, my_device_id);
#endif

    /* init the blocks sizes for cuda kernels */
    init_blocks( system );

    /* measure total simulation time after input is read */
    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }

    /* initialize data structures */
    Cuda_Initialize( system, control, data, workspace, lists, out_control, mpi_data );

#if defined(__CUDA_DEBUG__)
    Pure_Initialize( system, control, data, workspace, lists, out_control, mpi_data );
#endif

//#if defined(DEBUG)
    print_device_mem_usage( );
//#endif

    /* init the blocks sizes for cuda kernels */
    init_blocks( system );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initializated data structures\n", system->my_rank );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
    //END OF FIRST STEP

    // compute f_0
    Comm_Atoms( system, control, data, workspace, lists, mpi_data, TRUE );
    Sync_Atoms( system );
    Sync_Grid( &system->my_grid, &system->d_my_grid );
    init_blocks( system );

#if defined(__CUDA_DEBUG_LOG__)
    fprintf( stderr, "p%d: Comm_Atoms synchronized \n", system->my_rank );
#endif

    //Second step
    Cuda_Reset( system, control, data, workspace, lists );

#if defined(__CUDA_DEBUG__)
    Reset( system, control, data, workspace, lists );
#endif
#if defined(DEBUG)
    fprintf( stderr, "p%d: Cuda_Reset done...\n", system->my_rank );
#endif

    //Third Step
    Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );

#if defined(__CUDA_DEBUG__)
    Generate_Neighbor_Lists( system, data, workspace, lists );
#endif

#if defined(DEBUG)
    fprintf( stderr, "p%d: Cuda_Generate_Neighbor_Lists done...\n", system->my_rank );
#endif

    //Fourth Step
#if defined(DEBUG)
    fprintf( stderr, " Host Compute Forces begin.... \n" );
#endif

#if defined(__CUDA_DEBUG__)
    Compute_Forces( system, control, data, workspace,
            lists, out_control, mpi_data );
#endif

    Cuda_Compute_Forces( system, control, data, workspace, lists,
            out_control, mpi_data );

#if defined(DEBUG)
    fprintf (stderr, "p%d: Cuda_Compute_Forces done...\n", system->my_rank );
#endif

#if defined (__CUDA_DEBUG__)
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
#endif

    Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

#if defined(DEBUG)
    fprintf (stderr, "p%d: Cuda_Compute_Kinetic_Energy done ... \n", system->my_rank);
#endif

#if defined(__CUDA_DEBUG__)
    validate_device (system, data, workspace, lists);
#endif

#if !defined(__CUDA_DEBUG__)
    Output_Results( system, control, data, lists, out_control, mpi_data );
#endif
#if defined(DEBUG)
    fprintf (stderr, "p%d: Output_Results done ... \n", system->my_rank);
#endif

#if defined(DEBUG)
    fprintf( stderr, "p%d: computed forces at t0\n", system->my_rank );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    // start the simulation
    ++data->step;
    while ( data->step <= control->nsteps )
    {
        fprintf( stderr, "[BEGIN] STEP %d\n", data->step );
        ret = SUCCESS;

        if ( control->T_mode )
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
        fprintf( stderr, "[EVOLVE] STEP %d\n", data->step );
    
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
        fprintf( stderr, "[POST EVOLVE] STEP %d\n", data->step );

#if defined(__CUDA_DEBUG__)
        Post_Evolve(system, control, data, workspace, lists, out_control, mpi_data);
#endif

#if defined(DEBUG)
        t_end = Get_Timing_Info( t_begin );
        fprintf( stderr, " Post Evolve time: %f \n", t_end );
#endif

#if !defined(__CUDA_DEBUG__)
        if ( ret == SUCCESS )
        {
            Output_Results( system, control, data, lists, out_control, mpi_data );
        }
#endif

//        if ( ret == SUCCESS )
//        {
//            Analysis(system, control, data, workspace, lists, out_control, mpi_data);
//        }

        /* dump restart info */
//        if ( ret == SUCCESS && out_control->restart_freq &&
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

        if ( ret == SUCCESS )
        {
            ++data->step;
        }
        else
        {
            fprintf( stderr, "INFO: retrying step %d...\n", data->step );
        }
    }

#if defined(__CUDA_DEBUG__)
    /* vaildate the results in debug mode */
    validate_device( system, data, workspace, lists );
#endif

#else 
    /* allocate main data structures */
    system = (reax_system *) smalloc( sizeof(reax_system), "system" );
    control = (control_params *) smalloc( sizeof(control_params), "control" );
    data = (simulation_data *) smalloc( sizeof(simulation_data), "data" );
    workspace = (storage *) smalloc( sizeof(storage), "workspace" );

    lists = (reax_list **) smalloc( LIST_N * sizeof(reax_list*), "lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        // initialize here
	lists[i] = (reax_list *) smalloc( sizeof(reax_list), "lists[i]" );
        lists[i]->allocated = FALSE;
        lists[i]->n = 0; 
        lists[i]->num_intrs = 0;
        lists[i]->index = NULL;
        lists[i]->end_index = NULL;
        lists[i]->select.v = NULL;
    }
    out_control = (output_controls *) smalloc( sizeof(output_controls), "out_control" );
    mpi_data = (mpi_datatypes *) smalloc( sizeof(mpi_datatypes), "mpi_data" );

    //TODO: remove?
    /* allocate the cuda auxiliary data structures */
    dev_workspace = (storage *) smalloc( sizeof(storage), "dev_workspace" );
    dev_lists = (reax_list **) smalloc ( LIST_N * sizeof (reax_list *), "dev_lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        dev_lists[i] = (reax_list *) smalloc( sizeof(reax_list), "lists[i]" );
        dev_lists[i]->allocated = FALSE;
    }

    /* setup MPI environment */
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &(control->nprocs) );
    MPI_Comm_rank( MPI_COMM_WORLD, &(system->my_rank) );

    /* read system config files */
    Read_System( argv[1], argv[2], argv[3], system, control,
            data, workspace, out_control, mpi_data );

#if defined(DEBUG)
    fprintf( stderr, "p%d: read simulation info\n", system->my_rank );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    /* measure total simulation time after input is read */
    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }

    /* initialize datastructures */
    Initialize( system, control, data, workspace, lists, out_control, mpi_data );
   
#if defined(DEBUG)
    fprintf( stderr, "p%d: initializated data structures\n", system->my_rank );
    MPI_Barrier( mpi_data->world );
#endif

    /* compute f_0 */
    Comm_Atoms( system, control, data, workspace, lists, mpi_data, TRUE );
    Reset( system, control, data, workspace, lists );

#if defined(DEBUG)
    Print_List(*lists + BONDS);
#endif

    Generate_Neighbor_Lists( system, data, workspace, lists );
    Compute_Forces( system, control, data, workspace, lists, out_control, mpi_data );
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
    Output_Results( system, control, data, lists, out_control, mpi_data );

#if defined(DEBUG)
    fprintf( stderr, "p%d: computed forces at t0\n", system->my_rank );
    MPI_Barrier( mpi_data->world );
#endif

    /* start the simulation */
    for ( ++data->step; data->step <= control->nsteps; data->step++ )
    {
        if ( control->T_mode )
        {
            Temperature_Control( control, data );
        }

        Evolve( system, control, data, workspace, lists, out_control, mpi_data );
        Post_Evolve(system, control, data, workspace, lists, out_control, mpi_data);
        Output_Results( system, control, data, lists, out_control, mpi_data );
        //Analysis(system, control, data, workspace, lists, out_control, mpi_data);

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

#if defined(DEBUG)
        fprintf( stderr, "p%d: step%d completed\n", system->my_rank, data->step );
        MPI_Barrier( mpi_data->world );
#endif
    }
    
#endif

    /* end of the simulation, write total simulation time */
    if ( system->my_rank == MASTER_NODE )
    {
        t_elapsed = Get_Timing_Info( t_start );
        fprintf( out_control->out, "Total Simulation Time: %.2f secs\n", t_elapsed );
    }

    // Write_PDB( &system, &(lists[BOND]), &out_control );
    Close_Output_Files( system, control, out_control, mpi_data );

    MPI_Finalize( );

    /* de-allocate data structures */
    //for( i = 0; i < LIST_N; ++i ) {
    //if (lists[i]->index) free (lists[i]->index);
    //if (lists[i]->end_index) free (lists[i]->end_index);
    //if (lists[i]->select.v) free (lists[i]->select.v);
    //free (lists[i] );
    //}

    free( system );
    free( control );
    free( data );
    free( workspace );
    free( lists );
    free( out_control );
    free( mpi_data );

#if defined(TEST_ENERGY) || defined(TEST_FORCES)
//  Integrate_Results(control);
#endif

#if defined(DEBUG)
    fprintf( stderr, "p%d has reached the END\n", system->my_rank );
#endif

    return 0;
}
