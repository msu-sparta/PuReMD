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

#include "puremd.h"

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


/* CUDA-specific globals */
#if defined(HAVE_CUDA)
reax_list **dev_lists;
storage *dev_workspace;
void *scratch;
void *host_scratch;
int BLOCKS, BLOCKS_POW_2, BLOCK_SIZE;
int BLOCKS_N, BLOCKS_POW_2_N;
int MATVEC_BLOCKS;
#endif


static void Read_Config_Files( const char * const geo_file, const char * const ffield_file,
        const char * const control_file,
        reax_system * const system, control_params * const control, simulation_data * const data,
        storage * const workspace, output_controls * const out_control, mpi_datatypes * const mpi_data )
{
    Read_Force_Field_File( ffield_file, &system->reax_param, system, control );

    Read_Control_File( control_file, control, out_control );

    if ( control->geo_format == CUSTOM )
    {
        Read_Geo_File( geo_file, system, control, data, workspace, mpi_data );
    }
    else if ( control->geo_format == PDB )
    {
        Read_PDB_File( geo_file, system, control, data, workspace, mpi_data );
    }
    else if ( control->geo_format == ASCII_RESTART )
    {
        Read_Restart_File( geo_file, system, control, data, workspace, mpi_data );
        control->restart = 1;
    }
    else if ( control->geo_format == BINARY_RESTART )
    {
        Read_Binary_Restart_File( geo_file, system, control, data, workspace, mpi_data );
        control->restart = 1;
    }
    else
    {
        fprintf( stderr, "[ERROR] unknown geo file format. terminating!\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_GEO );
    }
}


static void Post_Evolve( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
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
static void Cuda_Post_Evolve( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
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
}
#endif


void* setup( const char * const geo_file, const char * const ffield_file,
        const char * const control_file )
{
    int i;
    puremd_handle *pmd_handle;

    /* top-level allocation */
    pmd_handle = (puremd_handle*) smalloc( sizeof(puremd_handle),
            "setup::pmd_handle" );

    /* second-level allocations */
    pmd_handle->system = smalloc( sizeof(reax_system),
           "Setup::pmd_handle->system" );
    pmd_handle->control = smalloc( sizeof(control_params),
           "Setup::pmd_handle->control" );
    pmd_handle->data = smalloc( sizeof(simulation_data),
           "Setup::pmd_handle->data" );
    pmd_handle->workspace = smalloc( sizeof(storage),
           "Setup::pmd_handle->workspace" );
    pmd_handle->lists = smalloc( sizeof(reax_list *) * LIST_N,
           "Setup::pmd_handle->lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        pmd_handle->lists[i] = smalloc( sizeof(reax_list),
                "Setup::pmd_handle->lists[i]" );
        pmd_handle->lists[i]->allocated = FALSE;
    }
    pmd_handle->out_control = smalloc( sizeof(output_controls),
           "Setup::pmd_handle->out_control" );
    pmd_handle->mpi_data = smalloc( sizeof(mpi_datatypes),
           "Setup::pmd_handle->mpi_data" );

#ifdef HAVE_CUDA
    /* allocate auxiliary data structures (GPU) */
    dev_workspace = smalloc( sizeof(storage), "Setup::dev_workspace" );
    dev_lists = smalloc ( sizeof(reax_list *) * LIST_N, "Setup::dev_lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        dev_lists[i] = smalloc( sizeof(reax_list), "Setup::dev_lists[i]" );
        dev_lists[i]->allocated = FALSE;
    }
#endif

    pmd_handle->output_enabled = TRUE;
    pmd_handle->callback = NULL;

    /* setup MPI environment */
    MPI_Comm_size( MPI_COMM_WORLD, &pmd_handle->control->nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &pmd_handle->system->my_rank );

    /* read system config files */
    Read_Config_Files( geo_file, ffield_file, control_file,
            pmd_handle->system, pmd_handle->control, pmd_handle->data,
            pmd_handle->workspace, pmd_handle->out_control, pmd_handle->mpi_data );

#ifdef HAVE_CUDA
    /* setup the CUDA Device for this process */
    Setup_Cuda_Environment( pmd_handle->system->my_rank,
            pmd_handle->control->nprocs, pmd_handle->control->gpus_per_node );

#if defined(DEBUG)
    print_device_mem_usage( );
#endif

    /* init blocks sizes */
    init_blocks( pmd_handle->system );
#endif

    return (void*) pmd_handle;
}


int setup_callback( const void * const handle, const callback_function callback  )
{
    int ret;
    puremd_handle *pmd_handle;


    ret = PUREMD_FAILURE;

    if ( handle != NULL && callback != NULL )
    {
        pmd_handle = (puremd_handle*) handle;
        pmd_handle->callback = callback;
        ret = PUREMD_SUCCESS;
    }

    return ret;
}


int simulate( const void * const handle )
{
    int ret, retries;
    reax_system *system;
    control_params *control;
    simulation_data *data;
    storage *workspace;
    reax_list **lists;
    output_controls *out_control;
    mpi_datatypes *mpi_data;
    puremd_handle *pmd_handle;
    real t_start, t_elapsed;
#if defined(DEBUG)
    real t_begin, t_end;
#endif

    t_start = 0;
    ret = PUREMD_FAILURE;

    if ( handle != NULL )
    {
        pmd_handle = (puremd_handle*) handle;

        system = pmd_handle->system;
        control = pmd_handle->control;
        data = pmd_handle->data;
        workspace = pmd_handle->workspace;
        lists = pmd_handle->lists;
        out_control = pmd_handle->out_control;
        mpi_data = pmd_handle->mpi_data;

#ifdef HAVE_CUDA
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
            ret = control->Evolve( system, control, data, workspace,
                    lists, out_control, mpi_data );
#endif
    
            ret = control->Cuda_Evolve( system, control, data, workspace,
                    lists, out_control, mpi_data );
    
#if defined(DEBUG)
            t_end = Get_Timing_Info( t_begin );
            fprintf( stderr, " Evolve time: %f \n", t_end );
#endif

#if defined(DEBUG)
            t_begin = Get_Time( );
#endif

            if ( ret == SUCCESS )
            {
                Cuda_Post_Evolve( system, control, data, workspace, lists,
                        out_control, mpi_data );

#if defined(__CUDA_DEBUG__)
                Post_Evolve( system, control, data, workspace, lists, out_control, mpi_data );
#endif
            }

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

//          Analysis( system, control, data, workspace, lists, out_control, mpi_data );

            /* dump restart info */
//            if ( out_control->restart_freq &&
//                    (data->step-data->prev_steps) % out_control->restart_freq == 0 )
//            {
//                if( out_control->restart_format == WRITE_ASCII )
//                {
//                    Write_Restart_File( system, control, data, out_control, mpi_data );
//                }
//                else if( out_control->restart_format == WRITE_BINARY )
//                {
//                    Write_Binary_Restart_File( system, control, data, out_control, mpi_data );
//                }
//            }

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

            ret = control->Evolve( system, control, data, workspace,
                    lists, out_control, mpi_data );

            if ( ret == SUCCESS )
            {
                Post_Evolve(system, control, data, workspace, lists, out_control, mpi_data);
            }

            if ( ret == SUCCESS )
            {
                data->timing.num_retries = retries;

                Output_Results( system, control, data, lists, out_control, mpi_data );

//              Analysis( system, control, data, workspace, lists, out_control, mpi_data );

                /* dump restart info */
                if ( out_control->restart_freq &&
                        (data->step - data->prev_steps) % out_control->restart_freq == 0 )
                {
                    if ( out_control->restart_format == WRITE_ASCII )
                    {
                        Write_Restart_File( system, control, data, out_control, mpi_data );
                    }
                    else if ( out_control->restart_format == WRITE_BINARY )
                    {
                        Write_Binary_Restart_File( system, control, data, out_control, mpi_data );
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

//      Write_PDB_File( &system, &lists[BONDS], &out_control );

#if defined(TEST_ENERGY) || defined(TEST_FORCES)
//      Integrate_Results(control);
#endif

#if defined(DEBUG)
        fprintf( stderr, "p%d has reached the END\n", system->my_rank );
#endif

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


int cleanup( const void * const handle )
{
    int i, ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL )
    {
        pmd_handle = (puremd_handle*) handle;

        Finalize( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control,
                pmd_handle->mpi_data, pmd_handle->output_enabled );

        sfree( pmd_handle->mpi_data, "cleanup::pmd_handle->mpi_data" );
        sfree( pmd_handle->out_control, "cleanup::pmd_handle->out_control" );
        for ( i = 0; i < LIST_N; ++i )
        {
            sfree( pmd_handle->lists[i], "cleanup::pmd_handle->lists[i]" );
        }
        sfree( pmd_handle->lists, "cleanup::pmd_handle->lists" );
        sfree( pmd_handle->workspace, "cleanup::pmd_handle->workspace" );
        sfree( pmd_handle->data, "cleanup::pmd_handle->data" );
        sfree( pmd_handle->control, "cleanup::pmd_handle->control" );
        sfree( pmd_handle->system, "cleanup::pmd_handle->system" );

        sfree( pmd_handle, "cleanup::pmd_handle" );

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


reax_atom* get_atoms( const void * const handle )
{
    puremd_handle *pmd_handle;
    reax_atom *atoms;

    atoms = NULL;

    if ( handle != NULL )
    {
        pmd_handle = (puremd_handle*) handle;
        atoms = pmd_handle->system->my_atoms;
    }

    return atoms;
}


int set_output_enabled( const void * const handle, const int enabled )
{
    int ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL )
    {
        pmd_handle = (puremd_handle*) handle;
        pmd_handle->output_enabled = enabled;
        ret = PUREMD_SUCCESS;
    }

    return ret;
}
