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

#if defined(HAVE_CUDA)
  #include "cuda/gpu_copy.h"
  #include "cuda/gpu_environment.h"
  #include "cuda/gpu_forces.h"
  #include "cuda/gpu_init_md.h"
  #include "cuda/gpu_neighbors.h"
  #include "cuda/gpu_post_evolve.h"
  #include "cuda/gpu_reset_tools.h"
  #include "cuda/gpu_system_props.h"
#elif defined(HAVE_HIP)
  #include "hip/gpu_copy.h"
  #include "hip/gpu_environment.h"
  #include "hip/gpu_forces.h"
  #include "hip/gpu_init_md.h"
  #include "hip/gpu_neighbors.h"
  #include "hip/gpu_post_evolve.h"
  #include "hip/gpu_reset_tools.h"
  #include "hip/gpu_system_props.h"
#endif


static void Read_Config_Files( const char * const geo_file,
        const char * const ffield_file,
        const char * const control_file,
        reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
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
    else if ( control->geo_format == BGF )
    {
        Read_BGF( geo_file, system, control, data, workspace, mpi_data );
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


#if defined(HAVE_CUDA) || defined(HAVE_HIP)
static void GPU_Post_Evolve( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
{
    /* remove translational and rotational velocity of the center of mass from system */
    if ( control->ensemble != NVE && control->remove_CoM_vel > 0
            && data->step % control->remove_CoM_vel == 0 )
    {
        /* compute velocity of the center of mass */
        GPU_Compute_Center_of_Mass( system, control, workspace,
                data, mpi_data, mpi_data->comm_mesh3D );

        GPU_Remove_CoM_Velocities( system, control, data );
    }

    if ( control->ensemble == NVE )
    {
        /* compute kinetic energy of the system */
        GPU_Compute_Kinetic_Energy( system, control, workspace,
                data, mpi_data->comm_mesh3D );
    }

    if ( (out_control->energy_update_freq > 0
                && (data->step - data->prev_steps) % out_control->energy_update_freq == 0)
            || (out_control->write_steps > 0
                && data->step % out_control->write_steps == 0) )
    {
        Compute_Total_Energy( system, control, data, MPI_COMM_WORLD );
    }
}


#else
static void Post_Evolve( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
{
    int i;
    rvec diff, cross;

    /* remove translational and rotational velocity of the center of mass from system */
    if ( control->ensemble != NVE && control->remove_CoM_vel > 0
            && data->step % control->remove_CoM_vel == 0 )
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

    if ( control->ensemble == NVE )
    {
        /* compute kinetic energy of system */
        Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
    }

    if ( (out_control->energy_update_freq > 0
                && (data->step - data->prev_steps) % out_control->energy_update_freq == 0)
            || (out_control->write_steps > 0
                && data->step % out_control->write_steps == 0) )
    {
        Compute_Total_Energy( system, control, data, MPI_COMM_WORLD );
    }
}
#endif


void* setup( const char * const geo_file, const char * const ffield_file,
        const char * const control_file )
{
    int i;
    puremd_handle *pmd_handle;

    /* top-level allocation */
    pmd_handle = (puremd_handle*) smalloc( sizeof(puremd_handle),
            __FILE__, __LINE__ );

    /* second-level allocations */
    pmd_handle->system = smalloc( sizeof(reax_system), __FILE__, __LINE__ );
    pmd_handle->control = smalloc( sizeof(control_params), __FILE__, __LINE__ );
    pmd_handle->data = smalloc( sizeof(simulation_data), __FILE__, __LINE__ );
    pmd_handle->workspace = smalloc( sizeof(storage), __FILE__, __LINE__ );
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
    pmd_handle->workspace->d_workspace = smalloc( sizeof(storage), __FILE__, __LINE__ );
#endif
    pmd_handle->lists = smalloc( sizeof(reax_list *) * LIST_N, __FILE__, __LINE__ );
    for ( i = 0; i < LIST_N; ++i )
    {
        pmd_handle->lists[i] = smalloc( sizeof(reax_list), __FILE__, __LINE__ );
        pmd_handle->lists[i]->allocated = FALSE;
    }
    pmd_handle->out_control = smalloc( sizeof(output_controls), __FILE__, __LINE__ );
    pmd_handle->mpi_data = smalloc( sizeof(mpi_datatypes), __FILE__, __LINE__ );

    pmd_handle->output_enabled = TRUE;
    pmd_handle->callback = NULL;

    /* setup MPI environment */
    MPI_Comm_size( MPI_COMM_WORLD, &pmd_handle->control->nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &pmd_handle->system->my_rank );

#if defined(DEBUG)
    fprintf( stderr, "[INFO] MPI timer resolution: %f\n", MPI_Wtick( ) );
#endif

    /* initialize logging timing and
     * globally synchronize clocks across all MPI processes */
    MPI_Barrier( MPI_COMM_WORLD );
    pmd_handle->data->timing.start = Get_Time( );

    /* read system config files */
    Read_Config_Files( geo_file, ffield_file, control_file,
            pmd_handle->system, pmd_handle->control, pmd_handle->data,
            pmd_handle->workspace, pmd_handle->out_control, pmd_handle->mpi_data );

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
    GPU_Setup_Environment( pmd_handle->system, pmd_handle->control );
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
    int ret, ret_pmd, retries;
    reax_system *system;
    control_params *control;
    simulation_data *data;
    storage *workspace;
    reax_list **lists;
    output_controls *out_control;
    mpi_datatypes *mpi_data;
    puremd_handle *pmd_handle;

    ret_pmd = PUREMD_FAILURE;

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

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
        GPU_Initialize( system, control, data, workspace, lists, out_control, mpi_data );

        /* compute f_0 */
        Comm_Atoms( system, control, data, workspace, mpi_data, TRUE );

#if defined(GPU_DEVICE_PACK)
        //TODO: remove once Comm_Atoms ported
        GPU_Copy_MPI_Data_Host_to_Device( control, mpi_data );
#endif

        GPU_Init_Block_Sizes( system, control );

        GPU_Copy_Atoms_Host_to_Device( system, control );
        GPU_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );

        GPU_Reset( system, control, data, workspace, lists );

        GPU_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        GPU_Compute_Forces( system, control, data, workspace, lists,
                out_control, mpi_data );

        GPU_Compute_Kinetic_Energy( system, control, workspace,
                data, mpi_data->comm_mesh3D );

        Compute_Total_Energy( system, control, data, MPI_COMM_WORLD );

        Output_Results( system, control, data, lists, out_control, mpi_data );

        Check_Energy( data );

#if defined(DEBUG_FOCUS)
        GPU_Print_Mem_Usage( data );
#endif

        ++data->step;
        retries = 0;
        while ( data->step <= control->nsteps && retries < MAX_RETRIES )
        {
            ret = SUCCESS;

            if ( control->T_mode > 0 && retries == 0 )
            {
                Temperature_Control( control, data );
            }
    
            ret = control->GPU_Evolve( system, control, data, workspace,
                    lists, out_control, mpi_data );

            if ( ret == SUCCESS )
            {
                GPU_Post_Evolve( system, control, data, workspace, lists,
                        out_control, mpi_data );
            }

            if ( ret == SUCCESS )
            {
                data->timing.num_retries = retries;

                Output_Results( system, control, data, lists, out_control, mpi_data );

//              Analysis( system, control, data, workspace, lists, out_control, mpi_data );

                if ( out_control->restart_freq
                        && (data->step - data->prev_steps) % out_control->restart_freq == 0 )
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

#if defined(DEBUG_FOCUS)
                fprintf( stderr, "[INFO] p%d: retrying step %d...\n", system->my_rank, data->step );
#endif
            }

#if defined(DEBUG_FOCUS)
            GPU_Print_Mem_Usage( data );
#endif
        }

        if ( retries >= MAX_RETRIES )
        {
            fprintf( stderr, "[ERROR] Maximum retries reached for this step (%d). Terminating...\n",
                  retries );
            MPI_Abort( MPI_COMM_WORLD, MAX_RETRIES_REACHED );
        }

#else 
        Initialize( system, control, data, workspace, lists, out_control, mpi_data );
       
        /* compute f_0 */
        Comm_Atoms( system, control, data, workspace, mpi_data, TRUE );

        Reset( system, control, data, workspace, lists );

        ret = Generate_Neighbor_Lists( system, control, data, workspace, lists );

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

        Compute_Total_Energy( system, control, data, MPI_COMM_WORLD );

        Output_Results( system, control, data, lists, out_control, mpi_data );

        Check_Energy( data );

        retries = 0;
        ++data->step;
        while ( data->step <= control->nsteps && retries < MAX_RETRIES )
        {
            ret = SUCCESS;

            if ( control->T_mode > 0 && retries == 0 )
            {
                Temperature_Control( control, data );
            }

            ret = control->Evolve( system, control, data, workspace,
                    lists, out_control, mpi_data );

            if ( ret == SUCCESS )
            {
                Post_Evolve( system, control, data, workspace,
                        lists, out_control, mpi_data );

                data->timing.num_retries = retries;
                Output_Results( system, control, data, lists, out_control, mpi_data );

//              Analysis( system, control, data, workspace, lists, out_control, mpi_data );

                if ( out_control->restart_freq
                        && (data->step - data->prev_steps) % out_control->restart_freq == 0 )
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

#if defined(DEBUG_FOCUS)
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

//      Write_PDB_File( system, lists[BONDS], data, control, mpi_data, out_control );

        /* end of simulation, write total simulation time
         * (excluding deallocation routine time) after
         * globally synchronizing clocks across all MPI processes */
        MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            fprintf( out_control->out, "Total Simulation Time: %.2f secs\n",
                    Get_Time( ) - data->timing.start );
        }

        ret_pmd = PUREMD_SUCCESS;
    }

    return ret_pmd;
}


int cleanup( const void * const handle )
{
    int ret_pmd;
    puremd_handle *pmd_handle;

    ret_pmd = PUREMD_FAILURE;

    if ( handle != NULL )
    {
        pmd_handle = (puremd_handle*) handle;

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
        //TODO: add GPU_Finalize( ... )

        GPU_Cleanup_Environment( pmd_handle->control );
#else
        Finalize( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control,
                pmd_handle->mpi_data, pmd_handle->output_enabled );
#endif

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
        sfree( pmd_handle->workspace->d_workspace, __FILE__, __LINE__ );
#endif
        sfree( pmd_handle->mpi_data, __FILE__, __LINE__ );
        sfree( pmd_handle->out_control, __FILE__, __LINE__ );
        sfree( pmd_handle->lists, __FILE__, __LINE__ );
        sfree( pmd_handle->workspace, __FILE__, __LINE__ );
        sfree( pmd_handle->data, __FILE__, __LINE__ );
        sfree( pmd_handle->control, __FILE__, __LINE__ );
        sfree( pmd_handle->system, __FILE__, __LINE__ );

        sfree( pmd_handle, __FILE__, __LINE__ );

        ret_pmd = PUREMD_SUCCESS;
    }

    return ret_pmd;
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
