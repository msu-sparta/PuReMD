/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

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

#include "spuremd.h"

#include "analyze.h"
#if defined(DEBUG_FOCUS)
  #include "box.h"
#endif
#include "control.h"
#include "ffield.h"
#include "forces.h"
#include "init_md.h"
#include "io_tools.h"
#include "neighbors.h"
#include "geo_tools.h"
#include "reset_tools.h"
#include "restart.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


/* Handles additional entire geometry calculations after
 * perturbing atom positions during a simulation step
 */
static void Post_Evolve( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        reax_list ** const lists, output_controls * const out_control )
{
    int i;
    rvec diff, cross;

    /* remove rotational and translational velocity of the center of mass */
    if ( control->ensemble != NVE && control->remove_CoM_vel > 0
            && data->step % control->remove_CoM_vel == 0 )
    {
        /* compute velocity of the center of mass */
        Compute_Center_of_Mass( system, data );

        for ( i = 0; i < system->N; i++ )
        {
            /* remove translational */
            rvec_ScaledAdd( system->atoms[i].v, -1.0, data->vcm );

            /* remove rotational */
            rvec_ScaledSum( diff, 1.0, system->atoms[i].x, -1.0, data->xcm );
            rvec_Cross( cross, data->avcm, diff );
            rvec_ScaledAdd( system->atoms[i].v, -1.0, cross );
        }
    }

    Compute_Kinetic_Energy( system, data );

    if ( (out_control->log_update_freq > 0
                && data->step % out_control->log_update_freq == 0)
            || (out_control->write_steps > 0
                && data->step % out_control->write_steps == 0) )
    {
        Compute_Total_Energy( data );
    }

    if ( control->compute_pressure == TRUE && control->ensemble != sNPT
            && control->ensemble != iNPT && control->ensemble != aNPT )
    {
        Compute_Pressure_Isotropic( system, control, data, out_control );
    }
}


/* Parse input files
 *
 * geo_file: file containing geometry info of the structure to simulate
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 */
static void Read_Input_Files( const char * const geo_file,
        const char * const ffield_file, const char * const control_file,
        reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        output_controls * const out_control, int first_run )
{
    FILE *ffield, *ctrl;

    ffield = sfopen( ffield_file, "r" );
    ctrl = sfopen( control_file, "r" );

    Read_Force_Field( ffield, &system->reax_param, first_run );

    Read_Control_File( ctrl, system, control, out_control );

    if ( control->geo_format == CUSTOM )
    {
        Read_Geo( geo_file, system, control, data, workspace, first_run );
    }
    else if ( control->geo_format == PDB )
    {
        Read_PDB( geo_file, system, control, data, workspace, first_run );
    }
    else if ( control->geo_format == BGF )
    {
        Read_BGF( geo_file, system, control, data, workspace, first_run );
    }
    else if ( control->geo_format == ASCII_RESTART )
    {
        Read_ASCII_Restart( geo_file, system, control, data, workspace, first_run );
        control->restart = TRUE;
    }
    else if ( control->geo_format == BINARY_RESTART )
    {
        Read_Binary_Restart( geo_file, system, control, data, workspace, first_run );
        control->restart = TRUE;
    }
    else
    {
        fprintf( stderr, "[ERROR] unknown geo file format. terminating!\n" );
        exit( INVALID_GEO );
    }

    sfclose( ffield, "Read_Input_Files::ffield" );
    sfclose( ctrl, "Read_Input_Files::ctrl" );

#if defined(DEBUG_FOCUS)
    Print_Box( &system->box, stderr );
#endif
}


/* Allocate top-level data structures and parse input files
 * for the first simulation
 *
 * geo_file: file containing geometry info of the structure to simulate
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 */
void* setup( const char * const geo_file, const char * const ffield_file,
        const char * const control_file )
{
    int i;
    spuremd_handle *spmd_handle;

    /* top-level allocation */
    spmd_handle = (spuremd_handle*) smalloc( sizeof(spuremd_handle),
            "setup::spmd_handle" );

    /* second-level allocations */
    spmd_handle->system = smalloc( sizeof(reax_system),
           "Setup::spmd_handle->system" );
    spmd_handle->control = smalloc( sizeof(control_params),
           "Setup::spmd_handle->control" );
    spmd_handle->data = smalloc( sizeof(simulation_data),
           "Setup::spmd_handle->data" );
    spmd_handle->workspace = smalloc( sizeof(static_storage),
           "Setup::spmd_handle->workspace" );
    spmd_handle->lists = smalloc( sizeof(reax_list *) * LIST_N,
           "Setup::spmd_handle->lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        spmd_handle->lists[i] = smalloc( sizeof(reax_list),
                "Setup::spmd_handle->lists[i]" );
//        spmd_handle->lists[i]->allocated = FALSE;
    }
    spmd_handle->out_control = smalloc( sizeof(output_controls),
           "Setup::spmd_handle->out_control" );

    spmd_handle->first_run = TRUE;
    spmd_handle->output_enabled = TRUE;
    spmd_handle->realloc = TRUE;
    spmd_handle->callback = NULL;
    spmd_handle->data->sim_id = 0;

    Read_Input_Files( geo_file, ffield_file, control_file,
            spmd_handle->system, spmd_handle->control,
            spmd_handle->data, spmd_handle->workspace,
            spmd_handle->out_control, spmd_handle->first_run );

    spmd_handle->system->N_max = (int) CEIL( SAFE_ZONE * spmd_handle->system->N );

    return (void*) spmd_handle;
}


/* Setup callback function to be run after each simulation step
 *
 * handle: pointer to wrapper struct with top-level data structures
 * callback: function pointer to attach for callback
 */
int setup_callback( const void * const handle, const callback_function callback  )
{
    int ret;
    spuremd_handle *spmd_handle;


    ret = SPUREMD_FAILURE;

    if ( handle != NULL && callback != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;
        spmd_handle->callback = callback;
        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Run the simulation according to the prescribed parameters
 *
 * handle: pointer to wrapper struct with top-level data structures
 */
int simulate( const void * const handle )
{
    int steps, ret;
    evolve_function Evolve;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        Initialize( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, spmd_handle->lists,
                spmd_handle->out_control, &Evolve,
                spmd_handle->output_enabled,
                spmd_handle->first_run, spmd_handle->realloc );

        spmd_handle->realloc = FALSE;

        /* compute f_0 */
        //if( control.restart == FALSE ) {
        Reset( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, spmd_handle->lists );

        Generate_Neighbor_Lists( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control );

        Compute_Forces( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control );

        Compute_Kinetic_Energy( spmd_handle->system, spmd_handle->data );

        if ( spmd_handle->control->compute_pressure == TRUE && spmd_handle->control->ensemble != sNPT
                && spmd_handle->control->ensemble != iNPT && spmd_handle->control->ensemble != aNPT )
        {
            Compute_Pressure_Isotropic( spmd_handle->system, spmd_handle->control,
                    spmd_handle->data, spmd_handle->out_control );
        }

        if ( spmd_handle->output_enabled == TRUE || spmd_handle->callback != NULL )
        {
            if ( ((spmd_handle->out_control->log_update_freq > 0
                        && spmd_handle->data->step % spmd_handle->out_control->log_update_freq == 0)
                    || (spmd_handle->out_control->write_steps > 0
                        && spmd_handle->data->step % spmd_handle->out_control->write_steps == 0))
                || spmd_handle->callback != NULL )
            {
                Compute_Total_Energy( spmd_handle->data );
            }
        }

        if ( spmd_handle->output_enabled == TRUE )
        {
            Output_Results( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                    spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control );
        }

        Check_Energy( spmd_handle->data );

        if ( spmd_handle->output_enabled == TRUE )
        {
            if ( spmd_handle->out_control->write_steps > 0
                    && spmd_handle->data->step % spmd_handle->out_control->write_steps == 0 )
            {
                Write_PDB( spmd_handle->system, spmd_handle->lists[BONDS], spmd_handle->data,
                        spmd_handle->control, spmd_handle->workspace, spmd_handle->out_control );
            }
        }

        if ( spmd_handle->callback != NULL )
        {
            spmd_handle->callback( spmd_handle->system->atoms, spmd_handle->data,
                    spmd_handle->lists );
        }
        //}

        for ( ++spmd_handle->data->step; spmd_handle->data->step <= spmd_handle->control->nsteps; spmd_handle->data->step++ )
        {
            if ( spmd_handle->control->T_mode != 0 )
            {
                Temperature_Control( spmd_handle->control, spmd_handle->data,
                        spmd_handle->out_control );
            }

            Evolve( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                    spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control );

            Post_Evolve( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                    spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control );

            if ( spmd_handle->output_enabled == TRUE )
            {
                Output_Results( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                        spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control );
            }

            Check_Energy( spmd_handle->data );

            if ( spmd_handle->output_enabled == TRUE )
            {
                steps = spmd_handle->data->step - spmd_handle->data->prev_steps;

                Analysis( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                        spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control );

                if ( spmd_handle->out_control->restart_freq > 0
                        && steps % spmd_handle->out_control->restart_freq == 0 )
                {
                    Write_Restart( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                            spmd_handle->workspace, spmd_handle->out_control );
                }

                if ( spmd_handle->out_control->write_steps > 0
                        && steps % spmd_handle->out_control->write_steps == 0 )
                {
                    Write_PDB( spmd_handle->system, spmd_handle->lists[BONDS], spmd_handle->data,
                            spmd_handle->control, spmd_handle->workspace, spmd_handle->out_control );
                }
            }

            if ( spmd_handle->callback != NULL )
            {
                spmd_handle->callback( spmd_handle->system->atoms, spmd_handle->data,
                        spmd_handle->lists );
            }
        }

        spmd_handle->data->timing.end = Get_Time( );
        spmd_handle->data->timing.elapsed = Get_Timing_Info( spmd_handle->data->timing.start );

        if ( spmd_handle->output_enabled == TRUE )
        {
            fprintf( spmd_handle->out_control->log, "total: %.2f secs\n", spmd_handle->data->timing.elapsed );
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Deallocate all data structures post-simulation
 *
 * handle: pointer to wrapper struct with top-level data structures
 */
int cleanup( const void * const handle )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        Finalize( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control,
                spmd_handle->output_enabled, FALSE );

        sfree( spmd_handle->out_control, "cleanup::spmd_handle->out_control" );
        for ( i = 0; i < LIST_N; ++i )
        {
            sfree( spmd_handle->lists[i], "cleanup::spmd_handle->lists[i]" );
        }
        sfree( spmd_handle->lists, "cleanup::spmd_handle->lists" );
        sfree( spmd_handle->workspace, "cleanup::spmd_handle->workspace" );
        sfree( spmd_handle->data, "cleanup::spmd_handle->data" );
        sfree( spmd_handle->control, "cleanup::spmd_handle->control" );
        sfree( spmd_handle->system, "cleanup::spmd_handle->system" );

        sfree( spmd_handle, "cleanup::spmd_handle" );

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Reset for the next simulation by parsing input files and triggering
 * reallocation if more space is needed
 *
 * handle: pointer to wrapper struct with top-level data structures
 * geo_file: file containing geometry info of the structure to simulate
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 */
int reset( const void * const handle, const char * const geo_file,
        const char * const ffield_file, const char * const control_file )
{
    int ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        spmd_handle->first_run = FALSE;
        spmd_handle->realloc = FALSE;
        spmd_handle->data->sim_id++;

        Read_Input_Files( geo_file, ffield_file, control_file,
                spmd_handle->system, spmd_handle->control,
                spmd_handle->data, spmd_handle->workspace,
                spmd_handle->out_control, spmd_handle->first_run );

        if ( spmd_handle->system->N > spmd_handle->system->N_max )
        {
            /* deallocate everything which needs more space
             * (i.e., structures whose space is a function of the number of atoms),
             * except for data structures allocated while parsing input files */
            Finalize( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                    spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control,
                    spmd_handle->output_enabled, TRUE );

            spmd_handle->system->N_max = (int) CEIL( SAFE_ZONE * spmd_handle->system->N );
            spmd_handle->realloc = TRUE;
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom info
 *
 * handle: pointer to wrapper struct with top-level data structures
 */
reax_atom* get_atoms( const void * const handle )
{
    spuremd_handle *spmd_handle;
    reax_atom *atoms;

    atoms = NULL;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;
        atoms = spmd_handle->system->atoms;
    }

    return atoms;
}


/* Setter for writing output to files
 *
 * handle: pointer to wrapper struct with top-level data structures
 * enabled: TRUE enables writing output to files, FALSE otherwise
 */
int set_output_enabled( const void * const handle, const int enabled )
{
    int ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;
        spmd_handle->output_enabled = enabled;
        ret = SPUREMD_SUCCESS;
    }

    return ret;
}
