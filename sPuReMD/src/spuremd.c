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

#include "mytypes.h"

#include "analyze.h"
#include "control.h"
#include "ffield.h"
#include "forces.h"
#include "init_md.h"
#include "neighbors.h"
#include "geo_tools.h"
#include "print_utils.h"
#include "reset_utils.h"
#include "restart.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


static void Post_Evolve( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        reax_list ** const lists, output_controls * const out_control )
{
    int i;
    rvec diff, cross;

    /* if velocity dependent force then
       {
       Generate_Neighbor_Lists( &system, &control, &lists );
       Compute_Charges(system, control, workspace, lists[FAR_NBRS]);
       Introduce compute_force here if we are using velocity dependent forces
       Compute_Forces(system,control,data,workspace,lists);
       } */

    /* compute kinetic energy of the system */
    Compute_Kinetic_Energy( system, data );

    /* remove rotational and translational velocity of the center of mass */
    if ( control->ensemble != NVE &&
            control->remove_CoM_vel &&
            data->step && data->step % control->remove_CoM_vel == 0 )
    {

        /* compute velocity of the center of mass */
        Compute_Center_of_Mass( system, data, out_control->prs );

        for ( i = 0; i < system->N; i++ )
        {
            // remove translational
            rvec_ScaledAdd( system->atoms[i].v, -1., data->vcm );

            // remove rotational
            rvec_ScaledSum( diff, 1., system->atoms[i].x, -1., data->xcm );
            rvec_Cross( cross, data->avcm, diff );
            rvec_ScaledAdd( system->atoms[i].v, -1., cross );
        }
    }
}


static void Read_System( const char * const geo_file,
        const char * const ffield_file,
        const char * const control_file,
        reax_system * const system,
        control_params * const control,
        simulation_data * const data,
        static_storage * const workspace,
        output_controls * const out_control )
{
    FILE *ffield, *ctrl;

    if ( (ffield = fopen( ffield_file, "r" )) == NULL )
    {
        fprintf( stderr, "[ERROR] Error opening the ffield file!\n" );
        fprintf( stderr, "    [INFO] (%s)\n", ffield_file );
        exit( FILE_NOT_FOUND );
    }
    if ( (ctrl = fopen( control_file, "r" )) == NULL )
    {
        fprintf( stderr, "[ERROR] Error opening the ffield file!\n" );
        fprintf( stderr, "    [INFO] (%s)\n", control_file );
        exit( FILE_NOT_FOUND );
    }

    /* ffield file */
    Read_Force_Field( ffield, &(system->reaxprm) );

    /* control file */
    Read_Control_File( ctrl, system, control, out_control );

    /* geo file */
    if ( control->geo_format == CUSTOM )
    {
        Read_Geo( geo_file, system, control, data, workspace );
    }
    else if ( control->geo_format == PDB )
    {
        Read_PDB( geo_file, system, control, data, workspace );
    }
    else if ( control->geo_format == BGF )
    {
        Read_BGF( geo_file, system, control, data, workspace );
    }
    else if ( control->geo_format == ASCII_RESTART )
    {
        Read_ASCII_Restart( geo_file, system, control, data, workspace );
        control->restart = 1;
    }
    else if ( control->geo_format == BINARY_RESTART )
    {
        Read_Binary_Restart( geo_file, system, control, data, workspace );
        control->restart = 1;
    }
    else
    {
        fprintf( stderr, "[ERROR] unknown geo file format. terminating!\n" );
        exit( INVALID_GEO );
    }

    fclose( ffield );
    fclose( ctrl );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "input files have been read...\n" );
    Print_Box( &(system->box), stderr );
#endif
}


void* setup( const char * const geo_file, const char * const ffield_file,
        const char * const control_file )
{
    spuremd_handle *spmd_handle;

    /* top-level allocation */
    spmd_handle = (spuremd_handle*) smalloc( sizeof(spuremd_handle),
            "setup::spmd_handle" );

    /* second-level allocations */
    spmd_handle->system = (reax_system*) smalloc( sizeof(reax_system),
           "Setup::spmd_handle->system" );
    spmd_handle->control = (control_params*) smalloc( sizeof(control_params),
           "Setup::spmd_handle->control" );
    spmd_handle->data = (simulation_data*) smalloc( sizeof(simulation_data),
           "Setup::spmd_handle->data" );
    spmd_handle->workspace = (static_storage*) smalloc( sizeof(static_storage),
           "Setup::spmd_handle->workspace" );
    spmd_handle->lists = (reax_list*) smalloc( sizeof(reax_list) * LIST_N,
           "Setup::spmd_handle->lists" );
    spmd_handle->out_control = (output_controls*) smalloc( sizeof(output_controls),
           "Setup::spmd_handle->out_control" );

    spmd_handle->output_enabled = TRUE;

    /* parse geometry file */
    Read_System( geo_file, ffield_file, control_file,
            spmd_handle->system, spmd_handle->control,
            spmd_handle->data, spmd_handle->workspace,
            spmd_handle->out_control );

    //TODO: if errors detected, set handle to NULL to indicate failure

    return (void*) spmd_handle;
}


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
                spmd_handle->workspace, &spmd_handle->lists,
                spmd_handle->out_control, &Evolve,
                spmd_handle->output_enabled );

        /* compute f_0 */
        //if( control.restart == 0 ) {
        Reset( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, &spmd_handle->lists );
        Generate_Neighbor_Lists( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, &spmd_handle->lists, spmd_handle->out_control );

        //fprintf( stderr, "total: %.2f secs\n", data.timing.nbrs);
        Compute_Forces( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, &spmd_handle->lists, spmd_handle->out_control );
        Compute_Kinetic_Energy( spmd_handle->system, spmd_handle->data );
        if ( spmd_handle->output_enabled == TRUE )
        {
            Output_Results( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                    spmd_handle->workspace, &spmd_handle->lists, spmd_handle->out_control );
        }
        ++spmd_handle->data->step;
        //}
        
        for ( ; spmd_handle->data->step <= spmd_handle->control->nsteps; spmd_handle->data->step++ )
        {
            if ( spmd_handle->control->T_mode )
            {
                Temperature_Control( spmd_handle->control, spmd_handle->data,
                        spmd_handle->out_control );
            }

            Evolve( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                    spmd_handle->workspace, &spmd_handle->lists, spmd_handle->out_control );
            Post_Evolve( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                    spmd_handle->workspace, &spmd_handle->lists, spmd_handle->out_control );

            if ( spmd_handle->output_enabled == TRUE )
            {
                Output_Results( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                        spmd_handle->workspace, &spmd_handle->lists, spmd_handle->out_control );
                Analysis( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                        spmd_handle->workspace, &spmd_handle->lists, spmd_handle->out_control );
            }

            steps = spmd_handle->data->step - spmd_handle->data->prev_steps;
            if ( steps && spmd_handle->out_control->restart_freq &&
                    steps % spmd_handle->out_control->restart_freq == 0 &&
                    spmd_handle->output_enabled == TRUE )
            {
                Write_Restart( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                        spmd_handle->workspace, spmd_handle->out_control );
            }
        }

        if ( spmd_handle->out_control->write_steps > 0 && spmd_handle->output_enabled == TRUE )
        {
            fclose( spmd_handle->out_control->trj );
            Write_PDB( spmd_handle->system, &(spmd_handle->lists[BONDS]), spmd_handle->data,
                    spmd_handle->control, spmd_handle->workspace, spmd_handle->out_control );
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


int cleanup( const void * const handle )
{
    int ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        Finalize( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, &spmd_handle->lists, spmd_handle->out_control,
                spmd_handle->output_enabled );

        sfree( spmd_handle->out_control, "cleanup::spmd_handle->out_control" );
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