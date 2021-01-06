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

#include "allocate.h"
#include "analyze.h"
#include "box.h"
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

    if ( control->ensemble == NVE )
    {
        Compute_Kinetic_Energy( system, data );
    }

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
        output_controls * const out_control )
{
    if ( ffield_file != NULL )
    {
        Read_Force_Field( ffield_file, system, &system->reax_param );
    }

    if ( control_file != NULL )
    {
        Read_Control_File( control_file, system, control, out_control );
    }

    if ( geo_file != NULL )
    {
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
            control->restart = TRUE;
        }
        else if ( control->geo_format == BINARY_RESTART )
        {
            Read_Binary_Restart( geo_file, system, control, data, workspace );
            control->restart = TRUE;
        }
        else
        {
            fprintf( stderr, "[ERROR] unknown geo file format. terminating!\n" );
            exit( INVALID_GEO );
        }
    }

#if defined(DEBUG_FOCUS)
    Print_Box( &system->box, stderr );
#endif
}


/* Allocate top-level data structures and parse input files
 * for the first simulation
 *
 * qm_num_atoms: num. atoms in the QM region
 * qm_types: element types for QM atoms
 * qm_pos_x: x-coordinate of QM atom positions, in Angstroms
 * qm_pos_y: y-coordinate of QM atom positions, in Angstroms
 * qm_pos_z: z-coordinate of QM atom positions, in Angstroms
 * mm_num_atoms: num. atoms in the MM region
 * mm_types: element types for MM atoms
 * mm_pos_x: x-coordinate of MM atom positions, in Angstroms
 * mm_pos_y: y-coordinate of MM atom positions, in Angstroms
 * mm_pos_z: z-coordinate of MM atom positions, in Angstroms
 * mm_q: charge of MM atom, in Coulombs
 * sim_box: simulation box information, where the entries are
 *  - box length per dimension (3 entries)
 *  - angles per dimension (3 entries)
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 */
void * setup_qmmm_( int qm_num_atoms, const int * const qm_types,
        const double * const qm_pos_x, const double * const qm_pos_y,
        const double * const qm_pos_z,
        int mm_num_atoms, const int * const mm_types,
        const double * const mm_pos_x, const double * const mm_pos_y,
        const double * const mm_pos_z, const double * const mm_q,
        const double * const sim_box,
        const char * const ffield_file,
        const char * const control_file )
{
    int i;
//    char atom_name[9];
    rvec x;
    spuremd_handle *spmd_handle;

    /* top-level allocation */
    spmd_handle = (spuremd_handle*) smalloc( sizeof(spuremd_handle),
            "setup::spmd_handle" );

    /* second-level allocations */
    spmd_handle->system = smalloc( sizeof(reax_system),
           "Setup::spmd_handle->system" );
    spmd_handle->system->prealloc_allocated = FALSE;
    spmd_handle->system->ffield_params_allocated = FALSE;
    spmd_handle->system->g.allocated = FALSE;

    spmd_handle->control = smalloc( sizeof(control_params),
           "Setup::spmd_handle->control" );

    spmd_handle->data = smalloc( sizeof(simulation_data),
           "Setup::spmd_handle->data" );

    spmd_handle->workspace = smalloc( sizeof(static_storage),
           "Setup::spmd_handle->workspace" );
    spmd_handle->workspace->H.allocated = FALSE;
    spmd_handle->workspace->H_full.allocated = FALSE;
    spmd_handle->workspace->H_sp.allocated = FALSE;
    spmd_handle->workspace->H_p.allocated = FALSE;
    spmd_handle->workspace->H_spar_patt.allocated = FALSE;
    spmd_handle->workspace->H_spar_patt_full.allocated = FALSE;
    spmd_handle->workspace->H_app_inv.allocated = FALSE;
    spmd_handle->workspace->L.allocated = FALSE;
    spmd_handle->workspace->U.allocated = FALSE;

    spmd_handle->lists = smalloc( sizeof(reax_list *) * LIST_N,
           "Setup::spmd_handle->lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        spmd_handle->lists[i] = smalloc( sizeof(reax_list),
                "Setup::spmd_handle->lists[i]" );
        spmd_handle->lists[i]->allocated = FALSE;
    }
    spmd_handle->out_control = smalloc( sizeof(output_controls),
           "Setup::spmd_handle->out_control" );

    spmd_handle->output_enabled = FALSE;
    spmd_handle->realloc = TRUE;
    spmd_handle->callback = NULL;
    spmd_handle->data->sim_id = 0;

    spmd_handle->system->N_qm = qm_num_atoms;
    spmd_handle->system->N_mm = mm_num_atoms;
    spmd_handle->system->N = qm_num_atoms + mm_num_atoms;

    PreAllocate_Space( spmd_handle->system, spmd_handle->control,
            spmd_handle->workspace, spmd_handle->system->N );

    Setup_Box( sim_box[0], sim_box[1], sim_box[2],
            sim_box[3], sim_box[4], sim_box[5],
            &spmd_handle->system->box );

    for ( i = 0; i < qm_num_atoms; ++i )
    {
        x[0] = qm_pos_x[i];
        x[1] = qm_pos_y[i];
        x[2] = qm_pos_z[i];

        Fit_to_Periodic_Box( &spmd_handle->system->box, x );

        spmd_handle->workspace->orig_id[i] = i + 1;
//        spmd_handle->system->atoms[i].type = Get_Atom_Type( &system->reax_param,
//                element, sizeof(element) );
        spmd_handle->system->atoms[i].type = qm_types[i];
//        strncpy( spmd_handle->system->atoms[i].name, atom_name,
//                sizeof(spmd_handle->system->atoms[i].name) - 1 );
//        spmd_handle->system->atoms[i].name[sizeof(spmd_handle->system->atoms[i].name) - 1] = '\0';
        rvec_Copy( spmd_handle->system->atoms[i].x, x );
        rvec_MakeZero( spmd_handle->system->atoms[i].v );
        rvec_MakeZero( spmd_handle->system->atoms[i].f );
        spmd_handle->system->atoms[i].q = 0.0;

//        mask[i] = 1;
    }

    for ( i = qm_num_atoms; i < qm_num_atoms + mm_num_atoms; ++i )
    {
        x[0] = mm_pos_x[i - qm_num_atoms];
        x[1] = mm_pos_y[i - qm_num_atoms];
        x[2] = mm_pos_z[i - qm_num_atoms];

        Fit_to_Periodic_Box( &spmd_handle->system->box, x );

        spmd_handle->workspace->orig_id[i] = i + 1;
//        spmd_handle->system->atoms[i].type = Get_Atom_Type( &system->reax_param,
//                element, sizeof(element) );
        spmd_handle->system->atoms[i].type = mm_types[i - qm_num_atoms];
//        strncpy( spmd_handle->system->atoms[i].name, atom_name,
//                sizeof(spmd_handle->system->atoms[i].name) - 1 );
//        spmd_handle->system->atoms[i].name[sizeof(spmd_handle->system->atoms[i].name) - 1] = '\0';
        rvec_Copy( spmd_handle->system->atoms[i].x, x );
        rvec_MakeZero( spmd_handle->system->atoms[i].v );
        rvec_MakeZero( spmd_handle->system->atoms[i].f );
        spmd_handle->system->atoms[i].q = mm_q[i - qm_num_atoms];

//        mask[i] = 0;
    }

    Read_Input_Files( NULL, ffield_file, control_file,
            spmd_handle->system, spmd_handle->control,
            spmd_handle->data, spmd_handle->workspace,
            spmd_handle->out_control );

    spmd_handle->system->N_max = (int) CEIL( SAFE_ZONE * spmd_handle->system->N );

    return (void *) spmd_handle;
}


/* Allocate top-level data structures and parse input files
 * for the first simulation
 *
 * geo_file: file containing geometry info of the structure to simulate
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 */
void * setup( const char * const geo_file, const char * const ffield_file,
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
    spmd_handle->system->prealloc_allocated = FALSE;
    spmd_handle->system->ffield_params_allocated = FALSE;
    spmd_handle->system->g.allocated = FALSE;

    spmd_handle->control = smalloc( sizeof(control_params),
           "Setup::spmd_handle->control" );

    spmd_handle->data = smalloc( sizeof(simulation_data),
           "Setup::spmd_handle->data" );

    spmd_handle->workspace = smalloc( sizeof(static_storage),
           "Setup::spmd_handle->workspace" );
    spmd_handle->workspace->H.allocated = FALSE;
    spmd_handle->workspace->H_full.allocated = FALSE;
    spmd_handle->workspace->H_sp.allocated = FALSE;
    spmd_handle->workspace->H_p.allocated = FALSE;
    spmd_handle->workspace->H_spar_patt.allocated = FALSE;
    spmd_handle->workspace->H_spar_patt_full.allocated = FALSE;
    spmd_handle->workspace->H_app_inv.allocated = FALSE;
    spmd_handle->workspace->L.allocated = FALSE;
    spmd_handle->workspace->U.allocated = FALSE;

    spmd_handle->lists = smalloc( sizeof(reax_list *) * LIST_N,
           "Setup::spmd_handle->lists" );
    for ( i = 0; i < LIST_N; ++i )
    {
        spmd_handle->lists[i] = smalloc( sizeof(reax_list),
                "Setup::spmd_handle->lists[i]" );
        spmd_handle->lists[i]->allocated = FALSE;
    }
    spmd_handle->out_control = smalloc( sizeof(output_controls),
           "Setup::spmd_handle->out_control" );

    spmd_handle->output_enabled = TRUE;
    spmd_handle->realloc = TRUE;
    spmd_handle->callback = NULL;
    spmd_handle->data->sim_id = 0;

    Read_Input_Files( geo_file, ffield_file, control_file,
            spmd_handle->system, spmd_handle->control,
            spmd_handle->data, spmd_handle->workspace,
            spmd_handle->out_control );

    spmd_handle->system->N_max = (int) CEIL( SAFE_ZONE * spmd_handle->system->N );

    return (void *) spmd_handle;
}


/* Setup callback function to be run after each simulation step
 *
 * handle: pointer to wrapper struct with top-level data structures
 * callback: function pointer to attach for callback
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
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
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
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
                spmd_handle->realloc );

        spmd_handle->realloc = FALSE;

        /* compute f_0 */
        //if( control.restart == FALSE ) {
        Reset( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, spmd_handle->lists );

        Generate_Neighbor_Lists( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control );

        Compute_Forces( spmd_handle->system, spmd_handle->control, spmd_handle->data,
                spmd_handle->workspace, spmd_handle->lists, spmd_handle->out_control,
                spmd_handle->realloc );

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
            spmd_handle->callback( spmd_handle->system->N, spmd_handle->system->atoms,
                    spmd_handle->data );
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
                spmd_handle->callback( spmd_handle->system->N, spmd_handle->system->atoms,
                        spmd_handle->data );
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
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
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
 * qm_num_atoms: num. atoms in the QM region
 * qm_types: element types for QM atoms
 * qm_pos_x: x-coordinate of QM atom positions, in Angstroms
 * qm_pos_y: y-coordinate of QM atom positions, in Angstroms
 * qm_pos_z: z-coordinate of QM atom positions, in Angstroms
 * mm_num_atoms: num. atoms in the MM region
 * mm_types: element types for MM atoms
 * mm_pos_x: x-coordinate of MM atom positions, in Angstroms
 * mm_pos_y: y-coordinate of MM atom positions, in Angstroms
 * mm_pos_z: z-coordinate of MM atom positions, in Angstroms
 * mm_q: charge of MM atom, in Coulombs
 * sim_box: simulation box information, where the entries are
 *  - box length per dimension (3 entries)
 *  - angles per dimension (3 entries)
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int reset_qmmm_( const void * const handle,
        int qm_num_atoms, const int * const qm_types,
        const double * const qm_pos_x, const double * const qm_pos_y,
        const double * const qm_pos_z,
        int mm_num_atoms, const int * const mm_types,
        const double * const mm_pos_x, const double * const mm_pos_y,
        const double * const mm_pos_z, const double * const mm_q,
        const double * const sim_box,
        const char * const ffield_file, const char * const control_file )
{
    int i, ret;
    rvec x;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        /* close files used in previous simulation */
        if ( spmd_handle->output_enabled == TRUE )
        {
            Finalize_Out_Controls( spmd_handle->system, spmd_handle->control,
                    spmd_handle->workspace, spmd_handle->out_control );
        }

        spmd_handle->realloc = FALSE;
        spmd_handle->data->sim_id++;

        spmd_handle->system->N_qm = qm_num_atoms;
        spmd_handle->system->N_mm = mm_num_atoms;
        spmd_handle->system->N = qm_num_atoms + mm_num_atoms;

        PreAllocate_Space( spmd_handle->system, spmd_handle->control,
                spmd_handle->workspace, spmd_handle->system->N );

        Setup_Box( sim_box[0], sim_box[1], sim_box[2],
                sim_box[3], sim_box[4], sim_box[5],
                &spmd_handle->system->box );

        for ( i = 0; i < qm_num_atoms; ++i )
        {
            x[0] = qm_pos_x[i];
            x[1] = qm_pos_y[i];
            x[2] = qm_pos_z[i];

            Fit_to_Periodic_Box( &spmd_handle->system->box, x );

            spmd_handle->workspace->orig_id[i] = i + 1;
//            spmd_handle->system->atoms[i].type = Get_Atom_Type( &system->reax_param,
//                    element, sizeof(element) );
            spmd_handle->system->atoms[i].type = qm_types[i];
//            strncpy( spmd_handle->system->atoms[i].name, atom_name,
//                    sizeof(spmd_handle->system->atoms[i].name) - 1 );
//            spmd_handle->system->atoms[i].name[sizeof(spmd_handle->system->atoms[i].name) - 1] = '\0';
            rvec_Copy( spmd_handle->system->atoms[i].x, x );
            rvec_MakeZero( spmd_handle->system->atoms[i].v );
            rvec_MakeZero( spmd_handle->system->atoms[i].f );
            spmd_handle->system->atoms[i].q = 0.0;

//            mask[i] = 1;
        }

        for ( i = qm_num_atoms; i < qm_num_atoms + mm_num_atoms; ++i )
        {
            x[0] = mm_pos_x[i - qm_num_atoms];
            x[1] = mm_pos_y[i - qm_num_atoms];
            x[2] = mm_pos_z[i - qm_num_atoms];

            Fit_to_Periodic_Box( &spmd_handle->system->box, x );

            spmd_handle->workspace->orig_id[i] = i + 1;
//            spmd_handle->system->atoms[i].type = Get_Atom_Type( &system->reax_param,
//                    element, sizeof(element) );
            spmd_handle->system->atoms[i].type = mm_types[i - qm_num_atoms];
//            strncpy( spmd_handle->system->atoms[i].name, atom_name,
//                    sizeof(spmd_handle->system->atoms[i].name) - 1 );
//            spmd_handle->system->atoms[i].name[sizeof(spmd_handle->system->atoms[i].name) - 1] = '\0';
            rvec_Copy( spmd_handle->system->atoms[i].x, x );
            rvec_MakeZero( spmd_handle->system->atoms[i].v );
            rvec_MakeZero( spmd_handle->system->atoms[i].f );
            spmd_handle->system->atoms[i].q = mm_q[i - qm_num_atoms];

//            mask[i] = 0;
        }

        Read_Input_Files( NULL, ffield_file, control_file,
                spmd_handle->system, spmd_handle->control,
                spmd_handle->data, spmd_handle->workspace,
                spmd_handle->out_control );

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


/* Reset for the next simulation by parsing input files and triggering
 * reallocation if more space is needed
 *
 * handle: pointer to wrapper struct with top-level data structures
 * geo_file: file containing geometry info of the structure to simulate
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
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

        /* close files used in previous simulation */
        if ( spmd_handle->output_enabled == TRUE )
        {
            Finalize_Out_Controls( spmd_handle->system, spmd_handle->control,
                    spmd_handle->workspace, spmd_handle->out_control );
        }

        spmd_handle->realloc = FALSE;
        spmd_handle->data->sim_id++;

        Read_Input_Files( geo_file, ffield_file, control_file,
                spmd_handle->system, spmd_handle->control,
                spmd_handle->data, spmd_handle->workspace,
                spmd_handle->out_control );

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


/* Getter for atom positions in QMMM mode
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_pos_x: x-coordinate of QM atom positions, in Angstroms (allocated by caller)
 * qm_pos_y: y-coordinate of QM atom positions, in Angstroms (allocated by caller)
 * qm_pos_z: z-coordinate of QM atom positions, in Angstroms (allocated by caller)
 * mm_pos_x: x-coordinate of MM atom positions, in Angstroms (allocated by caller)
 * mm_pos_y: y-coordinate of MM atom positions, in Angstroms (allocated by caller)
 * mm_pos_z: z-coordinate of MM atom positions, in Angstroms (allocated by caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_atom_positions_qmmm_( const void * const handle, double * const qm_pos_x,
        double * const qm_pos_y, double * const qm_pos_z, double * const mm_pos_x,
        double * const mm_pos_y, double * const mm_pos_z )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        for ( i = 0; i < spmd_handle->system->N_qm; ++i )
        {
            qm_pos_x[i] = spmd_handle->system->atoms[i].x[0];
            qm_pos_y[i] = spmd_handle->system->atoms[i].x[1];
            qm_pos_z[i] = spmd_handle->system->atoms[i].x[2];
        }

        for ( i = spmd_handle->system->N_qm; i < spmd_handle->system->N; ++i )
        {
            mm_pos_x[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].x[0];
            mm_pos_y[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].x[1];
            mm_pos_z[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].x[2];
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom velocities in QMMM mode
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_vel_x: x-coordinate of QM atom velocities, in Angstroms / ps (allocated by caller)
 * qm_vel_y: y-coordinate of QM atom velocities, in Angstroms / ps (allocated by caller)
 * qm_vel_z: z-coordinate of QM atom velocities, in Angstroms / ps (allocated by caller)
 * mm_vel_x: x-coordinate of MM atom velocities, in Angstroms / ps (allocated by caller)
 * mm_vel_y: y-coordinate of MM atom velocities, in Angstroms / ps (allocated by caller)
 * mm_vel_z: z-coordinate of MM atom velocities, in Angstroms / ps (allocated by caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_atom_velocities_qmmm_( const void * const handle, double * const qm_vel_x,
        double * const qm_vel_y, double * const qm_vel_z, double * const mm_vel_x,
        double * const mm_vel_y, double * const mm_vel_z )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        for ( i = 0; i < spmd_handle->system->N_qm; ++i )
        {
            qm_vel_x[i] = spmd_handle->system->atoms[i].v[0];
            qm_vel_y[i] = spmd_handle->system->atoms[i].v[1];
            qm_vel_z[i] = spmd_handle->system->atoms[i].v[2];
        }

        for ( i = spmd_handle->system->N_qm; i < spmd_handle->system->N; ++i )
        {
            mm_vel_x[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].v[0];
            mm_vel_y[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].v[1];
            mm_vel_z[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].v[2];
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom forces in QMMM mode
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_f_x: x-coordinate of QM atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 * qm_f_y: y-coordinate of QM atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 * qm_f_z: z-coordinate of QM atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 * mm_f_x: x-coordinate of MM atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 * mm_f_y: y-coordinate of MM atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 * mm_f_z: z-coordinate of MM atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_atom_forces_qmmm_( const void * const handle, double * const qm_f_x,
        double * const qm_f_y, double * const qm_f_z, double * const mm_f_x,
        double * const mm_f_y, double * const mm_f_z )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        for ( i = 0; i < spmd_handle->system->N_qm; ++i )
        {
            qm_f_x[i] = spmd_handle->system->atoms[i].f[0];
            qm_f_y[i] = spmd_handle->system->atoms[i].f[1];
            qm_f_z[i] = spmd_handle->system->atoms[i].f[2];
        }

        for ( i = spmd_handle->system->N_qm; i < spmd_handle->system->N; ++i )
        {
            mm_f_x[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].f[0];
            mm_f_y[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].f[1];
            mm_f_z[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].f[2];
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom charges in QMMM mode
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_q: QM atom charges, in Coulombs (allocated by caller)
 * mm_q: MM atom charges, in Coulombs (allocated by caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_atom_charges_qmmm_( const void * const handle, double * const qm_q,
        double * const mm_q )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        for ( i = 0; i < spmd_handle->system->N_qm; ++i )
        {
            qm_q[i] = spmd_handle->system->atoms[i].q;
        }

        for ( i = spmd_handle->system->N_qm; i < spmd_handle->system->N; ++i )
        {
            mm_q[i - spmd_handle->system->N_qm] = spmd_handle->system->atoms[i].q;
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom positions
 *
 * handle: pointer to wrapper struct with top-level data structures
 * pos_x: x-coordinate of atom positions, in Angstroms (allocated by caller)
 * pos_y: y-coordinate of atom positions, in Angstroms (allocated by caller)
 * pos_z: z-coordinate of atom positions, in Angstroms (allocated by caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_atom_positions( const void * const handle, double * const pos_x,
        double * const pos_y, double * const pos_z )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        for ( i = 0; i < spmd_handle->system->N; ++i )
        {
            pos_x[i] = spmd_handle->system->atoms[i].x[0];
            pos_y[i] = spmd_handle->system->atoms[i].x[1];
            pos_z[i] = spmd_handle->system->atoms[i].x[2];
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom velocities
 *
 * handle: pointer to wrapper struct with top-level data structures
 * vel_x: x-coordinate of atom velocities, in Angstroms / ps (allocated by caller)
 * vel_y: y-coordinate of atom velocities, in Angstroms / ps (allocated by caller)
 * vel_z: z-coordinate of atom velocities, in Angstroms / ps (allocated by caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_atom_velocities( const void * const handle, double * const vel_x,
        double * const vel_y, double * const vel_z )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        for ( i = 0; i < spmd_handle->system->N; ++i )
        {
            vel_x[i] = spmd_handle->system->atoms[i].v[0];
            vel_y[i] = spmd_handle->system->atoms[i].v[1];
            vel_z[i] = spmd_handle->system->atoms[i].v[2];
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom forces
 *
 * handle: pointer to wrapper struct with top-level data structures
 * f_x: x-coordinate of atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 * f_y: y-coordinate of atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 * f_z: z-coordinate of atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_atom_forces( const void * const handle, double * const f_x,
        double * const f_y, double * const f_z )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        for ( i = 0; i < spmd_handle->system->N; ++i )
        {
            f_x[i] = spmd_handle->system->atoms[i].f[0];
            f_y[i] = spmd_handle->system->atoms[i].f[1];
            f_z[i] = spmd_handle->system->atoms[i].f[2];
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom charges
 *
 * handle: pointer to wrapper struct with top-level data structures
 * q: atom charges, in Coulombs (allocated by caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_atom_charges( const void * const handle, double * const q )
{
    int i, ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        for ( i = 0; i < spmd_handle->system->N; ++i )
        {
            q[i] = spmd_handle->system->atoms[i].q;
        }

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for system energies
 *
 * handle: pointer to wrapper struct with top-level data structures
 * e_pot: system potential energy, in kcal / mol (reference from caller)
 * e_kin: system kinetic energy, in kcal / mol (reference from caller)
 * e_tot: system total energy, in kcal / mol (reference from caller)
 * t_scalar: temperature scalar, in K (reference from caller)
 * vol: volume of the simulation box, in Angstroms^3 (reference from caller)
 * pres: average pressure, in K (reference from caller)
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int get_system_info( const void * const handle, double * const e_pot,
        double * const e_kin, double * const e_tot, double * const temp,
        double * const vol, double * const pres )
{
    int ret;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;

        *e_pot = spmd_handle->data->E_Pot;
        *e_kin = spmd_handle->data->E_Kin;
        *e_tot = spmd_handle->data->E_Tot;
        *temp = spmd_handle->data->therm.T;
        *vol = spmd_handle->system->box.volume;
        *pres = (spmd_handle->control->P[0] + spmd_handle->control->P[1]
                + spmd_handle->control->P[2]) / 3.0;

        ret = SPUREMD_SUCCESS;
    }

    return ret;
}


/* Setter for writing output to files
 *
 * handle: pointer to wrapper struct with top-level data structures
 * enabled: TRUE enables writing output to files, FALSE otherwise
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
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


/* Setter for simulation parameter values as defined in the input control file
 *
 * handle: pointer to wrapper struct with top-level data structures
 * control_keyword: keyword from the control file to set the value for
 * control_value: value to set
 *
 * returns: SPUREMD_SUCCESS upon success, SPUREMD_FAILURE otherwise
 */
int set_control_parameter( const void * const handle, const char * const keyword,
       const char ** const values )
{
    int ret, ret_;
    spuremd_handle *spmd_handle;

    ret = SPUREMD_FAILURE;

    if ( handle != NULL )
    {
        spmd_handle = (spuremd_handle*) handle;
        ret_ = Set_Control_Parameter( keyword, values, spmd_handle->control,
                spmd_handle->out_control );
        if ( ret_ == SUCCESS )
        {
            ret = SPUREMD_SUCCESS;
        }
    }

    return ret;
}
