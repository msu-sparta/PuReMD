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

#include "puremd.h"

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

#include <ctype.h>


/* Handles additional entire geometry calculations after
 * perturbing atom positions during a simulation step
 */
static void Post_Evolve( reax_system * const system, control_params * const control,
        simulation_data * const data, static_storage * const workspace,
        reax_list ** const lists, output_controls * const out_control )
{
    uint32_t i;
    rvec diff, cross;

    /* remove rotational and translational velocity of the center of mass */
    if ( control->ensemble != NVE && control->remove_CoM_vel > 0
            && data->step % control->remove_CoM_vel == 0 ) {
        /* compute velocity of the center of mass */
        Compute_Center_of_Mass( system, data );

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic,32) \
            private(i, diff, cross) shared(system, data)
#endif
        for ( i = 0; i < system->N; i++ ) {
            /* remove translational */
            rvec_ScaledAdd( system->atoms[i].v, -1.0, data->vcm );

            /* remove rotational */
            rvec_ScaledSum( diff, 1.0, system->atoms[i].x, -1.0, data->xcm );
            rvec_Cross( cross, data->avcm, diff );
            rvec_ScaledAdd( system->atoms[i].v, -1.0, cross );
        }
    }

    if ( control->ensemble == NVE ) {
        Compute_Kinetic_Energy( system, data );
    }

    Compute_Total_Energy( data );

    if ( control->compute_pressure == TRUE && control->ensemble != sNPT
            && control->ensemble != iNPT && control->ensemble != aNPT ) {
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
        output_controls * const out_control, bool reset )
{
    if ( ffield_file != NULL ) {
        Read_Force_Field( ffield_file, system, &system->reax_param );
    }

    if ( reset == FALSE || control_file != NULL ) {
        Set_Control_Defaults( system, control, out_control );
    }

    if ( control_file != NULL ) {
        Read_Control_File( control_file, control, out_control );
    }

    if ( reset == FALSE || control_file != NULL ) {
        Set_Control_Derived_Values( system, control );
    }

    if ( geo_file != NULL ) {
        if ( control->geo_format == CUSTOM ) {
            Read_Geo( geo_file, system, control, data, workspace );
        } else if ( control->geo_format == PDB ) {
            Read_PDB( geo_file, system, control, data, workspace );
        } else if ( control->geo_format == BGF ) {
            Read_BGF( geo_file, system, control, data, workspace );
        } else if ( control->geo_format == ASCII_RESTART ) {
            Read_ASCII_Restart( geo_file, system, control, data, workspace );
            control->restart = TRUE;
        } else if ( control->geo_format == BINARY_RESTART ) {
            Read_Binary_Restart( geo_file, system, control, data, workspace );
            control->restart = TRUE;
        } else {
            fprintf( stderr, "[ERROR] unknown geo file format. terminating!\n" );
            exit( INVALID_GEO );
        }
    }

#if defined(DEBUG_FOCUS)
    Print_Box( &system->box, stderr );
#endif
}


static void Allocate_Top_Level_Structs( puremd_handle ** handle )
{
    uint32_t i;

    /* top-level allocation */
    *handle = smalloc( sizeof(puremd_handle), __FILE__, __LINE__ );

    /* second-level allocations */
    (*handle)->system = smalloc( sizeof(reax_system), __FILE__, __LINE__ );
    (*handle)->control = smalloc( sizeof(control_params), __FILE__, __LINE__ );
    (*handle)->data = smalloc( sizeof(simulation_data), __FILE__, __LINE__ );
    (*handle)->workspace = smalloc( sizeof(static_storage), __FILE__, __LINE__ );
    (*handle)->lists = smalloc( sizeof(reax_list *) * LIST_N, __FILE__, __LINE__ );
    for ( i = 0; i < LIST_N; ++i ) {
        (*handle)->lists[i] = smalloc( sizeof(reax_list), __FILE__, __LINE__ );
    }
    (*handle)->out_control = smalloc( sizeof(output_controls), __FILE__, __LINE__ );
}


static void Initialize_Top_Level_Structs( puremd_handle * handle )
{
    uint32_t i;

    /* top-level initializations */
    handle->output_enabled = TRUE;
    handle->realloc = TRUE;
    handle->callback = NULL;
    handle->data->sim_id = 0;

    /* second-level initializations */
    handle->system->prealloc_allocated = FALSE;
    handle->system->allocated = FALSE;
    handle->system->ffield_params_allocated = FALSE;
    handle->system->g.allocated = FALSE;
    handle->system->N_max = 0;
    handle->system->max_num_molec_charge_constraints = 0;
    handle->system->num_molec_charge_constraints = 0;
    handle->system->max_num_custom_charge_constraints = 0;
    handle->system->num_custom_charge_constraints = 0;
    handle->system->max_num_custom_charge_constraint_entries = 0;
    handle->system->num_custom_charge_constraint_entries = 0;

    handle->workspace->allocated = FALSE;
    handle->workspace->H.allocated = FALSE;
    handle->workspace->H_full.allocated = FALSE;
    handle->workspace->H_sp.allocated = FALSE;
    handle->workspace->H_p.allocated = FALSE;
    handle->workspace->H_spar_patt.allocated = FALSE;
    handle->workspace->H_spar_patt_full.allocated = FALSE;
    handle->workspace->H_app_inv.allocated = FALSE;
    handle->workspace->L.allocated = FALSE;
    handle->workspace->U.allocated = FALSE;
    for ( i = 0; i < LIST_N; ++i ) {
        handle->lists[i]->allocated = FALSE;
    }
    handle->out_control->allocated = FALSE;
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
    puremd_handle *pmd_handle;

    Allocate_Top_Level_Structs( &pmd_handle );
    Initialize_Top_Level_Structs( pmd_handle );

    Read_Input_Files( geo_file, ffield_file, control_file,
            pmd_handle->system, pmd_handle->control,
            pmd_handle->data, pmd_handle->workspace,
            pmd_handle->out_control, FALSE );

    pmd_handle->system->N_max = (uint32_t) CEIL( SAFE_ZONE * pmd_handle->system->N );

    return (void *) pmd_handle;
}


/* Allocate top-level data structures and parse input files
 * for the first simulation
 *
 * num_atoms: num. atoms in this simulation
 * types: integer representation of atom element (type)
 *  NOTE: must match the 0-based index from section 2 in the ReaxFF parameter file
 * sim_box_info: simulation box information, where the entries are
 *  - box length per dimension (3 entries)
 *  - angles per dimension (3 entries)
 * pos: coordinates of atom positions (consecutively arranged), in Angstroms
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 */
void * setup2( int32_t num_atoms, const int32_t * const atom_type,
        const double * const pos, const double * const sim_box_info,
        const char * const ffield_file, const char * const control_file )
{
    uint32_t i, j;
    rvec x, dx;
    puremd_handle *pmd_handle;

    Allocate_Top_Level_Structs( &pmd_handle );
    Initialize_Top_Level_Structs( pmd_handle );

    /* override default */
    pmd_handle->output_enabled = FALSE;

    Read_Input_Files( NULL, ffield_file, control_file,
            pmd_handle->system, pmd_handle->control,
            pmd_handle->data, pmd_handle->workspace,
            pmd_handle->out_control, FALSE );

    pmd_handle->system->N = num_atoms;

    PreAllocate_Space( pmd_handle->system, pmd_handle->control,
            pmd_handle->workspace, (uint32_t) CEIL( SAFE_ZONE * pmd_handle->system->N ) );

    Setup_Box( sim_box_info[0], sim_box_info[1], sim_box_info[2],
            sim_box_info[3], sim_box_info[4], sim_box_info[5],
            &pmd_handle->system->box );

    if ( pmd_handle->control->periodic_boundaries == FALSE ) {
        dx[0] = pos[0];
        dx[1] = pos[1];
        dx[2] = pos[2];

        for ( i = 1; i < pmd_handle->system->N; ++i ) {
            for ( j = 0; j < 3; ++j ) {
                if ( pos[3 * i + j] < dx[j] ) {
                    dx[j] = pos[3 * i + j];
                }
            }
        }

        for ( i = 0; i < 3; ++i ) {
            dx[i] *= -1.0;
        }
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i, x) shared(pmd_handle, dx)
#endif
    for ( i = 0; i < pmd_handle->system->N; ++i ) {
        assert( atom_type[i] >= 0
                && atom_type[i] < pmd_handle->system->reax_param.num_atom_types );

        x[0] = pos[3 * i];
        x[1] = pos[3 * i + 1];
        x[2] = pos[3 * i + 2];

        if ( pmd_handle->control->periodic_boundaries == TRUE ) {
            Fit_To_Periodic_Box( &pmd_handle->system->box, x );
        } else {
            Fit_To_Non_Periodic_Box( x, dx );
        }

        pmd_handle->workspace->orig_id[i] = i + 1;
        pmd_handle->system->atoms[i].type = atom_type[i];
        strncpy( pmd_handle->system->atoms[i].name,
                pmd_handle->system->reax_param.sbp[atom_type[i]].name,
                sizeof(pmd_handle->system->atoms[i].name) );
        pmd_handle->system->atoms[i].name[sizeof(pmd_handle->system->atoms[i].name) - 1] = '\0';
        rvec_Copy( pmd_handle->system->atoms[i].x, x );
        rvec_MakeZero( pmd_handle->system->atoms[i].v );
        rvec_MakeZero( pmd_handle->system->atoms[i].f );
        pmd_handle->system->atoms[i].q = 0.0;
            
        /* check for dummy atom */
        if ( strncmp( pmd_handle->system->atoms[i].name, "X\0", 2 ) == 0 ) {
           pmd_handle->system->atoms[i].is_dummy = TRUE;
        } else {
            pmd_handle->system->atoms[i].is_dummy = FALSE;            
        }		
    }

    pmd_handle->system->N_max = (int32_t) CEIL( SAFE_ZONE * pmd_handle->system->N );

    return (void *) pmd_handle;
}


/* Setup callback function to be run after each simulation step
 *
 * handle: pointer to wrapper struct with top-level data structures
 * callback: function pointer to attach for callback
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t setup_callback( const void * const handle, const callback_function callback  )
{
    int32_t ret;
    puremd_handle *pmd_handle;


    ret = PUREMD_FAILURE;

    if ( handle != NULL && callback != NULL ) {
        pmd_handle = (puremd_handle*) handle;
        pmd_handle->callback = callback;
        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Run the simulation according to the prescribed parameters
 *
 * handle: pointer to wrapper struct with top-level data structures
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t simulate( const void * const handle )
{
    int32_t steps, ret, ret_forces, retries;
    evolve_function Evolve;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        Initialize( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                pmd_handle->workspace, pmd_handle->lists,
                pmd_handle->out_control, &Evolve,
                pmd_handle->output_enabled,
                pmd_handle->realloc );

        /* compute f_0 */
        //if( control.restart == FALSE ) {
        Reset( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                pmd_handle->workspace );

        ret_forces = Compute_Forces( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control,
                pmd_handle->realloc );

        if ( ret_forces != SUCCESS ) {
            Estimate_Storages( pmd_handle->system, pmd_handle->control,
                   pmd_handle->workspace, pmd_handle->lists,
                   pmd_handle->workspace->realloc.cm,
                   pmd_handle->workspace->realloc.bonds
                   || pmd_handle->workspace->realloc.hbonds );

            Reallocate_Part2( pmd_handle->system, pmd_handle->control,
                    pmd_handle->data, pmd_handle->workspace,
                    pmd_handle->lists );

            ret_forces = Compute_Forces( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                    pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control,
                    pmd_handle->realloc );

            if ( ret_forces != SUCCESS ) {
                fprintf( stderr, "[ERROR] Unrecoverable memory allocation issue (Compute_Forces). Terminating...\n" );
                exit( INVALID_GEO );
            }
        }

        Compute_Kinetic_Energy( pmd_handle->system, pmd_handle->data );

        if ( pmd_handle->control->compute_pressure == TRUE && pmd_handle->control->ensemble != sNPT
                && pmd_handle->control->ensemble != iNPT && pmd_handle->control->ensemble != aNPT ) {
            Compute_Pressure_Isotropic( pmd_handle->system, pmd_handle->control,
                    pmd_handle->data, pmd_handle->out_control );
        }

        Compute_Total_Energy( pmd_handle->data );

        if ( pmd_handle->output_enabled == TRUE ) {
            Output_Results( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                    pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control );
        }

        Check_Energy( pmd_handle->data );

        if ( pmd_handle->output_enabled == TRUE ) {
            if ( pmd_handle->out_control->write_steps > 0
                    && pmd_handle->data->step % pmd_handle->out_control->write_steps == 0 ) {
                Write_PDB( pmd_handle->system, pmd_handle->lists[BONDS], pmd_handle->data,
                        pmd_handle->control, pmd_handle->workspace, pmd_handle->out_control );
            }
        }

        if ( pmd_handle->callback != NULL ) {
            pmd_handle->callback( pmd_handle->system->N, pmd_handle->system->atoms,
                    pmd_handle->data );
        }
        //}

        ++pmd_handle->data->step;
        retries = 0;
        while ( pmd_handle->data->step <= pmd_handle->control->nsteps
                && retries < MAX_RETRIES ) {
            ret = SUCCESS;

            if ( pmd_handle->control->T_mode != 0 ) {
                Temperature_Control( pmd_handle->control, pmd_handle->data,
                        pmd_handle->out_control );
            }

            ret = Evolve( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                    pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control );

            if ( ret == SUCCESS ) {
                Post_Evolve( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                        pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control );
            }

            if ( ret == SUCCESS ) {
                if ( pmd_handle->output_enabled == TRUE ) {
                    Output_Results( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                            pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control );
                }

                Check_Energy( pmd_handle->data );

                if ( pmd_handle->output_enabled == TRUE ) {
                    steps = pmd_handle->data->step - pmd_handle->data->prev_steps;

                    Analysis( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                            pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control );

                    if ( pmd_handle->out_control->restart_freq > 0
                            && steps % pmd_handle->out_control->restart_freq == 0 ) {
                        Write_Restart( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                                pmd_handle->workspace, pmd_handle->out_control );
                    }

                    if ( pmd_handle->out_control->write_steps > 0
                            && steps % pmd_handle->out_control->write_steps == 0 ) {
                        Write_PDB( pmd_handle->system, pmd_handle->lists[BONDS], pmd_handle->data,
                                pmd_handle->control, pmd_handle->workspace, pmd_handle->out_control );
                    }
                }

                if ( pmd_handle->callback != NULL ) {
                    pmd_handle->callback( pmd_handle->system->N, pmd_handle->system->atoms,
                            pmd_handle->data );
                }

                ++pmd_handle->data->step;
                retries = 0;
            } else {
                ++retries;

#if defined(DEBUG_FOCUS)
                fprintf( stderr, "[INFO] p%d: retrying step %d...\n", system->my_rank, data->step );
#endif
            }
        }

        pmd_handle->data->timing.end = Get_Time( );
        pmd_handle->data->timing.elapsed = Get_Timing_Info( pmd_handle->data->timing.start );

        if ( pmd_handle->output_enabled == TRUE
                && pmd_handle->out_control->log_update_freq > 0 ) {
            fprintf( pmd_handle->out_control->out, "Total Simulation Time: %.2f secs\n",
                    pmd_handle->data->timing.elapsed );
        }

        pmd_handle->realloc = FALSE;
        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Deallocate all data structures post-simulation
 *
 * handle: pointer to wrapper struct with top-level data structures
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t cleanup( const void * const handle )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        Finalize( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control,
                pmd_handle->output_enabled, FALSE );

        sfree( pmd_handle->out_control, __FILE__, __LINE__ );
        for ( i = 0; i < LIST_N; ++i ) {
            sfree( pmd_handle->lists[i], __FILE__, __LINE__ );
        }
        sfree( pmd_handle->lists, __FILE__, __LINE__ );
        sfree( pmd_handle->workspace, __FILE__, __LINE__ );
        sfree( pmd_handle->data, __FILE__, __LINE__ );
        sfree( pmd_handle->control, __FILE__, __LINE__ );
        sfree( pmd_handle->system, __FILE__, __LINE__ );

        sfree( pmd_handle, __FILE__, __LINE__ );

        ret = PUREMD_SUCCESS;
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
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t reset( const void * const handle, const char * const geo_file,
        const char * const ffield_file, const char * const control_file )
{
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        /* close files used in previous simulation */
        if ( pmd_handle->output_enabled == TRUE ) {
            Finalize_Out_Controls( pmd_handle->system, pmd_handle->control,
                    pmd_handle->workspace, pmd_handle->out_control );
        }

        pmd_handle->realloc = FALSE;
        pmd_handle->data->sim_id++;

        Read_Input_Files( geo_file, ffield_file, control_file,
                pmd_handle->system, pmd_handle->control,
                pmd_handle->data, pmd_handle->workspace,
                pmd_handle->out_control, TRUE );

        pmd_handle->system->num_molec_charge_constraints = 0;
        pmd_handle->system->num_custom_charge_constraints = 0;
        pmd_handle->system->num_custom_charge_constraint_entries = 0;

        if ( pmd_handle->system->N > pmd_handle->system->N_max ) {
            /* deallocate everything which needs more space
             * (i.e., structures whose space is a function of the number of atoms),
             * except for data structures allocated while parsing input files */
            Finalize( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                    pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control,
                    pmd_handle->output_enabled, TRUE );

            pmd_handle->system->N_max = (int32_t) CEIL( SAFE_ZONE * pmd_handle->system->N );
            pmd_handle->realloc = TRUE;
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Allocate top-level data structures and parse input files
 * for the first simulation
 *
 * handle: pointer to wrapper struct with top-level data structures
 * num_atoms: num. atoms in this simulation
 * types: integer representation of atom element (type)
 *  NOTE: must match the 0-based index from section 2 in the ReaxFF parameter file
 * sim_box_info: simulation box information, where the entries are
 *  - box length per dimension (3 entries)
 *  - angles per dimension (3 entries)
 * pos: coordinates of atom positions (consecutively arranged), in Angstroms
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 */
int32_t reset2( const void * const handle, int32_t num_atoms,
        const int32_t * const atom_type, const double * const pos,
        const double * const sim_box_info, const char * const ffield_file,
        const char * const control_file )
{
    uint32_t i, j;
    int32_t ret;
    rvec x, dx;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        /* close files used in previous simulation */
        if ( pmd_handle->output_enabled == TRUE ) {
            Finalize_Out_Controls( pmd_handle->system, pmd_handle->control,
                    pmd_handle->workspace, pmd_handle->out_control );
        }

        pmd_handle->realloc = FALSE;
        pmd_handle->data->sim_id++;

        Read_Input_Files( NULL, ffield_file, control_file,
                pmd_handle->system, pmd_handle->control,
                pmd_handle->data, pmd_handle->workspace,
                pmd_handle->out_control, TRUE );

        pmd_handle->system->N = num_atoms;
        pmd_handle->system->num_molec_charge_constraints = 0;
        pmd_handle->system->num_custom_charge_constraints = 0;
        pmd_handle->system->num_custom_charge_constraint_entries = 0;

        if ( pmd_handle->system->prealloc_allocated == FALSE
                || pmd_handle->system->N > pmd_handle->system->N_max ) {
            PreAllocate_Space( pmd_handle->system, pmd_handle->control,
                    pmd_handle->workspace, (int32_t) CEIL( SAFE_ZONE * pmd_handle->system->N ) );
        }

        Setup_Box( sim_box_info[0], sim_box_info[1], sim_box_info[2],
                sim_box_info[3], sim_box_info[4], sim_box_info[5],
                &pmd_handle->system->box );

        if ( pmd_handle->control->periodic_boundaries == FALSE ) {
            dx[0] = pos[0];
            dx[1] = pos[1];
            dx[2] = pos[2];

            for ( i = 1; i < pmd_handle->system->N; ++i ) {
                for ( j = 0; j < 3; ++j ) {
                    if ( pos[3 * i + j] < dx[j] ) {
                        dx[j] = pos[3 * i + j];
                    }
                }
            }

            for ( i = 0; i < 3; ++i ) {
                dx[i] *= -1.0;
            }
        }

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic,32) private(i, x) shared(pmd_handle, dx)
#endif
        for ( i = 0; i < pmd_handle->system->N; ++i ) {
            assert( atom_type[i] >= 0
                    && atom_type[i] < pmd_handle->system->reax_param.num_atom_types );

            x[0] = pos[3 * i];
            x[1] = pos[3 * i + 1];
            x[2] = pos[3 * i + 2];

            if ( pmd_handle->control->periodic_boundaries == TRUE ) {
                Fit_To_Periodic_Box( &pmd_handle->system->box, x );
            } else {
                Fit_To_Non_Periodic_Box( x, dx );
            }

            pmd_handle->workspace->orig_id[i] = i + 1;
            pmd_handle->system->atoms[i].type = atom_type[i];
            strncpy( pmd_handle->system->atoms[i].name,
                    pmd_handle->system->reax_param.sbp[atom_type[i]].name,
                    sizeof(pmd_handle->system->atoms[i].name) );
            pmd_handle->system->atoms[i].name[sizeof(pmd_handle->system->atoms[i].name) - 1] = '\0';
            rvec_Copy( pmd_handle->system->atoms[i].x, x );
            rvec_MakeZero( pmd_handle->system->atoms[i].v );
            rvec_MakeZero( pmd_handle->system->atoms[i].f );
            pmd_handle->system->atoms[i].q = 0.0;
                
            /* check for dummy atom */
            if ( strncmp( pmd_handle->system->atoms[i].name, "X\0", 2 ) == 0 ) {
               pmd_handle->system->atoms[i].is_dummy = TRUE;
            } else {
                pmd_handle->system->atoms[i].is_dummy = FALSE;            
            }		
        }

        if ( pmd_handle->system->N > pmd_handle->system->N_max ) {
            /* deallocate everything which needs more space
             * (i.e., structures whose space is a function of the number of atoms),
             * except for data structures allocated while parsing input files */
            Finalize( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                    pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control,
                    pmd_handle->output_enabled, TRUE );

            pmd_handle->system->N_max = (int32_t) CEIL( SAFE_ZONE * pmd_handle->system->N );
            pmd_handle->realloc = TRUE;
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;    
}


/* Getter for atom positions
 *
 * handle: pointer to wrapper struct with top-level data structures
 * pos: coordinates of atom positions, in Angstroms (allocated by caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_atom_positions( const void * const handle, double * const pos )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        for ( i = 0; i < pmd_handle->system->N; ++i ) {
            pos[3 * i] = pmd_handle->system->atoms[i].x[0];
            pos[3 * i + 1] = pmd_handle->system->atoms[i].x[1];
            pos[3 * i + 2] = pmd_handle->system->atoms[i].x[2];
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom velocities
 *
 * handle: pointer to wrapper struct with top-level data structures
 * vel: coordinates of atom velocities, in Angstroms / ps (allocated by caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_atom_velocities( const void * const handle, double * const vel )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        for ( i = 0; i < pmd_handle->system->N; ++i ) {
            vel[3 * i] = pmd_handle->system->atoms[i].v[0];
            vel[3 * i + 1] = pmd_handle->system->atoms[i].v[1];
            vel[3 * i + 2] = pmd_handle->system->atoms[i].v[2];
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom forces
 *
 * handle: pointer to wrapper struct with top-level data structures
 * f: coordinates of atom forces, in Angstroms * Daltons / ps^2 (allocated by caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_atom_forces( const void * const handle, double * const f )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        for ( i = 0; i < pmd_handle->system->N; ++i ) {
            f[3 * i] = pmd_handle->system->atoms[i].f[0];
            f[3 * i + 1] = pmd_handle->system->atoms[i].f[1];
            f[3 * i + 2] = pmd_handle->system->atoms[i].f[2];
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom charges
 *
 * handle: pointer to wrapper struct with top-level data structures
 * q: atom charges, in Coulombs (allocated by caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_atom_charges( const void * const handle, double * const q )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        for ( i = 0; i < pmd_handle->system->N; ++i ) {
            q[i] = pmd_handle->system->atoms[i].q;
        }

        ret = PUREMD_SUCCESS;
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
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_system_info( const void * const handle, double * const e_pot,
        double * const e_kin, double * const e_tot, double * const temp,
        double * const vol, double * const pres )
{
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        if ( e_pot != NULL ) {
            *e_pot = pmd_handle->data->E_Pot;
        }

        if ( e_kin != NULL ) {
            *e_kin = pmd_handle->data->E_Kin;
        }

        if ( e_tot != NULL ) {
            *e_tot = pmd_handle->data->E_Tot;
        }

        if ( temp != NULL ) {
            *temp = pmd_handle->data->therm.T;
        }

        if ( vol != NULL ) {
            *vol = pmd_handle->system->box.volume;
        }

        if ( pres != NULL ) {
            *pres = (pmd_handle->control->P[0] + pmd_handle->control->P[1]
                    + pmd_handle->control->P[2]) / 3.0;
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for total energy
 *
 * handle: pointer to wrapper struct with top-level data structures
 * e_tot: system total energy, in kcal / mol (reference from caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_total_energy( const void * const handle, double * const e_tot )
{
    int32_t ret;

    ret = get_system_info( handle, e_tot, NULL, NULL, NULL, NULL, NULL );

    if ( ret == SUCCESS ) {
        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Setter for writing output to files
 *
 * handle: pointer to wrapper struct with top-level data structures
 * enabled: TRUE enables writing output to files, FALSE otherwise
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t set_output_enabled( const void * const handle, const int32_t enabled )
{
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;
        pmd_handle->output_enabled = enabled;
        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Setter for simulation parameter values as defined in the input control file
 *
 * handle: pointer to wrapper struct with top-level data structures
 * control_keyword: keyword from the control file to set the value for
 * control_value: value to set
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t set_control_parameter( const void * const handle, const char * const keyword,
       const char ** const values )
{
    int32_t ret, ret_;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        ret_ = Set_Control_Parameter( keyword, values, pmd_handle->control,
                pmd_handle->out_control );

        if ( ret_ == SUCCESS ) {
            ret = PUREMD_SUCCESS;
        }
    }

    return ret;
}


/* Setter for contiguous charge constraints
 *
 * handle: pointer to wrapper struct with top-level data structures
 * num_charge_constraint_contig: num. of contiguous charge constraints for charge model
 * charge_constraint_contig_start: starting atom num. (1-based) of atom group for a charge constraint
 * charge_constraint_contig_end: ending atom num. (1-based) of atom group for a charge constraint
 * charge_constraint_contig_value: charge constraint value for atom group
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t set_contiguous_charge_constraints( const void * const handle,
        int32_t num_charge_constraint_contig, const int32_t * const charge_constraint_contig_start,
        const int32_t * const charge_constraint_contig_end,
        const double * const charge_constraint_contig_value )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        pmd_handle->system->num_molec_charge_constraints = num_charge_constraint_contig;

        if ( pmd_handle->system->num_molec_charge_constraints
                > pmd_handle->system->max_num_molec_charge_constraints ) {
            if ( pmd_handle->system->max_num_molec_charge_constraints > 0 ) {
                sfree( pmd_handle->system->molec_charge_constraints,
                        __FILE__, __LINE__ );
                sfree( pmd_handle->system->molec_charge_constraint_ranges,
                        __FILE__, __LINE__ );
            }

            pmd_handle->system->molec_charge_constraints = smalloc(
                    sizeof(real) * pmd_handle->system->num_molec_charge_constraints,
                    __FILE__, __LINE__ );
            pmd_handle->system->molec_charge_constraint_ranges = smalloc(
                    sizeof(int32_t) * 2 * pmd_handle->system->num_molec_charge_constraints,
                    __FILE__, __LINE__ );

            pmd_handle->system->max_num_molec_charge_constraints
                = pmd_handle->system->num_molec_charge_constraints;
        }

        if ( pmd_handle->system->num_molec_charge_constraints > 0 ) {
            for ( i = 0; i < pmd_handle->system->num_molec_charge_constraints; ++i ) {
                pmd_handle->system->molec_charge_constraint_ranges[2 * i] = charge_constraint_contig_start[i];
                pmd_handle->system->molec_charge_constraint_ranges[2 * i + 1] = charge_constraint_contig_end[i];
            }

            for ( i = 0; i < pmd_handle->system->num_molec_charge_constraints; ++i ) {
                pmd_handle->system->molec_charge_constraints[i] = charge_constraint_contig_value[i];
            }
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Setter for custom charge constraints
 *
 * handle: pointer to wrapper struct with top-level data structures
 * num_charge_constraint_custom: num. of custom charge constraints for charge model
 * charge_constraint_custom_count: counts for each custom charge constraint
 * charge_constraint_custom_atom_index: atom indices (1-based) for custom charge constraints
 * charge_constraint_custom_coeff: coefficients for custom charge constraints
 * charge_constraint_custom_rhs: right-hand side (RHS) constants for custom charge constraints
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t set_custom_charge_constraints( const void * const handle,
        int32_t num_charge_constraint_custom,
        const int32_t * const charge_constraint_custom_count,
        const int32_t * const charge_constraint_custom_atom_index,
        const double * const charge_constraint_custom_coeff,
        const double * const charge_constraint_custom_rhs )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        pmd_handle->system->num_custom_charge_constraints = num_charge_constraint_custom;

        if ( pmd_handle->system->num_custom_charge_constraints
                > pmd_handle->system->max_num_custom_charge_constraints ) {
            if ( pmd_handle->system->max_num_custom_charge_constraints > 0 ) {
                sfree( pmd_handle->system->custom_charge_constraint_count,
                        __FILE__, __LINE__ );
                sfree( pmd_handle->system->custom_charge_constraint_start,
                        __FILE__, __LINE__ );
                sfree( pmd_handle->system->custom_charge_constraint_rhs,
                        __FILE__, __LINE__ );
            }

            pmd_handle->system->custom_charge_constraint_count = smalloc(
                    sizeof(int32_t) * pmd_handle->system->num_custom_charge_constraints,
                    __FILE__, __LINE__ );
            pmd_handle->system->custom_charge_constraint_start = smalloc(
                    sizeof(int32_t) * (pmd_handle->system->num_custom_charge_constraints + 1),
                    __FILE__, __LINE__ );
            pmd_handle->system->custom_charge_constraint_rhs = smalloc(
                    sizeof(real) * pmd_handle->system->num_custom_charge_constraints,
                    __FILE__, __LINE__ );

            pmd_handle->system->max_num_custom_charge_constraints
                = pmd_handle->system->num_custom_charge_constraints;
        }

        pmd_handle->system->num_custom_charge_constraint_entries = 0;

        if ( pmd_handle->system->num_custom_charge_constraints > 0 ) {
            for ( i = 0; i < pmd_handle->system->num_custom_charge_constraints; ++i ) {
                pmd_handle->system->custom_charge_constraint_count[i] = charge_constraint_custom_count[i];
                pmd_handle->system->custom_charge_constraint_start[i] = (i == 0 ? 0 :
                        pmd_handle->system->custom_charge_constraint_start[i - 1] + charge_constraint_custom_count[i - 1]);
                pmd_handle->system->num_custom_charge_constraint_entries += charge_constraint_custom_count[i];
                pmd_handle->system->custom_charge_constraint_rhs[i] = charge_constraint_custom_rhs[i];
            }
            pmd_handle->system->custom_charge_constraint_start[pmd_handle->system->num_custom_charge_constraints]
                = pmd_handle->system->custom_charge_constraint_start[pmd_handle->system->num_custom_charge_constraints - 1]
                + charge_constraint_custom_count[pmd_handle->system->num_custom_charge_constraints - 1];
        }

        if ( pmd_handle->system->num_custom_charge_constraint_entries
                > pmd_handle->system->max_num_custom_charge_constraint_entries ) {
            if ( pmd_handle->system->max_num_custom_charge_constraint_entries > 0 ) {
                sfree( pmd_handle->system->custom_charge_constraint_atom_index,
                        __FILE__, __LINE__ );
                sfree( pmd_handle->system->custom_charge_constraint_coeff,
                        __FILE__, __LINE__ );
            }

            pmd_handle->system->custom_charge_constraint_atom_index = smalloc(
                    sizeof(int32_t) * pmd_handle->system->num_custom_charge_constraint_entries,
                    __FILE__, __LINE__ );
            pmd_handle->system->custom_charge_constraint_coeff = smalloc(
                    sizeof(real) * pmd_handle->system->num_custom_charge_constraint_entries,
                    __FILE__, __LINE__ );

            pmd_handle->system->max_num_custom_charge_constraint_entries
                = pmd_handle->system->num_custom_charge_constraint_entries;
        }

        if ( pmd_handle->system->num_custom_charge_constraint_entries > 0 ) {
            for ( i = 0; i < pmd_handle->system->num_custom_charge_constraint_entries; ++i ) {
                pmd_handle->system->custom_charge_constraint_atom_index[i] = charge_constraint_custom_atom_index[i];
            }

            for ( i = 0; i < pmd_handle->system->num_custom_charge_constraint_entries; ++i ) {
                pmd_handle->system->custom_charge_constraint_coeff[i] = charge_constraint_custom_coeff[i];
            }
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


#if defined(QMMM)
/* Allocate top-level data structures and parse input files
 * for the first simulation
 *
 * qm_num_atoms: num. atoms in the QM region
 * qm_symbols: element types for QM atoms
 * qm_pos: coordinates of QM atom positions (consecutively arranged), in Angstroms
 * mm_num_atoms: num. atoms in the MM region
 * mm_symbols: element types for MM atoms
 * mm_pos_q: coordinates and charges of MM atom positions (consecutively arranged), in Angstroms / Coulombs
 * sim_box_info: simulation box information, where the entries are
 *  - box length per dimension (3 entries)
 *  - angles per dimension (3 entries)
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 */
void * setup_qmmm( int32_t qm_num_atoms, const char * const qm_symbols,
        const double * const qm_pos, int32_t mm_num_atoms, const char * const mm_symbols,
        const double * const mm_pos_q, const double * const sim_box_info,
        const char * const ffield_file, const char * const control_file )
{
    uint32_t i, j;
    char element[3];
    rvec x, dx;
    puremd_handle *pmd_handle;

    Allocate_Top_Level_Structs( &pmd_handle );
    Initialize_Top_Level_Structs( pmd_handle );

    /* override default */
    pmd_handle->output_enabled = FALSE;

    Read_Input_Files( NULL, ffield_file, control_file,
            pmd_handle->system, pmd_handle->control,
            pmd_handle->data, pmd_handle->workspace,
            pmd_handle->out_control, FALSE );

    pmd_handle->system->N_qm = qm_num_atoms;
    pmd_handle->system->N_mm = mm_num_atoms;
    pmd_handle->system->N = pmd_handle->system->N_qm + pmd_handle->system->N_mm;

    PreAllocate_Space( pmd_handle->system, pmd_handle->control,
            pmd_handle->workspace, (int32_t) CEIL( SAFE_ZONE * pmd_handle->system->N ) );

    Setup_Box( sim_box_info[0], sim_box_info[1], sim_box_info[2],
            sim_box_info[3], sim_box_info[4], sim_box_info[5],
            &pmd_handle->system->box );

    if ( handle->control->periodic_boundaries == FALSE ) {
        dx[0] = pos[0];
        dx[1] = pos[1];
        dx[2] = pos[2];

        for ( i = 1; i < handle->system->N_qm; ++i ) {
            for ( j = 0; j < 3; ++j ) {
                if ( qm_pos[3 * i + j] < dx[j] ) {
                    dx[j] = qm_pos[3 * i + j];
                }
            }
        }

        for ( i = 0; i < handle->system->N - handle->system->N_qm; ++i ) {
            for ( j = 0; j < 3; ++j ) {
                if ( mm_pos_q[4 * i + j] < dx[j] ) {
                    dx[j] = mm_pos_q[4 * i + j];
                }
            }
        }

        for ( i = 0; i < 3; ++i ) {
            dx[i] *= -1.0;
        }
    }

    element[2] = '\0';

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i, x, element) shared(pmd_handle, dx)
#endif
    for ( i = 0; i < pmd_handle->system->N_qm; ++i ) {
        x[0] = qm_pos[3 * i];
        x[1] = qm_pos[3 * i + 1];
        x[2] = qm_pos[3 * i + 2];

        if ( handle->control->periodic_boundaries == TRUE ) {
            Fit_To_Periodic_Box( &pmd_handle->system->box, x );
        } else {
            Fit_To_Non_Periodic_Box( x, dx );
        }

        pmd_handle->workspace->orig_id[i] = i + 1;
        element[0] = toupper( qm_symbols[2 * i] );
        element[1] = toupper( qm_symbols[2 * i + 1] );
        Trim_Spaces( element, sizeof(element) );
        pmd_handle->system->atoms[i].type = Get_Atom_Type( &pmd_handle->system->reax_param,
                element, sizeof(element) );
        strncpy( pmd_handle->system->atoms[i].name, element,
                sizeof(pmd_handle->system->atoms[i].name) - 1 );
        pmd_handle->system->atoms[i].name[sizeof(pmd_handle->system->atoms[i].name) - 1] = '\0';
        rvec_Copy( pmd_handle->system->atoms[i].x, x );
        rvec_MakeZero( pmd_handle->system->atoms[i].v );
        rvec_MakeZero( pmd_handle->system->atoms[i].f );
        pmd_handle->system->atoms[i].q = 0.0;
        pmd_handle->system->atoms[i].q_init = 0.0;
        pmd_handle->system->atoms[i].qmmm_mask = TRUE;
            
        /* check for dummy atom */
        if ( strncmp( element, "X\0", 2 ) == 0 ) {
           pmd_handle->system->atoms[i].is_dummy = TRUE;
        } else {
            pmd_handle->system->atoms[i].is_dummy = FALSE;            
        }		
    }

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i, x) firstprivate(element) shared(pmd_handle, dx)
#endif
    for ( i = pmd_handle->system->N_qm; i < pmd_handle->system->N; ++i ) {
        x[0] = mm_pos_q[4 * (i - pmd_handle->system->N_qm)];
        x[1] = mm_pos_q[4 * (i - pmd_handle->system->N_qm) + 1];
        x[2] = mm_pos_q[4 * (i - pmd_handle->system->N_qm) + 2];

        if ( handle->control->periodic_boundaries == TRUE ) {
            Fit_To_Periodic_Box( &pmd_handle->system->box, x );
        } else {
            Fit_To_Non_Periodic_Box( x, dx );
        }

        pmd_handle->workspace->orig_id[i] = i + 1;
        element[0] = toupper( mm_symbols[2 * (i - pmd_handle->system->N_qm)] );
        element[1] = toupper( mm_symbols[2 * (i - pmd_handle->system->N_qm) + 1] );
        Trim_Spaces( element, sizeof(element) );
        pmd_handle->system->atoms[i].type = Get_Atom_Type( &pmd_handle->system->reax_param,
                element, sizeof(element) );
        strncpy( pmd_handle->system->atoms[i].name, element,
                sizeof(pmd_handle->system->atoms[i].name) - 1 );
        pmd_handle->system->atoms[i].name[sizeof(pmd_handle->system->atoms[i].name) - 1] = '\0';
        rvec_Copy( pmd_handle->system->atoms[i].x, x );
        rvec_MakeZero( pmd_handle->system->atoms[i].v );
        rvec_MakeZero( pmd_handle->system->atoms[i].f );
        pmd_handle->system->atoms[i].q = mm_pos_q[4 * (i - pmd_handle->system->N_qm) + 3];
        pmd_handle->system->atoms[i].q_init = mm_pos_q[4 * (i - pmd_handle->system->N_qm) + 3];
        pmd_handle->system->atoms[i].qmmm_mask = FALSE;
            
        /* check for dummy atom */
        if ( strncmp( element, "X\0", 2 ) == 0 ) {
           pmd_handle->system->atoms[i].is_dummy = TRUE;
        } else {
            pmd_handle->system->atoms[i].is_dummy = FALSE;            
        }		
    }

    pmd_handle->system->N_max = (int32_t) CEIL( SAFE_ZONE * pmd_handle->system->N );

    return (void *) pmd_handle;
}


/* Reset for the next simulation by parsing input files and triggering
 * reallocation if more space is needed
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_num_atoms: num. atoms in the QM region
 * qm_symbols: element types for QM atoms
 * qm_pos: coordinates of QM atom positions (consecutively arranged), in Angstroms
 * mm_num_atoms: num. atoms in the MM region
 * mm_symbols: element types for MM atoms
 * mm_pos_q: coordinates and charges of MM atom positions (consecutively arranged), in Angstroms / Coulombs
 * sim_box_info: simulation box information, where the entries are
 *  - box length per dimension (3 entries)
 *  - angles per dimension (3 entries)
 * ffield_file: file containing force field parameters
 * control_file: file containing simulation parameters
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t reset_qmmm( const void * const handle, int32_t qm_num_atoms,
        const char * const qm_symbols, const double * const qm_pos,
        int32_t mm_num_atoms, const char * const mm_symbols,
        const double * const mm_pos_q, const double * const sim_box_info,
        const char * const ffield_file, const char * const control_file )
{
    uint32_t i, j;
    int32_t ret;
    char element[3];
    rvec x, dx;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        /* close files used in previous simulation */
        if ( pmd_handle->output_enabled == TRUE ) {
            Finalize_Out_Controls( pmd_handle->system, pmd_handle->control,
                    pmd_handle->workspace, pmd_handle->out_control );
        }

        pmd_handle->realloc = FALSE;
        pmd_handle->data->sim_id++;

        Read_Input_Files( NULL, ffield_file, control_file,
                pmd_handle->system, pmd_handle->control,
                pmd_handle->data, pmd_handle->workspace,
                pmd_handle->out_control, TRUE );

        pmd_handle->system->N_qm = qm_num_atoms;
        pmd_handle->system->N_mm = mm_num_atoms;
        pmd_handle->system->N = pmd_handle->system->N_qm + pmd_handle->system->N_mm;
        pmd_handle->system->num_molec_charge_constraints = 0;
        pmd_handle->system->num_custom_charge_constraints = 0;
        pmd_handle->system->num_custom_charge_constraint_entries = 0;

        if ( pmd_handle->system->prealloc_allocated == FALSE
                || pmd_handle->system->N > pmd_handle->system->N_max ) {
            PreAllocate_Space( pmd_handle->system, pmd_handle->control,
                    pmd_handle->workspace, (int32_t) CEIL( SAFE_ZONE * pmd_handle->system->N ) );
        }

        Setup_Box( sim_box_info[0], sim_box_info[1], sim_box_info[2],
                sim_box_info[3], sim_box_info[4], sim_box_info[5],
                &pmd_handle->system->box );

        if ( handle->control->periodic_boundaries == FALSE ) {
            dx[0] = pos[0];
            dx[1] = pos[1];
            dx[2] = pos[2];

            for ( i = 1; i < handle->system->N_qm; ++i ) {
                for ( j = 0; j < 3; ++j ) {
                    if ( qm_pos[3 * i + j] < dx[j] ) {
                        dx[j] = qm_pos[3 * i + j];
                    }
                }
            }

            for ( i = 0; i < handle->system->N - handle->system->N_qm; ++i ) {
                for ( j = 0; j < 3; ++j ) {
                    if ( mm_pos_q[4 * i + j] < dx[j] ) {
                        dx[j] = mm_pos_q[4 * i + j];
                    }
                }
            }

            for ( i = 0; i < 3; ++i ) {
                dx[i] *= -1.0;
            }
        }

        element[2] = '\0';

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic,32) private(i, x) firstprivate(element) shared(pmd_handle, dx)
#endif
        for ( i = 0; i < pmd_handle->system->N_qm; ++i ) {
            x[0] = qm_pos[3 * i];
            x[1] = qm_pos[3 * i + 1];
            x[2] = qm_pos[3 * i + 2];

            if ( handle->control->periodic_boundaries == TRUE ) {
                Fit_To_Periodic_Box( &pmd_handle->system->box, x );
            } else {
                Fit_To_Non_Periodic_Box( x, dx );
            }

            pmd_handle->workspace->orig_id[i] = i + 1;
            element[0] = toupper( qm_symbols[2 * i] );
            element[1] = toupper( qm_symbols[2 * i + 1] );
            Trim_Spaces( element, sizeof(element) );
            pmd_handle->system->atoms[i].type = Get_Atom_Type( &pmd_handle->system->reax_param,
                    element, sizeof(element) );
            strncpy( pmd_handle->system->atoms[i].name, element,
                    sizeof(pmd_handle->system->atoms[i].name) - 1 );
            pmd_handle->system->atoms[i].name[sizeof(pmd_handle->system->atoms[i].name) - 1] = '\0';
            rvec_Copy( pmd_handle->system->atoms[i].x, x );
            rvec_MakeZero( pmd_handle->system->atoms[i].v );
            rvec_MakeZero( pmd_handle->system->atoms[i].f );
            pmd_handle->system->atoms[i].q = 0.0;
            pmd_handle->system->atoms[i].q_init = 0.0;
            pmd_handle->system->atoms[i].qmmm_mask = TRUE;
            
            /* check for dummy atom */
            if ( strncmp( element, "X\0", 2 ) == 0 ) {
               pmd_handle->system->atoms[i].is_dummy = TRUE;
            } else {
                pmd_handle->system->atoms[i].is_dummy = FALSE;            
            }		
        }

#if defined(_OPENMP)
        #pragma omp parallel for schedule(dynamic,32) private(i, x) firstprivate(element) shared(pmd_handle, dx)
#endif
        for ( i = pmd_handle->system->N_qm; i < pmd_handle->system->N; ++i ) {
            x[0] = mm_pos_q[4 * (i - pmd_handle->system->N_qm)];
            x[1] = mm_pos_q[4 * (i - pmd_handle->system->N_qm) + 1];
            x[2] = mm_pos_q[4 * (i - pmd_handle->system->N_qm) + 2];

            if ( handle->control->periodic_boundaries == TRUE ) {
                Fit_To_Periodic_Box( &pmd_handle->system->box, x );
            } else {
                Fit_To_Non_Periodic_Box( x, dx );
            }

            pmd_handle->workspace->orig_id[i] = i + 1;
            element[0] = toupper( mm_symbols[2 * (i - pmd_handle->system->N_qm)] );
            element[1] = toupper( mm_symbols[2 * (i - pmd_handle->system->N_qm) + 1] );
            Trim_Spaces( element, sizeof(element) );
            pmd_handle->system->atoms[i].type = Get_Atom_Type( &pmd_handle->system->reax_param,
                    element, sizeof(element) );
            strncpy( pmd_handle->system->atoms[i].name, element,
                    sizeof(pmd_handle->system->atoms[i].name) - 1 );
            pmd_handle->system->atoms[i].name[sizeof(pmd_handle->system->atoms[i].name) - 1] = '\0';
            rvec_Copy( pmd_handle->system->atoms[i].x, x );
            rvec_MakeZero( pmd_handle->system->atoms[i].v );
            rvec_MakeZero( pmd_handle->system->atoms[i].f );
            pmd_handle->system->atoms[i].q = mm_pos_q[4 * (i - pmd_handle->system->N_qm) + 3];
            pmd_handle->system->atoms[i].q_init = mm_pos_q[4 * (i - pmd_handle->system->N_qm) + 3];
            pmd_handle->system->atoms[i].qmmm_mask = FALSE;
            
            /* check for dummy atom */
            if ( strncmp( element, "X\0", 2 ) == 0 ) {
               pmd_handle->system->atoms[i].is_dummy = TRUE;
            } else {
                pmd_handle->system->atoms[i].is_dummy = FALSE;            
            }		
        }

        if ( pmd_handle->system->N > pmd_handle->system->N_max ) {
            /* deallocate everything which needs more space
             * (i.e., structures whose space is a function of the number of atoms),
             * except for data structures allocated while parsing input files */
            Finalize( pmd_handle->system, pmd_handle->control, pmd_handle->data,
                    pmd_handle->workspace, pmd_handle->lists, pmd_handle->out_control,
                    pmd_handle->output_enabled, TRUE );

            pmd_handle->system->N_max = (int32_t) CEIL( SAFE_ZONE * pmd_handle->system->N );
            pmd_handle->realloc = TRUE;
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom positions in QMMM mode
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_pos: coordinates of QM atom positions (consecutively arranged), in Angstroms (allocated by caller)
 * mm_pos: coordinates of MM atom positions (consecutively arranged), in Angstroms (allocated by caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_atom_positions_qmmm( const void * const handle, double * const qm_pos,
        double * const mm_pos )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        if ( qm_pos != NULL ) {
            for ( i = 0; i < pmd_handle->system->N_qm; ++i ) {
                qm_pos[3 * i] = pmd_handle->system->atoms[i].x[0];
                qm_pos[3 * i + 1] = pmd_handle->system->atoms[i].x[1];
                qm_pos[3 * i + 2] = pmd_handle->system->atoms[i].x[2];
            }
        }

        if ( mm_pos != NULL ) {
            for ( i = pmd_handle->system->N_qm; i < pmd_handle->system->N; ++i ) {
                mm_pos[3 * (i - pmd_handle->system->N_qm)] = pmd_handle->system->atoms[i].x[0];
                mm_pos[3 * (i - pmd_handle->system->N_qm) + 1] = pmd_handle->system->atoms[i].x[1];
                mm_pos[3 * (i - pmd_handle->system->N_qm) + 2] = pmd_handle->system->atoms[i].x[2];
            }
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom velocities in QMMM mode
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_vel: coordinates of QM atom velocities (consecutively arranged), in Angstroms / ps (allocated by caller)
 * mm_vel: coordinates of MM atom velocities (consecutively arranged), in Angstroms / ps (allocated by caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_atom_velocities_qmmm( const void * const handle, double * const qm_vel,
        double * const mm_vel )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        if ( qm_vel != NULL ) {
            for ( i = 0; i < pmd_handle->system->N_qm; ++i ) {
                qm_vel[3 * i] = pmd_handle->system->atoms[i].v[0];
                qm_vel[3 * i + 1] = pmd_handle->system->atoms[i].v[1];
                qm_vel[3 * i + 2] = pmd_handle->system->atoms[i].v[2];
            }
        }

        if ( mm_vel != NULL ) {
            for ( i = pmd_handle->system->N_qm; i < pmd_handle->system->N; ++i ) {
                mm_vel[3 * (i - pmd_handle->system->N_qm)] = pmd_handle->system->atoms[i].v[0];
                mm_vel[3 * (i - pmd_handle->system->N_qm) + 1] = pmd_handle->system->atoms[i].v[1];
                mm_vel[3 * (i - pmd_handle->system->N_qm) + 2] = pmd_handle->system->atoms[i].v[2];
            }
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom forces in QMMM mode
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_f: coordinates of QM atom forces (consecutively arranged), in Angstroms * Daltons / ps^2 (allocated by caller)
 * mm_f: coordinates of MM atom forces (consecutively arranged), in Angstroms * Daltons / ps^2 (allocated by caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_atom_forces_qmmm( const void * const handle, double * const qm_f,
        double * const mm_f )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        if ( qm_f != NULL ) {
            for ( i = 0; i < pmd_handle->system->N_qm; ++i ) {
                qm_f[3 * i] = pmd_handle->system->atoms[i].f[0];
                qm_f[3 * i + 1] = pmd_handle->system->atoms[i].f[1];
                qm_f[3 * i + 2] = pmd_handle->system->atoms[i].f[2];
            }
        }

        if ( mm_f != NULL ) {
            for ( i = pmd_handle->system->N_qm; i < pmd_handle->system->N; ++i ) {
                mm_f[3 * (i - pmd_handle->system->N_qm)] = pmd_handle->system->atoms[i].f[0];
                mm_f[3 * (i - pmd_handle->system->N_qm) + 1] = pmd_handle->system->atoms[i].f[1];
                mm_f[3 * (i - pmd_handle->system->N_qm) + 2] = pmd_handle->system->atoms[i].f[2];
            }
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}


/* Getter for atom charges in QMMM mode
 *
 * handle: pointer to wrapper struct with top-level data structures
 * qm_q: QM atom charges, in Coulombs (allocated by caller)
 * mm_q: MM atom charges, in Coulombs (allocated by caller)
 *
 * returns: PUREMD_SUCCESS upon success, PUREMD_FAILURE otherwise
 */
int32_t get_atom_charges_qmmm( const void * const handle, double * const qm_q,
        double * const mm_q )
{
    uint32_t i;
    int32_t ret;
    puremd_handle *pmd_handle;

    ret = PUREMD_FAILURE;

    if ( handle != NULL ) {
        pmd_handle = (puremd_handle*) handle;

        if ( qm_q != NULL ) {
            for ( i = 0; i < pmd_handle->system->N_qm; ++i ) {
                qm_q[i] = pmd_handle->system->atoms[i].q;
            }
        }

        if ( mm_q != NULL ) {
            for ( i = pmd_handle->system->N_qm; i < pmd_handle->system->N; ++i ) {
                mm_q[i - pmd_handle->system->N_qm] = pmd_handle->system->atoms[i].q;
            }
        }

        ret = PUREMD_SUCCESS;
    }

    return ret;
}
#endif
