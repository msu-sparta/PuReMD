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

#include "init_md.h"

#include "allocate.h"
#include "box.h"
#include "forces.h"
#include "grid.h"
#include "integrate.h"
#if defined(TEST_FORCES)
  #include "io_tools.h"
#endif
#include "neighbors.h"
#include "list.h"
#include "lookup.h"
#include "reset_tools.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


static void Generate_Initial_Velocities( reax_system *system,
        control_params *control, real T )
{
    int i;
    real scale, norm;

    if ( T <= 0.1 || control->random_vel == FALSE )
    {
        /* warnings if conflicts between initial temperature and control file parameter */
        if ( control->random_vel == TRUE )
        {
            fprintf( stderr, "[ERROR] conflicting control file parameters\n" );
            fprintf( stderr, "[INFO] random_vel = 1 and small initial temperature (t_init = %f)\n", T );
            fprintf( stderr, "[INFO] set random_vel = 0 to resolve this (atom initial velocites set to zero)\n" );
            exit( INVALID_INPUT );
        }
        else if ( T > 0.1 )
        {
            fprintf( stderr, "[ERROR] conflicting control file paramters\n" );
            fprintf( stderr, "[INFO] random_vel = 0 and large initial temperature (t_init = %f)\n", T );
            fprintf( stderr, "[INFO] set random_vel = 1 to resolve this (random atom initial velocites according to t_init)\n" );
            exit( INVALID_INPUT );
        }

        for ( i = 0; i < system->N; i++ )
        {
            rvec_MakeZero( system->atoms[i].v );
        }
    }
    else
    {
        if ( T <= 0.0 )
        {
            fprintf( stderr, "[ERROR] random atom initial velocities specified with invalid temperature (%f). Terminating...\n",
                  T );
            exit( INVALID_INPUT );
        }

        for ( i = 0; i < system->N; i++ )
        {
            rvec_Random( system->atoms[i].v );

            norm = rvec_Norm_Sqr( system->atoms[i].v );
            scale = SQRT( system->reax_param.sbp[ system->atoms[i].type ].mass
                    * norm / (3.0 * K_B * T) );

            rvec_Scale( system->atoms[i].v, 1.0 / scale, system->atoms[i].v );

#if defined(DEBUG_FOCUS)
            fprintf( stderr, "[INFO] atom %d, scale = %f, v = (%f, %f, %f)\n",
                    i, scale,
                    system->atoms[i].v[0],
                    system->atoms[i].v[1],
                    system->atoms[i].v[2] );
#endif
        }
    }
}


static void Init_System( reax_system *system, control_params *control,
        simulation_data *data )
{
    int i;
    rvec dx;

    if ( control->restart == FALSE )
    {
        Reset_Atomic_Forces( system );
    }

    Compute_Total_Mass( system, data );
    Compute_Center_of_Mass( system, data );

    /* just fit the atoms to the periodic box */
    if ( control->reposition_atoms == 0 )
    {
        rvec_MakeZero( dx );
    }
    /* put the center of mass to the center of the box */
    else if ( control->reposition_atoms == 1 )
    {
        rvec_Scale( dx, 0.5, system->box.box_norms );
        rvec_ScaledAdd( dx, -1.0, data->xcm );
    }
    /* put the center of mass to the origin */
    else if ( control->reposition_atoms == 2 )
    {
        rvec_Scale( dx, -1.0, data->xcm );
    }
    else
    {
        fprintf( stderr, "[ERROR] Unknown option for reposition_atoms (%d). Terminating...\n",
              control->reposition_atoms );
        exit( UNKNOWN_OPTION );
    }

    for ( i = 0; i < system->N; ++i )
    {
        /* re-map the atom positions to fall within the simulation box,
         * where the corners of the box are (0,0,0) and (d_x, d_y, d_z)
         * with d_i being the box length along dimension i */
        Update_Atom_Position_Periodic( system->atoms[i].x, dx,
                system->atoms[i].rel_map, &system->box );

        /* zero out rel_map (which was set by the above function) */
        ivec_MakeZero( system->atoms[i].rel_map );
    }

    /* Initialize velocities so that desired init T can be attained */
    if ( control->restart == FALSE
            || (control->restart == TRUE && control->random_vel == TRUE) )
    {
        Generate_Initial_Velocities( system, control, control->T_init );
    }

    Setup_Grid( system );
}


static void Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data, output_controls *out_control,
        evolve_function *Evolve, int realloc )
{
#if defined(_OPENMP)
    if ( realloc == TRUE && (control->ensemble == sNPT || control->ensemble == iNPT
            || control->ensemble == aNPT || control->compute_pressure == TRUE) )
    {
        data->press_local = smalloc( sizeof( rtensor ) * control->num_threads,
               "Init_Simulation_Data::data->press_local" );
    }
#endif

    Reset_Pressures( control, data );
    Reset_Energies( data );

    data->therm.T = 0.0;
    data->therm.xi = 0.0;
    data->therm.v_xi = 0.0;
    data->therm.v_xi_old = 0.0;
    data->therm.G_xi = 0.0;

    /* initialize for non-restarted run,
     * code in restart.c (restart file parser) initializes otherwise */
    if ( control->restart == FALSE )
    {
        data->step = 0;
        data->prev_steps = 0;
    }

    data->time = 0.0;

    switch ( control->ensemble )
    {
    case NVE:
        data->N_f = 3 * system->N;
        *Evolve = &Velocity_Verlet_NVE;
        break;

    case bNVT:
        data->N_f = 3 * system->N;
        *Evolve = &Velocity_Verlet_Berendsen_NVT;
        break;

    case nhNVT:
        data->N_f = 3 * system->N + 1;
        *Evolve = &Velocity_Verlet_Nose_Hoover_NVT_Klein;

        if ( control->restart == FALSE
                || (control->restart == TRUE && control->random_vel == TRUE) )
        {
            data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin
                    - data->N_f * K_B * control->T);
            data->therm.v_xi = data->therm.G_xi * control->dt;
            data->therm.v_xi_old = 0.0;
            data->therm.xi = 0.0;
        }
        break;

    /* anisotropic NPT */
    case aNPT:
        fprintf( stderr, "[ERROR] THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
        exit( UNKNOWN_OPTION );

        data->N_f = 3 * system->N + 9;
        *Evolve = &Velocity_Verlet_Berendsen_Isotropic_NPT;

        if ( control->restart == FALSE )
        {
            data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin -
                    data->N_f * K_B * control->T);
            data->therm.v_xi = data->therm.G_xi * control->dt;
            data->iso_bar.eps = 1.0 / 3.0 * LOG( system->box.volume );
//            data->inv_W = 1.0 / (data->N_f * K_B * control->T * SQR(control->Tau_P));
//            Compute_Pressure( system, data, workspace );
        }
        break;

    /* semi-isotropic NPT */
    case sNPT:
        data->N_f = 3 * system->N + 4;
        *Evolve = &Velocity_Verlet_Berendsen_Semi_Isotropic_NPT;
        break;

    /* isotropic NPT */
    case iNPT:
        data->N_f = 3 * system->N + 2;
        *Evolve = &Velocity_Verlet_Berendsen_Isotropic_NPT;
        break;

    default:
        fprintf( stderr, "[ERROR] Unknown ensemble type (%d). Terminating...\n", control->ensemble );
        exit( UNKNOWN_OPTION );
        break;
    }

    Compute_Kinetic_Energy( system, data );

    /* init timing info */
    data->timing.start = Get_Time( );
    data->timing.total = data->timing.start;
    data->timing.nbrs = 0.0;
    data->timing.init_forces = 0.0;
    data->timing.bonded = 0.0;
    data->timing.nonb = 0.0;
    data->timing.cm = 0.0;
    data->timing.cm_sort_mat_rows = 0.0;
    data->timing.cm_solver_pre_comp = 0.0;
    data->timing.cm_solver_pre_app = 0.0;
    data->timing.cm_solver_iters = 0;
    data->timing.cm_solver_spmv = 0.0;
    data->timing.cm_solver_vector_ops = 0.0;
    data->timing.cm_solver_orthog = 0.0;
    data->timing.cm_solver_tri_solve = 0.0;
    data->timing.cm_last_pre_comp = 0.0;
    data->timing.cm_total_loss = 0.0;
    data->timing.cm_optimum = 0.0;
}


/* Initialize Taper params */
static void Init_Taper( control_params *control, static_storage *workspace )
{
    real d1, d7;
    real swa, swa2, swa3;
    real swb, swb2, swb3;

    swa = control->nonb_low;
    swb = control->nonb_cut;

    if ( FABS( swa ) > 0.01 )
    {
        fprintf( stderr, "[WARNING] non-zero value for lower Taper-radius cutoff (%f)\n", swa );
    }

    if ( swb < 0.0 )
    {
        fprintf( stderr, "[ERROR] Negative value for upper Taper-radius cutoff\n" );
        exit( INVALID_INPUT );
    }
    else if ( swb < 5.0 )
    {
        fprintf( stderr, "[WARNING] Low value for upper Taper-radius cutoff (%f)\n", swb );
    }

    d1 = swb - swa;
    d7 = POW( d1, 7.0 );
    swa2 = SQR( swa );
    swa3 = swa2 * swa;
    swb2 = SQR( swb );
    swb3 = swb2 * swb;

    workspace->Tap[7] =  20.0 / d7;
    workspace->Tap[6] = -70.0 * (swa + swb) / d7;
    workspace->Tap[5] =  84.0 * (swa2 + 3.0 * swa * swb + swb2) / d7;
    workspace->Tap[4] = -35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3 ) / d7;
    workspace->Tap[3] = 140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3 ) / d7;
    workspace->Tap[2] = -210.0 * (swa3 * swb2 + swa2 * swb3) / d7;
    workspace->Tap[1] = 140.0 * swa3 * swb3 / d7;
    workspace->Tap[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2 +
            7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb ) / d7;
}


static void Init_Workspace( reax_system *system, control_params *control,
        static_storage *workspace, int realloc )
{
    int i;

    if ( realloc == TRUE )
    {
        /* hydrogen bond list */
        workspace->hbond_index = smalloc( system->N_max * sizeof( int ),
               "Init_Workspace::workspace->hbond_index" );

        /* bond order related storage  */
        workspace->total_bond_order = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->bond_order" );
        workspace->Deltap = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->Deltap" );
        workspace->Deltap_boc = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->Deltap_boc" );
        workspace->dDeltap_self = smalloc( system->N_max * sizeof( rvec ),
               "Init_Workspace::workspace->dDeltap_self" );

        workspace->Delta = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->Delta" );
        workspace->Delta_lp = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->Delta_lp" );
        workspace->Delta_lp_temp = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->Delta_lp_temp" );
        workspace->dDelta_lp = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->dDelta_lp" );
        workspace->dDelta_lp_temp = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->dDelta_lp_temp" );
        workspace->Delta_e = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->Delta_e" );
        workspace->Delta_boc = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->Delta_boc" );
        workspace->nlp = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->nlp" );
        workspace->nlp_temp = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->nlp_temp" );
        workspace->Clp = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->Clp" );
        workspace->CdDelta = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->CdDelta" );
        workspace->vlpex = smalloc( system->N_max * sizeof( real ),
               "Init_Workspace::workspace->vlpex" );
    }

    /* charge method storage */
    switch ( control->charge_method )
    {
        case QEQ_CM:
            system->N_cm = system->N;
            system->N_cm_max = system->N_max;
            break;
        case EE_CM:
            system->N_cm = system->N + 1;
            system->N_cm_max = system->N_max + 1;
            break;
        case ACKS2_CM:
            system->N_cm = 2 * system->N + 2;
            system->N_cm_max = 2 * system->N_max + 2;
            break;
        default:
            fprintf( stderr, "[ERROR] Unknown charge method type. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    if ( realloc == TRUE )
    {
        workspace->Hdia_inv = NULL;

        if ( control->cm_solver_pre_comp_type == ICHOLT_PC
                || (control->cm_solver_pre_comp_type == ILUT_PC && control->cm_solver_pre_comp_droptol > 0.0 )
                || control->cm_solver_pre_comp_type == ILUTP_PC
                || control->cm_solver_pre_comp_type == FG_ILUT_PC )
        {
            workspace->droptol = scalloc( system->N_cm_max, sizeof( real ),
                    "Init_Workspace::workspace->droptol" );
        }

        workspace->b_s = scalloc( system->N_cm_max, sizeof( real ),
                "Init_Workspace::workspace->b_s" );
        workspace->b_t = scalloc( system->N_cm_max, sizeof( real ),
                "Init_Workspace::workspace->b_t" );
        workspace->b_prc = scalloc( system->N_cm_max * 2, sizeof( real ),
                "Init_Workspace::workspace->b_prc" );
        workspace->b_prm = scalloc( system->N_cm_max * 2, sizeof( real ),
                "Init_Workspace::workspace->b_prm" );
        workspace->s = scalloc( 5, sizeof( real* ),
                "Init_Workspace::workspace->s" );
        workspace->t = scalloc( 5, sizeof( real* ),
                "Init_Workspace::workspace->t" );
        for ( i = 0; i < 5; ++i )
        {
            workspace->s[i] = scalloc( system->N_cm_max, sizeof( real ),
                    "Init_Workspace::workspace->s[i]" );
            workspace->t[i] = scalloc( system->N_cm_max, sizeof( real ),
                    "Init_Workspace::workspace->t[i]" );
        }
    }

    switch ( control->charge_method )
    {
        case QEQ_CM:
            for ( i = 0; i < system->N; ++i )
            {
                workspace->b_s[i] = -1.0 * system->reax_param.sbp[ system->atoms[i].type ].chi;
                workspace->b_t[i] = -1.0;
            }
            break;

        case EE_CM:
            for ( i = 0; i < system->N; ++i )
            {
                workspace->b_s[i] = -system->reax_param.sbp[ system->atoms[i].type ].chi;
            }

            workspace->b_s[system->N] = control->cm_q_net;
            break;

        case ACKS2_CM:
            for ( i = 0; i < system->N; ++i )
            {
                workspace->b_s[i] = -system->reax_param.sbp[ system->atoms[i].type ].chi;
            }

            /* Non-zero total charge can lead to unphysical results.
             * As such, set the ACKS2 reference charge of every atom
             * to the total charge divided by the number of atoms.
             * Except for trivial cases, this leads to fractional
             * reference charges, which is usually not desirable. */
            for ( i = 0; i < system->N; ++i )
            {
                workspace->b_s[system->N + i] = control->cm_q_net / system->N;
            }

            /* system charge defines the total charge constraint */
            workspace->b_s[system->N_cm - 1] = control->cm_q_net;
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method type. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    if ( realloc == TRUE )
    {
        switch ( control->cm_solver_type )
        {
            case GMRES_S:
            case GMRES_H_S:
                workspace->y = scalloc( control->cm_solver_restart + 1, sizeof( real ),
                        "Init_Workspace::workspace->y" );
                workspace->z = scalloc( control->cm_solver_restart + 1, sizeof( real ),
                        "Init_Workspace::workspace->z" );
                workspace->g = scalloc( control->cm_solver_restart + 1, sizeof( real ),
                        "Init_Workspace::workspace->g" );
                workspace->h = scalloc( control->cm_solver_restart + 1, sizeof( real*),
                        "Init_Workspace::workspace->h" );
                workspace->hs = scalloc( control->cm_solver_restart + 1, sizeof( real ),
                        "Init_Workspace::workspace->hs" );
                workspace->hc = scalloc( control->cm_solver_restart + 1, sizeof( real ),
                        "Init_Workspace::workspace->hc" );
                workspace->rn = scalloc( control->cm_solver_restart + 1, sizeof( real*),
                        "Init_Workspace::workspace->rn" );
                workspace->v = scalloc( control->cm_solver_restart + 1, sizeof( real*),
                        "Init_Workspace::workspace->v" );

                for ( i = 0; i < control->cm_solver_restart + 1; ++i )
                {
                    workspace->h[i] = scalloc( control->cm_solver_restart + 1, sizeof( real ),
                            "Init_Workspace::workspace->h[i]" );
                    workspace->rn[i] = scalloc( system->N_cm_max * 2, sizeof( real ),
                            "Init_Workspace::workspace->rn[i]" );
                    workspace->v[i] = scalloc( system->N_cm_max, sizeof( real ),
                            "Init_Workspace::workspace->v[i]" );
                }

                workspace->r = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->r" );
                workspace->d = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->d" );
                workspace->q = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->q" );
                workspace->p = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->p" );
                break;

            case CG_S:
                workspace->r = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->r" );
                workspace->d = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->d" );
                workspace->q = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->q" );
                workspace->p = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->p" );
                break;

            case SDM_S:
                workspace->r = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->r" );
                workspace->d = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->d" );
                workspace->q = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->q" );
                break;

            case BiCGStab_S:
                workspace->r = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->r" );
                workspace->r_hat = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->r_hat" );
                workspace->d = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->d" );
                workspace->q = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->q" );
                workspace->q_hat = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->q_hat" );
                workspace->p = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->p" );
                workspace->y = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->y" );
                workspace->z = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->z" );
                workspace->g = scalloc( system->N_cm_max, sizeof( real ),
                        "Init_Workspace::workspace->g" );
                break;

            default:
                fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
                exit( INVALID_INPUT );
                break;
        }

#if defined(_OPENMP)
        /* SpMV related */
        workspace->b_local = smalloc( control->num_threads * system->N_cm_max * sizeof(real),
                "Init_Workspace::b_local" );
#endif
    }

    /* level scheduling related */
    workspace->levels_L = 1;
    workspace->levels_U = 1;
    if ( realloc == TRUE )
    {
        if ( control->cm_solver_pre_app_type == TRI_SOLVE_LEVEL_SCHED_PA ||
                control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
        {
            workspace->row_levels_L = smalloc( system->N_cm_max * sizeof(unsigned int),
                    "Init_Workspace::row_levels_L" );
            workspace->level_rows_L = smalloc( system->N_cm_max * sizeof(unsigned int),
                    "Init_Workspace::level_rows_L" );
            workspace->level_rows_cnt_L = smalloc( (system->N_cm_max + 1) * sizeof(unsigned int),
                    "Init_Workspace::level_rows_cnt_L" );
            workspace->row_levels_U = smalloc( system->N_cm_max * sizeof(unsigned int),
                    "Init_Workspace::row_levels_U" );
            workspace->level_rows_U = smalloc( system->N_cm_max * sizeof(unsigned int),
                    "Init_Workspace::level_rows_U" );
            workspace->level_rows_cnt_U = smalloc( (system->N_cm_max + 1) * sizeof(unsigned int),
                    "Init_Workspace::level_rows_cnt_U" );
            workspace->top = smalloc( (system->N_cm_max + 1) * sizeof(unsigned int),
                    "Init_Workspace::top" );
        }
        else
        {
            workspace->row_levels_L = NULL;
            workspace->level_rows_L = NULL;
            workspace->level_rows_cnt_L = NULL;
            workspace->row_levels_U = NULL;
            workspace->level_rows_U = NULL;
            workspace->level_rows_cnt_U = NULL;
            workspace->top = NULL;
        }
    }

    /* graph coloring related */
    workspace->recolor_cnt = 0;
    if ( realloc == TRUE )
    {
        if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
        {
            workspace->color = smalloc( sizeof(unsigned int) * system->N_cm_max,
                    "Init_Workspace::color" );
            workspace->to_color = smalloc( sizeof(unsigned int) * system->N_cm_max,
                    "Init_Workspace::to_color" );
            workspace->conflict = smalloc( sizeof(unsigned int) * system->N_cm_max,
                    "setup_graph_coloring::conflict" );
            workspace->conflict_cnt = smalloc( sizeof(unsigned int) * (control->num_threads + 1),
                    "Init_Workspace::conflict_cnt" );
            workspace->recolor = smalloc( sizeof(unsigned int) * system->N_cm_max,
                    "Init_Workspace::recolor" );
            workspace->color_top = smalloc( sizeof(unsigned int) * (system->N_cm_max + 1),
                    "Init_Workspace::color_top" );
            workspace->permuted_row_col = smalloc( sizeof(unsigned int) * system->N_cm_max,
                    "Init_Workspace::premuted_row_col" );
            workspace->permuted_row_col_inv = smalloc( sizeof(unsigned int) * system->N_cm_max,
                    "Init_Workspace::premuted_row_col_inv" );
        }
        else
        {
            workspace->color = NULL;
            workspace->to_color = NULL;
            workspace->conflict = NULL;
            workspace->conflict_cnt = NULL;
            workspace->recolor = NULL;
            workspace->color_top = NULL;
            workspace->permuted_row_col = NULL;
            workspace->permuted_row_col_inv = NULL;
        }

        /* graph coloring related OR ILUTP preconditioner */
        if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA 
                || control->cm_solver_pre_comp_type == ILUTP_PC )
        {
            workspace->y_p = smalloc( sizeof(real) * system->N_cm_max, "Init_Workspace::y_p" );
            workspace->x_p = smalloc( sizeof(real) * system->N_cm_max, "Init_Workspace::x_p" );
        }
        else
        {
            workspace->y_p = NULL;
            workspace->x_p = NULL;
        }

        /* Jacobi iteration related */
        if ( control->cm_solver_pre_app_type == JACOBI_ITER_PA )
        {
            workspace->Dinv_L = smalloc( sizeof(real) * system->N_cm_max,
                    "Init_Workspace::Dinv_L" );
            workspace->Dinv_U = smalloc( sizeof(real) * system->N_cm_max,
                    "Init_Workspace::Dinv_U" );
            workspace->Dinv_b = smalloc( sizeof(real) * system->N_cm_max,
                    "Init_Workspace::Dinv_b" );
            workspace->rp = smalloc( sizeof(real) * system->N_cm_max,
                    "Init_Workspace::rp" );
            workspace->rp2 = smalloc( sizeof(real) * system->N_cm_max,
                    "Init_Workspace::rp2" );
        }
        else
        {
            workspace->Dinv_L = NULL;
            workspace->Dinv_U = NULL;
            workspace->Dinv_b = NULL;
            workspace->rp = NULL;
            workspace->rp2 = NULL;
        }

        /* ILUTP preconditioner related */
        if ( control->cm_solver_pre_comp_type == ILUTP_PC )
        {
            workspace->perm_ilutp = smalloc( sizeof( int ) * system->N_cm_max,
                   "Init_Workspace::workspace->perm_ilutp" );
        }
        else
        {
            workspace->perm_ilutp = NULL;
        }

#if defined(QMMM)
        workspace->mask_qmmm = smalloc( system->N_cm_max * sizeof( int ),
               "Init_Workspace::workspace->mask_qmmm" );
#endif

        /* integrator storage */
        workspace->a = smalloc( system->N_max * sizeof( rvec ),
               "Init_Workspace::workspace->a" );
        workspace->f_old = smalloc( system->N_max * sizeof( rvec ),
               "Init_Workspace::workspace->f_old" );
        workspace->v_const = smalloc( system->N_max * sizeof( rvec ),
               "Init_Workspace::workspace->v_const" );

#if defined(_OPENMP)
        workspace->f_local = smalloc( control->num_threads * system->N_max * sizeof( rvec ),
               "Init_Workspace::workspace->f_local" );
#endif

        /* storage for analysis */
        if ( control->molec_anal || control->diffusion_coef )
        {
            workspace->mark = scalloc( system->N_max, sizeof(int),
                    "Init_Workspace::workspace->mark" );
            workspace->old_mark = scalloc( system->N_max, sizeof(int),
                    "Init_Workspace::workspace->old_mark" );
        }
        else
        {
            workspace->mark = workspace->old_mark = NULL;
        }

        if ( control->diffusion_coef )
        {
            workspace->x_old = scalloc( system->N_max, sizeof( rvec ),
                    "Init_Workspace::workspace->x_old" );
        }
        else
        {
            workspace->x_old = NULL;
        }
    }

#if defined(TEST_FORCES)
    workspace->dDelta = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->dDelta" );
    workspace->f_ele = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_ele" );
    workspace->f_vdw = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_vdw" );
    workspace->f_be = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_be" );
    workspace->f_lp = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_lp" );
    workspace->f_ov = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_ov" );
    workspace->f_un = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_un" );
    workspace->f_ang = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_ang" );
    workspace->f_coa = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_coa" );
    workspace->f_pen = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_pen" );
    workspace->f_hb = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_hb" );
    workspace->f_tor = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_tor" );
    workspace->f_con = smalloc( system->N_max * sizeof( rvec ),
           "Init_Workspace::workspace->f_con" );
#endif

    workspace->realloc.num_far = -1;
    workspace->realloc.Htop = -1;
    workspace->realloc.hbonds = -1;
    workspace->realloc.bonds = -1;
    workspace->realloc.num_3body = -1;
    workspace->realloc.gcell_atoms = -1;

    Reset_Workspace( system, workspace );

    /* Initialize Taper function */
    Init_Taper( control, workspace );
}


static void Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control, int realloc )
{
    int i, num_nbrs, num_bonds, num_hbonds, num_3body, Htop, max_nnz;
    int *hb_top, *bond_top;

    num_nbrs = Estimate_Num_Neighbors( system, control, workspace, lists );

    if ( lists[FAR_NBRS]->allocated == FALSE )
    {
        Make_List( system->N, system->N_max, num_nbrs, TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );
    }
    else if ( realloc == TRUE || lists[FAR_NBRS]->total_intrs < num_nbrs )
    {
        if ( lists[FAR_NBRS]->allocated == TRUE )
        {
            Delete_List( TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );
        }
        Make_List( system->N, system->N_max, 
                MAX( num_nbrs, lists[FAR_NBRS]->total_intrs ),
                TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );
    }
    else
    {
        lists[FAR_NBRS]->n = system->N;
    }

    Generate_Neighbor_Lists( system, control, data, workspace, lists, out_control );

    Htop = 0;
    hb_top = scalloc( system->N, sizeof(int), "Init_Lists::hb_top" );
    bond_top = scalloc( system->N, sizeof(int), "Init_Lists::bond_top" );
    num_3body = 0;

    Estimate_Storage_Sizes( system, control, lists, &Htop,
            hb_top, bond_top, &num_3body );
    num_3body = MAX( num_3body, MIN_BONDS );

    switch ( control->charge_method )
    {
        case QEQ_CM:
            max_nnz = Htop;
            break;
        case EE_CM:
            max_nnz = Htop + system->N_cm;
            break;
        case ACKS2_CM:
            max_nnz = 2 * Htop + 3 * system->N + 2;
            break;
        default:
            max_nnz = Htop;
            break;
    }

    if ( workspace->H.allocated == FALSE )
    {
        Allocate_Matrix( &workspace->H, system->N_cm, system->N_cm_max, max_nnz );
    }
    else if ( realloc == TRUE || workspace->H.m < max_nnz )
    {
        if ( workspace->H.allocated == TRUE )
        {
            Deallocate_Matrix( &workspace->H );
        }
        Allocate_Matrix( &workspace->H, system->N_cm, system->N_cm_max, max_nnz );
    }
    else
    {
        workspace->H.n = system->N_cm;
    }

    if ( workspace->H_sp.allocated == FALSE )
    {
        /* TODO: better estimate for H_sp?
         *   If so, need to refactor Estimate_Storage_Sizes
         *   to use various cut-off distances as parameters
         *   (non-bonded, hydrogen, 3body, etc.) */
        Allocate_Matrix( &workspace->H_sp, system->N_cm, system->N_cm_max, max_nnz );
    }
    else if ( realloc == TRUE || workspace->H_sp.m < max_nnz )
    {
        if ( workspace->H_sp.allocated == TRUE )
        {
            Deallocate_Matrix( &workspace->H_sp );
        }
        /* TODO: better estimate for H_sp?
         *   If so, need to refactor Estimate_Storage_Sizes
         *   to use various cut-off distances as parameters
         *   (non-bonded, hydrogen, 3body, etc.) */
        Allocate_Matrix( &workspace->H_sp, system->N_cm, system->N_cm_max, max_nnz );
    }
    else
    {
        workspace->H_sp.n = system->N_cm;
    }

    workspace->num_H = 0;
    if ( control->hbond_cut > 0.0 )
    {
        /* init hydrogen atom indexes */
        for ( i = 0; i < system->N; ++i )
        {
            if ( system->reax_param.sbp[ system->atoms[i].type ].p_hbond == H_ATOM )
            {
                workspace->hbond_index[i] = workspace->num_H++;
            }
            else
            {
                workspace->hbond_index[i] = -1;
            }
        }

        if ( workspace->num_H == 0 )
        {
            control->hbond_cut = 0.0;
        }
        else
        {
            num_hbonds = 0;
            for ( i = 0; i < system->N; ++i )
            {
                num_hbonds += hb_top[i];
            }

            if ( lists[HBONDS]->allocated == FALSE )
            {
                workspace->num_H_max = (int) CEIL( SAFE_ZONE * workspace->num_H );

                Make_List( workspace->num_H, workspace->num_H_max,
                        (int) CEIL( SAFE_ZONE * num_hbonds ),
                        TYP_HBOND, lists[HBONDS] );
            }
            else if ( workspace->num_H_max < workspace->num_H
                    || lists[HBONDS]->total_intrs < num_hbonds )
            {
                if ( workspace->num_H_max < workspace->num_H )
                {
                    workspace->num_H_max = (int) CEIL( SAFE_ZONE * workspace->num_H );
                }

                if ( lists[HBONDS]->allocated == TRUE )
                {
                    Delete_List( TYP_HBOND, lists[HBONDS] );
                }
                Make_List( workspace->num_H, workspace->num_H_max,
                        MAX( num_hbonds, lists[HBONDS]->total_intrs ),
                        TYP_HBOND, lists[HBONDS] );
            }
            else
            {
                lists[HBONDS]->n = workspace->num_H;
            }

            Initialize_HBond_List( system->N, workspace->hbond_index, hb_top, lists[HBONDS] );
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "estimated storage - num_hbonds: %d\n", num_hbonds );
    fprintf( stderr, "memory allocated: hbonds = %ldMB\n",
             num_hbonds * sizeof(hbond_data) / (1024 * 1024) );
#endif

    num_bonds = 0;
    for ( i = 0; i < system->N; ++i )
    {
        num_bonds += bond_top[i];
    }

    /* bonds list */
    if ( lists[BONDS]->allocated == FALSE )
    {
        Make_List( system->N, system->N_max, (int) CEIL( num_bonds * SAFE_ZONE ),
                TYP_BOND, lists[BONDS] );
    }
    else if ( realloc == TRUE || lists[BONDS]->total_intrs < num_bonds )
    {
        if ( lists[BONDS]->allocated == TRUE )
        {
            Delete_List( TYP_BOND, lists[BONDS] );
        }
        Make_List( system->N, system->N_max,
                MAX( num_bonds, lists[BONDS]->total_intrs ),
                TYP_BOND, lists[BONDS] );
    }
    else
    {
        lists[BONDS]->n = system->N;
    }

    Initialize_Bond_List( bond_top, lists[BONDS] );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "estimated storage - num_bonds: %d\n", num_bonds );
    fprintf( stderr, "memory allocated: bonds = %ldMB\n",
             num_bonds * sizeof(bond_data) / (1024 * 1024) );
#endif

    /* 3bodies list */
    if ( lists[THREE_BODIES]->allocated == FALSE )
    {
        Make_List( num_bonds, num_bonds, num_3body, TYP_THREE_BODY, lists[THREE_BODIES] );
    }
    else if ( lists[THREE_BODIES]->n_max < num_bonds
            || lists[THREE_BODIES]->total_intrs < num_3body )
    {
        if ( lists[THREE_BODIES]->allocated == TRUE )
        {
            Delete_List( TYP_THREE_BODY, lists[THREE_BODIES] );
        }
        Make_List( MAX( num_bonds, lists[THREE_BODIES]->n_max),
                MAX( num_bonds, lists[THREE_BODIES]->n_max),
                MAX( num_3body, lists[THREE_BODIES]->total_intrs ),
                TYP_THREE_BODY, lists[THREE_BODIES] );
    }
    else
    {
        lists[THREE_BODIES]->n = num_bonds;
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "estimated storage - num_3body: %d\n", num_3body );
    fprintf( stderr, "memory allocated: 3-body = %ldMB\n",
             num_3body * sizeof(three_body_interaction_data) / (1024 * 1024) );
#endif

#if defined(TEST_FORCES)
    //TODO: increased num. of DDELTA list elements, find a better count later
    Make_List( system->N, num_bonds * 20, TYP_DDELTA, lists[DDELTA] );

    for ( i = 0; i < lists[DDELTA]->n; ++i )
    {
        Set_Start_Index( i, 0, lists[DDELTA] );
        Set_End_Index( i, 0, lists[DDELTA] );
    }

    Make_List( num_bonds, num_bonds * MAX_BONDS * 3, TYP_DBO, lists[DBO] );

    for ( i = 0; i < lists[DBO]->n; ++i )
    {
        Set_Start_Index( i, 0, lists[DBO] );
        Set_End_Index( i, 0, lists[DBO] );
    }
#endif

    sfree( hb_top, "Init_Lists::hb_top" );
    sfree( bond_top, "Init_Lists::bond_top" );
}


static void Init_Out_Controls( reax_system *system, control_params *control,
        static_storage *workspace, output_controls *out_control, int output_enabled )
{
#define TEMP_SIZE (1000)
    char temp[TEMP_SIZE];

    if ( output_enabled == TRUE && out_control->write_steps > 0 )
    {
        strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
        temp[TEMP_SIZE - 5] = '\0';
        strcat( temp, ".trj" );
        out_control->trj = sfopen( temp, "w" );
        out_control->write_header( system, control, workspace, out_control );
    }
    else
    {
        out_control->trj = NULL;
    }

    if ( output_enabled == TRUE && out_control->log_update_freq > 0 )
    {
        strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
        temp[TEMP_SIZE - 5] = '\0';
        strcat( temp, ".out" );
        out_control->out = sfopen( temp, "w" );
        fprintf( out_control->out, "%-6s%16s%16s%16s%11s%11s%13s%13s%13s\n",
                 "step", "total_energy", "poten_energy", "kin_energy",
                 "temp", "target", "volume", "press", "target" );
        fflush( out_control->out );

        strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
        temp[TEMP_SIZE - 5] = '\0';
        strcat( temp, ".pot" );
        out_control->pot = sfopen( temp, "w" );
        fprintf( out_control->pot,
                 "%-6s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s\n",
                 "step", "ebond", "eatom", "elp", "eang", "ecoa", "ehb",
                 "etor", "econj", "evdw", "ecoul", "epol" );
        fflush( out_control->pot );

        strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
        temp[TEMP_SIZE - 5] = '\0';
        strcat( temp, ".log" );
        out_control->log = sfopen( temp, "w" );
        fprintf( out_control->log, "%-6s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                 "step", "total", "neighbors", "init", "bonded",
                 "nonbonded", "cm", "cm_sort", "s_iters", "pre_comp", "pre_app",
                 "s_spmv", "s_vec_ops", "s_orthog", "s_tsolve" );
    }
    else
    {
        out_control->out = NULL;
        out_control->pot = NULL;
        out_control->log = NULL;
    }

    if ( output_enabled == TRUE && (control->ensemble == sNPT || control->ensemble == iNPT
            || control->ensemble == aNPT || control->compute_pressure == TRUE) )
    {
        strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
        temp[TEMP_SIZE - 5] = '\0';
        strcat( temp, ".prs" );
        out_control->prs = sfopen( temp, "w" );
#if defined(DEBUG) || defined(DEBUG_FOCUS)
        fprintf( out_control->prs, "%-8s %13s %13s %13s %13s %13s %13s\n",
                "step", "KExx", "KEyy", "KEzz",
                "Virialxx", "Virialyy", "Virialzz" );
#endif
        fprintf( out_control->prs, "%-8s %13s %13s %13s %13s %13s %13s %13s %13s\n",
                "step", "Lx", "Ly", "Lz",
                "Pxx", "Pyy", "Pzz", "Pavg", "Volume" );

        fflush( out_control->prs );
    }
    else
    {
        out_control->prs = NULL;
    }

    /* Init molecular analysis file */
    if ( output_enabled == TRUE && control->molec_anal )
    {
        snprintf( temp, TEMP_SIZE, "%.*s.mol", TEMP_SIZE - 5, control->sim_name );
        out_control->mol = sfopen( temp, "w" );
        if ( control->num_ignored )
        {
            snprintf( temp, TEMP_SIZE, "%.*s.ign", TEMP_SIZE - 5, control->sim_name );
            out_control->ign = sfopen( temp, "w" );
        }
    }
    else
    {
        out_control->mol = NULL;
        out_control->ign = NULL;
    }

    if ( output_enabled == TRUE && control->dipole_anal )
    {
        strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
        temp[TEMP_SIZE - 5] = '\0';
        strcat( temp, ".dpl" );
        out_control->dpl = sfopen( temp, "w" );
        fprintf( out_control->dpl,
                 "Step      Molecule Count  Avg. Dipole Moment Norm\n" );
        fflush( out_control->dpl );
    }
    else
    {
        out_control->dpl = NULL;
    }

    if ( output_enabled == TRUE && control->diffusion_coef )
    {
        strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
        temp[TEMP_SIZE - 6] = '\0';
        strcat( temp, ".drft" );
        out_control->drft = sfopen( temp, "w" );
        fprintf( out_control->drft, "Step     Type Count   Avg Squared Disp\n" );
        fflush( out_control->drft );
    }
    else
    {
        out_control->drft = NULL;
    }


#if defined(TEST_ENERGY)
    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".ebond" );
    out_control->ebond = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
    temp[TEMP_SIZE - 5] = '\0';
    strcat( temp, ".elp" );
    out_control->elp = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
    temp[TEMP_SIZE - 5] = '\0';
    strcat( temp, ".eov" );
    out_control->eov = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
    temp[TEMP_SIZE - 5] = '\0';
    strcat( temp, ".eun" );
    out_control->eun = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".eval" );
    out_control->eval = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".epen" );
    out_control->epen = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".ecoa" );
    out_control->ecoa = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
    temp[TEMP_SIZE - 5] = '\0';
    strcat( temp, ".ehb" );
    out_control->ehb = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".etor" );
    out_control->etor = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".econ" );
    out_control->econ = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".evdw" );
    out_control->evdw = sfopen( temp, "w" );

    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".ecou" );
    out_control->ecou = sfopen( temp, "w" );
#endif


#if defined(TEST_FORCES)
    /* open bond orders file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
    temp[TEMP_SIZE - 5] = '\0';
    strcat( temp, ".fbo" );
    out_control->fbo = sfopen( temp, "w" );

    /* open bond orders derivatives file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".fdbo" );
    out_control->fdbo = sfopen( temp, "w" );

    /* open bond forces file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 7 );
    temp[TEMP_SIZE - 7] = '\0';
    strcat( temp, ".fbond" );
    out_control->fbond = sfopen( temp, "w" );

    /* open lone-pair forces file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".flp" );
    out_control->flp = sfopen( temp, "w" );

    /* open overcoordination forces file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 7 );
    temp[TEMP_SIZE - 7] = '\0';
    strcat( temp, ".fatom" );
    out_control->fatom = sfopen( temp, "w" );

    /* open angle forces file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 8 );
    temp[TEMP_SIZE - 8] = '\0';
    strcat( temp, ".f3body" );
    out_control->f3body = sfopen( temp, "w" );

    /* open hydrogen bond forces file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 5 );
    temp[TEMP_SIZE - 5] = '\0';
    strcat( temp, ".fhb" );
    out_control->fhb = sfopen( temp, "w" );

    /* open torsion forces file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 8 );
    temp[TEMP_SIZE - 8] = '\0';
    strcat( temp, ".f4body" );
    out_control->f4body = sfopen( temp, "w" );

    /* open nonbonded forces file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 7 );
    temp[TEMP_SIZE - 7] = '\0';
    strcat( temp, ".fnonb" );
    out_control->fnonb = sfopen( temp, "w" );

    /* open total force file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 6 );
    temp[TEMP_SIZE - 6] = '\0';
    strcat( temp, ".ftot" );
    out_control->ftot = sfopen( temp, "w" );

    /* open coulomb forces file */
    strncpy( temp, control->sim_name, TEMP_SIZE - 7 );
    temp[TEMP_SIZE - 7] = '\0';
    strcat( temp, ".ftot2" );
    out_control->ftot2 = sfopen( temp, "w" );
#endif

#undef TEMP_SIZE
}


void Initialize( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, reax_list **lists,
        output_controls *out_control, evolve_function *Evolve,
        int output_enabled, int realloc )
{
#if defined(_OPENMP)
    #pragma omp parallel default(none) shared(control)
    {
        #pragma omp single
        control->num_threads = omp_get_num_threads( );
    }
#else
    control->num_threads = 1;
#endif

    Randomize( );

    Init_System( system, control, data );

    Init_Simulation_Data( system, control, data, out_control, Evolve, realloc );

    Init_Workspace( system, control, workspace, realloc );

    Init_Lists( system, control, data, workspace, lists, out_control, realloc );

    Init_Out_Controls( system, control, workspace, out_control, output_enabled );

    /* These are done in forces.c, only forces.c can see all those functions */
    Init_Bonded_Force_Functions( control );

#if defined(TEST_FORCES)
    Init_Force_Test_Functions( control );
#endif

    if ( control->tabulate )
    {
        Make_LR_Lookup_Table( system, control, workspace );
    }
}


static void Finalize_System( reax_system *system, control_params *control,
        simulation_data *data, int reset )
{
    int i, j, k;
    reax_interaction *reax;

    system->prealloc_allocated = FALSE;
    system->ffield_params_allocated = FALSE;

    reax = &system->reax_param;

    Finalize_Grid( system );

    if ( reset == FALSE )
    {
        sfree( reax->gp.l, "Finalize_System::reax->gp.l" );

        for ( i = 0; i < reax->max_num_atom_types; i++ )
        {
            for ( j = 0; j < reax->max_num_atom_types; j++ )
            {
                for ( k = 0; k < reax->max_num_atom_types; k++ )
                {
                    sfree( reax->fbp[i][j][k], "Finalize_System::reax->fbp[i][j][k]" );
                }

                sfree( reax->thbp[i][j], "Finalize_System::reax->thbp[i][j]" );
                sfree( reax->hbp[i][j], "Finalize_System::reax->hbp[i][j]" );
                sfree( reax->fbp[i][j], "Finalize_System::reax->fbp[i][j]" );
            }

            sfree( reax->tbp[i], "Finalize_System::reax->tbp[i]" );
            sfree( reax->thbp[i], "Finalize_System::reax->thbp[i]" );
            sfree( reax->hbp[i], "Finalize_System::reax->hbp[i]" );
            sfree( reax->fbp[i], "Finalize_System::reax->fbp[i]" );
        }

        sfree( reax->sbp, "Finalize_System::reax->sbp" );
        sfree( reax->tbp, "Finalize_System::reax->tbp" );
        sfree( reax->thbp, "Finalize_System::reax->thbp" );
        sfree( reax->hbp, "Finalize_System::reax->hbp" );
        sfree( reax->fbp, "Finalize_System::reax->fbp" );

        sfree( system->atoms, "Finalize_System::system->atoms" );
    }
}


static void Finalize_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data, output_controls *out_control )
{
#if defined(_OPENMP)
    if ( control->ensemble == sNPT || control->ensemble == iNPT
            || control->ensemble == aNPT || control->compute_pressure == TRUE )
    {
        sfree( data->press_local, "Finalize_Simulation_Data::data->press_local" );
    }
#endif
}


static void Finalize_Workspace( reax_system *system, control_params *control,
        static_storage *workspace, int reset )
{
    int i;

    sfree( workspace->hbond_index, "Finalize_Workspace::workspace->hbond_index" );
    sfree( workspace->total_bond_order, "Finalize_Workspace::workspace->total_bond_order" );
    sfree( workspace->Deltap, "Finalize_Workspace::workspace->Deltap" );
    sfree( workspace->Deltap_boc, "Finalize_Workspace::workspace->Deltap_boc" );
    sfree( workspace->dDeltap_self, "Finalize_Workspace::workspace->dDeltap_self" );
    sfree( workspace->Delta, "Finalize_Workspace::workspace->Delta" );
    sfree( workspace->Delta_lp, "Finalize_Workspace::workspace->Delta_lp" );
    sfree( workspace->Delta_lp_temp, "Finalize_Workspace::workspace->Delta_lp_temp" );
    sfree( workspace->dDelta_lp, "Finalize_Workspace::workspace->dDelta_lp" );
    sfree( workspace->dDelta_lp_temp, "Finalize_Workspace::workspace->dDelta_lp_temp" );
    sfree( workspace->Delta_e, "Finalize_Workspace::workspace->Delta_e" );
    sfree( workspace->Delta_boc, "Finalize_Workspace::workspace->Delta_boc" );
    sfree( workspace->nlp, "Finalize_Workspace::workspace->nlp" );
    sfree( workspace->nlp_temp, "Finalize_Workspace::workspace->nlp_temp" );
    sfree( workspace->Clp, "Finalize_Workspace::workspace->Clp" );
    sfree( workspace->CdDelta, "Finalize_Workspace::workspace->CdDelta" );
    sfree( workspace->vlpex, "Finalize_Workspace::workspace->vlpex" );

    if ( reset == FALSE && (control->geo_format == BGF
            || control->geo_format == ASCII_RESTART
            || control->geo_format == BINARY_RESTART) )
    {
        sfree( workspace->map_serials, "Finalize_Workspace::workspace->map_serials" );
    }

    if ( workspace->H.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->H );
    }
    if ( workspace->H_full.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->H_full );
    }
    if ( workspace->H_sp.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->H_sp );
    }
    if ( workspace->H_p.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->H_p );
    }
    if ( workspace->H_spar_patt.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->H_spar_patt );
    }
    if ( workspace->H_spar_patt_full.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->H_spar_patt_full );
    }
    if ( workspace->H_app_inv.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->H_app_inv );
    }
    if ( workspace->L.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->L );
    }
    if ( workspace->U.allocated == TRUE )
    {
        Deallocate_Matrix( &workspace->U );
    }

    for ( i = 0; i < 5; ++i )
    {
        sfree( workspace->s[i], "Finalize_Workspace::workspace->s[i]" );
        sfree( workspace->t[i], "Finalize_Workspace::workspace->t[i]" );
    }

    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        sfree( workspace->Hdia_inv, "Finalize_Workspace::workspace->Hdia_inv" );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || (control->cm_solver_pre_comp_type == ILUT_PC && control->cm_solver_pre_comp_droptol > 0.0 )
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        sfree( workspace->droptol, "Finalize_Workspace::workspace->droptol" );
    }
    sfree( workspace->b_s, "Finalize_Workspace::workspace->b_s" );
    sfree( workspace->b_t, "Finalize_Workspace::workspace->b_t" );
    sfree( workspace->b_prc, "Finalize_Workspace::workspace->b_prc" );
    sfree( workspace->b_prm, "Finalize_Workspace::workspace->b_prm" );
    sfree( workspace->s, "Finalize_Workspace::workspace->s" );
    sfree( workspace->t, "Finalize_Workspace::workspace->t" );

    switch ( control->cm_solver_type )
    {
        case GMRES_S:
        case GMRES_H_S:
            for ( i = 0; i < control->cm_solver_restart + 1; ++i )
            {
                sfree( workspace->h[i], "Finalize_Workspace::workspace->h[i]" );
                sfree( workspace->rn[i], "Finalize_Workspace::workspace->rn[i]" );
                sfree( workspace->v[i], "Finalize_Workspace::workspace->v[i]" );
            }

            sfree( workspace->y, "Finalize_Workspace::workspace->y" );
            sfree( workspace->z, "Finalize_Workspace::workspace->z" );
            sfree( workspace->g, "Finalize_Workspace::workspace->g" );
            sfree( workspace->h, "Finalize_Workspace::workspace->h" );
            sfree( workspace->hs, "Finalize_Workspace::workspace->hs" );
            sfree( workspace->hc, "Finalize_Workspace::workspace->hc" );
            sfree( workspace->rn, "Finalize_Workspace::workspace->rn" );
            sfree( workspace->v, "Finalize_Workspace::workspace->v" );

            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            break;

        case CG_S:
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            break;

        case SDM_S:
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            break;

        case BiCGStab_S:
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->r_hat, "Finalize_Workspace::workspace->r_hat" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->q_hat, "Finalize_Workspace::workspace->q_hat" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            sfree( workspace->y, "Finalize_Workspace::workspace->y" );
            sfree( workspace->z, "Finalize_Workspace::workspace->z" );
            sfree( workspace->g, "Finalize_Workspace::workspace->g" );
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    /* SpMV related */
#if defined(_OPENMP)
    sfree( workspace->b_local, "Finalize_Workspace::b_local" );
#endif

    /* level scheduling related */
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_LEVEL_SCHED_PA ||
            control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        sfree( workspace->row_levels_L, "Finalize_Workspace::row_levels_L" );
        sfree( workspace->level_rows_L, "Finalize_Workspace::level_rows_L" );
        sfree( workspace->level_rows_cnt_L, "Finalize_Workspace::level_rows_cnt_L" );
        sfree( workspace->row_levels_U, "Finalize_Workspace::row_levels_U" );
        sfree( workspace->level_rows_U, "Finalize_Workspace::level_rows_U" );
        sfree( workspace->level_rows_cnt_U, "Finalize_Workspace::level_rows_cnt_U" );
        sfree( workspace->top, "Finalize_Workspace::top" );
    }

    /* graph coloring related */
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA )
    {
        sfree( workspace->color, "Finalize_Workspace::workspace->color" );
        sfree( workspace->to_color, "Finalize_Workspace::workspace->to_color" );
        sfree( workspace->conflict, "Finalize_Workspace::workspace->conflict" );
        sfree( workspace->conflict_cnt, "Finalize_Workspace::workspace->conflict_cnt" );
        sfree( workspace->recolor, "Finalize_Workspace::workspace->recolor" );
        sfree( workspace->color_top, "Finalize_Workspace::workspace->color_top" );
        sfree( workspace->permuted_row_col, "Finalize_Workspace::workspace->permuted_row_col" );
        sfree( workspace->permuted_row_col_inv, "Finalize_Workspace::workspace->permuted_row_col_inv" );
    }

    /* graph coloring related OR ILUTP preconditioner */
    if ( control->cm_solver_pre_app_type == TRI_SOLVE_GC_PA 
            || control->cm_solver_pre_comp_type == ILUTP_PC )
    {
        sfree( workspace->y_p, "Finalize_Workspace::workspace->y_p" );
        sfree( workspace->x_p, "Finalize_Workspace::workspace->x_p" );
    }

    /* Jacobi iteration related */
    if ( control->cm_solver_pre_app_type == JACOBI_ITER_PA )
    {
        sfree( workspace->Dinv_L, "Finalize_Workspace::Dinv_L" );
        sfree( workspace->Dinv_U, "Finalize_Workspace::Dinv_U" );
        sfree( workspace->Dinv_b, "Finalize_Workspace::Dinv_b" );
        sfree( workspace->rp, "Finalize_Workspace::rp" );
        sfree( workspace->rp2, "Finalize_Workspace::rp2" );
    }

    /* ILUTP preconditioner related */
    if ( control->cm_solver_pre_comp_type == ILUTP_PC )
    {
        sfree( workspace->perm_ilutp, "Finalize_Workspace::workspace->perm_ilutp" );
    }

#if defined(QMMM)
    sfree( workspace->mask_qmmm, "Init_Workspace::workspace->mask_qmmm" );
#endif

    /* integrator storage */
    sfree( workspace->a, "Finalize_Workspace::workspace->a" );
    sfree( workspace->f_old, "Finalize_Workspace::workspace->f_old" );
    sfree( workspace->v_const, "Finalize_Workspace::workspace->v_const" );

#if defined(_OPENMP)
    sfree( workspace->f_local, "Finalize_Workspace::workspace->f_local" );
#endif

    /* storage for analysis */
    if ( control->molec_anal || control->diffusion_coef )
    {
        sfree( workspace->mark, "Finalize_Workspace::workspace->mark" );
        sfree( workspace->old_mark, "Finalize_Workspace::workspace->old_mark" );
    }

    if ( control->diffusion_coef )
    {
        sfree( workspace->x_old, "Finalize_Workspace::workspace->x_old" );
    }

    if ( reset == FALSE )
    {
        sfree( workspace->orig_id, "Finalize_Workspace::workspace->orig_id" );

        /* space for keeping restriction info, if any */
        if ( control->restrict_bonds )
        {
            for ( i = 0; i < system->N; ++i )
            {
                sfree( workspace->restricted_list[i],
                        "Finalize_Workspace::workspace->restricted_list[i]" );
            }

            sfree( workspace->restricted, "Finalize_Workspace::workspace->restricted" );
            sfree( workspace->restricted_list, "Finalize_Workspace::workspace->restricted_list" );
        }
    }

#if defined(TEST_FORCES)
    sfree( workspace->dDelta, "Finalize_Workspace::workspace->dDelta" );
    sfree( workspace->f_ele, "Finalize_Workspace::workspace->f_ele" );
    sfree( workspace->f_vdw, "Finalize_Workspace::workspace->f_vdw" );
    sfree( workspace->f_be, "Finalize_Workspace::workspace->f_be" );
    sfree( workspace->f_lp, "Finalize_Workspace::workspace->f_lp" );
    sfree( workspace->f_ov, "Finalize_Workspace::workspace->f_ov" );
    sfree( workspace->f_un, "Finalize_Workspace::workspace->f_un" );
    sfree( workspace->f_ang, "Finalize_Workspace::workspace->f_ang" );
    sfree( workspace->f_coa, "Finalize_Workspace::workspace->f_coa" );
    sfree( workspace->f_pen, "Finalize_Workspace::workspace->f_pen" );
    sfree( workspace->f_hb, "Finalize_Workspace::workspace->f_hb" );
    sfree( workspace->f_tor, "Finalize_Workspace::workspace->f_tor" );
    sfree( workspace->f_con, "Finalize_Workspace::workspace->f_con" );
#endif
}


static void Finalize_Lists( reax_list **lists )
{
    if ( lists[FAR_NBRS]->allocated == TRUE )
    {
        Delete_List( TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );
    }
    if ( lists[HBONDS]->allocated == TRUE )
    {
        Delete_List( TYP_HBOND, lists[HBONDS] );
    }
    if ( lists[BONDS]->allocated == TRUE )
    {
        Delete_List( TYP_BOND, lists[BONDS] );
    }
    if ( lists[THREE_BODIES]->allocated == TRUE )
    {
        Delete_List( TYP_THREE_BODY, lists[THREE_BODIES] );
    }

#if defined(TEST_FORCES)
    if ( lists[DDELTA]->allocated == TRUE )
    {
        Delete_List( TYP_DDELTA, lists[DDELTA] );
    }
    if ( lists[DBO]->allocated == TRUE )
    {
        Delete_List( TYP_DBO, lists[DBO] );
    }
#endif
}


void Finalize_Out_Controls( reax_system *system, control_params *control,
        static_storage *workspace, output_controls *out_control )
{
    if ( out_control->write_steps > 0 )
    {
        sfclose( out_control->trj, "Finalize_Out_Controls::out_control->trj" );
    }

    if ( out_control->log_update_freq > 0 )
    {
        sfclose( out_control->out, "Finalize_Out_Controls::out_control->out" );
        sfclose( out_control->pot, "Finalize_Out_Controls::out_control->pot" );
        sfclose( out_control->log, "Finalize_Out_Controls::out_control->log" );
    }

    if ( control->ensemble == sNPT || control->ensemble == iNPT
            || control->ensemble == aNPT || control->compute_pressure == TRUE )
    {
        sfclose( out_control->prs, "Finalize_Out_Controls::out_control->prs" );
    }

    if ( control->molec_anal )
    {
        sfclose( out_control->mol, "Finalize_Out_Controls::out_control->mol" );

        if ( control->num_ignored )
        {
            sfclose( out_control->ign, "Finalize_Out_Controls::out_control->ign" );
        }
    }

    if ( control->dipole_anal )
    {
        sfclose( out_control->dpl, "Finalize_Out_Controls::out_control->dpl" );
    }

    if ( control->diffusion_coef )
    {
        sfclose( out_control->drft, "Finalize_Out_Controls::out_control->drft" );
    }


#if defined(TEST_ENERGY)
    sfclose( out_control->ebond, "Finalize_Out_Controls::out_control->ebond" );
    sfclose( out_control->elp, "Finalize_Out_Controls::out_control->elp" );
    sfclose( out_control->eov, "Finalize_Out_Controls::out_control->eov" );
    sfclose( out_control->eun, "Finalize_Out_Controls::out_control->eun" );
    sfclose( out_control->eval, "Finalize_Out_Controls::out_control->eval" );
    sfclose( out_control->epen, "Finalize_Out_Controls::out_control->epen" );
    sfclose( out_control->ecoa, "Finalize_Out_Controls::out_control->ecoa" );
    sfclose( out_control->ehb, "Finalize_Out_Controls::out_control->ehb" );
    sfclose( out_control->etor, "Finalize_Out_Controls::out_control->etor" );
    sfclose( out_control->econ, "Finalize_Out_Controls::out_control->econ" );
    sfclose( out_control->evdw, "Finalize_Out_Controls::out_control->evdw" );
    sfclose( out_control->ecou, "Finalize_Out_Controls::out_control->ecou" );
#endif

#if defined(TEST_FORCES)
    sfclose( out_control->fbo, "Finalize_Out_Controls::out_control->fbo" );
    sfclose( out_control->fdbo, "Finalize_Out_Controls::out_control->fdbo" );
    sfclose( out_control->fbond, "Finalize_Out_Controls::out_control->fbond" );
    sfclose( out_control->flp, "Finalize_Out_Controls::out_control->flp" );
    sfclose( out_control->fatom, "Finalize_Out_Controls::out_control->fatom" );
    sfclose( out_control->f3body, "Finalize_Out_Controls::out_control->f3body" );
    sfclose( out_control->fhb, "Finalize_Out_Controls::out_control->fhb" );
    sfclose( out_control->f4body, "Finalize_Out_Controls::out_control->f4body" );
    sfclose( out_control->fnonb, "Finalize_Out_Controls::out_control->fnonb" );
    sfclose( out_control->ftot, "Finalize_Out_Controls::out_control->ftot" );
    sfclose( out_control->ftot2, "Finalize_Out_Controls::out_control->ftot2" );
#endif
}


/* Deallocate top-level data structures, close file handles, etc.
 *
 */
void Finalize( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, reax_list **lists,
        output_controls *out_control, int output_enabled, int reset )
{
    if ( control->tabulate )
    {
        Finalize_LR_Lookup_Table( system, control, workspace );
    }

    if ( output_enabled == TRUE && reset == FALSE )
    {
        Finalize_Out_Controls( system, control, workspace, out_control );
    }

    Finalize_Lists( lists );

    Finalize_Workspace( system, control, workspace, reset );

    Finalize_Simulation_Data( system, control, data, out_control );

    Finalize_System( system, control, data, reset );
}
