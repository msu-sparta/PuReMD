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

#include "forces.h"

#include "box.h"
#include "bond_orders.h"
#include "bonds.h"
#include "charges.h"
#include "hydrogen_bonds.h"
#if defined(TEST_FORCES)
  #include "io_tools.h"
#endif
#include "list.h"
#include "multi_body.h"
#include "nonbonded.h"
#include "system_props.h"
#include "torsion_angles.h"
#include "tool_box.h"
#include "valence_angles.h"
#include "vector.h"


typedef enum
{
    DIAGONAL = 0,
    OFF_DIAGONAL = 1,
} MATRIX_ENTRY_POSITION;


#if defined(TEST_FORCES)
static int compare_bonds( const void *p1, const void *p2 )
{
    return ((bond_data *)p1)->nbr - ((bond_data *)p2)->nbr;
}
#endif


void Init_Bonded_Force_Functions( control_params *control )
{
    control->intr_funcs[0] = &BO;
    control->intr_funcs[1] = &Bonds;
    control->intr_funcs[2] = &Atom_Energy;
    control->intr_funcs[3] = &Valence_Angles;
    control->intr_funcs[4] = &Torsion_Angles;
    if ( control->hbond_cut > 0.0 )
    {
        control->intr_funcs[5] = &Hydrogen_Bonds;
    }
    else
    {
        control->intr_funcs[5] = NULL;
    }
    control->intr_funcs[6] = NULL;
    control->intr_funcs[7] = NULL;
    control->intr_funcs[8] = NULL;
    control->intr_funcs[9] = NULL;
}


static void Compute_Bonded_Forces( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i;

#if defined(TEST_ENERGY)
    /* Mark beginning of a new timestep in each energy file */
    fprintf( out_control->ebond, "step: %d\n%6s%6s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "bo", "ebond", "total" );
    fprintf( out_control->elp, "step: %d\n%6s%12s%12s%12s\n",
             data->step, "atom", "nlp", "elp", "total" );
    fprintf( out_control->eov, "step: %d\n%6s%12s%12s\n",
             data->step, "atom", "eov", "total" );
    fprintf( out_control->eun, "step: %d\n%6s%12s%12s\n",
             data->step, "atom", "eun", "total" );
    fprintf( out_control->eval, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3",
             "angle", "bo(12)", "bo(23)", "eval", "epen", "total" );
    fprintf( out_control->epen, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3",
             "angle", "bo(12)", "bo(23)", "epen", "total" );
    fprintf( out_control->ecoa, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3",
             "angle", "bo(12)", "bo(23)", "ecoa", "total" );
    fprintf( out_control->ehb,  "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3",
             "r(23)", "angle", "bo(12)", "ehb", "total" );
    fprintf( out_control->etor, "step: %d\n%6s%6s%6s%6s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3", "atom4",
             "phi", "bo(23)", "etor", "total" );
    fprintf( out_control->econ, "step:%d\n%6s%6s%6s%6s%12s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3", "atom4",
             "phi", "bo(12)", "bo(23)", "bo(34)", "econ", "total" );
#endif

    /* function calls for bonded interactions */
    for ( i = 0; i < NUM_INTRS; i++ )
    {
        if ( control->intr_funcs[i] != NULL )
        {
            (control->intr_funcs[i])( system, control, data, workspace,
                    lists, out_control );
        }
    }

#if defined(TEST_FORCES)
    /* function calls for printing bonded interactions */
    for ( i = 0; i < NUM_INTRS; i++ )
    {
        if ( control->print_intr_funcs[i] != NULL )
        {
            (control->print_intr_funcs[i])( system, control, data, workspace,
                    lists, out_control );
        }
    }
#endif
}


static void Compute_NonBonded_Forces( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list** lists, output_controls *out_control, int realloc )
{
    real t_start, t_elapsed;

#if defined(TEST_ENERGY)
    fprintf( out_control->evdw, "step: %d\n%6s%6s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "r12", "evdw", "total" );
    fprintf( out_control->ecou, "step: %d\n%6s%6s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "r12", "q1", "q2", "ecou", "total" );
#endif

    t_start = Get_Time( );
    Compute_Charges( system, control, data, workspace, out_control, realloc );
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.cm += t_elapsed;
    
    if ( control->cm_solver_pre_comp_refactor == -1 )
    {
        if ( data->step <= 4 || is_refactoring_step( control, data ) )
        {
            if ( is_refactoring_step( control, data ) )
            {
                data->timing.cm_last_pre_comp = data->timing.cm_solver_pre_comp;
            }

            data->timing.cm_optimum = data->timing.cm_solver_pre_app
                + data->timing.cm_solver_spmv
                + data->timing.cm_solver_vector_ops
                + data->timing.cm_solver_orthog
                + data->timing.cm_solver_tri_solve;
            data->timing.cm_total_loss = 0.0;
        }
        else
        {
            data->timing.cm_total_loss += data->timing.cm_solver_pre_app
                + data->timing.cm_solver_spmv
                + data->timing.cm_solver_vector_ops
                + data->timing.cm_solver_orthog
                + data->timing.cm_solver_tri_solve
                - data->timing.cm_optimum;
        }
    }

    if ( control->tabulate <= 0 )
    {
        vdW_Coulomb_Energy( system, control, data, workspace, lists, out_control );
    }
    else
    {
        Tabulated_vdW_Coulomb_Energy( system, control, data, workspace,
                lists, out_control );
    }

#if defined(TEST_FORCES)
    Print_vdW_Coulomb_Forces( system, control, data, workspace,
            lists, out_control );
#endif
}


/* This version of Compute_Total_Force computes forces from coefficients
   accumulated by all interaction functions. Saves enormous time & space! */
static void Compute_Total_Force( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, reax_list **lists )
{
    int i;
    reax_list *bonds;

    bonds = lists[BONDS];

#if defined(_OPENMP)
    #pragma omp parallel default(shared)
#endif
    {
        int pj;
#if defined(_OPENMP)
        int j;
#endif

        if ( control->compute_pressure == FALSE
                && (control->ensemble == NVE || control->ensemble == nhNVT
                    || control->ensemble == bNVT) )
        {
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for ( i = 0; i < system->N; ++i )
            {
                for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
                {
                    if ( i <= bonds->bond_list[pj].nbr )
                    {
                        Add_dBond_to_Forces( i, pj, system, data, workspace, lists );
                    }
                }
            }
        }
        else if ( control->ensemble == sNPT || control->ensemble == iNPT
                || control->ensemble == aNPT || control->compute_pressure == TRUE )
        {
#if defined(_OPENMP)
            #pragma omp for schedule(static)
#endif
            for ( i = 0; i < system->N; ++i )
            {
                for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
                {
                    if ( i <= bonds->bond_list[pj].nbr )
                    {
                        Add_dBond_to_Forces_NPT( i, pj, system, data, workspace, lists );
                    }
                }
            }
        }

#if defined(_OPENMP)
        /* reduction (sum) on thread-local force vectors */
        #pragma omp for schedule(static)
        for ( i = 0; i < system->N; ++i )
        {
            for ( j = 0; j < control->num_threads; ++j )
            {
                rvec_Add( system->atoms[i].f, workspace->f_local[j * system->N + i] );
            }
        }
#endif
    }
}


static void Validate_Lists( static_storage *workspace, reax_list **lists, int step, int n,
        int Hmax, int Htop, int num_bonds, int num_hbonds )
{
    int i, flag;
    reax_list *bonds, *hbonds;

    bonds = lists[BONDS];
    hbonds = lists[HBONDS];

    /* far neighbors */
    if ( Htop > Hmax * DANGER_ZONE )
    {
        workspace->realloc.Htop = Htop;
        if ( Htop > Hmax )
        {
            fprintf( stderr,
                     "[ERROR] step%d - ran out of space on H matrix: Htop=%d, max = %d",
                     step, Htop, Hmax );
            exit( INSUFFICIENT_MEMORY );
        }
    }

    /* bond list */
    flag = -1;
    workspace->realloc.num_bonds = num_bonds;
    for ( i = 0; i < n - 1; ++i )
    {
        if ( End_Index(i, bonds) >= Start_Index(i + 1, bonds) - 2 )
        {
            workspace->realloc.bonds = 1;
            if ( End_Index(i, bonds) > Start_Index(i + 1, bonds) )
            {
                flag = i;
            }
        }
    }

    if ( flag > -1 )
    {
        fprintf( stderr, "[ERROR] step%d-bondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                 step, flag, End_Index(flag, bonds), Start_Index(flag + 1, bonds) );
        exit( INSUFFICIENT_MEMORY );
    }

    if ( End_Index(i, bonds) >= bonds->total_intrs - 2 )
    {
        workspace->realloc.bonds = 1;

        if ( End_Index(i, bonds) > bonds->total_intrs )
        {
            fprintf( stderr, "[ERROR] step%d-bondchk failed: i=%d end(i)=%d bond_end=%d\n",
                     step, flag, End_Index(i, bonds), bonds->total_intrs );
            exit( INSUFFICIENT_MEMORY );
        }
    }


    /* hbonds list */
    if ( workspace->num_H > 0 )
    {
        flag = -1;
        workspace->realloc.num_hbonds = num_hbonds;
        for ( i = 0; i < workspace->num_H - 1; ++i )
        {
            if ( Num_Entries(i, hbonds) >=
                    (Start_Index(i + 1, hbonds) - Start_Index(i, hbonds)) * DANGER_ZONE )
            {
                workspace->realloc.hbonds = 1;
                if ( End_Index(i, hbonds) > Start_Index(i + 1, hbonds) )
                {
                    flag = i;
                }
            }
        }

        if ( flag > -1 )
        {
            fprintf( stderr, "[ERROR] step%d-hbondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                     step, flag, End_Index(flag, hbonds), Start_Index(flag + 1, hbonds) );
            exit( INSUFFICIENT_MEMORY );
        }

        if ( Num_Entries(i, hbonds) >=
                (hbonds->total_intrs - Start_Index(i, hbonds)) * DANGER_ZONE )
        {
            workspace->realloc.hbonds = 1;

            if ( End_Index(i, hbonds) > hbonds->total_intrs )
            {
                fprintf( stderr, "[ERROR] step%d-hbondchk failed: i=%d end(i)=%d hbondend=%d\n",
                         step, flag, End_Index(i, hbonds), hbonds->total_intrs );
                exit( INSUFFICIENT_MEMORY );
            }
        }
    }
}


static inline real Init_Charge_Matrix_Entry_Tab( reax_system *system,
        control_params *control, LR_lookup_table **LR, int i, int j,
        real r_ij, MATRIX_ENTRY_POSITION pos )
{
    int r;
    real base, dif, val, ret;
    LR_lookup_table *t;

    ret = 0.0;

    switch ( control->charge_method )
    {
    case QEQ_CM:
    //TODO: tabulate other portions of matrices for EE, ACKS2?
    case EE_CM:
    case ACKS2_CM:
        switch ( pos )
        {
            case OFF_DIAGONAL:
                t = &LR[MIN( system->atoms[i].type, system->atoms[j].type )]
                       [MAX( system->atoms[i].type, system->atoms[j].type )];

                /* cubic spline interpolation */
                r = (int)(r_ij * t->inv_dx);
                if ( r == 0 ) 
                {
                    ++r;
                }
                base = (real)(r + 1) * t->dx;
                dif = r_ij - base;
                val = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b)
                    * dif + t->ele[r].a;
                val *= EV_to_KCALpMOL / C_ELE;

                ret = ((i == j) ? 0.5 : 1.0) * val;
            break;

            case DIAGONAL:
                ret = system->reax_param.sbp[system->atoms[i].type].eta;
            break;

            default:
                fprintf( stderr, "[ERROR] Invalid matrix position. Terminating...\n" );
                exit( INVALID_INPUT );
            break;
        }
        break;

    default:
        fprintf( stderr, "[ERROR] Invalid charge method. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    return ret;
}


static inline real Init_Charge_Matrix_Entry( reax_system *system,
        control_params *control, static_storage *workspace,
        int i, int j, real r_ij, MATRIX_ENTRY_POSITION pos )
{
    real Tap, dr3gamij_1, dr3gamij_3, ret;

    ret = 0.0;

    switch ( control->charge_method )
    {
    case QEQ_CM:
    case EE_CM:
    case ACKS2_CM:
        switch ( pos )
        {
            case OFF_DIAGONAL:
                Tap = workspace->Tap[7] * r_ij + workspace->Tap[6];
                Tap = Tap * r_ij + workspace->Tap[5];
                Tap = Tap * r_ij + workspace->Tap[4];
                Tap = Tap * r_ij + workspace->Tap[3];
                Tap = Tap * r_ij + workspace->Tap[2];
                Tap = Tap * r_ij + workspace->Tap[1];
                Tap = Tap * r_ij + workspace->Tap[0];

                /* shielding */
                dr3gamij_1 = r_ij * r_ij * r_ij
                        + POW( system->reax_param.tbp[system->atoms[i].type][system->atoms[j].type].gamma, -3.0 );
                dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

                /* i == j: periodic self-interaction term
                 * i != j: general interaction term */
                ret = ((i == j) ? 0.5 : 1.0) * Tap * EV_to_KCALpMOL / dr3gamij_3;
            break;

            case DIAGONAL:
                ret = system->reax_param.sbp[system->atoms[i].type].eta;
            break;

            default:
                fprintf( stderr, "[ERROR] Invalid matrix position. Terminating...\n" );
                exit( INVALID_INPUT );
            break;
        }
        break;

    default:
        fprintf( stderr, "[ERROR] Invalid charge method. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    return ret;
}


static void Init_Charge_Matrix_Remaining_Entries( reax_system *system,
        control_params *control, reax_list *far_nbr_list,
        sparse_matrix * H, sparse_matrix * H_sp,
        int * Htop, int * H_sp_top )
{
    int i, j, pj, target, val_flag;
    real d, xcut, bond_softness, * X_diag;

    switch ( control->charge_method )
    {
        case QEQ_CM:
            break;

        case EE_CM:
            H->start[system->N_cm - 1] = *Htop;
            H_sp->start[system->N_cm - 1] = *H_sp_top;

            for ( i = 0; i < system->N_cm - 1; ++i )
            {
                H->j[*Htop] = i;
                H->val[*Htop] = 1.0;
                *Htop = *Htop + 1;

                H_sp->j[*H_sp_top] = i;
                H_sp->val[*H_sp_top] = 1.0;
                *H_sp_top = *H_sp_top + 1;
            }

            H->j[*Htop] = system->N_cm - 1;
            H->val[*Htop] = 0.0;
            *Htop = *Htop + 1;

            H_sp->j[*H_sp_top] = system->N_cm - 1;
            H_sp->val[*H_sp_top] = 0.0;
            *H_sp_top = *H_sp_top + 1;
            break;

        case ACKS2_CM:
            X_diag = smalloc( sizeof(real) * system->N,
                    "Init_Charge_Matrix_Remaining_Entries::X_diag" );

            for ( i = 0; i < system->N; ++i )
            {
                X_diag[i] = 0.0;
            }

            for ( i = 0; i < system->N; ++i )
            {
                H->start[system->N + i] = *Htop;
                H_sp->start[system->N + i] = *H_sp_top;

                /* constraint on ref. value for kinetic energy potential */
                H->j[*Htop] = i;
                H->val[*Htop] = 1.0;
                *Htop = *Htop + 1;

                H_sp->j[*H_sp_top] = i;
                H_sp->val[*H_sp_top] = 1.0;
                *H_sp_top = *H_sp_top + 1;

                /* kinetic energy terms */
                for ( pj = Start_Index(i, far_nbr_list); pj < End_Index(i, far_nbr_list); ++pj )
                {
                    /* exclude self-periodic images of atoms for
                     * kinetic energy term because the effective
                     * potential is the same on an atom and its periodic image */
                    if ( far_nbr_list->far_nbr_list[pj].d <= control->nonb_cut )
                    {
                        j = far_nbr_list->far_nbr_list[pj].nbr;

                        xcut = 0.5 * ( system->reax_param.sbp[ system->atoms[i].type ].b_s_acks2
                                + system->reax_param.sbp[ system->atoms[j].type ].b_s_acks2 );

                        if ( far_nbr_list->far_nbr_list[pj].d < xcut )
                        {
                            d = far_nbr_list->far_nbr_list[pj].d / xcut;
                            bond_softness = system->reax_param.gp.l[34] * POW( d, 3.0 )
                                * POW( 1.0 - d, 6.0 );

                            if ( bond_softness > 0.0 )
                            {
                                val_flag = FALSE;

                                for ( target = H->start[system->N + i]; target < *Htop; ++target )
                                {
                                    if ( H->j[target] == system->N + j )
                                    {
                                        H->val[target] += bond_softness;
                                        val_flag = TRUE;
                                        break;
                                    }
                                }

                                if ( val_flag == FALSE )
                                {
                                    H->j[*Htop] = system->N + j;
                                    H->val[*Htop] = bond_softness;
                                    ++(*Htop);
                                }

                                val_flag = FALSE;

                                for ( target = H_sp->start[system->N + i]; target < *H_sp_top; ++target )
                                {
                                    if ( H_sp->j[target] == system->N + j )
                                    {
                                        H_sp->val[target] += bond_softness;
                                        val_flag = TRUE;
                                        break;
                                    }
                                }

                                if ( val_flag == FALSE )
                                {
                                    H_sp->j[*H_sp_top] = system->N + j;
                                    H_sp->val[*H_sp_top] = bond_softness;
                                    ++(*H_sp_top);
                                }

                                X_diag[i] -= bond_softness;
                                X_diag[j] -= bond_softness;
                            }
                        }
                    }
                }

                /* placeholders for diagonal entries, to be replaced below */
                H->j[*Htop] = system->N + i;
                H->val[*Htop] = 0.0;
                *Htop = *Htop + 1;

                H_sp->j[*H_sp_top] = system->N + i;
                H_sp->val[*H_sp_top] = 0.0;
                *H_sp_top = *H_sp_top + 1;
            }

            /* second to last row */
            H->start[system->N_cm - 2] = *Htop;
            H_sp->start[system->N_cm - 2] = *H_sp_top;

            /* place accumulated diagonal entries (needed second to last row marker above before this code) */
            for ( i = system->N; i < 2 * system->N; ++i )
            {
                for ( pj = H->start[i]; pj < H->start[i + 1]; ++pj )
                {
                    if ( H->j[pj] == i )
                    {
                        H->val[pj] = X_diag[i - system->N];
                        break;
                    }
                }

                for ( pj = H_sp->start[i]; pj < H_sp->start[i + 1]; ++pj )
                {
                    if ( H_sp->j[pj] == i )
                    {
                        H_sp->val[pj] = X_diag[i - system->N];
                        break;
                    }
                }
            }

            /* coupling with the kinetic energy potential */
            for ( i = 0; i < system->N; ++i )
            {
                H->j[*Htop] = system->N + i;
                H->val[*Htop] = 1.0;
                *Htop = *Htop + 1;

                H_sp->j[*H_sp_top] = system->N + i;
                H_sp->val[*H_sp_top] = 1.0;
                *H_sp_top = *H_sp_top + 1;
            }

            /* explicitly store zero on diagonal */
            H->j[*Htop] = system->N_cm - 2;
            H->val[*Htop] = 0.0;
            *Htop = *Htop + 1;

            H_sp->j[*H_sp_top] = system->N_cm - 2;
            H_sp->val[*H_sp_top] = 0.0;
            *H_sp_top = *H_sp_top + 1;

            /* last row */
            H->start[system->N_cm - 1] = *Htop;
            H_sp->start[system->N_cm - 1] = *H_sp_top;

            for ( i = 0; i < system->N; ++i )
            {
                H->j[*Htop] = i;
                H->val[*Htop] = 1.0;
                *Htop = *Htop + 1;

                H_sp->j[*H_sp_top] = i;
                H_sp->val[*H_sp_top] = 1.0;
                *H_sp_top = *H_sp_top + 1;
            }

            /* explicitly store zero on diagonal */
            H->j[*Htop] = system->N_cm - 1;
            H->val[*Htop] = 0.0;
            *Htop = *Htop + 1;

            H_sp->j[*H_sp_top] = system->N_cm - 1;
            H_sp->val[*H_sp_top] = 0.0;
            *H_sp_top = *H_sp_top + 1;

            sfree( X_diag, "Init_Charge_Matrix_Remaining_Entries::X_diag" );
            break;

        default:
            break;
    }
}


/* Generate bond list (full format), hydrogen bond list (full format),
 * and charge matrix (half symmetric format)
 * from the far neighbors list (with distance updates, if necessary)  */
static void Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, j, pj, target;
    int start_i, end_i;
    int type_i, type_j;
    int Htop, H_sp_top, btop_i, btop_j, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top, jhb_top;
    int flag, flag_sp, val_flag, renbr;
    real r_ij, r2, val;
    real C12, C34, C56;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
    real BO, BO_s, BO_pi, BO_pi2;
    sparse_matrix *H, *H_sp;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
    bond_data *ibond, *jbond;
    bond_order_data *bo_ij, *bo_ji;

    far_nbrs = lists[FAR_NBRS];
    bonds = lists[BONDS];
    hbonds = lists[HBONDS];
    H = &workspace->H;
    H_sp = &workspace->H_sp;
    Htop = 0;
    H_sp_top = 0;
    num_bonds = 0;
    num_hbonds = 0;
    btop_i = 0;
    btop_j = 0;
    renbr = ((data->step - data->prev_steps) % control->reneighbor) == 0 ? TRUE : FALSE;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );
        H->start[i] = Htop;
        H_sp->start[i] = H_sp_top;
        btop_i = End_Index( i, bonds );
        sbp_i = &system->reax_param.sbp[type_i];

        if ( control->hbond_cut > 0.0 )
        {
            ihb = sbp_i->p_hbond;

            if ( ihb == H_ATOM )
            {
                ihb_top = End_Index( workspace->hbond_index[i], hbonds );
            }
            else
            {
                ihb_top = -1;
            }
        }
        else
        {
            ihb = NON_H_BONDING_ATOM;
            ihb_top = -1;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &far_nbrs->far_nbr_list[pj];
            j = nbr_pj->nbr;
            flag = FALSE;
            flag_sp = FALSE;

            /* check if reneighboring step --
             * atomic distances just computed via
             * Verlet list, so use current distances */
            if ( renbr == TRUE )
            {
                if ( nbr_pj->d <= control->nonb_cut )
                {
                    flag = TRUE;

                    if ( nbr_pj->d <= control->nonb_sp_cut )
                    {
                        flag_sp = TRUE;
                    }
                }
            }
            /* update atomic distances */
            else
            {
                atom_j = &system->atoms[j];
                nbr_pj->d = control->compute_atom_distance( &system->box,
                        atom_i->x, atom_j->x, atom_i->rel_map,
                        atom_j->rel_map, nbr_pj->rel_box,
                        nbr_pj->dvec );

                if ( nbr_pj->d <= control->nonb_cut )
                {
                    flag = TRUE;

                    if ( nbr_pj->d <= control->nonb_sp_cut )
                    {
                        flag_sp = TRUE;
                    }
                }
            }

            if ( flag == TRUE )
            {
                type_j = system->atoms[j].type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[type_i][type_j];
                r_ij = nbr_pj->d;

                val = Init_Charge_Matrix_Entry( system, control,
                            workspace, i, j, r_ij, OFF_DIAGONAL );
                val_flag = FALSE;

                for ( target = H->start[i]; target < Htop; ++target )
                {
                    if ( H->j[target] == j )
                    {
                        H->val[target] += val;
                        val_flag = TRUE;
                        break;
                    }
                }

                if ( val_flag == FALSE )
                {
                    H->j[Htop] = j;
                    H->val[Htop] = val;
                    ++Htop;
                }

                /* H_sp matrix entry */
                if ( flag_sp == TRUE )
                {
                    val_flag = FALSE;

                    for ( target = H_sp->start[i]; target < H_sp_top; ++target )
                    {
                        if ( H_sp->j[target] == j )
                        {
                            H_sp->val[target] += val;
                            val_flag = TRUE;
                            break;
                        }
                    }

                    if ( val_flag == FALSE )
                    {
                        H_sp->j[H_sp_top] = j;
                        H_sp->val[H_sp_top] = val;
                        ++H_sp_top;
                    }
                }

                /* hydrogen bond lists */
                if ( control->hbond_cut > 0.0
                        && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                        && nbr_pj->d <= control->hbond_cut )
                {
                    jhb = sbp_j->p_hbond;

                    if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                    {
                        hbonds->hbond_list[ihb_top].nbr = j;
                        hbonds->hbond_list[ihb_top].scl = 1;
                        hbonds->hbond_list[ihb_top].ptr = nbr_pj;
                        ++ihb_top;
                        ++num_hbonds;
                    }
                    else if ( ihb == H_BONDING_ATOM && jhb == H_ATOM )
                    {
                        jhb_top = End_Index( workspace->hbond_index[j], hbonds );
                        hbonds->hbond_list[jhb_top].nbr = i;
                        hbonds->hbond_list[jhb_top].scl = -1;
                        hbonds->hbond_list[jhb_top].ptr = nbr_pj;
                        Set_End_Index( workspace->hbond_index[j], jhb_top + 1, hbonds );
                        ++num_hbonds;
                    }
                }

                /* uncorrected bond orders */
                if ( nbr_pj->d <= control->bond_cut )
                {
                    r2 = SQR( r_ij );

                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0 )
                    {
                        C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                    }
                    else
                    {
                        C12 = 0.0;
                        BO_s = 0.0;
                    }

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0 )
                    {
                        C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = EXP( C34 );
                    }
                    else
                    {
                        C34 = 0.0;
                        BO_pi = 0.0;
                    }

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0 )
                    {
                        C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = EXP( C56 );
                    }
                    else
                    {
                        C56 = 0.0;
                        BO_pi2 = 0.0;
                    }

                    /* Initially BO values are the uncorrected ones, page 1 */
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        num_bonds += 2;
                        /****** bonds i-j and j-i ******/
                        ibond = &bonds->bond_list[btop_i];
                        btop_j = End_Index( j, bonds );
                        jbond = &bonds->bond_list[btop_j];

                        ibond->nbr = j;
                        jbond->nbr = i;
                        ibond->d = r_ij;
                        jbond->d = r_ij;
                        rvec_Copy( ibond->dvec, nbr_pj->dvec );
                        rvec_Scale( jbond->dvec, -1, nbr_pj->dvec );
                        ivec_Copy( ibond->rel_box, nbr_pj->rel_box );
                        ivec_Scale( jbond->rel_box, -1, nbr_pj->rel_box );
                        ibond->dbond_index = btop_i;
                        jbond->dbond_index = btop_i;
                        ibond->sym_index = btop_j;
                        jbond->sym_index = btop_i;
                        ++btop_i;
                        Set_End_Index( j, btop_j + 1, bonds );

                        bo_ij = &ibond->bo_data;
                        bo_ij->BO = BO;
                        bo_ij->BO_s = BO_s;
                        bo_ij->BO_pi = BO_pi;
                        bo_ij->BO_pi2 = BO_pi2;
                        bo_ji = &jbond->bo_data;
                        bo_ji->BO = BO;
                        bo_ji->BO_s = BO_s;
                        bo_ji->BO_pi = BO_pi;
                        bo_ji->BO_pi2 = BO_pi2;

                        /* Bond Order page2-3, derivative of total bond order prime */
                        Cln_BOp_s = twbp->p_bo2 * C12 / r2;
                        Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
                        Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

                        /* Only dln_BOp_xx wrt. dr_i is stored here, note that
                           dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
                        rvec_Scale( bo_ij->dln_BOp_s, -bo_ij->BO_s * Cln_BOp_s, ibond->dvec );
                        rvec_Scale( bo_ij->dln_BOp_pi, -bo_ij->BO_pi * Cln_BOp_pi, ibond->dvec );
                        rvec_Scale( bo_ij->dln_BOp_pi2,
                                -bo_ij->BO_pi2 * Cln_BOp_pi2, ibond->dvec );
                        rvec_Scale( bo_ji->dln_BOp_s, -1., bo_ij->dln_BOp_s );
                        rvec_Scale( bo_ji->dln_BOp_pi, -1., bo_ij->dln_BOp_pi );
                        rvec_Scale( bo_ji->dln_BOp_pi2, -1., bo_ij->dln_BOp_pi2 );

                        /* Only dBOp wrt. dr_i is stored here, note that
                           dBOp/dr_i = -dBOp/dr_j and all others are 0 */
                        rvec_Scale( bo_ij->dBOp, -(bo_ij->BO_s * Cln_BOp_s
                                    + bo_ij->BO_pi * Cln_BOp_pi
                                    + bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );
                        rvec_Scale( bo_ji->dBOp, -1., bo_ij->dBOp );

                        rvec_Add( workspace->dDeltap_self[i], bo_ij->dBOp );
                        rvec_Add( workspace->dDeltap_self[j], bo_ji->dBOp );

                        bo_ij->BO_s -= control->bo_cut;
                        bo_ij->BO -= control->bo_cut;
                        bo_ji->BO_s -= control->bo_cut;
                        bo_ji->BO -= control->bo_cut;
                        workspace->total_bond_order[i] += bo_ij->BO;
                        workspace->total_bond_order[j] += bo_ji->BO;
                        bo_ij->Cdbo = 0.0;
                        bo_ij->Cdbopi = 0.0;
                        bo_ij->Cdbopi2 = 0.0;
                        bo_ji->Cdbo = 0.0;
                        bo_ji->Cdbopi = 0.0;
                        bo_ji->Cdbopi2 = 0.0;

                        Set_End_Index( j, btop_j + 1, bonds );
                    }
                }
            }
        }

        /* diagonal entry */
        H->j[Htop] = i;
        H->val[Htop] = Init_Charge_Matrix_Entry( system, control,
                workspace, i, i, r_ij, DIAGONAL );
        ++Htop;

        H_sp->j[H_sp_top] = i;
        H_sp->val[H_sp_top] = H->val[Htop - 1];
        ++H_sp_top;

        Set_End_Index( i, btop_i, bonds );
        if ( ihb == H_ATOM )
        {
            Set_End_Index( workspace->hbond_index[i], ihb_top, hbonds );
        }
    }

    Init_Charge_Matrix_Remaining_Entries( system, control, far_nbrs,
            H, H_sp, &Htop, &H_sp_top );

    H->start[system->N_cm] = Htop;
    H_sp->start[system->N_cm] = H_sp_top;

    /* validate lists - decide if reallocation is required! */
    Validate_Lists( workspace, lists,
            data->step, system->N, H->m, Htop, num_bonds, num_hbonds );

#if defined(TEST_FORCES)
    /* Calculate_dBO requires a sorted bonds list */
//    for ( i = 0; i < bonds->n; ++i )
//    {
//        if ( Num_Entries(i, bonds) > 0 )
//        {
//            qsort( &bonds->bond_list[Start_Index(i, bonds)],
//                    Num_Entries(i, bonds), sizeof(bond_data), compare_bonds );
//        }
//    }
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "step%d: Htop = %d, num_bonds = %d, num_hbonds = %d\n",
             data->step, Htop, num_bonds, num_hbonds );

#endif
}


static void Init_Forces_Tab( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, j, pj, target;
    int start_i, end_i;
    int type_i, type_j;
    int Htop, H_sp_top, btop_i, btop_j, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top, jhb_top;
    int flag, flag_sp, val_flag, renbr;
    real r_ij, r2, val;
    real C12, C34, C56;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
    real BO, BO_s, BO_pi, BO_pi2;
    sparse_matrix *H, *H_sp;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
    bond_data *ibond, *jbond;
    bond_order_data *bo_ij, *bo_ji;

    far_nbrs = lists[FAR_NBRS];
    bonds = lists[BONDS];
    hbonds = lists[HBONDS];
    H = &workspace->H;
    H_sp = &workspace->H_sp;
    Htop = 0;
    H_sp_top = 0;
    num_bonds = 0;
    num_hbonds = 0;
    btop_i = 0;
    btop_j = 0;
    renbr = ((data->step - data->prev_steps) % control->reneighbor) == 0 ? TRUE : FALSE;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );
        H->start[i] = Htop;
        H_sp->start[i] = H_sp_top;
        btop_i = End_Index( i, bonds );
        sbp_i = &system->reax_param.sbp[type_i];

        if ( control->hbond_cut > 0.0 )
        {
            ihb = sbp_i->p_hbond;
            if ( ihb == H_ATOM )
            {
                ihb_top = End_Index( workspace->hbond_index[i], hbonds );
            }
            else
            {
                ihb_top = -1;
            }
        }
        else
        {
            ihb = NON_H_BONDING_ATOM;
            ihb_top = -1;
        }

        /* update i-j distance - check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &far_nbrs->far_nbr_list[pj];
            j = nbr_pj->nbr;
            flag = FALSE;
            flag_sp = FALSE;

            /* check if reneighboring step --
             * atomic distances just computed via
             * Verlet list, so use current distances */
            if ( renbr == TRUE )
            {
                if ( nbr_pj->d <= control->nonb_cut )
                {
                    flag = TRUE;

                    if ( nbr_pj->d <= control->nonb_sp_cut )
                    {
                        flag_sp = TRUE;
                    }
                }
            }
            /* update atomic distances */
            else
            {
                atom_j = &system->atoms[j];
                nbr_pj->d = control->compute_atom_distance( &system->box,
                        atom_i->x, atom_j->x, atom_i->rel_map,
                        atom_j->rel_map, nbr_pj->rel_box,
                        nbr_pj->dvec );

                if ( nbr_pj->d <= control->nonb_cut )
                {
                    flag = TRUE;

                    if ( nbr_pj->d <= control->nonb_sp_cut )
                    {
                        flag_sp = TRUE;
                    }
                }
            }

            if ( flag == TRUE )
            {
                type_j = system->atoms[j].type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[type_i][type_j];
                r_ij = nbr_pj->d;

                val = Init_Charge_Matrix_Entry( system, control,
                        workspace, i, j, r_ij, OFF_DIAGONAL );
                val_flag = FALSE;

                for ( target = H->start[i]; target < Htop; ++target )
                {
                    if ( H->j[target] == j )
                    {
                        H->val[target] += val;
                        val_flag = TRUE;
                        break;
                    }
                }

                if ( val_flag == FALSE )
                {
                    H->j[Htop] = j;
                    H->val[Htop] = val;
                    ++Htop;
                }

                /* H_sp matrix entry */
                if ( flag_sp == TRUE )
                {
                    val_flag = FALSE;

                    for ( target = H_sp->start[i]; target < H_sp_top; ++target )
                    {
                        if ( H_sp->j[target] == j )
                        {
                            H_sp->val[target] += val;
                            val_flag = TRUE;
                            break;
                        }
                    }

                    if ( val_flag == FALSE )
                    {
                        H_sp->j[H_sp_top] = j;
                        H_sp->val[H_sp_top] = val;
                        ++H_sp_top;
                    }
                }

                /* hydrogen bond lists */
                if ( control->hbond_cut > 0.0
                        && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                        && nbr_pj->d <= control->hbond_cut )
                {
                    jhb = sbp_j->p_hbond;

                    if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                    {
                        hbonds->hbond_list[ihb_top].nbr = j;
                        hbonds->hbond_list[ihb_top].scl = 1;
                        hbonds->hbond_list[ihb_top].ptr = nbr_pj;
                        ++ihb_top;
                        ++num_hbonds;
                    }
                    else if ( ihb == H_BONDING_ATOM && jhb == H_ATOM )
                    {
                        jhb_top = End_Index( workspace->hbond_index[j], hbonds );
                        hbonds->hbond_list[jhb_top].nbr = i;
                        hbonds->hbond_list[jhb_top].scl = -1;
                        hbonds->hbond_list[jhb_top].ptr = nbr_pj;
                        Set_End_Index( workspace->hbond_index[j], jhb_top + 1, hbonds );
                        ++num_hbonds;
                    }
                }

                /* uncorrected bond orders */
                if ( nbr_pj->d <= control->bond_cut )
                {
                    r2 = SQR( r_ij );

                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0 )
                    {
                        C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                    }
                    else
                    {
                        C12 = 0.0;
                        BO_s = 0.0;
                    }

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0 )
                    {
                        C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = EXP( C34 );
                    }
                    else
                    {
                        C34 = 0.0;
                        BO_pi = 0.0;
                    }

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0 )
                    {
                        C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = EXP( C56 );
                    }
                    else
                    {
                        C56 = 0.0;
                        BO_pi2 = 0.0;
                    }

                    /* Initially BO values are the uncorrected ones, page 1 */
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        num_bonds += 2;
                        /****** bonds i-j and j-i ******/
                        ibond = &bonds->bond_list[btop_i];
                        btop_j = End_Index( j, bonds );
                        jbond = &bonds->bond_list[btop_j];

                        ibond->nbr = j;
                        jbond->nbr = i;
                        ibond->d = r_ij;
                        jbond->d = r_ij;
                        rvec_Copy( ibond->dvec, nbr_pj->dvec );
                        rvec_Scale( jbond->dvec, -1, nbr_pj->dvec );
                        ivec_Copy( ibond->rel_box, nbr_pj->rel_box );
                        ivec_Scale( jbond->rel_box, -1, nbr_pj->rel_box );
                        ibond->dbond_index = btop_i;
                        jbond->dbond_index = btop_i;
                        ibond->sym_index = btop_j;
                        jbond->sym_index = btop_i;
                        ++btop_i;
                        Set_End_Index( j, btop_j + 1, bonds );

                        bo_ij = &ibond->bo_data;
                        bo_ij->BO = BO;
                        bo_ij->BO_s = BO_s;
                        bo_ij->BO_pi = BO_pi;
                        bo_ij->BO_pi2 = BO_pi2;
                        bo_ji = &jbond->bo_data;
                        bo_ji->BO = BO;
                        bo_ji->BO_s = BO_s;
                        bo_ji->BO_pi = BO_pi;
                        bo_ji->BO_pi2 = BO_pi2;

                        /* Bond Order page2-3, derivative of total bond order prime */
                        Cln_BOp_s = twbp->p_bo2 * C12 / r2;
                        Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
                        Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

                        /* Only dln_BOp_xx wrt. dr_i is stored here, note that
                           dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
                        rvec_Scale( bo_ij->dln_BOp_s, -bo_ij->BO_s * Cln_BOp_s, ibond->dvec );
                        rvec_Scale( bo_ij->dln_BOp_pi, -bo_ij->BO_pi * Cln_BOp_pi, ibond->dvec );
                        rvec_Scale( bo_ij->dln_BOp_pi2,
                                -bo_ij->BO_pi2 * Cln_BOp_pi2, ibond->dvec );
                        rvec_Scale( bo_ji->dln_BOp_s, -1., bo_ij->dln_BOp_s );
                        rvec_Scale( bo_ji->dln_BOp_pi, -1., bo_ij->dln_BOp_pi );
                        rvec_Scale( bo_ji->dln_BOp_pi2, -1., bo_ij->dln_BOp_pi2 );

                        /* Only dBOp wrt. dr_i is stored here, note that
                           dBOp/dr_i = -dBOp/dr_j and all others are 0 */
                        rvec_Scale( bo_ij->dBOp, -(bo_ij->BO_s * Cln_BOp_s
                                    + bo_ij->BO_pi * Cln_BOp_pi
                                    + bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );
                        rvec_Scale( bo_ji->dBOp, -1., bo_ij->dBOp );

                        rvec_Add( workspace->dDeltap_self[i], bo_ij->dBOp );
                        rvec_Add( workspace->dDeltap_self[j], bo_ji->dBOp );

                        bo_ij->BO_s -= control->bo_cut;
                        bo_ij->BO -= control->bo_cut;
                        bo_ji->BO_s -= control->bo_cut;
                        bo_ji->BO -= control->bo_cut;
                        workspace->total_bond_order[i] += bo_ij->BO;
                        workspace->total_bond_order[j] += bo_ji->BO;
                        bo_ij->Cdbo = 0.0;
                        bo_ij->Cdbopi = 0.0;
                        bo_ij->Cdbopi2 = 0.0;
                        bo_ji->Cdbo = 0.0;
                        bo_ji->Cdbopi = 0.0;
                        bo_ji->Cdbopi2 = 0.0;

                        Set_End_Index( j, btop_j + 1, bonds );
                    }
                }
            }
        }

        /* diagonal entry */
        H->j[Htop] = i;
        H->val[Htop] = Init_Charge_Matrix_Entry( system, control,
                workspace, i, i, r_ij, DIAGONAL );
        ++Htop;

        H_sp->j[H_sp_top] = i;
        H_sp->val[H_sp_top] = H->val[Htop - 1];
        ++H_sp_top;

        Set_End_Index( i, btop_i, bonds );
        if ( ihb == H_ATOM )
        {
            Set_End_Index( workspace->hbond_index[i], ihb_top, hbonds );
        }
    }

    Init_Charge_Matrix_Remaining_Entries( system, control, far_nbrs,
            H, H_sp, &Htop, &H_sp_top );

    H->start[system->N_cm] = Htop;
    H_sp->start[system->N_cm] = H_sp_top;

    /* validate lists - decide if reallocation is required! */
    Validate_Lists( workspace, lists,
            data->step, system->N, H->m, Htop, num_bonds, num_hbonds );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "step%d: Htop = %d, num_bonds = %d, num_hbonds = %d\n",
             data->step, Htop, num_bonds, num_hbonds );
    //Print_Bonds( system, bonds, "sbonds.out" );
    //Print_Bond_List2( system, bonds, "sbonds.out" );
    //Print_Sparse_Matrix2( H, "H.out", NULL );
#endif
}


void Estimate_Storage_Sizes( reax_system *system, control_params *control,
        reax_list **lists, int *Htop, int *hb_top, int *bond_top, int *num_3body )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    real r_ij;
    real C12, C34, C56;
    real BO, BO_s, BO_pi, BO_pi2;
    reax_list *far_nbrs;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    far_nbrs = lists[FAR_NBRS];

    for ( i = 0; i < far_nbrs->n; ++i )
    {
        atom_i = &system->atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );
        sbp_i = &system->reax_param.sbp[type_i];
        ihb = sbp_i->p_hbond;

        /* update i-j distance - check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &far_nbrs->far_nbr_list[pj];

            if ( nbr_pj->d <= control->nonb_cut )
            {
                j = nbr_pj->nbr;
                atom_j = &system->atoms[j];
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[type_i][type_j];
                ++(*Htop);

                /* hydrogen bond lists */
                if ( control->hbond_cut > 0.0
                        && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                        && nbr_pj->d <= control->hbond_cut )
                {
                    jhb = sbp_j->p_hbond;

                    if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                    {
                        ++hb_top[i];
                    }
                    else if ( ihb == H_BONDING_ATOM && jhb == H_ATOM )
                    {
                        ++hb_top[j];
                    }
                }

                /* uncorrected bond orders */
                if ( nbr_pj->d <= control->bond_cut )
                {
                    r_ij = nbr_pj->d;

                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0 )
                    {
                        C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                    }
                    else
                    {
                        C12 = 0.0;
                        BO_s = 0.0;
                    }

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0 )
                    {
                        C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = EXP( C34 );
                    }
                    else
                    {
                        C34 = 0.0;
                        BO_pi = 0.0;
                    }

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0 )
                    {
                        C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = EXP( C56 );
                    }
                    else
                    {
                        C56 = 0.0;
                        BO_pi2 = 0.0;
                    }

                    /* Initially BO values are the uncorrected ones, page 1 */
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        ++bond_top[i];
                        ++bond_top[j];
                    }
                }
            }
        }
    }

    *Htop += system->N;
    *Htop *= SAFE_ZONE;
    for ( i = 0; i < system->N; ++i )
    {
        hb_top[i] = MAX( hb_top[i] * SAFE_HBONDS, MIN_HBONDS );
        *num_3body += SQR( bond_top[i] );
        bond_top[i] = MAX( bond_top[i] * 2, MIN_BONDS );
    }
    *num_3body *= SAFE_ZONE;
}


void Compute_Forces( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list** lists, output_controls *out_control, int realloc )
{
    real t_start, t_elapsed;

    t_start = Get_Time( );
    if ( control->tabulate <= 0 )
    {
        Init_Forces( system, control, data, workspace, lists, out_control );
    }
    else
    {
        Init_Forces_Tab( system, control, data, workspace, lists, out_control );
    }
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.init_forces += t_elapsed;

    t_start = Get_Time( );
    Compute_Bonded_Forces( system, control, data, workspace, lists, out_control );
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.bonded += t_elapsed;

    t_start = Get_Time( );
    Compute_NonBonded_Forces( system, control, data, workspace,
            lists, out_control, realloc );
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.nonb += t_elapsed;

    Compute_Total_Force( system, control, data, workspace, lists );

#if defined(DEBUG_FOCUS)
    //Print_Total_Force( system, control, data, workspace, lists, out_control );
#endif

#if defined(TEST_FORCES)
    Print_Total_Force( system, control, data, workspace, lists, out_control );
    Compare_Total_Forces( system, control, data, workspace, lists, out_control );
#endif
}
