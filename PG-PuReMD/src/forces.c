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

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "../../common/include/config.h"
#endif

#if defined(PURE_REAX)
  #include "forces.h"

  #include "allocate.h"
  #include "bond_orders.h"
  #include "bonds.h"
  #include "basic_comm.h"
  #include "charges.h"
  #include "comm_tools.h"
  #include "hydrogen_bonds.h"
  #include "io_tools.h"
  #include "list.h"
  #include "lookup.h"
  #include "multi_body.h"
  #include "nonbonded.h"
  #include "tool_box.h"
  #include "torsion_angles.h"
  #include "valence_angles.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_forces.h"

  #include "reax_allocate.h"
  #include "reax_bond_orders.h"
  #include "reax_bonds.h"
  #include "reax_basic_comm.h"
  #include "reax_charges.h"
  #include "reax_comm_tools.h"
  #include "reax_hydrogen_bonds.h"
  #include "reax_io_tools.h"
  #include "reax_list.h"
  #include "reax_lookup.h"
  #include "reax_multi_body.h"
  #include "reax_nonbonded.h"
  #include "reax_tool_box.h"
  #include "reax_torsion_angles.h"
  #include "reax_valence_angles.h"
  #include "reax_vector.h"
#endif

#include "index_utils.h"


typedef enum
{
    DIAGONAL = 0,
    OFF_DIAGONAL = 1,
} MATRIX_ENTRY_POSITION;


/* placeholder for unused interactions in interaction list
 * Interaction_Functions, which is initialized in Init_Force_Functions */
void Dummy_Interaction( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
}


void Init_Force_Functions( control_params * const control )
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
        control->intr_funcs[5] = &Dummy_Interaction;
    }
    control->intr_funcs[6] = &Dummy_Interaction;
    control->intr_funcs[7] = &Dummy_Interaction;
    control->intr_funcs[8] = &Dummy_Interaction;
    control->intr_funcs[9] = &Dummy_Interaction;
}


static inline real Init_Charge_Matrix_Entry_Tab( const reax_system * const system,
        const control_params * const control, LR_lookup_table * const LR,
        int i, int j, real r_ij, MATRIX_ENTRY_POSITION pos )
{
    int tmin, tmax;
    real val, ret;
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
                tmin = MIN( system->my_atoms[i].type, system->my_atoms[j].type );
                tmax = MAX( system->my_atoms[i].type, system->my_atoms[j].type );
                t = &LR[ index_lr( tmin, tmax, system->reax_param.num_atom_types ) ];

                val = LR_Lookup_Entry( t, r_ij, LR_CM );

                ret = ((i == j) ? 0.5 : 1.0) * val;
            break;

            case DIAGONAL:
                ret = system->reax_param.sbp[system->my_atoms[i].type].eta;
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


static inline real Init_Charge_Matrix_Entry( const reax_system * const system,
        const control_params * const control, const storage * const workspace,
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
                        + POW( system->reax_param.tbp[
                                index_tbp( system->my_atoms[i].type,
                                    system->my_atoms[j].type,
                                    system->reax_param.num_atom_types )
                        ].gamma, -3.0 );
                dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

                /* i == j: periodic self-interaction term
                 * i != j: general interaction term */
                ret = ((i == j) ? 0.5 : 1.0) * Tap * EV_to_KCALpMOL / dr3gamij_3;
                break;

            case DIAGONAL:
                ret = system->reax_param.sbp[system->my_atoms[i].type].eta;
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
        sparse_matrix * H, //sparse_matrix * H_sp,
        int * Htop )//, int * H_sp_top )
{
    int i, j, pj, target, val_flag;
    real d, xcut, bond_softness, * X_diag;

    switch ( control->charge_method )
    {
        case QEQ_CM:
            break;

        case EE_CM:
            H->start[system->n_cm - 1] = *Htop;
//            H_sp->start[system->n_cm - 1] = *H_sp_top;

            for ( i = 0; i < system->n_cm - 1; ++i )
            {
                H->j[*Htop] = i;
                H->val[*Htop] = 1.0;
                *Htop = *Htop + 1;

//                H_sp->j[*H_sp_top] = i;
//                H_sp->val[*H_sp_top] = 1.0;
//                *H_sp_top = *H_sp_top + 1;
            }

            H->j[*Htop] = system->n_cm - 1;
            H->val[*Htop] = 0.0;
            *Htop = *Htop + 1;

//            H_sp->j[*H_sp_top] = system->n_cm - 1;
//            H_sp->val[*H_sp_top] = 0.0;
//            *H_sp_top = *H_sp_top + 1;
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
//                H_sp->start[system->N + i] = *H_sp_top;

                /* constraint on ref. value for kinetic energy potential */
                H->j[*Htop] = i;
                H->val[*Htop] = 1.0;
                *Htop = *Htop + 1;

//                H_sp->j[*H_sp_top] = i;
//                H_sp->val[*H_sp_top] = 1.0;
//                *H_sp_top = *H_sp_top + 1;

                /* kinetic energy terms */
                for ( pj = Start_Index(i, far_nbr_list); pj < End_Index(i, far_nbr_list); ++pj )
                {
                    /* exclude self-periodic images of atoms for
                     * kinetic energy term because the effective
                     * potential is the same on an atom and its periodic image */
                    if ( far_nbr_list->far_nbr_list.d[pj] <= control->nonb_cut )
                    {
                        j = far_nbr_list->far_nbr_list.nbr[pj];

                        xcut = 0.5 * ( system->reax_param.sbp[ system->my_atoms[i].type ].b_s_acks2
                                + system->reax_param.sbp[ system->my_atoms[j].type ].b_s_acks2 );

                        if ( far_nbr_list->far_nbr_list.d[pj] < xcut )
                        {
                            d = far_nbr_list->far_nbr_list.d[pj] / xcut;
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

//                                val_flag = FALSE;

//                                for ( target = H_sp->start[system->N + i]; target < *H_sp_top; ++target )
//                                {
//                                    if ( H_sp->j[target] == system->N + j )
//                                    {
//                                        H_sp->val[target] += bond_softness;
//                                        val_flag = TRUE;
//                                        break;
//                                    }
//                                }

//                                if ( val_flag == FALSE )
//                                {
//                                    H_sp->j[*H_sp_top] = system->N + j;
//                                    H_sp->val[*H_sp_top] = bond_softness;
//                                    ++(*H_sp_top);
//                                }

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

//                H_sp->j[*H_sp_top] = system->N + i;
//                H_sp->val[*H_sp_top] = 0.0;
//                *H_sp_top = *H_sp_top + 1;
            }

            /* second to last row */
            H->start[system->n_cm - 2] = *Htop;
//            H_sp->start[system->n_cm - 2] = *H_sp_top;

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

//                for ( pj = H_sp->start[i]; pj < H_sp->start[i + 1]; ++pj )
//                {
//                    if ( H_sp->j[pj] == i )
//                    {
//                        H_sp->val[pj] = X_diag[i - system->N];
//                        break;
//                    }
//                }
            }

            /* coupling with the kinetic energy potential */
            for ( i = 0; i < system->N; ++i )
            {
                H->j[*Htop] = system->N + i;
                H->val[*Htop] = 1.0;
                *Htop = *Htop + 1;

//                H_sp->j[*H_sp_top] = system->N + i;
//                H_sp->val[*H_sp_top] = 1.0;
//                *H_sp_top = *H_sp_top + 1;
            }

            /* explicitly store zero on diagonal */
            H->j[*Htop] = system->n_cm - 2;
            H->val[*Htop] = 0.0;
            *Htop = *Htop + 1;

//            H_sp->j[*H_sp_top] = system->n_cm - 2;
//            H_sp->val[*H_sp_top] = 0.0;
//            *H_sp_top = *H_sp_top + 1;

            /* last row */
            H->start[system->n_cm - 1] = *Htop;
//            H_sp->start[system->n_cm - 1] = *H_sp_top;

            for ( i = 0; i < system->N; ++i )
            {
                H->j[*Htop] = i;
                H->val[*Htop] = 1.0;
                *Htop = *Htop + 1;

//                H_sp->j[*H_sp_top] = i;
//                H_sp->val[*H_sp_top] = 1.0;
//                *H_sp_top = *H_sp_top + 1;
            }

            /* explicitly store zero on diagonal */
            H->j[*Htop] = system->n_cm - 1;
            H->val[*Htop] = 0.0;
            *Htop = *Htop + 1;

//            H_sp->j[*H_sp_top] = system->n_cm - 1;
//            H_sp->val[*H_sp_top] = 0.0;
//            *H_sp_top = *H_sp_top + 1;

            sfree( X_diag, "Init_Charge_Matrix_Remaining_Entries::X_diag" );
            break;

        default:
            break;
    }
}


/* Compute the distances and displacement vectors for entries
 * in the far neighbors list if it's a NOT re-neighboring step */
static void Init_Distance( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    reax_list *far_nbr_list;
    reax_atom *atom_i, *atom_j;

    far_nbr_list = lists[FAR_NBRS];

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );

        /* update distance and displacement vector between atoms i and j (i-j) */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];
            
            far_nbr_list->far_nbr_list.dvec[pj][0] = atom_j->x[0] - atom_i->x[0];
            far_nbr_list->far_nbr_list.dvec[pj][1] = atom_j->x[1] - atom_i->x[1];
            far_nbr_list->far_nbr_list.dvec[pj][2] = atom_j->x[2] - atom_i->x[2];
            far_nbr_list->far_nbr_list.d[pj] = rvec_Norm( far_nbr_list->far_nbr_list.dvec[pj] );
        }
    }
}


#if defined(NEUTRAL_TERRITORY)
/* Compute the charge matrix entries and store the matrix in half format
 * using the far neighbors list (stored in full format) and according to
 * the neutral territory communication method */
static void Init_CM_Half_NT( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int cm_top;
    int local, renbr;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbr_list;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int mark[6];
    int total_cnt[6];
    int bin[6];
    int total_sum[6];
    int nt_flag;

    far_nbr_list = lists[FAR_NBRS];
    H = workspace->H;
    H->n = system->n;
    cm_top = 0;
    renbr = is_refactoring_step( control, data );
    nt_flag = TRUE;

    if ( renbr == TRUE )
    {
        for ( i = 0; i < 6; ++i )
        {
            total_cnt[i] = 0;
            bin[i] = 0;
            total_sum[i] = 0;
        }

        for ( i = system->n; i < system->N; ++i )
        {
            atom_i = &system->my_atoms[i];

            if ( atom_i->nt_dir != -1 )
            {
                total_cnt[ atom_i->nt_dir ]++;
            }
        }

        total_sum[0] = system->n;
        for ( i = 1; i < 6; ++i )
        {
            total_sum[i] = total_sum[i - 1] + total_cnt[i - 1];
        }

        for ( i = system->n; i < system->N; ++i )
        {
            atom_i = &system->my_atoms[i];

            if ( atom_i->nt_dir != -1 )
            {
                atom_i->pos = total_sum[ atom_i->nt_dir ] + bin[ atom_i->nt_dir ];
                bin[ atom_i->nt_dir ]++;
            }
        }
        H->NT = total_sum[5] + total_cnt[5];
    }

    mark[0] = 1;
    mark[1] = 1;
    mark[2] = 2;
    mark[3] = 2;
    mark[4] = 2;
    mark[5] = 2;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );

        if ( i < system->n )
        {
            local = 1;
        }
        else if ( atom_i->nt_dir != -1 )
        {
            local = 2;
            nt_flag = FALSE;
        }
        else
        {
            continue;
        }

        if ( local == 1 )
        {
            H->start[i] = cm_top;
            H->j[cm_top] = i;
            H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                    workspace, i, i, 0.0, DIAGONAL );
            ++cm_top;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];

            if ( far_nbr_list->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                type_j = atom_j->type;
                r_ij = far_nbr_list->far_nbr_list.d[pj];
                twbp = &system->reax_param.tbp[
                    index_tbp(type_i, type_j, system->reax_param.num_atom_types) ];

                if ( local == 1 )
                {
                    /* H matrix entry */
                    if ( atom_j->nt_dir > 0 || (j < system->n && i < j) )
                    {
                        if ( j < system->n )
                        {
                            H->j[cm_top] = j;
                        }
                        else
                        {
                            H->j[cm_top] = atom_j->pos;
                        }

                        if ( control->tabulate == 0 )
                        {
                            H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else 
                        {
                            H->val[cm_top] = Init_Charge_Matrix_Entry_Tab( system, control,
                                    workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                        }

                        ++cm_top;
                    }

                }
                else if ( local == 2 )
                {
                    /* H matrix entry */
                    if ( atom_j->nt_dir != -1
                            && mark[atom_i->nt_dir] != mark[atom_j->nt_dir]
                            && atom_i->pos < atom_j->pos )
                    {
                        if ( nt_flag == FALSE )
                        {
                            nt_flag = TRUE;
                            H->start[atom_i->pos] = cm_top;
                        }

                        //TODO: necessary?
                        if ( j < system->n )
                        {
                            H->j[cm_top] = j;
                        }
                        else
                        {
                            H->j[cm_top] = atom_j->pos;
                        }

                        if ( control->tabulate == 0 )
                        {
                            H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else 
                        {
                            H->val[cm_top] = Init_Charge_Matrix_Entry_Tab( system, control,
                                    workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                        }

                        ++cm_top;
                    }
                }

            }
        }

        if ( local == 1 )
        {
            H->end[i] = cm_top;
        }
        else if ( local == 2 )
        {
            if ( nt_flag == TRUE )
            {
                H->end[atom_i->pos] = cm_top;
            }
            else
            {
                 H->start[atom_i->pos] = 0;
                 H->end[atom_i->pos] = 0;
            }
        }
    }

    /* reallocation check */
    for ( i = 0; i < system->N; ++i )
    {
        if ( i < system->n && H->end[i] - H->start[i] > system->max_cm_entries[i] )
        {
            workspace->realloc.cm = TRUE;
            break;
        }
    }
}


/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the neutral territory communication method */
static void Init_CM_Full_NT( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int cm_top;
    int local, renbr;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbr_list;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int mark[6];
    int total_cnt[6];
    int bin[6];
    int total_sum[6];
    int nt_flag;

    far_nbr_list = lists[FAR_NBRS];
    H = workspace->H;
    H->n = system->n;
    cm_top = 0;
    renbr = is_refactoring_step( control, data );
    nt_flag = TRUE;

    if ( renbr == TRUE )
    {
        for ( i = 0; i < 6; ++i )
        {
            total_cnt[i] = 0;
            bin[i] = 0;
            total_sum[i] = 0;
        }

        for ( i = system->n; i < system->N; ++i )
        {
            atom_i = &system->my_atoms[i];

            if ( atom_i->nt_dir != -1 )
            {
                total_cnt[ atom_i->nt_dir ]++;
            }
        }

        total_sum[0] = system->n;
        for ( i = 1; i < 6; ++i )
        {
            total_sum[i] = total_sum[i - 1] + total_cnt[i - 1];
        }

        for ( i = system->n; i < system->N; ++i )
        {
            atom_i = &system->my_atoms[i];

            if ( atom_i->nt_dir != -1 )
            {
                atom_i->pos = total_sum[ atom_i->nt_dir ] + bin[ atom_i->nt_dir ];
                bin[ atom_i->nt_dir ]++;
            }
        }
        H->NT = total_sum[5] + total_cnt[5];
    }

    mark[0] = 1;
    mark[1] = 1;
    mark[2] = 2;
    mark[3] = 2;
    mark[4] = 2;
    mark[5] = 2;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );

        if ( i < system->n )
        {
            local = 1;
        }
        else if ( atom_i->nt_dir != -1 )
        {
            local = 2;
            nt_flag = FALSE;
        }
        else
        {
            continue;
        }

        if ( local == 1 )
        {
            H->start[i] = cm_top;
            H->j[cm_top] = i;
            H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                    workspace, i, i, 0.0, DIAGONAL );
            ++cm_top;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbr_list->far_nbr_list.nbr[pj];
                atom_j = &system->my_atoms[j];

                type_j = atom_j->type;
                r_ij = far_nbr_list->far_nbr_list.d[pj];
                twbp = &system->reax_param.tbp[
                    index_tbp(type_i, type_j, system->reax_param.num_atom_types) ];

                if ( local == 1 )
                {
                    /* H matrix entry */
                    if ( atom_j->nt_dir > 0 || (j < system->n) )
                    {
                        if ( j < system->n )
                        {
                            H->j[cm_top] = j;
                        }
                        else
                        {
                            H->j[cm_top] = atom_j->pos;
                        }

                        if ( control->tabulate == 0 )
                        {
                            H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else 
                        {
                            H->val[cm_top] = Init_Charge_Matrix_Entry_Tab( system, control,
                                    workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                        }

                        ++cm_top;
                    }

                }
                else if ( local == 2 )
                {
                    /* H matrix entry */
                    if ( ( atom_j->nt_dir != -1
                                && mark[atom_i->nt_dir] != mark[atom_j->nt_dir] )
                            || ( j < system->n && atom_i->nt_dir != 0 ) )
                    {
                        if ( nt_flag == FALSE )
                        {
                            nt_flag = TRUE;
                            H->start[atom_i->pos] = cm_top;
                        }

                        if ( j < system->n )
                        {
                            H->j[cm_top] = j;
                        }
                        else
                        {
                            H->j[cm_top] = atom_j->pos;
                        }

                        if ( control->tabulate == 0 )
                        {
                            H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else 
                        {
                            H->val[cm_top] = Init_Charge_Matrix_Entry_Tab( system, control,
                                    workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                        }

                        ++cm_top;
                    }
                }

            }
        }

        if ( local == 1 )
        {
            H->end[i] = cm_top;
        }
        else if ( local == 2 )
        {
            if ( nt_flag == TRUE )
            {
                H->end[atom_i->pos] = cm_top;
            }
            else
            {
                 H->start[atom_i->pos] = 0;
                 H->end[atom_i->pos] = 0;
            }
        }
    }

    /* reallocation check */
    for ( i = 0; i < system->N; ++i )
    {
        if ( i < system->n && H->end[i] - H->start[i] > system->max_cm_entries[i] )
        {
            workspace->realloc.cm = TRUE;
            break;
        }
    }
}


#else
/* Compute the charge matrix entries and store the matrix in half format
 * using the far neighbors list (stored in half format) and according to
 * the full shell communication method */
static void Init_CM_Half_FS( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int cm_top;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbr_list;
    reax_atom *atom_i, *atom_j;

    far_nbr_list = lists[FAR_NBRS];

    H = &workspace->H;
    H->n = system->n;
    cm_top = 0;

    for ( i = 0; i < system->n; ++i )
    {
        atom_i = &system->my_atoms[i];
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );

        /* diagonal entry in the matrix */
        H->start[i] = cm_top;
        H->j[cm_top] = i;
        H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                workspace, i, i, r_ij, DIAGONAL );
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbr_list->far_nbr_list.nbr[pj];
                atom_j = &system->my_atoms[j];
            
                /* if j is a local atom OR
                 * if j is a ghost atom in the upper triangular region of the matrix */
                if ( j < system->n || atom_i->orig_id < atom_j->orig_id )
                {
                    r_ij = far_nbr_list->far_nbr_list.d[pj];

                    H->j[cm_top] = j;

                    if ( control->tabulate == 0 )
                    {
                        H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                                workspace, i, j, r_ij, OFF_DIAGONAL );
                    }
                    else
                    {
                        H->val[cm_top] = Init_Charge_Matrix_Entry_Tab( system, control,
                                workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                    }

                    ++cm_top;
                }
            }
        }

        H->end[i] = cm_top;
    }

    /* reallocation check */
    for ( i = 0; i < system->n; ++i )
    {
        if ( H->end[i] - H->start[i] > system->max_cm_entries[i] )
        {
            workspace->realloc.cm = TRUE;
            break;
        }
    }
}


/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
static void Init_CM_Full_FS( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int cm_top;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbr_list;

    far_nbr_list = lists[FAR_NBRS];

    H = &workspace->H;
    H->n = system->n;
    cm_top = 0;

    for ( i = 0; i < system->n; ++i )
    {
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );

        /* diagonal entry in the matrix */
        H->start[i] = cm_top;
        H->j[cm_top] = i;
        H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                workspace, i, i, r_ij, DIAGONAL );
        ++cm_top;

        /* off-diagonal entries in the matrix */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbr_list->far_nbr_list.nbr[pj];
                r_ij = far_nbr_list->far_nbr_list.d[pj];

                H->j[cm_top] = j;

                if ( control->tabulate == 0 )
                {
                    H->val[cm_top] = Init_Charge_Matrix_Entry( system, control,
                            workspace, i, j, r_ij, OFF_DIAGONAL );
                }
                else
                {
                    H->val[cm_top] = Init_Charge_Matrix_Entry_Tab( system, control,
                            workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                }

                ++cm_top;
            }
        }

        H->end[i] = cm_top;
    }

    /* reallocation check */
    for ( i = 0; i < system->n; ++i )
    {
        if ( H->end[i] - H->start[i] > system->max_cm_entries[i] )
        {
            workspace->realloc.cm = TRUE;
            break;
        }
    }
}
#endif


/* Compute entries of the bonds/hbonds lists and store the lists in full format
 * using the far neighbors list (stored in half format)
 * 
 * Note: this version does NOT contain an optimization to restrict the bond_mark
 *  array to at most the 3-hop neighborhood */
static void Init_Bond_Half( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int btop_i;
    int ihb, jhb, ihb_top;
    int local;
    real cutoff;
    reax_list *far_nbr_list, *bond_list, *hbond_list;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int jhb_top;
    
    far_nbr_list = lists[FAR_NBRS];
    bond_list = lists[BONDS];
    hbond_list = lists[HBONDS];

    for ( i = 0; i < system->n; ++i )
    {
        workspace->bond_mark[i] = 0;
    }
    for ( i = system->n; i < system->N; ++i )
    {
        /* put ghost atoms to an infinite distance (i.e., 1000) */
        workspace->bond_mark[i] = 1000;
    }

    btop_i = 0;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );

        /* start at end because other atoms
         * can add to this atom's list (half-list) */
        btop_i = End_Index( i, bond_list );
        sbp_i = &system->reax_param.sbp[type_i];
        ihb = NON_H_BONDING_ATOM;
        ihb_top = -1;

        if ( i < system->n )
        {
            local = TRUE;
            cutoff = control->nonb_cut;
        }
        else
        {
            local = FALSE;
            cutoff = control->bond_cut;
        }

        if ( local == TRUE )
        {
            if ( control->hbond_cut > 0.0 )
            {
                ihb = sbp_i->p_hbond;

                if ( ihb == H_ATOM )
                {
                    /* start at end because other atoms
                     * can add to this atom's list (half-list) */ 
                    ihb_top = End_Index( atom_i->Hindex, hbond_list );
                }
                else
                {
                    ihb_top = -1;
                }
            }
        }

        /* update i-j distance - check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];
            
            if ( far_nbr_list->far_nbr_list.d[pj] <= cutoff )
            {
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[
                    index_tbp(type_i, type_j, system->reax_param.num_atom_types) ];

                if ( local == TRUE )
                {
                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.0
                            && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                            && far_nbr_list->far_nbr_list.d[pj] <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;

                        if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                        {
                            hbond_list->hbond_list[ihb_top].nbr = j;
                            hbond_list->hbond_list[ihb_top].scl = 1;
                            hbond_list->hbond_list[ihb_top].ptr = pj;
                            ++ihb_top;
                        }
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( j < system->n && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                        {
                            jhb_top = End_Index( atom_j->Hindex, hbond_list );
                            hbond_list->hbond_list[jhb_top].nbr = i;
                            hbond_list->hbond_list[jhb_top].scl = -1;
                            hbond_list->hbond_list[jhb_top].ptr = pj;
                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbond_list );
                        }
                    }
                }

                /* uncorrected bond orders */
                if ( far_nbr_list->far_nbr_list.d[pj] <= control->bond_cut
                        && BOp( workspace, bond_list, control->bo_cut,
                            i, btop_i, far_nbr_list->far_nbr_list.nbr[pj],
                            &far_nbr_list->far_nbr_list.rel_box[pj], far_nbr_list->far_nbr_list.d[pj],
                            &far_nbr_list->far_nbr_list.dvec[pj], far_nbr_list->format,
                            sbp_i, sbp_j, twbp ) == TRUE )
                {
                    ++btop_i;

                    if ( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
                    {
                        workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
                    }
                    else if ( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
                    {
                        workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
                    }
                }

            }
        }

        Set_End_Index( i, btop_i, bond_list );

        if ( local == TRUE && ihb == H_ATOM )
        {
            Set_End_Index( atom_i->Hindex, ihb_top, hbond_list );
        }
    }

    /* reallocation checks */
    for ( i = 0; i < system->N; ++i )
    {
        if ( Num_Entries( i, bond_list ) > system->max_bonds[i] )
        {
            workspace->realloc.bonds = TRUE;
            break;
        }
    }

    for ( i = 0; i < system->n; ++i )
    {
        if ( system->reax_param.sbp[ system->my_atoms[i].type ].p_hbond == H_ATOM
                && Num_Entries( system->my_atoms[i].Hindex, hbond_list )
                > system->max_hbonds[system->my_atoms[i].Hindex] )
        {
            workspace->realloc.hbonds = TRUE;
            break;
        }
    }
}


/* Compute entries of the bonds/hbonds lists and store the lists in full format
 * using the far neighbors list (stored in full format) */
static void Init_Bond_Full( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, ihb_top;
    reax_list *far_nbr_list, *bond_list, *hbond_list;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int btop_i;
    int k, push;
    int *q;

    far_nbr_list = lists[FAR_NBRS];
    bond_list = lists[BONDS];
    hbond_list = lists[HBONDS];
    push = 0;
    btop_i = 0;
    bond_list = lists[BONDS];

    q = smalloc( sizeof(int) * (system->N - system->n),
            "Init_Bond_Full::q" );

    for ( i = 0; i < system->n; ++i )
    {
        workspace->bond_mark[i] = 0;
    }
    for ( i = system->n; i < system->N; ++i )
    {
        /* put ghost atoms to an infinite distance (i.e., 1000) */
        workspace->bond_mark[i] = 1000;
    }

    /* bonds that are directly connected to local atoms */
    for ( i = 0; i < system->n; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        btop_i = End_Index( i, bond_list );
        sbp_i = &system->reax_param.sbp[type_i];
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );
        ihb = sbp_i->p_hbond;
        ihb_top = Start_Index( atom_i->Hindex, hbond_list );

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];

            if ( control->hbond_cut > 0.0 && ihb == H_ATOM )
            {
                /* check if j is within cutoff */
                if ( far_nbr_list->far_nbr_list.d[pj] <= control->hbond_cut
                        && system->reax_param.sbp[atom_j->type].p_hbond == H_BONDING_ATOM )
                {
                    hbond_list->hbond_list[ihb_top].nbr = j;
                    hbond_list->hbond_list[ihb_top].scl = 1;
                    hbond_list->hbond_list[ihb_top].ptr = pj;
                    ++ihb_top;
                }
            }

            if ( i <= j && far_nbr_list->far_nbr_list.d[pj] <= control->bond_cut )
            {
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[
                    index_tbp(type_i, type_j, system->reax_param.num_atom_types) ];

                if ( BOp( workspace, bond_list, control->bo_cut,
                            i, btop_i, far_nbr_list->far_nbr_list.nbr[pj],
                            &far_nbr_list->far_nbr_list.rel_box[pj], far_nbr_list->far_nbr_list.d[pj],
                            &far_nbr_list->far_nbr_list.dvec[pj], far_nbr_list->format,
                            sbp_i, sbp_j, twbp ) == TRUE )
                {
                    ++btop_i;

                    /* if j is a non-local atom, push it on the queue
                     * to search for it's bonded neighbors later */
                    if ( workspace->bond_mark[j] == 1000 )
                    {
                        workspace->bond_mark[j] = 101;
                        q[ push++ ] = j;
                    }
                }
            }
        }

        if ( control->hbond_cut > 0.0 && ihb == H_ATOM )
        {
            Set_End_Index( atom_i->Hindex, ihb_top, hbond_list );
        }

        Set_End_Index( i, btop_i, bond_list );
    }

    /* bonds that are indirectly connected to local atoms */
    for ( k = 0; k < push; ++k )
    {
        i = q[k];
        workspace->bond_mark[i] -= 100;
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        btop_i = End_Index( i, bond_list );
        sbp_i = &system->reax_param.sbp[type_i];
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list->far_nbr_list.nbr[pj];

            if ( workspace->bond_mark[i] == 3
                    && workspace->bond_mark[j] == 1000 )
            {
                continue;
            }

            atom_j = &system->my_atoms[j];

            if (  workspace->bond_mark[j] > 100
                    && far_nbr_list->far_nbr_list.d[pj] <= control->bond_cut )
            {
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[
                    index_tbp(type_i, type_j, system->reax_param.num_atom_types) ];

                if ( BOp( workspace, bond_list, control->bo_cut,
                            i, btop_i, far_nbr_list->far_nbr_list.nbr[pj],
                            &far_nbr_list->far_nbr_list.rel_box[pj], far_nbr_list->far_nbr_list.d[pj],
                            &far_nbr_list->far_nbr_list.dvec[pj], far_nbr_list->format,
                            sbp_i, sbp_j, twbp ) == TRUE )
                {
                    ++btop_i;

                    if ( workspace->bond_mark[j] == 1000 )
                    {
                        workspace->bond_mark[j] = workspace->bond_mark[i] + 100;

                        if ( workspace->bond_mark[i] < 3 )
                        {
                            q[ push++ ] = j;
                        }
                    }
                }
            }
        }

        Set_End_Index( i, btop_i, bond_list );
    }

    /* reallocation checks */
    for ( i = 0; i < system->N; ++i )
    {
        if ( Num_Entries( i, bond_list ) > system->max_bonds[i] )
        {
            workspace->realloc.bonds = TRUE;
            break;
        }
    }

    for ( i = 0; i < system->n; ++i )
    {
        if ( system->reax_param.sbp[ system->my_atoms[i].type ].p_hbond == H_ATOM
                && Num_Entries( system->my_atoms[i].Hindex, hbond_list )
                > system->max_hbonds[system->my_atoms[i].Hindex] )
        {
            workspace->realloc.hbonds = TRUE;
            break;
        }
    }

    sfree( q, "Init_Bond_Full::q" );
}


static void Compute_Bonded_Forces( reax_system * const system,
        control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control )
{
    int i;

#if defined(TEST_ENERGY)
    /* Mark beginning of a new timestep in bonded energy files */
    Debug_Marker_Bonded( out_control, data->step );
#endif

    /* Implement all force calls as function pointers */
    for ( i = 0; i < NUM_INTRS; i++ )
    {
        if ( control->intr_funcs[i] != NULL )
        {
            (control->intr_funcs[i])( system, control, data, workspace,
                    lists, out_control );
        }
    }
}


static void Compute_NonBonded_Forces( reax_system * const system,
        control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
#if defined(TEST_ENERGY)
    /* Mark beginning of a new timestep in nonbonded energy files */
    Debug_Marker_Nonbonded( out_control, data->step );
#endif

    /* van der Waals and Coulomb interactions */
    if ( control->tabulate == 0 )
    {
        vdW_Coulomb_Energy( system, control, data, workspace, lists, out_control );
    }
    else
    {
        Tabulated_vdW_Coulomb_Energy( system, control, data, workspace, lists, out_control );
    }
}



static void Estimate_Storages_CM( reax_system * const system, control_params * const control,
        reax_list ** const lists, int * const matrix_dim, int cm_format )
{
    int i, j, pj;
    int start_i, end_i;
    int local;
    real cutoff;
    reax_list *far_nbr_list;
    reax_atom *atom_i, *atom_j;
#if defined(NEUTRAL_TERRITORY)
    int mark[6] = {1, 1, 2, 2, 2, 2};
#endif

    far_nbr_list = lists[FAR_NBRS];
    *matrix_dim = 0;

    for ( i = 0; i < system->total_cap; ++i )
    {
        if ( i < system->local_cap )
        {
            system->cm_entries[i] = 0;
        }
    }

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );

        if ( i < system->n )
        {
            local = 1;
            cutoff = control->nonb_cut;
            ++system->cm_entries[i];
            ++(*matrix_dim);
        }
#if defined(NEUTRAL_TERRITORY)
        else if ( atom_i->nt_dir != -1 )
        {
            local = 2;
            cutoff = control->nonb_cut;
            ++(*matrix_dim);
        }
#endif
        else
        {
            local = 0;
            cutoff = control->bond_cut;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list->far_nbr_list.nbr[pj];

#if !defined(NEUTRAL_TERRITORY)
            if ( far_nbr_list->format == HALF_LIST )
#endif
            {
                atom_j = &system->my_atoms[j];
            }

            if ( far_nbr_list->far_nbr_list.d[pj] <= cutoff )
            {
                if ( local == 1 )
                {
#if defined(NEUTRAL_TERRITORY)
                    if( atom_j->nt_dir > 0 || j < system->n )
                    {
                        ++system->cm_entries[i];
                    }
#else
                    if ( (far_nbr_list->format == HALF_LIST
                                && (j < system->n || atom_i->orig_id < atom_j->orig_id))
                            || far_nbr_list->format == FULL_LIST )
                    {
                        ++system->cm_entries[i];
                    }
#endif
                }

#if defined(NEUTRAL_TERRITORY)
                else if ( local == 2 )
                {
                    if( ( atom_j->nt_dir != -1 && mark[atom_i->nt_dir] != mark[atom_j->nt_dir] ) 
                            || ( j < system->n && atom_i->nt_dir != 0 ))
                    {
                        ++system->cm_entries[i];
                    }
                }
#endif
            }
        }
    }

#if defined(NEUTRAL_TERRITORY)
    /* Since we don't know the NT atoms' position yet, Htop cannot be calculated accurately.
     * Therefore, we assume it is full and divide 2 if necessary. */
    if ( cm_format == SYM_HALF_MATRIX )
    {
        for ( i = 0; i < system->local_cap; ++i )
        {
            system->cm_entries[i] = (system->cm_entries[i] + system->n + 1) / 2;
        }
    }
#endif

#if defined(NEUTRAL_TERRITORY)
    *matrix_dim = (int) MAX( *matrix_dim * SAFE_ZONE_NT, MIN_CAP );
#else
    *matrix_dim = (int) MAX( *matrix_dim * SAFE_ZONE, MIN_CAP );
#endif

    for ( i = 0; i < system->N; ++i )
    {
        if ( i < system->local_cap )
        {
#if defined(NEUTRAL_TERRITORY)
            system->max_cm_entries[i] = MAX( (int)(system->cm_entries[i] * SAFE_ZONE_NT), MIN_CM_ENTRIES );
#else
            system->max_cm_entries[i] = MAX( (int)(system->cm_entries[i] * SAFE_ZONE), MIN_CM_ENTRIES );
#endif
        }
    }

    /* set currently unused space to min. capacity */
    for ( i = system->N; i < system->local_cap; ++i )
    {
        system->max_cm_entries[i] = MIN_CM_ENTRIES;
    }

    /* reductions to get totals */
    system->total_cm_entries = 0;

    for ( i = 0; i < system->local_cap; ++i )
    {
        system->total_cm_entries += system->max_cm_entries[i];
    }

    system->total_cm_entries = MAX( system->total_cm_entries, MIN_CAP * MIN_CM_ENTRIES );
}


static void Estimate_Storages_Bonds( reax_system * const system,
        control_params * const control, reax_list ** const lists )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    int local;
    real cutoff;
    real r_ij;
    real C12, C34, C56;
    real BO, BO_s, BO_pi, BO_pi2;
    reax_list *far_nbr_list;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
#if defined(NEUTRAL_TERRITORY)
    int mark[6] = {1, 1, 2, 2, 2, 2};
#endif

    far_nbr_list = lists[FAR_NBRS];

    for ( i = 0; i < system->total_cap; ++i )
    {
        system->bonds[i] = 0;
    }

    for ( i = 0; i < system->total_cap; ++i )
    {
        system->hbonds[i] = 0;
    }

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );
        sbp_i = &system->reax_param.sbp[type_i];

        if ( i < system->n )
        {
            local = 1;
            cutoff = control->nonb_cut;
            ihb = sbp_i->p_hbond;
        }
#if defined(NEUTRAL_TERRITORY)
        else if ( atom_i->nt_dir != -1 )
        {
            local = 2;
            cutoff = control->nonb_cut;
            ihb = -1;
        }
#endif
        else
        {
            local = 0;
            cutoff = control->bond_cut;
            ihb = NON_H_BONDING_ATOM;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list->far_nbr_list.nbr[pj];

#if !defined(NEUTRAL_TERRITORY)
            if ( far_nbr_list->format == HALF_LIST )
#endif
            {
                atom_j = &system->my_atoms[j];
            }

            if ( far_nbr_list->far_nbr_list.d[pj] <= cutoff )
            {
                type_j = system->my_atoms[j].type;
                r_ij = far_nbr_list->far_nbr_list.d[pj];
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[
                    index_tbp(type_i, type_j, system->reax_param.num_atom_types) ];

                if ( local == 1 )
                {
                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.1
                            && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                            && far_nbr_list->far_nbr_list.d[pj] <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;

                        if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                        {
                            ++system->hbonds[atom_i->Hindex];
                        }
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( far_nbr_list->format == HALF_LIST
                                && (j < system->n && ihb == H_BONDING_ATOM && jhb == H_ATOM) )
                        {
                            ++system->hbonds[atom_j->Hindex];
                        }
                    }
                }

                /* uncorrected bond orders */
                if ( far_nbr_list->far_nbr_list.d[pj] <= control->bond_cut )
                {
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

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0)
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
                        ++system->bonds[i];
                        if ( far_nbr_list->format == HALF_LIST )
                        {
                            ++system->bonds[j];
                        }
                    }
                }
            }
        }
    }

    for ( i = 0; i < system->total_cap; ++i )
    {
        system->max_hbonds[i] = 0;
    }

    for ( i = 0; i < system->N; ++i )
    {
        if ( far_nbr_list->format == HALF_LIST )
        {
            system->max_bonds[i] = MAX( (int)(2.0 * system->bonds[i] * SAFE_ZONE), MIN_BONDS );
        }
        else
        {
            system->max_bonds[i] = MAX( (int)(system->bonds[i] * SAFE_ZONE), MIN_BONDS );
        }
        if ( system->reax_param.sbp[ system->my_atoms[i].type ].p_hbond == H_ATOM )
        {
            system->max_hbonds[ system->my_atoms[i].Hindex ] = MAX(
                    (int)(system->hbonds[ system->my_atoms[i].Hindex ] * SAFE_ZONE), MIN_HBONDS );
        }
    }

    /* set currently unused space to min. capacity */
    for ( i = system->N; i < system->total_cap; ++i )
    {
        system->max_bonds[i] = MIN_BONDS;
    }
    for ( i = system->N; i < system->total_cap; ++i )
    {
        system->max_hbonds[i] = MIN_HBONDS;
    }

    /* reductions to get totals */
    system->total_bonds = 0;
    system->total_hbonds = 0;
    system->total_thbodies = 0;

    for ( i = 0; i < system->total_cap; ++i )
    {
        system->total_bonds += system->max_bonds[i];
    }
    for ( i = 0; i < system->total_cap; ++i )
    {
        system->total_hbonds += system->max_hbonds[i];
    }
    if ( far_nbr_list->format == HALF_LIST )
    {
        for ( i = 0; i < system->total_cap; ++i )
        {
            system->total_thbodies += SQR( system->max_bonds[i] / 2.0 );
        }
    }
    else
    {
        for ( i = 0; i < system->total_cap; ++i )
        {
            system->total_thbodies += SQR( system->max_bonds[i] );
        }
    }

    system->total_bonds = MAX( system->total_bonds, MIN_CAP * MIN_BONDS );
    system->total_hbonds = MAX( system->total_hbonds, MIN_CAP * MIN_HBONDS );
    system->total_thbodies = MAX( system->total_thbodies * SAFE_ZONE, MIN_3BODIES );

    /* duplicate info in atom structs in case of
     * ownership transfer across processor boundaries */
    for ( i = 0; i < system->n; ++i )
    {
        system->my_atoms[i].num_bonds = system->bonds[i];
    }
    for ( i = 0; i < system->N; ++i )
    {
        system->my_atoms[i].num_hbonds = system->hbonds[i];
    }
}


/* this version of Compute_Total_Force computes forces from
 * coefficients accumulated by all interaction functions.
 * Saves enormous time & space! */
static void Compute_Total_Force( reax_system * const system,
        control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        mpi_datatypes * const mpi_data )
{
    int i, pj;
    reax_list * const bond_list = lists[BONDS];

    if ( control->virial == 0 )
    {
        for ( i = 0; i < system->N; ++i )
        {
            for ( pj = Start_Index(i, bond_list); pj < End_Index(i, bond_list); ++pj )
            {
                if ( i < bond_list->bond_list[pj].nbr )
                {
                    Add_dBond_to_Forces( i, pj, system, data, workspace, lists );
                }
            }
        }
    }
    else
    {
        for ( i = 0; i < system->N; ++i )
        {
            for ( pj = Start_Index(i, bond_list); pj < End_Index(i, bond_list); ++pj )
            {
                if ( i < bond_list->bond_list[pj].nbr )
                {
                    Add_dBond_to_Forces_NPT( i, pj, system, data, workspace, lists );
                }
            }
        }
    }

#if defined(PURE_REAX)
    /* now all forces are computed to their partially-final values
     * based on the neighbors information each processor has had.
     * final values of force on each atom needs to be computed by adding up
     * all partially-final pieces */
    Coll_FS( system, mpi_data, workspace->f, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    for ( i = 0; i < system->n; ++i )
    {
        rvec_Copy( system->my_atoms[i].f, workspace->f[i] );
    }

#if defined(TEST_FORCES)
    Coll_FS( system, mpi_data, workspace->f_ele, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_vdw, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_be, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_lp, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_ov, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_un, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_ang, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_coa, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_pen, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_hb, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_tor, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll_FS( system, mpi_data, workspace->f_con, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
#endif

#endif
}


static int Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, renbr, ret;
    static int dist_done = FALSE, cm_done = FALSE, bonds_done = FALSE;
#if defined(LOG_PERFORMANCE)
    double time;
    
    time = Get_Time( );
#endif

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    if ( renbr == FALSE && dist_done == FALSE )
    {
        Init_Distance( system, control, data, workspace, lists, out_control );

        dist_done = TRUE;
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.init_dist );
#endif

    if ( cm_done == FALSE )
    {
        Init_Matrix_Row_Indices( &workspace->H, system->max_cm_entries );

#if defined(NEUTRAL_TERRITORY)
        if ( workspace->H.format == SYM_HALF_MATRIX )
        {
            Init_CM_Half_NT( system, control, data, workspace, lists, out_control );
        }
        else
        {
            Init_CM_Full_NT( system, control, data, workspace, lists, out_control );
        }
#else
        if ( workspace->H.format == SYM_HALF_MATRIX )
        {
            Init_CM_Half_FS( system, control, data, workspace, lists, out_control );
        }
        else
        {
            Init_CM_Full_FS( system, control, data, workspace, lists, out_control );
        }
#endif
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.init_cm );
#endif

    if ( bonds_done == FALSE )
    {
        for ( i = 0; i < system->total_cap; ++i )
        {
            workspace->total_bond_order[i] = 0.0;
        }
        for ( i = 0; i < system->total_cap; ++i )
        {
            rvec_MakeZero( workspace->dDeltap_self[i] );
        }

        Init_List_Indices( lists[BONDS], system->max_bonds );
        Init_List_Indices( lists[HBONDS], system->max_hbonds );

        if ( lists[FAR_NBRS]->format == HALF_LIST )
        {
            Init_Bond_Half( system, control, data, workspace, lists, out_control );
        }
        else
        {
            Init_Bond_Full( system, control, data, workspace, lists, out_control );
        }
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.init_bond );
#endif

    ret = (workspace->realloc.cm == FALSE
            && workspace->realloc.bonds == FALSE
            && workspace->realloc.hbonds == FALSE
            ? SUCCESS : FAILURE);

    if ( workspace->realloc.cm == FALSE )
    {
        cm_done = TRUE;
    }
    if ( workspace->realloc.bonds == FALSE && workspace->realloc.hbonds == FALSE )
    {
        bonds_done = TRUE;
    }

    if ( ret == SUCCESS )
    {
        dist_done = FALSE;
        cm_done = FALSE;
        bonds_done = FALSE;
    }

    return ret;
}


static int Init_Forces_No_Charges( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int btop_i;
    int ihb, jhb, ihb_top;
    int local, flag, renbr;
    real cutoff;
    reax_list *far_nbr_list, *bond_list, *hbond_list;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int jhb_top;
    int start_j, end_j;
    int btop_j;

    far_nbr_list = lists[FAR_NBRS];
    bond_list = lists[BONDS];
    hbond_list = lists[HBONDS];

    for ( i = 0; i < system->n; ++i )
    {
        workspace->bond_mark[i] = 0;
    }
    for ( i = system->n; i < system->N; ++i )
    {
        /* put ghost atoms to an infinite distance */
        workspace->bond_mark[i] = 1000;
    }

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );
        if ( far_nbr_list->format == HALF_LIST )
        {
            /* start at end because other atoms
             * can add to this atom's list (half-list) */
            btop_i = End_Index( i, bond_list );
        }
        else if ( far_nbr_list->format == FULL_LIST )
        {
            btop_i = Start_Index( i, bond_list );
        }
        else
        {
            btop_i = 0;
        }
        sbp_i = &system->reax_param.sbp[type_i];
        ihb = NON_H_BONDING_ATOM;
        ihb_top = -1;

        if ( i < system->n )
        {
            local = TRUE;
            cutoff = MAX( control->hbond_cut, control->bond_cut );
        }
        else
        {
            local = FALSE;
            cutoff = control->bond_cut;
        }

        if ( local == TRUE && control->hbond_cut > 0.0 )
        {
            ihb = sbp_i->p_hbond;

            if ( ihb == H_ATOM )
            {
                if ( far_nbr_list->format == HALF_LIST )
                {
                    /* start at end because other atoms
                     * can add to this atom's list (half-list) */
                    ihb_top = End_Index( atom_i->Hindex, hbond_list );
                }
                else if ( far_nbr_list->format == FULL_LIST )
                {
                    ihb_top = Start_Index( atom_i->Hindex, hbond_list );
                }
            }
            else
            {
                ihb_top = -1;
            }
        }

        /* update i-j distance - check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];

            if ( renbr == TRUE )
            {
                if ( far_nbr_list->far_nbr_list.d[pj] <= cutoff )
                {
                    flag = TRUE;
                }
                else
                {
                    flag = FALSE;
                }
            }
            else
            {
                far_nbr_list->far_nbr_list.dvec[pj][0] = atom_j->x[0] - atom_i->x[0];
                far_nbr_list->far_nbr_list.dvec[pj][1] = atom_j->x[1] - atom_i->x[1];
                far_nbr_list->far_nbr_list.dvec[pj][2] = atom_j->x[2] - atom_i->x[2];
                far_nbr_list->far_nbr_list.d[pj] = rvec_Norm_Sqr( far_nbr_list->far_nbr_list.dvec[pj] );

                if ( far_nbr_list->far_nbr_list.d[pj] <= SQR(cutoff) )
                {
                    far_nbr_list->far_nbr_list.d[pj] = SQRT( far_nbr_list->far_nbr_list.d[pj] );
                    flag = TRUE;
                }
                else
                {
                    flag = FALSE;
                }
            }

            if ( flag == TRUE )
            {
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[
                    index_tbp(type_i, type_j, system->reax_param.num_atom_types) ];

                if ( local == TRUE )
                {
                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.0
                            && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                            && far_nbr_list->far_nbr_list.d[pj] <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;

                        if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                        {
                            hbond_list->hbond_list[ihb_top].nbr = j;
                            hbond_list->hbond_list[ihb_top].scl = 1;
                            hbond_list->hbond_list[ihb_top].ptr = pj;
                            ++ihb_top;
                        }
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( far_nbr_list->format == HALF_LIST
                                && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                        {
                            jhb_top = End_Index( atom_j->Hindex, hbond_list );
                            hbond_list->hbond_list[jhb_top].nbr = i;
                            hbond_list->hbond_list[jhb_top].scl = -1;
                            hbond_list->hbond_list[jhb_top].ptr = pj;
                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbond_list );
                        }
                    }
                }

                /* uncorrected bond orders */
                if ( //(workspace->bond_mark[i] < 3 || workspace->bond_mark[j] < 3) &&
                    far_nbr_list->far_nbr_list.d[pj] <= control->bond_cut
                    && BOp( workspace, bond_list, control->bo_cut,
                         i, btop_i, far_nbr_list->far_nbr_list.nbr[pj],
                         &far_nbr_list->far_nbr_list.rel_box[pj], far_nbr_list->far_nbr_list.d[pj],
                         &far_nbr_list->far_nbr_list.dvec[pj], far_nbr_list->format,
                         sbp_i, sbp_j, twbp ) == TRUE )
                {
                    ++btop_i;

                    if ( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
                    {
                        workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
                    }
                    else if ( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
                    {
                        workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
                    }
                }
            }
        }

        Set_End_Index( i, btop_i, bond_list );

        if ( local == TRUE && ihb == H_ATOM )
        {
            Set_End_Index( atom_i->Hindex, ihb_top, hbond_list );
        }
    }

    if ( far_nbr_list->format == FULL_LIST )
    {
        /* set sym_index for bonds list (far_nbrs full list) */
        for ( i = 0; i < system->N; ++i )
        {
            start_i = Start_Index( i, bond_list );
            end_i = End_Index( i, bond_list );

            for ( btop_i = start_i; btop_i < end_i; ++btop_i )
            {
                j = bond_list->bond_list[btop_i].nbr;
                start_j = Start_Index( j, bond_list );
                end_j = End_Index( j, bond_list );

                for ( btop_j = start_j; btop_j < end_j; ++btop_j )
                {
                    if ( bond_list->bond_list[btop_j].nbr == i )
                    {
                        bond_list->bond_list[btop_i].sym_index = btop_j;
                        break;
                    }
                }
            }
        }
    }

    /* reallocation checks */
    for ( i = 0; i < system->N; ++i )
    {
        if ( Num_Entries( i, bond_list ) > system->max_bonds[i] )
        {
            workspace->realloc.bonds = TRUE;
        }

        if ( i < system->n
                && system->reax_param.sbp[ system->my_atoms[i].type ].p_hbond == H_ATOM
                && Num_Entries( system->my_atoms[i].Hindex, hbond_list )
                > system->max_hbonds[system->my_atoms[i].Hindex] )
        {
            workspace->realloc.hbonds = TRUE;
        }
    }

    return (workspace->realloc.bonds == TRUE 
            || workspace->realloc.hbonds == TRUE) ? FAILURE : SUCCESS;
}


void Estimate_Storages( reax_system * const system, control_params * const control,
        reax_list ** const lists, storage *workspace, int realloc_cm,
        int realloc_bonds, int * const matrix_dim, int cm_format )
{
    if ( realloc_cm == TRUE )
    {
        Estimate_Storages_CM( system, control, lists, matrix_dim, cm_format );
    }

    if ( realloc_bonds == TRUE )
    {
        Estimate_Storages_Bonds( system, control, lists );
    }
}


int Compute_Forces( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
    int charge_flag, matrix_dim, ret, ret_mpi;
#if defined(LOG_PERFORMANCE)
    real time;

    time = Get_Time( );
#endif

    /********* init forces ************/
    if ( control->charge_freq
            && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        charge_flag = TRUE;
    }
    else
    {
        charge_flag = FALSE;
    }

    if ( charge_flag == TRUE )
    {
        ret = Init_Forces( system, control, data, workspace, lists, out_control, mpi_data );
    }
    else
    {
        ret = Init_Forces_No_Charges( system, control, data, workspace, lists, out_control );
    }

    if ( ret != SUCCESS )
    {
        Estimate_Storages( system, control, lists, workspace,
                workspace->realloc.cm,
                (workspace->realloc.bonds == TRUE
                 || workspace->realloc.hbonds == TRUE ? TRUE : FALSE),
                &matrix_dim, workspace->H.format );
    }

#if defined(LOG_PERFORMANCE)
    Update_Timing_Info( &time, &data->timing.init_forces );
#endif

    if ( ret == SUCCESS )
    {
        Compute_Bonded_Forces( system, control, data, workspace,
                lists, out_control );

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.bonded );
#endif

    /**************** charges ************************/
#if defined(PURE_REAX)
        if ( charge_flag == TRUE )
        {
            Compute_Charges( system, control, data, workspace, out_control, mpi_data );
        }

#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm );
#endif

        /* dynamically determine preconditioner recomputation rate */
        if ( control->cm_solver_pre_comp_refactor == -1 )
        {
            /* root MPI process determines recomputation
             * and broadcasts to other processes */
            if ( system->my_rank == MASTER_NODE )
            {
                /* preconditioner just recomputed, record timings */
                if ( data->refactor == TRUE )
                {
                    data->refactor = FALSE;
                    data->timing.cm_last_pre_comp = data->timing.cm_solver_pre_comp;
                    data->timing.last_nbrs = data->timing.nbrs;
                    data->last_pc_step = data->step;
                    data->timing.cm_optimum = data->timing.cm_solver_pre_app
                        + data->timing.cm_solver_spmv 
                        + data->timing.cm_solver_vector_ops
                        + data->timing.cm_solver_orthog 
                        + data->timing.cm_solver_tri_solve;
                    data->timing.cm_total_loss = ZERO;
                }
                /* not enough initial guesses for spline extrapolation of initial guesses,
                 *  so do not record metrics */
                else if ( data->step <= 4 )
                {
                    data->timing.cm_optimum = data->timing.cm_solver_pre_app
                        + data->timing.cm_solver_spmv
                        + data->timing.cm_solver_vector_ops
                        + data->timing.cm_solver_orthog
                        + data->timing.cm_solver_tri_solve;
                }
                else
                {
                    /* record time lost from preconditioner degradation */
                    data->timing.cm_total_loss += data->timing.cm_solver_pre_app
                        + data->timing.cm_solver_spmv 
                        + data->timing.cm_solver_vector_ops
                        + data->timing.cm_solver_orthog 
                        + data->timing.cm_solver_tri_solve
                        - data->timing.cm_optimum;

                    /* cases:
                     *  - check if cumulative loss exceeds recomputation time,
                     *    and if so, schedule recomputation
                     *  - since preconditioner recomputation is coupled with
                     *    neighbor list regeneration, schedule precomputation
                     *    recomputation if neighbor lists are recomputed */
                    if ( data->timing.cm_total_loss > data->timing.cm_last_pre_comp + data->timing.last_nbrs
                            || data->step - data->last_pc_step + 1 >= control->reneighbor )
                    {
                        data->refactor = TRUE;
                    }
                }
            }

            ret_mpi = MPI_Bcast( &data->refactor, 1, MPI_INT, MASTER_NODE, MPI_COMM_WORLD );
            Check_MPI_Error( ret_mpi, __FILE__, __LINE__ );
        }
#endif //PURE_REAX
    
        Compute_NonBonded_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    
#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.nonb );
#endif
    
        Compute_Total_Force( system, control, data, workspace, lists, mpi_data );
    
#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.bonded );
#endif

#if defined(TEST_FORCES)
        Print_Force_Files( system, control, data, workspace, lists, out_control, mpi_data );
#endif
    }

    return ret;
}
