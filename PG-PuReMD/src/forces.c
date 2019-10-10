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

#if defined(PURE_REAX)
  #include "forces.h"
  #include "bond_orders.h"
  #include "bonds.h"
  #include "basic_comm.h"
  #include "hydrogen_bonds.h"
  #include "io_tools.h"
  #include "list.h"
  #include "lookup.h"
  #include "multi_body.h"
  #include "nonbonded.h"
  #include "charges.h"
  #include "tool_box.h"
  #include "torsion_angles.h"
  #include "valence_angles.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_forces.h"
  #include "reax_bond_orders.h"
  #include "reax_bonds.h"
  #include "reax_basic_comm.h"
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


int compare_bonds( const void *p1, const void *p2 )
{
    return ((bond_data *) p1)->nbr - ((bond_data *) p2)->nbr;
}


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


void Compute_Bonded_Forces( reax_system * const system, control_params * const control,
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
        (control->intr_funcs[i])( system, control, data, workspace, lists, out_control );
    }
}


void Compute_NonBonded_Forces( reax_system * const system, control_params * const control,
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


/* this version of Compute_Total_Force computes forces from
 * coefficients accumulated by all interaction functions.
 * Saves enormous time & space! */
void Compute_Total_Force( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        mpi_datatypes * const mpi_data )
{
    int i, pj;
    reax_list * const bond_list = lists[BONDS];

    for ( i = 0; i < system->N; ++i )
    {
        for ( pj = Start_Index(i, bond_list); pj < End_Index(i, bond_list); ++pj )
        {
            if ( i < bond_list->bond_list[pj].nbr )
            {
                if ( control->virial == 0 )
                {
                    Add_dBond_to_Forces( i, pj, workspace, lists );
                }
                else
                {
                    Add_dBond_to_Forces_NPT( i, pj, data, workspace, lists );
                }
            }
        }
    }

#if defined(PURE_REAX)
    /* now all forces are computed to their partially-final values
     * based on the neighbors information each processor has had.
     * final values of force on each atom needs to be computed by adding up
     * all partially-final pieces */
    Coll( system, mpi_data, workspace->f, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    for ( i = 0; i < system->n; ++i )
    {
        rvec_Copy( system->my_atoms[i].f, workspace->f[i] );
    }

#if defined(TEST_FORCES)
    Coll( system, mpi_data, workspace->f_ele, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_vdw, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_be, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_lp, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_ov, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_un, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_ang, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_coa, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_pen, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_hb, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_tor, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
    Coll( system, mpi_data, workspace->f_con, RVEC_PTR_TYPE, mpi_data->mpi_rvec );
#endif

#endif
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
                dr3gamij_1 = ( r_ij * r_ij * r_ij +
                        system->reax_param.tbp[ index_tbp(system->my_atoms[i].type,
                            system->my_atoms[j].type, system->reax_param.num_atom_types) ].gamma );
                dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

//                ret = ((i == j) ? 0.5 : 1.0) * Tap * EV_to_KCALpMOL / dr3gamij_3;
                ret = Tap * EV_to_KCALpMOL / dr3gamij_3;
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


static inline real Compute_tabH( storage * const workspace, real r_ij,
        int ti, int tj, int num_atom_types )
{
    int r;
    real val, dif, base;
    int tmin = MIN( ti, tj );
    int tmax = MAX( ti, tj );
    const LR_lookup_table * const t = &workspace->LR[ index_lr( tmin, tmax, num_atom_types ) ];

    /* cubic spline interpolation */
    r = (int)(r_ij * t->inv_dx);
    if ( r == 0 )
    {
        ++r;
    }
    base = (real)(r + 1) * t->dx;
    dif = r_ij - base;
    val = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b) * dif
        + t->ele[r].a;
    val *= EV_to_KCALpMOL / C_ele;

    return val;
}


int Init_Forces( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int cm_top, btop_i;
    int ihb, jhb, ihb_top;
    int local, flag, renbr;
    real r_ij, cutoff;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
#if defined(HALF_LIST)
    int jhb_top;
#else
    int start_j, end_j;
    int btop_j;
#endif
    sparse_matrix * const H = &workspace->H;
    reax_list * const far_nbr_list = lists[FAR_NBRS];
    reax_list * const bond_list = lists[BONDS];
    reax_list * const hbond_list = lists[HBONDS];

    for ( i = 0; i < system->n; ++i )
    {
        workspace->bond_mark[i] = 0;
    }
    /* put ghost atoms to an infinite distance */
    for ( i = system->n; i < system->N; ++i )
    {
        workspace->bond_mark[i] = 1000;
    }

    cm_top = 0;
    H->n = system->n;
    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbr_list );
        end_i = End_Index( i, far_nbr_list );
#if defined(HALF_LIST)
        /* start at end because other atoms
         * can add to this atom's list (half-list) */
        btop_i = End_Index( i, bond_list );
#else
        btop_i = Start_Index( i, bond_list );
#endif
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
            cm_top = H->start[i];
            H->entries[cm_top].j = i;
            H->entries[cm_top].val = Init_Charge_Matrix_Entry( system, control, 
                    workspace, i, j, 0.0, DIAGONAL );
            ++cm_top;

            if ( control->hbond_cut > 0.0 )
            {
                ihb = sbp_i->p_hbond;

                if ( ihb == H_ATOM )
                {
#if defined(HALF_LIST)
                    /* start at end because other atoms
                     * can add to this atom's list (half-list) */
                    ihb_top = End_Index( atom_i->Hindex, hbond_list );
#else
                    ihb_top = Start_Index( atom_i->Hindex, hbond_list );
#endif
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
            nbr_pj = &far_nbr_list->far_nbr_list[pj];
            j = nbr_pj->nbr;
            atom_j = &system->my_atoms[j];

            if ( renbr == TRUE )
            {
                if ( nbr_pj->d <= cutoff )
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
                nbr_pj->dvec[0] = atom_j->x[0] - atom_i->x[0];
                nbr_pj->dvec[1] = atom_j->x[1] - atom_i->x[1];
                nbr_pj->dvec[2] = atom_j->x[2] - atom_i->x[2];
                nbr_pj->d = rvec_Norm_Sqr( nbr_pj->dvec );

                if ( nbr_pj->d <= SQR( cutoff ) )
                {
                    nbr_pj->d = SQRT( nbr_pj->d );
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
                r_ij = nbr_pj->d;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[ index_tbp(type_i, type_j,
                            system->reax_param.num_atom_types) ];

                if ( local == TRUE )
                {
                    /* H matrix entry */
#if defined(HALF_LIST)
                    if ( j < system->n || atom_i->orig_id < atom_j->orig_id ) //tryQEq||1
#endif
                    {
                        H->entries[cm_top].j = j;

                        if ( control->tabulate == 0 )
                        {
                            H->entries[cm_top].val = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else
                        {
                            H->entries[cm_top].val = Compute_tabH( workspace, r_ij, type_i, type_j,
                                    system->reax_param.num_atom_types );
                        }
                        ++cm_top;
                    }

                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.0
                            && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                            && nbr_pj->d <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;

                        if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                        {
                            hbond_list->hbond_list[ihb_top].nbr = j;
                            hbond_list->hbond_list[ihb_top].scl = 1;
                            hbond_list->hbond_list[ihb_top].ptr = nbr_pj;
                            ++ihb_top;
                        }
#if defined(HALF_LIST)
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( j < system->n && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                        {
                            jhb_top = End_Index( atom_j->Hindex, hbond_list );
                            hbond_list->hbond_list[jhb_top].nbr = i;
                            hbond_list->hbond_list[jhb_top].scl = -1;
                            hbond_list->hbond_list[jhb_top].ptr = nbr_pj;
                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbond_list );
                        }
#endif
                    }
                }

                /* uncorrected bond orders */
                if ( //(workspace->bond_mark[i] < 3 || workspace->bond_mark[j] < 3) &&
                    nbr_pj->d <= control->bond_cut
                    && BOp( workspace, bond_list, control->bo_cut,
                        i, btop_i, nbr_pj, sbp_i, sbp_j, twbp ) == TRUE )
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
        if ( local == TRUE )
        {
            H->end[i] = cm_top;
            if ( ihb == H_ATOM )
            {
                Set_End_Index( atom_i->Hindex, ihb_top, hbond_list );
            }
        }
    }

#if !defined(HALF_LIST)
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
#endif

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

        if ( i < system->n && H->end[i] - H->start[i] > system->max_cm_entries[i] )
        {
            workspace->realloc.cm = TRUE;
        }
    }

    return (workspace->realloc.bonds == FALSE 
            && workspace->realloc.hbonds == FALSE
            && workspace->realloc.cm == FALSE) ? SUCCESS : FAILURE;
}


int Init_Forces_No_Charges( reax_system * const system, control_params * const control,
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
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;
#if defined(HALF_LIST)
    int jhb_top;
#else
    int start_j, end_j;
    int btop_j;
#endif
    reax_list * const far_nbr_list = lists[FAR_NBRS];
    reax_list * const bond_list = lists[BONDS];
    reax_list * const hbond_list = lists[HBONDS];

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
#if defined(HALF_LIST)
        /* start at end because other atoms
         * can add to this atom's list (half-list) */
        btop_i = End_Index( i, bond_list );
#else
        btop_i = Start_Index( i, bond_list );
#endif
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
#if defined(HALF_LIST)
                /* start at end because other atoms
                 * can add to this atom's list (half-list) */
                ihb_top = End_Index( atom_i->Hindex, hbond_list );
#else
                ihb_top = Start_Index( atom_i->Hindex, hbond_list );
#endif
            }
            else
            {
                ihb_top = -1;
            }
        }

        /* update i-j distance - check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &far_nbr_list->far_nbr_list[pj];
            j = nbr_pj->nbr;
            atom_j = &system->my_atoms[j];

            if ( renbr == TRUE )
            {
                if ( nbr_pj->d <= cutoff )
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
                nbr_pj->dvec[0] = atom_j->x[0] - atom_i->x[0];
                nbr_pj->dvec[1] = atom_j->x[1] - atom_i->x[1];
                nbr_pj->dvec[2] = atom_j->x[2] - atom_i->x[2];
                nbr_pj->d = rvec_Norm_Sqr( nbr_pj->dvec );

                if ( nbr_pj->d <= SQR( cutoff ) )
                {
                    nbr_pj->d = SQRT( nbr_pj->d );
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
                twbp = &system->reax_param.tbp[ index_tbp(type_i, type_j,
                            system->reax_param.num_atom_types) ];

                if ( local == TRUE )
                {
                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.0
                            && (ihb == H_ATOM || ihb == H_BONDING_ATOM)
                            && nbr_pj->d <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;

                        if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                        {
                            hbond_list->hbond_list[ihb_top].nbr = j;
                            hbond_list->hbond_list[ihb_top].scl = 1;
                            hbond_list->hbond_list[ihb_top].ptr = nbr_pj;
                            ++ihb_top;
                        }
#if defined(HALF_LIST)
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( j < system->n && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                        {
                            jhb_top = End_Index( atom_j->Hindex, hbond_list );
                            hbond_list->hbond_list[jhb_top].nbr = i;
                            hbond_list->hbond_list[jhb_top].scl = -1;
                            hbond_list->hbond_list[jhb_top].ptr = nbr_pj;
                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbond_list );
                        }
#endif
                    }
                }

                /* uncorrected bond orders */
                if ( //(workspace->bond_mark[i] < 3 || workspace->bond_mark[j] < 3) &&
                    nbr_pj->d <= control->bond_cut
                    && BOp( workspace, bond_list, control->bo_cut,
                        i, btop_i, nbr_pj, sbp_i, sbp_j, twbp ) == TRUE )
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

#if !defined(HALF_LIST)
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
#endif

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
        reax_list ** const lists )
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
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i;
#if defined(HALF_LIST)
    reax_atom *atom_j;
#endif
    reax_list * const far_nbr_list = lists[FAR_NBRS];

    for ( i = 0; i < system->total_cap; ++i )
    {
        system->bonds[i] = 0;
        system->hbonds[i] = 0;
        if ( i < system->local_cap )
        {
            system->cm_entries[i] = 0;
        }
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
            local = TRUE;
            cutoff = control->nonb_cut;
            ++system->cm_entries[i];
            ihb = sbp_i->p_hbond;
        }
        else
        {
            local = FALSE;
            cutoff = control->bond_cut;
            ihb = NON_H_BONDING_ATOM;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &far_nbr_list->far_nbr_list[pj];
            j = nbr_pj->nbr;
#if defined(HALF_LIST)
            atom_j = &system->my_atoms[j];
#endif

            if ( nbr_pj->d <= cutoff )
            {
                type_j = system->my_atoms[j].type;
                r_ij = nbr_pj->d;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[ index_tbp(type_i, type_j,
                            system->reax_param.num_atom_types) ];

                if ( local == TRUE )
                {
#if defined(HALF_LIST)
                    if ( j < system->n || atom_i->orig_id < atom_j->orig_id ) //tryQEq ||1
#endif
                    {
                        ++system->cm_entries[i];
                    }

                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.1 && (ihb == H_ATOM || ihb == H_BONDING_ATOM) &&
                            nbr_pj->d <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;

                        if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                        {
                            ++system->hbonds[atom_i->Hindex];
                        }
#if defined(HALF_LIST)
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( j < system->n && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                        {
                            ++system->hbonds[atom_j->Hindex];
                        }
#endif
                    }
                }

                /* uncorrected bond orders */
                if ( nbr_pj->d <= control->bond_cut )
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
#if defined(HALF_LIST)
                        ++system->bonds[j];
#endif
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
#if defined(HALF_LIST)
        system->max_bonds[i] = MAX( (int)(2.0 * system->bonds[i] * SAFE_ZONE), MIN_BONDS );
#else
        system->max_bonds[i] = MAX( (int)(system->bonds[i] * SAFE_ZONE), MIN_BONDS );
#endif
        if ( system->reax_param.sbp[ system->my_atoms[i].type ].p_hbond == H_ATOM )
        {
            system->max_hbonds[ system->my_atoms[i].Hindex ] = MAX(
                    (int)(system->hbonds[ system->my_atoms[i].Hindex ] * SAFE_ZONE), MIN_HBONDS );
        }
        if ( i < system->local_cap )
        {
            system->max_cm_entries[i] = MAX( (int)(system->cm_entries[i] * SAFE_ZONE), MIN_CM_ENTRIES );
        }
    }
    for ( i = system->N; i < system->total_cap; ++i )
    {
        system->max_bonds[i] = MIN_BONDS;
        system->max_hbonds[i] = MIN_HBONDS;
        if ( i < system->local_cap )
        {
            system->max_cm_entries[i] = MIN_CM_ENTRIES;
        }
    }

    /* reductions to get totals */
    system->total_bonds = 0;
    system->total_hbonds = 0;
    system->total_cm_entries = 0;
    system->total_thbodies = 0;
    for ( i = 0; i < system->total_cap; ++i )
    {
        /* duplicate info in atom structs in case of
         * ownership transfer across processor boundaries */
        if ( i < system->n )
        {
            system->my_atoms[i].num_hbonds = system->hbonds[i];
        }
        if ( i < system->N )
        {
            system->my_atoms[i].num_bonds = system->bonds[i];
        }

        system->total_bonds += system->max_bonds[i];
        system->total_hbonds += system->max_hbonds[i];
        if ( i < system->local_cap )
        {
            system->total_cm_entries += system->max_cm_entries[i];
        }
#if defined(HALF_LIST)
        system->total_thbodies += SQR( system->max_bonds[i] / 2.0 );
#else
        system->total_thbodies += SQR( system->max_bonds[i] );
#endif
    }

    system->total_bonds = MAX( system->total_bonds, MIN_CAP * MIN_BONDS );
    system->total_hbonds = MAX( system->total_hbonds, MIN_CAP * MIN_HBONDS );
    system->total_cm_entries = MAX( system->total_cm_entries, MIN_CAP * MIN_CM_ENTRIES );
    system->total_thbodies = MAX( system->total_thbodies * SAFE_ZONE, MIN_3BODIES );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ estimate storages: total_cm_entries = %d, total_thbodies = %d\n",
            system->my_rank, system->total_cm_entries, system->total_thbodies );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
}


int Compute_Forces( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
    int charge_flag, ret;
#if defined(LOG_PERFORMANCE)
    real t_start = 0;

    if ( system->my_rank == MASTER_NODE )
    {
        t_start = Get_Time( );
    }
#endif

    /********* init forces ************/
    if ( control->charge_freq && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        charge_flag = TRUE;
    }
    else
    {
        charge_flag = FALSE;
    }

    if ( charge_flag == TRUE )
    {
        ret = Init_Forces( system, control, data, workspace, lists, out_control );
    }
    else
    {
        ret = Init_Forces_No_Charges( system, control, data, workspace, lists, out_control );
    }

    if ( ret == FAILURE )
    {
        Estimate_Storages( system, control, lists );
    }

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        Update_Timing_Info( &t_start, &(data->timing.init_forces) );
    }
#endif

    if ( ret == SUCCESS )
    {
        Compute_Bonded_Forces( system, control, data, workspace,
                lists, out_control );

#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.bonded) );
        }
#endif

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: completed bonded\n",
                 system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

    /**************** charges ************************/
#if defined(PURE_REAX)
        if ( charge_flag == TRUE )
        {
            Compute_Charges( system, control, data, workspace, out_control, mpi_data );
        }

#if defined(LOG_PERFORMANCE)
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.cm) );
        }
#endif

#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: qeq completed\n", system->my_rank, data->step);
        MPI_Barrier( MPI_COMM_WORLD );
#endif
#endif //PURE_REAX
    
        Compute_NonBonded_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    
#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.nonb) );
        }
#endif

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: nonbonded forces completed\n",
                 system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    
        Compute_Total_Force( system, control, data, workspace, lists, mpi_data );
    
#if defined(LOG_PERFORMANCE)
        //MPI_Barrier( MPI_COMM_WORLD );
        if ( system->my_rank == MASTER_NODE )
        {
            Update_Timing_Info( &t_start, &(data->timing.bonded) );
        }
#endif

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: total forces computed\n",
                 system->my_rank, data->step );
        //Print_Total_Force( system, data, workspace );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

#if defined(TEST_FORCES)
        Print_Force_Files( system, control, data, workspace, lists, out_control, mpi_data );
#endif
    }

    return ret;
}
