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
  #include "charges.h"
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

  #include "reax_bond_orders.h"
  #include "reax_bonds.h"
  #include "reax_basic_comm.h"
  #include "reax_charges.h"
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


typedef enum
{
    DIAGONAL = 0,
    OFF_DIAGONAL = 1,
} MATRIX_ENTRY_POSITION;


static int compare_bonds( const void *p1, const void *p2 )
{
    return ((bond_data *)p1)->nbr - ((bond_data *)p2)->nbr;
}


static void Validate_Lists( reax_system *system, storage *workspace,
        reax_list **lists, int step, int n, int N, int numH, MPI_Comm comm )
{
    int i, comp, Hindex;
    reax_list *bonds, *hbonds;
    reallocate_data *realloc;
    realloc = &workspace->realloc;

    /* bond list */
    if ( N > 0 )
    {
        bonds = lists[BONDS];

        for ( i = 0; i < N; ++i )
        {
            // if( i < n ) - we need to update ghost estimates for delayed nbrings
            system->my_atoms[i].num_bonds = MAX(Num_Entries(i, bonds) * 2, MIN_BONDS);

            //if( End_Index(i, bonds) >= Start_Index(i+1, bonds)-2 )
            //workspace->realloc.bonds = 1;

            if ( i < N - 1 )
            {
                comp = Start_Index(i + 1, bonds);
            }
            else
            {
                comp = bonds->num_intrs;
            }

            if ( End_Index(i, bonds) > comp )
            {
                fprintf( stderr, "[ERROR] step%d-bondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                         step, i, End_Index(i, bonds), comp );
                MPI_Abort( comm, INSUFFICIENT_MEMORY );
            }
        }
    }

    /* hbonds list */
    if ( numH > 0 )
    {
        hbonds = lists[HBONDS];

        for ( i = 0; i < n; ++i )
        {
            Hindex = system->my_atoms[i].Hindex;

            if ( Hindex > -1 )
            {
                system->my_atoms[i].num_hbonds =
                    (int)(MAX( Num_Entries(Hindex, hbonds) * SAFER_ZONE, MIN_HBONDS ));

                //if( Num_Entries(i, hbonds) >=
                //(Start_Index(i+1,hbonds)-Start_Index(i,hbonds))*0.90/*DANGER_ZONE*/){
                //  workspace->realloc.hbonds = 1;

                if ( Hindex < numH - 1 )
                {
                    comp = Start_Index( Hindex + 1, hbonds );
                }
                else
                {
                    comp = hbonds->num_intrs;
                }

                if ( End_Index(Hindex, hbonds) > comp )
                {
                    fprintf(stderr, "[ERROR] step%d-hbondchk failed: H=%d end(H)=%d str(H+1)=%d\n",
                            step, Hindex, End_Index(Hindex, hbonds), comp );
                    MPI_Abort( comm, INSUFFICIENT_MEMORY );
                }
            }

//            if ( Hindex > -1 )
//            {
//                system->my_atoms[i].num_hbonds =
//                    MAX( Num_Entries(Hindex, hbonds) * SAFER_ZONE, MIN_HBONDS );

                //if( Num_Entries(i, hbonds) >=
                //(Start_Index(i+1,hbonds)-Start_Index(i,hbonds))*0.90/*DANGER_ZONE*/){
                //  workspace->realloc.hbonds = 1;
                
                //TODO
//                if ( Hindex < system->n - 1 )
//                {
//                    comp = Start_Index(Hindex + 1, hbonds);
//                }
//                else
//                {
//                    comp = hbonds->num_intrs;
//                }
//
//                if ( End_Index(Hindex, hbonds) > comp )
//                {
//                    fprintf(stderr, "[ERROR] step%d-hbondchk failed: H=%d end(H)=%d str(H+1)=%d\n",
//                            step, Hindex, End_Index(Hindex, hbonds), comp );
//                    MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
//                }
//            }
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
                t = &LR[MIN( system->my_atoms[i].type, system->my_atoms[j].type )]
                       [MAX( system->my_atoms[i].type, system->my_atoms[j].type )];

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


static inline real Init_Charge_Matrix_Entry( reax_system *system,
        control_params *control, storage *workspace,
        int i, int j, real r_ij, MATRIX_ENTRY_POSITION pos )
{
    real Tap, dr3gamij_1, dr3gamij_3, ret;

    ret = 0.0;

    switch ( control->charge_method )
    {
    case QEQ_CM:
    case EE_CM:
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
                        + POW( system->reax_param.tbp[system->my_atoms[i].type][system->my_atoms[j].type].gamma, -3.0 );
                dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

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

    case ACKS2_CM:
        switch ( pos )
        {
            case OFF_DIAGONAL:
                Tap = workspace->Tap[7] * r_ij
                    + workspace->Tap[6];
                Tap = Tap * r_ij + workspace->Tap[5];
                Tap = Tap * r_ij + workspace->Tap[4];
                Tap = Tap * r_ij + workspace->Tap[3];
                Tap = Tap * r_ij + workspace->Tap[2];
                Tap = Tap * r_ij + workspace->Tap[1];
                Tap = Tap * r_ij + workspace->Tap[0];

                /* shielding */
                dr3gamij_1 = r_ij * r_ij * r_ij
                        + POW( system->reax_param.tbp[system->my_atoms[i].type][system->my_atoms[j].type].gamma, -3.0 );
                dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

                ret = Tap * EV_to_KCALpMOL / dr3gamij_3;
            break;

            case DIAGONAL:
                /* parameters in electron-volts */
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
        control_params *control, reax_list *far_nbrs,
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
            H->start[system->N_cm - 1] = *Htop;
//            H_sp->start[system->N_cm - 1] = *H_sp_top;

            for ( i = 0; i < system->N_cm - 1; ++i )
            {
                H->entries[*Htop].j = i;
                H->entries[*Htop].val = 1.0;
                *Htop = *Htop + 1;

//                H_sp->entries[*H_sp_top].j = i;
//                H_sp->entries[*H_sp_top].val = 1.0;
//                *H_sp_top = *H_sp_top + 1;
            }

            H->entries[*Htop].j = system->N_cm - 1;
            H->entries[*Htop].val = 0.0;
            *Htop = *Htop + 1;

//            H_sp->entries[*H_sp_top].j = system->N_cm - 1;
//            H_sp->entries[*H_sp_top].val = 0.0;
//            *H_sp_top = *H_sp_top + 1;
            break;

        case ACKS2_CM:
            X_diag = smalloc( sizeof(real) * system->N,
                    "Init_Charge_Matrix_Remaining_Entries::X_diag", MPI_COMM_WORLD );

            for ( i = 0; i < system->N; ++i )
            {
                X_diag[i] = 0.0;
            }

            for ( i = 0; i < system->N; ++i )
            {
                H->start[system->N + i] = *Htop;
//                H_sp->start[system->N + i] = *H_sp_top;

                /* constraint on ref. value for kinetic energy potential */
                H->entries[*Htop].j = i;
                H->entries[*Htop].val = 1.0;
                *Htop = *Htop + 1;

//                H_sp->entries[*H_sp_top].j = i;
//                H_sp->entries[*H_sp_top].val = 1.0;
//                *H_sp_top = *H_sp_top + 1;

                /* kinetic energy terms */
                for ( pj = Start_Index(i, far_nbrs); pj < End_Index(i, far_nbrs); ++pj )
                {
                    /* exclude self-periodic images of atoms for
                     * kinetic energy term because the effective
                     * potential is the same on an atom and its periodic image */
                    if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut )
                    {
                        j = far_nbrs->far_nbr_list.nbr[pj];

                        xcut = 0.5 * ( system->reax_param.sbp[ system->my_atoms[i].type ].b_s_acks2
                                + system->reax_param.sbp[ system->my_atoms[j].type ].b_s_acks2 );

                        if ( far_nbrs->far_nbr_list.d[pj] < xcut )
                        {
                            d = far_nbrs->far_nbr_list.d[pj] / xcut;
                            bond_softness = system->reax_param.gp.l[34] * POW( d, 3.0 )
                                * POW( 1.0 - d, 6.0 );

                            if ( bond_softness > 0.0 )
                            {
                                val_flag = FALSE;

                                for ( target = H->start[system->N + i]; target < *Htop; ++target )
                                {
                                    if ( H->entries[target].j == system->N + j )
                                    {
                                        H->entries[target].val += bond_softness;
                                        val_flag = TRUE;
                                        break;
                                    }
                                }

                                if ( val_flag == FALSE )
                                {
                                    H->entries[*Htop].j = system->N + j;
                                    H->entries[*Htop].val = bond_softness;
                                    ++(*Htop);
                                }

//                                val_flag = FALSE;

//                                for ( target = H_sp->start[system->N + i]; target < *H_sp_top; ++target )
//                                {
//                                    if ( H_sp->entries[target].j == system->N + j )
//                                    {
//                                        H_sp->entries[target].val += bond_softness;
//                                        val_flag = TRUE;
//                                        break;
//                                    }
//                                }

//                                if ( val_flag == FALSE )
//                                {
//                                    H_sp->entries[*H_sp_top].j = system->N + j;
//                                    H_sp->entries[*H_sp_top].val = bond_softness;
//                                    ++(*H_sp_top);
//                                }

                                X_diag[i] -= bond_softness;
                                X_diag[j] -= bond_softness;
                            }
                        }
                    }
                }

                /* placeholders for diagonal entries, to be replaced below */
                H->entries[*Htop].j = system->N + i;
                H->entries[*Htop].val = 0.0;
                *Htop = *Htop + 1;

//                H_sp->entries[*H_sp_top].j = system->N + i;
//                H_sp->entries[*H_sp_top].val = 0.0;
//                *H_sp_top = *H_sp_top + 1;
            }

            /* second to last row */
            H->start[system->N_cm - 2] = *Htop;
//            H_sp->start[system->N_cm - 2] = *H_sp_top;

            /* place accumulated diagonal entries (needed second to last row marker above before this code) */
            for ( i = system->N; i < 2 * system->N; ++i )
            {
                for ( pj = H->start[i]; pj < H->start[i + 1]; ++pj )
                {
                    if ( H->entries[pj].j == i )
                    {
                        H->entries[pj].val = X_diag[i - system->N];
                        break;
                    }
                }

//                for ( pj = H_sp->start[i]; pj < H_sp->start[i + 1]; ++pj )
//                {
//                    if ( H_sp->entries[pj].j == i )
//                    {
//                        H_sp->entries[pj].val = X_diag[i - system->N];
//                        break;
//                    }
//                }
            }

            /* coupling with the kinetic energy potential */
            for ( i = 0; i < system->N; ++i )
            {
                H->entries[*Htop].j = system->N + i;
                H->entries[*Htop].val = 1.0;
                *Htop = *Htop + 1;

//                H_sp->entries[*H_sp_top].j = system->N + i;
//                H_sp->entries[*H_sp_top].val = 1.0;
//                *H_sp_top = *H_sp_top + 1;
            }

            /* explicitly store zero on diagonal */
            H->entries[*Htop].j = system->N_cm - 2;
            H->entries[*Htop].val = 0.0;
            *Htop = *Htop + 1;

//            H_sp->entries[*H_sp_top].j = system->N_cm - 2;
//            H_sp->entries[*H_sp_top].val = 0.0;
//            *H_sp_top = *H_sp_top + 1;

            /* last row */
            H->start[system->N_cm - 1] = *Htop;
//            H_sp->start[system->N_cm - 1] = *H_sp_top;

            for ( i = 0; i < system->N; ++i )
            {
                H->entries[*Htop].j = i;
                H->entries[*Htop].val = 1.0;
                *Htop = *Htop + 1;

//                H_sp->entries[*H_sp_top].j = i;
//                H_sp->entries[*H_sp_top].val = 1.0;
//                *H_sp_top = *H_sp_top + 1;
            }

            /* explicitly store zero on diagonal */
            H->entries[*Htop].j = system->N_cm - 1;
            H->entries[*Htop].val = 0.0;
            *Htop = *Htop + 1;

//            H_sp->entries[*H_sp_top].j = system->N_cm - 1;
//            H_sp->entries[*H_sp_top].val = 0.0;
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
        output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
{
    int i, j, pj;
    int start_i, end_i;
    int renbr;
    reax_list *far_nbrs;
    reax_atom *atom_i, *atom_j;

    far_nbrs = lists[FAR_NBRS];
    renbr = is_refactoring_step( control, data );

    if ( renbr == FALSE )
    {
        for ( i = 0; i < system->N; ++i )
        {
            atom_i = &system->my_atoms[i];
            start_i = Start_Index( i, far_nbrs );
            end_i = End_Index( i, far_nbrs );

            /* update distance and displacement vector between atoms i and j (i-j) */
            for ( pj = start_i; pj < end_i; ++pj )
            {
                j = far_nbrs->far_nbr_list.nbr[pj];
                atom_j = &system->my_atoms[j];
                
                far_nbrs->far_nbr_list.dvec[pj][0] = atom_j->x[0] - atom_i->x[0];
                far_nbrs->far_nbr_list.dvec[pj][1] = atom_j->x[1] - atom_i->x[1];
                far_nbrs->far_nbr_list.dvec[pj][2] = atom_j->x[2] - atom_i->x[2];
                far_nbrs->far_nbr_list.d[pj] = rvec_Norm_Sqr( far_nbrs->far_nbr_list.dvec[pj] );
                far_nbrs->far_nbr_list.d[pj] = sqrt( far_nbrs->far_nbr_list.d[pj] );
            }
        }
    }
}


#if defined(NEUTRAL_TERRITORY)
/* Compute the charge matrix entries and store the matrix in half format
 * using the far neighbors list (stored in full format) and according to
 * the neutral territory communication method */
static void Init_CM_Half_NT( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop;
    int local, renbr;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbrs;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int mark[6];
    int total_cnt[6];
    int bin[6];
    int total_sum[6];
    int nt_flag;

    far_nbrs = lists[FAR_NBRS];

    H = workspace->H;
    H->n = system->n;
    Htop = 0;
    renbr = is_refactoring_step( control, data );
    nt_flag = 1;

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

    mark[0] = mark[1] = 1;
    mark[2] = mark[3] = mark[4] = mark[5] = 2;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );

        if ( i < system->n )
        {
            local = 1;
        }
        else if ( atom_i->nt_dir != -1 )
        {
            local = 2;
            nt_flag = 0;
        }
        else
        {
            continue;
        }

        if ( local == 1 )
        {
            H->start[i] = Htop;
            H->entries[Htop].j = i;
            H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                    workspace, i, i, r_ij, DIAGONAL );
            ++Htop;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbrs->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];

            if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                type_j = atom_j->type;
                r_ij = far_nbrs->far_nbr_list.d[pj];
                twbp = &system->reax_param.tbp[type_i][type_j];

                if ( local == 1 )
                {
                    /* H matrix entry */
                    if ( atom_j->nt_dir > 0 || (j < system->n && i < j) )
                    {
                        if ( j < system->n )
                        {
                            H->entries[Htop].j = j;
                        }
                        else
                        {
                            H->entries[Htop].j = atom_j->pos;
                        }

                        if ( control->tabulate == 0 )
                        {
                            H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else 
                        {
                            H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab( system, control,
                                    workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                        }

                        ++Htop;
                    }

                }
                else if ( local == 2 )
                {
                    /* H matrix entry */
                    if ( atom_j->nt_dir != -1
                            && mark[atom_i->nt_dir] != mark[atom_j->nt_dir]
                            && atom_i->pos < atom_j->pos )
                    {
                        if ( !nt_flag )
                        {
                            nt_flag = 1;
                            H->start[atom_i->pos] = Htop;
                        }

                        //TODO: necessary?
                        if ( j < system->n )
                        {
                            H->entries[Htop].j = j;
                        }
                        else
                        {
                            H->entries[Htop].j = atom_j->pos;
                        }

                        if ( control->tabulate == 0 )
                        {
                            H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else 
                        {
                            H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab( system, control,
                                    workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                        }

                        ++Htop;
                    }
                }

            }
        }

        if ( local == 1 )
        {
            H->end[i] = Htop;
        }
        else if ( local == 2 )
        {
            if ( nt_flag )
            {
                H->end[atom_i->pos] = Htop;
            }
            else
            {
                 H->start[atom_i->pos] = 0;
                 H->end[atom_i->pos] = 0;
            }
        }
    }

    workspace->realloc.Htop = Htop;

#if defined( DEBUG_FOCUS )
    Print_Sparse_Matrix( system, H );
    for ( i = 0; i < H->n; ++i )
        for ( j = H->start[i]; j < H->end[i]; ++j )
            fprintf( stderr, "%d %d %.15e\n",
                     MIN(system->my_atoms[i].orig_id,
                         system->my_atoms[H->entries[j].j].orig_id),
                     MAX(system->my_atoms[i].orig_id,
                         system->my_atoms[H->entries[j].j].orig_id),
                     H->entries[j].val );
#endif
}


/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the neutral territory communication method */
static void Init_CM_Full_NT( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop;
    int local, renbr;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbrs;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int mark[6];
    int total_cnt[6];
    int bin[6];
    int total_sum[6];
    int nt_flag;

    far_nbrs = lists[FAR_NBRS];

    H = workspace->H;
    H->n = system->n;
    Htop = 0;
    renbr = is_refactoring_step( control, data );
    nt_flag = 1;

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

    mark[0] = mark[1] = 1;
    mark[2] = mark[3] = mark[4] = mark[5] = 2;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );

        if ( i < system->n )
        {
            local = 1;
        }
        else if ( atom_i->nt_dir != -1 )
        {
            local = 2;
            nt_flag = 0;
        }
        else
        {
            continue;
        }

        if ( local == 1 )
        {
            H->start[i] = Htop;
            H->entries[Htop].j = i;
            H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                    workspace, i, i, r_ij, DIAGONAL );
            ++Htop;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbrs->far_nbr_list.nbr[pj];
                atom_j = &system->my_atoms[j];

                type_j = atom_j->type;
                r_ij = far_nbrs->far_nbr_list.d[pj];
                twbp = &system->reax_param.tbp[type_i][type_j];

                if ( local == 1 )
                {
                    /* H matrix entry */
                    if ( atom_j->nt_dir > 0 || (j < system->n) )
                    {
                        if ( j < system->n )
                        {
                            H->entries[Htop].j = j;
                        }
                        else
                        {
                            H->entries[Htop].j = atom_j->pos;
                        }

                        if ( control->tabulate == 0 )
                        {
                            H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else 
                        {
                            H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab( system, control,
                                    workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                        }

                        ++Htop;
                    }

                }
                else if ( local == 2 )
                {
                    /* H matrix entry */
                    if ( ( atom_j->nt_dir != -1
                                && mark[atom_i->nt_dir] != mark[atom_j->nt_dir] )
                            || ( j < system->n && atom_i->nt_dir != 0 ) )
                    {
                        if ( !nt_flag )
                        {
                            nt_flag = 1;
                            H->start[atom_i->pos] = Htop;
                        }

                        if ( j < system->n )
                        {
                            H->entries[Htop].j = j;
                        }
                        else
                        {
                            H->entries[Htop].j = atom_j->pos;
                        }

                        if ( control->tabulate == 0 )
                        {
                            H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                                    workspace, i, j, r_ij, OFF_DIAGONAL );
                        }
                        else 
                        {
                            H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab( system, control,
                                    workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                        }

                        ++Htop;
                    }
                }

            }
        }

        if ( local == 1 )
        {
            H->end[i] = Htop;
        }
        else if ( local == 2 )
        {
            if ( nt_flag )
            {
                H->end[atom_i->pos] = Htop;
            }
            else
            {
                 H->start[atom_i->pos] = 0;
                 H->end[atom_i->pos] = 0;
            }
        }
    }

    workspace->realloc.Htop = Htop;

#if defined( DEBUG_FOCUS )
    Print_Sparse_Matrix( system, H );
    for ( i = 0; i < H->n; ++i )
        for ( j = H->start[i]; j < H->end[i]; ++j )
            fprintf( stderr, "%d %d %.15e\n",
                     MIN(system->my_atoms[i].orig_id,
                         system->my_atoms[H->entries[j].j].orig_id),
                     MAX(system->my_atoms[i].orig_id,
                         system->my_atoms[H->entries[j].j].orig_id),
                     H->entries[j].val );
#endif
}


#else
/* Compute the charge matrix entries and store the matrix in half format
 * using the far neighbors list (stored in half format) and according to
 * the full shell communication method */
static void Init_CM_Half_FS( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
{
    int i, j, pj;
    int start_i, end_i;
    int Htop;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbrs;
    reax_atom *atom_i, *atom_j;

    far_nbrs = lists[FAR_NBRS];

    H = workspace->H;
    H->n = system->n;
    Htop = 0;

    for ( i = 0; i < system->n; ++i )
    {
        atom_i = &system->my_atoms[i];
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );

        /* diagonal entry in the matrix */
        H->start[i] = Htop;
        H->entries[Htop].j = i;
        H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                workspace, i, i, r_ij, DIAGONAL );
        ++Htop;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbrs->far_nbr_list.nbr[pj];
                atom_j = &system->my_atoms[j];
            
                /* if j is a local atom OR
                 * if j is a ghost atom in the upper triangular region of the matrix */
                if ( j < system->n || atom_i->orig_id < atom_j->orig_id )
                {
                    r_ij = far_nbrs->far_nbr_list.d[pj];

                    H->entries[Htop].j = j;

                    if ( control->tabulate == 0 )
                    {
                        H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                                workspace, i, j, r_ij, OFF_DIAGONAL );
                    }
                    else
                    {
                        H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab( system, control,
                                workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                    }

                    ++Htop;
                }
            }
        }

        H->end[i] = Htop;
    }

    workspace->realloc.Htop = Htop;

#if defined( DEBUG_FOCUS )
    Print_Sparse_Matrix( system, H );
    for ( i = 0; i < H->n; ++i )
        for ( j = H->start[i]; j < H->end[i]; ++j )
            fprintf( stderr, "%d %d %.15e\n",
                     MIN(system->my_atoms[i].orig_id,
                         system->my_atoms[H->entries[j].j].orig_id),
                     MAX(system->my_atoms[i].orig_id,
                         system->my_atoms[H->entries[j].j].orig_id),
                     H->entries[j].val );
#endif
}


/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
static void Init_CM_Full_FS( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
{
    int i, j, pj;
    int start_i, end_i;
    int Htop;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbrs;

    far_nbrs = lists[FAR_NBRS];

    H = workspace->H;
    H->n = system->n;
    Htop = 0;

    for ( i = 0; i < system->n; ++i )
    {
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );

        /* diagonal entry in the matrix */
        H->start[i] = Htop;
        H->entries[Htop].j = i;
        H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                workspace, i, i, r_ij, DIAGONAL );
        ++Htop;

        /* off-diagonal entries in the matrix */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbrs->far_nbr_list.nbr[pj];
                r_ij = far_nbrs->far_nbr_list.d[pj];

                H->entries[Htop].j = j;

                if ( control->tabulate == 0 )
                {
                    H->entries[Htop].val = Init_Charge_Matrix_Entry( system, control,
                            workspace, i, j, r_ij, OFF_DIAGONAL );
                }
                else
                {
                    H->entries[Htop].val = Init_Charge_Matrix_Entry_Tab( system, control,
                            workspace->LR, i, j, r_ij, OFF_DIAGONAL );
                }

                ++Htop;
            }
        }

        H->end[i] = Htop;
    }

    workspace->realloc.Htop = Htop;

#if defined( DEBUG_FOCUS )
    Print_Sparse_Matrix( system, H );
    for ( i = 0; i < H->n; ++i )
        for ( j = H->start[i]; j < H->end[i]; ++j )
            fprintf( stderr, "%d %d %.15e\n",
                     MIN(system->my_atoms[i].orig_id,
                         system->my_atoms[H->entries[j].j].orig_id),
                     MAX(system->my_atoms[i].orig_id,
                         system->my_atoms[H->entries[j].j].orig_id),
                     H->entries[j].val );
#endif
}
#endif


/* Compute entries of the bonds/hbonds lists and store the lists in full format
 * using the far neighbors list (stored in half format)
 * 
 * Note: this version does NOT contain an optimization to restrict the bond_mark
 *  array to at most the 3-hop neighborhood */
static void Init_Bond_Half( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int btop_i, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top;
    int local;
    real cutoff;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int jhb_top;
    
    far_nbrs = lists[FAR_NBRS];
    bonds = lists[BONDS];
    hbonds = lists[HBONDS];

    for ( i = 0; i < system->n; ++i )
    {
        workspace->bond_mark[i] = 0;
    }
    for ( i = system->n; i < system->N; ++i )
    {
        /* put ghost atoms to an infinite distance (i.e., 1000) */
        workspace->bond_mark[i] = 1000;
    }

    num_bonds = 0;
    num_hbonds = 0;
    btop_i = 0;

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );

        /* start at end because other atoms
         * can add to this atom's list (half-list) */
        btop_i = End_Index( i, bonds );
        sbp_i = &system->reax_param.sbp[type_i];

        if ( i < system->n )
        {
            local = 1;
            cutoff = control->nonb_cut;
        }
        else
        {
            local = 0;
            cutoff = control->bond_cut;
        }

        ihb = -1;
        ihb_top = -1;
        if ( local == 1 )
        {
            if ( control->hbond_cut > 0 )
            {
                ihb = sbp_i->p_hbond;

                if ( ihb == 1 )
                {
                    /* start at end because other atoms
                     * can add to this atom's list (half-list) */ 
                    ihb_top = End_Index( atom_i->Hindex, hbonds );
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
            j = far_nbrs->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];
            
            if ( far_nbrs->far_nbr_list.d[pj] <= cutoff )
            {
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[type_i][type_j];

                if ( local == 1 )
                {
                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0
                            && (ihb == 1 || ihb == 2)
                            && far_nbrs->far_nbr_list.d[pj] <= control->hbond_cut )
                    {
                        // fprintf( stderr, "%d %d\n", atom1, atom2 );
                        jhb = sbp_j->p_hbond;

                        if ( ihb == 1 && jhb == 2 )
                        {
                            hbonds->hbond_list[ihb_top].nbr = j;
                            hbonds->hbond_list[ihb_top].scl = 1;
                            hbonds->hbond_list[ihb_top].ptr = pj;
                            ++ihb_top;
                            ++num_hbonds;
                        }
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( j < system->n && ihb == 2 && jhb == 1 ) 
                        {
                            jhb_top = End_Index( atom_j->Hindex, hbonds );
                            hbonds->hbond_list[jhb_top].nbr = i;
                            hbonds->hbond_list[jhb_top].scl = -1;
                            hbonds->hbond_list[jhb_top].ptr = pj;
                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbonds );
                            ++num_hbonds;
                        }
                    }
                }

                /* uncorrected bond orders */
                if ( far_nbrs->far_nbr_list.d[pj] <= control->bond_cut
                        && BOp( workspace, bonds, control->bo_cut,
                            i, btop_i, far_nbrs->far_nbr_list.nbr[pj],
                            &far_nbrs->far_nbr_list.rel_box[pj], far_nbrs->far_nbr_list.d[pj],
                            &far_nbrs->far_nbr_list.dvec[pj], far_nbrs->format,
                            sbp_i, sbp_j, twbp ) )
                {
                    num_bonds += 2;
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

        Set_End_Index( i, btop_i, bonds );

        if ( local == 1 && ihb == 1 )
        {
            Set_End_Index( atom_i->Hindex, ihb_top, hbonds );
        }
    }

    workspace->realloc.num_bonds = num_bonds;
    workspace->realloc.num_hbonds = num_hbonds;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: Htop = %d num_bonds = %d num_hbonds = %d\n",
        system->my_rank, data->step, workspace->realloc.Htop, num_bonds, num_hbonds );
    MPI_Barrier( comm );
#endif

#if defined( DEBUG_FOCUS )
    Print_Bonds( system, bonds, "debugbonds.out" );
    Print_Bond_List2( system, bonds, "pbonds.out" );
#endif

    Validate_Lists( system, workspace, lists, data->step,
            system->n, system->N, system->numH, comm );

}


/* Compute entries of the bonds/hbonds lists and store the lists in full format
 * using the far neighbors list (stored in full format) */
static void Init_Bond_Full( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int num_bonds, num_hbonds;
    int ihb, ihb_top;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int btop_i;
    int k, push;
    int *q;

    far_nbrs = lists[FAR_NBRS];
    bonds = lists[BONDS];
    hbonds = lists[HBONDS];
    num_hbonds = 0;
    push = 0;
    num_bonds = 0;
    btop_i = 0;
    bonds = lists[BONDS];

    q = smalloc( sizeof(int) * (system->N - system->n),
            "Init_Distance::q", MPI_COMM_WORLD );

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
        btop_i = End_Index( i, bonds );
        sbp_i = &system->reax_param.sbp[type_i];
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );
        ihb = sbp_i->p_hbond;
        ihb_top = Start_Index( atom_i->Hindex, hbonds );

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbrs->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];

            if ( control->hbond_cut > 0.0 && ihb == 1 )
            {
                /* check if j is within cutoff */
                if ( far_nbrs->far_nbr_list.d[pj] <= control->hbond_cut
                  && system->reax_param.sbp[atom_j->type].p_hbond == 2 )
                {
                    hbonds->hbond_list[ihb_top].nbr = j;
                    hbonds->hbond_list[ihb_top].scl = 1;
                    hbonds->hbond_list[ihb_top].ptr = pj;
                    ++ihb_top;
                    ++num_hbonds;
                }
            }

            if ( i <= j && far_nbrs->far_nbr_list.d[pj] <= control->bond_cut )
            {
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[type_i][type_j];

                if ( BOp( workspace, bonds, control->bo_cut,
                            i, btop_i, far_nbrs->far_nbr_list.nbr[pj],
                            &far_nbrs->far_nbr_list.rel_box[pj], far_nbrs->far_nbr_list.d[pj],
                            &far_nbrs->far_nbr_list.dvec[pj], far_nbrs->format,
                            sbp_i, sbp_j, twbp ) )
                {
                    num_bonds += 2;
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

        if ( control->hbond_cut > 0.0 && ihb == 1 )
        {
            Set_End_Index( atom_i->Hindex, ihb_top, hbonds );
        }

        Set_End_Index( i, btop_i, bonds );
    }

    /* bonds that are indirectly connected to local atoms */
    for ( k = 0; k < push; ++k )
    {
        i = q[k];
        workspace->bond_mark[i] -= 100;
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        btop_i = End_Index( i, bonds );
        sbp_i = &system->reax_param.sbp[type_i];
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbrs->far_nbr_list.nbr[pj];

            if ( workspace->bond_mark[i] == 3
                    && workspace->bond_mark[j] == 1000 )
            {
                continue;
            }

            atom_j = &system->my_atoms[j];

            if (  workspace->bond_mark[j] > 100
                    && far_nbrs->far_nbr_list.d[pj] <= control->bond_cut )
            {
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[type_i][type_j];

                if ( BOp( workspace, bonds, control->bo_cut,
                            i, btop_i, far_nbrs->far_nbr_list.nbr[pj],
                            &far_nbrs->far_nbr_list.rel_box[pj], far_nbrs->far_nbr_list.d[pj],
                            &far_nbrs->far_nbr_list.dvec[pj], far_nbrs->format,
                            sbp_i, sbp_j, twbp ) )
                {
                    num_bonds += 2;
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

        Set_End_Index( i, btop_i, bonds );
    }

    workspace->realloc.num_bonds = num_bonds;
    sfree( q, "Init_Bond_Full::q" );

    workspace->realloc.num_hbonds = num_hbonds;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: Htop = %d num_bonds = %d num_hbonds = %d\n",
             system->my_rank, data->step, workspace->realloc.Htop, workspace->realloc.num_bonds, num_hbonds );
    MPI_Barrier( comm );
#endif

#if defined( DEBUG_FOCUS )
    Print_Bonds( system, bonds, "debugbonds.out" );
    Print_Bond_List2( system, bonds, "pbonds.out" );
#endif

    Validate_Lists( system, workspace, lists, data->step,
            system->n, system->N, system->numH, comm );

}


static void Compute_Bonded_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm )
{
    int i;

    /* Mark beginning of a new timestep in bonded energy files */
#if defined(TEST_ENERGY)
    Debug_Marker_Bonded( out_control, data->step );
#endif

    /* Implement all force calls as function pointers */
    for ( i = 0; i < NUM_INTRS; i++ )
    {
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: starting f%d\n", system->my_rank, i );
        MPI_Barrier( comm );
#endif

        if ( control->intr_funcs[i] != NULL )
        {
            (control->intr_funcs[i])( system, control, data, workspace,
                    lists, out_control );
        }

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: f%d done\n", system->my_rank, i );
        MPI_Barrier( comm );
#endif
    }
}


static void Compute_NonBonded_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm )
{
    /* Mark beginning of a new timestep in nonbonded energy files */
#if defined(TEST_ENERGY)
    Debug_Marker_Nonbonded( out_control, data->step );
#endif

    /* van der Waals and Coulomb interactions */
    if ( control->tabulate == 0 )
    {
        vdW_Coulomb_Energy( system, control, data, workspace,
                lists, out_control );
    }
    else
    {
        Tabulated_vdW_Coulomb_Energy( system, control, data, workspace,
                lists, out_control );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: nonbonded forces done\n", system->my_rank );
    MPI_Barrier( comm );
#endif
}


/* this version of Compute_Total_Force computes forces from
 * coefficients accumulated by all interaction functions.
 * Saves enormous time & space! */
static void Compute_Total_Force( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, mpi_datatypes *mpi_data )
{
    int i, pj;
    reax_list *bonds;

    bonds = lists[BONDS];

    for ( i = 0; i < system->N; ++i )
    {
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            if ( i < bonds->bond_list[pj].nbr )
            {
                if ( control->virial == 0 )
                {
                    Add_dBond_to_Forces( i, pj, system, data, workspace, lists );
                }
                else
                {
                    Add_dBond_to_Forces_NPT( i, pj, system, data, workspace, lists );
                }
            }
        }
    }

    //Print_Total_Force( system, data, workspace );

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


static void Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
{
    double t_start, t_dist, t_cm, t_bond;
    double timings[3], t_total[3];
    
    t_start = MPI_Wtime( );

    Init_Distance( system, control, data, workspace, lists, out_control, comm, mpi_data );

    t_dist = MPI_Wtime( );

#if defined(NEUTRAL_TERRITORY)
    if ( workspace->H->format == SYM_HALF_MATRIX )
    {
        Init_CM_Half_NT( system, control, data, workspace, lists, out_control, comm, mpi_data );
    }
    else
    {
        Init_CM_Full_NT( system, control, data, workspace, lists, out_control, comm, mpi_data );
    }
#else
    if ( workspace->H->format == SYM_HALF_MATRIX )
    {
        Init_CM_Half_FS( system, control, data, workspace, lists, out_control, comm, mpi_data );
    }
    else
    {
        Init_CM_Full_FS( system, control, data, workspace, lists, out_control, comm, mpi_data );
    }
#endif

    t_cm = MPI_Wtime();

    if ( lists[FAR_NBRS]->format == HALF_LIST )
    {
        Init_Bond_Half( system, control, data, workspace, lists, out_control, comm, mpi_data );
    }
    else
    {
        Init_Bond_Full( system, control, data, workspace, lists, out_control, comm, mpi_data );
    }

    t_bond = MPI_Wtime();

    timings[0] = t_dist - t_start;
    timings[1] = t_cm - t_dist;
    timings[2] = t_bond - t_cm;

    MPI_Reduce( timings, t_total, 3, MPI_DOUBLE, MPI_SUM, MASTER_NODE, mpi_data->world );

    if ( system->my_rank == MASTER_NODE ) 
    {
        data->timing.init_dist += t_total[0] / control->nprocs;
        data->timing.init_cm += t_total[1] / control->nprocs;
        data->timing.init_bond += t_total[2] / control->nprocs;
    }

}


static void Init_Forces_No_Charges( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control, MPI_Comm comm )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int btop_i, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top;
    int local, flag, renbr;
    real cutoff;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int jhb_top;
    int start_j, end_j;
    int btop_j;

    far_nbrs = lists[FAR_NBRS];
    bonds = lists[BONDS];
    hbonds = lists[HBONDS];

    for ( i = 0; i < system->n; ++i )
    {
        workspace->bond_mark[i] = 0;
    }
    for ( i = system->n; i < system->N; ++i )
    {
        /* put ghost atoms to an infinite distance (i.e., 1000) */
        workspace->bond_mark[i] = 1000;
    }

    num_bonds = 0;
    num_hbonds = 0;
    btop_i = 0;
    renbr = is_refactoring_step( control, data );

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i = End_Index(i, far_nbrs);

        if ( far_nbrs->format == HALF_LIST )
        {
            /* start at end because other atoms
             * can add to this atom's list (half-list) */
            btop_i = End_Index( i, bonds );
        }
        else if ( far_nbrs->format == FULL_LIST )
        {
            btop_i = Start_Index( i, bonds );
        }
        sbp_i = &system->reax_param.sbp[type_i];

        if ( i < system->n )
        {
            local = 1;
            cutoff = MAX( control->hbond_cut, control->bond_cut );
        }
        else
        {
            local = 0;
            cutoff = control->bond_cut;
        }

        ihb = -1;
        ihb_top = -1;
        if ( local && control->hbond_cut > 0 )
        {
            ihb = sbp_i->p_hbond;
            if ( ihb == 1 )
            {
                if ( far_nbrs->format == HALF_LIST )
                {
                    /* start at end because other atoms
                     * can add to this atom's list (half-list) */
                    ihb_top = End_Index( atom_i->Hindex, hbonds );
                }
                else if ( far_nbrs->format == FULL_LIST )
                {
                    ihb_top = Start_Index( atom_i->Hindex, hbonds );
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
            j = far_nbrs->far_nbr_list.nbr[pj];
            atom_j = &system->my_atoms[j];

            if ( renbr == TRUE )
            {
                if ( far_nbrs->far_nbr_list.d[pj] <= cutoff )
                {
                    flag = 1;
                }
                else
                {
                    flag = 0;
                }
            }
            else
            {
                far_nbrs->far_nbr_list.dvec[pj][0] = atom_j->x[0] - atom_i->x[0];
                far_nbrs->far_nbr_list.dvec[pj][1] = atom_j->x[1] - atom_i->x[1];
                far_nbrs->far_nbr_list.dvec[pj][2] = atom_j->x[2] - atom_i->x[2];
                far_nbrs->far_nbr_list.d[pj] = rvec_Norm_Sqr( far_nbrs->far_nbr_list.dvec[pj] );

                if ( far_nbrs->far_nbr_list.d[pj] <= SQR(cutoff) )
                {
                    far_nbrs->far_nbr_list.d[pj] = sqrt( far_nbrs->far_nbr_list.d[pj] );
                    flag = 1;
                }
                else
                {
                    flag = 0;
                }
            }

            if ( flag )
            {
                type_j = atom_j->type;
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[type_i][type_j];

                if ( local )
                {
                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.0
                            && (ihb == 1 || ihb == 2)
                            && far_nbrs->far_nbr_list.d[pj] <= control->hbond_cut )
                    {
                        // fprintf( stderr, "%d %d\n", atom1, atom2 );
                        jhb = sbp_j->p_hbond;
                        if ( ihb == 1 && jhb == 2 )
                        {
                            hbonds->hbond_list[ihb_top].nbr = j;
                            hbonds->hbond_list[ihb_top].scl = 1;
                            hbonds->hbond_list[ihb_top].ptr = pj;
                            ++ihb_top;
                            ++num_hbonds;
                        }
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( far_nbrs->format == HALF_LIST
                                && (j < system->n && ihb == 2 && jhb == 1) )
                        {
                            jhb_top = End_Index( atom_j->Hindex, hbonds );
                            hbonds->hbond_list[jhb_top].nbr = i;
                            hbonds->hbond_list[jhb_top].scl = -1;
                            hbonds->hbond_list[jhb_top].ptr = pj;
                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbonds );
                            ++num_hbonds;
                        }
                    }
                }


                /* uncorrected bond orders */
                if ( //(workspace->bond_mark[i] < 3 || workspace->bond_mark[j] < 3) &&
                    far_nbrs->far_nbr_list.d[pj] <= control->bond_cut
                    && BOp( workspace, bonds, control->bo_cut,
                         i, btop_i, far_nbrs->far_nbr_list.nbr[pj],
                         &far_nbrs->far_nbr_list.rel_box[pj], far_nbrs->far_nbr_list.d[pj],
                         &far_nbrs->far_nbr_list.dvec[pj], far_nbrs->format,
                         sbp_i, sbp_j, twbp ) )
                {
                    num_bonds += 2;
                    ++btop_i;

                    if ( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
                        workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
                    else if ( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
                    {
                        workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
                        //if( workspace->bond_mark[i] == 1000 )
                        //  workspace->done_after[i] = pj;
                    }
                    //fprintf( stdout, "%d%d - %d(%d) %d(%d)\n",
                    //   i , j, i, workspace->bond_mark[i], j, workspace->bond_mark[j] );
                }
            }
        }

        Set_End_Index( i, btop_i, bonds );
        if ( local && ihb == 1 )
            Set_End_Index( atom_i->Hindex, ihb_top, hbonds );
    }

    if ( far_nbrs->format == FULL_LIST )
    {
        /* set sym_index for bonds list (far_nbrs full list) */
        for ( i = 0; i < system->N; ++i )
        {
            start_i = Start_Index( i, bonds );
            end_i = End_Index( i, bonds );

            for ( btop_i = start_i; btop_i < end_i; ++btop_i )
            {
                j = bonds->bond_list[btop_i].nbr;
                start_j = Start_Index( j, bonds );
                end_j = End_Index( j, bonds );

                for ( btop_j = start_j; btop_j < end_j; ++btop_j )
                {
                    if ( bonds->bond_list[btop_j].nbr == i )
                    {
                        bonds->bond_list[btop_i].sym_index = btop_j;
                        break;
                    }
                }
            }
        }
    }

    workspace->realloc.num_bonds = num_bonds;
    workspace->realloc.num_hbonds = num_hbonds;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: num_bonds = %d num_hbonds = %d\n",
             system->my_rank, data->step, num_bonds, num_hbonds );
    MPI_Barrier( comm );
#endif
#if defined(DEBUG_FOCUS)
    Print_Bonds( system, bonds, "debugbonds.out" );
    Print_Bond_List2( system, bonds, "pbonds.out" );
#endif

    Validate_Lists( system, workspace, lists, data->step,
            system->n, system->N, system->numH, comm );
}


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


void Estimate_Storages( reax_system *system, control_params *control,
        reax_list **lists, int *Htop, int *hb_top,
        int *bond_top, int *num_3body, MPI_Comm comm,
        int *matrix_dim, int cm_format )
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
    reax_list *far_nbrs;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;

    far_nbrs = lists[FAR_NBRS];
    *Htop = 0;
    *matrix_dim = 0;
    memset( hb_top, 0, sizeof(int) * system->local_cap );
    memset( bond_top, 0, sizeof(int) * system->total_cap );
    *num_3body = 0;

#if defined(NEUTRAL_TERRITORY)
    int mark[6] = {1, 1, 2, 2, 2, 2};
#endif

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i = End_Index(i, far_nbrs);
        sbp_i = &system->reax_param.sbp[type_i];

        if ( i < system->n )
        {
            local = 1;
            cutoff = control->nonb_cut;
            ++(*Htop);
            ++(*matrix_dim);
            ihb = sbp_i->p_hbond;
        }
#if defined(NEUTRAL_TERRITORY)
        else if ( atom_i->nt_dir != -1 )
        {
            local = 2;
            cutoff = control->nonb_cut;
            ++(*matrix_dim);
            ihb = -1;
        }
#endif
        else
        {
            local = 0;
            cutoff = control->bond_cut;
            ihb = -1;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbrs->far_nbr_list.nbr[pj];

#if !defined(NEUTRAL_TERRITORY)
            if ( far_nbrs->format == HALF_LIST )
#endif
            {
                atom_j = &system->my_atoms[j];
            }

            if ( far_nbrs->far_nbr_list.d[pj] <= cutoff )
            {
                type_j = system->my_atoms[j].type;
                r_ij = far_nbrs->far_nbr_list.d[pj];
                sbp_j = &system->reax_param.sbp[type_j];
                twbp = &system->reax_param.tbp[type_i][type_j];

                if ( local == 1 )
                {
#if defined(NEUTRAL_TERRITORY)
                    if( atom_j->nt_dir > 0 || j < system->n )
                    {
                        ++(*Htop);
                    }
#else
                    if ( (far_nbrs->format == HALF_LIST
                                && (j < system->n || atom_i->orig_id < atom_j->orig_id))
                            || far_nbrs->format == FULL_LIST )
                    {
                        ++(*Htop);
                    }
#endif

                    /* hydrogen bond lists */
                    if ( control->hbond_cut > 0.1
                            && (ihb == 1 || ihb == 2)
                            && far_nbrs->far_nbr_list.d[pj] <= control->hbond_cut )
                    {
                        jhb = sbp_j->p_hbond;

                        if ( ihb == 1 && jhb == 2 )
                        {
                            ++hb_top[i];
                        }
                        /* only add to list for local j (far nbrs is half-list) */
                        else if ( far_nbrs->format == HALF_LIST
                                && (j < system->n && ihb == 2 && jhb == 1) )
                        {
                            ++hb_top[j];
                        }
                    }
                }

#if defined(NEUTRAL_TERRITORY)
                else if ( local == 2 )
                {
                    if( ( atom_j->nt_dir != -1 && mark[atom_i->nt_dir] != mark[atom_j->nt_dir] ) 
                            || ( j < system->n && atom_i->nt_dir != 0 ))
                    {
                        ++(*Htop);
                    }
                }
#endif

                /* uncorrected bond orders */
                if ( far_nbrs->far_nbr_list.d[pj] <= control->bond_cut )
                {
                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0)
                    {
                        C12 = twbp->p_bo1 * pow( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * exp( C12 );
                    }
                    else
                    {
                        C12 = 0.0;
                        BO_s = 0.0;
                    }

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0)
                    {
                        C34 = twbp->p_bo3 * pow( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = exp( C34 );
                    }
                    else
                    {
                        C34 = 0.0;
                        BO_pi = 0.0;
                    }

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0)
                    {
                        C56 = twbp->p_bo5 * pow( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = exp( C56 );
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
                        if ( far_nbrs->format == HALF_LIST )
                        {
                            ++bond_top[j];
                        }
                    }
                }
            }
        }
    }

#if defined(NEUTRAL_TERRITORY)
    /* Since we don't know the NT atoms' position yet, Htop cannot be calculated accurately.
     * Therefore, we assume it is full and divide 2 if necessary. */
    if ( cm_format == SYM_HALF_MATRIX )
    {
        *Htop = (*Htop + system->n + 1) / 2;
    }
#endif

#if defined(NEUTRAL_TERRITORY)
    *matrix_dim = (int) MAX( *matrix_dim * SAFE_ZONE_NT, MIN_CAP );
    *Htop = (int) MAX( *Htop * SAFE_ZONE_NT, MIN_CAP * MIN_HENTRIES );
#else
    *matrix_dim = (int) MAX( *matrix_dim * SAFE_ZONE, MIN_CAP );
    *Htop = (int) MAX( *Htop * SAFE_ZONE, MIN_CAP * MIN_HENTRIES );
#endif

    for ( i = 0; i < system->n; ++i )
    {
        hb_top[i] = (int) MAX( hb_top[i] * SAFER_ZONE, MIN_HBONDS );
    }

    for ( i = 0; i < system->N; ++i )
    {
        *num_3body += SQR( bond_top[i] );
        //TODO: why x2?
        bond_top[i] = MAX( bond_top[i] * 2, MIN_BONDS );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ estimate storages: Htop = %d, num_3body = %d\n",
            system->my_rank, *Htop, *num_3body );
    MPI_Barrier( comm );
#endif
}


void Compute_Forces( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    MPI_Comm comm;
    int charge_flag;
#if defined(LOG_PERFORMANCE)
    real t_start = 0.0, t_end;

    if ( system->my_rank == MASTER_NODE )
    {
        t_start = MPI_Wtime();
    }
#endif

    comm = mpi_data->world;

    /********* init forces ************/
#if defined(PURE_REAX)
    if ( control->charge_freq > 0
            && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        charge_flag = TRUE;
    }
    else
    {
        charge_flag = FALSE;
    }
#elif defined(LAMMPS_REAX)
    charge_flag = FALSE;
#endif

    if ( charge_flag == TRUE )
    {
        Init_Forces( system, control, data, workspace,
                lists, out_control, comm, mpi_data );
    }
    else
    {
        Init_Forces_No_Charges( system, control, data, workspace,
                lists, out_control, comm );
    }

#if defined(LOG_PERFORMANCE)
    //MPI_Barrier( mpi_data->world );
    if ( system->my_rank == MASTER_NODE )
    {
        t_end = MPI_Wtime( );
        data->timing.init_forces += t_end - t_start;
        t_start = t_end;
    }
#endif

    /********* bonded interactions ************/
    Compute_Bonded_Forces( system, control, data, workspace,
            lists, out_control, mpi_data->world );

#if defined(LOG_PERFORMANCE)
    //MPI_Barrier( mpi_data->world );
    if ( system->my_rank == MASTER_NODE )
    {
        t_end = MPI_Wtime( );
        data->timing.bonded += t_end - t_start;
        t_start = t_end;
    }
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: completed bonded\n",
             system->my_rank, data->step );
    MPI_Barrier( mpi_data->world );
#endif

    /**************** qeq ************************/
#if defined(PURE_REAX)
    if ( charge_flag == TRUE )
    {
        Compute_Charges( system, control, data, workspace, out_control, mpi_data );
    }

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        t_end = MPI_Wtime( );
        data->timing.cm += t_end - t_start;
        /* dynamically determine preconditioner recomputation rate */
        if ( control->cm_solver_pre_comp_refactor == -1 )
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
        t_start = t_end;
    }
    if ( control->cm_solver_pre_comp_refactor == -1 )
    {
        MPI_Bcast( &data->refactor, 1, MPI_INT, MASTER_NODE, MPI_COMM_WORLD );
    }
#endif

#if defined(DEBUG_FOCUS)
    fprintf(stderr, "p%d @ step%d: qeq completed\n", system->my_rank, data->step);
    MPI_Barrier( mpi_data->world );
#endif
#endif //PURE_REAX

    /********* nonbonded interactions ************/
    Compute_NonBonded_Forces( system, control, data, workspace,
            lists, out_control, mpi_data->world );

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        t_end = MPI_Wtime( );
        data->timing.nonb += t_end - t_start;
        t_start = t_end;
    }
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: nonbonded forces completed\n",
             system->my_rank, data->step );
    MPI_Barrier( mpi_data->world );
#endif

    /*********** total force ***************/
    Compute_Total_Force( system, control, data, workspace, lists, mpi_data );

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        t_end = MPI_Wtime( );
        data->timing.bonded += t_end - t_start;
    }
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: total forces computed\n",
             system->my_rank, data->step );

    //Print_Total_Force( system, data, workspace );
    MPI_Barrier( mpi_data->world );
#endif

#if defined(TEST_FORCES)
    Print_Force_Files( system, control, data, workspace,
            lists, out_control, mpi_data );
#endif
}
