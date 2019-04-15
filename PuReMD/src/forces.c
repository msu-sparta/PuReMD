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
  #include "qEq.h"
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


interaction_function Interaction_Functions[NUM_INTRS];


static int compare_bonds( const void *p1, const void *p2 )
{
    return ((bond_data *)p1)->nbr - ((bond_data *)p2)->nbr;
}


static void Dummy_Interaction( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    ;
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


/* Computes a charge matrix entry using the Taper function */
static real Compute_H( real r, real gamma, real *ctap )
{
    real taper, dr3gamij_1, dr3gamij_3;

    taper = ctap[7] * r + ctap[6];
    taper = taper * r + ctap[5];
    taper = taper * r + ctap[4];
    taper = taper * r + ctap[3];
    taper = taper * r + ctap[2];
    taper = taper * r + ctap[1];
    taper = taper * r + ctap[0];

    dr3gamij_1 = ( r * r * r + gamma );
    dr3gamij_3 = pow( dr3gamij_1, 1.0 / 3.0 );

    return taper * EV_to_KCALpMOL / dr3gamij_3;
}


/* Computes a charge matrix entry using the force tabulation
 * (i.e., an arithmetic-reducing optimization) */
static real Compute_tabH( real r_ij, int ti, int tj )
{
    int r, tmin, tmax;
    real val, dif, base;
    LR_lookup_table *t;

    tmin = MIN( ti, tj );
    tmax = MAX( ti, tj );
    t = &LR[tmin][tmax];

    /* cubic spline interpolation */
    r = (int)(r_ij * t->inv_dx);

    if ( r == 0 )
    {
        ++r;
    }

    base = (real)(r + 1) * t->dx;
    dif = r_ij - base;
    val = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b) * dif +
          t->ele[r].a;
    val *= EV_to_KCALpMOL / C_ele;

    return val;
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
    //renbr = (data->step - data->prev_steps) % control->reneighbor == 0;
    renbr = is_refactoring_step( control, data );

    if ( !renbr )
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
    single_body_parameters *sbp_i;
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
    //renbr = (data->step - data->prev_steps) % control->reneighbor == 0;
    is_refactoring_step( control, data );

    nt_flag = 1;
    if( renbr )
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

            if( atom_i->nt_dir != -1 )
            {
                total_cnt[ atom_i->nt_dir ]++;
            }
        }

        total_sum[0] = system->n;
        for ( i = 1; i < 6; ++i )
        {
            total_sum[i] = total_sum[i-1] + total_cnt[i-1];
        }

        for ( i = system->n; i < system->N; ++i )
        {
            atom_i = &system->my_atoms[i];

            if( atom_i->nt_dir != -1 )
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

        sbp_i = &system->reax_param.sbp[type_i];

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
            H->entries[Htop].val = sbp_i->eta;
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
                            H->entries[Htop].val = Compute_H( r_ij, twbp->gamma, workspace->Tap );
                        }
                        else 
                        {
                            H->entries[Htop].val = Compute_tabH( r_ij, type_i, type_j );
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
                            H->entries[Htop].val = Compute_H( r_ij, twbp->gamma, workspace->Tap );
                        }
                        else 
                        {
                            H->entries[Htop].val = Compute_tabH( r_ij, type_i, type_j );
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

#if defined( DEBUG )
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
    single_body_parameters *sbp_i;
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
    //renbr = (data->step - data->prev_steps) % control->reneighbor == 0;
    renbr = is_refactoring_step( control, data );

    nt_flag = 1;
    if ( renbr )
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
            total_sum[i] = total_sum[i-1] + total_cnt[i-1];
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

        sbp_i = &system->reax_param.sbp[type_i];

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
            H->entries[Htop].val = sbp_i->eta;
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
                            H->entries[Htop].val = Compute_H(r_ij, twbp->gamma, workspace->Tap);
                        }
                        else 
                        {
                            H->entries[Htop].val = Compute_tabH(r_ij, type_i, type_j);
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
                            H->entries[Htop].val = Compute_H( r_ij, twbp->gamma, workspace->Tap );
                        }
                        else 
                        {
                            H->entries[Htop].val = Compute_tabH( r_ij, type_i, type_j );
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

#if defined( DEBUG )
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
    int type_i, type_j;
    int Htop;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbrs;
    single_body_parameters *sbp_i;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;

    far_nbrs = lists[FAR_NBRS];

    H = workspace->H;
    H->n = system->n;
    Htop = 0;

    for ( i = 0; i < system->n; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );

        sbp_i = &system->reax_param.sbp[type_i];

        H->start[i] = Htop;
        H->entries[Htop].j = i;
        H->entries[Htop].val = sbp_i->eta;
        ++Htop;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            // H matrix entry
            if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbrs->far_nbr_list.nbr[pj];
                atom_j = &system->my_atoms[j];
            
                if ( j < system->n || atom_i->orig_id < atom_j->orig_id )
                {
                    type_j = atom_j->type;
                    r_ij = far_nbrs->far_nbr_list.d[pj];
                    twbp = &system->reax_param.tbp[type_i][type_j];

                    H->entries[Htop].j = j;

                    if ( control->tabulate == 0 )
                    {
                        H->entries[Htop].val = Compute_H( r_ij, twbp->gamma, workspace->Tap );
                    }
                    else
                    {
                        H->entries[Htop].val = Compute_tabH( r_ij, type_i, type_j );
                    }

                    ++Htop;
                }
            }
        }

        H->end[i] = Htop;
    }

    workspace->realloc.Htop = Htop;

#if defined( DEBUG )
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
    int type_i, type_j;
    int Htop;
    real r_ij;
    sparse_matrix *H;
    reax_list *far_nbrs;
    single_body_parameters *sbp_i;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;

    far_nbrs = lists[FAR_NBRS];

    H = workspace->H;
    H->n = system->n;
    Htop = 0;

    for ( i = 0; i < system->n; ++i )
    {
        atom_i = &system->my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );

        sbp_i = &system->reax_param.sbp[type_i];

        H->start[i] = Htop;
        H->entries[Htop].j = i;
        H->entries[Htop].val = sbp_i->eta;
        ++Htop;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut )
            {
                j = far_nbrs->far_nbr_list.nbr[pj];
                atom_j = &system->my_atoms[j];
                type_j = atom_j->type;
                r_ij = far_nbrs->far_nbr_list.d[pj];
                twbp = &system->reax_param.tbp[type_i][type_j];

                // H matrix entry
                H->entries[Htop].j = j;

                if ( control->tabulate == 0 )
                {
                    H->entries[Htop].val = Compute_H(r_ij, twbp->gamma, workspace->Tap);
                }
                else
                {
                    H->entries[Htop].val = Compute_tabH(r_ij, type_i, type_j);
                }

                ++Htop;
            }
        }

        H->end[i] = Htop;
    }

    workspace->realloc.Htop = Htop;

#if defined( DEBUG )
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

#if defined( DEBUG )
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
    int ihb, jhb, ihb_top;
    real cutoff;
    reax_list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    reax_atom *atom_i, *atom_j;
    int start_j, end_j;
    int btop_i, btop_j;
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

#if defined( DEBUG )
    Print_Bonds( system, bonds, "debugbonds.out" );
    Print_Bond_List2( system, bonds, "pbonds.out" );
#endif

    Validate_Lists( system, workspace, lists, data->step,
            system->n, system->N, system->numH, comm );

}


void Init_Force_Functions( control_params *control )
{
    Interaction_Functions[0] = &BO;
    Interaction_Functions[1] = &Bonds; //Dummy_Interaction;
    Interaction_Functions[2] = &Atom_Energy; //Dummy_Interaction;
    Interaction_Functions[3] = &Valence_Angles; //Dummy_Interaction;
    Interaction_Functions[4] = &Torsion_Angles; //Dummy_Interaction;
    if ( control->hbond_cut > 0.0 )
    {
        Interaction_Functions[5] = &Hydrogen_Bonds;
    }
    else
    {
        Interaction_Functions[5] = &Dummy_Interaction;
    }
    Interaction_Functions[6] = &Dummy_Interaction; //empty
    Interaction_Functions[7] = &Dummy_Interaction; //empty
    Interaction_Functions[8] = &Dummy_Interaction; //empty
    Interaction_Functions[9] = &Dummy_Interaction; //empty
}


void Compute_Bonded_Forces( reax_system *system, control_params *control,
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
#if defined(DEBUG)
        fprintf( stderr, "p%d: starting f%d\n", system->my_rank, i );
        MPI_Barrier( comm );
#endif

        (Interaction_Functions[i])( system, control, data, workspace,
                lists, out_control );

#if defined(DEBUG)
        fprintf( stderr, "p%d: f%d done\n", system->my_rank, i );
        MPI_Barrier( comm );
#endif
    }
}


void Compute_NonBonded_Forces( reax_system *system, control_params *control,
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

#if defined(DEBUG)
    fprintf( stderr, "p%d: nonbonded forces done\n", system->my_rank );
    MPI_Barrier( comm );
#endif
}


/* this version of Compute_Total_Force computes forces from
 * coefficients accumulated by all interaction functions.
 * Saves enormous time & space! */
void Compute_Total_Force( reax_system *system, control_params *control,
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
                    Add_dBond_to_Forces( i, pj, workspace, lists );
                }
                else
                {
                    Add_dBond_to_Forces_NPT( i, pj, data, workspace, lists );
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


void Init_Forces( reax_system *system, control_params *control,
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


//void Init_Forces( reax_system *system, control_params *control,
//                  simulation_data *data, storage *workspace, reax_list **lists,
//                  output_controls *out_control, MPI_Comm comm, mpi_datatypes *mpi_data )
//{
//    int i, j, pj;
//    int start_i, end_i;
//    int type_i, type_j;
//    int Htop, btop_i, num_bonds, num_hbonds;
//    int ihb, jhb, ihb_top;
//    int local, flag, renbr;
//    real r_ij, cutoff;
//    sparse_matrix *H;
//    reax_list *far_nbrs, *bonds, *hbonds;
//    single_body_parameters *sbp_i, *sbp_j;
//    two_body_parameters *twbp;
//    reax_atom *atom_i, *atom_j;
//    int jhb_top;
//    int start_j, end_j;
//    int btop_j;
//#if defined(NEUTRAL_TERRITORY)
//    int mark[6];
//    int total_cnt[6];
//    int bin[6];
//    int total_sum[6];
//    int nt_flag;
//#endif
//
//    far_nbrs = lists[FAR_NBRS];
//    bonds = lists[BONDS];
//    hbonds = lists[HBONDS];
//
//
//    for ( i = 0; i < system->n; ++i )
//        workspace->bond_mark[i] = 0;
//    for ( i = system->n; i < system->N; ++i )
//    {
//        /* put ghost atoms to an infinite distance (i.e., 1000) */
//        workspace->bond_mark[i] = 1000;
//    }
//
//    H = workspace->H;
//    H->n = system->n;
//    Htop = 0;
//    num_bonds = 0;
//    num_hbonds = 0;
//    btop_i = 0;
//    renbr = (data->step - data->prev_steps) % control->reneighbor == 0;
//
//#if defined(NEUTRAL_TERRITORY)
//    nt_flag = 1;
//    if( renbr )
//    {
//        for ( i = 0; i < 6; ++i )
//        {
//            total_cnt[i] = 0;
//            bin[i] = 0;
//            total_sum[i] = 0;
//        }
//
//        for ( i = system->n; i < system->N; ++i )
//        {
//            atom_i = &system->my_atoms[i];
//
//            if( atom_i->nt_dir != -1 )
//            {
//                total_cnt[ atom_i->nt_dir ]++;
//            }
//        }
//
//        total_sum[0] = system->n;
//        for ( i = 1; i < 6; ++i )
//        {
//            total_sum[i] = total_sum[i-1] + total_cnt[i-1];
//        }
//
//        for ( i = system->n; i < system->N; ++i )
//        {
//            atom_i = &system->my_atoms[i];
//
//            if( atom_i->nt_dir != -1 )
//            {
//                atom_i->pos = total_sum[ atom_i->nt_dir ] + bin[ atom_i->nt_dir ];
//                bin[ atom_i->nt_dir ]++;
//            }
//        }
//        H->NT = total_sum[5] + total_cnt[5];
//    }
//
//    mark[0] = mark[1] = 1;
//    mark[2] = mark[3] = mark[4] = mark[5] = 2;
//#endif
//
//    for ( i = 0; i < system->N; ++i )
//    {
//        atom_i = &system->my_atoms[i];
//        type_i  = atom_i->type;
//        start_i = Start_Index(i, far_nbrs);
//        end_i = End_Index(i, far_nbrs);
//
//        if ( far_nbrs->format == HALF_LIST )
//        {
//            // start at end because other atoms
//            // can add to this atom's list (half-list)
//            btop_i = End_Index( i, bonds );
//        }
//        else if ( far_nbrs->format == FULL_LIST )
//        {
//            btop_i = Start_Index( i, bonds );
//        }
//        sbp_i = &system->reax_param.sbp[type_i];
//
//        if ( i < system->n )
//        {
//            local = 1;
//            cutoff = control->nonb_cut;
//        }
//#if defined(NEUTRAL_TERRITORY)
//        else if ( atom_i->nt_dir != -1 )
//        {
//            local = 2;
//            cutoff = control->nonb_cut;
//            nt_flag = 0;
//        }
//#endif
//        else
//        {
//            local = 0;
//            cutoff = control->bond_cut;
//        }
//
//        ihb = -1;
//        ihb_top = -1;
//        if ( local == 1 )
//        {
//            H->start[i] = Htop;
//            H->entries[Htop].j = i;
//            H->entries[Htop].val = sbp_i->eta;
//            ++Htop;
//
//            if ( control->hbond_cut > 0 )
//            {
//                ihb = sbp_i->p_hbond;
//                if ( ihb == 1 )
//                {
//                    if ( far_nbrs->format == HALF_LIST )
//                    {
//                        // start at end because other atoms
//                        // can add to this atom's list (half-list)
//                        ihb_top = End_Index( atom_i->Hindex, hbonds );
//                    }
//                    else if ( far_nbrs->format == FULL_LIST )
//                    {
//                        ihb_top = Start_Index( atom_i->Hindex, hbonds );
//                    }
//                }
//                else
//                {
//                    ihb_top = -1;
//                }
//            }
//        }
//
//        // update i-j distance - check if j is within cutoff
//        for ( pj = start_i; pj < end_i; ++pj )
//        {
//            j = far_nbrs->far_nbr_list.nbr[pj];
//            atom_j = &system->my_atoms[j];
//
//            if ( renbr )
//            {
//                if ( far_nbrs->far_nbr_list.d[pj] <= cutoff )
//                    flag = 1;
//                else
//                    flag = 0;
//            }
//            else
//            {
//                far_nbrs->far_nbr_list.dvec[pj][0] = atom_j->x[0] - atom_i->x[0];
//                far_nbrs->far_nbr_list.dvec[pj][1] = atom_j->x[1] - atom_i->x[1];
//                far_nbrs->far_nbr_list.dvec[pj][2] = atom_j->x[2] - atom_i->x[2];
//                far_nbrs->far_nbr_list.d[pj] = rvec_Norm_Sqr( far_nbrs->far_nbr_list.dvec[pj] );
//
//                if ( far_nbrs->far_nbr_list.d[pj] <= SQR(cutoff) )
//                {
//                    far_nbrs->far_nbr_list.d[pj] = sqrt( far_nbrs->far_nbr_list.d[pj] );
//                    flag = 1;
//                }
//                else
//                {
//                    flag = 0;
//                }
//            }
//
//            if ( flag )
//            {
//                type_j = atom_j->type;
//                r_ij = far_nbrs->far_nbr_list.d[pj];
//                sbp_j = &system->reax_param.sbp[type_j];
//                twbp = &system->reax_param.tbp[type_i][type_j];
//
//                if ( local == 1 )
//                {
//                    // H matrix entry
//#if defined(NEUTRAL_TERRITORY)
//                    if ( atom_j->nt_dir > 0 || (j < system->n
//                                && (H->format == SYM_FULL_MATRIX
//                                    || (H->format == SYM_HALF_MATRIX && i < j))) )
//                    {
//                        if( j < system->n )
//                        {
//                            H->entries[Htop].j = j;
//                        }
//                        else
//                        {
//                            H->entries[Htop].j = atom_j->pos;
//                        }
//
//                        if ( control->tabulate == 0 )
//                        {
//                            H->entries[Htop].val = Compute_H(r_ij, twbp->gamma, workspace->Tap);
//                        }
//                        else 
//                        {
//                            H->entries[Htop].val = Compute_tabH(r_ij, type_i, type_j);
//                        }
//
//                        ++Htop;
//                    }
//#else
//                    if ( (far_nbrs->format == HALF_LIST
//                            && (j < system->n || atom_i->orig_id < atom_j->orig_id))
//                      || far_nbrs->format == FULL_LIST )
//                    {
//                        H->entries[Htop].j = j;
//
//                        if ( control->tabulate == 0 )
//                        {
//                            H->entries[Htop].val = Compute_H(r_ij, twbp->gamma, workspace->Tap);
//                        }
//                        else
//                        {
//                            H->entries[Htop].val = Compute_tabH(r_ij, type_i, type_j);
//                        }
//
//                        ++Htop;
//                    }
//#endif
//
//                    // hydrogen bond lists
//                    if ( control->hbond_cut > 0.0
//                            && (ihb == 1 || ihb == 2)
//                            && far_nbrs->far_nbr_list.d[pj] <= control->hbond_cut )
//                    {
//                        // fprintf( stderr, "%d %d\n", atom1, atom2 );
//                        jhb = sbp_j->p_hbond;
//                        if ( ihb == 1 && jhb == 2 )
//                        {
//                            hbonds->hbond_list[ihb_top].nbr = j;
//                            hbonds->hbond_list[ihb_top].scl = 1;
//                            hbonds->hbond_list[ihb_top].ptr = pj;
//                            ++ihb_top;
//                            ++num_hbonds;
//                        }
//                        // only add to list for local j (far nbrs is half-list)
//                        else if ( far_nbrs->format == HALF_LIST
//                                && (j < system->n && ihb == 2 && jhb == 1) )
//                        {
//                            jhb_top = End_Index( atom_j->Hindex, hbonds );
//                            hbonds->hbond_list[jhb_top].nbr = i;
//                            hbonds->hbond_list[jhb_top].scl = -1;
//                            hbonds->hbond_list[jhb_top].ptr = pj;
//                            Set_End_Index( atom_j->Hindex, jhb_top + 1, hbonds );
//                            ++num_hbonds;
//                        }
//                    }
//                }
//#if defined(NEUTRAL_TERRITORY)
//                else if ( local == 2 )
//                {
//                    // H matrix entry 
//                    if( ( atom_j->nt_dir != -1 && mark[atom_i->nt_dir] != mark[atom_j->nt_dir] 
//                                && ( H->format == SYM_FULL_MATRIX
//                                    || (H->format == SYM_HALF_MATRIX && atom_i->pos < atom_j->pos))) 
//                            || ( j < system->n && atom_i->nt_dir != 0 && H->format == SYM_FULL_MATRIX ))
//                    {
//                        if( !nt_flag )
//                        {
//                            nt_flag = 1;
//                            H->start[atom_i->pos] = Htop;
//                        }
//
//                        if( j < system->n )
//                        {
//                            H->entries[Htop].j = j;
//                        }
//                        else
//                        {
//                            H->entries[Htop].j = atom_j->pos;
//                        }
//
//                        if ( control->tabulate == 0 )
//                        {
//                            H->entries[Htop].val = Compute_H(r_ij, twbp->gamma, workspace->Tap);
//                        }
//                        else 
//                        {
//                            H->entries[Htop].val = Compute_tabH(r_ij, type_i, type_j);
//                        }
//
//                        ++Htop;
//                    }
//                }
//#endif
//
//                // uncorrected bond orders
//                if ( //(workspace->bond_mark[i] < 3 || workspace->bond_mark[j] < 3) &&
//                    far_nbrs->far_nbr_list.d[pj] <= control->bond_cut
//                    && BOp( workspace, bonds, control->bo_cut,
//                         i, btop_i, far_nbrs->far_nbr_list.nbr[pj],
//                         &far_nbrs->far_nbr_list.rel_box[pj], far_nbrs->far_nbr_list.d[pj],
//                         &far_nbrs->far_nbr_list.dvec[pj], far_nbrs->format,
//                         sbp_i, sbp_j, twbp ) )
//                {
//                    num_bonds += 2;
//                    ++btop_i;
//
//                    if ( workspace->bond_mark[j] > workspace->bond_mark[i] + 1 )
//                        workspace->bond_mark[j] = workspace->bond_mark[i] + 1;
//                    else if ( workspace->bond_mark[i] > workspace->bond_mark[j] + 1 )
//                    {
//                        workspace->bond_mark[i] = workspace->bond_mark[j] + 1;
//                    }
//                }
//            }
//        }
//
//        Set_End_Index( i, btop_i, bonds );
//        if ( local == 1 )
//        {
//            H->end[i] = Htop;
//            if ( ihb == 1 )
//                Set_End_Index( atom_i->Hindex, ihb_top, hbonds );
//        }
//#if defined(NEUTRAL_TERRITORY)
//        else if ( local == 2 )
//        {
//            if( nt_flag )
//            {
//                H->end[atom_i->pos] = Htop;
//            }
//            else
//            {
//                 H->start[atom_i->pos] = 0;
//                 H->end[atom_i->pos] = 0;
//            }
//        }
//#endif
//    }
//
//    if ( far_nbrs->format == FULL_LIST )
//    {
//
//        for( i = 0; i < system->N; ++i )
//            qsort( &bonds->bond_list[Start_Index(i, bonds)],
//                    Num_Entries(i, bonds), sizeof(bond_data), compare_bonds );
//
//        // set sym_index for bonds list (far_nbrs full list)
//        for ( i = 0; i < system->N; ++i )
//        {
//            start_i = Start_Index( i, bonds );
//            end_i = End_Index( i, bonds );
//
//            for ( btop_i = start_i; btop_i < end_i; ++btop_i )
//            {
//                j = bonds->bond_list[btop_i].nbr;
//                start_j = Start_Index( j, bonds );
//                end_j = End_Index( j, bonds );
//
//                for ( btop_j = start_j; btop_j < end_j; ++btop_j )
//                {
//                    if ( bonds->bond_list[btop_j].nbr == i )
//                    {
//                        bonds->bond_list[btop_i].sym_index = btop_j;
//                        break;
//                    }
//                }
//            }
//        }
//    }
//
//#if defined(DEBUG)
//    Print_Sparse_Matrix2( system, H, NULL );
//#endif
//
//    workspace->realloc.Htop = Htop;
//    workspace->realloc.num_bonds = num_bonds;
//    workspace->realloc.num_hbonds = num_hbonds;
//
//#if defined(DEBUG_FOCUS)
//    fprintf( stderr, "p%d @ step%d: Htop = %d num_bonds = %d num_hbonds = %d\n",
//             system->my_rank, data->step, Htop, num_bonds, num_hbonds );
//    MPI_Barrier( comm );
//#endif
//
//#if defined( DEBUG )
//    Print_Bonds( system, bonds, "debugbonds.out" );
//    Print_Bond_List2( system, bonds, "pbonds.out" );
//    Print_Sparse_Matrix( system, H );
//    for ( i = 0; i < H->n; ++i )
//        for ( j = H->start[i]; j < H->end[i]; ++j )
//            fprintf( stderr, "%d %d %.15e\n",
//                     MIN(system->my_atoms[i].orig_id,
//                         system->my_atoms[H->entries[j].j].orig_id),
//                     MAX(system->my_atoms[i].orig_id,
//                         system->my_atoms[H->entries[j].j].orig_id),
//                     H->entries[j].val );
//#endif
//
//    Validate_Lists( system, workspace, lists, data->step,
//                    system->n, system->N, system->numH, comm );
//
//}


void Init_Forces_noQEq( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control, MPI_Comm comm )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int btop_i, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top;
    int local, flag, renbr;
    real r_ij, cutoff;
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
    //renbr = (data->step - data->prev_steps) % control->reneighbor == 0;
    renbr = is_refactoring_step( control, data );

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &(system->my_atoms[i]);
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);
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
        sbp_i = &(system->reax_param.sbp[type_i]);

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

            if ( renbr )
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
                r_ij = far_nbrs->far_nbr_list.d[pj];
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
#if defined( DEBUG )
    Print_Bonds( system, bonds, "debugbonds.out" );
    Print_Bond_List2( system, bonds, "pbonds.out" );
#endif

    Validate_Lists( system, workspace, lists, data->step,
            system->n, system->N, system->numH, comm );
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
    real r_ij, r2;
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
                    r2 = SQR(r_ij);

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
    int qeq_flag;
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
    if ( control->charge_freq && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        qeq_flag = 1;
    }
    else
    {
        qeq_flag = 0;
    }
#elif defined(LAMMPS_REAX)
    qeq_flag = 0;
#endif

    if ( qeq_flag )
    {
        Init_Forces( system, control, data, workspace, lists, out_control, comm, mpi_data );
    }
    else
    {
        Init_Forces_noQEq( system, control, data, workspace,
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
    if ( qeq_flag )
    {
        QEq( system, control, data, workspace, out_control, mpi_data );
    }

#if defined(LOG_PERFORMANCE)
    if ( system->my_rank == MASTER_NODE )
    {
        t_end = MPI_Wtime( );
        data->timing.cm += t_end - t_start;
        if ( control->cm_solver_pre_comp_refactor == -1 )
        {
            // fprintf( stdout, "step =  %d --- pc time = %.3lf --- loss time = %.3lf\n", data->step, data->timing.cm_last_pre_comp + data->timing.last_nbrs, data->timing.cm_total_loss);
            // fflush( stdout );
            if ( data->refactor )
            {
                data->refactor = 0;
                
                data->timing.cm_last_pre_comp = data->timing.cm_solver_pre_comp;

                data->timing.last_nbrs = data->timing.nbrs;

                data->last_pc_step = data->step;
                
                data->timing.cm_optimum = data->timing.cm_solver_pre_app + data->timing.cm_solver_spmv 
                    + data->timing.cm_solver_vector_ops + data->timing.cm_solver_orthog 
                    + data->timing.cm_solver_tri_solve;

                data->timing.cm_total_loss = ZERO;

            }
            else if ( data->step <= 4 )
            {
                data->timing.cm_optimum = data->timing.cm_solver_pre_app + data->timing.cm_solver_spmv
                    + data->timing.cm_solver_vector_ops + data->timing.cm_solver_orthog
                    + data->timing.cm_solver_tri_solve;
            }
            else
            {
                data->timing.cm_total_loss += data->timing.cm_solver_pre_app + data->timing.cm_solver_spmv 
                    + data->timing.cm_solver_vector_ops + data->timing.cm_solver_orthog 
                    + data->timing.cm_solver_tri_solve - data->timing.cm_optimum;

                if ( data->timing.cm_total_loss > data->timing.cm_last_pre_comp + data->timing.last_nbrs || 
                        data->step - data->last_pc_step + 1 >= control->reneighbor )
                {
                    data->refactor = 1;
                }
            }
        }
        t_start = t_end;
    }
    if ( control->cm_solver_pre_comp_refactor == -1 )
    {
        MPI_Bcast( &(data->refactor), 1, MPI_INT, MASTER_NODE, MPI_COMM_WORLD );
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
