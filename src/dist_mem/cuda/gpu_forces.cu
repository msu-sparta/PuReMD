
#include "gpu_forces.h"

#include "gpu_bonds.h"
#include "gpu_bond_orders.h"
#include "gpu_charges.h"
#include "gpu_helpers.h"
#include "gpu_hydrogen_bonds.h"
#include "gpu_list.h"
#include "gpu_multi_body.h"
#include "gpu_neighbors.h"
#include "gpu_nonbonded.h"
#include "gpu_reduction.h"
#include "gpu_spar_lin_alg.h"
#include "gpu_torsion_angles.h"
#include "gpu_utils.h"
#include "gpu_valence_angles.h"

#include "../basic_comm.h"
#include "../forces.h"
#include "../index_utils.h"
#include "../tool_box.h"
#include "../vector.h"

#include <cub/util_ptx.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>


#define FULL_WARP_MASK (0xFFFFFFFF)


typedef enum
{
    DIAGONAL = 0,
    OFF_DIAGONAL = 1,
} MATRIX_ENTRY_POSITION;


GPU_DEVICE real Init_Charge_Matrix_Entry_Tab( LR_lookup_table const * const t_LR, real r_ij,
        int ti, int tj, int num_atom_types )
{
    int r, tmin, tmax;
    real val, dif, base;

    tmin = min( ti, tj );
    tmax = max( ti, tj );
    LR_lookup_table const * const t = &t_LR[ index_lr(tmin,tmax, num_atom_types) ];

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
    val *= EV_to_KCALpMOL / C_ELE;

    return val;
}


GPU_GLOBAL void k_init_end_index( int const * const intr_cnt,
        int const * const indices, int * const end_indices, int N )
{
    int i;

    i = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    end_indices[i] = indices[i] + intr_cnt[i];
}


GPU_GLOBAL void k_init_hbond_indices( reax_atom * const atoms,
        single_body_parameters const * const sbp,
        int const * const hbonds, int const * const max_hbonds,
        int * const indices, int * const end_indices, int N, int N_max )
{
    int i, hindex, flag;

    i = blockIdx.x * blockDim.x  + threadIdx.x;

    if ( i >= N_max )
    {
        return;
    }

    if ( i < N )
    {
        hindex = atoms[i].Hindex;

        flag = (sbp[atoms[i].type].p_hbond == H_ATOM
                || sbp[atoms[i].type].p_hbond == H_BONDING_ATOM ? TRUE : FALSE);
    }
    else
    {
        flag = FALSE;
    }

    indices[hindex] = (flag == TRUE ? max_hbonds[i] : 0);
    end_indices[hindex] = (flag == TRUE ? indices[hindex] + hbonds[i] : 0);
    atoms[i].num_hbonds = (flag == TRUE ? hbonds[i] : 0);
}


GPU_GLOBAL void k_print_hbond_info( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, reax_list hbond_list, int N )
{
    int i, type_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    type_i = my_atoms[i].type;

    printf( "atom %6d: ihb = %2d, ihb_top = %2d\n", i, sbp[type_i].p_hbond,
            Start_Index( my_atoms[i].Hindex, &hbond_list ) );
}


/* 1 thread computes the distances and displacement vectors of an atom for its neighbors
 * in the far neighbors list if it's a NOT re-neighboring step
 */
GPU_GLOBAL void k_init_dist( reax_atom const * const my_atoms,
        reax_list far_nbr_list, int N )
{
    int i, j, pj, start_i, end_i;
    rvec x_i, d;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    rvec_Copy( x_i, my_atoms[i].x );

    /* update distance and displacement vector between atoms i and j (i-j) */
    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];

        d[0] = my_atoms[j].x[0] - x_i[0];
        d[1] = my_atoms[j].x[1] - x_i[1];
        d[2] = my_atoms[j].x[2] - x_i[2];

        far_nbr_list.far_nbr_list.d[pj] = norm3d( d[0], d[1], d[2] );
        rvec_Copy( far_nbr_list.far_nbr_list.dvec[pj], d );
    }
}


/* 1 warp of threads computes the distances and displacement vectors of an atom for its neighbors
 * in the far neighbors list if it's a NOT re-neighboring step
 */
GPU_GLOBAL void k_init_dist_opt( reax_atom const * const my_atoms,
        reax_list far_nbr_list, int N )
{
    int j, pj, start_i, end_i, i, lane_id;
    rvec x_i, d;

    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( i >= N )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize; 
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    rvec_Copy( x_i, my_atoms[i].x );

    /* update distance and displacement vector between atoms i and j (i-j) */
    for ( pj = start_i + lane_id; pj < end_i; pj += warpSize )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];

        d[0] = my_atoms[j].x[0] - x_i[0];
        d[1] = my_atoms[j].x[1] - x_i[1];
        d[2] = my_atoms[j].x[2] - x_i[2];

        far_nbr_list.far_nbr_list.d[pj] = norm3d( d[0], d[1], d[2] );
        rvec_Copy( far_nbr_list.far_nbr_list.dvec[pj], d );
    }
}


/* Compute the charge matrix entries and store the matrix in half format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
GPU_GLOBAL void k_init_cm_qeq_half_fs( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        sparse_matrix H, real const * const tap_coef, real cutoff,
        reax_list far_nbr_list, int num_atom_types,
        int * const max_cm_entries, int * const realloc, int N )
{
    int i, j, pj, start_i, end_i, type_i, orig_id_i;
    int cm_top, num_cm_entries;
    real tap_coef_[TAPER_COEF_SIZE], tap, dr3gamij_1, dr3gamij_3, r_ij;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    cm_top = H.start[i];

    if ( i < H.n )
    {
        type_i = my_atoms[i].type;
        orig_id_i = my_atoms[i].orig_id;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        for ( pj = 0; pj < TAPER_COEF_SIZE; ++pj )
        {
            tap_coef_[pj] = tap_coef[pj];
        }

        /* diagonal entry in the matrix */
        H.j[cm_top] = i;
        H.val[cm_top] = sbp[type_i].eta;
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];

                /* if j is a local OR ghost atom in the upper triangular region of the matrix */
                if ( orig_id_i < my_atoms[j].orig_id )
                {
                    r_ij = far_nbr_list.far_nbr_list.d[pj];

                    H.j[cm_top] = j;

                    tap = tap_coef_[7] * r_ij + tap_coef_[6];
                    tap = tap * r_ij + tap_coef_[5];
                    tap = tap * r_ij + tap_coef_[4];
                    tap = tap * r_ij + tap_coef_[3];
                    tap = tap * r_ij + tap_coef_[2];
                    tap = tap * r_ij + tap_coef_[1];
                    tap = tap * r_ij + tap_coef_[0];    

                    /* shielding */
                    dr3gamij_1 = r_ij * r_ij * r_ij
                        + tbp[index_tbp(type_i, my_atoms[j].type, num_atom_types)].gamma;
                    dr3gamij_3 = RCBRT( dr3gamij_1 );

                    /* i == j: periodic self-interaction term
                     * i != j: general interaction term */
                    H.val[cm_top] = ((i == j) ? 0.5 : 1.0) * tap * EV_to_KCALpMOL * dr3gamij_3;

                    ++cm_top;
                }
            }
        }
    }

    H.end[i] = cm_top;
    num_cm_entries = cm_top - H.start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc = TRUE;
    }
}


/* Compute the tabulated charge matrix entries and store the matrix in half format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
GPU_GLOBAL void k_init_cm_qeq_half_fs_tab( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, sparse_matrix H, real cutoff,
        reax_list far_nbr_list, LR_lookup_table const * const t_LR, int num_atom_types,
        int * const max_cm_entries, int * const realloc, int N )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int cm_top;
    int num_cm_entries;
    real r_ij;
    reax_atom *atom_i, *atom_j;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    cm_top = H.start[i];

    if ( i < H.n )
    {
        atom_i = &my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry in the matrix */
        H.j[cm_top] = i;
        H.val[cm_top] = sbp[type_i].eta;
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list.far_nbr_list.nbr[pj];

            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                atom_j = &my_atoms[j];
                type_j = atom_j->type;

                /* if j is a local OR ghost atom in the upper triangular region of the matrix */
                if ( atom_i->orig_id < atom_j->orig_id )
                {
                    r_ij = far_nbr_list.far_nbr_list.d[pj];

                    H.j[cm_top] = j;
                    H.val[cm_top] = Init_Charge_Matrix_Entry_Tab( t_LR, r_ij,
                            type_i, type_j, num_atom_types );
                    ++cm_top;
                }
            }
        }
    }

    H.end[i] = cm_top;
    num_cm_entries = cm_top - H.start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc = TRUE;
    }
}


/* Compute the charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
GPU_GLOBAL void k_init_cm_qeq_full_fs( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        sparse_matrix H, real const * const tap_coef, real cutoff,
        reax_list far_nbr_list, int num_atom_types, int * const max_cm_entries,
        int * const realloc, int N )
{
    int i, j, pj, start_i, end_i, type_i;
    int cm_top, num_cm_entries;
    real tap_coef_[TAPER_COEF_SIZE], tap, dr3gamij_1, dr3gamij_3, r_ij;
    reax_atom *atom_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    cm_top = H.start[i];

    if ( i < H.n )
    {
        atom_i = &my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        for ( pj = 0; pj < TAPER_COEF_SIZE; ++pj )
        {
            tap_coef_[pj] = tap_coef[pj];
        }

        /* diagonal entry in the matrix */
        H.j[cm_top] = i;
        H.val[cm_top] = sbp[type_i].eta;
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];

                H.j[cm_top] = j;

                r_ij = far_nbr_list.far_nbr_list.d[pj];
                tap = tap_coef_[7] * r_ij + tap_coef_[6];
                tap = tap * r_ij + tap_coef_[5];
                tap = tap * r_ij + tap_coef_[4];
                tap = tap * r_ij + tap_coef_[3];
                tap = tap * r_ij + tap_coef_[2];
                tap = tap * r_ij + tap_coef_[1];
                tap = tap * r_ij + tap_coef_[0];    

                /* shielding */
                dr3gamij_1 = r_ij * r_ij * r_ij
                    + tbp[index_tbp(type_i, my_atoms[j].type, num_atom_types)].gamma;
                dr3gamij_3 = RCBRT( dr3gamij_1 );

                /* i == j: periodic self-interaction term
                 * i != j: general interaction term */
                H.val[cm_top] = ((i == j) ? 0.5 : 1.0) * tap * EV_to_KCALpMOL * dr3gamij_3;

                ++cm_top;
            }
        }
    }

    H.end[i] = cm_top;
    num_cm_entries = cm_top - H.start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc = TRUE;
    }
}


/* Compute the charge matrix entries for QEq and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
GPU_GLOBAL void k_init_cm_qeq_full_fs_opt( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        sparse_matrix H, real const * const tap_coef, real cutoff,
        reax_list far_nbr_list, int num_atom_types, int * const max_cm_entries,
        int * const realloc, int N )
{
    extern __shared__ cub::WarpScan<int>::TempStorage temp1[];
    int i, j, pj, lane_id, itr;
    int start_i, end_i, type_i;
    int cm_top, num_cm_entries, offset, flag;
    real tap_coef_[TAPER_COEF_SIZE], tap, dr3gamij_1, dr3gamij_3, r_ij;
    reax_atom *atom_i;

    /* all threads within a warp are assigned the same unique row 
     * in the charge matrix */
    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( i >= N )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
    cm_top = H.start[i];

    if ( i < H.n )
    {
        atom_i = &my_atoms[i];
        type_i = atom_i->type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        for ( pj = 0; pj < TAPER_COEF_SIZE; ++pj )
        {
            tap_coef_[pj] = tap_coef[pj];
        }

        /* diagonal entry in the matrix */
        if ( lane_id == 0 )
        {
            H.j[cm_top] = i;
            H.val[cm_top] = sbp[type_i].eta; 
        }
        ++cm_top;

        for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
        {
            offset = (pj < end_i && far_nbr_list.far_nbr_list.d[pj] <= cutoff) ? 1 : 0;
            flag = (offset == 1) ? TRUE : FALSE;
            cub::WarpScan<int>(temp1[threadIdx.x / warpSize]).ExclusiveSum(offset, offset);

            if ( flag == TRUE )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];

                H.j[cm_top + offset] = j;

                r_ij = far_nbr_list.far_nbr_list.d[pj];
                tap = tap_coef_[7] * r_ij + tap_coef_[6];
                tap = tap * r_ij + tap_coef_[5];
                tap = tap * r_ij + tap_coef_[4];
                tap = tap * r_ij + tap_coef_[3];
                tap = tap * r_ij + tap_coef_[2];
                tap = tap * r_ij + tap_coef_[1];
                tap = tap * r_ij + tap_coef_[0];    

                /* shielding */
                dr3gamij_1 = r_ij * r_ij * r_ij
                    + tbp[index_tbp(type_i, my_atoms[j].type, num_atom_types)].gamma;
                dr3gamij_3 = RCBRT( dr3gamij_1 );

                /* i == j: periodic self-interaction term
                 * i != j: general interaction term */
                H.val[cm_top + offset] = ((i == j) ? 0.5 : 1.0) * tap * EV_to_KCALpMOL * dr3gamij_3;
            }

            /* get cm_top from thread in last lane */
            cm_top = cm_top + offset + (flag == TRUE ? 1 : 0);
            cm_top = cub::ShuffleIndex<WARP_SIZE>( cm_top, warpSize - 1, FULL_WARP_MASK );

            pj += warpSize;
        }
    }

    if ( lane_id == 0 )
    {
        H.end[i] = cm_top;
        num_cm_entries = cm_top - H.start[i];

        /* reallocation check */
        if ( num_cm_entries > max_cm_entries[i] )
        {
            *realloc = TRUE;
        }
    }
}


/* Compute the tabulated charge matrix entries and store the matrix in full format
 * using the far neighbors list (stored in full format) and according to
 * the full shell communication method */
GPU_GLOBAL void k_init_cm_qeq_full_fs_tab( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, sparse_matrix H, real cutoff,
        reax_list far_nbr_list, LR_lookup_table *t_LR, int num_atom_types,
        int * const max_cm_entries, int * const realloc, int N )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i;
    int cm_top;
    int num_cm_entries;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    cm_top = H.start[i];

    if ( i < H.n )
    {
        type_i = my_atoms[i].type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry in the matrix */
        H.j[cm_top] = i;
        H.val[cm_top] = sbp[type_i].eta;
        ++cm_top;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbr_list.far_nbr_list.nbr[pj];

            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                H.j[cm_top] = j;
                H.val[cm_top] = Init_Charge_Matrix_Entry_Tab( t_LR,
                        far_nbr_list.far_nbr_list.d[pj],
                        type_i, my_atoms[j].type, num_atom_types );
                ++cm_top;
            }
        }
    }

    H.end[i] = cm_top;
    num_cm_entries = cm_top - H.start[i];

    /* reallocation check */
    if ( num_cm_entries > max_cm_entries[i] )
    {
        *realloc = TRUE;
    }
}


GPU_GLOBAL void k_init_bonds( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        real * const total_bond_order, rvec * const dDeltap_self,
        real cutoff1, real cutoff2, real bo_cut,
        reax_list far_nbr_list, reax_list bond_list, int n, int N,
        int num_atom_types, int * const max_bonds, int * const realloc )
{
    int i, j, pj, start_i, end_i;
    int type_i, type_j, tbp_ij;
    int btop_i, num_bonds;
    real total_bond_order_i;
    rvec dDeltap_self_i;
    real cutoff, r_ij;
    real C12, C34, C56;
    real BO_s, BO_pi, BO_pi2, BO;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    type_i = my_atoms[i].type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    btop_i = Start_Index( i, &bond_list );
    total_bond_order_i = 0.0;
    rvec_MakeZero( dDeltap_self_i );

    if ( i < n )
    {
        cutoff = cutoff1;
//        bond_mark[i] = 0;
    }
    else
    {
        cutoff = cutoff2;
        /* put ghost atoms to an infinite distance (i.e., 1000) */
//        bond_mark[i] = 1000;
    }

    /* check if j is within cutoff */
    for ( pj = start_i; pj < end_i; ++pj )
    {
        /* uncorrected bond orders */
        if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
        {
            j = far_nbr_list.far_nbr_list.nbr[pj];
            type_j = my_atoms[j].type;
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            tbp_ij = index_tbp(type_i, type_j, num_atom_types);

            /* uncorrected bond orders */
            if ( sbp[type_i].r_s > 0.0 && sbp[type_j].r_s > 0.0 )
            {
                C12 = tbp[tbp_ij].p_bo1 * POW( r_ij / tbp[tbp_ij].r_s, tbp[tbp_ij].p_bo2 );
                BO_s = (1.0 + bo_cut) * EXP( C12 );
            }
            else
            {
                C12 = 0.0;
                BO_s = 0.0;
            }

            if ( sbp[type_i].r_pi > 0.0 && sbp[type_j].r_pi > 0.0 )
            {
                C34 = tbp[tbp_ij].p_bo3 * POW( r_ij / tbp[tbp_ij].r_p, tbp[tbp_ij].p_bo4 );
                BO_pi = EXP( C34 );
            }
            else
            {
                C34 = 0.0;
                BO_pi = 0.0;
            }

            if ( sbp[type_i].r_pi_pi > 0.0 && sbp[type_j].r_pi_pi > 0.0 )
            {
                C56 = tbp[tbp_ij].p_bo5 * POW( r_ij / tbp[tbp_ij].r_pp, tbp[tbp_ij].p_bo6 );
                BO_pi2 = EXP( C56 );
            }
            else
            {
                C56 = 0.0;
                BO_pi2 = 0.0;
            }

            /* initially BO values are the uncorrected ones, page 1 */
            BO = BO_s + BO_pi + BO_pi2;

            if ( BO >= bo_cut )
            {
                /* compute and append bond info to list */
                GPU_Compute_BOp( bond_list, bo_cut, i, btop_i,
                        far_nbr_list.far_nbr_list.nbr[pj],
                        C12, C34, C56, BO_s, BO_pi, BO_pi2, BO,
                        &far_nbr_list.far_nbr_list.rel_box[pj],
                        far_nbr_list.far_nbr_list.d[pj],
                        &far_nbr_list.far_nbr_list.dvec[pj], far_nbr_list.format,
                        &tbp[tbp_ij], dDeltap_self_i, &total_bond_order_i );

                ++btop_i;

                /* TODO: future optimization if bond_mark implemented */
//                if ( bond_mark[j] > bond_mark[i] + 1 )
//                {
//                    bond_mark[j] = bond_mark[i] + 1;
//                }
//                else if ( bond_mark[i] > bond_mark[j] + 1 )
//                {
//                    bond_mark[i] = bond_mark[j] + 1;
//                }
            }
        }
    }

    Set_End_Index( i, btop_i, &bond_list );

    num_bonds = btop_i - Start_Index( i, &bond_list );

    /* copy bond info to atom structure
     * (needed for atom ownership transfer via MPI) */
    my_atoms[i].num_bonds = num_bonds;

    total_bond_order[i] = total_bond_order_i;
    rvec_Copy( dDeltap_self[i], dDeltap_self_i );

    /* reallocation check */
    if ( num_bonds > max_bonds[i] )
    {
        *realloc = TRUE;
    }
}


GPU_GLOBAL void k_init_bonds_opt( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        real * const total_bond_order, rvec * const dDeltap_self,
        real cutoff1, real cutoff2, real bo_cut,
        reax_list far_nbr_list, reax_list bond_list, int n, int N,
        int num_atom_types, int * const max_bonds, int * const realloc )
{
    extern __shared__ cub::WarpScan<int>::TempStorage temp21[];
    cub::WarpReduce<double>::TempStorage *temp22;
    int i, j, pj, warp_id, lane_id, itr;
    int start_i, end_i, tbp_ij;
    int type_i, type_j;
    int btop_i, offset, flag, num_bonds;
    real cutoff, r_ij;
    real C12, C34, C56;
    real BO_s, BO_pi, BO_pi2, BO;
    real total_bond_order_i;
    rvec dDeltap_self_i;

    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( i >= N )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
    temp22 = (cub::WarpReduce<double>::TempStorage *) &temp21[warp_id];
    type_i = my_atoms[i].type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    btop_i = Start_Index( i, &bond_list );
    total_bond_order_i = 0.0;
    rvec_MakeZero( dDeltap_self_i );

    if ( i < n )
    {
        cutoff = cutoff1;
//        bond_mark[i] = 0;
    }
    else
    {
        cutoff = cutoff2;
        /* put ghost atoms to an infinite distance (i.e., 1000) */
//        bond_mark[i] = 1000;
    }

    for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
    {
        /* uncorrected bond orders */
        if ( pj < end_i && far_nbr_list.far_nbr_list.d[pj] <= cutoff )
        {
            j = far_nbr_list.far_nbr_list.nbr[pj];
            type_j = my_atoms[j].type;
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            tbp_ij = index_tbp(type_i, type_j, num_atom_types);

            /* uncorrected bond orders */
            if ( sbp[type_i].r_s > 0.0 && sbp[type_j].r_s > 0.0 )
            {
                C12 = tbp[tbp_ij].p_bo1 * POW( r_ij / tbp[tbp_ij].r_s, tbp[tbp_ij].p_bo2 );
                BO_s = (1.0 + bo_cut) * EXP( C12 );
            }
            else
            {
                C12 = 0.0;
                BO_s = 0.0;
            }

            if ( sbp[type_i].r_pi > 0.0 && sbp[type_j].r_pi > 0.0 )
            {
                C34 = tbp[tbp_ij].p_bo3 * POW( r_ij / tbp[tbp_ij].r_p, tbp[tbp_ij].p_bo4 );
                BO_pi = EXP( C34 );
            }
            else
            {
                C34 = 0.0;
                BO_pi = 0.0;
            }

            if ( sbp[type_i].r_pi_pi > 0.0 && sbp[type_j].r_pi_pi > 0.0 )
            {
                C56 = tbp[tbp_ij].p_bo5 * POW( r_ij / tbp[tbp_ij].r_pp, tbp[tbp_ij].p_bo6 );
                BO_pi2 = EXP( C56 );
            }
            else
            {
                C56 = 0.0;
                BO_pi2 = 0.0;
            }
        }
        else
        {
            BO_s = 0.0;
            BO_pi = 0.0;
            BO_pi2 = 0.0;
        }

        /* initially BO values are the uncorrected ones, page 1 */
        BO = BO_s + BO_pi + BO_pi2;

        offset = (pj < end_i && far_nbr_list.far_nbr_list.d[pj] <= cutoff && BO >= bo_cut) ? 1 : 0;
        flag = (offset == 1) ? TRUE : FALSE;
        cub::WarpScan<int>(temp21[warp_id]).ExclusiveSum(offset, offset);

        if ( flag == TRUE )
        {
            /* compute and append bond info to list */
            GPU_Compute_BOp( bond_list, bo_cut, i, btop_i + offset,
                    far_nbr_list.far_nbr_list.nbr[pj],
                    C12, C34, C56, BO_s, BO_pi, BO_pi2, BO,
                    &far_nbr_list.far_nbr_list.rel_box[pj],
                    far_nbr_list.far_nbr_list.d[pj],
                    &far_nbr_list.far_nbr_list.dvec[pj], far_nbr_list.format,
                    &tbp[tbp_ij], dDeltap_self_i, &total_bond_order_i );

            /* TODO: future optimization if bond_mark implemented */
//            if ( bond_mark[j] > bond_mark[i] + 1 )
//            {
//                bond_mark[j] = bond_mark[i] + 1;
//            }
//            else if ( bond_mark[i] > bond_mark[j] + 1 )
//            {
//                bond_mark[i] = bond_mark[j] + 1;
//            }
        }

        /* get btop_i from thread in last lane */
        btop_i = btop_i + offset + (flag == TRUE ? 1 : 0);
        btop_i = cub::ShuffleIndex<WARP_SIZE>( btop_i, warpSize - 1, FULL_WARP_MASK );

        pj += warpSize;
    }

    total_bond_order_i = cub::WarpReduce<double>(temp22[warp_id]).Sum(total_bond_order_i);
    dDeltap_self_i[0] = cub::WarpReduce<double>(temp22[warp_id]).Sum(dDeltap_self_i[0]);
    dDeltap_self_i[1] = cub::WarpReduce<double>(temp22[warp_id]).Sum(dDeltap_self_i[1]);
    dDeltap_self_i[2] = cub::WarpReduce<double>(temp22[warp_id]).Sum(dDeltap_self_i[2]);

    if ( lane_id == 0 )
    {
        Set_End_Index( i, btop_i, &bond_list );

        num_bonds = btop_i - Start_Index( i, &bond_list );

        /* copy bond info to atom structure
         * (needed for atom ownership transfer via MPI) */
        my_atoms[i].num_bonds = num_bonds;

        total_bond_order[i] = total_bond_order_i;
        rvec_Copy( dDeltap_self[i], dDeltap_self_i );

        /* reallocation check */
        if ( num_bonds > max_bonds[i] )
        {
            *realloc = TRUE;
        }
    }
}


/* Construct the interaction list for hydrogen bonds */
GPU_GLOBAL void k_init_hbonds( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, real cutoff,
        reax_list far_nbr_list, reax_list hbond_list,
        int n, int N, int num_atom_types, int * const max_hbonds, int * const realloc )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb, ihb_top;
    int num_hbonds;
    reax_atom *atom_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    atom_i = &my_atoms[i];
    type_i = atom_i->type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    ihb = sbp[type_i].p_hbond;

    ihb_top = Start_Index( atom_i->Hindex, &hbond_list );

    if ( (i < n && ihb == H_ATOM) || ihb == H_BONDING_ATOM )
    {
        /* check if j is within cutoff */
        for ( pj = start_i; pj < end_i; ++pj )
        {
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                jhb = sbp[type_j].p_hbond;

                /* atom i: H bonding, ghost
                 * atom j: H atom, native */
                if ( i >= n && j < n
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                {
                    hbond_list.hbond_list.nbr[ihb_top] = j;
                    hbond_list.hbond_list.scl[ihb_top] = -1;
                    hbond_list.hbond_list.ptr[ihb_top] = pj;
#if !defined(GPU_KERNEL_ATOMIC)
//                    hbond_list.hbond_list.sym_index[ihb_top] = -1;
//                    rvec_MakeZero( hbond_list.hbond_list.f_hb[ihb_top] );
#endif

                    ++ihb_top;
                }
                /* atom i: H atom, native
                 * atom j: H bonding atom */
                else if ( i < n
                        && ihb == H_ATOM && jhb == H_BONDING_ATOM )
                {
                    hbond_list.hbond_list.nbr[ihb_top] = j;
                    hbond_list.hbond_list.scl[ihb_top] = 1;
                    hbond_list.hbond_list.ptr[ihb_top] = pj;
#if !defined(GPU_KERNEL_ATOMIC)
//                    hbond_list.hbond_list.sym_index[ihb_top] = -1;
//                    rvec_MakeZero( hbond_list.hbond_list.f_hb[ihb_top] );
#endif

                    ++ihb_top;
                }
                /* atom i: H bonding atom, native
                 * atom j: H atom, native */
                else if ( i < n
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                {
                    hbond_list.hbond_list.nbr[ihb_top] = j;
                    hbond_list.hbond_list.scl[ihb_top] = -1;
                    hbond_list.hbond_list.ptr[ihb_top] = pj;
#if !defined(GPU_KERNEL_ATOMIC)
//                    hbond_list.hbond_list.sym_index[ihb_top] = -1;
//                    rvec_MakeZero( hbond_list.hbond_list.f_hb[ihb_top] );
#endif

                    ++ihb_top;
                }
            }
        }
    }

    Set_End_Index( atom_i->Hindex, ihb_top, &hbond_list );

    num_hbonds = ihb_top - Start_Index( atom_i->Hindex, &hbond_list );

    /* copy hbond info to atom structure
     * (needed for atom ownership transfer via MPI) */
    my_atoms[i].num_hbonds = num_hbonds;

    /* reallocation check */
    if ( num_hbonds > max_hbonds[i] )
    {
        *realloc = TRUE;
    }
}


/* Construct the interaction list for hydrogen bonds */
GPU_GLOBAL void k_init_hbonds_opt( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, real cutoff,
        reax_list far_nbr_list, reax_list hbond_list,
        int n, int N, int num_atom_types, int * const max_hbonds, int * const realloc )
{
    extern __shared__ cub::WarpScan<int>::TempStorage temp3[];
    int i, j, pj, lane_id, itr;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb, ihb_top, offset, flag;
    int num_hbonds;
    reax_atom *atom_i;

    /* all threads within a warp are assigned the bonds
     * for a unique atom */
    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( i >= N )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
    atom_i = &my_atoms[i];
    type_i = atom_i->type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    ihb = sbp[type_i].p_hbond;

    ihb_top = Start_Index( atom_i->Hindex, &hbond_list );

    if ( (i < n && ihb == H_ATOM) || ihb == H_BONDING_ATOM )
    {
        for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
        {
            if ( pj < end_i )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                jhb = sbp[type_j].p_hbond;

                offset = (pj < end_i && far_nbr_list.far_nbr_list.d[pj] <= cutoff
                        && ((i >= n && j < n && ihb == H_BONDING_ATOM && jhb == H_ATOM)
                            || (i < n && ihb == H_ATOM && jhb == H_BONDING_ATOM)
                            || (i < n && ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n))) ? 1 : 0;
            }
            else
            {
                offset = 0;
            }

            flag = (offset == 1) ? TRUE : FALSE;
            cub::WarpScan<int>(temp3[threadIdx.x / warpSize]).ExclusiveSum(offset, offset);

            if ( flag == TRUE )
            {
                /* atom i: H bonding, ghost
                 * atom j: H atom, native */
                if ( i >= n && j < n
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM )
                {
                    hbond_list.hbond_list.nbr[ihb_top + offset] = j;
                    hbond_list.hbond_list.scl[ihb_top + offset] = -1;
                    hbond_list.hbond_list.ptr[ihb_top + offset] = pj;
#if !defined(GPU_KERNEL_ATOMIC)
//                    hbond_list.hbond_list.sym_index[ihb_top + offset] = -1;
//                    rvec_MakeZero( hbond_list.hbond_list.f_hb[ihb_top + offset] );
#endif
                }
                /* atom i: H atom, native
                 * atom j: H bonding atom */
                else if ( i < n
                        && ihb == H_ATOM && jhb == H_BONDING_ATOM )
                {
                    hbond_list.hbond_list.nbr[ihb_top + offset] = j;
                    hbond_list.hbond_list.scl[ihb_top + offset] = 1;
                    hbond_list.hbond_list.ptr[ihb_top + offset] = pj;
#if !defined(GPU_KERNEL_ATOMIC)
//                    hbond_list.hbond_list.sym_index[ihb_top + offset] = -1;
//                    rvec_MakeZero( hbond_list.hbond_list.f_hb[ihb_top + offset] );
#endif
                }
                /* atom i: H bonding atom, native
                 * atom j: H atom, native */
                else if ( i < n
                        && ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                {
                    hbond_list.hbond_list.nbr[ihb_top + offset] = j;
                    hbond_list.hbond_list.scl[ihb_top + offset] = -1;
                    hbond_list.hbond_list.ptr[ihb_top + offset] = pj;
#if !defined(GPU_KERNEL_ATOMIC)
//                    hbond_list.hbond_list.sym_index[ihb_top + offset] = -1;
//                    rvec_MakeZero( hbond_list.hbond_list.f_hb[ihb_top + offset] );
#endif
                }
            }

            /* get ihb_top from thread in last lane */
            ihb_top = ihb_top + offset + (flag == TRUE ? 1 : 0);
            ihb_top = cub::ShuffleIndex<WARP_SIZE>( ihb_top, warpSize - 1, FULL_WARP_MASK );

            pj += warpSize;
        }
    }

    if ( lane_id == 0 )
    {
        Set_End_Index( atom_i->Hindex, ihb_top, &hbond_list );

        num_hbonds = ihb_top - Start_Index( atom_i->Hindex, &hbond_list );

        /* copy hbond info to atom structure
         * (needed for atom ownership transfer via MPI) */
        my_atoms[i].num_hbonds = num_hbonds;

        /* reallocation check */
        if ( num_hbonds > max_hbonds[i] )
        {
            *realloc = TRUE;
        }
    }
}


/* Construct the interaction list for bonds */
GPU_GLOBAL void k_estimate_storages_cm_half( reax_atom const * const my_atoms,
        real cutoff, reax_list far_nbr_list, int cm_n, int cm_n_max,
        int * const cm_entries, int * const max_cm_entries )
{
    int i, j, pj; 
    int start_i, end_i;
    int num_cm_entries;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= cm_n_max )
    {
        return;
    }

    num_cm_entries = 0;

    if ( i < cm_n )
    {
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry */
        ++num_cm_entries;

        for ( pj = start_i; pj < end_i; ++pj )
        { 
            j = far_nbr_list.far_nbr_list.nbr[pj];

            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff
                    && (j < cm_n || my_atoms[i].orig_id < my_atoms[j].orig_id) )
            {
                ++num_cm_entries;
            }
        }
    }

    cm_entries[i] = num_cm_entries;
    /* round up to the nearest multiple of warp size to ensure that reads along
     * rows can be coalesced */
    max_cm_entries[i] = max( ((int) CEIL( num_cm_entries * SAFE_ZONE )
                + warpSize - 1) / warpSize * warpSize, MIN_CM_ENTRIES );
}


GPU_GLOBAL void k_estimate_storages_cm_full( real cutoff,
        reax_list far_nbr_list, int cm_n, int cm_n_max,
        int * const cm_entries, int * const max_cm_entries )
{
    int i, pj; 
    int start_i, end_i;
    int num_cm_entries;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= cm_n_max )
    {
        return;
    }

    num_cm_entries = 0;

    if ( i < cm_n )
    {
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        /* diagonal entry */
        ++num_cm_entries;

        for ( pj = start_i; pj < end_i; ++pj )
        { 
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                ++num_cm_entries;
            }
        }
    }

    cm_entries[i] = num_cm_entries;
    /* round up to the nearest multiple of warp size to ensure that reads along
     * rows can be coalesced */
    max_cm_entries[i] = max( ((int) CEIL( num_cm_entries * SAFE_ZONE )
                + warpSize - 1) / warpSize * warpSize, MIN_CM_ENTRIES );
}


GPU_GLOBAL void k_estimate_storages_cm_full_opt( real cutoff,
        reax_list far_nbr_list, int cm_n, int cm_n_max,
        int * const cm_entries, int * const max_cm_entries )
{
    extern __shared__ cub::WarpReduce<int>::TempStorage temp4[];
    int i, pj, start_i, end_i, lane_id, num_cm_entries;

    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( i >= cm_n_max )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize; 
    num_cm_entries = 0;

    if ( i < cm_n )
    {
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        for ( pj = start_i + lane_id; pj < end_i; pj += warpSize )
        { 
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                ++num_cm_entries;
            }
        }

        num_cm_entries = cub::WarpReduce<int>(temp4[threadIdx.x / warpSize]).Sum(num_cm_entries);

        /* diagonal entry -- only matters for thread in lane 0 (add once) */
        ++num_cm_entries;
    }

    if ( lane_id == 0 )
    {
        cm_entries[i] = num_cm_entries;
        /* round up to the nearest multiple of warp size to ensure that reads along
         * rows can be coalesced */
        max_cm_entries[i] = max( ((int) CEIL( num_cm_entries * SAFE_ZONE )
                    + warpSize - 1) / warpSize * warpSize, MIN_CM_ENTRIES );
    }
}


GPU_GLOBAL void k_estimate_storage_bonds( reax_atom const * const my_atoms, 
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        real cutoff1, real cutoff2, real bo_cut, reax_list far_nbr_list, 
        int num_atom_types, int n, int N, int total_cap,
        int * const bonds, int * const max_bonds )
{
    int i, j, pj; 
    int start_i, end_i;
    int type_i, type_j, tbp_ij;
    int num_bonds;
    real cutoff, r_ij; 
    real C12, C34, C56;
    real BO_s, BO_pi, BO_pi2;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= total_cap )
    {
        return;
    }

    num_bonds = 0;

    if ( i < N )
    {
        type_i = my_atoms[i].type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );

        if ( i < n )
        {
            cutoff = cutoff1;
        }
        else
        {
            cutoff = cutoff2;
        }

        for ( pj = start_i; pj < end_i; ++pj )
        { 
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                r_ij = far_nbr_list.far_nbr_list.d[pj];
                tbp_ij = index_tbp(type_i ,type_j, num_atom_types);

                /* uncorrected bond orders */
                if ( sbp[type_i].r_s > 0.0 && sbp[type_j].r_s > 0.0 )
                {
                    C12 = tbp[tbp_ij].p_bo1 * POW( r_ij / tbp[tbp_ij].r_s, tbp[tbp_ij].p_bo2 );
                    BO_s = (1.0 + bo_cut) * EXP( C12 );
                }
                else
                {
                    C12 = 0.0;
                    BO_s = 0.0;
                }

                if ( sbp[type_i].r_pi > 0.0 && sbp[type_j].r_pi > 0.0 )
                {
                    C34 = tbp[tbp_ij].p_bo3 * POW( r_ij / tbp[tbp_ij].r_p, tbp[tbp_ij].p_bo4 );
                    BO_pi = EXP( C34 );
                }
                else
                {
                    C34 = 0.0;
                    BO_pi = 0.0;
                }

                if ( sbp[type_i].r_pi_pi > 0.0 && sbp[type_j].r_pi_pi > 0.0 )
                {
                    C56 = tbp[tbp_ij].p_bo5 * POW( r_ij / tbp[tbp_ij].r_pp, tbp[tbp_ij].p_bo6 );
                    BO_pi2= EXP( C56 );
                }
                else
                {
                    C56 = 0.0;
                    BO_pi2 = 0.0;
                }

                /* initially BO values are the uncorrected ones, page 1 */
                if ( BO_s + BO_pi + BO_pi2 >= bo_cut )
                {
                    ++num_bonds;
                }
            }
        }
    }

    bonds[i] = num_bonds;
    /* round up to the nearest multiple of warp size to ensure that reads along
     * rows can be coalesced */
    max_bonds[i] = max( ((int) CEIL(2 * num_bonds * SAFE_ZONE)
                + warpSize - 1) / warpSize * warpSize, MIN_BONDS );
}


GPU_GLOBAL void k_estimate_storage_bonds_opt( reax_atom const * const my_atoms, 
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        real cutoff1, real cutoff2, real bo_cut, reax_list far_nbr_list, 
        int num_atom_types, int n, int N, int total_cap,
        int * const bonds, int * const max_bonds )
{
    extern __shared__ cub::WarpReduce<int>::TempStorage temp5[];
    int i, j, pj, lane_id; 
    int start_i, end_i;
    int type_i, type_j, tbp_ij;
    int num_bonds;
    real cutoff, r_ij; 
    real C12, C34, C56;
    real BO_s, BO_pi, BO_pi2;
    real r_s, r_pi, r_pi_pi;

    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( i >= total_cap )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize;
    num_bonds = 0;

    if ( i < N )
    {
        type_i = my_atoms[i].type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );
        r_s = sbp[type_i].r_s;
        r_pi = sbp[type_i].r_pi;
        r_pi_pi = sbp[type_i].r_pi_pi;

        if ( i < n )
        {
            cutoff = cutoff1;
        }
        else
        {
            cutoff = cutoff2;
        }

        for ( pj = start_i + lane_id; pj < end_i; pj += warpSize )
        { 
            if ( far_nbr_list.far_nbr_list.d[pj] <= cutoff )
            {
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                r_ij = far_nbr_list.far_nbr_list.d[pj];
                tbp_ij = index_tbp(type_i ,type_j, num_atom_types);

                /* uncorrected bond orders */
                if ( r_s > 0.0 && sbp[type_j].r_s > 0.0 )
                {
                    C12 = tbp[tbp_ij].p_bo1 * POW( r_ij / tbp[tbp_ij].r_s, tbp[tbp_ij].p_bo2 );
                    BO_s = (1.0 + bo_cut) * EXP( C12 );
                }
                else
                {
                    C12 = 0.0;
                    BO_s = 0.0;
                }

                if ( r_pi > 0.0 && sbp[type_j].r_pi > 0.0 )
                {
                    C34 = tbp[tbp_ij].p_bo3 * POW( r_ij / tbp[tbp_ij].r_p, tbp[tbp_ij].p_bo4 );
                    BO_pi = EXP( C34 );
                }
                else
                {
                    C34 = 0.0;
                    BO_pi = 0.0;
                }

                if ( r_pi_pi > 0.0 && sbp[type_j].r_pi_pi > 0.0 )
                {
                    C56 = tbp[tbp_ij].p_bo5 * POW( r_ij / tbp[tbp_ij].r_pp, tbp[tbp_ij].p_bo6 );
                    BO_pi2= EXP( C56 );
                }
                else
                {
                    C56 = 0.0;
                    BO_pi2 = 0.0;
                }

                /* initially BO values are the uncorrected ones, page 1 */
                if ( BO_s + BO_pi + BO_pi2 >= bo_cut )
                {
                    ++num_bonds;
                }
            }
        }

        num_bonds = cub::WarpReduce<int>(temp5[threadIdx.x / warpSize]).Sum(num_bonds);
    }

    if ( lane_id == 0 )
    {
        bonds[i] = num_bonds;
        /* round up to the nearest multiple of warp size to ensure that reads along
         * rows can be coalesced */
        max_bonds[i] = max( ((int) CEIL(2 * num_bonds * SAFE_ZONE)
                    + warpSize - 1) / warpSize * warpSize, MIN_BONDS );
    }
}


GPU_GLOBAL void k_estimate_storage_hbonds( reax_atom const * const my_atoms, 
        single_body_parameters const * const sbp, real cutoff1, real cutoff2, real cutoff3,
        reax_list far_nbr_list, int num_atom_types, int n, int N,
        int total_cap, int * const hbonds, int * const max_hbonds )
{
    int i, j, pj; 
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    int num_hbonds;
    real cutoff;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= total_cap )
    {
        return;
    }

    num_hbonds = 0;

    if ( i < N )
    {
        type_i = my_atoms[i].type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );
        ihb = sbp[type_i].p_hbond;

        if ( i < n )
        { 
            cutoff = cutoff1;
        }   
        else
        {
            cutoff = cutoff2;
        } 

        if ( (i < n && ihb == H_ATOM) || ihb == H_BONDING_ATOM )
        {
            for ( pj = start_i; pj < end_i; ++pj )
            { 
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                jhb = sbp[type_j].p_hbond;

                /* atom i: H bonding, ghost
                 * atom j: H atom, native */
                if ( i >= n && j < n && ihb == H_BONDING_ATOM && jhb == H_ATOM
                        && far_nbr_list.far_nbr_list.d[pj] <= cutoff1
                        && far_nbr_list.far_nbr_list.d[pj] <= cutoff3 )
                {
                    ++num_hbonds;
                }
                else if ( i < n && far_nbr_list.far_nbr_list.d[pj] <= cutoff
                        && far_nbr_list.far_nbr_list.d[pj] <= cutoff3 )
                {
                    /* atom i: H atom, native
                     * atom j: H bonding atom */
                    if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                    {
                        ++num_hbonds;
                    }
                    /* atom i: H bonding atom, native
                     * atom j: H atom, native */
                    else if ( ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                    {
                        ++num_hbonds;
                    }
                }
            }
        }
    }

    hbonds[i] = num_hbonds;
    /* round up to the nearest multiple of warp size to ensure that reads along
     * rows can be coalesced */
    max_hbonds[i] = max( ((int) CEIL(num_hbonds * SAFE_ZONE)
                + warpSize - 1) / warpSize * warpSize, MIN_HBONDS );
}


GPU_GLOBAL void k_estimate_storage_hbonds_opt( reax_atom const * const my_atoms, 
        single_body_parameters const * const sbp, real cutoff1, real cutoff2, real cutoff3,
        reax_list far_nbr_list, int num_atom_types, int n, int N,
        int total_cap, int * const hbonds, int * const max_hbonds )
{
    extern __shared__ cub::WarpReduce<int>::TempStorage temp6[];
    int i, j, pj, lane_id;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    int num_hbonds;
    real cutoff;

    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( i >= total_cap )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize; 
    num_hbonds = 0;

    if ( i < N )
    {
        type_i = my_atoms[i].type;
        start_i = Start_Index( i, &far_nbr_list );
        end_i = End_Index( i, &far_nbr_list );
        ihb = sbp[type_i].p_hbond;

        if ( i < n )
        { 
            cutoff = cutoff1;
        }   
        else
        {
            cutoff = cutoff2;
        } 

        if ( (i < n && ihb == H_ATOM) || ihb == H_BONDING_ATOM )
        {
            for ( pj = start_i + lane_id; pj < end_i; pj += warpSize )
            { 
                j = far_nbr_list.far_nbr_list.nbr[pj];
                type_j = my_atoms[j].type;
                jhb = sbp[type_j].p_hbond;

                /* atom i: H bonding, ghost
                 * atom j: H atom, native */
                if ( i >= n && j < n && ihb == H_BONDING_ATOM && jhb == H_ATOM
                        && far_nbr_list.far_nbr_list.d[pj] <= cutoff1
                        && far_nbr_list.far_nbr_list.d[pj] <= cutoff3 )
                {
                    ++num_hbonds;
                }
                else if ( i < n && far_nbr_list.far_nbr_list.d[pj] <= cutoff
                        && far_nbr_list.far_nbr_list.d[pj] <= cutoff3 )
                {
                    /* atom i: H atom, native
                     * atom j: H bonding atom */
                    if ( ihb == H_ATOM && jhb == H_BONDING_ATOM )
                    {
                        ++num_hbonds;
                    }
                    /* atom i: H bonding atom, native
                     * atom j: H atom, native */
                    else if ( ihb == H_BONDING_ATOM && jhb == H_ATOM && j < n )
                    {
                        ++num_hbonds;
                    }
                }
            }

            num_hbonds = cub::WarpReduce<int>(temp6[threadIdx.x / warpSize]).Sum(num_hbonds);
        }
    }

    if ( lane_id == 0 )
    {
        hbonds[i] = num_hbonds;
        /* round up to the nearest multiple of warp size to ensure that reads along
         * rows can be coalesced */
        max_hbonds[i] = max( ((int) CEIL(num_hbonds * SAFE_ZONE)
                    + warpSize - 1) / warpSize * warpSize, MIN_HBONDS );
    }
}


GPU_GLOBAL void k_update_sym_dbond_indices( reax_list bond_list, int N )
{
    int i, pj, pk, nbr_ij, nbr_jk;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    /* i-j bonds */
    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        nbr_ij = BL.nbr[pj];

        /* j-k bonds */
        for ( pk = Start_Index(nbr_ij, &bond_list); pk < End_Index(nbr_ij, &bond_list); ++pk )
        {
            nbr_jk = BL.nbr[pk];

            if ( i == nbr_jk && i > nbr_ij )
            {
                BL.sym_index[pj] = pk;
                BL.sym_index[pk] = pj;
                break;
            }
        }
    }

#undef BL
}


GPU_GLOBAL void k_update_sym_dbond_indices_opt( reax_list bond_list, int N )
{
    int i, pj, pk, start_i, end_i, nbr_ij, nbr_jk, flag, lane_id;
#define BL (bond_list.bond_list_gpu)

    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    if ( i >= N )
    {
        return;
    }

    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize; 
    start_i = Start_Index(i, &bond_list);
    end_i = End_Index(i, &bond_list);

    /* i-j bonds */
    for ( pj = start_i + lane_id; pj < end_i; pj += warpSize )
    {
        nbr_ij = BL.nbr[pj];
        flag = FALSE;

        /* j-k bonds */
        for ( pk = Start_Index(nbr_ij, &bond_list); pk < End_Index(nbr_ij, &bond_list); ++pk )
        {
            nbr_jk = BL.nbr[pk];

            if ( i == nbr_jk && i > nbr_ij )
            {
                flag = TRUE;
                break;
            }
        }

        if ( flag == TRUE )
        {
            BL.sym_index[pj] = pk;
            BL.sym_index[pk] = pj;
        }
    }

#undef BL
}


#if !defined(GPU_KERNEL_ATOMIC)
//GPU_GLOBAL void k_update_sym_hbond_indices_opt( reax_atom const * const my_atoms,
//        reax_list hbond_list, int N )
//{
//    int i, pj, pk, nbr_ij;
//    int start_i, end_i, flag, lane_id;
//
//    i = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
//
//    if ( i >= N )
//    {
//        return;
//    }
//
//    lane_id = (blockIdx.x * blockDim.x + threadIdx.x) % warpSize; 
//    start_i = Start_Index( my_atoms[i].Hindex, &hbond_list );
//    end_i = End_Index( my_atoms[i].Hindex, &hbond_list );
//
//    /* i-j H-bonds */
//    for ( pj = start_i + lane_id; pj < end_i; pj += warpSize )
//    {
//        nbr_ij = hbond_list.hbond_list.nbr[pj];
//        flag = FALSE;
//
//        /* j-k H-bonds */
//        for ( pk = Start_Index( my_atoms[nbr_ij].Hindex, &hbond_list );
//                pk < End_Index( my_atoms[nbr_ij].Hindex, &hbond_list ); ++pk )
//        {
//            if ( i == hbond_list.hbond_list.nbr[pk] )
//            {
//                flag = TRUE;
//                break;
//            }
//        }
//
//        if ( flag == TRUE )
//        {
//            hbond_list.hbond_list.sym_index[pj] = pk;
//            hbond_list.hbond_list.sym_index[pk] = pj;
//        }
//    }
//}
#endif


#if defined(DEBUG_FOCUS)
GPU_GLOBAL void k_print_forces( reax_atom const * const my_atoms,
        rvec const * const f, int n )
{
    int i; 

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    printf( "%8d: %24.15f, %24.15f, %24.15f\n",
            my_atoms[i].orig_id, f[i][0], f[i][1], f[i][2] );
}


GPU_GLOBAL void k_print_hbonds( reax_atom const * const my_atoms,
        reax_list hbond_list, int n, int rank, int step )
{
    int i, k, pj, start, end; 
    hbond_data *hbond_jk;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    start = Start_Index( my_atoms[i].Hindex, &hbond_list );
    end = End_Index( my_atoms[i].Hindex, &hbond_list );

    for ( pj = start; pj < end; ++pj )
    {
        k = hbond_list.hbond_list[pj].nbr;
        hbond_jk = &hbond_list.hbond_list[pj];

#if !defined(GPU_KERNEL_ATOMIC)
        printf( "p%03d, step %05d: %8d: %8d, %24.15f, %24.15f, %24.15f\n",
                rank, step, my_atoms[i].Hindex, k,
                hbond_jk->f_hb[0],
                hbond_jk->f_hb[1],
                hbond_jk->f_hb[2] );
#else
        printf( "p%03d, step %05d: %8d: %8d\n",
                rank, step, my_atoms[i].Hindex, k );
#endif
    }
}
#endif


#if defined(DEBUG_FOCUS)
static void Print_Forces( reax_system const * const system,
        control_params const * const control )
{
    k_print_forces <<< control->blocks_n, control->gpu_block_size,
                   0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, workspace->d_workspace->f, system->n );
    cudaCheckError( );
}


static void Print_HBonds( reax_system const * const system,
        control_params const * const control, int step )
{
    k_print_hbonds <<< control->blocks_n, control->gpu_block_size,
                   0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, *(lists[HBONDS]), system->n, system->my_rank, step );
    cudaCheckError( );
}
#endif


/* Initialize indices for far neighbors list post reallocation
 *
 * system: atomic system info. */
void GPU_Init_Neighbor_Indices( reax_system * const system,
        control_params const * const control, reax_list * const far_nbr_list )
{
    int blocks;

    blocks = far_nbr_list->n / control->gpu_block_size
        + (far_nbr_list->n % control->gpu_block_size == 0 ? 0 : 1);

    /* init indices */
    GPU_Scan_Excl_Sum( system->d_max_far_nbrs, far_nbr_list->index,
            far_nbr_list->n, 0, control->gpu_streams[0] );

    /* init end_indices */
    k_init_end_index <<< blocks, control->gpu_block_size,
                     0, control->gpu_streams[0] >>>
        ( system->d_far_nbrs, far_nbr_list->index, far_nbr_list->end_index,
          far_nbr_list->n );
    cudaCheckError( );
}


/* Initialize indices for far hydrogen bonds list post reallocation
 *
 * system: atomic system info. */
void GPU_Init_HBond_Indices( reax_system * const system, storage * const workspace,
        reax_list * const hbond_list, int block_size, cudaStream_t s )
{
    int blocks, *temp;

    blocks = hbond_list->n / block_size
        + (hbond_list->n % block_size == 0 ? 0 : 1);

    sCudaCheckMalloc( &workspace->d_workspace->scratch[2],
            &workspace->d_workspace->scratch_size[2],
            sizeof(int) * hbond_list->n, __FILE__, __LINE__ );
    temp = (int *) workspace->d_workspace->scratch[2];

    /* init indices and end_indices */
    GPU_Scan_Excl_Sum( system->d_max_hbonds, temp, hbond_list->n, 2, s );

    k_init_hbond_indices <<< blocks, block_size, 0, s >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, system->d_hbonds, temp, 
          hbond_list->index, hbond_list->end_index, system->N, hbond_list->n );
    cudaCheckError( );
}


/* Initialize indices for far bonds list post reallocation
 *
 * system: atomic system info. */
void GPU_Init_Bond_Indices( reax_system * const system, reax_list * const bond_list,
        int block_size, cudaStream_t s )
{
    int blocks;

    blocks = system->total_cap / block_size
        + (system->total_cap % block_size == 0 ? 0 : 1);

    /* init indices */
    GPU_Scan_Excl_Sum( system->d_max_bonds, bond_list->index,
            system->total_cap, 1, s );

    /* init end_indices */
    k_init_end_index <<< blocks, block_size, 0, s >>>
        ( system->d_bonds, bond_list->index, bond_list->end_index, system->total_cap );
    cudaCheckError( );
}


/* Initialize indices for charge matrix post reallocation
 *
 * system: atomic system info.
 * H: charge matrix */
void GPU_Init_Sparse_Matrix_Indices( reax_system * const system, sparse_matrix * const H,
        int block_size, cudaStream_t s )
{
    int blocks;

    blocks = H->n_max / block_size
        + (H->n_max % block_size == 0 ? 0 : 1);

    /* init start indices */
    GPU_Scan_Excl_Sum( system->d_max_cm_entries, H->start, H->n_max, 5, s );

    if ( H->format == SYM_HALF_MATRIX )
    {
        /* init end_indices */
        k_init_end_index <<< blocks, block_size, 0, s >>>
            ( system->d_cm_entries, H->start, H->end, H->n_max );
        cudaCheckError( );
    }
}


void GPU_Estimate_Storages( reax_system * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        int realloc_cm, int realloc_bonds, int realloc_hbonds, int step )
{
    int blocks;
#if defined(LOG_PERFORMANCE)
    float time_elapsed;
#endif

    if ( realloc_cm == TRUE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_CM_START], control->gpu_streams[5] );
#endif

//        blocks = workspace->d_workspace->H.n_max / control->gpu_block_size
//            + (workspace->d_workspace->H.n_max % control->gpu_block_size == 0 ? 0 : 1);
        blocks = workspace->d_workspace->H.n_max * WARP_SIZE / control->gpu_block_size
            + (workspace->d_workspace->H.n_max * WARP_SIZE % control->gpu_block_size == 0 ? 0 : 1);

        if ( workspace->d_workspace->H.format == SYM_HALF_MATRIX )
        {
            k_estimate_storages_cm_half <<< blocks, control->gpu_block_size, 0,
                                        control->gpu_streams[5] >>>
                ( system->d_my_atoms, control->nonb_cut,
                  *(lists[FAR_NBRS]), workspace->d_workspace->H.n,
                  workspace->d_workspace->H.n_max,
                  system->d_cm_entries, system->d_max_cm_entries );
        }
        else
        {
//            k_estimate_storages_cm_full <<< blocks, control->gpu_block_size, 0,
//                                        control->gpu_streams[5] >>>
//                ( control->nonb_cut, *(lists[FAR_NBRS]), workspace->d_workspace->H.n,
//                  workspace->d_workspace->H.n_max,
//                  system->d_cm_entries, system->d_max_cm_entries );

            k_estimate_storages_cm_full_opt <<< blocks, control->gpu_block_size,
                                            sizeof(cub::WarpReduce<int>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                                            control->gpu_streams[5] >>>
                ( control->nonb_cut, *(lists[FAR_NBRS]), workspace->d_workspace->H.n,
                  workspace->d_workspace->H.n_max,
                  system->d_cm_entries, system->d_max_cm_entries );
        }
        cudaCheckError( );

        GPU_Reduction_Sum( system->d_max_cm_entries, system->d_total_cm_entries,
                workspace->d_workspace->H.n_max, 5, control->gpu_streams[5] );
        sCudaMemcpyAsync( &system->total_cm_entries, system->d_total_cm_entries,
                sizeof(int), cudaMemcpyDeviceToHost, control->gpu_streams[5], __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_CM_STOP], control->gpu_streams[5] );
#endif
    }

    if ( realloc_bonds == TRUE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_START], control->gpu_streams[1] );
#endif

//        blocks = system->total_cap / control->gpu_block_size
//            + (system->total_cap % control->gpu_block_size == 0 ? 0 : 1);
        blocks = system->total_cap * WARP_SIZE / control->gpu_block_size
            + (system->total_cap * WARP_SIZE % control->gpu_block_size == 0 ? 0 : 1);

//        k_estimate_storage_bonds <<< blocks, control->gpu_block_size, 0,
//                                 control->gpu_streams[1] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp,
//              MIN( control->nonb_cut, control->bond_cut ), control->bond_cut, control->bo_cut,
//              *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
//              system->n, system->N, system->total_cap,
//              system->d_bonds, system->d_max_bonds );
        k_estimate_storage_bonds_opt <<< blocks, control->gpu_block_size,
                                     sizeof(cub::WarpReduce<int>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                                     control->gpu_streams[1] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp, 
              MIN( control->nonb_cut, control->bond_cut ), control->bond_cut, control->bo_cut,
              *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
              system->n, system->N, system->total_cap,
              system->d_bonds, system->d_max_bonds );
        cudaCheckError( );

        GPU_Reduction_Sum( system->d_max_bonds, system->d_total_bonds,
                system->total_cap, 1, control->gpu_streams[1] );
        sCudaMemcpyAsync( &system->total_bonds, system->d_total_bonds, sizeof(int), 
                cudaMemcpyDeviceToHost, control->gpu_streams[1], __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_STOP], control->gpu_streams[1] );
#endif
    }

    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 && realloc_hbonds == TRUE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_START], control->gpu_streams[2] );
#endif

//        blocks = system->total_cap / control->gpu_block_size
//            + (system->total_cap % control->gpu_block_size == 0 ? 0 : 1);
        blocks = system->total_cap * WARP_SIZE / control->gpu_block_size
            + (system->total_cap * WARP_SIZE % control->gpu_block_size == 0 ? 0 : 1);

//        k_estimate_storage_hbonds <<< blocks, control->gpu_block_size, 0,
//                                  control->gpu_streams[2] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              control->nonb_cut, control->bond_cut, control->hbond_cut,
//              *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
//              system->n, system->N, system->total_cap,
//              system->d_hbonds, system->d_max_hbonds );
        k_estimate_storage_hbonds_opt <<< blocks, control->gpu_block_size,
                                      sizeof(cub::WarpReduce<int>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                                      control->gpu_streams[2] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              control->nonb_cut, control->bond_cut, control->hbond_cut,
              *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
              system->n, system->N, system->total_cap,
              system->d_hbonds, system->d_max_hbonds );
        cudaCheckError( );

        GPU_Reduction_Sum( system->d_max_hbonds, system->d_total_hbonds,
                system->total_cap, 2, control->gpu_streams[2] );
        sCudaMemcpyAsync( &system->total_hbonds, system->d_total_hbonds, sizeof(int), 
                cudaMemcpyDeviceToHost, control->gpu_streams[2], __FILE__, __LINE__ );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_STOP], control->gpu_streams[2] );
#endif
    }

    if ( realloc_cm == TRUE )
    {
        cudaStreamSynchronize( control->gpu_streams[5] );

#if defined(LOG_PERFORMANCE)
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_CM_START],
                control->gpu_time_events[TE_INIT_CM_STOP] ); 
        data->timing.init_cm += (real) (time_elapsed / 1000.0);
#endif
    }
    if ( realloc_bonds == TRUE )
    {
        cudaStreamSynchronize( control->gpu_streams[1] );

#if defined(LOG_PERFORMANCE)
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_BOND_START],
                control->gpu_time_events[TE_INIT_BOND_STOP] ); 
        data->timing.init_bond += (real) (time_elapsed / 1000.0);
#endif
    }
    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 && realloc_hbonds == TRUE )
    {
        cudaStreamSynchronize( control->gpu_streams[2] );

#if defined(LOG_PERFORMANCE)
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_HBOND_START],
                control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
        data->timing.init_hbond += (real) (time_elapsed / 1000.0);
#endif
    }
}


/* Initialize the bond list, hydrogen bond list, and charge matrix
 * data structures along with updating the pairwise distances in the
 * far neighbor (Verlet) list if required.
 *
 * NOTE: the control flow of the code follows a
 * try-compute-else-reallocate-and-retry logic which requires that
 * the initialization kernels be atomic transactions. Locks are used
 * to mark if the transaction succeeds (and thus should not be repeated).
 */
int GPU_Init_Forces( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control ) 
{
    int renbr, blocks, ret;
    static int dist_done = FALSE, cm_done = FALSE, bonds_done = FALSE, hbonds_done = FALSE;
#if defined(LOG_PERFORMANCE)
    float time_elapsed;
#endif

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* reset reallocation flags on device */
    if ( cm_done == FALSE )
    {
        sCudaMemsetAsync( &workspace->d_workspace->realloc[RE_CM], FALSE, sizeof(int), 
                control->gpu_streams[5], __FILE__, __LINE__ );
    }
    if ( bonds_done == FALSE )
    {
        sCudaMemsetAsync( &workspace->d_workspace->realloc[RE_BONDS], FALSE, sizeof(int), 
                control->gpu_streams[1], __FILE__, __LINE__ );
    }
    if ( hbonds_done == FALSE )
    {
        sCudaMemsetAsync( &workspace->d_workspace->realloc[RE_HBONDS], FALSE, sizeof(int), 
                control->gpu_streams[2], __FILE__, __LINE__ );
    }

    if ( renbr == FALSE && dist_done == FALSE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_DIST_START], control->gpu_streams[0] );
#endif

//        k_init_dist <<< control->blocks_N, control->gpu_block_size,
//                    0, control->gpu_streams[0] >>>
//            ( system->d_my_atoms, *(lists[FAR_NBRS]), system->N );

        k_init_dist_opt <<< control->blocks_warp_N, control->gpu_block_size,
                        0, control->gpu_streams[0] >>>
            ( system->d_my_atoms, *(lists[FAR_NBRS]), system->N );
        cudaCheckError( );

        cudaEventRecord( control->gpu_stream_events[SE_INIT_DIST_DONE], control->gpu_streams[0] );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_DIST_STOP], control->gpu_streams[0] );
#endif

        dist_done = TRUE;
    }

    if ( cm_done == FALSE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_CM_START], control->gpu_streams[5] );
#endif

        blocks = workspace->d_workspace->H.n_max / control->gpu_block_size
            + (workspace->d_workspace->H.n_max % control->gpu_block_size == 0 ? 0 : 1);

        /* update num. rows in matrix for this GPU */
        workspace->d_workspace->H.n = system->n;

        GPU_Init_Sparse_Matrix_Indices( system, &workspace->d_workspace->H,
                control->gpu_block_size, control->gpu_streams[5] );

        if ( renbr == FALSE )
        {
            cudaStreamWaitEvent( control->gpu_streams[5], control->gpu_stream_events[SE_INIT_DIST_DONE], 0 );
        }

        if ( workspace->d_workspace->H.format == SYM_HALF_MATRIX && control->tabulate <= 0 )
        {
            k_init_cm_qeq_half_fs <<< blocks, control->gpu_block_size,
                                  0, control->gpu_streams[5] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp,
                  workspace->d_workspace->H, workspace->d_workspace->tap_coef,
                  control->nonb_cut, *(lists[FAR_NBRS]),
                  system->reax_param.num_atom_types, system->d_max_cm_entries,
                  &workspace->d_workspace->realloc[RE_CM], workspace->d_workspace->H.n_max );
        }
        else if ( workspace->d_workspace->H.format == SYM_HALF_MATRIX && control->tabulate > 0 )
        {
            k_init_cm_qeq_half_fs_tab <<< blocks, control->gpu_block_size,
                                      0, control->gpu_streams[5] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp,
                  workspace->d_workspace->H, control->nonb_cut, *(lists[FAR_NBRS]),
                  workspace->d_workspace->LR, system->reax_param.num_atom_types,
                  system->d_max_cm_entries, &workspace->d_workspace->realloc[RE_CM],
                  workspace->d_workspace->H.n_max );
        }
        else if ( workspace->d_workspace->H.format == SYM_FULL_MATRIX && control->tabulate <= 0 )
        {
//            k_init_cm_qeq_full_fs <<< blocks, control->gpu_block_size, 0, control->gpu_streams[5] >>>
//                ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp,
//                  workspace->d_workspace->H, workspace->d_workspace->tap_coef, control->nonb_cut,
//                  *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
//                  system->d_max_cm_entries, &workspace->d_workspace->realloc[RE_CM],
//                  workspace->d_workspace->H.n_max );

            blocks = workspace->d_workspace->H.n_max * WARP_SIZE / control->gpu_block_size
                + (workspace->d_workspace->H.n_max * WARP_SIZE % control->gpu_block_size == 0 ? 0 : 1);

            k_init_cm_qeq_full_fs_opt <<< blocks, control->gpu_block_size,
                                      sizeof(cub::WarpScan<int>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                                      control->gpu_streams[5] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_tbp,
                  workspace->d_workspace->H, workspace->d_workspace->tap_coef, control->nonb_cut,
                  *(lists[FAR_NBRS]), system->reax_param.num_atom_types,
                  system->d_max_cm_entries, &workspace->d_workspace->realloc[RE_CM],
                  workspace->d_workspace->H.n_max );
        }
        else if ( workspace->d_workspace->H.format == SYM_FULL_MATRIX && control->tabulate > 0 )
        {
            k_init_cm_qeq_full_fs_tab <<< blocks, control->gpu_block_size, 0,
                                      control->gpu_streams[5] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp, workspace->d_workspace->H,
                  control->nonb_cut, *(lists[FAR_NBRS]), workspace->d_workspace->LR,
                  system->reax_param.num_atom_types, system->d_max_cm_entries,
                  &workspace->d_workspace->realloc[RE_CM], workspace->d_workspace->H.n_max );
        }
        cudaCheckError( );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_CM_STOP], control->gpu_streams[5] );
#endif
    }

    if ( bonds_done == FALSE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_START], control->gpu_streams[1] );
#endif

        blocks = system->total_cap / control->gpu_block_size
            + ((system->total_cap % control->gpu_block_size == 0 ) ? 0 : 1);

        GPU_Init_Bond_Indices( system, lists[BONDS], control->gpu_block_size,
                control->gpu_streams[1] );

        if ( renbr == FALSE )
        {
            cudaStreamWaitEvent( control->gpu_streams[1], control->gpu_stream_events[SE_INIT_DIST_DONE], 0 );
        }

//        k_init_bonds <<< control->blocks_N, control->gpu_block_size, 0, control->gpu_streams[1] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              system->reax_param.d_tbp, workspace->d_workspace->total_bond_order,
//              workspace->d_workspace->dDeltap_self, MIN( control->nonb_cut, control->bond_cut ),
//              control->bond_cut, control->bo_cut, *(lists[FAR_NBRS]), *(lists[BONDS]),
//              system->n, system->N, system->reax_param.num_atom_types,
//              system->d_max_bonds, &workspace->d_workspace->realloc[RE_BONDS] );
//        cudaCheckError( );

        k_init_bonds_opt <<< control->blocks_warp_N, control->gpu_block_size,
                     (sizeof(cub::WarpScan<int>::TempStorage)
                      + sizeof(cub::WarpReduce<double>::TempStorage)) * (control->gpu_block_size / WARP_SIZE),
                     control->gpu_streams[1] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              system->reax_param.d_tbp, workspace->d_workspace->total_bond_order,
              workspace->d_workspace->dDeltap_self, MIN( control->nonb_cut, control->bond_cut ),
              control->bond_cut, control->bo_cut, *(lists[FAR_NBRS]), *(lists[BONDS]),
              system->n, system->N, system->reax_param.num_atom_types,
              system->d_max_bonds, &workspace->d_workspace->realloc[RE_BONDS] );
        cudaCheckError( );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_STOP], control->gpu_streams[1] );
#endif
    }

    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 && hbonds_done == FALSE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_START], control->gpu_streams[2] );
#endif

        GPU_Init_HBond_Indices( system, workspace, lists[HBONDS],
                control->gpu_block_size, control->gpu_streams[2] );

        if ( renbr == FALSE )
        {
            cudaStreamWaitEvent( control->gpu_streams[2], control->gpu_stream_events[SE_INIT_DIST_DONE], 0 );
        }

//        k_init_hbonds <<< control->blocks_N, control->gpu_block_size, 0, control->gpu_streams[2] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              MIN( control->nonb_cut, control->hbond_cut ), *(lists[FAR_NBRS]), *(lists[HBONDS]),
//              system->n, system->N, system->reax_param.num_atom_types,
//              system->d_max_hbonds, &workspace->d_workspace->realloc[RE_HBONDS] );
//        cudaCheckError( );

        k_init_hbonds_opt <<< control->blocks_warp_N, control->gpu_block_size,
                          sizeof(cub::WarpScan<int>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                          control->gpu_streams[2] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              MIN( control->nonb_cut, control->hbond_cut ), *(lists[FAR_NBRS]), *(lists[HBONDS]),
              system->n, system->N, system->reax_param.num_atom_types,
              system->d_max_hbonds, &workspace->d_workspace->realloc[RE_HBONDS] );
        cudaCheckError( );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_STOP], control->gpu_streams[2] );
#endif
    }

    /* check reallocation flags on device */
    if ( cm_done == FALSE )
    {
        sCudaMemcpyAsync( &workspace->realloc[RE_CM],
                &workspace->d_workspace->realloc[RE_CM], sizeof(int), 
                cudaMemcpyDeviceToHost, control->gpu_streams[5], __FILE__, __LINE__ );
    }
    else
    {
        workspace->realloc[RE_CM] = FALSE;
    }
    if ( bonds_done == FALSE )
    {
        sCudaMemcpyAsync( &workspace->realloc[RE_BONDS],
                &workspace->d_workspace->realloc[RE_BONDS], sizeof(int), 
                cudaMemcpyDeviceToHost, control->gpu_streams[1], __FILE__, __LINE__ );
    }
    else
    {
        workspace->realloc[RE_BONDS] = FALSE;
    }
    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 && hbonds_done == FALSE )
    {
        sCudaMemcpyAsync( &workspace->realloc[RE_HBONDS],
                &workspace->d_workspace->realloc[RE_HBONDS], sizeof(int), 
                cudaMemcpyDeviceToHost, control->gpu_streams[2], __FILE__, __LINE__ );
    }
    else
    {
        workspace->realloc[RE_HBONDS] = FALSE;
    }

    cudaStreamSynchronize( control->gpu_streams[0] );
    cudaStreamSynchronize( control->gpu_streams[5] );
    cudaStreamSynchronize( control->gpu_streams[1] );
    cudaStreamSynchronize( control->gpu_streams[2] );

    ret = (workspace->realloc[RE_CM] == FALSE
            && workspace->realloc[RE_BONDS] == FALSE
            && workspace->realloc[RE_HBONDS] == FALSE ? SUCCESS : FAILURE);

    if ( workspace->realloc[RE_CM] == FALSE )
    {
        cm_done = TRUE;
    }
#if defined(LOG_PERFORMANCE)
    else
    {
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_CM_START],
                control->gpu_time_events[TE_INIT_CM_STOP] ); 
        data->timing.init_cm += (real) (time_elapsed / 1000.0);
    }
#endif
    if ( workspace->realloc[RE_BONDS] == FALSE )
    {
        bonds_done = TRUE;
    }
#if defined(LOG_PERFORMANCE)
    else
    {
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_BOND_START],
                control->gpu_time_events[TE_INIT_BOND_STOP] ); 
        data->timing.init_bond += (real) (time_elapsed / 1000.0);
    }
#endif
    if ( workspace->realloc[RE_HBONDS] == FALSE )
    {
        hbonds_done = TRUE;
    }
#if defined(LOG_PERFORMANCE)
    else if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
    {
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_HBOND_START],
                control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
        data->timing.init_hbond += (real) (time_elapsed / 1000.0);
    }
#endif

    if ( ret == SUCCESS )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_BOND_START],
                control->gpu_time_events[TE_INIT_BOND_STOP] ); 
        data->timing.init_bond += (real) (time_elapsed / 1000.0);

        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_START], control->gpu_streams[1] );
#endif

//        k_update_sym_dbond_indices <<< control->blocks_N, control->gpu_block_size,
//                                   0, control->gpu_streams[1] >>> 
//            ( *(lists[BONDS]), system->N );
        k_update_sym_dbond_indices_opt <<< control->blocks_warp_N, control->gpu_block_size,
                                       0, control->gpu_streams[1] >>>
            ( *(lists[BONDS]), system->N );
        cudaCheckError( );

        cudaEventRecord( control->gpu_stream_events[SE_INIT_BOND_DONE], control->gpu_streams[1] );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_STOP], control->gpu_streams[1] );
#endif

#if !defined(GPU_KERNEL_ATOMIC)
//        if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
//        {
//#if defined(LOG_PERFORMANCE)
//            cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_HBOND_START],
//                    control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
//            data->timing.init_hbond += (real) (time_elapsed / 1000.0);
//
//            cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_START], control->gpu_streams[2] );
//#endif
//
//            /* make hbond_list symmetric */
//            k_update_sym_hbond_indices_opt <<< control->blocks_warp_N, control->gpu_block_size,
//                                           0, control->gpu_streams[2] >>>
//                ( system->d_my_atoms, *(lists[HBONDS]), system->N );
//            cudaCheckError( );
//
//#if defined(LOG_PERFORMANCE)
//            cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_STOP], control->gpu_streams[2] );
//#endif
//        }
#endif

        dist_done = FALSE;
        cm_done = FALSE;
        bonds_done = FALSE;
        hbonds_done = FALSE;
    }
    else
    {
        GPU_Estimate_Storages( system, control, data, workspace, lists,
               workspace->realloc[RE_CM], workspace->realloc[RE_BONDS],
               workspace->realloc[RE_HBONDS], data->step - data->prev_steps );
    }

    return ret;
}


int GPU_Init_Forces_No_Charges( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control ) 
{
    int renbr, ret;
    static int dist_done = FALSE, bonds_done = FALSE, hbonds_done = FALSE;
#if defined(LOG_PERFORMANCE)
    float time_elapsed;
#endif

    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* reset reallocation flags on device */
    if ( bonds_done == FALSE )
    {
        sCudaMemsetAsync( &workspace->d_workspace->realloc[RE_BONDS], FALSE, sizeof(int), 
                control->gpu_streams[1], __FILE__, __LINE__ );
    }
    if ( hbonds_done == FALSE )
    {
        sCudaMemsetAsync( &workspace->d_workspace->realloc[RE_HBONDS], FALSE, sizeof(int), 
                control->gpu_streams[2], __FILE__, __LINE__ );
    }

    if ( renbr == FALSE && dist_done == FALSE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_DIST_START], control->gpu_streams[0] );
#endif

//        k_init_dist <<< control->blocks_N, control->gpu_block_size,
//                    0, control->gpu_streams[0] >>>
//            ( system->d_my_atoms, *(lists[FAR_NBRS]), system->N );

        k_init_dist_opt <<< control->blocks_warp_N, control->gpu_block_size,
                        0, control->gpu_streams[0] >>>
            ( system->d_my_atoms, *(lists[FAR_NBRS]), system->N );
        cudaCheckError( );

        cudaEventRecord( control->gpu_stream_events[SE_INIT_DIST_DONE], control->gpu_streams[0] );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_DIST_STOP], control->gpu_streams[0] );
#endif

        dist_done = TRUE;
    }

    if ( bonds_done == FALSE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_START], control->gpu_streams[1] );
#endif

        GPU_Init_Bond_Indices( system, lists[BONDS], control->gpu_block_size,
                control->gpu_streams[1] );

        if ( renbr == FALSE )
        {
            cudaStreamWaitEvent( control->gpu_streams[1], control->gpu_stream_events[SE_INIT_DIST_DONE], 0 );
        }

//        k_init_bonds <<< control->blocks_N, control->gpu_block_size, 0, control->gpu_streams[1] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              system->reax_param.d_tbp, workspace->d_workspace->total_bond_order,
//              workspace->d_workspace->dDeltap_self, MIN( control->nonb_cut, control->bond_cut ),
//              control->bond_cut, control->bo_cut, *(lists[FAR_NBRS]), *(lists[BONDS]),
//              system->n, system->N, system->reax_param.num_atom_types,
//              system->d_max_bonds, &workspace->d_workspace->realloc[RE_BONDS] );

        k_init_bonds_opt <<< control->blocks_warp_N, control->gpu_block_size,
                     sizeof(cub::WarpScan<int>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                     control->gpu_streams[1] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              system->reax_param.d_tbp, workspace->d_workspace->total_bond_order,
              workspace->d_workspace->dDeltap_self, MIN( control->nonb_cut, control->bond_cut ),
              control->bond_cut, control->bo_cut, *(lists[FAR_NBRS]), *(lists[BONDS]),
              system->n, system->N, system->reax_param.num_atom_types,
              system->d_max_bonds, &workspace->d_workspace->realloc[RE_BONDS] );
        cudaCheckError( );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_STOP], control->gpu_streams[1] );
#endif
    }

    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 && hbonds_done == FALSE )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_START], control->gpu_streams[2] );
#endif

        GPU_Init_HBond_Indices( system, workspace, lists[HBONDS],
                control->gpu_block_size, control->gpu_streams[2] );

        if ( renbr == FALSE )
        {
            cudaStreamWaitEvent( control->gpu_streams[2], control->gpu_stream_events[SE_INIT_DIST_DONE], 0 );
        }

//        k_init_hbonds <<< control->blocks_N, control->gpu_block_size, 0, control->gpu_streams[2] >>>
//            ( system->d_my_atoms, system->reax_param.d_sbp,
//              MIN( control->nonb_cut, control->hbond_cut ), *(lists[FAR_NBRS]), *(lists[HBONDS]),
//              system->n, system->N, system->reax_param.num_atom_types,
//              system->d_max_hbonds, &workspace->d_workspace->realloc[RE_HBONDS] );
//        cudaCheckError( );

        k_init_hbonds_opt <<< control->blocks_warp_N, control->gpu_block_size,
                          sizeof(cub::WarpScan<int>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                          control->gpu_streams[2] >>>
            ( system->d_my_atoms, system->reax_param.d_sbp,
              MIN( control->nonb_cut, control->hbond_cut ), *(lists[FAR_NBRS]), *(lists[HBONDS]),
              system->n, system->N, system->reax_param.num_atom_types,
              system->d_max_hbonds, &workspace->d_workspace->realloc[RE_HBONDS] );
        cudaCheckError( );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_STOP], control->gpu_streams[2] );
#endif
    }

    /* check reallocation flags on device */
    if ( bonds_done == FALSE )
    {
        sCudaMemcpyAsync( &workspace->realloc[RE_BONDS],
                &workspace->d_workspace->realloc[RE_BONDS], sizeof(int), 
                cudaMemcpyDeviceToHost, control->gpu_streams[1], __FILE__, __LINE__ );
    }
    else
    {
        workspace->realloc[RE_BONDS] = FALSE;
    }
    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 && hbonds_done == FALSE )
    {
        sCudaMemcpyAsync( &workspace->realloc[RE_HBONDS],
                &workspace->d_workspace->realloc[RE_HBONDS], sizeof(int), 
                cudaMemcpyDeviceToHost, control->gpu_streams[2], __FILE__, __LINE__ );
    }
    else
    {
        workspace->realloc[RE_HBONDS] = FALSE;
    }

    cudaStreamSynchronize( control->gpu_streams[0] );
    cudaStreamSynchronize( control->gpu_streams[1] );
    cudaStreamSynchronize( control->gpu_streams[2] );

    ret = (workspace->realloc[RE_BONDS] == FALSE
            && workspace->realloc[RE_HBONDS] == FALSE ? SUCCESS : FAILURE);

    if ( workspace->realloc[RE_BONDS] == FALSE )
    {
        bonds_done = TRUE;
    }
#if defined(LOG_PERFORMANCE)
    else
    {
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_BOND_START],
                control->gpu_time_events[TE_INIT_BOND_STOP] ); 
        data->timing.init_bond += (real) (time_elapsed / 1000.0);
    }
#endif
    if ( workspace->realloc[RE_HBONDS] == FALSE )
    {
        hbonds_done = TRUE;
    }
#if defined(LOG_PERFORMANCE)
    else if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
    {
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_HBOND_START],
                control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
        data->timing.init_hbond += (real) (time_elapsed / 1000.0);
    }
#endif

    if ( ret == SUCCESS )
    {
#if defined(LOG_PERFORMANCE)
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_BOND_START],
                control->gpu_time_events[TE_INIT_BOND_STOP] ); 
        data->timing.init_bond += (real) (time_elapsed / 1000.0);

        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_START], control->gpu_streams[1] );
#endif

//        k_update_sym_dbond_indices <<< control->blocks_N, control->gpu_block_size,
//                                   0, control->gpu_streams[1] >>>
//            ( *(lists[BONDS]), system->N );
        k_update_sym_dbond_indices_opt <<< control->blocks_warp_N, control->gpu_block_size,
                                       0, control->gpu_streams[1] >>>
            ( *(lists[BONDS]), system->N );
        cudaCheckError( );

        cudaEventRecord( control->gpu_stream_events[SE_INIT_BOND_DONE], control->gpu_streams[1] );

#if defined(LOG_PERFORMANCE)
        cudaEventRecord( control->gpu_time_events[TE_INIT_BOND_STOP], control->gpu_streams[1] );
#endif

#if !defined(GPU_KERNEL_ATOMIC)
//        if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
//        {
//#if defined(LOG_PERFORMANCE)
//            cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_HBOND_START],
//                    control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
//            data->timing.init_hbond += (real) (time_elapsed / 1000.0);
//
//            cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_START], control->gpu_streams[2] );
//#endif
//
//            /* make hbond_list symmetric */
//            k_update_sym_hbond_indices_opt <<< control->blocks_warp_N, control->gpu_block_size,
//                                           0, control->gpu_streams[2] >>>
//                ( system->d_my_atoms, *(lists[HBONDS]), system->N );
//            cudaCheckError( );
//
//#if defined(LOG_PERFORMANCE)
//            cudaEventRecord( control->gpu_time_events[TE_INIT_HBOND_STOP], control->gpu_streams[2] );
//#endif
//        }
#endif

        dist_done = FALSE;
        bonds_done = FALSE;
        hbonds_done = FALSE;
    }
    else
    {
        GPU_Estimate_Storages( system, control, data, workspace, lists,
               FALSE, workspace->realloc[RE_BONDS],
               workspace->realloc[RE_HBONDS], data->step - data->prev_steps );
    }

    return ret;
}


int GPU_Compute_Bonded_Forces( reax_system * const system, control_params * const control, 
        simulation_data * const data, storage * const workspace, 
        reax_list ** const lists, output_controls * const out_control )
{
    int ret;
    static int compute_bonded_part1 = FALSE;

    ret = SUCCESS;

    if ( compute_bonded_part1 == FALSE )
    {
        GPU_Compute_Bond_Orders( system, control, data, workspace, lists,
                out_control );

        GPU_Compute_Bonds( system, control, data, workspace, lists,
                out_control );

        GPU_Compute_Atom_Energy( system, control, data, workspace, lists,
                out_control );

        if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
        {
            GPU_Compute_Hydrogen_Bonds( system, control, data, workspace,
                    lists, out_control );
        }

        compute_bonded_part1 = TRUE;
    }

    ret = GPU_Compute_Valence_Angles( system, control, data, workspace,
            lists, out_control );

    if ( ret == SUCCESS )
    {
        GPU_Compute_Torsion_Angles( system, control, data, workspace, lists,
                out_control );

        compute_bonded_part1 = FALSE;
    }

    return ret;
}


static void GPU_Compute_Total_Force( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, mpi_datatypes * const mpi_data )
{
    sCudaHostAllocCheck( &workspace->scratch[0], &workspace->scratch_size[0],
            sizeof(rvec) * system->N, cudaHostAllocPortable, TRUE, SAFE_ZONE,
            __FILE__, __LINE__ );
    memset( workspace->scratch[0], 0, sizeof(rvec) * system->N );

    GPU_Total_Forces_Part1( system, control, data, workspace, lists );

    /* now all forces are computed to their partially-final values
     * based on the neighbors information each processor has had.
     * final values of force on each atom needs to be computed by adding up
     * all partially-final pieces */
    sCudaMemcpyAsync( workspace->scratch[0], workspace->d_workspace->f,
            sizeof(rvec) * system->N, cudaMemcpyDeviceToHost,
            control->gpu_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->gpu_streams[0] );

    Coll( system, mpi_data, workspace->scratch[0], RVEC_PTR_TYPE,
            mpi_data->mpi_rvec );

    sCudaMemcpyAsync( workspace->d_workspace->f, workspace->scratch[0],
            sizeof(rvec) * system->N, cudaMemcpyHostToDevice,
            control->gpu_streams[0], __FILE__, __LINE__ );

    GPU_Total_Forces_Part2( system, control, workspace );
}


extern "C" int GPU_Compute_Forces( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
{
    int i, charge_flag, ret;
    static int init_forces_done = FALSE, nonbonded_forces_part1_done = FALSE;
#if defined(LOG_PERFORMANCE)
    float time_elapsed, time_elapsed2, time_elapsed3;
    float time_elapsed4, time_elapsed5, time_elapsed6;
    float time_elapsed7, time_elapsed8, time_elapsed9;
#endif

    ret = SUCCESS;

    if ( control->charge_freq > 0
            && (data->step - data->prev_steps) % control->charge_freq == 0 )
    {
        charge_flag = TRUE;
    }
    else
    {
        charge_flag = FALSE;
    }

    if ( init_forces_done == FALSE )
    {
        if ( charge_flag == TRUE )
        {
            ret = GPU_Init_Forces( system, control, data,
                    workspace, lists, out_control );
        }
        else
        {
            ret = GPU_Init_Forces_No_Charges( system, control, data,
                    workspace, lists, out_control );
        }

        if ( ret == SUCCESS )
        {
            init_forces_done = TRUE;
        }
    }

    if ( nonbonded_forces_part1_done == FALSE )
    {
        GPU_Compute_NonBonded_Forces_Part1( system, control, data, workspace,
                lists, out_control );

        nonbonded_forces_part1_done = TRUE;
    }

    if ( ret == SUCCESS )
    {
        ret = GPU_Compute_Bonded_Forces( system, control, data,
                workspace, lists, out_control );
    }

    if ( ret == SUCCESS )
    {
#if defined(LOG_PERFORMANCE)
        real time;

        time = Get_Time( );
#endif

        if ( charge_flag == TRUE )
        {
            GPU_Compute_Charges( system, control, data,
                    workspace, out_control, mpi_data, control->gpu_streams[5] );
        }
    
#if defined(LOG_PERFORMANCE)
        Update_Timing_Info( &time, &data->timing.cm );
#endif

        GPU_Compute_NonBonded_Forces_Part2( system, control, data, workspace,
                lists, out_control );

        for ( i = 0; i < MAX_GPU_STREAMS; ++i )
        {
            cudaStreamSynchronize( control->gpu_streams[i] );
        }

        GPU_Compute_Total_Force( system, control, data, workspace, lists, mpi_data );

#if defined(LOG_PERFORMANCE)
        if ( (data->step - data->prev_steps) % control->reneighbor == 0 )
        {
            cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_NBRS_START],
                    control->gpu_time_events[TE_NBRS_STOP] ); 
            data->timing.nbrs += (real) (time_elapsed / 1000.0);

            cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_CM_START],
                    control->gpu_time_events[TE_INIT_CM_STOP] ); 
            cudaEventElapsedTime( &time_elapsed2, control->gpu_time_events[TE_INIT_CM_START],
                    control->gpu_time_events[TE_INIT_BOND_STOP] ); 
            if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
            {
                cudaEventElapsedTime( &time_elapsed3, control->gpu_time_events[TE_INIT_CM_START],
                        control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
            }
            else
            {
                time_elapsed3 = 0.0;
            }
            cudaEventElapsedTime( &time_elapsed4, control->gpu_time_events[TE_INIT_BOND_START],
                    control->gpu_time_events[TE_INIT_CM_STOP] ); 
            cudaEventElapsedTime( &time_elapsed5, control->gpu_time_events[TE_INIT_BOND_START],
                    control->gpu_time_events[TE_INIT_BOND_STOP] ); 
            if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
            {
                cudaEventElapsedTime( &time_elapsed6, control->gpu_time_events[TE_INIT_BOND_START],
                        control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
                cudaEventElapsedTime( &time_elapsed7, control->gpu_time_events[TE_INIT_HBOND_START],
                        control->gpu_time_events[TE_INIT_CM_STOP] ); 
                cudaEventElapsedTime( &time_elapsed8, control->gpu_time_events[TE_INIT_HBOND_START],
                        control->gpu_time_events[TE_INIT_BOND_STOP] ); 
                cudaEventElapsedTime( &time_elapsed9, control->gpu_time_events[TE_INIT_HBOND_START],
                        control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
            }
            else
            {
                time_elapsed6 = 0.0;
                time_elapsed7 = 0.0;
                time_elapsed8 = 0.0;
                time_elapsed9 = 0.0;
            }
            data->timing.init_forces += (real) MAX3( 
                MAX3(time_elapsed / 1000.0, time_elapsed2 / 1000.0, time_elapsed3 / 1000.0),
                MAX3(time_elapsed4 / 1000.0, time_elapsed5 / 1000.0, time_elapsed6 / 1000.0),
                MAX3(time_elapsed7 / 1000.0, time_elapsed8 / 1000.0, time_elapsed9 / 1000.0) );
        }
        else
        {
            cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_DIST_START],
                    control->gpu_time_events[TE_INIT_CM_STOP] ); 
            cudaEventElapsedTime( &time_elapsed2, control->gpu_time_events[TE_INIT_DIST_START],
                    control->gpu_time_events[TE_INIT_BOND_STOP] ); 
            if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
            {
                cudaEventElapsedTime( &time_elapsed3, control->gpu_time_events[TE_INIT_DIST_START],
                        control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
            }
            else
            {
                time_elapsed3 = 0.0;
            }
            data->timing.init_forces += (real) MAX3(time_elapsed / 1000.0, time_elapsed2 / 1000.0,
                    time_elapsed3 / 1000.0);

            cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_DIST_START],
                    control->gpu_time_events[TE_INIT_DIST_STOP] ); 
            data->timing.init_dist += (real) (time_elapsed / 1000.0);
        }

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_CM_START],
                control->gpu_time_events[TE_INIT_CM_STOP] ); 
        data->timing.init_cm += (real) (time_elapsed / 1000.0);

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_BOND_START],
                control->gpu_time_events[TE_INIT_BOND_STOP] ); 
        data->timing.init_bond += (real) (time_elapsed / 1000.0);

        if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
        {
            cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_INIT_HBOND_START],
                    control->gpu_time_events[TE_INIT_HBOND_STOP] ); 
            data->timing.init_hbond += (real) (time_elapsed / 1000.0);
        }

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_BOND_ORDER_START],
                control->gpu_time_events[TE_LPOVUN_STOP] ); 
        cudaEventElapsedTime( &time_elapsed2, control->gpu_time_events[TE_BOND_ORDER_START],
                control->gpu_time_events[TE_BONDS_STOP] ); 
        cudaEventElapsedTime( &time_elapsed3, control->gpu_time_events[TE_BOND_ORDER_START],
                control->gpu_time_events[TE_TORSION_STOP] ); 
        if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
        {
            cudaEventElapsedTime( &time_elapsed4, control->gpu_time_events[TE_BOND_ORDER_START],
                    control->gpu_time_events[TE_HBONDS_STOP] ); 
        }
        else
        {
            time_elapsed4 = 0.0;
        }
        data->timing.bonded += (real) MAX(MAX3(time_elapsed / 1000.0, time_elapsed2 / 1000.0,
                time_elapsed3 / 1000.0), time_elapsed4 / 1000.0);

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_BOND_ORDER_START],
                control->gpu_time_events[TE_BOND_ORDER_STOP] ); 
        data->timing.bond_order += (real) (time_elapsed / 1000.0);

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_BONDS_START],
                control->gpu_time_events[TE_BONDS_STOP] ); 
        data->timing.bonds += (real) (time_elapsed / 1000.0);

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_LPOVUN_START],
                control->gpu_time_events[TE_LPOVUN_STOP] ); 
        data->timing.lpovun += (real) (time_elapsed / 1000.0);

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_VALENCE_START],
                control->gpu_time_events[TE_VALENCE_STOP] ); 
        data->timing.valence += (real) (time_elapsed / 1000.0);

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_TORSION_START],
                control->gpu_time_events[TE_TORSION_STOP] ); 
        data->timing.torsion += (real) (time_elapsed / 1000.0);

        if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
        {
            cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_HBONDS_START],
                    control->gpu_time_events[TE_HBONDS_STOP] ); 
            data->timing.hbonds += (real) (time_elapsed / 1000.0);
        }

#if !defined(FUSED_VDW_COULOMB)
        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_VDW_START],
                control->gpu_time_events[TE_VDW_STOP] ); 
        data->timing.nonb += (real) (time_elapsed / 1000.0);
#endif

        cudaEventElapsedTime( &time_elapsed, control->gpu_time_events[TE_COULOMB_START],
                control->gpu_time_events[TE_COULOMB_STOP] ); 
        data->timing.nonb += (real) (time_elapsed / 1000.0);
#endif

        init_forces_done = FALSE;
        nonbonded_forces_part1_done = FALSE;
    }

    return ret;
}
