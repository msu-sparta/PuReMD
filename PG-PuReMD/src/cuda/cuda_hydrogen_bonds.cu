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

#include "cuda_hydrogen_bonds.h"

#include "cuda_valence_angles.h"
#include "cuda_helpers.h"
#include "cuda_list.h"
#include "cuda_reduction.h"
#include "cuda_utils.h"

#include "../index_utils.h"
#include "../vector.h"

#include <cub/warp/warp_reduce.cuh>


/* one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part1( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, hbond_parameters const * const hbp,
        global_parameters gp, control_params const * const control, storage workspace,
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real * const e_hb_g )
{
    int i, j, k, pi, pk;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int *hblist, hblist_size;
    int itr, top;
    int nbr_jk, hbp_ijk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk;
    rvec f_j, f_k;
    bond_order_data *bo_ij;
    bond_data *pbond_ij;
    hbond_data *phbond_jk;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    /* discover the Hydrogen bonds between i-j-k triplets.
     * here j is H atom and there has to be some bond between i and j.
     * Hydrogen bond is between j and k.
     * so in this function i->X, j->H, k->Z when we map 
     * variables onto the ones in the handout. */

    /* j must be a hydrogen atom */
    if ( sbp[ my_atoms[j].type ].p_hbond == H_ATOM )
    {
        hblist = NULL;
        hblist_size = 0;
        type_j = my_atoms[j].type;
        start_j = Start_Index( j, &bond_list );
        end_j = End_Index( j, &bond_list );
        hb_start_j = Start_Index( my_atoms[j].Hindex, &hbond_list );
        hb_end_j = End_Index( my_atoms[j].Hindex, &hbond_list );
        top = 0;
        e_hb_ = 0.0;
        rvec_MakeZero( f_j );

        if ( Num_Entries( j, &bond_list ) > hblist_size )
        {
            hblist_size = Num_Entries( j, &bond_list );
            hblist = (int *) malloc( sizeof(int) * hblist_size );
        }

        /* search bonded atoms i to atom j (hydrogen atom) for potential hydrogen bonding */
        for ( pi = start_j; pi < end_j; ++pi )
        {
            pbond_ij = &bond_list.bond_list[pi];
            i = pbond_ij->nbr;
            bo_ij = &pbond_ij->bo_data;
            type_i = my_atoms[i].type;

            if ( sbp[type_i].p_hbond == H_BONDING_ATOM
                    && bo_ij->BO >= HB_THRESHOLD )
            {
                hblist[top] = pi;
                ++top;
            }
        }

        /* for each hbond of atom j */
        for ( pk = hb_start_j; pk < hb_end_j; ++pk )
        {
            phbond_jk = &hbond_list.hbond_list[pk];
            k = phbond_jk->nbr;
            type_k = my_atoms[k].type;
            nbr_jk = phbond_jk->ptr;
            r_jk = far_nbr_list.far_nbr_list.d[nbr_jk];

            rvec_MakeZero( f_k );

            rvec_Scale( dvec_jk, hbond_list.hbond_list[pk].scl,
                    far_nbr_list.far_nbr_list.dvec[nbr_jk] );

            /* find matching hbond to atoms j and k */
            for ( itr = 0; itr < top; ++itr )
            {
                pi = hblist[itr];
                pbond_ij = &bond_list.bond_list[pi];
                i = pbond_ij->nbr;

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id
                        && hbp[ index_hbp(my_atoms[i].type, type_j, type_k, num_atom_types) ].is_valid == TRUE )
                {
                    bo_ij = &pbond_ij->bo_data;
                    type_i = my_atoms[i].type;
                    r_ij = pbond_ij->d;
                    hbp_ijk = index_hbp(type_i, type_j, type_k, num_atom_types);

                    Calculate_Theta( pbond_ij->dvec, r_ij, dvec_jk, r_jk,
                            &theta, &cos_theta );

                    /* the derivative of cos(theta) */
                    Calculate_dCos_Theta( pbond_ij->dvec, r_ij, dvec_jk, r_jk,
                            &dcos_theta_di, &dcos_theta_dj, &dcos_theta_dk );

                    /* hydrogen bond energy */
                    sin_theta2 = SIN( theta / 2.0 );
                    sin_xhz4 = SQR( sin_theta2 );
                    sin_xhz4 *= sin_xhz4;
                    cos_xhz1 = ( 1.0 - cos_theta );
                    exp_hb2 = EXP( -1.0 * hbp[hbp_ijk].p_hb2 * bo_ij->BO );
                    exp_hb3 = EXP( -1.0 * hbp[hbp_ijk].p_hb3 * ( hbp[hbp_ijk].r0_hb / r_jk
                                + r_jk / hbp[hbp_ijk].r0_hb - 2.0 ) );

                    e_hb = hbp[hbp_ijk].p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                    e_hb_ += e_hb;

                    CEhb1 = hbp[hbp_ijk].p_hb1 * hbp[hbp_ijk].p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                    CEhb2 = -0.5 * hbp[hbp_ijk].p_hb1 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                    CEhb3 = hbp[hbp_ijk].p_hb3 * e_hb * (hbp[hbp_ijk].r0_hb / SQR( r_jk )
                            + -1.0 / hbp[hbp_ijk].r0_hb);

                    /* hydrogen bond forces */
                    /* dbo term */
                    atomicAdd( &bo_ij->Cdbo, CEhb1 );

                    /* dcos terms */
#if !defined(GPU_ACCUM_ATOMIC)
                    rvec_ScaledAdd( pbond_ij->hb_f, CEhb2, dcos_theta_di ); 
#else
                    atomic_rvecScaledAdd( workspace.f[i], CEhb2, dcos_theta_di );
#endif
                    rvec_ScaledAdd( f_j, CEhb2, dcos_theta_dj );
                    rvec_ScaledAdd( f_k, CEhb2, dcos_theta_dk );

                    /* dr terms */
                    rvec_ScaledAdd( f_j, -1.0 * CEhb3 / r_jk, dvec_jk ); 
                    rvec_ScaledAdd( f_k, CEhb3 / r_jk, dvec_jk );
                }
            }

#if !defined(GPU_ACCUM_ATOMIC)
            rvec_Copy( phbond_jk->hb_f, f_k );
#else
            atomic_rvecAdd( workspace.f[k], f_k );
#endif
        }

        if ( hblist != NULL )
        {
            free( hblist );
        }

#if !defined(GPU_ACCUM_ATOMIC)
        rvecCopy( workspace.f[j], f_j );
        e_hb_g[j] = e_hb_;
#else
        atomic_rvecAdd( workspace.f[j], f_j );
        atomicAdd( (double *) e_hb_g, (double) e_hb_ );
#endif
    }
}


/* one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part1_opt( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, hbond_parameters const * const hbp,
        global_parameters gp, control_params const * const control, storage workspace,
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real * const e_hb_g )
{
    extern __shared__ cub::WarpReduce<double>::TempStorage temp_d[];
    int i, j, k, pi, pk, thread_id, warp_id, lane_id, itr;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int nbr_jk, hbp_ijk;
    real r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk;
    rvec f_j, f_k;
    bond_order_data *bo_ij;
    bond_data *pbond_ij;
    hbond_data *phbond_jk;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    j = thread_id / warpSize;

    if ( j >= n )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize; 

    /* discover the Hydrogen bonds between i-j-k triplets.
     * here j is H atom and there has to be some bond between i and j.
     * Hydrogen bond is between j and k.
     * so in this function i->X, j->H, k->Z when we map 
     * variables onto the ones in the handout. */

    /* j must be a hydrogen atom */
    if ( sbp[ my_atoms[j].type ].p_hbond == H_ATOM )
    {
        type_j = my_atoms[j].type;
        start_j = Start_Index( j, &bond_list );
        end_j = End_Index( j, &bond_list );
        hb_start_j = Start_Index( my_atoms[j].Hindex, &hbond_list );
        hb_end_j = End_Index( my_atoms[j].Hindex, &hbond_list );
        e_hb_ = 0.0;
        rvec_MakeZero( f_j );

        /* for each hbond of atom j */
        for ( pk = hb_start_j; pk < hb_end_j; ++pk )
        {
            phbond_jk = &hbond_list.hbond_list[pk];
            k = phbond_jk->nbr;
            type_k = my_atoms[k].type;
            nbr_jk = phbond_jk->ptr;
            r_jk = far_nbr_list.far_nbr_list.d[nbr_jk];

            rvec_MakeZero( f_k );

            rvec_Scale( dvec_jk, hbond_list.hbond_list[pk].scl,
                    far_nbr_list.far_nbr_list.dvec[nbr_jk] );

            /* search bonded atoms i to atom j (hydrogen atom) for potential hydrogen bonding */
            for ( itr = 0, pi = start_j + lane_id; itr < (end_j - start_j + warpSize - 1) / warpSize; ++itr )
            {
                if ( pi < end_j )
                {
                    pbond_ij = &bond_list.bond_list[pi];
                    i = pbond_ij->nbr;
                    bo_ij = &pbond_ij->bo_data;
                    type_i = my_atoms[i].type;

                    if ( sbp[type_i].p_hbond == H_BONDING_ATOM
                            && bo_ij->BO >= HB_THRESHOLD
                            && my_atoms[i].orig_id != my_atoms[k].orig_id )
                    {
                        hbp_ijk = index_hbp(type_i, type_j, type_k, num_atom_types);

                        Calculate_Theta( pbond_ij->dvec, pbond_ij->d, dvec_jk, r_jk,
                                &theta, &cos_theta );

                        /* the derivative of cos(theta) */
                        Calculate_dCos_Theta( pbond_ij->dvec, pbond_ij->d, dvec_jk, r_jk,
                                &dcos_theta_di, &dcos_theta_dj, &dcos_theta_dk );

                        /* hydrogen bond energy */
                        sin_theta2 = SIN( theta / 2.0 );
                        sin_xhz4 = SQR( sin_theta2 );
                        sin_xhz4 *= sin_xhz4;
                        cos_xhz1 = ( 1.0 - cos_theta );
                        exp_hb2 = EXP( -1.0 * hbp[hbp_ijk].p_hb2 * bo_ij->BO );
                        exp_hb3 = EXP( -1.0 * hbp[hbp_ijk].p_hb3 * ( hbp[hbp_ijk].r0_hb / r_jk
                                    + r_jk / hbp[hbp_ijk].r0_hb - 2.0 ) );

                        e_hb = hbp[hbp_ijk].p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                        e_hb_ += e_hb;

                        CEhb1 = hbp[hbp_ijk].p_hb1 * hbp[hbp_ijk].p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                        CEhb2 = -0.5 * hbp[hbp_ijk].p_hb1 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                        CEhb3 = hbp[hbp_ijk].p_hb3 * e_hb * (hbp[hbp_ijk].r0_hb / SQR( r_jk )
                                + -1.0 / hbp[hbp_ijk].r0_hb);

                        /* hydrogen bond forces */
                        /* dbo term */
                        atomicAdd( &bo_ij->Cdbo, CEhb1 );

                        /* dcos terms */
#if !defined(GPU_ACCUM_ATOMIC)
                        rvec_ScaledAdd( pbond_ij->hb_f, CEhb2, dcos_theta_di ); 
#else
                        atomic_rvecScaledAdd( workspace.f[i], CEhb2, dcos_theta_di );
#endif
                        rvec_ScaledAdd( f_j, CEhb2, dcos_theta_dj );
                        rvec_ScaledAdd( f_k, CEhb2, dcos_theta_dk );

                        /* dr terms */
                        rvec_ScaledAdd( f_j, -1.0 * CEhb3 / r_jk, dvec_jk ); 
                        rvec_ScaledAdd( f_k, CEhb3 / r_jk, dvec_jk );
                    }
                }

                pi += warpSize;
            }

            f_k[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[0]);
            f_k[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[1]);
            f_k[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[2]);

            if ( lane_id == 0 )
            {
#if !defined(GPU_ACCUM_ATOMIC)
                rvec_Copy( phbond_jk->hb_f, f_k );
#else
                atomic_rvecAdd( workspace.f[k], f_k );
#endif
            }
        }

        f_j[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[0]);
        f_j[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[1]);
        f_j[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[2]);
        e_hb_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_hb_);

        if ( lane_id == 0 )
        {
#if !defined(GPU_ACCUM_ATOMIC)
            /* write conflicts for accumulating partial forces resolved by subsequent kernels */
            rvecCopy( workspace.f[j], f_j );
            e_hb_g[j] = e_hb_;
#else
            atomic_rvecAdd( workspace.f[j], f_j );
            atomicAdd( (double *) e_hb_g, (double) e_hb_ );
#endif
        }
    }
}


/* one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_virial_part1( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, hbond_parameters const * const hbp,
        global_parameters gp, control_params const * const control, storage workspace,
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real * const e_hb_g, rvec * const ext_press_g )
{
    int i, j, k, pi, pk;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int *hblist, hblist_size;
    int itr, top;
    int nbr_jk, hbp_ijk;
    ivec rel_jk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk, temp, ext_press_l;
#if defined(GPU_ACCUM_ATOMIC)
    rvec f_j;
#endif
    bond_order_data *bo_ij;
    bond_data *pbond_ij;
    hbond_data *phbond_jk;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    e_hb_ = 0.0;
    rvec_MakeZero( ext_press_l );
#if defined(GPU_ACCUM_ATOMIC)
    rvec_MakeZero( f_j );
#endif

    /* discover the Hydrogen bonds between i-j-k triplets.
     * here j is H atom and there has to be some bond between i and j.
     * Hydrogen bond is between j and k.
     * so in this function i->X, j->H, k->Z when we map 
     * variables onto the ones in the handout. */

    /* j must be a hydrogen atom */
    if ( sbp[ my_atoms[j].type ].p_hbond == H_ATOM )
    {
        hblist = NULL;
        hblist_size = 0;
        type_j = my_atoms[j].type;
        start_j = Start_Index( j, &bond_list );
        end_j = End_Index( j, &bond_list );
        hb_start_j = Start_Index( my_atoms[j].Hindex, &hbond_list );
        hb_end_j = End_Index( my_atoms[j].Hindex, &hbond_list );
        top = 0;

        if ( Num_Entries( j, &bond_list ) > hblist_size )
        {
            hblist_size = Num_Entries( j, &bond_list );
            hblist = (int *) malloc( sizeof(int) * hblist_size );
        }

        /* search bonded atoms i to atom j (hydrogen atom) for potential hydrogen bonding */
        for ( pi = start_j; pi < end_j; ++pi )
        {
            pbond_ij = &bond_list.bond_list[pi];
            i = pbond_ij->nbr;
            bo_ij = &pbond_ij->bo_data;
            type_i = my_atoms[i].type;

            if ( sbp[type_i].p_hbond == H_BONDING_ATOM
                    && bo_ij->BO >= HB_THRESHOLD )
            {
                hblist[top] = pi;
                ++top;
            }
        }

        /* for each hbond of atom j */
        for ( pk = hb_start_j; pk < hb_end_j; ++pk )
        {
            phbond_jk = &hbond_list.hbond_list[pk];
            k = phbond_jk->nbr;
            type_k = my_atoms[k].type;
            nbr_jk = phbond_jk->ptr;
            r_jk = far_nbr_list.far_nbr_list.d[nbr_jk];

            rvec_Scale( dvec_jk, hbond_list.hbond_list[pk].scl,
                    far_nbr_list.far_nbr_list.dvec[nbr_jk] );

#if !defined(GPU_ACCUM_ATOMIC)
            rvec_MakeZero( phbond_jk->hb_f );
#endif

            /* find matching hbond to atoms j and k */
            for ( itr = 0; itr < top; ++itr )
            {
                pi = hblist[itr];
                pbond_ij = &bond_list.bond_list[pi];
                i = pbond_ij->nbr;

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id
                        && hbp[ index_hbp(my_atoms[i].type, type_j, type_k, num_atom_types) ].is_valid == TRUE )
                {
                    bo_ij = &pbond_ij->bo_data;
                    type_i = my_atoms[i].type;
                    r_ij = pbond_ij->d;
                    hbp_ijk = index_hbp(type_i, type_j, type_k, num_atom_types);

                    Calculate_Theta( pbond_ij->dvec, r_ij, dvec_jk, r_jk,
                            &theta, &cos_theta );

                    /* the derivative of cos(theta) */
                    Calculate_dCos_Theta( pbond_ij->dvec, r_ij, dvec_jk, r_jk,
                            &dcos_theta_di, &dcos_theta_dj, &dcos_theta_dk );

                    /* hydrogen bond energy */
                    sin_theta2 = SIN( theta / 2.0 );
                    sin_xhz4 = SQR( sin_theta2 );
                    sin_xhz4 *= sin_xhz4;
                    cos_xhz1 = ( 1.0 - cos_theta );
                    exp_hb2 = EXP( -1.0 * hbp[hbp_ijk].p_hb2 * bo_ij->BO );
                    exp_hb3 = EXP( -1.0 * hbp[hbp_ijk].p_hb3 * ( hbp[hbp_ijk].r0_hb / r_jk
                                + r_jk / hbp[hbp_ijk].r0_hb - 2.0 ) );

                    e_hb = hbp[hbp_ijk].p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                    e_hb_ += e_hb;

                    CEhb1 = hbp[hbp_ijk].p_hb1 * hbp[hbp_ijk].p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                    CEhb2 = -0.5 * hbp[hbp_ijk].p_hb1 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                    CEhb3 = hbp[hbp_ijk].p_hb3 * e_hb * (hbp[hbp_ijk].r0_hb / SQR( r_jk )
                            + -1.0 / hbp[hbp_ijk].r0_hb);

                    /* hydrogen bond forces */
                    /* dbo term */
                    atomicAdd( &bo_ij->Cdbo, CEhb1 );

#if !defined(GPU_ACCUM_ATOMIC)
                    /* for pressure coupling, terms that are not related to bond order
                     * derivatives are added directly into pressure vector/tensor */
                    /* dcos terms */
                    rvec_Scale( temp, CEhb2, dcos_theta_di );
                    rvec_Add( pbond_ij->hb_f, temp );
                    rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                    rvec_Add( ext_press_l, temp );

                    rvec_ScaledAdd( workspace.f[j], CEhb2, dcos_theta_dj );

                    ivec_Scale( rel_jk, hbond_list.hbond_list[pk].scl,
                            far_nbr_list.far_nbr_list.rel_box[nbr_jk] );
                    rvec_Scale( temp, CEhb2, dcos_theta_dk );
                    rvec_Add( phbond_jk->hb_f, temp );
                    rvec_iMultiply( temp, rel_jk, temp );
                    rvec_Add( ext_press_l, temp );

                    /* dr terms */
                    rvec_ScaledAdd( workspace.f[j], -1.0 * CEhb3 / r_jk, dvec_jk ); 

                    rvec_Scale( temp, CEhb3 / r_jk, dvec_jk );
                    rvec_Add( phbond_jk->hb_f, temp );
                    rvec_iMultiply( temp, rel_jk, temp );
                    rvec_Add( ext_press_l, temp );
#else
                    /* for pressure coupling, terms that are not related to bond order
                     * derivatives are added directly into pressure vector/tensor */
                    /* dcos terms */
                    rvec_Scale( temp, CEhb2, dcos_theta_di );
                    atomic_rvecAdd( workspace.f[i], temp );
                    rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                    rvec_Add( ext_press_l, temp );

                    rvec_ScaledAdd( f_j, CEhb2, dcos_theta_dj );

                    ivec_Scale( rel_jk, hbond_list.hbond_list[pk].scl,
                            far_nbr_list.far_nbr_list.rel_box[nbr_jk] );
                    rvec_Scale( temp, CEhb2, dcos_theta_dk );
                    atomic_rvecAdd( workspace.f[k], temp );
                    rvec_iMultiply( temp, rel_jk, temp );
                    rvec_Add( ext_press_l, temp );

                    /* dr terms */
                    rvec_ScaledAdd( f_j, -1.0 * CEhb3 / r_jk, dvec_jk ); 

                    rvec_Scale( temp, CEhb3 / r_jk, dvec_jk );
                    atomic_rvecAdd( workspace.f[k], temp );
                    rvec_iMultiply( temp, rel_jk, temp );
                    rvec_Add( ext_press_l, temp );
#endif
                }
            }
        }

        if ( hblist != NULL )
        {
            free( hblist );
        }
    }

#if !defined(GPU_ACCUM_ATOMIC)
    /* write conflicts for accumulating partial forces resolved by subsequent kernels */
    rvecCopy( workspace.f[j], f_j );
    e_hb_g[j] = e_hb_;
    rvecCopy( ext_press_g[j], ext_press_l );
#else
    atomic_rvecAdd( workspace.f[j], f_j );
    atomicAdd( (double *) e_hb_g, (double) e_hb_ );
    atomic_rvecAdd( *ext_press_g, ext_press_l );
#endif
}


#if !defined(GPU_ACCUM_ATOMIC)
/* Accumulate forces stored in the bond list
 * using a one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part2( reax_atom *atoms,
        storage workspace, reax_list bond_list, int n )
{
    int j, pj;
    bond_data *pbond, *sym_index_bond;
    rvec hb_f;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    rvec_MakeZero( hb_f );

    for ( pj = Start_Index(j, &bond_list); pj < End_Index(j, &bond_list); ++pj )
    {
        pbond = &bond_list.bond_list[pj];
        sym_index_bond = &bond_list.bond_list[pbond->sym_index];

        rvec_Add( hb_f, sym_index_bond->hb_f );
    }

    rvec_Add( workspace.f[j], hb_f );
}


/* Accumulate forces stored in the bond list
 * using a one warp threads per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part2_opt( reax_atom *atoms,
        storage workspace, reax_list bond_list, int n )
{
    extern __shared__ cub::WarpReduce<double>::TempStorage temp_storage[];
    int j, pj, start, end;
    bond_data *pbond, *sym_index_bond;
    /* thread-local variables */
    int thread_id, warp_id, lane_id;
    rvec hb_f;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    j = thread_id / warpSize;

    if ( j >= n )
    {
        return;
    }

    lane_id = thread_id % warpSize; 
    warp_id = threadIdx.x / warpSize;
    start = Start_Index( j, &bond_list );
    end = End_Index( j, &bond_list );
    pj = start + lane_id;
    rvec_MakeZero( hb_f );

    while ( pj < end )
    {
        pbond = &bond_list.bond_list[pj];
        sym_index_bond = &bond_list.bond_list[pbond->sym_index];

        rvec_Add( hb_f, sym_index_bond->hb_f );

        pj += warpSize;
    }

    hb_f[0] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(hb_f[0]);
    hb_f[1] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(hb_f[1]);
    hb_f[2] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(hb_f[2]);

    /* first thread within a warp writes warp-level sums to global memory */
    if ( lane_id == 0 )
    {
        rvec_Add( workspace.f[j], hb_f );
    }
}


/* Accumulate forces stored in the hbond list
 * using a one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part3( reax_atom *atoms,
        storage workspace, reax_list hbond_list, int n )
{
    int j, pj;
    hbond_data *nbr_pj, *sym_index_nbr;
    rvec hb_f;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    rvec_MakeZero( hb_f );

    for ( pj = Start_Index(atoms[j].Hindex, &hbond_list); pj < End_Index(atoms[j].Hindex, &hbond_list); ++pj )
    {
        nbr_pj = &hbond_list.hbond_list[pj];
        sym_index_nbr = &hbond_list.hbond_list[ nbr_pj->sym_index ];

        rvec_Add( hb_f, sym_index_nbr->hb_f );
    }

    rvec_Add( workspace.f[j], hb_f );
}


/* Accumulate forces stored in the hbond list
 * using a one warp of threads per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part3_opt( reax_atom *atoms,
        storage workspace, reax_list hbond_list, int n )
{
    extern __shared__ cub::WarpReduce<double>::TempStorage temp_storage[];
    int j, pj, start, end;
    hbond_data *nbr_pj, *sym_index_nbr;
    /* thread-local variables */
    int thread_id, warp_id, lane_id, offset;
    rvec hb_f_l;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    j = thread_id / warpSize;

    if ( j >= n )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize; 
    start = Start_Index( atoms[j].Hindex, &hbond_list );
    end = End_Index( atoms[j].Hindex, &hbond_list );
    pj = start + lane_id;
    rvec_MakeZero( hb_f_l );

    while ( pj < end )
    {
        nbr_pj = &hbond_list.hbond_list[pj];
        sym_index_nbr = &hbond_list.hbond_list[ nbr_pj->sym_index ];

        rvec_Add( hb_f_l, sym_index_nbr->hb_f );

        pj += warpSize;
    }

    hb_f_l[0] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(hb_f_l[0]);
    hb_f_l[1] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(hb_f_l[1]);
    hb_f_l[2] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(hb_f_l[2]);

    /* first thread within a warp writes warp-level sums to global memory */
    if ( lane_id == 0 )
    {
        rvec_Add( workspace.f[j], hb_f_l );
    }
}
#endif



void Cuda_Compute_Hydrogen_Bonds( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list ** lists,
        output_controls const * const out_control )
{
//    int hbs, hnbrs_blocks;
#if !defined(GPU_ACCUM_ATOMIC)
    int update_energy;
    real *spad;
    rvec *rvec_spad;
#endif

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->cuda_time_events[TE_HBONDS_START], control->cuda_streams[2] );
#endif

#if !defined(GPU_ACCUM_ATOMIC)
    sCudaCheckMalloc( &workspace->scratch[2], &workspace->scratch_size[2],
            (sizeof(real) * 3 + sizeof(rvec)) * system->N + sizeof(rvec) * control->blocks_N,
            __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[2];
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#else
    sCudaMemsetAsync( &data->d_my_en->e_hb,
            0, sizeof(real), control->cuda_streams[2], __FILE__, __LINE__ );
    if ( control->virial == 1 )
    {
        sCudaMemsetAsync( &((simulation_data *)data->d_simulation_data)->my_ext_press,
                0, sizeof(rvec), control->cuda_streams[2], __FILE__, __LINE__ );
    }
#endif

    cudaStreamWaitEvent( control->cuda_streams[2], control->cuda_stream_events[SE_BOND_ORDER_DONE], 0 );

    if ( control->virial == 1 )
    {
        k_hydrogen_bonds_virial_part1 <<< control->blocks_n, control->gpu_block_size,
                                      0, control->cuda_streams[2] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp,
                  system->reax_param.d_hbp, system->reax_param.d_gp,
                  (control_params *) control->d_control_params,
                  *(workspace->d_workspace),
                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
                  system->n, system->reax_param.num_atom_types,
#if !defined(GPU_ACCUM_ATOMIC)
                  spad, (rvec *) (&spad[system->n])
#else
                  &data->d_my_en->e_hb,
                  &((simulation_data *)data->d_simulation_data)->my_ext_press
#endif
                );
        cudaCheckError( );
    }
    else
    {
//        k_hydrogen_bonds_part1 <<< control->blocks_n, control->gpu_block_size, 0, control->cuda_streams[2] >>>
//                ( system->d_my_atoms, system->reax_param.d_sbp,
//                  system->reax_param.d_hbp, system->reax_param.d_gp,
//                  (control_params *) control->d_control_params,
//                  *(workspace->d_workspace),
//                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
//                  system->n, system->reax_param.num_atom_types,
//#if !defined(GPU_ACCUM_ATOMIC)
//                  spad
//#else
//                  &data->d_my_en->e_hb
//#endif
//                );
//        cudaCheckError( );
        
        k_hydrogen_bonds_part1_opt <<< control->blocks_warp_n, control->gpu_block_size,
                                   sizeof(cub::WarpReduce<double>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                                   control->cuda_streams[2] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp,
                  system->reax_param.d_hbp, system->reax_param.d_gp,
                  (control_params *) control->d_control_params,
                  *(workspace->d_workspace),
                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
                  system->n, system->reax_param.num_atom_types,
#if !defined(GPU_ACCUM_ATOMIC)
                  spad
#else
                  &data->d_my_en->e_hb
#endif
                );
        cudaCheckError( );
    }

#if !defined(GPU_ACCUM_ATOMIC)
    if ( update_energy == TRUE )
    {
        Cuda_Reduction_Sum( spad, &data->d_my_en->e_hb,
                system->n, 2, control->cuda_streams[2] );
    }

    if ( control->virial == 1 )
    {
        rvec_spad = (rvec *) (&spad[system->n]);

        Cuda_Reduction_Sum( rvec_spad,
                &((simulation_data *)data->d_simulation_data)->my_ext_press,
                system->n, 2, control->cuda_streams[2] );
    }

    k_hydrogen_bonds_part2 <<< control->blocks_n, control->gpu_block_size,
                           0, control->cuda_streams[2] >>>
        ( system->d_my_atoms, *(workspace->d_workspace),
          *(lists[BONDS]), system->n );
    cudaCheckError( );

//    hnbrs_blocks = (system->n * HB_POST_PROC_KER_THREADS_PER_ATOM / HB_POST_PROC_BLOCK_SIZE) +
//        (((system->n * HB_POST_PROC_KER_THREADS_PER_ATOM) % HB_POST_PROC_BLOCK_SIZE) == 0 ? 0 : 1);

    k_hydrogen_bonds_part3 <<< control->blocks_n, control->gpu_block_size,
                           0, control->cuda_streams[2] >>>
        ( system->d_my_atoms, *(workspace->d_workspace), *(lists[HBONDS]), system->n );
//    k_hydrogen_bonds_part3_opt <<< hnbrs_blocks, HB_POST_PROC_BLOCK_SIZE, 
//            sizeof(rvec) * HB_POST_PROC_BLOCK_SIZE, control->cuda_streams[2] >>>
//        ( system->d_my_atoms, *(workspace->d_workspace), *(lists[HBONDS]), system->n );
    cudaCheckError( );
#endif

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->cuda_time_events[TE_HBONDS_STOP], control->cuda_streams[2] );
#endif
}
