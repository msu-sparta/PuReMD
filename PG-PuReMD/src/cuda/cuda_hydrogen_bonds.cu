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

#include "../cub/cub/warp/warp_reduce.cuh"
//#include <cub/warp/warp_reduce.cuh>


/* one thread per atom implementation */
CUDA_GLOBAL void k_hydrogen_bonds_part1( reax_atom *my_atoms, single_body_parameters *sbp, 
        hbond_parameters *d_hbp, global_parameters gp,
        control_params *control, storage workspace,
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real *e_hb_g )
{
    int i, j, k, pi, pk;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int *hblist, hblist_size;
    int itr, top;
    int nbr_jk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_l, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk;
    rvec f_j_l, f_k_l;
    hbond_parameters *hbp;
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
        e_hb_l = 0.0;
        rvec_MakeZero( f_j_l );

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

            rvec_MakeZero( f_k_l );

            rvec_Scale( dvec_jk, hbond_list.hbond_list[pk].scl,
                    far_nbr_list.far_nbr_list.dvec[nbr_jk] );

            /* find matching hbond to atoms j and k */
            for ( itr = 0; itr < top; ++itr )
            {
                pi = hblist[itr];
                pbond_ij = &bond_list.bond_list[pi];
                i = pbond_ij->nbr;

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id )
                {
                    bo_ij = &pbond_ij->bo_data;
                    type_i = my_atoms[i].type;
                    r_ij = pbond_ij->d;
                    hbp = &d_hbp[ index_hbp(type_i, type_j, type_k, num_atom_types) ];

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
                    exp_hb2 = EXP( -1.0 * hbp->p_hb2 * bo_ij->BO );
                    exp_hb3 = EXP( -1.0 * hbp->p_hb3 * ( hbp->r0_hb / r_jk
                                + r_jk / hbp->r0_hb - 2.0 ) );

                    e_hb = hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                    e_hb_l += e_hb;

                    CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                    CEhb2 = -0.5 * hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                    CEhb3 = hbp->p_hb3 * e_hb * (hbp->r0_hb / SQR( r_jk )
                            + -1.0 / hbp->r0_hb);

                    /* hydrogen bond forces */
                    /* dbo term */
                    bo_ij->Cdbo += CEhb1;

                    /* dcos terms */
#if !defined(CUDA_ACCUM_ATOMIC)
                    rvec_ScaledAdd( pbond_ij->hb_f, CEhb2, dcos_theta_di ); 
#else
                    atomic_rvecScaledAdd( workspace.f[i], CEhb2, dcos_theta_di );
#endif
                    rvec_ScaledAdd( f_j_l, CEhb2, dcos_theta_dj );
                    rvec_ScaledAdd( f_k_l, CEhb2, dcos_theta_dk );

                    /* dr terms */
                    rvec_ScaledAdd( f_j_l, -1.0 * CEhb3 / r_jk, dvec_jk ); 
                    rvec_ScaledAdd( f_k_l, CEhb3 / r_jk, dvec_jk );
                }
            }

#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_Copy( phbond_jk->hb_f, f_k_l );
#else
            atomic_rvecAdd( workspace.f[k], f_k_l );
#endif
        }

        if ( hblist != NULL )
        {
            free( hblist );
        }

#if !defined(CUDA_ACCUM_ATOMIC)
        /* write conflicts for accumulating partial forces resolved by subsequent kernels */
        rvecCopy( workspace.f[j], f_j_l );
        e_hb_g[j] = e_hb_l;
#else
        atomic_rvecAdd( workspace.f[j], f_j_l );
        atomicAdd( (double *) e_hb_g, (double) e_hb_l );
#endif
    }
}


/* one thread per atom implementation */
CUDA_GLOBAL void k_hydrogen_bonds_part1_opt( reax_atom *my_atoms, single_body_parameters *sbp, 
        hbond_parameters *d_hbp, global_parameters gp,
        control_params *control, storage workspace,
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real *e_hb_g )
{
    extern __shared__ cub::WarpReduce<double>::TempStorage temp_d[];
    int i, j, k, pi, pk, thread_id, lane_id, itr;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int nbr_jk;
    real r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_l, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk;
    rvec f_j_l, f_k_l;
    hbond_parameters *hbp;
    bond_order_data *bo_ij;
    bond_data *pbond_ij;
    hbond_data *phbond_jk;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    j = thread_id / warpSize;

    if ( j >= n )
    {
        return;
    }

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
        e_hb_l = 0.0;
        rvec_MakeZero( f_j_l );

        /* for each hbond of atom j */
        for ( pk = hb_start_j; pk < hb_end_j; ++pk )
        {
            phbond_jk = &hbond_list.hbond_list[pk];
            k = phbond_jk->nbr;
            type_k = my_atoms[k].type;
            nbr_jk = phbond_jk->ptr;
            r_jk = far_nbr_list.far_nbr_list.d[nbr_jk];

            rvec_MakeZero( f_k_l );

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
                        hbp = &d_hbp[ index_hbp(type_i, type_j, type_k, num_atom_types) ];

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
                        exp_hb2 = EXP( -1.0 * hbp->p_hb2 * bo_ij->BO );
                        exp_hb3 = EXP( -1.0 * hbp->p_hb3 * ( hbp->r0_hb / r_jk
                                    + r_jk / hbp->r0_hb - 2.0 ) );

                        e_hb = hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                        e_hb_l += e_hb;

                        CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                        CEhb2 = -0.5 * hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                        CEhb3 = hbp->p_hb3 * e_hb * (hbp->r0_hb / SQR( r_jk )
                                + -1.0 / hbp->r0_hb);

                        /* hydrogen bond forces */
                        /* dbo term */
                        bo_ij->Cdbo += CEhb1;

                        /* dcos terms */
#if !defined(CUDA_ACCUM_ATOMIC)
                        rvec_ScaledAdd( pbond_ij->hb_f, CEhb2, dcos_theta_di ); 
#else
                        atomic_rvecScaledAdd( workspace.f[i], CEhb2, dcos_theta_di );
#endif
                        rvec_ScaledAdd( f_j_l, CEhb2, dcos_theta_dj );
                        rvec_ScaledAdd( f_k_l, CEhb2, dcos_theta_dk );

                        /* dr terms */
                        rvec_ScaledAdd( f_j_l, -1.0 * CEhb3 / r_jk, dvec_jk ); 
                        rvec_ScaledAdd( f_k_l, CEhb3 / r_jk, dvec_jk );
                    }
                }

                pi += warpSize;
            }

            f_k_l[0] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_k_l[0]);
            f_k_l[1] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_k_l[1]);
            f_k_l[2] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_k_l[2]);

            if ( lane_id == 0 )
            {
#if !defined(CUDA_ACCUM_ATOMIC)
                rvec_Copy( phbond_jk->hb_f, f_k_l );
#else
                atomic_rvecAdd( workspace.f[k], f_k_l );
#endif
            }
        }

        f_j_l[0] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_j_l[0]);
        f_j_l[1] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_j_l[1]);
        f_j_l[2] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_j_l[2]);
        e_hb_l = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(e_hb_l);

        if ( lane_id == 0 )
        {
#if !defined(CUDA_ACCUM_ATOMIC)
            /* write conflicts for accumulating partial forces resolved by subsequent kernels */
            rvecCopy( workspace.f[j], f_j_l );
            e_hb_g[j] = e_hb_l;
#else
            atomic_rvecAdd( workspace.f[j], f_j_l );
            atomicAdd( (double *) e_hb_g, (double) e_hb_l );
#endif
        }
    }
}


/* one thread per atom implementation */
CUDA_GLOBAL void k_hydrogen_bonds_virial_part1( reax_atom *my_atoms, single_body_parameters *sbp, 
        hbond_parameters *d_hbp, global_parameters gp,
        control_params *control, storage workspace,
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real *e_hb_g, rvec *ext_press_g )
{
    int i, j, k, pi, pk;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int *hblist, hblist_size;
    int itr, top;
    int nbr_jk;
    ivec rel_jk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_l, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk, temp, ext_press_l;
#if defined(CUDA_ACCUM_ATOMIC)
    rvec f_j_l;
#endif
    hbond_parameters *hbp;
    bond_order_data *bo_ij;
    bond_data *pbond_ij;
    hbond_data *phbond_jk;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    e_hb_l = 0.0;
    rvec_MakeZero( ext_press_l );
#if defined(CUDA_ACCUM_ATOMIC)
    rvec_MakeZero( f_j_l );
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

#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_MakeZero( phbond_jk->hb_f );
#endif

            /* find matching hbond to atoms j and k */
            for ( itr = 0; itr < top; ++itr )
            {
                pi = hblist[itr];
                pbond_ij = &bond_list.bond_list[pi];
                i = pbond_ij->nbr;

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id )
                {
                    bo_ij = &pbond_ij->bo_data;
                    type_i = my_atoms[i].type;
                    r_ij = pbond_ij->d;
                    hbp = &d_hbp[ index_hbp(type_i, type_j, type_k, num_atom_types) ];

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
                    exp_hb2 = EXP( -1.0 * hbp->p_hb2 * bo_ij->BO );
                    exp_hb3 = EXP( -1.0 * hbp->p_hb3 * ( hbp->r0_hb / r_jk
                                + r_jk / hbp->r0_hb - 2.0 ) );

                    e_hb = hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                    e_hb_l += e_hb;

                    CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                    CEhb2 = -0.5 * hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                    CEhb3 = hbp->p_hb3 * e_hb * (hbp->r0_hb / SQR( r_jk )
                            + -1.0 / hbp->r0_hb);

                    /* hydrogen bond forces */
                    /* dbo term */
                    bo_ij->Cdbo += CEhb1;

#if !defined(CUDA_ACCUM_ATOMIC)
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

                    rvec_ScaledAdd( f_j_l, CEhb2, dcos_theta_dj );

                    ivec_Scale( rel_jk, hbond_list.hbond_list[pk].scl,
                            far_nbr_list.far_nbr_list.rel_box[nbr_jk] );
                    rvec_Scale( temp, CEhb2, dcos_theta_dk );
                    atomic_rvecAdd( workspace.f[k], temp );
                    rvec_iMultiply( temp, rel_jk, temp );
                    rvec_Add( ext_press_l, temp );

                    /* dr terms */
                    rvec_ScaledAdd( f_j_l, -1.0 * CEhb3 / r_jk, dvec_jk ); 

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

#if !defined(CUDA_ACCUM_ATOMIC)
    /* write conflicts for accumulating partial forces resolved by subsequent kernels */
    rvecCopy( workspace.f[j], f_j_l );
    e_hb_g[j] = e_hb_l;
    rvecCopy( ext_press_g[j], ext_press_l );
#else
    atomic_rvecAdd( workspace.f[j], f_j_l );
    atomicAdd( (double *) e_hb_g, (double) e_hb_l );
    atomic_rvecAdd( *ext_press_g, ext_press_l );
#endif
}


#if !defined(CUDA_ACCUM_ATOMIC)
/* Accumulate forces stored in the bond list
 * using a one thread per atom implementation */
CUDA_GLOBAL void k_hydrogen_bonds_part2( reax_atom *atoms,
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
CUDA_GLOBAL void k_hydrogen_bonds_part2_opt( reax_atom *atoms,
        storage workspace, reax_list bond_list, int n )
{
    typedef cub::WarpReduce<double> WarpReduce;
    extern __shared__ typename WarpReduce::TempStorage temp_storage[];
    int j, pj, start, end;
    bond_data *pbond, *sym_index_bond;
    /* thread-local variables */
    int thread_id, warp_id, lane_id;
    rvec hb_f;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id >> 5;
    lane_id = thread_id & 0x0000001F; 

    if ( warp_id >= n )
    {
        return;
    }

    j = warp_id;
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

    __syncthreads( );

    hb_f[0] = WarpReduce(temp_storage[warp_id]).Sum(hb_f[0]);
    hb_f[1] = WarpReduce(temp_storage[warp_id]).Sum(hb_f[1]);
    hb_f[2] = WarpReduce(temp_storage[warp_id]).Sum(hb_f[2]);

    /* first thread within a warp writes warp-level sums to global memory */
    if ( lane_id == 0 )
    {
        rvec_Add( workspace.f[j], hb_f );
    }
}


/* Accumulate forces stored in the hbond list
 * using a one thread per atom implementation */
CUDA_GLOBAL void k_hydrogen_bonds_part3( reax_atom *atoms,
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
CUDA_GLOBAL void k_hydrogen_bonds_part3_opt( reax_atom *atoms,
        storage workspace, reax_list hbond_list, int n )
{
    typedef cub::WarpReduce<double> WarpReduce;
    extern __shared__ typename WarpReduce::TempStorage temp_storage[];
    int j, pj, start, end;
    hbond_data *nbr_pj, *sym_index_nbr;
    /* thread-local variables */
    int thread_id, warp_id, lane_id, offset;
    rvec hb_f_l;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id >> 5;
    lane_id = thread_id & 0x0000001F; 

    if ( warp_id >= n )
    {
        return;
    }

    j = warp_id;
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

    __syncthreads( );

    hb_f_l[0] = WarpReduce(temp_storage[warp_id]).Sum(hb_f_l[0]);
    hb_f_l[1] = WarpReduce(temp_storage[warp_id]).Sum(hb_f_l[1]);
    hb_f_l[2] = WarpReduce(temp_storage[warp_id]).Sum(hb_f_l[2]);

    /* first thread within a warp writes warp-level sums to global memory */
    if ( lane_id == 0 )
    {
        rvec_Add( workspace.f[j], hb_f_l );
    }
}
#endif



void Cuda_Compute_Hydrogen_Bonds( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control )
{
    int blocks;
//    int hbs, hnbrs_blocks;
#if !defined(CUDA_ACCUM_ATOMIC)
    int update_energy;
    real *spad;
    rvec *rvec_spad;

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
                (sizeof(real) * 3 + sizeof(rvec)) * system->N + sizeof(rvec) * control->blocks_n,
            "Cuda_Compute_Hydrogen_Bonds::workspace->scratch" );
    spad = (real *) workspace->scratch;
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#else
    cuda_memset( &((simulation_data *)data->d_simulation_data)->my_en.e_hb,
            0, sizeof(real), "Cuda_Compute_Hydrogen_Bonds::e_hb" );
    if ( control->virial == 1 )
    {
        cuda_memset( &((simulation_data *)data->d_simulation_data)->my_ext_press,
                0, sizeof(rvec), "Cuda_Compute_Hydrogen_Bonds::my_ext_press" );
    }
#endif

    if ( control->virial == 1 )
    {
        k_hydrogen_bonds_virial_part1 <<< control->blocks, control->block_size,
                                      0, control->streams[0] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp,
                  system->reax_param.d_hbp, system->reax_param.d_gp,
                  (control_params *) control->d_control_params,
                  *(workspace->d_workspace),
                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
                  system->n, system->reax_param.num_atom_types,
#if !defined(CUDA_ACCUM_ATOMIC)
                  spad, (rvec *) (&spad[system->n])
#else
                  &((simulation_data *)data->d_simulation_data)->my_en.e_hb,
                  &((simulation_data *)data->d_simulation_data)->my_ext_press
#endif
                );
        cudaCheckError( );
    }
    else
    {
//        k_hydrogen_bonds_part1 <<< control->blocks, control->block_size, 0, control->streams[0] >>>
//                ( system->d_my_atoms, system->reax_param.d_sbp,
//                  system->reax_param.d_hbp, system->reax_param.d_gp,
//                  (control_params *) control->d_control_params,
//                  *(workspace->d_workspace),
//                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
//                  system->n, system->reax_param.num_atom_types,
//#if !defined(CUDA_ACCUM_ATOMIC)
//                  spad
//#else
//                  &((simulation_data *)data->d_simulation_data)->my_en.e_hb
//#endif
//                );
//        cudaCheckError( );

        blocks = system->n * 32 / DEF_BLOCK_SIZE
            + (system->n * 32 % DEF_BLOCK_SIZE == 0 ? 0 : 1);
        
        k_hydrogen_bonds_part1_opt <<< blocks, DEF_BLOCK_SIZE,
                                   sizeof(cub::WarpReduce<double>::TempStorage) * (DEF_BLOCK_SIZE / 32),
                                   control->streams[0] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp,
                  system->reax_param.d_hbp, system->reax_param.d_gp,
                  (control_params *) control->d_control_params,
                  *(workspace->d_workspace),
                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
                  system->n, system->reax_param.num_atom_types,
#if !defined(CUDA_ACCUM_ATOMIC)
                  spad
#else
                  &((simulation_data *)data->d_simulation_data)->my_en.e_hb
#endif
                );
        cudaCheckError( );
    }

#if !defined(CUDA_ACCUM_ATOMIC)
    if ( update_energy == TRUE )
    {
        Cuda_Reduction_Sum( spad,
                &((simulation_data *)data->d_simulation_data)->my_en.e_hb,
                system->n );
    }

    if ( control->virial == 1 )
    {
        rvec_spad = (rvec *) (&spad[system->n]);

        k_reduction_rvec <<< control->blocks, control->block_size,
                         sizeof(rvec) * (control->block_size / 32),
                         control->streams[0] >>>
            ( rvec_spad, &rvec_spad[system->n], system->n );
        cudaCheckError( );

        k_reduction_rvec <<< 1, control->blocks_pow_2,
                         sizeof(rvec) * (control->blocks_pow_2 / 32),
                         control->streams[0] >>>
            ( &rvec_spad[system->n],
              &((simulation_data *)data->d_simulation_data)->my_ext_press,
              control->blocks );
        cudaCheckError( );
//        Cuda_Reduction_Sum( rvec_spad,
//                &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                system->n );
    }
#endif

#if !defined(CUDA_ACCUM_ATOMIC)
    k_hydrogen_bonds_part2 <<< control->blocks, control->block_size, 0,
                           control->streams[0] >>>
        ( system->d_my_atoms, *(workspace->d_workspace),
          *(lists[BONDS]), system->n );
    cudaCheckError( );

//    hnbrs_blocks = (system->n * HB_POST_PROC_KER_THREADS_PER_ATOM / HB_POST_PROC_BLOCK_SIZE) +
//        (((system->n * HB_POST_PROC_KER_THREADS_PER_ATOM) % HB_POST_PROC_BLOCK_SIZE) == 0 ? 0 : 1);

    k_hydrogen_bonds_part3 <<< control->blocks, control->block_size, 0,
                           control->streams[0] >>>
        ( system->d_my_atoms, *(workspace->d_workspace), *(lists[HBONDS]), system->n );
//    k_hydrogen_bonds_part3_opt <<< hnbrs_blocks, HB_POST_PROC_BLOCK_SIZE, 
//            sizeof(rvec) * HB_POST_PROC_BLOCK_SIZE, control->streams[0] >>>
//        ( system->d_my_atoms, *(workspace->d_workspace), *(lists[HBONDS]), system->n );
    cudaCheckError( );
#endif
}
