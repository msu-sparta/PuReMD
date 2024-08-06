#include "hip/hip_runtime.h"
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

#include "gpu_hydrogen_bonds.h"

#include "gpu_valence_angles.h"
#include "gpu_helpers.h"
#include "gpu_list.h"
#if !defined(GPU_ATOMIC_EV)
  #include "gpu_reduction.h"
#endif
#include "gpu_utils.h"

#include "../index_utils.h"
#include "../vector.h"

#include <hipcub/warp/warp_reduce.hpp>


/* one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part1( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp,
        hbond_parameters const * const hbp,
        rvec * const f, reax_list far_nbr_list,
        reax_list bond_list, reax_list hbond_list, int n, 
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
#define BL (bond_list.bond_list_gpu)

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    e_hb_ = 0.0;

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
        rvec_MakeZero( f_j );

        if ( Num_Entries( j, &bond_list ) > hblist_size )
        {
            hblist_size = Num_Entries( j, &bond_list );
            hblist = (int *) malloc( sizeof(int) * hblist_size );
        }

        /* search bonded atoms i to atom j (hydrogen atom) for potential hydrogen bonding */
        for ( pi = start_j; pi < end_j; ++pi )
        {
            i = BL.nbr[pi];
            type_i = my_atoms[i].type;

            if ( sbp[type_i].p_hbond == H_BONDING_ATOM
                    && BL.BO[pi] >= HB_THRESHOLD )
            {
                hblist[top] = pi;
                ++top;
            }
        }

        /* for each hbond of atom j */
        for ( pk = hb_start_j; pk < hb_end_j; ++pk )
        {
            k = hbond_list.hbond_list.nbr[pk];
            type_k = my_atoms[k].type;
            nbr_jk = hbond_list.hbond_list.ptr[pk];
            r_jk = far_nbr_list.far_nbr_list.d[nbr_jk];

            rvec_MakeZero( f_k );

            rvec_Scale( dvec_jk, hbond_list.hbond_list.scl[pk],
                    far_nbr_list.far_nbr_list.dvec[nbr_jk] );

            /* find matching hbond to atoms j and k */
            for ( itr = 0; itr < top; ++itr )
            {
                pi = hblist[itr];
                i = BL.nbr[pi];

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id
                        && hbp[ index_hbp(my_atoms[i].type, type_j, type_k, num_atom_types) ].is_valid == TRUE )
                {
                    type_i = my_atoms[i].type;
                    r_ij = BL.d[pi];
                    hbp_ijk = index_hbp(type_i, type_j, type_k, num_atom_types);

                    Calculate_Theta( BL.dvec[pi], r_ij, dvec_jk, r_jk,
                            &theta, &cos_theta );

                    /* the derivative of cos(theta) */
                    Calculate_dCos_Theta( BL.dvec[pi], r_ij, dvec_jk, r_jk,
                            &dcos_theta_di, &dcos_theta_dj, &dcos_theta_dk );

                    /* hydrogen bond energy */
                    sin_theta2 = SIN( theta / 2.0 );
                    sin_xhz4 = SQR( sin_theta2 );
                    sin_xhz4 *= sin_xhz4;
                    cos_xhz1 = ( 1.0 - cos_theta );
                    exp_hb2 = EXP( -1.0 * hbp[hbp_ijk].p_hb2 * BL.BO[pi] );
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
#if defined(GPU_STREAM_SINGLE_ACCUM)
                    atomicAdd( &BL.Cdbo[pi], CEhb1 );
#else
                    BL.Cdbo_hbonds[pi] += CEhb1;
#endif

                    /* dcos terms */
#if defined(GPU_KERNEL_ATOMIC)
                    atomic_rvecScaledAdd( f[i], CEhb2, dcos_theta_di );
#else
                    rvec_ScaledAdd( BL.f_hb[pi], CEhb2, dcos_theta_di ); 
#endif
                    rvec_ScaledAdd( f_j, CEhb2, dcos_theta_dj );
                    rvec_ScaledAdd( f_k, CEhb2, dcos_theta_dk );

                    /* dr terms */
                    rvec_ScaledAdd( f_j, -1.0 * CEhb3 / r_jk, dvec_jk ); 
                    rvec_ScaledAdd( f_k, CEhb3 / r_jk, dvec_jk );
                }
            }

//#if defined(GPU_KERNEL_ATOMIC)
            atomic_rvecAdd( f[k], f_k );
//#else
//            rvec_Copy( hbond_list.hbond_list.f_hb[pk], f_k );
//#endif
        }

        if ( hblist != NULL )
        {
            free( hblist );
        }

#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
        atomic_rvecAdd( f[j], f_j );
#else
        rvec_Copy( f[j], f_j );
#endif
#if defined(GPU_ATOMIC_EV)
        atomicAdd( (double *) e_hb_g, (double) e_hb_ );
    }
#else
    }

    e_hb_g[j] = e_hb_;
#endif

#undef BL
}


/* one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part1_opt( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp,
        hbond_parameters const * const hbp,
        rvec * const f, reax_list far_nbr_list,
        reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real * const e_hb_g )
{
    extern __shared__ hipcub::WarpReduce<double>::TempStorage temp_d[];
    int i, j, k, pi, pk, thread_id, warp_id, lane_id, itr;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int nbr_jk, hbp_ijk;
    real r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk;
    rvec f_j, f_k;
#define BL (bond_list.bond_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    j = thread_id / warpSize;

    if ( j >= n )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize; 
    e_hb_ = 0.0;

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
        rvec_MakeZero( f_j );

        /* for each hbond of atom j */
        for ( pk = hb_start_j; pk < hb_end_j; ++pk )
        {
            k = hbond_list.hbond_list.nbr[pk];
            type_k = my_atoms[k].type;
            nbr_jk = hbond_list.hbond_list.ptr[pk];
            r_jk = far_nbr_list.far_nbr_list.d[nbr_jk];

            rvec_MakeZero( f_k );

            rvec_Scale( dvec_jk, hbond_list.hbond_list.scl[pk],
                    far_nbr_list.far_nbr_list.dvec[nbr_jk] );

            /* search bonded atoms i to atom j (hydrogen atom) for potential hydrogen bonding */
            for ( itr = 0, pi = start_j + lane_id; itr < (end_j - start_j + warpSize - 1) / warpSize; ++itr )
            {
                if ( pi < end_j )
                {
                    i = BL.nbr[pi];
                    type_i = my_atoms[i].type;

                    if ( sbp[type_i].p_hbond == H_BONDING_ATOM
                            && BL.BO[pi] >= HB_THRESHOLD
                            && my_atoms[i].orig_id != my_atoms[k].orig_id )
                    {
                        hbp_ijk = index_hbp(type_i, type_j, type_k, num_atom_types);

                        Calculate_Theta( BL.dvec[pi], BL.d[pi], dvec_jk, r_jk,
                                &theta, &cos_theta );

                        /* the derivative of cos(theta) */
                        Calculate_dCos_Theta( BL.dvec[pi], BL.d[pi], dvec_jk, r_jk,
                                &dcos_theta_di, &dcos_theta_dj, &dcos_theta_dk );

                        /* hydrogen bond energy */
                        sin_theta2 = SIN( theta / 2.0 );
                        sin_xhz4 = SQR( sin_theta2 );
                        sin_xhz4 *= sin_xhz4;
                        cos_xhz1 = ( 1.0 - cos_theta );
                        exp_hb2 = EXP( -1.0 * hbp[hbp_ijk].p_hb2 * BL.BO[pi] );
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
#if defined(GPU_STREAM_SINGLE_ACCUM)
                        atomicAdd( &BL.Cdbo[pi], CEhb1 );
#else
                        BL.Cdbo_hbonds[pi] += CEhb1;
#endif

                        /* dcos terms */
#if defined(GPU_KERNEL_ATOMIC)
                        atomic_rvecScaledAdd( f[i], CEhb2, dcos_theta_di );
#else
                        rvec_ScaledAdd( BL.f_hb[pi], CEhb2, dcos_theta_di ); 
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

            f_k[0] = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[0]);
            f_k[1] = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[1]);
            f_k[2] = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[2]);

            if ( lane_id == 0 )
            {
//#if defined(GPU_KERNEL_ATOMIC)
                atomic_rvecAdd( f[k], f_k );
//#else
//                rvec_Copy( hbond_list.hbond_list.f_hb[pk], f_k );
//#endif
            }
        }

        f_j[0] = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[0]);
        f_j[1] = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[1]);
        f_j[2] = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[2]);
        e_hb_ = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(e_hb_);

        if ( lane_id == 0 )
        {
#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
            atomic_rvecAdd( f[j], f_j );
#else
            rvec_Copy( f[j], f_j );
#endif
#if defined(GPU_ATOMIC_EV)
            atomicAdd( (double *) e_hb_g, (double) e_hb_ );
        }
    }
#else
        }
    }

    if ( lane_id == 0 )
    {
        e_hb_g[j] = e_hb_;
    }
#endif

#undef BL
}


/* one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_virial_part1( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp,
        hbond_parameters const * const hbp, rvec * const f, reax_list far_nbr_list,
        reax_list bond_list, reax_list hbond_list, int n, 
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
    rvec dvec_jk, temp, ext_press_l, f_j;
#define BL (bond_list.bond_list_gpu)

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    e_hb_ = 0.0;
    rvec_MakeZero( ext_press_l );
    rvec_MakeZero( f_j );

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
            i = BL.nbr[pi];
            type_i = my_atoms[i].type;

            if ( sbp[type_i].p_hbond == H_BONDING_ATOM
                    && BL.BO[pi] >= HB_THRESHOLD )
            {
                hblist[top] = pi;
                ++top;
            }
        }

        /* for each hbond of atom j */
        for ( pk = hb_start_j; pk < hb_end_j; ++pk )
        {
            k = hbond_list.hbond_list.nbr[pk];
            type_k = my_atoms[k].type;
            nbr_jk = hbond_list.hbond_list.ptr[pk];
            r_jk = far_nbr_list.far_nbr_list.d[nbr_jk];

            rvec_Scale( dvec_jk, hbond_list.hbond_list.scl[pk],
                    far_nbr_list.far_nbr_list.dvec[nbr_jk] );

            /* find matching hbond to atoms j and k */
            for ( itr = 0; itr < top; ++itr )
            {
                pi = hblist[itr];
                i = BL.nbr[pi];

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id
                        && hbp[ index_hbp(my_atoms[i].type, type_j, type_k, num_atom_types) ].is_valid == TRUE )
                {
                    type_i = my_atoms[i].type;
                    r_ij = BL.d[pi];
                    hbp_ijk = index_hbp(type_i, type_j, type_k, num_atom_types);

                    Calculate_Theta( BL.dvec[pi], r_ij, dvec_jk, r_jk,
                            &theta, &cos_theta );

                    /* the derivative of cos(theta) */
                    Calculate_dCos_Theta( BL.dvec[pi], r_ij, dvec_jk, r_jk,
                            &dcos_theta_di, &dcos_theta_dj, &dcos_theta_dk );

                    /* hydrogen bond energy */
                    sin_theta2 = SIN( theta / 2.0 );
                    sin_xhz4 = SQR( sin_theta2 );
                    sin_xhz4 *= sin_xhz4;
                    cos_xhz1 = ( 1.0 - cos_theta );
                    exp_hb2 = EXP( -1.0 * hbp[hbp_ijk].p_hb2 * BL.BO[pi] );
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
#if defined(GPU_STREAM_SINGLE_ACCUM)
                    atomicAdd( &BL.Cdbo[pi], CEhb1 );
#else
                    BL.Cdbo_hbonds[pi] += CEhb1;
#endif

                    /* for pressure coupling, terms that are not related to bond order
                     * derivatives are added directly into pressure vector/tensor */
                    /* dcos terms */
                    rvec_Scale( temp, CEhb2, dcos_theta_di );
#if defined(GPU_KERNEL_ATOMIC)
                    atomic_rvecAdd( f[i], temp );
#else
                    rvec_Add( BL.f_hb[pi], temp );
#endif
                    rvec_iMultiply( temp, BL.rel_box[pi], temp );
                    rvec_Add( ext_press_l, temp );

                    rvec_ScaledAdd( f_j, CEhb2, dcos_theta_dj );

                    ivec_Scale( rel_jk, hbond_list.hbond_list.scl[pk],
                            far_nbr_list.far_nbr_list.rel_box[nbr_jk] );
                    rvec_Scale( temp, CEhb2, dcos_theta_dk );
//#if defined(GPU_KERNEL_ATOMIC)
                    atomic_rvecAdd( f[k], temp );
//#else
//                    rvec_Add( hbond_list.hbond_list.f_hb[pk], temp );
//#endif
                    rvec_iMultiply( temp, rel_jk, temp );
                    rvec_Add( ext_press_l, temp );

                    /* dr terms */
                    rvec_ScaledAdd( f_j, -1.0 * CEhb3 / r_jk, dvec_jk ); 

                    rvec_Scale( temp, CEhb3 / r_jk, dvec_jk );
//#if defined(GPU_KERNEL_ATOMIC)
                    atomic_rvecAdd( f[k], temp );
//#else
//                    rvec_Add( hbond_list.hbond_list.f_hb[pk], temp );
//#endif
                    rvec_iMultiply( temp, rel_jk, temp );
                    rvec_Add( ext_press_l, temp );
                }
            }
        }

        if ( hblist != NULL )
        {
            free( hblist );
        }
    }

#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
    atomic_rvecAdd( f[j], f_j );
#else
    rvec_Add( f[j], f_j );
#endif
#if defined(GPU_ATOMIC_EV)
    atomicAdd( (double *) e_hb_g, (double) e_hb_ );
    atomic_rvecAdd( *ext_press_g, ext_press_l );
#else
    e_hb_g[j] = e_hb_;
    rvec_Copy( ext_press_g[j], ext_press_l );
#endif

#undef BL
}


#if !defined(GPU_KERNEL_ATOMIC)
/* Accumulate forces stored in the bond list
 * using a one thread per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part2( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, rvec * const f,
        reax_list bond_list, int n )
{
    int j, pj;
#define BL (bond_list.bond_list_gpu)

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    if ( sbp[my_atoms[j].type].p_hbond == H_ATOM )
    {
        for ( pj = Start_Index(j, &bond_list); pj < End_Index(j, &bond_list); ++pj )
        {
            if ( sbp[my_atoms[BL.nbr[pj]].type].p_hbond == H_BONDING_ATOM )
            {
                atomic_rvecAdd( f[BL.nbr[pj]], BL.f_hb[pj] );
            }
        }
    }

#undef BL
}


/* Accumulate forces stored in the bond list
 * using a one warp threads per atom implementation */
GPU_GLOBAL void k_hydrogen_bonds_part2_opt( reax_atom const * const my_atoms,
        single_body_parameters const * const sbp, rvec * const f,
        reax_list bond_list, int n )
{
    int j, pj, start, end, thread_id, lane_id;
#define BL (bond_list.bond_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    j = thread_id / warpSize;

    if ( j >= n )
    {
        return;
    }

    if ( sbp[my_atoms[j].type].p_hbond == H_ATOM )
    {
        lane_id = thread_id % warpSize; 
        start = Start_Index( j, &bond_list );
        end = End_Index( j, &bond_list );
        pj = start + lane_id;

        while ( pj < end )
        {
            if ( sbp[my_atoms[BL.nbr[pj]].type].p_hbond == H_BONDING_ATOM )
            {
                atomic_rvecAdd( f[BL.nbr[pj]], BL.f_hb[pj] );
            }

            pj += warpSize;
        }
    }

#undef BL
}


///* Accumulate forces stored in the hbond list
// * using a one thread per atom implementation */
//GPU_GLOBAL void k_hydrogen_bonds_part3( reax_atom const * const my_atoms,
//        single_body_parameters const * const sbp, rvec * const f,
//        reax_list hbond_list, int n )
//{
//    int j, pj;
//    rvec f_hb;
//#define HBL (hbond_list.hbond_list)
//
//    j = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if ( j >= n )
//    {
//        return;
//    }
//
//    if ( sbp[ my_atoms[j].type ].p_hbond == H_ATOM )
//    {
//        rvec_MakeZero( f_hb );
//
//        for ( pj = Start_Index(my_atoms[j].Hindex, &hbond_list);
//                pj < End_Index(my_atoms[j].Hindex, &hbond_list); ++pj )
//        {
//            rvec_Add( f_hb, HBL.f_hb[HBL.sym_index[pj]] );
//        }
//
//        rvec_Add( f[j], f_hb );
//    }
//
//#undef HBL
//}
//
//
///* Accumulate forces stored in the hbond list
// * using a one warp of threads per atom implementation */
//GPU_GLOBAL void k_hydrogen_bonds_part3_opt( reax_atom const * const my_atoms,
//        rvec * const f, reax_list hbond_list, int n )
//{
//    extern __shared__ hipcub::WarpReduce<double>::TempStorage temp_storage[];
//    int j, pj, start, end;
//    /* thread-local variables */
//    int thread_id, warp_id, lane_id;
//    rvec f_hb;
//#define HBL (hbond_list.hbond_list)
//
//    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
//    j = thread_id / warpSize;
//
//    if ( j >= n )
//    {
//        return;
//    }
//
//    warp_id = threadIdx.x / warpSize;
//    lane_id = thread_id % warpSize; 
//    start = Start_Index( my_atoms[j].Hindex, &hbond_list );
//    end = End_Index( my_atoms[j].Hindex, &hbond_list );
//    pj = start + lane_id;
//    rvec_MakeZero( f_hb );
//
//    while ( pj < end )
//    {
//        rvec_Add( f_hb, HBL.f_hb[HBL.sym_index[pj]] );
//
//        pj += warpSize;
//    }
//
//    f_hb[0] = hipcub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_hb[0]);
//    f_hb[1] = hipcub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_hb[1]);
//    f_hb[2] = hipcub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_hb[2]);
//
//    /* first thread within a warp writes warp-level sums to global memory */
//    if ( lane_id == 0 )
//    {
//        rvec_Add( f[j], f_hb );
//    }
//
//#undef HBL
//}
#endif



void GPU_Compute_Hydrogen_Bonds( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list ** lists,
        output_controls const * const out_control )
{
//    int hbs, hnbrs_blocks;
#if !defined(GPU_ATOMIC_EV)
    int update_energy;
    real *spad;
    rvec *rvec_spad;
#endif

#if defined(LOG_PERFORMANCE)
    hipEventRecord( control->gpu_time_events[TE_HBONDS_START], control->gpu_streams[2] );
#endif

#if defined(GPU_ATOMIC_EV)
    sHipMemsetAsync( &data->d_my_en[E_HB], 0, sizeof(real),
            control->gpu_streams[2], __FILE__, __LINE__ );
    if ( control->virial == 1 )
    {
        sHipMemsetAsync( &data->d_my_ext_press, 0, sizeof(rvec),
                control->gpu_streams[2], __FILE__, __LINE__ );
    }
#else
    sHipCheckMalloc( &workspace->d_workspace->scratch[2],
            &workspace->d_workspace->scratch_size[2],
            (sizeof(real) * 3 + sizeof(rvec)) * system->N + sizeof(rvec) * control->blocks_N,
            __FILE__, __LINE__ );
    spad = (real *) workspace->d_workspace->scratch[2];
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#endif

    hipStreamWaitEvent( control->gpu_streams[2], control->gpu_stream_events[SE_BOND_ORDER_DONE], 0 );

    if ( control->virial == 0 )
    {
//        k_hydrogen_bonds_part1 <<< control->blocks_n, control->gpu_block_size, 0, control->gpu_streams[2] >>>
//                ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_hbp,
//#if defined(GPU_STREAM_SINGLE_ACCUM)
//                  workspace->d_workspace->f,
//#else
//                  workspace->d_workspace->f_hb,
//#endif
//                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
//                  system->n, system->reax_param.num_atom_types,
//#if defined(GPU_ATOMIC_EV)
//                  &data->d_my_en[E_HB]
//#else
//                  spad
//#endif
//                );
        
        k_hydrogen_bonds_part1_opt <<< control->blocks_warp_n, control->gpu_block_size,
                                   sizeof(hipcub::WarpReduce<double>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                                   control->gpu_streams[2] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_hbp,
#if defined(GPU_STREAM_SINGLE_ACCUM)
                  workspace->d_workspace->f,
#else
                  workspace->d_workspace->f_hb,
#endif
                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
                  system->n, system->reax_param.num_atom_types,
#if defined(GPU_ATOMIC_EV)
                  &data->d_my_en[E_HB]
#else
                  spad
#endif
                );
    }
    else if ( control->virial == 1 )
    {
        k_hydrogen_bonds_virial_part1 <<< control->blocks_n, control->gpu_block_size,
                                      0, control->gpu_streams[2] >>>
                ( system->d_my_atoms, system->reax_param.d_sbp, system->reax_param.d_hbp,
#if defined(GPU_STREAM_SINGLE_ACCUM)
                  workspace->d_workspace->f,
#else
                  workspace->d_workspace->f_hb,
#endif
                  *(lists[FAR_NBRS]), *(lists[BONDS]), *(lists[HBONDS]),
                  system->n, system->reax_param.num_atom_types,
#if defined(GPU_ATOMIC_EV)
                  &data->d_my_en[E_HB], &data->d_my_ext_press
#else
                  spad, (rvec *) (&spad[system->n])
#endif
                );
    }
    hipCheckError( );

#if !defined(GPU_ATOMIC_EV)
    if ( update_energy == TRUE )
    {
        GPU_Reduction_Sum( spad, &data->d_my_en[E_HB], system->n, 2,
                control->gpu_streams[2] );
    }

    if ( control->virial == 1 )
    {
        rvec_spad = (rvec *) (&spad[system->n]);

        GPU_Reduction_Sum( rvec_spad, &data->d_my_ext_press, system->n, 2,
                control->gpu_streams[2] );
    }
#endif

#if !defined(GPU_KERNEL_ATOMIC)
//    k_hydrogen_bonds_part2 <<< control->blocks_n, control->gpu_block_size,
//                           0, control->gpu_streams[2] >>>
//        ( system->d_my_atoms, system->reax_param.d_sbp,
//#if defined(GPU_STREAM_SINGLE_ACCUM)
//          workspace->d_workspace->f,
//#else
//          workspace->d_workspace->f_hb,
//#endif
//          *(lists[BONDS]), system->n );
//    hipCheckError( );

    k_hydrogen_bonds_part2_opt <<< control->blocks_warp_n, control->gpu_block_size,
                           0, control->gpu_streams[2] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp,
#if defined(GPU_STREAM_SINGLE_ACCUM)
          workspace->d_workspace->f,
#else
          workspace->d_workspace->f_hb,
#endif
          *(lists[BONDS]), system->n );
    hipCheckError( );

////    k_hydrogen_bonds_part3 <<< control->blocks_n, control->gpu_block_size,
////                           0, control->gpu_streams[2] >>>
////        ( system->d_my_atoms, system->reax_param.d_sbp,
////#if defined(GPU_STREAM_SINGLE_ACCUM)
////          workspace->d_workspace->f,
////#else
////          workspace->d_workspace->f_hb,
////#endif
////          *(lists[HBONDS]), system->n );
////
//////    k_hydrogen_bonds_part3_opt <<< control->blocks_warp_n, control->gpu_block_size, 
//////                               sizeof(rvec) * (control->gpu_block_size / WARP_SIZE),
//////                               control->gpu_streams[2] >>>
//////        ( system->d_my_atoms,
//////#if defined(GPU_STREAM_SINGLE_ACCUM)
//////          workspace->d_workspace->f,
//////#else
//////          workspace->d_workspace->f_hb,
//////#endif
//////          *(lists[HBONDS]), system->n );
////    hipCheckError( );
#endif

#if defined(LOG_PERFORMANCE)
    hipEventRecord( control->gpu_time_events[TE_HBONDS_STOP], control->gpu_streams[2] );
#endif
}
