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

#include "../index_utils.h"
#include "../vector.h"


/* mask used to determine which threads within a warp participate in operations */
#define FULL_MASK (0xFFFFFFFF)


/* one thread per atom implementation */
CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part1( reax_atom *my_atoms, single_body_parameters *sbp, 
        hbond_parameters *d_hbp, global_parameters gp,
        control_params *control, storage workspace,
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real *e_hb_g, rvec *ext_press_g, int rank )
{
    int i, j, k, pi, pk;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    int *hblist, hblist_size;
    int itr, top;
    ivec rel_jk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_l, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk, temp, ext_press_l;
#if defined(CUDA_ACCUM_ATOMIC)
    rvec f_i_l, f_j_l, f_k_l;
#endif
    hbond_parameters *hbp;
    bond_order_data *bo_ij;
    int nbr_jk;
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
    rvec_MakeZero( f_i_l );
    rvec_MakeZero( f_j_l );
    rvec_MakeZero( f_k_l );
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

                    if ( control->virial == 0 )
                    {
#if !defined(CUDA_ACCUM_ATOMIC)
                        /* dcos terms */
                        rvec_ScaledAdd( pbond_ij->hb_f, CEhb2, dcos_theta_di ); 
                        rvec_ScaledAdd( workspace.f[j], CEhb2, dcos_theta_dj );
                        rvec_ScaledAdd( phbond_jk->hb_f, CEhb2, dcos_theta_dk );

                        /* dr terms */
                        rvec_ScaledAdd( workspace.f[j], -1.0 * CEhb3 / r_jk, dvec_jk ); 
                        rvec_ScaledAdd( phbond_jk->hb_f, CEhb3 / r_jk, dvec_jk );
#else
                        /* dcos terms */
                        rvec_ScaledAdd( f_i_l, CEhb2, dcos_theta_di );
                        rvec_ScaledAdd( f_j_l, CEhb2, dcos_theta_dj );
                        rvec_ScaledAdd( f_k_l, CEhb2, dcos_theta_dk );

                        /* dr terms */
                        rvec_ScaledAdd( f_j_l, -1.0 * CEhb3 / r_jk, dvec_jk );
                        rvec_ScaledAdd( f_k_l, CEhb3 / r_jk, dvec_jk );
#endif
                    }
                    else
                    {
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
                        rvec_Add( f_i_l, temp );
                        rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                        rvec_Add( ext_press_l, temp );

                        rvec_ScaledAdd( f_j_l, CEhb2, dcos_theta_dj );

                        ivec_Scale( rel_jk, hbond_list.hbond_list[pk].scl,
                                far_nbr_list.far_nbr_list.rel_box[nbr_jk] );
                        rvec_Scale( temp, CEhb2, dcos_theta_dk );
                        rvec_Add( f_k_l, temp );
                        rvec_iMultiply( temp, rel_jk, temp );
                        rvec_Add( ext_press_l, temp );

                        /* dr terms */
                        rvec_ScaledAdd( f_j_l, -1.0 * CEhb3 / r_jk, dvec_jk ); 

                        rvec_Scale( temp, CEhb3 / r_jk, dvec_jk );
                        rvec_Add( f_k_l, temp );
                        rvec_iMultiply( temp, rel_jk, temp );
                        rvec_Add( ext_press_l, temp );
#endif
                    }
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
    e_hb_g[j] = e_hb_l;
    rvecCopy( ext_press_g[j], ext_press_l );
#else
    atomic_rvecAdd( workspace.f[i], f_i_l );
    atomic_rvecAdd( workspace.f[j], f_j_l );
    atomic_rvecAdd( workspace.f[k], f_k_l );
    atomicAdd( (double *) e_hb_g, (double) e_hb_l );
    atomic_rvecAdd( *ext_press_g, ext_press_l );
#endif
}


/* one warp of threads per atom implementation */
CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part1_opt( reax_atom *my_atoms, single_body_parameters *sbp, 
        hbond_parameters *d_hbp, global_parameters gp, control_params *control, storage workspace,
        reax_list far_nbr_list, reax_list bond_list, reax_list hbond_list, int n, 
        int num_atom_types, real *e_hb_g, rvec *ext_press_g )
{
    int i, j, k, pi, pk;
    int type_i, type_j, type_k;
    int start_j, end_j, hb_start_j, hb_end_j;
    //TODO: re-write and remove
    int hblist[30];
    int itr, top;
    int loopcount, count;
    ivec rel_jk;
    real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
    real e_hb, e_hb_l, exp_hb2, exp_hb3, CEhb1, CEhb1_l, CEhb2, CEhb3;
    rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
    rvec dvec_jk, temp, ext_press_l;
#if !defined(CUDA_ACCUM_ATOMIC)
    rvec f_l, hb_f_l;
#else
    rvec f_i_l, f_j_l, f_k_l;
#endif
    hbond_parameters *hbp;
    bond_order_data *bo_ij;
    int nbr_jk;
    bond_data *pbond_ij;
    hbond_data *phbond_jk;
    /* thread-local variables */
    int thread_id, warp_id, lane_id, offset;
    unsigned int mask;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id >> 5;
    lane_id = thread_id & 0x0000001F; 
    mask = __ballot_sync( FULL_MASK, warp_id < n );

    if ( warp_id >= n )
    {
        return;
    }

    j = warp_id; // group of threads assigned to atom j

    /* discover the Hydrogen bonds between i-j-k triplets.
     * here j is H atom and there has to be some bond between i and j.
     * Hydrogen bond is between j and k.
     * so in this function i->X, j->H, k->Z when we map 
     * variables onto the ones in the handout.*/
    e_hb_l = 0.0;
    rvec_MakeZero( ext_press_l );
#if !defined(CUDA_ACCUM_ATOMIC)
    rvec_MakeZero( f_l );
#else
    rvec_MakeZero( f_i_l );
    rvec_MakeZero( f_j_l );
    rvec_MakeZero( f_k_l );
#endif

    /* j has to be of type H */
    if ( sbp[ my_atoms[j].type ].p_hbond == H_ATOM )
    {
        type_j = my_atoms[j].type;
        start_j = Start_Index( j, &bond_list );
        end_j = End_Index( j, &bond_list );
        hb_start_j = Start_Index( my_atoms[j].Hindex, &hbond_list );
        hb_end_j = End_Index( my_atoms[j].Hindex, &hbond_list );
        top = 0;

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

        /* find matching hbond to atoms j and k */
        for ( itr = 0; itr < top; ++itr )
        {
            pi = hblist[itr];
            pbond_ij = &bond_list.bond_list[pi];
            i = pbond_ij->nbr;
#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_MakeZero( hb_f_l );
#endif
            CEhb1_l = 0.0;

            //for( pk = hb_start_j; pk < hb_end_j; ++pk ) {
            loopcount = (hb_end_j - hb_start_j) / warpSize + 
                (((hb_end_j - hb_start_j) % warpSize == 0) ? 0 : 1);

            count = 0;
            pk = hb_start_j + lane_id;
            while ( count < loopcount )
            {
                /* only allow threads with an actual hbond */
                if ( pk < hb_end_j )
                {
                    phbond_jk = &hbond_list.hbond_list[pk];

                    /* set k's varibles */
                    k = hbond_list.hbond_list[pk].nbr;
                    type_k = my_atoms[k].type;
                    nbr_jk = hbond_list.hbond_list[pk].ptr;
                    r_jk = far_nbr_list.far_nbr_list.d[nbr_jk];

                    rvec_Scale( dvec_jk, phbond_jk->scl,
                            far_nbr_list.far_nbr_list.dvec[nbr_jk] );
                }
                else
                {
                    k = -1;
                }

                if ( my_atoms[i].orig_id != my_atoms[k].orig_id && k != -1 )
                {
                    bo_ij = &pbond_ij->bo_data;
                    type_i = my_atoms[i].type;
                    r_ij = pbond_ij->d;
                    hbp = &d_hbp[ index_hbp(type_i,type_j,type_k,num_atom_types) ];

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
                    CEhb1_l += CEhb1;

                    if ( control->virial == 0 )
                    {
#if !defined(CUDA_ACCUM_ATOMIC)
                        /* dcos terms */
                        rvec_ScaledAdd( hb_f_l, CEhb2, dcos_theta_di ); 
                        rvec_ScaledAdd( f_l, CEhb2, dcos_theta_dj );
                        rvec_ScaledAdd( phbond_jk->hb_f, CEhb2, dcos_theta_dk );

                        /* dr terms */
                        rvec_ScaledAdd( f_l, -1.0 * CEhb3 / r_jk, dvec_jk ); 
                        rvec_ScaledAdd( phbond_jk->hb_f, CEhb3 / r_jk, dvec_jk );
#else
                        /* dcos terms */
                        rvec_ScaledAdd( f_i_l, CEhb2, dcos_theta_di ); 
                        rvec_ScaledAdd( f_j_l, CEhb2, dcos_theta_dj );
                        rvec_ScaledAdd( f_k_l, CEhb2, dcos_theta_dk );

                        /* dr terms */
                        rvec_ScaledAdd( f_j_l, -1.0 * CEhb3 / r_jk, dvec_jk ); 
                        rvec_ScaledAdd( f_k_l, CEhb3 / r_jk, dvec_jk );
#endif
                    }
                    else
                    {
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
                        rvec_Add( f_i_l, temp );
                        rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                        rvec_Add( ext_press_l, temp );

                        rvec_ScaledAdd( f_j_l, CEhb2, dcos_theta_dj );

                        ivec_Scale( rel_jk, hbond_list.hbond_list[pk].scl,
                                far_nbr_list.far_nbr_list.rel_box[nbr_jk] );
                        rvec_Scale( temp, CEhb2, dcos_theta_dk );
                        rvec_Add( f_k_l, temp );
                        rvec_iMultiply( temp, rel_jk, temp );
                        rvec_Add( ext_press_l, temp );

                        /* dr terms */
                        rvec_ScaledAdd( f_j_l, -1.0 * CEhb3 / r_jk, dvec_jk ); 

                        rvec_Scale( temp, CEhb3 / r_jk, dvec_jk );
                        rvec_Add( f_k_l, temp );
                        rvec_iMultiply( temp, rel_jk, temp );
                        rvec_Add( ext_press_l, temp );
#endif
                    }

                } //orid id end

                pk += warpSize;
                count++;

            } //for itr loop end

            /* warp-level sums using registers within a warp */
            for ( offset = warpSize >> 1; offset > 0; offset /= 2 )
            {
                CEhb1_l += __shfl_down_sync( mask, CEhb1_l, offset );
#if !defined(CUDA_ACCUM_ATOMIC)
                hb_f_l[0] += __shfl_down_sync( mask, hb_f_l[0], offset );
                hb_f_l[1] += __shfl_down_sync( mask, hb_f_l[1], offset );
                hb_f_l[2] += __shfl_down_sync( mask, hb_f_l[2], offset );
#endif
            }

            /* first thread within a warp writes warp-level sum to shared memory */
            if ( lane_id == 0 )
            {
                bo_ij->Cdbo += CEhb1_l ;
#if !defined(CUDA_ACCUM_ATOMIC)
                rvec_Add( pbond_ij->hb_f, hb_f_l );
#endif
            }
        } // for loop hbonds end
    } //if Hbond check end

    /* warp-level sums using registers within a warp */
    for ( offset = warpSize >> 1; offset > 0; offset /= 2 )
    {
#if !defined(CUDA_ACCUM_ATOMIC)
        f_l[0] += __shfl_down_sync( mask, f_l[0], offset );
        f_l[1] += __shfl_down_sync( mask, f_l[1], offset );
        f_l[2] += __shfl_down_sync( mask, f_l[2], offset );
#else
        f_i_l[0] += __shfl_down_sync( mask, f_i_l[0], offset );
        f_i_l[1] += __shfl_down_sync( mask, f_i_l[1], offset );
        f_i_l[2] += __shfl_down_sync( mask, f_i_l[2], offset );
        f_j_l[0] += __shfl_down_sync( mask, f_j_l[0], offset );
        f_j_l[1] += __shfl_down_sync( mask, f_j_l[1], offset );
        f_j_l[2] += __shfl_down_sync( mask, f_j_l[2], offset );
        f_k_l[0] += __shfl_down_sync( mask, f_k_l[0], offset );
        f_k_l[1] += __shfl_down_sync( mask, f_k_l[1], offset );
        f_k_l[2] += __shfl_down_sync( mask, f_k_l[2], offset );
#endif
        e_hb_l += __shfl_down_sync( mask, e_hb_l, offset );
        ext_press_l[0] += __shfl_down_sync( mask, ext_press_l[0], offset );
        ext_press_l[1] += __shfl_down_sync( mask, ext_press_l[1], offset );
        ext_press_l[2] += __shfl_down_sync( mask, ext_press_l[2], offset );
    }

    /* first thread within a warp writes warp-level sums to global memory */
    if ( lane_id == 0 )
    {
#if !defined(CUDA_ACCUM_ATOMIC)
        rvec_Add( workspace.f[j], f_l );
        e_hb_g[j] = e_hb_l;
        rvecCopy( ext_press_g[j], ext_press_l );
#else
        atomic_rvecAdd( workspace.f[i], f_i_l );
        atomic_rvecAdd( workspace.f[j], f_j_l );
        atomic_rvecAdd( workspace.f[k], f_k_l );
        atomicAdd( (double *) e_hb_g, (double) e_hb_l );
        atomic_rvecAdd( *ext_press_g, ext_press_l );
#endif
    }
}


#if !defined(CUDA_ACCUM_ATOMIC)
/* Accumulate forces stored in the bond list
 * using a one thread per atom implementation */
CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part2( reax_atom *atoms,
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
CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part2_opt( reax_atom *atoms,
        storage workspace, reax_list bond_list, int n )
{
    int j, pj, start, end;
    bond_data *pbond, *sym_index_bond;
    /* thread-local variables */
    int thread_id, warp_id, lane_id, offset;
    unsigned int mask;
    rvec hb_f;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id >> 5;
    lane_id = thread_id & 0x0000001F; 
    mask = __ballot_sync( FULL_MASK, warp_id < n );

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

    /* warp-level sums using registers within a warp */
    for ( offset = warpSize >> 1; offset > 0; offset /= 2 )
    {
        hb_f[0] += __shfl_down_sync( mask, hb_f[0], offset );
        hb_f[1] += __shfl_down_sync( mask, hb_f[1], offset );
        hb_f[2] += __shfl_down_sync( mask, hb_f[2], offset );
    }

    /* first thread within a warp writes warp-level sums to global memory */
    if ( lane_id == 0 )
    {
        rvec_Add( workspace.f[j], hb_f );
    }
}


/* Accumulate forces stored in the hbond list
 * using a one thread per atom implementation */
CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part3( reax_atom *atoms,
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
CUDA_GLOBAL void Cuda_Hydrogen_Bonds_Part3_opt( reax_atom *atoms,
        storage workspace, reax_list hbond_list, int n )
{
    int j, pj, start, end;
    hbond_data *nbr_pj, *sym_index_nbr;
    /* thread-local variables */
    int thread_id, warp_id, lane_id, offset;
    unsigned int mask;
    rvec hb_f_l;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id >> 5;
    lane_id = thread_id & 0x0000001F; 
    mask = __ballot_sync( FULL_MASK, warp_id < n );

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

    /* warp-level sums using registers within a warp */
    for ( offset = warpSize >> 1; offset > 0; offset /= 2 )
    {
        hb_f_l[0] += __shfl_down_sync( mask, hb_f_l[0], offset );
        hb_f_l[1] += __shfl_down_sync( mask, hb_f_l[1], offset );
        hb_f_l[2] += __shfl_down_sync( mask, hb_f_l[2], offset );
    }

    /* first thread within a warp writes warp-level sums to global memory */
    if ( lane_id == 0 )
    {
        rvec_Add( workspace.f[j], hb_f_l );
    }
}
#endif
