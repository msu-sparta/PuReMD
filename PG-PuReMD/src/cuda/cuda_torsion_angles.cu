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

#include "cuda_torsion_angles.h"

#include "cuda_list.h"
#include "cuda_helpers.h"
#include "cuda_utils.h"

#include "../index_utils.h"
#include "../vector.h"

#include "../cub/cub/warp/warp_reduce.cuh"
//#include <cub/warp/warp_reduce.cuh>


#define MIN_SINE (1.0e-10)


CUDA_DEVICE static real Calculate_Omega( const rvec dvec_ij, real r_ij,
        const rvec dvec_jk, real r_jk, const rvec dvec_kl, real r_kl,
        const rvec dvec_li, real r_li, three_body_interaction_data const * const p_ijk,
        three_body_interaction_data const * const p_jkl, rvec dcos_omega_di,
        rvec dcos_omega_dj, rvec dcos_omega_dk, rvec dcos_omega_dl )
{
    real unnorm_cos_omega, unnorm_sin_omega, omega;
    real sin_ijk, cos_ijk, sin_jkl, cos_jkl;
    real htra, htrb, htrc, hthd, hthe, hnra, hnrc, hnhd, hnhe;
    real arg, poem, tel;
    rvec cross_jk_kl;

    assert( r_ij > 0.0 );
    assert( r_jk > 0.0 );
    assert( r_kl > 0.0 );
    assert( r_li > 0.0 );

    sin_ijk = SIN( p_ijk->theta );
    cos_ijk = COS( p_ijk->theta );
    sin_jkl = SIN( p_jkl->theta );
    cos_jkl = COS( p_jkl->theta );

    /* omega */
    unnorm_cos_omega = -1.0 * rvec_Dot( dvec_ij, dvec_jk )
        * rvec_Dot( dvec_jk, dvec_kl ) + SQR( r_jk )
        * rvec_Dot( dvec_ij, dvec_kl );

    rvec_Cross( cross_jk_kl, dvec_jk, dvec_kl );
    unnorm_sin_omega = -1.0 * r_jk * rvec_Dot( dvec_ij, cross_jk_kl );

    omega = ATAN2( unnorm_sin_omega, unnorm_cos_omega ); 

    /* derivatives */
    /* coef for adjusments to cos_theta's */
    /* rla = r_ij, rlb = r_jk, rlc = r_kl, r4 = r_li;
     * coshd = cos_ijk, coshe = cos_jkl;
     * sinhd = sin_ijk, sinhe = sin_jkl; */
    htra = r_ij + cos_ijk * ( r_kl * cos_jkl - r_jk );
    htrb = r_jk - r_ij * cos_ijk - r_kl * cos_jkl;
    htrc = r_kl + cos_jkl * ( r_ij * cos_ijk - r_jk );
    hthd = r_ij * sin_ijk * ( r_jk - r_kl * cos_jkl );
    hthe = r_kl * sin_jkl * ( r_jk - r_ij * cos_ijk );
    hnra = r_kl * sin_ijk * sin_jkl;
    hnrc = r_ij * sin_ijk * sin_jkl;
    hnhd = r_ij * r_kl * cos_ijk * sin_jkl;
    hnhe = r_ij * r_kl * sin_ijk * cos_jkl;

    poem = 2.0 * r_ij * r_kl * sin_ijk * sin_jkl;
    if ( poem < 1.0e-20 )
    {
        poem = 1.0e-20;
    }

    tel = SQR( r_ij ) + SQR( r_jk ) + SQR( r_kl ) - SQR( r_li )
        - 2.0 * ( r_ij * r_jk * cos_ijk - r_ij * r_kl * cos_ijk
                * cos_jkl + r_jk * r_kl * cos_jkl );

    arg  = tel / poem;
    if ( arg >  1.0 )
    {
        arg =  1.0;
    }
    if ( arg < -1.0 )
    {
        arg = -1.0;
    }

    if ( sin_ijk >= 0.0 && sin_ijk <= MIN_SINE )
    {
        sin_ijk = MIN_SINE;
    }
    else if ( sin_ijk <= 0.0 && sin_ijk >= -MIN_SINE )
    {
        sin_ijk = -MIN_SINE;
    }
    if ( sin_jkl >= 0.0 && sin_jkl <= MIN_SINE )
    {
        sin_jkl = MIN_SINE;
    }
    else if ( sin_jkl <= 0.0 && sin_jkl >= -MIN_SINE )
    {
        sin_jkl = -MIN_SINE;
    }

    /* dcos_omega_di */
    rvec_ScaledSum( dcos_omega_di, (htra - arg * hnra) / r_ij, dvec_ij, -1.0, dvec_li );
    rvec_ScaledAdd( dcos_omega_di, -(hthd - arg * hnhd) / sin_ijk, p_ijk->dcos_dk );
    rvec_Scale( dcos_omega_di, 2.0 / poem, dcos_omega_di );

    /* dcos_omega_dj */
    rvec_ScaledSum( dcos_omega_dj, -(htra - arg * hnra) / r_ij, dvec_ij,
            -htrb / r_jk, dvec_jk );
    rvec_ScaledAdd( dcos_omega_dj, -(hthd - arg * hnhd) / sin_ijk, p_ijk->dcos_dj );
    rvec_ScaledAdd( dcos_omega_dj, -(hthe - arg * hnhe) / sin_jkl, p_jkl->dcos_di );
    rvec_Scale( dcos_omega_dj, 2.0 / poem, dcos_omega_dj );

    /* dcos_omega_dk */
    rvec_ScaledSum( dcos_omega_dk, -(htrc - arg * hnrc) / r_kl, dvec_kl,
            htrb / r_jk, dvec_jk );
    rvec_ScaledAdd( dcos_omega_dk, -(hthd - arg * hnhd) / sin_ijk, p_ijk->dcos_di );
    rvec_ScaledAdd( dcos_omega_dk, -(hthe - arg * hnhe) / sin_jkl, p_jkl->dcos_dj );
    rvec_Scale( dcos_omega_dk, 2.0 / poem, dcos_omega_dk );

    /* dcos_omega_dl */
    rvec_ScaledSum( dcos_omega_dl, (htrc - arg * hnrc) / r_kl, dvec_kl, 1.0, dvec_li );
    rvec_ScaledAdd( dcos_omega_dl, -(hthe - arg * hnhe) / sin_jkl, p_jkl->dcos_dk );
    rvec_Scale( dcos_omega_dl, 2.0 / poem, dcos_omega_dl );

    return omega;  
}


CUDA_GLOBAL void k_torsion_angles_part1( reax_atom const * const my_atoms,
        global_parameters gp, four_body_header const * const fbph,
        control_params const * const control, reax_list bond_list,
        reax_list thb_list, storage workspace, int n, int num_atom_types, 
        real * const e_tor_g, real * const e_con_g )
{
    int i, j, k, l, pi, pj, pk, pl, pij, plk;
    int type_i, type_j, type_k, type_l;
    int start_j, end_j;
    int start_pj, end_pj, start_pk, end_pk;
    real Delta_j, Delta_k;
    real r_ij, r_jk, r_kl, r_li;
    real BOA_ij, BOA_jk, BOA_kl;
    real exp_tor2_ij, exp_tor2_jk, exp_tor2_kl;
    real exp_tor1, exp_tor3_DjDk, exp_tor4_DjDk, exp_tor34_inv;
    real exp_cot2_jk, exp_cot2_ij, exp_cot2_kl;
    real fn10, f11_DjDk, dfn11, fn12;
    real theta_ijk, theta_jkl;
    real sin_ijk, sin_jkl;
    real cos_ijk, cos_jkl;
    real tan_ijk_i, tan_jkl_i;
    real omega, cos_omega, cos2omega, cos3omega;
    rvec dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl;
    real CV, cmn, CEtors1, CEtors2, CEtors3, CEtors4;
    real CEtors5, CEtors6, CEtors7, CEtors8, CEtors9;
    real Cconj, CEconj1, CEconj2, CEconj3;
    real CEconj4, CEconj5, CEconj6, e_tor_, e_con_;
    real CdDelta_j, CdDelta_k, Cdbopi_jk, Cdbo_ij, Cdbo_jk;
    rvec dvec_li, f_i, f_j, f_k;
    bond_data *pbond_ij, *pbond_jk, *pbond_kl;
    bond_order_data *bo_ij, *bo_jk, *bo_kl;
    three_body_interaction_data *p_ijk, *p_jkl;
    real p_tor2, p_tor3, p_tor4, p_cot2;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    p_tor2 = gp.l[23];
    p_tor3 = gp.l[24];
    p_tor4 = gp.l[25];
    p_cot2 = gp.l[27];
    e_tor_ = 0.0;
    e_con_ = 0.0;
    CdDelta_j = 0.0;
    rvec_MakeZero( f_j );

    type_j = my_atoms[j].type;
    Delta_j = workspace.Delta_boc[j];
    start_j = Start_Index(j, &bond_list);
    end_j = End_Index(j, &bond_list);

    for ( pk = start_j; pk < end_j; ++pk )
    {
        pbond_jk = &bond_list.bond_list[pk];
        k = pbond_jk->nbr;
        bo_jk = &pbond_jk->bo_data;
        BOA_jk = bo_jk->BO - control->thb_cut;

        /* see if there are any 3-body interactions involving j and k
         * where j is the central atom. Otherwise there is no point in
         * trying to form a 4-body interaction out of this neighborhood */
        if ( my_atoms[j].orig_id < my_atoms[k].orig_id
                && bo_jk->BO > control->thb_cut
                && Num_Entries(pk, &thb_list) > 0 )
        {
            /* pj points to j on k's list */
            pj = pbond_jk->sym_index;

            /* do the same check as above:
             * are there any 3-body interactions
             * involving k and j where k is the central atom */
            if ( Num_Entries(pj, &thb_list) > 0 )
            {
                type_k = my_atoms[k].type;
                Delta_k = workspace.Delta_boc[k];
                r_jk = pbond_jk->d;
                CdDelta_k = 0.0;
                Cdbopi_jk = 0.0;
                Cdbo_jk = 0.0;
                rvec_MakeZero( f_k );

                start_pk = Start_Index( pk, &thb_list );
                end_pk = End_Index( pk, &thb_list );
                start_pj = Start_Index( pj, &thb_list );
                end_pj = End_Index( pj, &thb_list );        

                exp_tor2_jk = EXP( -p_tor2 * BOA_jk );
                exp_cot2_jk = EXP( -p_cot2 * SQR(BOA_jk - 1.5) );
                exp_tor3_DjDk = EXP( -p_tor3 * (Delta_j + Delta_k) );
                exp_tor4_DjDk = EXP( p_tor4  * (Delta_j + Delta_k) );
                exp_tor34_inv = 1.0 / (1.0 + exp_tor3_DjDk + exp_tor4_DjDk);
                f11_DjDk = (2.0 + exp_tor3_DjDk) * exp_tor34_inv;

                /* pick i up from j-k interaction where j is the central atom */
                for ( pi = start_pk; pi < end_pk; ++pi )
                {
                    p_ijk = &thb_list.three_body_list[pi];
                    /* pij is pointer to i on j's bond_list */
                    pij = p_ijk->pthb;
                    pbond_ij = &bond_list.bond_list[pij];
                    bo_ij = &pbond_ij->bo_data;

                    if ( bo_ij->BO > control->thb_cut )
                    {
                        i = p_ijk->thb;
                        type_i = my_atoms[i].type;
                        r_ij = pbond_ij->d;
                        BOA_ij = bo_ij->BO - control->thb_cut;
                        Cdbo_ij = 0.0;
                        rvec_MakeZero( f_i );

                        theta_ijk = p_ijk->theta;
                        sin_ijk = SIN( theta_ijk );
                        cos_ijk = COS( theta_ijk );
//                        tan_ijk_i = 1.0 / TAN( theta_ijk );
                        if ( sin_ijk >= 0.0 && sin_ijk <= MIN_SINE )
                        {
                            tan_ijk_i = cos_ijk / MIN_SINE;
                        }
                        else if ( sin_ijk <= 0.0 && sin_ijk >= -MIN_SINE )
                        {
                            tan_ijk_i = cos_ijk / -MIN_SINE;
                        }
                        else
                        {
                            tan_ijk_i = cos_ijk / sin_ijk;
                        }

                        exp_tor2_ij = EXP( -p_tor2 * BOA_ij );
                        exp_cot2_ij = EXP( -p_cot2 * SQR(BOA_ij -1.5) );

                        /* pick l up from j-k interaction where k is the central atom */
                        for ( pl = start_pj; pl < end_pj; ++pl )
                        {
                            p_jkl = &thb_list.three_body_list[pl];
                            l = p_jkl->thb;
                            /* a pointer to l on k's bond_list! */
                            plk = p_jkl->pthb;
                            pbond_kl = &bond_list.bond_list[plk];
                            bo_kl = &pbond_kl->bo_data;
                            type_l = my_atoms[l].type;
                            four_body_parameters const * const fbp = &fbph[
                                index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].prm[0];

                            if ( i != l
                                    && fbph[ index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].cnt > 0
                                    && bo_kl->BO > control->thb_cut
                                    && bo_ij->BO * bo_jk->BO * bo_kl->BO > control->thb_cut )
                            {
                                r_kl = pbond_kl->d;
                                BOA_kl = bo_kl->BO - control->thb_cut;

                                theta_jkl = p_jkl->theta;
                                sin_jkl = SIN( theta_jkl );
                                cos_jkl = COS( theta_jkl );
//                                tan_jkl_i = 1.0 / TAN( theta_jkl );
                                if ( sin_jkl >= 0.0 && sin_jkl <= MIN_SINE )
                                {
                                    tan_jkl_i = cos_jkl / MIN_SINE;
                                }
                                else if ( sin_jkl <= 0.0 && sin_jkl >= -MIN_SINE )
                                {
                                    tan_jkl_i = cos_jkl / -MIN_SINE;
                                }
                                else
                                {
                                    tan_jkl_i = cos_jkl / sin_jkl;
                                }

                                rvec_ScaledSum( dvec_li, 1.0, my_atoms[i].x, 
                                        -1.0, my_atoms[l].x );
                                r_li = rvec_Norm( dvec_li );                 

                                /* omega and its derivative */
//                                cos_omega = Calculate_Omega( pbond_ij->dvec, r_ij, pbond_jk->dvec,
                                omega = Calculate_Omega( pbond_ij->dvec, r_ij, pbond_jk->dvec, r_jk,
                                        pbond_kl->dvec, r_kl, dvec_li, r_li, p_ijk, p_jkl,
                                        dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl );

                                cos_omega = COS( omega );
                                cos2omega = COS( 2.0 * omega );
                                cos3omega = COS( 3.0 * omega );
                                /* end omega calculations */

                                /* torsion energy */
                                exp_tor1 = EXP( fbp->p_tor1
                                        * SQR(2.0 - bo_jk->BO_pi - f11_DjDk) );
                                exp_tor2_kl = EXP( -p_tor2 * BOA_kl );
                                exp_cot2_kl = EXP( -p_cot2 * SQR(BOA_kl - 1.5) );
                                fn10 = (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * 
                                    (1.0 - exp_tor2_kl);

                                CV = 0.5 * ( fbp->V1 * (1.0 + cos_omega)
                                        + fbp->V2 * exp_tor1 * (1.0 - cos2omega)
                                        + fbp->V3 * (1.0 + cos3omega) );
//                                CV = 0.5 * fbp->V1 * (1.0 + cos_omega)
//                                    + fbp->V2 * exp_tor1 * (1.0 - SQR(cos_omega))
//                                    + fbp->V3 * (0.5 + 2.0 * CUBE(cos_omega) - 1.5 * cos_omega);

                                e_tor_ += fn10 * sin_ijk * sin_jkl * CV;

                                dfn11 = (-p_tor3 * exp_tor3_DjDk
                                        + (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk)
                                        * (2.0 + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;

                                CEtors1 = sin_ijk * sin_jkl * CV;

                                CEtors2 = -fn10 * 2.0 * fbp->p_tor1 * fbp->V2 * exp_tor1
                                    * (2.0 - bo_jk->BO_pi - f11_DjDk)
                                    * (1.0 - SQR(cos_omega)) * sin_ijk * sin_jkl; 
                                CEtors3 = CEtors2 * dfn11;

                                CEtors4 = CEtors1 * p_tor2 * exp_tor2_ij
                                    * (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);
                                CEtors5 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij)
                                    * exp_tor2_jk * (1.0 - exp_tor2_kl);
                                CEtors6 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij)
                                    * (1.0 - exp_tor2_jk) * exp_tor2_kl;

                                cmn = -fn10 * CV;
                                CEtors7 = cmn * sin_jkl * tan_ijk_i;
                                CEtors8 = cmn * sin_ijk * tan_jkl_i;

                                CEtors9 = fn10 * sin_ijk * sin_jkl
                                    * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega
                                            + 1.5 * fbp->V3 * (cos2omega + 2.0 * SQR(cos_omega)));
//                                CEtors7 = cmn * sin_jkl * cos_ijk;
//                                CEtors8 = cmn * sin_ijk * cos_jkl;
//                                CEtors9 = fn10 * sin_ijk * sin_jkl
//                                    * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega
//                                            + fbp->V3 * (6.0 * SQR(cos_omega) - 1.50));
                                /* end  of torsion energy */

                                /* 4-body conjugation energy */
                                fn12 = exp_cot2_ij * exp_cot2_jk * exp_cot2_kl;
                                e_con_ += fbp->p_cot1 * fn12
                                    * (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);

                                Cconj = -2.0 * fn12 * fbp->p_cot1 * p_cot2
                                    * (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);

                                CEconj1 = Cconj * (BOA_ij - 1.5);
                                CEconj2 = Cconj * (BOA_jk - 1.5);
                                CEconj3 = Cconj * (BOA_kl - 1.5);

                                CEconj4 = -fbp->p_cot1 * fn12
                                    * (SQR(cos_omega) - 1.0) * sin_jkl * tan_ijk_i;
                                CEconj5 = -fbp->p_cot1 * fn12
                                    * (SQR(cos_omega) - 1.0) * sin_ijk * tan_jkl_i;
//                                CEconj4 = -fbp->p_cot1 * fn12
//                                    * (SQR(cos_omega) - 1.0) * sin_jkl * cos_ijk;
//                                CEconj5 = -fbp->p_cot1 * fn12
//                                    * (SQR(cos_omega) - 1.0) * sin_ijk * cos_jkl;
                                CEconj6 = 2.0 * fbp->p_cot1 * fn12
                                    * cos_omega * sin_ijk * sin_jkl;
                                /* end 4-body conjugation energy */

                                /* forces */
                                Cdbopi_jk += CEtors2;
                                CdDelta_j += CEtors3;
                                CdDelta_k += CEtors3;
                                Cdbo_ij += (CEtors4 + CEconj1);
                                Cdbo_jk += (CEtors5 + CEconj2);
#if !defined(CUDA_ACCUM_ATOMIC)
                                atomicAdd( &pbond_kl->ta_Cdbo, CEtors6 + CEconj3 );
#else
                                atomicAdd( &bo_kl->Cdbo, CEtors6 + CEconj3 );
#endif

                                /* dcos_theta_ijk */
                                rvec_ScaledAdd( f_i, CEtors7 + CEconj4, p_ijk->dcos_dk );
                                rvec_ScaledAdd( f_j, CEtors7 + CEconj4, p_ijk->dcos_dj );
                                rvec_ScaledAdd( f_k, CEtors7 + CEconj4, p_ijk->dcos_di );

                                /* dcos_theta_jkl */
                                rvec_ScaledAdd( f_j, CEtors8 + CEconj5, p_jkl->dcos_di );
                                rvec_ScaledAdd( f_k, CEtors8 + CEconj5, p_jkl->dcos_dj );
#if !defined(CUDA_ACCUM_ATOMIC)
                                atomic_rvecScaledAdd( pbond_kl->ta_f, CEtors8 + CEconj5, p_jkl->dcos_dk );
#else
                                atomic_rvecScaledAdd( workspace.f[l], CEtors8 + CEconj5, p_jkl->dcos_dk );
#endif

                                /* dcos_omega */
                                rvec_ScaledAdd( f_i, CEtors9 + CEconj6, dcos_omega_di );
                                rvec_ScaledAdd( f_j, CEtors9 + CEconj6, dcos_omega_dj );
                                rvec_ScaledAdd( f_k, CEtors9 + CEconj6, dcos_omega_dk );
#if !defined(CUDA_ACCUM_ATOMIC)
                                atomic_rvecScaledAdd( pbond_kl->ta_f, CEtors9 + CEconj6, dcos_omega_dl );
#else
                                atomic_rvecScaledAdd( workspace.f[l], CEtors9 + CEconj6, dcos_omega_dl );
#endif
                            } // pl check ends
                        } // pl loop ends

#if !defined(CUDA_ACCUM_ATOMIC)
                    bo_ij->Cdbo += Cdbo_ij;
                    atomic_rvecAdd( pbond_ij->ta_f, f_i );
#else
                    atomicAdd( &bo_ij->Cdbo, Cdbo_ij );
                    atomic_rvecAdd( workspace.f[i], f_i );
#endif
                    } // pi check ends
                } // pi loop ends

#if !defined(CUDA_ACCUM_ATOMIC)
                bo_jk->Cdbopi += Cdbopi_jk;
                pbond_jk->ta_CdDelta += CdDelta_k;
                bo_jk->Cdbo += Cdbo_jk;
                atomic_rvecAdd( pbond_jk->ta_f, f_k );
#else
                atomicAdd( &bo_jk->Cdbopi, Cdbopi_jk );
                atomicAdd( &workspace.CdDelta[k], CdDelta_k );
                atomicAdd( &bo_jk->Cdbo, Cdbo_jk );
                atomic_rvecAdd( workspace.f[k], f_k );
#endif
            } // k-j neighbor check ends
        } // j<k && j-k neighbor check ends
    } // pk loop ends

#if !defined(CUDA_ACCUM_ATOMIC)
    workspace.CdDelta[j] += CdDelta_j;
    rvec_Add( workspace.f[j], f_j );
    e_tor_g[j] = e_tor_;
    e_con_g[j] = e_con_;
#else
    atomicAdd( &workspace.CdDelta[j], CdDelta_j );
    atomic_rvecAdd( workspace.f[j], f_j );
    atomicAdd( (double *) e_tor_g, (double) e_tor_ );
    atomicAdd( (double *) e_con_g, (double) e_con_ );
#endif
}


CUDA_GLOBAL void k_torsion_angles_part1_opt( reax_atom const * const my_atoms,
        global_parameters gp, four_body_header const * const fbph,
        control_params const * const control, reax_list bond_list,
        reax_list thb_list, storage workspace, int n, int num_atom_types, 
        real * const e_tor_g, real * const e_con_g )
{
    extern __shared__ cub::WarpReduce<double>::TempStorage temp_d[];
    int i, j, k, l, pi, pj, pk, pl, pij, plk, thread_id, warp_id, lane_id, itr;
    int type_i, type_j, type_k, type_l;
    int start_j, end_j;
    int start_pj, end_pj, start_pk, end_pk;
    real Delta_j, Delta_k;
    real r_ij, r_jk, r_kl, r_li;
    real BOA_ij, BOA_jk, BOA_kl;
    real exp_tor2_ij, exp_tor2_jk, exp_tor2_kl;
    real exp_tor1, exp_tor3_DjDk, exp_tor4_DjDk, exp_tor34_inv;
    real exp_cot2_jk, exp_cot2_ij, exp_cot2_kl;
    real fn10, f11_DjDk, dfn11, fn12;
    real theta_ijk, theta_jkl;
    real sin_ijk, sin_jkl;
    real cos_ijk, cos_jkl;
    real tan_ijk_i, tan_jkl_i;
    real omega, cos_omega, cos2omega, cos3omega;
    rvec dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl;
    real CV, cmn, CEtors1, CEtors2, CEtors3, CEtors4;
    real CEtors5, CEtors6, CEtors7, CEtors8, CEtors9;
    real Cconj, CEconj1, CEconj2, CEconj3;
    real CEconj4, CEconj5, CEconj6, e_tor_, e_con_;
    real CdDelta_j, CdDelta_k, Cdbopi_jk, Cdbo_ij, Cdbo_jk;
    rvec dvec_li, f_i, f_j, f_k;
    bond_data *pbond_ij, *pbond_jk, *pbond_kl;
    bond_order_data *bo_ij, *bo_jk, *bo_kl;
    three_body_interaction_data *p_ijk, *p_jkl;
    real p_tor2, p_tor3, p_tor4, p_cot2;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    j = thread_id / warpSize;

    if ( j >= n )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize;
    p_tor2 = gp.l[23];
    p_tor3 = gp.l[24];
    p_tor4 = gp.l[25];
    p_cot2 = gp.l[27];
    e_tor_ = 0.0;
    e_con_ = 0.0;
    CdDelta_j = 0.0;
    rvec_MakeZero( f_j );

    type_j = my_atoms[j].type;
    Delta_j = workspace.Delta_boc[j];
    start_j = Start_Index(j, &bond_list);
    end_j = End_Index(j, &bond_list);

    for ( pk = start_j; pk < end_j; ++pk )
    {
        pbond_jk = &bond_list.bond_list[pk];
        k = pbond_jk->nbr;
        bo_jk = &pbond_jk->bo_data;
        BOA_jk = bo_jk->BO - control->thb_cut;

        /* see if there are any 3-body interactions involving j and k
         * where j is the central atom. Otherwise there is no point in
         * trying to form a 4-body interaction out of this neighborhood */
        if ( my_atoms[j].orig_id < my_atoms[k].orig_id
                && bo_jk->BO > control->thb_cut
                && Num_Entries(pk, &thb_list) > 0 )
        {
            /* pj points to j on k's list */
            pj = pbond_jk->sym_index;

            /* do the same check as above:
             * are there any 3-body interactions
             * involving k and j where k is the central atom */
            if ( Num_Entries(pj, &thb_list) > 0 )
            {
                type_k = my_atoms[k].type;
                Delta_k = workspace.Delta_boc[k];
                r_jk = pbond_jk->d;
                CdDelta_k = 0.0;
                Cdbopi_jk = 0.0;
                Cdbo_jk = 0.0;
                rvec_MakeZero( f_k );

                start_pk = Start_Index( pk, &thb_list );
                end_pk = End_Index( pk, &thb_list );
                start_pj = Start_Index( pj, &thb_list );
                end_pj = End_Index( pj, &thb_list );        

                exp_tor2_jk = EXP( -p_tor2 * BOA_jk );
                exp_cot2_jk = EXP( -p_cot2 * SQR(BOA_jk - 1.5) );
                exp_tor3_DjDk = EXP( -p_tor3 * (Delta_j + Delta_k) );
                exp_tor4_DjDk = EXP( p_tor4  * (Delta_j + Delta_k) );
                exp_tor34_inv = 1.0 / (1.0 + exp_tor3_DjDk + exp_tor4_DjDk);
                f11_DjDk = (2.0 + exp_tor3_DjDk) * exp_tor34_inv;

                /* pick i up from j-k interaction where j is the central atom */
                for ( pi = start_pk; pi < end_pk; ++pi )
                {
                    p_ijk = &thb_list.three_body_list[pi];
                    /* pij is pointer to i on j's bond_list */
                    pij = p_ijk->pthb;
                    pbond_ij = &bond_list.bond_list[pij];
                    bo_ij = &pbond_ij->bo_data;

                    if ( bo_ij->BO > control->thb_cut )
                    {
                        i = p_ijk->thb;
                        type_i = my_atoms[i].type;
                        r_ij = pbond_ij->d;
                        BOA_ij = bo_ij->BO - control->thb_cut;
                        Cdbo_ij = 0.0;
                        rvec_MakeZero( f_i );

                        theta_ijk = p_ijk->theta;
                        sin_ijk = SIN( theta_ijk );
                        cos_ijk = COS( theta_ijk );
//                        tan_ijk_i = 1.0 / TAN( theta_ijk );
                        if ( sin_ijk >= 0.0 && sin_ijk <= MIN_SINE )
                        {
                            tan_ijk_i = cos_ijk / MIN_SINE;
                        }
                        else if ( sin_ijk <= 0.0 && sin_ijk >= -MIN_SINE )
                        {
                            tan_ijk_i = cos_ijk / -MIN_SINE;
                        }
                        else
                        {
                            tan_ijk_i = cos_ijk / sin_ijk;
                        }

                        exp_tor2_ij = EXP( -p_tor2 * BOA_ij );
                        exp_cot2_ij = EXP( -p_cot2 * SQR(BOA_ij -1.5) );

                        /* pick l up from j-k interaction where k is the central atom */
                        for ( itr = 0, pl = start_pj + lane_id; itr < (end_pj - start_pj + warpSize - 1) / warpSize; ++itr )
                        {
                            if ( pl < end_pj )
                            {
                                p_jkl = &thb_list.three_body_list[pl];
                                l = p_jkl->thb;
                                /* a pointer to l on k's bond_list! */
                                plk = p_jkl->pthb;
                                pbond_kl = &bond_list.bond_list[plk];
                                bo_kl = &pbond_kl->bo_data;
                                type_l = my_atoms[l].type;
                                four_body_parameters const * const fbp = &fbph[
                                    index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].prm[0];

                                if ( i != l
                                        && fbph[ index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].cnt > 0
                                        && bo_kl->BO > control->thb_cut
                                        && bo_ij->BO * bo_jk->BO * bo_kl->BO > control->thb_cut )
                                {
                                    r_kl = pbond_kl->d;
                                    BOA_kl = bo_kl->BO - control->thb_cut;

                                    theta_jkl = p_jkl->theta;
                                    sin_jkl = SIN( theta_jkl );
                                    cos_jkl = COS( theta_jkl );
                                    //tan_jkl_i = 1.0 / TAN( theta_jkl );
                                    if ( sin_jkl >= 0.0 && sin_jkl <= MIN_SINE )
                                    {
                                        tan_jkl_i = cos_jkl / MIN_SINE;
                                    }
                                    else if ( sin_jkl <= 0.0 && sin_jkl >= -MIN_SINE )
                                    {
                                        tan_jkl_i = cos_jkl / -MIN_SINE;
                                    }
                                    else
                                    {
                                        tan_jkl_i = cos_jkl / sin_jkl;
                                    }

                                    rvec_ScaledSum( dvec_li, 1.0, my_atoms[i].x, 
                                            -1.0, my_atoms[l].x );
                                    r_li = rvec_Norm( dvec_li );                 

                                    /* omega and its derivative */
//                                    cos_omega = Calculate_Omega( pbond_ij->dvec, r_ij, pbond_jk->dvec,
                                    omega = Calculate_Omega( pbond_ij->dvec, r_ij, pbond_jk->dvec, r_jk,
                                            pbond_kl->dvec, r_kl, dvec_li, r_li, p_ijk, p_jkl,
                                            dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl );

                                    cos_omega = COS( omega );
                                    cos2omega = COS( 2.0 * omega );
                                    cos3omega = COS( 3.0 * omega );
                                    /* end omega calculations */

                                    /* torsion energy */
                                    exp_tor1 = EXP( fbp->p_tor1
                                            * SQR(2.0 - bo_jk->BO_pi - f11_DjDk) );
                                    exp_tor2_kl = EXP( -p_tor2 * BOA_kl );
                                    exp_cot2_kl = EXP( -p_cot2 * SQR(BOA_kl - 1.5) );
                                    fn10 = (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * 
                                        (1.0 - exp_tor2_kl);

                                    CV = 0.5 * ( fbp->V1 * (1.0 + cos_omega)
                                            + fbp->V2 * exp_tor1 * (1.0 - cos2omega)
                                            + fbp->V3 * (1.0 + cos3omega) );
//                                    CV = 0.5 * fbp->V1 * (1.0 + cos_omega)
//                                        + fbp->V2 * exp_tor1 * (1.0 - SQR(cos_omega))
//                                        + fbp->V3 * (0.5 + 2.0 * CUBE(cos_omega) - 1.5 * cos_omega);

                                    e_tor_ += fn10 * sin_ijk * sin_jkl * CV;

                                    dfn11 = (-p_tor3 * exp_tor3_DjDk
                                            + (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk)
                                            * (2.0 + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;

                                    CEtors1 = sin_ijk * sin_jkl * CV;

                                    CEtors2 = -fn10 * 2.0 * fbp->p_tor1 * fbp->V2 * exp_tor1
                                        * (2.0 - bo_jk->BO_pi - f11_DjDk)
                                        * (1.0 - SQR(cos_omega)) * sin_ijk * sin_jkl; 
                                    CEtors3 = CEtors2 * dfn11;

                                    CEtors4 = CEtors1 * p_tor2 * exp_tor2_ij
                                        * (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);
                                    CEtors5 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij)
                                        * exp_tor2_jk * (1.0 - exp_tor2_kl);
                                    CEtors6 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij)
                                        * (1.0 - exp_tor2_jk) * exp_tor2_kl;

                                    cmn = -fn10 * CV;
                                    CEtors7 = cmn * sin_jkl * tan_ijk_i;
                                    CEtors8 = cmn * sin_ijk * tan_jkl_i;

                                    CEtors9 = fn10 * sin_ijk * sin_jkl
                                        * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega
                                                + 1.5 * fbp->V3 * (cos2omega + 2.0 * SQR(cos_omega)));
//                                    CEtors7 = cmn * sin_jkl * cos_ijk;
//                                    CEtors8 = cmn * sin_ijk * cos_jkl;
//                                    CEtors9 = fn10 * sin_ijk * sin_jkl
//                                        * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega
//                                                + fbp->V3 * (6.0 * SQR(cos_omega) - 1.50));
                                    /* end  of torsion energy */

                                    /* 4-body conjugation energy */
                                    fn12 = exp_cot2_ij * exp_cot2_jk * exp_cot2_kl;
                                    e_con_ += fbp->p_cot1 * fn12
                                        * (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);

                                    Cconj = -2.0 * fn12 * fbp->p_cot1 * p_cot2
                                        * (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);

                                    CEconj1 = Cconj * (BOA_ij - 1.5);
                                    CEconj2 = Cconj * (BOA_jk - 1.5);
                                    CEconj3 = Cconj * (BOA_kl - 1.5);

                                    CEconj4 = -fbp->p_cot1 * fn12
                                        * (SQR(cos_omega) - 1.0) * sin_jkl * tan_ijk_i;
                                    CEconj5 = -fbp->p_cot1 * fn12
                                        * (SQR(cos_omega) - 1.0) * sin_ijk * tan_jkl_i;
//                                    CEconj4 = -fbp->p_cot1 * fn12
//                                        * (SQR(cos_omega) - 1.0) * sin_jkl * cos_ijk;
//                                    CEconj5 = -fbp->p_cot1 * fn12
//                                        * (SQR(cos_omega) - 1.0) * sin_ijk * cos_jkl;
                                    CEconj6 = 2.0 * fbp->p_cot1 * fn12
                                        * cos_omega * sin_ijk * sin_jkl;
                                    /* end 4-body conjugation energy */

                                    /* forces */
                                    Cdbopi_jk += CEtors2;
                                    CdDelta_j += CEtors3;
                                    CdDelta_k += CEtors3;
                                    Cdbo_ij += (CEtors4 + CEconj1);
                                    Cdbo_jk += (CEtors5 + CEconj2);
#if !defined(CUDA_ACCUM_ATOMIC)
                                    atomicAdd( &pbond_kl->ta_Cdbo, CEtors6 + CEconj3 );
#else
                                    atomicAdd( &bo_kl->Cdbo, CEtors6 + CEconj3 );
#endif

                                    /* dcos_theta_ijk */
                                    rvec_ScaledAdd( f_i, CEtors7 + CEconj4, p_ijk->dcos_dk );
                                    rvec_ScaledAdd( f_j, CEtors7 + CEconj4, p_ijk->dcos_dj );
                                    rvec_ScaledAdd( f_k, CEtors7 + CEconj4, p_ijk->dcos_di );

                                    /* dcos_theta_jkl */
                                    rvec_ScaledAdd( f_j, CEtors8 + CEconj5, p_jkl->dcos_di );
                                    rvec_ScaledAdd( f_k, CEtors8 + CEconj5, p_jkl->dcos_dj );
#if !defined(CUDA_ACCUM_ATOMIC)
                                    atomic_rvecScaledAdd( pbond_kl->ta_f, CEtors8 + CEconj5, p_jkl->dcos_dk );
#else
                                    atomic_rvecScaledAdd( workspace.f[l], CEtors8 + CEconj5, p_jkl->dcos_dk );
#endif

                                    /* dcos_omega */
                                    rvec_ScaledAdd( f_i, CEtors9 + CEconj6, dcos_omega_di );
                                    rvec_ScaledAdd( f_j, CEtors9 + CEconj6, dcos_omega_dj );
                                    rvec_ScaledAdd( f_k, CEtors9 + CEconj6, dcos_omega_dk );
#if !defined(CUDA_ACCUM_ATOMIC)
                                    atomic_rvecScaledAdd( pbond_kl->ta_f, CEtors9 + CEconj6, dcos_omega_dl );
#else
                                    atomic_rvecScaledAdd( workspace.f[l], CEtors9 + CEconj6, dcos_omega_dl );
#endif
                                } // pl check ends
                            }

                            pl += warpSize;
                        } // pl loop ends

                        Cdbo_ij = cub::WarpReduce<double>(temp_d[warp_id]).Sum(Cdbo_ij);
                        f_i[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[0]);
                        f_i[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[1]);
                        f_i[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[2]);

                        if ( lane_id == 0 )
                        {
#if !defined(CUDA_ACCUM_ATOMIC)
                            bo_ij->Cdbo += Cdbo_ij;
                            atomic_rvecAdd( pbond_ij->ta_f, f_i );
#else
                            atomicAdd( &bo_ij->Cdbo, Cdbo_ij );
                            atomic_rvecAdd( workspace.f[i], f_i );
#endif
                        }
                    } // pi check ends
                } // pi loop ends

                Cdbopi_jk = cub::WarpReduce<double>(temp_d[warp_id]).Sum(Cdbopi_jk);
                CdDelta_k = cub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_k);
                Cdbo_jk = cub::WarpReduce<double>(temp_d[warp_id]).Sum(Cdbo_jk);
                f_k[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[0]);
                f_k[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[1]);
                f_k[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[2]);

                if ( lane_id == 0 )
                {
#if !defined(CUDA_ACCUM_ATOMIC)
                    bo_jk->Cdbopi += Cdbopi_jk;
                    pbond_jk->ta_CdDelta += CdDelta_k;
                    bo_jk->Cdbo += Cdbo_jk;
                    atomic_rvecAdd( pbond_jk->ta_f, f_k );
#else
                    atomicAdd( &bo_jk->Cdbopi, Cdbopi_jk );
                    atomicAdd( &workspace.CdDelta[k], CdDelta_k );
                    atomicAdd( &bo_jk->Cdbo, Cdbo_jk );
                    atomic_rvecAdd( workspace.f[k], f_k );
#endif
                }
            } // k-j neighbor check ends
        } // j<k && j-k neighbor check ends
    } // pk loop ends

    CdDelta_j = cub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_j);
    f_j[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[0]);
    f_j[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[1]);
    f_j[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[2]);
    e_tor_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_tor_);
    e_con_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_con_);

    if ( lane_id == 0 )
    {
#if !defined(CUDA_ACCUM_ATOMIC)
        workspace.CdDelta[j] += CdDelta_j;
        rvec_Add( workspace.f[j], f_j );
        e_tor_g[j] = e_tor_;
        e_con_g[j] = e_con_;
#else
        atomicAdd( &workspace.CdDelta[j], CdDelta_j );
        atomic_rvecAdd( workspace.f[j], f_j );
        atomicAdd( (double *) e_tor_g, (double) e_tor_ );
        atomicAdd( (double *) e_con_g, (double) e_con_ );
#endif
    }
}


CUDA_GLOBAL void k_torsion_angles_virial_part1( reax_atom const * const my_atoms,
        global_parameters gp, four_body_header const * const fbph,
        control_params const * const control, reax_list bond_list,
        reax_list thb_list, storage workspace, int n, int num_atom_types, 
        real * const e_tor_g, real * const e_con_g, rvec * const ext_press_g )
{
    int i, j, k, l, pi, pj, pk, pl, pij, plk;
    int type_i, type_j, type_k, type_l;
    int start_j, end_j;
    int start_pj, end_pj, start_pk, end_pk;
    real Delta_j, Delta_k;
    real r_ij, r_jk, r_kl, r_li;
    real BOA_ij, BOA_jk, BOA_kl;
    real exp_tor2_ij, exp_tor2_jk, exp_tor2_kl;
    real exp_tor1, exp_tor3_DjDk, exp_tor4_DjDk, exp_tor34_inv;
    real exp_cot2_jk, exp_cot2_ij, exp_cot2_kl;
    real fn10, f11_DjDk, dfn11, fn12;
    real theta_ijk, theta_jkl;
    real sin_ijk, sin_jkl;
    real cos_ijk, cos_jkl;
    real tan_ijk_i, tan_jkl_i;
    real omega, cos_omega, cos2omega, cos3omega;
    rvec dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl;
    real CV, cmn, CEtors1, CEtors2, CEtors3, CEtors4;
    real CEtors5, CEtors6, CEtors7, CEtors8, CEtors9;
    real Cconj, CEconj1, CEconj2, CEconj3;
    real CEconj4, CEconj5, CEconj6, e_tor_, e_con_;
    real CdDelta_j, CdDelta_k, Cdbopi_jk, Cdbo_ij, Cdbo_jk;
    rvec dvec_li, temp, f_i, f_j, f_k, ext_press_;
    ivec rel_box_jl;
    bond_data *pbond_ij, *pbond_jk, *pbond_kl;
    bond_order_data *bo_ij, *bo_jk, *bo_kl;
    three_body_interaction_data *p_ijk, *p_jkl;
    real p_tor2, p_tor3, p_tor4, p_cot2;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    p_tor2 = gp.l[23];
    p_tor3 = gp.l[24];
    p_tor4 = gp.l[25];
    p_cot2 = gp.l[27];
    e_tor_ = 0.0;
    e_con_ = 0.0;
    CdDelta_j = 0.0;
    rvec_MakeZero( f_j );
    rvec_MakeZero( ext_press_ );

    type_j = my_atoms[j].type;
    Delta_j = workspace.Delta_boc[j];
    start_j = Start_Index(j, &bond_list);
    end_j = End_Index(j, &bond_list);

    for ( pk = start_j; pk < end_j; ++pk )
    {
        pbond_jk = &bond_list.bond_list[pk];
        k = pbond_jk->nbr;
        bo_jk = &pbond_jk->bo_data;
        BOA_jk = bo_jk->BO - control->thb_cut;

        /* see if there are any 3-body interactions involving j and k
         * where j is the central atom. Otherwise there is no point in
         * trying to form a 4-body interaction out of this neighborhood */
        if ( my_atoms[j].orig_id < my_atoms[k].orig_id
                && bo_jk->BO > control->thb_cut
                && Num_Entries(pk, &thb_list) )
        {
            /* pj points to j on k's list */
            pj = pbond_jk->sym_index;

            /* do the same check as above:
             * are there any 3-body interactions
             * involving k and j where k is the central atom */
            if ( Num_Entries(pj, &thb_list) > 0 )
            {
                type_k = my_atoms[k].type;
                Delta_k = workspace.Delta_boc[k];
                r_jk = pbond_jk->d;
                CdDelta_k = 0.0;
                Cdbopi_jk = 0.0;
                Cdbo_jk = 0.0;
                rvec_MakeZero( f_k );

                start_pk = Start_Index( pk, &thb_list );
                end_pk = End_Index( pk, &thb_list );
                start_pj = Start_Index( pj, &thb_list );
                end_pj = End_Index( pj, &thb_list );        

                exp_tor2_jk = EXP( -p_tor2 * BOA_jk );
                exp_cot2_jk = EXP( -p_cot2 * SQR(BOA_jk - 1.5) );
                exp_tor3_DjDk = EXP( -p_tor3 * (Delta_j + Delta_k) );
                exp_tor4_DjDk = EXP( p_tor4  * (Delta_j + Delta_k) );
                exp_tor34_inv = 1.0 / (1.0 + exp_tor3_DjDk + exp_tor4_DjDk);
                f11_DjDk = (2.0 + exp_tor3_DjDk) * exp_tor34_inv;

                /* pick i up from j-k interaction where j is the central atom */
                for ( pi = start_pk; pi < end_pk; ++pi )
                {
                    p_ijk = &thb_list.three_body_list[pi];
                    /* pij is pointer to i on j's bond_list */
                    pij = p_ijk->pthb;
                    pbond_ij = &bond_list.bond_list[pij];
                    bo_ij = &pbond_ij->bo_data;

                    if ( bo_ij->BO > control->thb_cut )
                    {
                        i = p_ijk->thb;
                        type_i = my_atoms[i].type;
                        r_ij = pbond_ij->d;
                        BOA_ij = bo_ij->BO - control->thb_cut;
                        Cdbo_ij = 0.0;
                        rvec_MakeZero( f_i );

                        theta_ijk = p_ijk->theta;
                        sin_ijk = SIN( theta_ijk );
                        cos_ijk = COS( theta_ijk );
//                        tan_ijk_i = 1.0 / TAN( theta_ijk );
                        if ( sin_ijk >= 0.0 && sin_ijk <= MIN_SINE )
                        {
                            tan_ijk_i = cos_ijk / MIN_SINE;
                        }
                        else if ( sin_ijk <= 0.0 && sin_ijk >= -MIN_SINE )
                        {
                            tan_ijk_i = cos_ijk / -MIN_SINE;
                        }
                        else
                        {
                            tan_ijk_i = cos_ijk / sin_ijk;
                        }

                        exp_tor2_ij = EXP( -p_tor2 * BOA_ij );
                        exp_cot2_ij = EXP( -p_cot2 * SQR(BOA_ij -1.5) );

                        /* pick l up from j-k interaction where k is the central atom */
                        for ( pl = start_pj; pl < end_pj; ++pl )
                        {
                            p_jkl = &thb_list.three_body_list[pl];
                            l = p_jkl->thb;
                            /* a pointer to l on k's bond_list! */
                            plk = p_jkl->pthb;
                            pbond_kl = &bond_list.bond_list[plk];
                            bo_kl = &pbond_kl->bo_data;
                            type_l = my_atoms[l].type;
                            four_body_parameters const * const fbp = &fbph[
                                index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].prm[0];

                            if ( i != l
                                    && fbph[ index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].cnt > 0
                                    && bo_kl->BO > control->thb_cut
                                    && bo_ij->BO * bo_jk->BO * bo_kl->BO > control->thb_cut )
                            {
                                r_kl = pbond_kl->d;
                                BOA_kl = bo_kl->BO - control->thb_cut;

                                theta_jkl = p_jkl->theta;
                                sin_jkl = SIN( theta_jkl );
                                cos_jkl = COS( theta_jkl );
//                                tan_jkl_i = 1.0 / TAN( theta_jkl );
                                if ( sin_jkl >= 0.0 && sin_jkl <= MIN_SINE )
                                {
                                    tan_jkl_i = cos_jkl / MIN_SINE;
                                }
                                else if ( sin_jkl <= 0.0 && sin_jkl >= -MIN_SINE )
                                {
                                    tan_jkl_i = cos_jkl / -MIN_SINE;
                                }
                                else
                                {
                                    tan_jkl_i = cos_jkl / sin_jkl;
                                }

                                rvec_ScaledSum( dvec_li, 1.0, my_atoms[i].x, 
                                        -1.0, my_atoms[l].x );
                                r_li = rvec_Norm( dvec_li );                 

                                /* omega and its derivative */
//                                cos_omega = Calculate_Omega( pbond_ij->dvec, r_ij, pbond_jk->dvec,
                                omega = Calculate_Omega( pbond_ij->dvec, r_ij, pbond_jk->dvec, r_jk,
                                        pbond_kl->dvec, r_kl, dvec_li, r_li, p_ijk, p_jkl,
                                        dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl );

                                cos_omega = COS( omega );
                                cos2omega = COS( 2.0 * omega );
                                cos3omega = COS( 3.0 * omega );
                                /* end omega calculations */

                                /* torsion energy */
                                exp_tor1 = EXP( fbp->p_tor1
                                        * SQR(2.0 - bo_jk->BO_pi - f11_DjDk) );
                                exp_tor2_kl = EXP( -p_tor2 * BOA_kl );
                                exp_cot2_kl = EXP( -p_cot2 * SQR(BOA_kl - 1.5) );
                                fn10 = (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * 
                                    (1.0 - exp_tor2_kl);

                                CV = 0.5 * ( fbp->V1 * (1.0 + cos_omega)
                                        + fbp->V2 * exp_tor1 * (1.0 - cos2omega)
                                        + fbp->V3 * (1.0 + cos3omega) );
//                                CV = 0.5 * fbp->V1 * (1.0 + cos_omega)
//                                    + fbp->V2 * exp_tor1 * (1.0 - SQR(cos_omega))
//                                    + fbp->V3 * (0.5 + 2.0 * CUBE(cos_omega) - 1.5 * cos_omega);

                                e_tor_ += fn10 * sin_ijk * sin_jkl * CV;

                                dfn11 = (-p_tor3 * exp_tor3_DjDk
                                        + (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk)
                                        * (2.0 + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;

                                CEtors1 = sin_ijk * sin_jkl * CV;

                                CEtors2 = -fn10 * 2.0 * fbp->p_tor1 * fbp->V2 * exp_tor1
                                    * (2.0 - bo_jk->BO_pi - f11_DjDk)
                                    * (1.0 - SQR(cos_omega)) * sin_ijk * sin_jkl; 
                                CEtors3 = CEtors2 * dfn11;

                                CEtors4 = CEtors1 * p_tor2 * exp_tor2_ij
                                    * (1.0 - exp_tor2_jk) * (1.0 - exp_tor2_kl);
                                CEtors5 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij)
                                    * exp_tor2_jk * (1.0 - exp_tor2_kl);
                                CEtors6 = CEtors1 * p_tor2 * (1.0 - exp_tor2_ij)
                                    * (1.0 - exp_tor2_jk) * exp_tor2_kl;

                                cmn = -fn10 * CV;
                                CEtors7 = cmn * sin_jkl * tan_ijk_i;
                                CEtors8 = cmn * sin_ijk * tan_jkl_i;

                                CEtors9 = fn10 * sin_ijk * sin_jkl
                                    * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega
                                            + 1.5 * fbp->V3 * (cos2omega + 2.0 * SQR(cos_omega)));
//                                CEtors7 = cmn * sin_jkl * cos_ijk;
//                                CEtors8 = cmn * sin_ijk * cos_jkl;
//                                CEtors9 = fn10 * sin_ijk * sin_jkl
//                                    * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega
//                                            + fbp->V3 * (6.0 * SQR(cos_omega) - 1.50));
                                /* end  of torsion energy */

                                /* 4-body conjugation energy */
                                fn12 = exp_cot2_ij * exp_cot2_jk * exp_cot2_kl;
                                e_con_ += fbp->p_cot1 * fn12
                                    * (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);

                                Cconj = -2.0 * fn12 * fbp->p_cot1 * p_cot2
                                    * (1.0 + (SQR(cos_omega) - 1.0) * sin_ijk * sin_jkl);

                                CEconj1 = Cconj * (BOA_ij - 1.5);
                                CEconj2 = Cconj * (BOA_jk - 1.5);
                                CEconj3 = Cconj * (BOA_kl - 1.5);

                                CEconj4 = -fbp->p_cot1 * fn12
                                    * (SQR(cos_omega) - 1.0) * sin_jkl * tan_ijk_i;
                                CEconj5 = -fbp->p_cot1 * fn12
                                    * (SQR(cos_omega) - 1.0) * sin_ijk * tan_jkl_i;
//                                CEconj4 = -fbp->p_cot1 * fn12
//                                    * (SQR(cos_omega) - 1.0) * sin_jkl * cos_ijk;
//                                CEconj5 = -fbp->p_cot1 * fn12
//                                    * (SQR(cos_omega) - 1.0) * sin_ijk * cos_jkl;
                                CEconj6 = 2.0 * fbp->p_cot1 * fn12
                                    * cos_omega * sin_ijk * sin_jkl;
                                /* end 4-body conjugation energy */

                                /* forces */
                                Cdbopi_jk += CEtors2;
                                CdDelta_j += CEtors3;
                                CdDelta_k += CEtors3;
                                Cdbo_ij += (CEtors4 + CEconj1);
                                Cdbo_jk += (CEtors5 + CEconj2);
#if !defined(CUDA_ACCUM_ATOMIC)
                                atomicAdd( &pbond_kl->ta_Cdbo, CEtors6 + CEconj3 );
#else
                                atomicAdd( &bo_kl->Cdbo, CEtors6 + CEconj3 );
#endif

                                ivec_Sum( rel_box_jl, pbond_jk->rel_box, pbond_kl->rel_box );

                                /* dcos_theta_ijk */
                                rvec_Scale( temp, CEtors7 + CEconj4, p_ijk->dcos_dk );
                                rvec_Add( f_i, temp );
                                rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                                rvec_Add( ext_press_, temp );

                                rvec_ScaledAdd( f_j, CEtors7 + CEconj4, p_ijk->dcos_dj );

                                rvec_Scale( temp, CEtors7 + CEconj4, p_ijk->dcos_di );
                                rvec_Add( f_k, temp );
                                rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                rvec_Add( ext_press_, temp );

                                /* dcos_theta_jkl */
                                rvec_ScaledAdd( f_j, CEtors8 + CEconj5, p_jkl->dcos_di );

                                rvec_Scale( temp, CEtors8 + CEconj5, p_jkl->dcos_dj );
                                rvec_Add( f_k, temp );
                                rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                rvec_Add( ext_press_, temp );

                                rvec_Scale( temp, CEtors8 + CEconj5, p_jkl->dcos_dk );
#if !defined(CUDA_ACCUM_ATOMIC)
                                rvec_Add( pbond_kl->ta_f, temp );
#else
                                atomic_rvecAdd( workspace.f[l], temp );
#endif
                                rvec_iMultiply( temp, rel_box_jl, temp );
                                rvec_Add( ext_press_, temp );

                                /* dcos_omega */                      
                                rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_di );
                                rvec_Add( f_i, temp );
                                rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                                rvec_Add( ext_press_, temp );

                                rvec_ScaledAdd( f_j, CEtors9 + CEconj6, dcos_omega_dj );

                                rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_dk );
                                rvec_Add( f_k, temp );
                                rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                rvec_Add( ext_press_, temp );

                                rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_dl );
#if !defined(CUDA_ACCUM_ATOMIC)
                                rvec_Add( pbond_kl->ta_f, temp );
#else
                                atomic_rvecAdd( workspace.f[l], temp );
#endif
                                rvec_iMultiply( temp, rel_box_jl, temp );
                                rvec_Add( ext_press_, temp );
                            } // pl check ends
                        } // pl loop ends

#if !defined(CUDA_ACCUM_ATOMIC)
                        bo_ij->Cdbo += Cdbo_ij;
                        atomic_rvecAdd( pbond_ij->ta_f, f_i );
#else
                        atomicAdd( &bo_ij->Cdbo, Cdbo_ij );
                        atomic_rvecAdd( workspace.f[i], f_i );
#endif
                    } // pi check ends
                } // pi loop ends

#if !defined(CUDA_ACCUM_ATOMIC)
                bo_jk->Cdbopi += Cdbopi_jk;
                pbond_jk->ta_CdDelta += CdDelta_k;
                bo_jk->Cdbo += Cdbo_jk;
                atomic_rvecAdd( pbond_jk->ta_f, f_k );
#else
                atomicAdd( &bo_jk->Cdbopi, Cdbopi_jk );
                atomicAdd( &workspace.CdDelta[k], CdDelta_k );
                atomicAdd( &bo_jk->Cdbo, Cdbo_jk );
                atomic_rvecAdd( workspace.f[k], f_k );
#endif
            } // k-j neighbor check ends
        } // j<k && j-k neighbor check ends
    } // pk loop ends

#if !defined(CUDA_ACCUM_ATOMIC)
    rvec_Add( workspace.f[j], f_j );
    e_tor_g[j] = e_tor_;
    e_con_g[j] = e_con_;
    rvec_Copy( e_ext_press_g[j], e_ext_press_l );
#else
    atomic_rvecAdd( workspace.f[j], f_j );
    atomicAdd( (double *) e_tor_g, (double) e_tor_ );
    atomicAdd( (double *) e_con_g, (double) e_con_ );
    atomic_rvecAdd( *ext_press_g, ext_press_ );
#endif
}


#if !defined(CUDA_ACCUM_ATOMIC)
CUDA_GLOBAL void k_torsion_angles_part2( storage workspace, reax_list bond_list, int N )
{
    int i, pj;
    bond_data *pbond_ij, *pbond_ji;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        pbond_ij = &bond_list.bond_list[pj];
        pbond_ji = &bond_list.bond_list[ pbond_ij->sym_index ];

        workspace.CdDelta[i] += pbond_ji->ta_CdDelta;
        pbond_ij->bo_data.Cdbo += pbond_ij->ta_Cdbo;
        rvec_Add( workspace.f[i], pbond_ji->ta_f ); 
    }
}
#endif


void Cuda_Compute_Torsion_Angles( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list **lists,
        output_controls const * const out_control )
{
    int blocks;
#if !defined(CUDA_ACCUM_ATOMIC)
    int update_energy;
    size_t s;
    real *spad;
    rvec *rvec_spad;

    if ( control->virial == 1 )
    {
        s = (sizeof(real) * 2 + sizeof(rvec)) * system->n + sizeof(rvec) * control->blocks;
    }
    else
    {
        s = (sizeof(real) * 2 * system->n;
    }
    sCudaCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            s, __FILE__, __LINE__ );

    spad = (real *) workspace->scratch[0];
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#else
    sCudaMemsetAsync( &((simulation_data *)data->d_simulation_data)->my_en.e_tor,
            0, sizeof(real), control->streams[0], __FILE__, __LINE__ );
    sCudaMemsetAsync( &((simulation_data *)data->d_simulation_data)->my_en.e_con,
            0, sizeof(real), control->streams[0], __FILE__, __LINE__ );
    if ( control->virial == 1 )
    {
        sCudaMemsetAsync( &((simulation_data *)data->d_simulation_data)->my_ext_press,
                0, sizeof(rvec), control->streams[0], __FILE__, __LINE__ );
    }
#endif

    if ( control->virial == 1 )
    {
        k_torsion_angles_virial_part1 <<< control->blocks, control->block_size,
                                      0, control->streams[0] >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_fbp,
              (control_params *) control->d_control_params, *(lists[BONDS]),
              *(lists[THREE_BODIES]), *(workspace->d_workspace), system->n,
              system->reax_param.num_atom_types, 
#if !defined(CUDA_ACCUM_ATOMIC)
              spad, &spad[system->n], (rvec *) (&spad[2 * system->n])
#else
              &((simulation_data *)data->d_simulation_data)->my_en.e_tor,
              &((simulation_data *)data->d_simulation_data)->my_en.e_con,
              &((simulation_data *)data->d_simulation_data)->my_ext_press
#endif
            );
    }
    else
    {
//        k_torsion_angles_part1 <<< control->blocks, control->block_size,
//                               0, control->streams[0] >>>
//            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_fbp,
//              (control_params *) control->d_control_params, *(lists[BONDS]),
//              *(lists[THREE_BODIES]), *(workspace->d_workspace), system->n,
//              system->reax_param.num_atom_types, 
//#if !defined(CUDA_ACCUM_ATOMIC)
//              spad, &spad[system->n]
//#else
//              &((simulation_data *)data->d_simulation_data)->my_en.e_tor,
//              &((simulation_data *)data->d_simulation_data)->my_en.e_con
//#endif
//            );

        blocks = system->n * 32 / DEF_BLOCK_SIZE
            + (system->n * 32 % DEF_BLOCK_SIZE == 0 ? 0 : 1);

        k_torsion_angles_part1_opt <<< blocks, DEF_BLOCK_SIZE,
                                   sizeof(cub::WarpReduce<double>::TempStorage) * (DEF_BLOCK_SIZE / 32),
                                   control->streams[0] >>>
            ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_fbp,
              (control_params *) control->d_control_params, *(lists[BONDS]),
              *(lists[THREE_BODIES]), *(workspace->d_workspace), system->n,
              system->reax_param.num_atom_types, 
#if !defined(CUDA_ACCUM_ATOMIC)
              spad, &spad[system->n]
#else
              &((simulation_data *)data->d_simulation_data)->my_en.e_tor,
              &((simulation_data *)data->d_simulation_data)->my_en.e_con
#endif
            );
    }
    cudaCheckError( );

#if !defined(CUDA_ACCUM_ATOMIC)
    if ( update_energy == TRUE )
    {
        Cuda_Reduction_Sum( spad,
                &((simulation_data *)data->d_simulation_data)->my_en.e_tor,
                system->n, 0, control->streams[0] );

        Cuda_Reduction_Sum( &spad[system->n],
                &((simulation_data *)data->d_simulation_data)->my_en.e_con,
                system->n, 0, control->streams[0] );
    }

    if ( control->virial == 1 )
    {
        rvec_spad = (rvec *) (&spad[2 * system->n]);

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
//            Cuda_Reduction_Sum( rvec_spad,
//                    &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                    system->n, 0, control->streams[0] );
    }
#endif

#if !defined(CUDA_ACCUM_ATOMIC)
    k_torsion_angles_part2 <<< control->blocks_n, control->block_size_n, 0,
                           control->streams[0] >>>
            ( *(workspace->d_workspace), *(lists[BONDS]), system->N );
    cudaCheckError( );
#endif
}
