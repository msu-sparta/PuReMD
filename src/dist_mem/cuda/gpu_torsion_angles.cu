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

#include "gpu_torsion_angles.h"

#include "gpu_list.h"
#include "gpu_helpers.h"
#if !defined(GPU_ATOMIC_EV)
  #include "gpu_reduction.h"
#endif
#include "gpu_valence_angles.h"
#include "gpu_utils.h"

#include "../index_utils.h"
#include "../vector.h"

#include <cub/warp/warp_reduce.cuh>


#define MIN_SINE (1.0e-10)


GPU_DEVICE static inline real Calculate_Omega( const rvec dvec_ij, real r_ij,
        const rvec dvec_jk, real r_jk, const rvec dvec_kl, real r_kl,
        const rvec dvec_li, real r_li, real ijk_theta,
        const rvec ijk_dcos_di, const rvec ijk_dcos_dj, const rvec ijk_dcos_dk,
        real jkl_theta, const rvec jkl_dcos_di, const rvec jkl_dcos_dj, const rvec jkl_dcos_dk,
        rvec dcos_omega_di, rvec dcos_omega_dj, rvec dcos_omega_dk, rvec dcos_omega_dl )
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

    sin_ijk = SIN( ijk_theta );
    cos_ijk = COS( ijk_theta );
    sin_jkl = SIN( jkl_theta );
    cos_jkl = COS( jkl_theta );

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
    rvec_ScaledAdd( dcos_omega_di, -(hthd - arg * hnhd) / sin_ijk, ijk_dcos_dk );
    rvec_Scale( dcos_omega_di, 2.0 / poem, dcos_omega_di );

    /* dcos_omega_dj */
    rvec_ScaledSum( dcos_omega_dj, -(htra - arg * hnra) / r_ij, dvec_ij,
            -htrb / r_jk, dvec_jk );
    rvec_ScaledAdd( dcos_omega_dj, -(hthd - arg * hnhd) / sin_ijk, ijk_dcos_dj );
    rvec_ScaledAdd( dcos_omega_dj, -(hthe - arg * hnhe) / sin_jkl, jkl_dcos_di );
    rvec_Scale( dcos_omega_dj, 2.0 / poem, dcos_omega_dj );

    /* dcos_omega_dk */
    rvec_ScaledSum( dcos_omega_dk, -(htrc - arg * hnrc) / r_kl, dvec_kl,
            htrb / r_jk, dvec_jk );
    rvec_ScaledAdd( dcos_omega_dk, -(hthd - arg * hnhd) / sin_ijk, ijk_dcos_di );
    rvec_ScaledAdd( dcos_omega_dk, -(hthe - arg * hnhe) / sin_jkl, jkl_dcos_dj );
    rvec_Scale( dcos_omega_dk, 2.0 / poem, dcos_omega_dk );

    /* dcos_omega_dl */
    rvec_ScaledSum( dcos_omega_dl, (htrc - arg * hnrc) / r_kl, dvec_kl, 1.0, dvec_li );
    rvec_ScaledAdd( dcos_omega_dl, -(hthe - arg * hnhe) / sin_jkl, jkl_dcos_dk );
    rvec_Scale( dcos_omega_dl, 2.0 / poem, dcos_omega_dl );

    return omega;  
}


GPU_GLOBAL void k_torsion_angles_part1( reax_atom const * const my_atoms,
        real const * const gp_l, single_body_parameters const * const sbp,
        two_body_parameters const * const tbp, three_body_header const * const thbh,
        four_body_header const * const fbph, real thb_cut, const reax_list bond_list,
        const reax_list thb_list, real const * const Delta_boc, real * const CdDelta,
        rvec * const f, int n, int num_atom_types, real * const e_tor_g,
        real * const e_con_g )
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
#define BL (bond_list.bond_list_gpu)
#define TBL (thb_list.three_body_list_gpu)

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    type_j = my_atoms[j].type;

    if ( sbp[type_j].fbp_cnt_j > 0 )
    {
        const real p_tor2 = gp_l[23];
        const real p_tor3 = gp_l[24];
        const real p_tor4 = gp_l[25];
        const real p_cot2 = gp_l[27];
        e_tor_ = 0.0;
        e_con_ = 0.0;
        CdDelta_j = 0.0;
        rvec_MakeZero( f_j );

        Delta_j = Delta_boc[j];
        start_j = Start_Index(j, &bond_list);
        end_j = End_Index(j, &bond_list);

        for ( pk = start_j; pk < end_j; ++pk )
        {
            k = BL.nbr[pk];
            type_k = my_atoms[k].type;

            if ( tbp[index_tbp(type_j, type_k, num_atom_types)].fbp_cnt_jk > 0 )
            {
                BOA_jk = BL.BO[pk] - thb_cut;

                /* see if there are any 3-body interactions involving j and k
                 * where j is the central atom. Otherwise there is no point in
                 * trying to form a 4-body interaction out of this neighborhood */
                if ( my_atoms[j].orig_id < my_atoms[k].orig_id
                        && BL.BO[pk] > thb_cut
                        && Num_Entries(pk, &thb_list) > 0 )
                {
                    /* pj points to j on k's list */
                    pj = BL.sym_index[pk];

                    /* do the same check as above:
                     * are there any 3-body interactions
                     * involving k and j where k is the central atom */
                    if ( Num_Entries(pj, &thb_list) > 0 )
                    {
                        Delta_k = Delta_boc[k];
                        r_jk = BL.d[pk];
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
                            i = TBL.thb[pi];
                            type_i = my_atoms[i].type;

                            if ( thbh[index_thbp(type_i, type_j, type_k, num_atom_types)].fbp_cnt_ijk > 0 )
                            {
                                /* pij is pointer to i on j's bond_list */
                                pij = TBL.pthb[pi];

                                if ( BL.BO[pij] > thb_cut )
                                {
                                    r_ij = BL.d[pij];
                                    BOA_ij = BL.BO[pij] - thb_cut;
                                    Cdbo_ij = 0.0;
                                    rvec_MakeZero( f_i );

                                    theta_ijk = TBL.theta[pi];
                                    sin_ijk = SIN( theta_ijk );
                                    cos_ijk = COS( theta_ijk );
//                                    tan_ijk_i = 1.0 / TAN( theta_ijk );
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
                                        l = TBL.thb[pl];
                                        /* a pointer to l on k's bond_list! */
                                        plk = TBL.pthb[pl];
                                        type_l = my_atoms[l].type;
                                        four_body_parameters const * const fbp = &fbph[
                                            index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].prm[0];

                                        if ( i != l
                                                && fbph[ index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].cnt > 0
                                                && BL.BO[plk] > thb_cut
                                                && BL.BO[pij] * BL.BO[pk] * BL.BO[plk] > thb_cut )
                                        {
                                            r_kl = BL.d[plk];
                                            BOA_kl = BL.BO[plk] - thb_cut;

                                            theta_jkl = TBL.theta[pl];
                                            sin_jkl = SIN( theta_jkl );
                                            cos_jkl = COS( theta_jkl );
//                                            tan_jkl_i = 1.0 / TAN( theta_jkl );
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
                                            r_li = norm3d( dvec_li[0], dvec_li[1], dvec_li[2] );                 

                                            /* omega and its derivative */
//                                            cos_omega = Calculate_Omega( BL.dvec[pij], r_ij, BL.dvec[pk],
                                            omega = Calculate_Omega( BL.dvec[pij], r_ij, BL.dvec[pk], r_jk,
                                                    BL.dvec[plk], r_kl, dvec_li, r_li, TBL.theta[pi],
                                                    TBL.dcos_di[pi], TBL.dcos_dj[pi], TBL.dcos_dk[pi],
                                                    TBL.theta[pl], TBL.dcos_di[pl], TBL.dcos_dj[pl], TBL.dcos_dk[pl],
                                                    dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl );

                                            cos_omega = COS( omega );
                                            cos2omega = COS( 2.0 * omega );
                                            cos3omega = COS( 3.0 * omega );
                                            /* end omega calculations */

                                            /* torsion energy */
                                            exp_tor1 = EXP( fbp->p_tor1
                                                    * SQR(2.0 - BL.BO_pi[pk] - f11_DjDk) );
                                            exp_tor2_kl = EXP( -p_tor2 * BOA_kl );
                                            exp_cot2_kl = EXP( -p_cot2 * SQR(BOA_kl - 1.5) );
                                            fn10 = (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * 
                                                (1.0 - exp_tor2_kl);

                                            CV = 0.5 * ( fbp->V1 * (1.0 + cos_omega)
                                                    + fbp->V2 * exp_tor1 * (1.0 - cos2omega)
                                                    + fbp->V3 * (1.0 + cos3omega) );
//                                            CV = 0.5 * fbp->V1 * (1.0 + cos_omega)
//                                                + fbp->V2 * exp_tor1 * (1.0 - SQR(cos_omega))
//                                                + fbp->V3 * (0.5 + 2.0 * CUBE(cos_omega) - 1.5 * cos_omega);

                                            e_tor_ += fn10 * sin_ijk * sin_jkl * CV;

                                            dfn11 = (-p_tor3 * exp_tor3_DjDk
                                                    + (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk)
                                                    * (2.0 + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;

                                            CEtors1 = sin_ijk * sin_jkl * CV;

                                            CEtors2 = -fn10 * 2.0 * fbp->p_tor1 * fbp->V2 * exp_tor1
                                                * (2.0 - BL.BO_pi[pk] - f11_DjDk)
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
//                                            CEtors7 = cmn * sin_jkl * cos_ijk;
//                                            CEtors8 = cmn * sin_ijk * cos_jkl;
//                                            CEtors9 = fn10 * sin_ijk * sin_jkl
//                                                * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega
//                                                        + fbp->V3 * (6.0 * SQR(cos_omega) - 1.50));
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
//                                            CEconj4 = -fbp->p_cot1 * fn12
//                                                * (SQR(cos_omega) - 1.0) * sin_jkl * cos_ijk;
//                                            CEconj5 = -fbp->p_cot1 * fn12
//                                                * (SQR(cos_omega) - 1.0) * sin_ijk * cos_jkl;
                                            CEconj6 = 2.0 * fbp->p_cot1 * fn12
                                                * cos_omega * sin_ijk * sin_jkl;
                                            /* end 4-body conjugation energy */

                                            /* forces */
                                            Cdbopi_jk += CEtors2;
                                            CdDelta_j += CEtors3;
                                            CdDelta_k += CEtors3;
                                            Cdbo_ij += (CEtors4 + CEconj1);
                                            Cdbo_jk += (CEtors5 + CEconj2);
#if defined(GPU_STREAM_SINGLE_ACCUM)
                                            atomicAdd( &BL.Cdbo[plk], CEtors6 + CEconj3 );
#else
                                            atomicAdd( &BL.Cdbo_tor[plk], CEtors6 + CEconj3 );
#endif

                                            /* dcos_theta_ijk */
                                            rvec_ScaledAdd( f_i, CEtors7 + CEconj4, TBL.dcos_dk[pi] );
                                            rvec_ScaledAdd( f_j, CEtors7 + CEconj4, TBL.dcos_dj[pi] );
                                            rvec_ScaledAdd( f_k, CEtors7 + CEconj4, TBL.dcos_di[pi] );

                                            /* dcos_theta_jkl */
                                            rvec_ScaledAdd( f_j, CEtors8 + CEconj5, TBL.dcos_di[pl] );
                                            rvec_ScaledAdd( f_k, CEtors8 + CEconj5, TBL.dcos_dj[pl] );
#if defined(GPU_KERNEL_ATOMIC)
                                            atomic_rvecScaledAdd( f[l], CEtors8 + CEconj5, TBL.dcos_dk[pl] );
#else
                                            atomic_rvecScaledAdd( BL.f_tor[plk], CEtors8 + CEconj5, TBL.dcos_dk[pl] );
#endif

                                            /* dcos_omega */
                                            rvec_ScaledAdd( f_i, CEtors9 + CEconj6, dcos_omega_di );
                                            rvec_ScaledAdd( f_j, CEtors9 + CEconj6, dcos_omega_dj );
                                            rvec_ScaledAdd( f_k, CEtors9 + CEconj6, dcos_omega_dk );
#if defined(GPU_KERNEL_ATOMIC)
                                            atomic_rvecScaledAdd( f[l], CEtors9 + CEconj6, dcos_omega_dl );
#else
                                            atomic_rvecScaledAdd( BL.f_tor[plk], CEtors9 + CEconj6, dcos_omega_dl );
#endif
                                        } // pl check ends
                                    } // pl loop ends

#if defined(GPU_STREAM_SINGLE_ACCUM)
                                atomicAdd( &BL.Cdbo[pij], Cdbo_ij );
#else
                                atomicAdd( &BL.Cdbo_tor[pij], Cdbo_ij );
#endif
#if defined(GPU_KERNEL_ATOMIC)
                                atomic_rvecAdd( f[i], f_i );
#else
                                atomic_rvecAdd( BL.f_tor[pij], f_i );
#endif
                                } // pi check ends
                            }
                        } // pi loop ends

#if defined(GPU_STREAM_SINGLE_ACCUM)
                        atomicAdd( &BL.Cdbo[pk], Cdbo_jk );
#else
                        atomicAdd( &BL.Cdbo_tor[pk], Cdbo_jk );
#endif
#if defined(GPU_STREAM_SINGLE_ACCUM)
                        atomicAdd( &BL.Cdbopi[pk], Cdbopi_jk );
#else
                        BL.Cdbopi_tor[pk] = Cdbopi_jk;
#endif
#if defined(GPU_KERNEL_ATOMIC)
                        atomicAdd( &CdDelta[k], CdDelta_k );
                        atomic_rvecAdd( f[k], f_k );
#else
                        atomicAdd( &BL.CdDelta_tor[pk], CdDelta_k );
                        atomic_rvecAdd( BL.f_tor[pk], f_k );
#endif
                    } // k-j neighbor check ends
                } // j<k && j-k neighbor check ends
            } 
        } // pk loop ends

#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
        atomicAdd( &CdDelta[j], CdDelta_j );
        atomic_rvecAdd( f[j], f_j );
#else
        CdDelta[j] += CdDelta_j;
        rvec_Add( f[j], f_j );
#endif
#if defined(GPU_ATOMIC_EV)
        atomicAdd( (double *) e_tor_g, (double) e_tor_ );
        atomicAdd( (double *) e_con_g, (double) e_con_ );
#else
        e_tor_g[j] = e_tor_;
        e_con_g[j] = e_con_;
#endif
    }

#undef BL
#undef TBL
}


GPU_GLOBAL void k_torsion_angles_part1_opt( reax_atom const * const my_atoms,
        real const * const gp_l, single_body_parameters const * const sbp,
        two_body_parameters const * const tbp, three_body_header const * const thbh,
        four_body_header const * const fbph, real thb_cut,
        const reax_list bond_list, const reax_list thb_list,
        real const * const Delta_boc, real * const CdDelta,
        rvec * const f, int n, int num_atom_types, 
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
#define BL (bond_list.bond_list_gpu)
#define TBL (thb_list.three_body_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    j = thread_id / warpSize;

    if ( j >= n )
    {
        return;
    }

    type_j = my_atoms[j].type;

    if ( sbp[type_j].fbp_cnt_j > 0 )
    {
        warp_id = threadIdx.x / warpSize;
        lane_id = thread_id % warpSize;
        real const p_tor2 = gp_l[23];
        real const p_tor3 = gp_l[24];
        real const p_tor4 = gp_l[25];
        real const p_cot2 = gp_l[27];
        e_tor_ = 0.0;
        e_con_ = 0.0;
        CdDelta_j = 0.0;
        rvec_MakeZero( f_j );

        Delta_j = Delta_boc[j];
        start_j = Start_Index(j, &bond_list);
        end_j = End_Index(j, &bond_list);

        for ( pk = start_j; pk < end_j; ++pk )
        {
            k = BL.nbr[pk];
            type_k = my_atoms[k].type;

            if ( tbp[index_tbp(type_j, type_k, num_atom_types)].fbp_cnt_jk > 0 )
            {
                BOA_jk = BL.BO[pk] - thb_cut;

                /* see if there are any 3-body interactions involving j and k
                 * where j is the central atom. Otherwise there is no point in
                 * trying to form a 4-body interaction out of this neighborhood */
                if ( my_atoms[j].orig_id < my_atoms[k].orig_id
                        && BL.BO[pk] > thb_cut
                        && Num_Entries(pk, &thb_list) > 0 )
                {
                    /* pj points to j on k's list */
                    pj = BL.sym_index[pk];

                    /* do the same check as above:
                     * are there any 3-body interactions
                     * involving k and j where k is the central atom */
                    if ( Num_Entries(pj, &thb_list) > 0 )
                    {
                        Delta_k = Delta_boc[k];
                        r_jk = BL.d[pk];
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
                            /* pij is pointer to i on j's bond_list */
                            pij = TBL.pthb[pi];
                            i = TBL.thb[pi];
                            type_i = my_atoms[i].type;

                            if ( thbh[index_thbp(type_i, type_j, type_k, num_atom_types)].fbp_cnt_ijk > 0 )
                            {

                                if ( BL.BO[pij] > thb_cut )
                                {
                                    r_ij = BL.d[pij];
                                    BOA_ij = BL.BO[pij] - thb_cut;
                                    Cdbo_ij = 0.0;
                                    rvec_MakeZero( f_i );

                                    theta_ijk = TBL.theta[pi];
                                    sin_ijk = SIN( theta_ijk );
                                    cos_ijk = COS( theta_ijk );
//                                    tan_ijk_i = 1.0 / TAN( theta_ijk );
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
                                            l = TBL.thb[pl];
                                            /* a pointer to l on k's bond_list! */
                                            plk = TBL.pthb[pl];
                                            type_l = my_atoms[l].type;
                                            four_body_parameters const * const fbp = &fbph[
                                                index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].prm[0];

                                            if ( i != l
                                                    && fbph[ index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].cnt > 0
                                                    && BL.BO[plk] > thb_cut
                                                    && BL.BO[pij] * BL.BO[pk] * BL.BO[plk] > thb_cut )
                                            {
                                                r_kl = BL.d[plk];
                                                BOA_kl = BL.BO[plk] - thb_cut;

                                                theta_jkl = TBL.theta[pl];
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
                                                r_li = norm3d( dvec_li[0], dvec_li[1], dvec_li[2] );                 

                                                /* omega and its derivative */
//                                                cos_omega = Calculate_Omega( BL.dvec[pij], r_ij, BL.dvec[pk],
                                                omega = Calculate_Omega( BL.dvec[pij], r_ij, BL.dvec[pk], r_jk,
                                                        BL.dvec[plk], r_kl, dvec_li, r_li, TBL.theta[pi],
                                                        TBL.dcos_di[pi], TBL.dcos_dj[pi], TBL.dcos_dk[pi],
                                                        TBL.theta[pl], TBL.dcos_di[pl], TBL.dcos_dj[pl], TBL.dcos_dk[pl],
                                                        dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl );

                                                cos_omega = COS( omega );
                                                cos2omega = COS( 2.0 * omega );
                                                cos3omega = COS( 3.0 * omega );
                                                /* end omega calculations */

                                                /* torsion energy */
                                                exp_tor1 = EXP( fbp->p_tor1
                                                        * SQR(2.0 - BL.BO_pi[pk] - f11_DjDk) );
                                                exp_tor2_kl = EXP( -p_tor2 * BOA_kl );
                                                exp_cot2_kl = EXP( -p_cot2 * SQR(BOA_kl - 1.5) );
                                                fn10 = (1.0 - exp_tor2_ij) * (1.0 - exp_tor2_jk) * 
                                                    (1.0 - exp_tor2_kl);

                                                CV = 0.5 * ( fbp->V1 * (1.0 + cos_omega)
                                                        + fbp->V2 * exp_tor1 * (1.0 - cos2omega)
                                                        + fbp->V3 * (1.0 + cos3omega) );
//                                                CV = 0.5 * fbp->V1 * (1.0 + cos_omega)
//                                                    + fbp->V2 * exp_tor1 * (1.0 - SQR(cos_omega))
//                                                    + fbp->V3 * (0.5 + 2.0 * CUBE(cos_omega) - 1.5 * cos_omega);

                                                e_tor_ += fn10 * sin_ijk * sin_jkl * CV;

                                                dfn11 = (-p_tor3 * exp_tor3_DjDk
                                                        + (p_tor3 * exp_tor3_DjDk - p_tor4 * exp_tor4_DjDk)
                                                        * (2.0 + exp_tor3_DjDk) * exp_tor34_inv) * exp_tor34_inv;

                                                CEtors1 = sin_ijk * sin_jkl * CV;

                                                CEtors2 = -fn10 * 2.0 * fbp->p_tor1 * fbp->V2 * exp_tor1
                                                    * (2.0 - BL.BO_pi[pk] - f11_DjDk)
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
//                                                CEtors7 = cmn * sin_jkl * cos_ijk;
//                                                CEtors8 = cmn * sin_ijk * cos_jkl;
//                                                CEtors9 = fn10 * sin_ijk * sin_jkl
//                                                    * (0.5 * fbp->V1 - 2.0 * fbp->V2 * exp_tor1 * cos_omega
//                                                            + fbp->V3 * (6.0 * SQR(cos_omega) - 1.50));
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
//                                                CEconj4 = -fbp->p_cot1 * fn12
//                                                    * (SQR(cos_omega) - 1.0) * sin_jkl * cos_ijk;
//                                                CEconj5 = -fbp->p_cot1 * fn12
//                                                    * (SQR(cos_omega) - 1.0) * sin_ijk * cos_jkl;
                                                CEconj6 = 2.0 * fbp->p_cot1 * fn12
                                                    * cos_omega * sin_ijk * sin_jkl;
                                                /* end 4-body conjugation energy */

                                                /* forces */
                                                Cdbopi_jk += CEtors2;
                                                CdDelta_j += CEtors3;
                                                CdDelta_k += CEtors3;
                                                Cdbo_ij += (CEtors4 + CEconj1);
                                                Cdbo_jk += (CEtors5 + CEconj2);
#if defined(GPU_STREAM_SINGLE_ACCUM)
                                                atomicAdd( &BL.Cdbo[plk], CEtors6 + CEconj3 );
#else
                                                atomicAdd( &BL.Cdbo_tor[plk], CEtors6 + CEconj3 );
#endif

                                                /* dcos_theta_ijk */
                                                rvec_ScaledAdd( f_i, CEtors7 + CEconj4, TBL.dcos_dk[pi] );
                                                rvec_ScaledAdd( f_j, CEtors7 + CEconj4, TBL.dcos_dj[pi] );
                                                rvec_ScaledAdd( f_k, CEtors7 + CEconj4, TBL.dcos_di[pi] );

                                                /* dcos_theta_jkl */
                                                rvec_ScaledAdd( f_j, CEtors8 + CEconj5, TBL.dcos_di[pl] );
                                                rvec_ScaledAdd( f_k, CEtors8 + CEconj5, TBL.dcos_dj[pl] );
#if defined(GPU_KERNEL_ATOMIC)
                                                atomic_rvecScaledAdd( f[l], CEtors8 + CEconj5, TBL.dcos_dk[pl] );
#else
                                                atomic_rvecScaledAdd( BL.f_tor[plk], CEtors8 + CEconj5, TBL.dcos_dk[pl] );
#endif

                                                /* dcos_omega */
                                                rvec_ScaledAdd( f_i, CEtors9 + CEconj6, dcos_omega_di );
                                                rvec_ScaledAdd( f_j, CEtors9 + CEconj6, dcos_omega_dj );
                                                rvec_ScaledAdd( f_k, CEtors9 + CEconj6, dcos_omega_dk );
#if defined(GPU_KERNEL_ATOMIC)
                                                atomic_rvecScaledAdd( f[l], CEtors9 + CEconj6, dcos_omega_dl );
#else
                                                atomic_rvecScaledAdd( BL.f_tor[plk], CEtors9 + CEconj6, dcos_omega_dl );
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
#if defined(GPU_STREAM_SINGLE_ACCUM)
                                        atomicAdd( &BL.Cdbo[pij], Cdbo_ij );
#else
                                        atomicAdd( &BL.Cdbo_tor[pij], Cdbo_ij );
#endif
#if defined(GPU_KERNEL_ATOMIC)
                                        atomic_rvecAdd( f[i], f_i );
#else
                                        atomic_rvecAdd( BL.f_tor[pij], f_i );
#endif
                                    }
                                } // pi check ends
                            }
                        } // pi loop ends

                        Cdbopi_jk = cub::WarpReduce<double>(temp_d[warp_id]).Sum(Cdbopi_jk);
                        CdDelta_k = cub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_k);
                        Cdbo_jk = cub::WarpReduce<double>(temp_d[warp_id]).Sum(Cdbo_jk);
                        f_k[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[0]);
                        f_k[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[1]);
                        f_k[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_k[2]);

                        if ( lane_id == 0 )
                        {
#if defined(GPU_STREAM_SINGLE_ACCUM)
                            atomicAdd( &BL.Cdbo[pk], Cdbo_jk );
#else
                            atomicAdd( &BL.Cdbo_tor[pk], Cdbo_jk );
#endif
#if defined(GPU_STREAM_SINGLE_ACCUM)
                            atomicAdd( &BL.Cdbopi[pk], Cdbopi_jk );
#else
                            BL.Cdbopi_tor[pk] = Cdbopi_jk;
#endif
#if defined(GPU_KERNEL_ATOMIC)
                            atomicAdd( &CdDelta[k], CdDelta_k );
                            atomic_rvecAdd( f[k], f_k );
#else
                            atomicAdd( &BL.CdDelta_tor[pk], CdDelta_k );
                            atomic_rvecAdd( BL.f_tor[pk], f_k );
#endif
                        }
                    } // k-j neighbor check ends
                } // j<k && j-k neighbor check ends
            }
        } // pk loop ends

        CdDelta_j = cub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_j);
        f_j[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[0]);
        f_j[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[1]);
        f_j[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[2]);
        e_tor_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_tor_);
        e_con_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_con_);

        if ( lane_id == 0 )
        {
#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
            atomicAdd( &CdDelta[j], CdDelta_j );
            atomic_rvecAdd( f[j], f_j );
#else
            CdDelta[j] += CdDelta_j;
            rvec_Add( f[j], f_j );
#endif
#if defined(GPU_ATOMIC_EV)
            atomicAdd( (double *) e_tor_g, (double) e_tor_ );
            atomicAdd( (double *) e_con_g, (double) e_con_ );
#else
            e_tor_g[j] = e_tor_;
            e_con_g[j] = e_con_;
#endif
        }
    }

#undef BL
}


GPU_GLOBAL void k_torsion_angles_virial_part1( reax_atom const * const my_atoms,
        real const * const gp_l, four_body_header const * const fbph,
        real thb_cut, const reax_list bond_list, const reax_list thb_list,
        real const * const Delta_boc, real * const CdDelta,
        rvec * const f, int n, int num_atom_types, real * const e_tor_g,
        real * const e_con_g, rvec * const ext_press_g )
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
#define BL (bond_list.bond_list_gpu)

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= n )
    {
        return;
    }

    real const p_tor2 = gp_l[23];
    real const p_tor3 = gp_l[24];
    real const p_tor4 = gp_l[25];
    real const p_cot2 = gp_l[27];
    e_tor_ = 0.0;
    e_con_ = 0.0;
    CdDelta_j = 0.0;
    rvec_MakeZero( f_j );
    rvec_MakeZero( ext_press_ );

    type_j = my_atoms[j].type;
    Delta_j = Delta_boc[j];
    start_j = Start_Index(j, &bond_list);
    end_j = End_Index(j, &bond_list);

    for ( pk = start_j; pk < end_j; ++pk )
    {
        k = BL.nbr[pk];
        BOA_jk = BL.BO[pk] - thb_cut;

        /* see if there are any 3-body interactions involving j and k
         * where j is the central atom. Otherwise there is no point in
         * trying to form a 4-body interaction out of this neighborhood */
        if ( my_atoms[j].orig_id < my_atoms[k].orig_id
                && BL.BO[pk] > thb_cut
                && Num_Entries(pk, &thb_list) )
        {
            /* pj points to j on k's list */
            pj = BL.sym_index[pk];

            /* do the same check as above:
             * are there any 3-body interactions
             * involving k and j where k is the central atom */
            if ( Num_Entries(pj, &thb_list) > 0 )
            {
                type_k = my_atoms[k].type;
                Delta_k = Delta_boc[k];
                r_jk = BL.d[pk];
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
                    /* pij is pointer to i on j's bond_list */
                    pij = TBL.pthb[pi];

                    if ( BL.BO[pij] > thb_cut )
                    {
                        i = TBL.thb[pi];
                        type_i = my_atoms[i].type;
                        r_ij = BL.d[pij];
                        BOA_ij = BL.BO[pij] - thb_cut;
                        Cdbo_ij = 0.0;
                        rvec_MakeZero( f_i );

                        theta_ijk = TBL.theta[pi];
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
                            l = TBL.thb[pl];
                            /* a pointer to l on k's bond_list! */
                            plk = TBL.pthb[pl];
                            type_l = my_atoms[l].type;
                            four_body_parameters const * const fbp = &fbph[
                                index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].prm[0];

                            if ( i != l
                                    && fbph[ index_fbp(type_i, type_j, type_k, type_l, num_atom_types) ].cnt > 0
                                    && BL.BO[plk] > thb_cut
                                    && BL.BO[pij] * BL.BO[pk] * BL.BO[plk] > thb_cut )
                            {
                                r_kl = BL.d[plk];
                                BOA_kl = BL.BO[plk] - thb_cut;

                                theta_jkl = TBL.theta[pl];
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
                                r_li = norm3d( dvec_li[0], dvec_li[1], dvec_li[2] );                 

                                /* omega and its derivative */
//                                cos_omega = Calculate_Omega( BL.dvec[pij], r_ij, BL.dvec[pk],
                                omega = Calculate_Omega( BL.dvec[pij], r_ij, BL.dvec[pk], r_jk,
                                        BL.dvec[plk], r_kl, dvec_li, r_li, TBL.theta[pi],
                                        TBL.dcos_di[pi], TBL.dcos_dj[pi], TBL.dcos_dk[pi],
                                        TBL.theta[pl], TBL.dcos_di[pl], TBL.dcos_dj[pl], TBL.dcos_dk[pl],
                                        dcos_omega_di, dcos_omega_dj, dcos_omega_dk, dcos_omega_dl );

                                cos_omega = COS( omega );
                                cos2omega = COS( 2.0 * omega );
                                cos3omega = COS( 3.0 * omega );
                                /* end omega calculations */

                                /* torsion energy */
                                exp_tor1 = EXP( fbp->p_tor1
                                        * SQR(2.0 - BL.BO_pi[pk] - f11_DjDk) );
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
                                    * (2.0 - BL.BO_pi[pk] - f11_DjDk)
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
#if defined(GPU_STREAM_SINGLE_ACCUM)
                                atomicAdd( &BL.Cdbo[plk], CEtors6 + CEconj3 );
#else
                                atomicAdd( &BL.Cdbo_tor[plk], CEtors6 + CEconj3 );
#endif

                                ivec_Sum( rel_box_jl, BL.rel_box[pk], BL.rel_box[plk] );

                                /* dcos_theta_ijk */
                                rvec_Scale( temp, CEtors7 + CEconj4, TBL.dcos_dk[pi] );
                                rvec_Add( f_i, temp );
                                rvec_iMultiply( temp, BL.rel_box[pij], temp );
                                rvec_Add( ext_press_, temp );

                                rvec_ScaledAdd( f_j, CEtors7 + CEconj4, TBL.dcos_dj[pi] );

                                rvec_Scale( temp, CEtors7 + CEconj4, TBL.dcos_di[pi] );
                                rvec_Add( f_k, temp );
                                rvec_iMultiply( temp, BL.rel_box[pk], temp );
                                rvec_Add( ext_press_, temp );

                                /* dcos_theta_jkl */
                                rvec_ScaledAdd( f_j, CEtors8 + CEconj5, TBL.dcos_di[pl] );

                                rvec_Scale( temp, CEtors8 + CEconj5, TBL.dcos_dj[pl] );
                                rvec_Add( f_k, temp );
                                rvec_iMultiply( temp, BL.rel_box[pk], temp );
                                rvec_Add( ext_press_, temp );

                                rvec_Scale( temp, CEtors8 + CEconj5, TBL.dcos_dk[pl] );
#if defined(GPU_KERNEL_ATOMIC)
                                atomic_rvecAdd( f[l], temp );
#else
                                atomic_rvecAdd( BL.f_tor[plk], temp );
#endif
                                rvec_iMultiply( temp, rel_box_jl, temp );
                                rvec_Add( ext_press_, temp );

                                /* dcos_omega */                      
                                rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_di );
                                rvec_Add( f_i, temp );
                                rvec_iMultiply( temp, BL.rel_box[pij], temp );
                                rvec_Add( ext_press_, temp );

                                rvec_ScaledAdd( f_j, CEtors9 + CEconj6, dcos_omega_dj );

                                rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_dk );
                                rvec_Add( f_k, temp );
                                rvec_iMultiply( temp, BL.rel_box[pk], temp );
                                rvec_Add( ext_press_, temp );

                                rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_dl );
#if defined(GPU_KERNEL_ATOMIC)
                                atomic_rvecAdd( f[l], temp );
#else
                                atomic_rvecAdd( BL.f_tor[plk], temp );
#endif
                                rvec_iMultiply( temp, rel_box_jl, temp );
                                rvec_Add( ext_press_, temp );
                            } // pl check ends
                        } // pl loop ends

#if defined(GPU_STREAM_SINGLE_ACCUM)
                        atomicAdd( &BL.Cdbo[pij], Cdbo_ij );
#else
                        atomicAdd( &BL.Cdbo_tor[pij], Cdbo_ij );
#endif
#if defined(GPU_KERNEL_ATOMIC)
                        atomic_rvecAdd( f[i], f_i );
#else
                        atomic_rvecAdd( BL.f_tor[pij], f_i );
#endif
                    } // pi check ends
                } // pi loop ends

#if defined(GPU_STREAM_SINGLE_ACCUM)
                atomicAdd( &BL.Cdbo[pk], Cdbo_jk );
#else
                atomicAdd( &BL.Cdbo_tor[pk], Cdbo_jk );
#endif
#if defined(GPU_STREAM_SINGLE_ACCUM)
                atomicAdd( &BL.Cdbopi[pk], Cdbopi_jk );
#else
                BL.Cdbopi_tor[pk] = Cdbopi_jk;
#endif
#if defined(GPU_KERNEL_ATOMIC)
                atomicAdd( &CdDelta[k], CdDelta_k );
                atomic_rvecAdd( f[k], f_k );
#else
                BL.CdDelta_tor[pk] += CdDelta_k;
                atomic_rvecAdd( BL.f_tor[pk], f_k );
#endif
            } // k-j neighbor check ends
        } // j<k && j-k neighbor check ends
    } // pk loop ends

#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
    atomicAdd( &CdDelta[j], CdDelta_j );
    atomic_rvecAdd( f[j], f_j );
#else
    CdDelta[j] += CdDelta_j;
    rvec_Add( f[j], f_j );
#endif
#if defined(GPU_ATOMIC_EV)
    atomicAdd( (double *) e_tor_g, (double) e_tor_ );
    atomicAdd( (double *) e_con_g, (double) e_con_ );
    atomic_rvecAdd( *ext_press_g, ext_press_ );
#else
    e_tor_g[j] = e_tor_;
    e_con_g[j] = e_con_;
    rvec_Copy( ext_press_g[j], ext_press_ );
#endif

#undef BL
}


#if !defined(GPU_KERNEL_ATOMIC)
GPU_GLOBAL void k_torsion_angles_part2( real * const CdDelta, rvec * const f,
        reax_list bond_list, int N )
{
    int i, pj;
    real CdDelta_i;
    rvec f_i;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    CdDelta_i = 0.0;
    rvec_MakeZero( f_i );

    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        CdDelta_i += BL.CdDelta_tor[BL.sym_index[pj]];
        rvec_Add( f_i, BL.f_tor[BL.sym_index[pj]] ); 
    }

#if defined(GPU_STREAM_SINGLE_ACCUM)
    atomicAdd( &CdDelta[i], CdDelta_i );
    atomic_rvecAdd( f[i], f_i ); 
#else
    CdDelta[i] += CdDelta_i;
    rvec_Add( f[i], f_i ); 
#endif

#undef BL
}
#endif


void GPU_Compute_Torsion_Angles( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list **lists,
        output_controls const * const out_control )
{
#if !defined(GPU_ATOMIC_EV)
    int update_energy;
    size_t s;
    real *spad;
    rvec *rvec_spad;
#endif

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->gpu_time_events[TE_TORSION_START], control->gpu_streams[3] );
#endif

#if defined(GPU_ATOMIC_EV)
    sCudaMemsetAsync( &data->d_my_en[E_TOR], 0, sizeof(real) * 2,
            control->gpu_streams[3], __FILE__, __LINE__ );
    if ( control->virial == 1 )
    {
        sCudaMemsetAsync( &data->d_my_ext_press,
                0, sizeof(rvec), control->gpu_streams[3], __FILE__, __LINE__ );
    }
#else
    if ( control->virial == 1 )
    {
        s = (sizeof(real) * 2 + sizeof(rvec)) * system->n;
    }
    else
    {
        s = (sizeof(real) * 2) * system->n;
    }
    sCudaCheckMalloc( &workspace->d_workspace->scratch[3],
            &workspace->d_workspace->scratch_size[3], s, __FILE__, __LINE__ );

    spad = (real *) workspace->d_workspace->scratch[3];
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#endif

    if ( control->virial == 0 )
    {
//        k_torsion_angles_part1 <<< control->blocks_n, control->gpu_block_size,
//                               0, control->gpu_streams[3] >>>
//            ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp,
//              system->reax_param.d_tbp, system->reax_param.d_thbp, system->reax_param.d_fbp,
//              control->thb_cut, *(lists[BONDS]), *(lists[THREE_BODIES]),
//              workspace->d_workspace->Delta_boc,
//#if defined(GPU_STREAM_SINGLE_ACCUM)
//              workspace->d_workspace->CdDelta, workspace->d_workspace->f,
//#else
//              workspace->d_workspace->CdDelta_tor, workspace->d_workspace->f_tor,
//#endif
//              system->n, system->reax_param.num_atom_types, 
//#if defined(GPU_ATOMIC_EV)
//              &data->d_my_en[E_TOR], &data->d_my_en[E_CON]
//#else
//              spad, &spad[system->n]
//#endif
//            );

        k_torsion_angles_part1_opt <<< control->blocks_warp_n, control->gpu_block_size,
                                   sizeof(cub::WarpReduce<double>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                                   control->gpu_streams[3] >>>
            ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp,
              system->reax_param.d_tbp, system->reax_param.d_thbp, system->reax_param.d_fbp,
              control->thb_cut, *(lists[BONDS]), *(lists[THREE_BODIES]),
              workspace->d_workspace->Delta_boc,
#if defined(GPU_STREAM_SINGLE_ACCUM)
              workspace->d_workspace->CdDelta, workspace->d_workspace->f,
#else
              workspace->d_workspace->CdDelta_tor, workspace->d_workspace->f_tor,
#endif
              system->n, system->reax_param.num_atom_types, 
#if defined(GPU_ATOMIC_EV)
              &data->d_my_en[E_TOR], &data->d_my_en[E_CON]
#else
              spad, &spad[system->n]
#endif
            );

//        k_torsion_angles_part1_no_cache_opt <<< control->blocks_warp_n, control->gpu_block_size,
//                                   sizeof(cub::WarpReduce<double>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
//                                   control->gpu_streams[3] >>>
//            ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp,
//              system->reax_param.d_tbp, system->reax_param.d_thbp, system->reax_param.d_fbp,
//              control->thb_cut, *(lists[BONDS]), workspace->d_workspace->Delta_boc,
//#if defined(GPU_STREAM_SINGLE_ACCUM)
//              workspace->d_workspace->CdDelta, workspace->d_workspace->f,
//#else
//              workspace->d_workspace->CdDelta_tor, workspace->d_workspace->f_tor,
//#endif
//              system->n, system->reax_param.num_atom_types, 
//#if defined(GPU_ATOMIC_EV)
//              &data->d_my_en[E_TOR], &data->d_my_en[E_CON]
//#else
//              spad, &spad[system->n]
//#endif
//            );
    }
    else if ( control->virial == 1 )
    {
        k_torsion_angles_virial_part1 <<< control->blocks_n, control->gpu_block_size,
                                      0, control->gpu_streams[3] >>>
            ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_fbp,
              control->thb_cut, *(lists[BONDS]), *(lists[THREE_BODIES]),
              workspace->d_workspace->Delta_boc,
#if defined(GPU_STREAM_SINGLE_ACCUM)
              workspace->d_workspace->CdDelta, workspace->d_workspace->f,
#else
              workspace->d_workspace->CdDelta_tor, workspace->d_workspace->f_tor,
#endif
              system->n, system->reax_param.num_atom_types, 
#if defined(GPU_ATOMIC_EV)
              &data->d_my_en[E_TOR], &data->d_my_en[E_CON],
              &data->d_my_ext_press
#else
              spad, &spad[system->n], (rvec *) (&spad[2 * system->n])
#endif
            );
    }
    cudaCheckError( );

#if !defined(GPU_ATOMIC_EV)
    if ( update_energy == TRUE )
    {
        GPU_Reduction_Sum( spad, &data->d_my_en[E_TOR], system->n, 3,
                control->gpu_streams[3] );

        GPU_Reduction_Sum( &spad[system->n], &data->d_my_en[E_CON], system->n, 3,
                control->gpu_streams[3] );
    }

    if ( control->virial == 1 )
    {
        rvec_spad = (rvec *) (&spad[2 * system->n]);

        GPU_Reduction_Sum( rvec_spad, &data->d_my_ext_press,
                system->n, 3, control->gpu_streams[3] );
    }
#endif

#if !defined(GPU_KERNEL_ATOMIC)
    k_torsion_angles_part2 <<< control->blocks_N, control->gpu_block_size,
                           0, control->gpu_streams[3] >>>
#if defined(GPU_STREAM_SINGLE_ACCUM)
            ( workspace->d_workspace->CdDelta, workspace->d_workspace->f,
#else
            ( workspace->d_workspace->CdDelta_tor, workspace->d_workspace->f_tor,
#endif
              *(lists[BONDS]), system->N );
    cudaCheckError( );
#endif

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->gpu_time_events[TE_TORSION_STOP], control->gpu_streams[3] );
#endif
}
