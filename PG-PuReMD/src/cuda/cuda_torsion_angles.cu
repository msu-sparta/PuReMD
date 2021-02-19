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

#include "../index_utils.h"
#include "../vector.h"

#define MIN_SINE (1.0e-10)


CUDA_DEVICE static real Calculate_Omega( const rvec dvec_ij, real r_ij, const rvec dvec_jk, real r_jk,
        const rvec dvec_kl, real r_kl, const rvec dvec_li, real r_li,
        const three_body_interaction_data * const p_ijk, const three_body_interaction_data * const p_jkl,
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


CUDA_GLOBAL void Cuda_Torsion_Angles_Part1( reax_atom *my_atoms, global_parameters gp, 
        four_body_header *d_fbp, control_params *control, reax_list bond_list,
        reax_list thb_list, storage workspace, int n, int num_atom_types, 
        real *e_tor_g, real *e_con_g, rvec *ext_press_g )
{
    int i, j, k, l, pi, pj, pk, pl, pij, plk;
    int type_i, type_j, type_k, type_l;
    int start_j, end_j;
    int start_pj, end_pj, start_pk, end_pk;
#if defined(DEBUG_FOCUS)
    int num_frb_intrs;
#endif
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
    real CEconj4, CEconj5, CEconj6, e_tor_l, e_con_l;
    rvec dvec_li, temp, f_j_l, ext_press_l;
    ivec rel_box_jl;
    four_body_header *fbh;
    four_body_parameters *fbp;
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
    e_tor_l = 0.0;
    e_con_l = 0.0;
    rvec_MakeZero( f_j_l );
    rvec_MakeZero( ext_press_l );
#if defined(DEBUG_FOCUS)
    num_frb_intrs = 0;
#endif

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

                        theta_ijk = p_ijk->theta;
                        sin_ijk = SIN( theta_ijk );
                        cos_ijk = COS( theta_ijk );
                        //tan_ijk_i = 1.0 / TAN( theta_ijk );
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
                            fbh = &d_fbp[
                                index_fbp(type_i,type_j,type_k,type_l,num_atom_types) ];
                            fbp = &d_fbp[
                                index_fbp(type_i,type_j,type_k,type_l,num_atom_types) ].prm[0];

                            if ( i != l && fbh->cnt > 0
                                    && bo_kl->BO > control->thb_cut
                                    && bo_ij->BO * bo_jk->BO * bo_kl->BO > control->thb_cut )
                            {
#if defined(DEBUG_FOCUS)
                                ++num_frb_intrs;
#endif

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
                                //cos_omega = Calculate_Omega( pbond_ij->dvec, r_ij, pbond_jk->dvec,
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

                                e_tor_l += fn10 * sin_ijk * sin_jkl * CV;

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
                                e_con_l += fbp->p_cot1 * fn12
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
#if !defined(CUDA_ACCUM_ATOMIC)
                                bo_jk->Cdbopi += CEtors2;
                                workspace.CdDelta[j] += CEtors3;
                                pbond_jk->ta_CdDelta += CEtors3;
                                bo_ij->Cdbo += (CEtors4 + CEconj1);
                                bo_jk->Cdbo += (CEtors5 + CEconj2);
                                atomicAdd( &pbond_kl->ta_Cdbo, CEtors6 + CEconj3 );
#else
                                atomicAdd( &bo_jk->Cdbopi, CEtors2 );
                                atomicAdd( &workspace.CdDelta[j], CEtors3 );
                                atomicAdd( &workspace.CdDelta[k], CEtors3 );
                                atomicAdd( &bo_ij->Cdbo, CEtors4 + CEconj1 );
                                atomicAdd( &bo_jk->Cdbo, CEtors5 + CEconj2 );
                                atomicAdd( &bo_kl->Cdbo, CEtors6 + CEconj3 );
#endif

                                if ( control->virial == 0 )
                                {
#if !defined(CUDA_ACCUM_ATOMIC)
                                    /* dcos_theta_ijk */
                                    atomic_rvecScaledAdd( pbond_ij->ta_f, 
                                            CEtors7 + CEconj4, p_ijk->dcos_dk );
                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors7 + CEconj4, p_ijk->dcos_dj );
                                    atomic_rvecScaledAdd( pbond_jk->ta_f,
                                            CEtors7 + CEconj4, p_ijk->dcos_di );

                                    /* dcos_theta_jkl */
                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors8 + CEconj5, p_jkl->dcos_di );
                                    atomic_rvecScaledAdd( pbond_jk->ta_f,
                                            CEtors8 + CEconj5, p_jkl->dcos_dj );
                                    atomic_rvecScaledAdd( pbond_kl->ta_f, 
                                            CEtors8 + CEconj5, p_jkl->dcos_dk );

                                    /* dcos_omega */
                                    atomic_rvecScaledAdd( pbond_ij->ta_f,
                                            CEtors9 + CEconj6, dcos_omega_di );
                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors9 + CEconj6, dcos_omega_dj );
                                    atomic_rvecScaledAdd( pbond_jk->ta_f,
                                            CEtors9 + CEconj6, dcos_omega_dk );
                                    atomic_rvecScaledAdd( pbond_kl->ta_f,
                                            CEtors9 + CEconj6, dcos_omega_dl );
#else
                                    /* dcos_theta_ijk */
                                    atomic_rvecScaledAdd( workspace.f[i], 
                                            CEtors7 + CEconj4, p_ijk->dcos_dk );
                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors7 + CEconj4, p_ijk->dcos_dj );
                                    atomic_rvecScaledAdd( workspace.f[k],
                                            CEtors7 + CEconj4, p_ijk->dcos_di );

                                    /* dcos_theta_jkl */
                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors8 + CEconj5, p_jkl->dcos_di );
                                    atomic_rvecScaledAdd( workspace.f[k],
                                            CEtors8 + CEconj5, p_jkl->dcos_dj );
                                    atomic_rvecScaledAdd( workspace.f[l], 
                                            CEtors8 + CEconj5, p_jkl->dcos_dk );

                                    /* dcos_omega */
                                    atomic_rvecScaledAdd( workspace.f[i],
                                            CEtors9 + CEconj6, dcos_omega_di );
                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors9 + CEconj6, dcos_omega_dj );
                                    atomic_rvecScaledAdd( workspace.f[k],
                                            CEtors9 + CEconj6, dcos_omega_dk );
                                    atomic_rvecScaledAdd( workspace.f[l],
                                            CEtors9 + CEconj6, dcos_omega_dl );
#endif
                                }
                                else
                                {
#if !defined(CUDA_ACCUM_ATOMIC)
                                    ivec_Sum( rel_box_jl, pbond_jk->rel_box, pbond_kl->rel_box );

                                    /* dcos_theta_ijk */
                                    rvec_Scale( temp, CEtors7 + CEconj4, p_ijk->dcos_dk );
                                    atomic_rvecAdd( pbond_ij->ta_f, temp );
                                    rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors7 + CEconj4, p_ijk->dcos_dj );

                                    rvec_Scale( temp, CEtors7 + CEconj4, p_ijk->dcos_di );
                                    atomic_rvecAdd( pbond_jk->ta_f, temp );
                                    rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    /* dcos_theta_jkl */
                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors8 + CEconj5, p_jkl->dcos_di );

                                    rvec_Scale( temp, CEtors8 + CEconj5, p_jkl->dcos_dj );
                                    atomic_rvecAdd( pbond_jk->ta_f, temp );
                                    rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    rvec_Scale( temp, CEtors8 + CEconj5, p_jkl->dcos_dk );
                                    rvec_Add( pbond_kl->ta_f, temp );
                                    rvec_iMultiply( temp, rel_box_jl, temp );
                                    rvec_Add( ext_press_l, temp );

                                    /* dcos_omega */                      
                                    rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_di );
                                    atomic_rvecAdd( pbond_ij->ta_f, temp );
                                    rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors9 + CEconj6, dcos_omega_dj );

                                    rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_dk );
                                    rvec_Add( pbond_jk->ta_f, temp );
                                    rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_dl );
                                    rvec_Add( pbond_kl->ta_f, temp );
                                    rvec_iMultiply( temp, rel_box_jl, temp );
                                    rvec_Add( ext_press_l, temp );
#else
                                    ivec_Sum( rel_box_jl, pbond_jk->rel_box, pbond_kl->rel_box );

                                    /* dcos_theta_ijk */
                                    rvec_Scale( temp, CEtors7 + CEconj4, p_ijk->dcos_dk );
                                    atomic_rvecAdd( workspace.f[i], temp );
                                    rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors7 + CEconj4, p_ijk->dcos_dj );

                                    rvec_Scale( temp, CEtors7 + CEconj4, p_ijk->dcos_di );
                                    atomic_rvecAdd( workspace.f[k], temp );
                                    rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    /* dcos_theta_jkl */
                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors8 + CEconj5, p_jkl->dcos_di );

                                    rvec_Scale( temp, CEtors8 + CEconj5, p_jkl->dcos_dj );
                                    atomic_rvecAdd( workspace.f[k], temp );
                                    rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    rvec_Scale( temp, CEtors8 + CEconj5, p_jkl->dcos_dk );
                                    atomic_rvecAdd( workspace.f[l], temp );
                                    rvec_iMultiply( temp, rel_box_jl, temp );
                                    rvec_Add( ext_press_l, temp );

                                    /* dcos_omega */                      
                                    rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_di );
                                    atomic_rvecAdd( workspace.f[i], temp );
                                    rvec_iMultiply( temp, pbond_ij->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    rvec_ScaledAdd( f_j_l, 
                                            CEtors9 + CEconj6, dcos_omega_dj );

                                    rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_dk );
                                    atomic_rvecAdd( workspace.f[k], temp );
                                    rvec_iMultiply( temp, pbond_jk->rel_box, temp );
                                    rvec_Add( ext_press_l, temp );

                                    rvec_Scale( temp, CEtors9 + CEconj6, dcos_omega_dl );
                                    rvec_Add( workspace.f[l], temp );
                                    rvec_iMultiply( temp, rel_box_jl, temp );
                                    rvec_Add( ext_press_l, temp );
#endif
                                }
                            } // pl check ends
                        } // pl loop ends
                    } // pi check ends
                } // pi loop ends
            } // k-j neighbor check ends
        } // j<k && j-k neighbor check ends
    } // pk loop ends
    //  } // j loop

#if !defined(CUDA_ACCUM_ATOMIC)
    rvec_Add( workspace.f[j], f_j_l );
    e_tor_g[j] = e_tor_l;
    e_con_g[j] = e_con_l;
    rvec_Copy( e_ext_press_g[j], e_ext_press_l );
#else
    atomic_rvecAdd( workspace.f[j], f_j_l );
    atomicAdd( (double *) e_tor_g, (double) e_tor_l );
    atomicAdd( (double *) e_con_g, (double) e_con_l );
    atomic_rvecAdd( *ext_press_g, ext_press_l );
#endif
}


#if !defined(CUDA_ACCUM_ATOMIC)
CUDA_GLOBAL void Cuda_Torsion_Angles_Part2( reax_atom *my_atoms, 
        storage workspace, reax_list bond_list, int N )
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
        /* update f vector */
//        rvec_Add( my_atoms[i].f, pbond_ji->ta_f ); 
        rvec_Add( workspace.f[i], pbond_ji->ta_f ); 
    }
}
#endif
