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

#include "cuda_valence_angles.h"

#if defined(CUDA_ACCUM_ATOMIC)
#include "cuda_helpers.h"
#endif
#include "cuda_list.h"
#include "cuda_reduction.h"
#include "cuda_utils.h"

#include "../index_utils.h"
#include "../vector.h"

#include "../cub/cub/warp/warp_scan.cuh"
//#include <cub/warp/warp_scan.cuh>
#include "../cub/cub/warp/warp_reduce.cuh"
//#include <cub/warp/warp_reduce.cuh>


#define FULL_MASK (0xFFFFFFFF)


/**
 * Default product (multiplication) functor
 */
struct Prod
{
    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a * b;
    }
};


/* Compute 3-body interactions, in which the main role is played by
   atom j, which sits in the middle of the other two atoms i and k. */
CUDA_GLOBAL void k_valence_angles_part1( reax_atom *my_atoms,
        global_parameters gp, single_body_parameters *sbp, three_body_header *d_thbh,
        control_params *control, storage workspace, reax_list bond_list,
        reax_list thb_list, int n, int N, int num_atom_types,
        real *e_ang_g, real *e_pen_g, real *e_coa_g )
{
    int i, j, pi, k, pk, t;
    int type_i, type_j, type_k;
    int start_j, end_j;
    int cnt, num_thb_intrs;
    real temp, temp_bo_jt, pBOjt7;
    real p_val1, p_val2, p_val3, p_val4, p_val5;
    real p_val6, p_val7, p_val8, p_val9, p_val10;
    real p_pen1, p_pen2, p_pen3, p_pen4;
    real p_coa1, p_coa2, p_coa3, p_coa4;
    real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    real CEpen1, CEpen2, CEpen3;
    real e_ang_l, e_coa, e_coa_l, e_pen, e_pen_l;
    real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    real Cf7ij, Cf7jk, Cf8j, Cf9j;
    real f7_ij, f7_jk, f8_Dj, f9_Dj;
    real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
    real BOA_ij, BOA_jk;
    rvec f_j_l;
    three_body_header *thbh;
    three_body_parameters *thbp;
    three_body_interaction_data *p_ijk;
    bond_data *pbond_ij, *pbond_jk, *pbond_jt;
    bond_order_data *bo_ij, *bo_jk, *bo_jt;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= N )
    {
        return;
    }

    p_pen2 = gp.l[19];
    p_pen3 = gp.l[20];
    p_pen4 = gp.l[21];
    p_coa2 = gp.l[2];
    p_coa3 = gp.l[38];
    p_coa4 = gp.l[30];
    p_val6 = gp.l[14];
    p_val8 = gp.l[33];
    p_val9 = gp.l[16];
    p_val10 = gp.l[17];
    e_ang_l = 0.0;
    e_coa_l = 0.0;
    e_pen_l = 0.0;
    rvec_MakeZero( f_j_l );

    type_j = my_atoms[j].type;
    start_j = Start_Index( j, &bond_list );
    end_j = End_Index( j, &bond_list );
    p_val3 = sbp[ type_j ].p_val3;
    p_val5 = sbp[ type_j ].p_val5;

    /* sum of pi and pi-pi BO terms for all neighbors of atom j,
     * used in determining the equilibrium angle between i-j-k */
    SBOp = 0.0;
    /* product of e^{-BO_j^8} terms for all neighbors of atom j,
     * used in determining the equilibrium angle between i-j-k */
    prod_SBO = 1.0;

    for ( t = start_j; t < end_j; ++t )
    {
        bo_jt = &bond_list.bond_list[t].bo_data;
        SBOp += bo_jt->BO_pi + bo_jt->BO_pi2;
        temp = SQR( bo_jt->BO );
        temp *= temp;
        temp *= temp;
        prod_SBO *= EXP( -temp );
    }

    /* modifications to match Adri's code - 09/01/09 */
    if ( workspace.vlpex[j] >= 0.0 )
    {
        vlpadj = 0.0;
        dSBO2 = prod_SBO - 1.0;
    }
    else
    {
        vlpadj = workspace.nlp[j];
        dSBO2 = (prod_SBO - 1.0) * (1.0 - p_val8 * workspace.dDelta_lp[j]);
    }

    SBO = SBOp + (1.0 - prod_SBO) * (-workspace.Delta_boc[j] - p_val8 * vlpadj);
    dSBO1 = -8.0 * prod_SBO * ( workspace.Delta_boc[j] + p_val8 * vlpadj );

    if ( SBO <= 0.0 )
    {
        SBO2 = 0.0;
        CSBO2 = 0.0;
    }
    else if ( SBO > 0.0 && SBO <= 1.0 )
    {
        SBO2 = POW( SBO, p_val9 );
        CSBO2 = p_val9 * POW( SBO, p_val9 - 1.0 );
    }
    else if( SBO > 1.0 && SBO < 2.0 )
    {
        SBO2 = 2.0 - POW( 2.0 - SBO, p_val9 );
        CSBO2 = p_val9 * POW( 2.0 - SBO, p_val9 - 1.0 );
    }
    else
    {
        SBO2 = 2.0;
        CSBO2 = 0.0;
    }

    expval6 = EXP( p_val6 * workspace.Delta_boc[j] );

    for ( pi = start_j; pi < end_j; ++pi )
    {
        num_thb_intrs = Start_Index( pi, &thb_list );
        pbond_ij = &bond_list.bond_list[pi];
        bo_ij = &pbond_ij->bo_data;
        BOA_ij = bo_ij->BO - control->thb_cut;

        if ( BOA_ij >= 0.0 && (j < n || pbond_ij->nbr < n) )
        {
            i = pbond_ij->nbr;
            type_i = my_atoms[i].type;

            /* compute _ALL_ 3-body intrs */
            for ( pk = start_j; pk < end_j; ++pk )
            {
                if ( pk == pi )
                {
                    continue;
                }

                pbond_jk = &bond_list.bond_list[pk];
                bo_jk = &pbond_jk->bo_data;
                BOA_jk = bo_jk->BO - control->thb_cut;

                if ( BOA_jk < 0.0 )
                {
                    continue;
                }

                k = pbond_jk->nbr;
                type_k = my_atoms[k].type;
                p_ijk = &thb_list.three_body_list[num_thb_intrs];

                Calculate_Theta( pbond_ij->dvec, pbond_ij->d,
                        pbond_jk->dvec, pbond_jk->d,
                        &theta, &cos_theta );

                Calculate_dCos_Theta( pbond_ij->dvec, pbond_ij->d,
                        pbond_jk->dvec, pbond_jk->d,
                        &p_ijk->dcos_di, &p_ijk->dcos_dj,
                        &p_ijk->dcos_dk );

                p_ijk->thb = k;
                p_ijk->pthb = pk;
                p_ijk->theta = theta;

                sin_theta = SIN( theta );
                if ( sin_theta < 1.0e-5 )
                {
                    sin_theta = 1.0e-5;
                }

                ++num_thb_intrs;

                /* Fortran ReaxFF code hard-codes the constant below
                 * as of 2019-02-27, so use that for now */
                if ( j >= n || BOA_jk < 0.0 || bo_ij->BO * bo_jk->BO < 0.00001 )
//                if ( j >= n || BOA_jk < 0.0 || bo_ij->BO * bo_jk->BO < SQR(control->thb_cut) )
                {
                    continue;
                }

                thbh = &d_thbh[
                    index_thbp(type_i, type_j, type_k, num_atom_types) ];

                for ( cnt = 0; cnt < thbh->cnt; ++cnt )
                {
                    /* valence angle does not exist in the force field */
                    if ( FABS(thbh->prm[cnt].p_val1) < 0.001 )
                    {
                        continue;
                    }

                    thbp = &thbh->prm[cnt];

                    /* calculate valence angle energy */
                    p_val1 = thbp->p_val1;
                    p_val2 = thbp->p_val2;
                    p_val4 = thbp->p_val4;
                    p_val7 = thbp->p_val7;
                    theta_00 = thbp->theta_00;

                    exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                    f7_ij = 1.0 - exp3ij;
                    Cf7ij = p_val3 * p_val4
                        * POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                    exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                    f7_jk = 1.0 - exp3jk;
                    Cf7jk = p_val3 * p_val4
                        * POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                    expval7 = EXP( -p_val7 * workspace.Delta_boc[j] );
                    trm8 = 1.0 + expval6 + expval7;
                    f8_Dj = p_val5 - (p_val5 - 1.0) * (2.0 + expval6) / trm8;
                    Cf8j = ( (1.0 - p_val5) / SQR(trm8) )
                        * (p_val6 * expval6 * trm8
                                - (2.0 + expval6) * (p_val6 * expval6 - p_val7 * expval7) );

                    theta_0 = 180.0 - theta_00 * (1.0 - EXP(-p_val10 * (2.0 - SBO2)));
                    theta_0 = DEG2RAD( theta_0 );

                    expval2theta = p_val1 * EXP(-p_val2 * SQR(theta_0 - theta));
                    if ( p_val1 >= 0.0 )
                    {
                        expval12theta = p_val1 - expval2theta;
                    }
                    /* to avoid linear Me-H-Me angles (6/6/06) */
                    else
                    {
                        expval12theta = -expval2theta;
                    }

                    CEval1 = Cf7ij * f7_jk * f8_Dj * expval12theta;
                    CEval2 = Cf7jk * f7_ij * f8_Dj * expval12theta;
                    CEval3 = Cf8j * f7_ij * f7_jk * expval12theta;
                    CEval4 = 2.0 * p_val2 * f7_ij * f7_jk * f8_Dj
                        * expval2theta * (theta_0 - theta);

                    Ctheta_0 = p_val10 * DEG2RAD(theta_00)
                        * EXP( -p_val10 * (2.0 - SBO2) );

                    CEval5 = CEval4 * Ctheta_0 * CSBO2;
                    CEval6 = CEval5 * dSBO1;
                    CEval7 = CEval5 * dSBO2;
                    CEval8 = CEval4 / sin_theta;

                    if ( pk < pi )
                    {
                        e_ang_l += f7_ij * f7_jk * f8_Dj * expval12theta;
                    }

                    /* calculate penalty for double bonds in valency angles */
                    p_pen1 = thbp->p_pen1;

                    exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                    exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                    exp_pen3 = EXP( -p_pen3 * workspace.Delta[j] );
                    exp_pen4 = EXP(  p_pen4 * workspace.Delta[j] );
                    trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                    f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                    Cf9j = (-p_pen3 * exp_pen3 * trm_pen34
                            - (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3
                                + p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                    e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                    if ( pk < pi )
                    {
                        e_pen_l += e_pen;
                    }

                    CEpen1 = e_pen * Cf9j / f9_Dj;
                    temp = -2.0 * p_pen2 * e_pen;
                    CEpen2 = temp * (BOA_ij - 2.0);
                    CEpen3 = temp * (BOA_jk - 2.0);

                    /* calculate valency angle conjugation energy */
                    p_coa1 = thbp->p_coa1;

                    exp_coa2 = EXP( p_coa2 * workspace.Delta_boc[j] );
                    e_coa = p_coa1
                        * EXP( -p_coa4 * SQR(BOA_ij - 1.5) )
                        * EXP( -p_coa4 * SQR(BOA_jk - 1.5) )
                        * EXP( -p_coa3 * SQR(workspace.total_bond_order[i] - BOA_ij) )
                        * EXP( -p_coa3 * SQR(workspace.total_bond_order[k] - BOA_jk) )
                        / (1.0 + exp_coa2);

                    if ( pk < pi )
                    {
                        e_coa_l += e_coa;
                    }

                    CEcoa1 = -2.0 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                    CEcoa2 = -2.0 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                    CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1.0 + exp_coa2);
                    CEcoa4 = -2.0 * p_coa3 * (workspace.total_bond_order[i] - BOA_ij) * e_coa;
                    CEcoa5 = -2.0 * p_coa3 * (workspace.total_bond_order[k] - BOA_jk) * e_coa;

                    /* calculate force contributions */
                    if ( pk < pi )
                    {
#if !defined(CUDA_ACCUM_ATOMIC)
                        bo_ij->Cdbo += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
                        bo_jk->Cdbo += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
                        workspace.CdDelta[j] += (CEval3 + CEval7) + CEpen1 + CEcoa3;
                        pbond_ij->va_CdDelta += CEcoa4;
                        pbond_jk->va_CdDelta += CEcoa5;
#else
                        atomicAdd( &bo_ij->Cdbo, CEval1 + CEpen2 + (CEcoa1 - CEcoa4) );
                        atomicAdd( &bo_jk->Cdbo, CEval2 + CEpen3 + (CEcoa2 - CEcoa5) );
                        atomicAdd( &workspace.CdDelta[j], (CEval3 + CEval7) + CEpen1 + CEcoa3 );
                        atomicAdd( &workspace.CdDelta[i], CEcoa4 );
                        atomicAdd( &workspace.CdDelta[k], CEcoa5 );
#endif

                        for ( t = start_j; t < end_j; ++t )
                        {
                            pbond_jt = &bond_list.bond_list[t];
                            bo_jt = &pbond_jt->bo_data;
                            temp_bo_jt = bo_jt->BO;
                            temp = CUBE( temp_bo_jt );
                            pBOjt7 = temp * temp * temp_bo_jt;

#if !defined(CUDA_ACCUM_ATOMIC)
                            bo_jt->Cdbo += CEval6 * pBOjt7;
                            bo_jt->Cdbopi += CEval5;
                            bo_jt->Cdbopi2 += CEval5;
#else
                            atomicAdd( &bo_jt->Cdbo, CEval6 * pBOjt7 );
                            atomicAdd( &bo_jt->Cdbopi, CEval5 );
                            atomicAdd( &bo_jt->Cdbopi2, CEval5 );
#endif
                        }

#if !defined(CUDA_ACCUM_ATOMIC)
                        rvec_ScaledAdd( pbond_ij->va_f, CEval8, p_ijk->dcos_di );
                        rvec_ScaledAdd( f_j_l, CEval8, p_ijk->dcos_dj );
                        rvec_ScaledAdd( pbond_jk->va_f, CEval8, p_ijk->dcos_dk );
#else
                        atomic_rvecScaledAdd( workspace.f[i], CEval8, p_ijk->dcos_di );
                        rvec_ScaledAdd( f_j_l, CEval8, p_ijk->dcos_dj );
                        atomic_rvecScaledAdd( workspace.f[k], CEval8, p_ijk->dcos_dk );
#endif
                    }
                }
            }
        }

        Set_End_Index( pi, num_thb_intrs, &thb_list );
    }

#if !defined(CUDA_ACCUM_ATOMIC)
    rvec_Add( workspace.f[j], f_j_l );
    e_ang_g[j] = e_ang_l;
    e_coa_g[j] = e_coa_l;
    e_pen_g[j] = e_pen_l;
#else
    atomic_rvecAdd( workspace.f[j], f_j_l );
    atomicAdd( (double *) e_ang_g, (double) e_ang_l );
    atomicAdd( (double *) e_coa_g, (double) e_coa_l );
    atomicAdd( (double *) e_pen_g, (double) e_pen_l );
#endif
}


/* Compute 3-body interactions, in which the main role is played by
   atom j, which sits in the middle of the other two atoms i and k. */
CUDA_GLOBAL void k_valence_angles_part1_opt( reax_atom *my_atoms,
        global_parameters gp, single_body_parameters *sbp, three_body_header *d_thbh,
        control_params *control, storage workspace, reax_list bond_list,
        reax_list thb_list, int n, int N, int num_atom_types,
        real *e_ang_g, real *e_pen_g, real *e_coa_g )
{
    extern __shared__ cub::WarpScan<int>::TempStorage temp_i[];
    __shared__ cub::WarpReduce<double>::TempStorage *temp_d;
    int i, j, pi, k, pk, t, thread_id, lane_id, itr;
    int type_i, type_j, type_k;
    int start_j, end_j;
    int cnt, num_thb_intrs, offset, flag;
    real temp, temp_bo_jt, pBOjt7;
    real p_val1, p_val2, p_val3, p_val4, p_val5;
    real p_val6, p_val7, p_val8, p_val9, p_val10;
    real p_pen1, p_pen2, p_pen3, p_pen4;
    real p_coa1, p_coa2, p_coa3, p_coa4;
    real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    real CEpen1, CEpen2, CEpen3;
    real e_ang_l, e_coa, e_coa_l, e_pen, e_pen_l;
    real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    real Cf7ij, Cf7jk, Cf8j, Cf9j;
    real f7_ij, f7_jk, f8_Dj, f9_Dj;
    real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
    real BOA_ij, BOA_jk;
    rvec f_j_l;
    three_body_header *thbh;
    three_body_parameters *thbp;
    three_body_interaction_data *p_ijk;
    bond_data *pbond_ij, *pbond_jk, *pbond_jt;
    bond_order_data *bo_ij, *bo_jk, *bo_jt;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    j = thread_id / warpSize;

    if ( j >= N )
    {
        return;
    }

    temp_d = (cub::WarpReduce<double>::TempStorage *) &temp_i[blockDim.x / warpSize];
    lane_id = thread_id % warpSize;
    p_pen2 = gp.l[19];
    p_pen3 = gp.l[20];
    p_pen4 = gp.l[21];
    p_coa2 = gp.l[2];
    p_coa3 = gp.l[38];
    p_coa4 = gp.l[30];
    p_val6 = gp.l[14];
    p_val8 = gp.l[33];
    p_val9 = gp.l[16];
    p_val10 = gp.l[17];
    e_ang_l = 0.0;
    e_coa_l = 0.0;
    e_pen_l = 0.0;
    rvec_MakeZero( f_j_l );

    type_j = my_atoms[j].type;
    start_j = Start_Index( j, &bond_list );
    end_j = End_Index( j, &bond_list );
    p_val3 = sbp[ type_j ].p_val3;
    p_val5 = sbp[ type_j ].p_val5;

    /* sum of pi and pi-pi BO terms for all neighbors of atom j,
     * used in determining the equilibrium angle between i-j-k */
    SBOp = 0.0;
    /* product of e^{-BO_j^8} terms for all neighbors of atom j,
     * used in determining the equilibrium angle between i-j-k */
    prod_SBO = 1.0;

    for ( itr = 0, t = start_j + lane_id; itr < (end_j - start_j + warpSize - 1) / warpSize; ++itr )
    {
        if ( t < end_j )
        {
            bo_jt = &bond_list.bond_list[t].bo_data;
            SBOp += bo_jt->BO_pi + bo_jt->BO_pi2;
            temp = SQR( bo_jt->BO );
            temp *= temp;
            temp *= temp;
            prod_SBO *= EXP( -temp );
        }

        t += warpSize;
    }

    SBOp = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(SBOp);
    prod_SBO = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Reduce(prod_SBO, Prod());

    /* broadcast redux results from lane 0 */
    SBOp = __shfl_sync( FULL_MASK, SBOp, 0 );
    prod_SBO = __shfl_sync( FULL_MASK, prod_SBO, 0 );

    /* modifications to match Adri's code - 09/01/09 */
    if ( workspace.vlpex[j] >= 0.0 )
    {
        vlpadj = 0.0;
        dSBO2 = prod_SBO - 1.0;
    }
    else
    {
        vlpadj = workspace.nlp[j];
        dSBO2 = (prod_SBO - 1.0) * (1.0 - p_val8 * workspace.dDelta_lp[j]);
    }

    SBO = SBOp + (1.0 - prod_SBO) * (-workspace.Delta_boc[j] - p_val8 * vlpadj);
    dSBO1 = -8.0 * prod_SBO * ( workspace.Delta_boc[j] + p_val8 * vlpadj );

    if ( SBO <= 0.0 )
    {
        SBO2 = 0.0;
        CSBO2 = 0.0;
    }
    else if ( SBO > 0.0 && SBO <= 1.0 )
    {
        SBO2 = POW( SBO, p_val9 );
        CSBO2 = p_val9 * POW( SBO, p_val9 - 1.0 );
    }
    else if( SBO > 1.0 && SBO < 2.0 )
    {
        SBO2 = 2.0 - POW( 2.0 - SBO, p_val9 );
        CSBO2 = p_val9 * POW( 2.0 - SBO, p_val9 - 1.0 );
    }
    else
    {
        SBO2 = 2.0;
        CSBO2 = 0.0;
    }

    expval6 = EXP( p_val6 * workspace.Delta_boc[j] );

    for ( pi = start_j; pi < end_j; ++pi )
    {
        num_thb_intrs = Start_Index( pi, &thb_list );
        pbond_ij = &bond_list.bond_list[pi];
        bo_ij = &pbond_ij->bo_data;
        BOA_ij = bo_ij->BO - control->thb_cut;

        if ( BOA_ij >= 0.0 && (j < n || pbond_ij->nbr < n) )
        {
            i = pbond_ij->nbr;
            type_i = my_atoms[i].type;

            /* compute _ALL_ 3-body intrs */
            for ( itr = 0, pk = start_j + lane_id; itr < (end_j - start_j + warpSize - 1) / warpSize; ++itr )
            {
                pbond_jk = &bond_list.bond_list[pk];
                bo_jk = &pbond_jk->bo_data;
                BOA_jk = bo_jk->BO - control->thb_cut;

                offset = (pk < end_j && pk != pi && BOA_jk >= 0.0) ? 1 : 0;
                flag = (offset == 1) ? TRUE : FALSE;
                cub::WarpScan<int>(temp_i[j % (blockDim.x / warpSize)]).ExclusiveSum(offset, offset);

                if ( flag == TRUE )
                {
                    k = pbond_jk->nbr;
                    type_k = my_atoms[k].type;
                    p_ijk = &thb_list.three_body_list[num_thb_intrs + offset];

                    Calculate_Theta( pbond_ij->dvec, pbond_ij->d,
                            pbond_jk->dvec, pbond_jk->d,
                            &theta, &cos_theta );

                    Calculate_dCos_Theta( pbond_ij->dvec, pbond_ij->d,
                            pbond_jk->dvec, pbond_jk->d,
                            &p_ijk->dcos_di, &p_ijk->dcos_dj,
                            &p_ijk->dcos_dk );

                    p_ijk->thb = k;
                    p_ijk->pthb = pk;
                    p_ijk->theta = theta;

                    sin_theta = SIN( theta );
                    if ( sin_theta < 1.0e-5 )
                    {
                        sin_theta = 1.0e-5;
                    }

                    /* Fortran ReaxFF code hard-codes the constant below
                     * as of 2019-02-27, so use that for now */
                    if ( j < n && BOA_jk >= 0.0 && bo_ij->BO * bo_jk->BO >= 0.00001 )
//                    if ( j < n && BOA_jk >= 0.0 && bo_ij->BO * bo_jk->BO >= SQR(control->thb_cut) )
                    {
                        thbh = &d_thbh[
                            index_thbp(type_i, type_j, type_k, num_atom_types) ];

                        for ( cnt = 0; cnt < thbh->cnt; ++cnt )
                        {
                            /* valence angle does not exist in the force field */
                            if ( FABS(thbh->prm[cnt].p_val1) < 0.001 )
                            {
                                continue;
                            }

                            thbp = &thbh->prm[cnt];

                            /* calculate valence angle energy */
                            p_val1 = thbp->p_val1;
                            p_val2 = thbp->p_val2;
                            p_val4 = thbp->p_val4;
                            p_val7 = thbp->p_val7;
                            theta_00 = thbp->theta_00;

                            exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                            f7_ij = 1.0 - exp3ij;
                            Cf7ij = p_val3 * p_val4
                                * POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                            exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                            f7_jk = 1.0 - exp3jk;
                            Cf7jk = p_val3 * p_val4
                                * POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                            expval7 = EXP( -p_val7 * workspace.Delta_boc[j] );
                            trm8 = 1.0 + expval6 + expval7;
                            f8_Dj = p_val5 - (p_val5 - 1.0) * (2.0 + expval6) / trm8;
                            Cf8j = ( (1.0 - p_val5) / SQR(trm8) )
                                * (p_val6 * expval6 * trm8
                                        - (2.0 + expval6) * ( p_val6 * expval6 - p_val7 * expval7) );

                            theta_0 = 180.0 - theta_00 * (1.0 - EXP(-p_val10 * (2.0 - SBO2)));
                            theta_0 = DEG2RAD( theta_0 );

                            expval2theta = p_val1 * EXP(-p_val2 * SQR(theta_0 - theta));
                            if ( p_val1 >= 0.0 )
                            {
                                expval12theta = p_val1 - expval2theta;
                            }
                            /* to avoid linear Me-H-Me angles (6/6/06) */
                            else
                            {
                                expval12theta = -expval2theta;
                            }

                            CEval1 = Cf7ij * f7_jk * f8_Dj * expval12theta;
                            CEval2 = Cf7jk * f7_ij * f8_Dj * expval12theta;
                            CEval3 = Cf8j * f7_ij * f7_jk * expval12theta;
                            CEval4 = 2.0 * p_val2 * f7_ij * f7_jk * f8_Dj
                                * expval2theta * (theta_0 - theta);

                            Ctheta_0 = p_val10 * DEG2RAD(theta_00)
                                * EXP( -p_val10 * (2.0 - SBO2) );

                            CEval5 = CEval4 * Ctheta_0 * CSBO2;
                            CEval6 = CEval5 * dSBO1;
                            CEval7 = CEval5 * dSBO2;
                            CEval8 = CEval4 / sin_theta;

                            if ( pk < pi )
                            {
                                e_ang_l += f7_ij * f7_jk * f8_Dj * expval12theta;
                            }

                            /* calculate penalty for double bonds in valency angles */
                            p_pen1 = thbp->p_pen1;

                            exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                            exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                            exp_pen3 = EXP( -p_pen3 * workspace.Delta[j] );
                            exp_pen4 = EXP(  p_pen4 * workspace.Delta[j] );
                            trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                            f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                            Cf9j = (-p_pen3 * exp_pen3 * trm_pen34
                                    - (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3
                                        + p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                            e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                            if ( pk < pi )
                            {
                                e_pen_l += e_pen;
                            }

                            CEpen1 = e_pen * Cf9j / f9_Dj;
                            temp = -2.0 * p_pen2 * e_pen;
                            CEpen2 = temp * (BOA_ij - 2.0);
                            CEpen3 = temp * (BOA_jk - 2.0);

                            /* calculate valency angle conjugation energy */
                            p_coa1 = thbp->p_coa1;

                            exp_coa2 = EXP( p_coa2 * workspace.Delta_boc[j] );
                            e_coa = p_coa1
                                * EXP( -p_coa4 * SQR(BOA_ij - 1.5) )
                                * EXP( -p_coa4 * SQR(BOA_jk - 1.5) )
                                * EXP( -p_coa3 * SQR(workspace.total_bond_order[i] - BOA_ij) )
                                * EXP( -p_coa3 * SQR(workspace.total_bond_order[k] - BOA_jk) )
                                / (1.0 + exp_coa2);

                            if ( pk < pi )
                            {
                                e_coa_l += e_coa;
                            }

                            CEcoa1 = -2.0 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                            CEcoa2 = -2.0 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                            CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1.0 + exp_coa2);
                            CEcoa4 = -2.0 * p_coa3 * (workspace.total_bond_order[i] - BOA_ij) * e_coa;
                            CEcoa5 = -2.0 * p_coa3 * (workspace.total_bond_order[k] - BOA_jk) * e_coa;

                            /* calculate force contributions */
                            if ( pk < pi )
                            {
#if !defined(CUDA_ACCUM_ATOMIC)
                                bo_ij->Cdbo += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
                                bo_jk->Cdbo += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
                                workspace.CdDelta[j] += (CEval3 + CEval7) + CEpen1 + CEcoa3;
                                pbond_ij->va_CdDelta += CEcoa4;
                                pbond_jk->va_CdDelta += CEcoa5;
#else
                                atomicAdd( &bo_ij->Cdbo, CEval1 + CEpen2 + (CEcoa1 - CEcoa4) );
                                atomicAdd( &bo_jk->Cdbo, CEval2 + CEpen3 + (CEcoa2 - CEcoa5) );
                                atomicAdd( &workspace.CdDelta[j], (CEval3 + CEval7) + CEpen1 + CEcoa3 );
                                atomicAdd( &workspace.CdDelta[i], CEcoa4 );
                                atomicAdd( &workspace.CdDelta[k], CEcoa5 );
#endif

                                for ( t = start_j; t < end_j; ++t )
                                {
                                    pbond_jt = &bond_list.bond_list[t];
                                    bo_jt = &pbond_jt->bo_data;
                                    temp_bo_jt = bo_jt->BO;
                                    temp = CUBE( temp_bo_jt );
                                    pBOjt7 = temp * temp * temp_bo_jt;

#if !defined(CUDA_ACCUM_ATOMIC)
                                    bo_jt->Cdbo += (CEval6 * pBOjt7);
                                    bo_jt->Cdbopi += CEval5;
                                    bo_jt->Cdbopi2 += CEval5;
#else
                                    atomicAdd( &bo_jt->Cdbo, CEval6 * pBOjt7 );
                                    atomicAdd( &bo_jt->Cdbopi, CEval5 );
                                    atomicAdd( &bo_jt->Cdbopi2, CEval5 );
#endif
                                }

#if !defined(CUDA_ACCUM_ATOMIC)
                                rvec_ScaledAdd( pbond_ij->va_f, CEval8, p_ijk->dcos_di );
                                rvec_ScaledAdd( f_j_l, CEval8, p_ijk->dcos_dj );
                                rvec_ScaledAdd( pbond_jk->va_f, CEval8, p_ijk->dcos_dk );
#else
                                atomic_rvecScaledAdd( workspace.f[i], CEval8, p_ijk->dcos_di );
                                rvec_ScaledAdd( f_j_l, CEval8, p_ijk->dcos_dj );
                                atomic_rvecScaledAdd( workspace.f[k], CEval8, p_ijk->dcos_dk );
#endif
                            }
                        }
                    }
                }

                /* get num_thb_intrs from thread in last lane */
                num_thb_intrs = num_thb_intrs + offset + (flag == TRUE ? 1 : 0);
                num_thb_intrs = __shfl_sync( FULL_MASK, num_thb_intrs, warpSize - 1 );

                pk += warpSize;
            }
        }

        if ( lane_id == 0 )
        {
            Set_End_Index( pi, num_thb_intrs, &thb_list );
        }
    }

    f_j_l[0] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_j_l[0]);
    f_j_l[1] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_j_l[1]);
    f_j_l[2] = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(f_j_l[2]);
    e_ang_l = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(e_ang_l);
    e_coa_l = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(e_coa_l);
    e_pen_l = cub::WarpReduce<double>(temp_d[j % (blockDim.x / warpSize)]).Sum(e_pen_l);

    if ( lane_id == 0 )
    {
#if !defined(CUDA_ACCUM_ATOMIC)
        rvec_Add( workspace.f[j], f_j_l );
        e_ang_g[j] = e_ang_l;
        e_coa_g[j] = e_coa_l;
        e_pen_g[j] = e_pen_l;
#else
        atomic_rvecAdd( workspace.f[j], f_j_l );
        atomicAdd( (double *) e_ang_g, (double) e_ang_l );
        atomicAdd( (double *) e_coa_g, (double) e_coa_l );
        atomicAdd( (double *) e_pen_g, (double) e_pen_l );
#endif
    }
}


/* Compute 3-body interactions, in which the main role is played by
   atom j, which sits in the middle of the other two atoms i and k. */
CUDA_GLOBAL void k_valence_angles_virial_part1( reax_atom *my_atoms,
        global_parameters gp, single_body_parameters *sbp, three_body_header *d_thbh,
        control_params *control, storage workspace, reax_list bond_list,
        reax_list thb_list, int n, int N, int num_atom_types,
        real *e_ang_g, real *e_pen_g, real *e_coa_g, rvec *ext_press_g )
{
    int i, j, pi, k, pk, t;
    int type_i, type_j, type_k;
    int start_j, end_j;
    int cnt, num_thb_intrs;
    real temp, temp_bo_jt, pBOjt7;
    real p_val1, p_val2, p_val3, p_val4, p_val5;
    real p_val6, p_val7, p_val8, p_val9, p_val10;
    real p_pen1, p_pen2, p_pen3, p_pen4;
    real p_coa1, p_coa2, p_coa3, p_coa4;
    real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    real CEpen1, CEpen2, CEpen3;
    real e_ang_l, e_coa, e_coa_l, e_pen, e_pen_l;
    real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    real Cf7ij, Cf7jk, Cf8j, Cf9j;
    real f7_ij, f7_jk, f8_Dj, f9_Dj;
    real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
    real BOA_ij, BOA_jk;
    rvec rvec_temp, f_j_l, ext_press_l;
    three_body_header *thbh;
    three_body_parameters *thbp;
    three_body_interaction_data *p_ijk;
    bond_data *pbond_ij, *pbond_jk, *pbond_jt;
    bond_order_data *bo_ij, *bo_jk, *bo_jt;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= N )
    {
        return;
    }

    p_pen2 = gp.l[19];
    p_pen3 = gp.l[20];
    p_pen4 = gp.l[21];
    p_coa2 = gp.l[2];
    p_coa3 = gp.l[38];
    p_coa4 = gp.l[30];
    p_val6 = gp.l[14];
    p_val8 = gp.l[33];
    p_val9 = gp.l[16];
    p_val10 = gp.l[17];
    e_ang_l = 0.0;
    e_coa_l = 0.0;
    e_pen_l = 0.0;
    rvec_MakeZero( f_j_l );
    rvec_MakeZero( ext_press_l );

    type_j = my_atoms[j].type;
    start_j = Start_Index( j, &bond_list );
    end_j = End_Index( j, &bond_list );
    p_val3 = sbp[ type_j ].p_val3;
    p_val5 = sbp[ type_j ].p_val5;

    /* sum of pi and pi-pi BO terms for all neighbors of atom j,
     * used in determining the equilibrium angle between i-j-k */
    SBOp = 0.0;
    /* product of e^{-BO_j^8} terms for all neighbors of atom j,
     * used in determining the equilibrium angle between i-j-k */
    prod_SBO = 1.0;

    for ( t = start_j; t < end_j; ++t )
    {
        bo_jt = &bond_list.bond_list[t].bo_data;
        SBOp += bo_jt->BO_pi + bo_jt->BO_pi2;
        temp = SQR( bo_jt->BO );
        temp *= temp;
        temp *= temp;
        prod_SBO *= EXP( -temp );
    }

    /* modifications to match Adri's code - 09/01/09 */
    if ( workspace.vlpex[j] >= 0.0 )
    {
        vlpadj = 0.0;
        dSBO2 = prod_SBO - 1.0;
    }
    else
    {
        vlpadj = workspace.nlp[j];
        dSBO2 = (prod_SBO - 1.0) * (1.0 - p_val8 * workspace.dDelta_lp[j]);
    }

    SBO = SBOp + (1.0 - prod_SBO) * (-workspace.Delta_boc[j] - p_val8 * vlpadj);
    dSBO1 = -8.0 * prod_SBO * ( workspace.Delta_boc[j] + p_val8 * vlpadj );

    if ( SBO <= 0.0 )
    {
        SBO2 = 0.0;
        CSBO2 = 0.0;
    }
    else if ( SBO > 0.0 && SBO <= 1.0 )
    {
        SBO2 = POW( SBO, p_val9 );
        CSBO2 = p_val9 * POW( SBO, p_val9 - 1.0 );
    }
    else if( SBO > 1.0 && SBO < 2.0 )
    {
        SBO2 = 2.0 - POW( 2.0 - SBO, p_val9 );
        CSBO2 = p_val9 * POW( 2.0 - SBO, p_val9 - 1.0 );
    }
    else
    {
        SBO2 = 2.0;
        CSBO2 = 0.0;
    }

    expval6 = EXP( p_val6 * workspace.Delta_boc[j] );

    for ( pi = start_j; pi < end_j; ++pi )
    {
        num_thb_intrs = Start_Index( pi, &thb_list );
        pbond_ij = &bond_list.bond_list[pi];
        bo_ij = &pbond_ij->bo_data;
        BOA_ij = bo_ij->BO - control->thb_cut;

        if ( BOA_ij >= 0.0 && (j < n || pbond_ij->nbr < n) )
        {
            i = pbond_ij->nbr;
            type_i = my_atoms[i].type;

            /* compute _ALL_ 3-body intrs */
            for ( pk = start_j; pk < end_j; ++pk )
            {
                if ( pk == pi )
                {
                    continue;
                }

                pbond_jk = &bond_list.bond_list[pk];
                bo_jk = &pbond_jk->bo_data;
                BOA_jk = bo_jk->BO - control->thb_cut;

                if ( BOA_jk < 0.0 )
                {
                    continue;
                }

                k = pbond_jk->nbr;
                type_k = my_atoms[k].type;
                p_ijk = &thb_list.three_body_list[num_thb_intrs];

                Calculate_Theta( pbond_ij->dvec, pbond_ij->d,
                        pbond_jk->dvec, pbond_jk->d,
                        &theta, &cos_theta );

                Calculate_dCos_Theta( pbond_ij->dvec, pbond_ij->d,
                        pbond_jk->dvec, pbond_jk->d,
                        &p_ijk->dcos_di, &p_ijk->dcos_dj,
                        &p_ijk->dcos_dk );

                p_ijk->thb = k;
                p_ijk->pthb = pk;
                p_ijk->theta = theta;

                sin_theta = SIN( theta );
                if ( sin_theta < 1.0e-5 )
                {
                    sin_theta = 1.0e-5;
                }

                ++num_thb_intrs;

                /* Fortran ReaxFF code hard-codes the constant below
                 * as of 2019-02-27, so use that for now */
                if ( j >= n || BOA_jk < 0.0 || (bo_ij->BO * bo_jk->BO) < 0.00001 )
//                if ( j >= n || BOA_jk < 0.0 || (bo_ij->BO * bo_jk->BO) < SQR(control->thb_cut) )
                {
                    continue;
                }

                thbh = &d_thbh[
                    index_thbp(type_i, type_j, type_k, num_atom_types) ];

                for ( cnt = 0; cnt < thbh->cnt; ++cnt )
                {
                    /* valence angle does not exist in the force field */
                    if ( FABS(thbh->prm[cnt].p_val1) < 0.001 )
                    {
                        continue;
                    }

                    thbp = &thbh->prm[cnt];

                    /* calculate valence angle energy */
                    p_val1 = thbp->p_val1;
                    p_val2 = thbp->p_val2;
                    p_val4 = thbp->p_val4;
                    p_val7 = thbp->p_val7;
                    theta_00 = thbp->theta_00;

                    exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                    f7_ij = 1.0 - exp3ij;
                    Cf7ij = p_val3 * p_val4
                        * POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                    exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                    f7_jk = 1.0 - exp3jk;
                    Cf7jk = p_val3 * p_val4
                        * POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                    expval7 = EXP( -p_val7 * workspace.Delta_boc[j] );
                    trm8 = 1.0 + expval6 + expval7;
                    f8_Dj = p_val5 - (p_val5 - 1.0) * (2.0 + expval6) / trm8;
                    Cf8j = ( (1.0 - p_val5) / SQR(trm8) )
                        * (p_val6 * expval6 * trm8
                                - (2.0 + expval6) * ( p_val6 * expval6 - p_val7 * expval7) );

                    theta_0 = 180.0 - theta_00 * (1.0 - EXP(-p_val10 * (2.0 - SBO2)));
                    theta_0 = DEG2RAD( theta_0 );

                    expval2theta = p_val1 * EXP(-p_val2 * SQR(theta_0 - theta));
                    if ( p_val1 >= 0.0 )
                    {
                        expval12theta = p_val1 - expval2theta;
                    }
                    /* to avoid linear Me-H-Me angles (6/6/06) */
                    else
                    {
                        expval12theta = -expval2theta;
                    }

                    CEval1 = Cf7ij * f7_jk * f8_Dj * expval12theta;
                    CEval2 = Cf7jk * f7_ij * f8_Dj * expval12theta;
                    CEval3 = Cf8j * f7_ij * f7_jk * expval12theta;
                    CEval4 = 2.0 * p_val2 * f7_ij * f7_jk * f8_Dj
                        * expval2theta * (theta_0 - theta);

                    Ctheta_0 = p_val10 * DEG2RAD(theta_00)
                        * EXP( -p_val10 * (2.0 - SBO2) );

                    CEval5 = CEval4 * Ctheta_0 * CSBO2;
                    CEval6 = CEval5 * dSBO1;
                    CEval7 = CEval5 * dSBO2;
                    CEval8 = CEval4 / sin_theta;

                    if ( pk < pi )
                    {
                        e_ang_l += f7_ij * f7_jk * f8_Dj * expval12theta;
                    }

                    /* calculate penalty for double bonds in valency angles */
                    p_pen1 = thbp->p_pen1;

                    exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                    exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                    exp_pen3 = EXP( -p_pen3 * workspace.Delta[j] );
                    exp_pen4 = EXP(  p_pen4 * workspace.Delta[j] );
                    trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                    f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                    Cf9j = (-p_pen3 * exp_pen3 * trm_pen34
                            - (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3
                                + p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                    e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                    if ( pk < pi )
                    {
                        e_pen_l += e_pen;
                    }

                    CEpen1 = e_pen * Cf9j / f9_Dj;
                    temp = -2.0 * p_pen2 * e_pen;
                    CEpen2 = temp * (BOA_ij - 2.0);
                    CEpen3 = temp * (BOA_jk - 2.0);

                    /* calculate valency angle conjugation energy */
                    p_coa1 = thbp->p_coa1;

                    exp_coa2 = EXP( p_coa2 * workspace.Delta_boc[j] );
                    e_coa = p_coa1
                        * EXP( -p_coa4 * SQR(BOA_ij - 1.5) )
                        * EXP( -p_coa4 * SQR(BOA_jk - 1.5) )
                        * EXP( -p_coa3 * SQR(workspace.total_bond_order[i] - BOA_ij) )
                        * EXP( -p_coa3 * SQR(workspace.total_bond_order[k] - BOA_jk) )
                        / (1.0 + exp_coa2);

                    if ( pk < pi )
                    {
                        e_coa_l += e_coa;
                    }

                    CEcoa1 = -2.0 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                    CEcoa2 = -2.0 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                    CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1.0 + exp_coa2);
                    CEcoa4 = -2.0 * p_coa3 * (workspace.total_bond_order[i] - BOA_ij) * e_coa;
                    CEcoa5 = -2.0 * p_coa3 * (workspace.total_bond_order[k] - BOA_jk) * e_coa;

                    /* calculate force contributions */
                    if ( pk < pi )
                    {
#if !defined(CUDA_ACCUM_ATOMIC)
                        bo_ij->Cdbo += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
                        bo_jk->Cdbo += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
                        workspace.CdDelta[j] += (CEval3 + CEval7) + CEpen1 + CEcoa3;
                        pbond_ij->va_CdDelta += CEcoa4;
                        pbond_jk->va_CdDelta += CEcoa5;
#else
                        atomicAdd( &bo_ij->Cdbo, CEval1 + CEpen2 + (CEcoa1 - CEcoa4) );
                        atomicAdd( &bo_jk->Cdbo, CEval2 + CEpen3 + (CEcoa2 - CEcoa5) );
                        atomicAdd( &workspace.CdDelta[j], (CEval3 + CEval7) + CEpen1 + CEcoa3 );
                        atomicAdd( &workspace.CdDelta[i], CEcoa4 );
                        atomicAdd( &workspace.CdDelta[k], CEcoa5 );
#endif

                        for ( t = start_j; t < end_j; ++t )
                        {
                            pbond_jt = &bond_list.bond_list[t];
                            bo_jt = &pbond_jt->bo_data;
                            temp_bo_jt = bo_jt->BO;
                            temp = CUBE( temp_bo_jt );
                            pBOjt7 = temp * temp * temp_bo_jt;

#if !defined(CUDA_ACCUM_ATOMIC)
                            bo_jt->Cdbo += (CEval6 * pBOjt7);
                            bo_jt->Cdbopi += CEval5;
                            bo_jt->Cdbopi2 += CEval5;
#else
                            atomicAdd( &bo_jt->Cdbo, CEval6 * pBOjt7 );
                            atomicAdd( &bo_jt->Cdbopi, CEval5 );
                            atomicAdd( &bo_jt->Cdbopi2, CEval5 );
#endif
                        }

#if !defined(CUDA_ACCUM_ATOMIC)
                        /* terms not related to bond order derivatives are
                         * added directly into forces and pressure vector/tensor */
                        rvec_Scale( rvec_temp, CEval8, p_ijk->dcos_di );
                        rvec_Add( pbond_ij->va_f, rvec_temp );
                        rvec_iMultiply( rvec_temp, pbond_ij->rel_box, rvec_temp );
                        rvec_Add( ext_press_l, rvec_temp );

                        rvec_ScaledAdd( f_j_l, CEval8, p_ijk->dcos_dj );

                        rvec_Scale( rvec_temp, CEval8, p_ijk->dcos_dk );
                        rvec_Add( pbond_jk->va_f, rvec_temp );
                        rvec_iMultiply( rvec_temp, pbond_jk->rel_box, rvec_temp );
                        rvec_Add( ext_press_l, rvec_temp );
#else
                        /* terms not related to bond order derivatives are
                         * added directly into forces and pressure vector/tensor */
                        rvec_Scale( rvec_temp, CEval8, p_ijk->dcos_di );
                        atomic_rvecAdd( workspace.f[i], rvec_temp );
                        rvec_iMultiply( rvec_temp, pbond_ij->rel_box, rvec_temp );
                        rvec_Add( ext_press_l, rvec_temp );

                        rvec_ScaledAdd( f_j_l, CEval8, p_ijk->dcos_dj );

                        rvec_Scale( rvec_temp, CEval8, p_ijk->dcos_dk );
                        atomic_rvecAdd( workspace.f[k], rvec_temp );
                        rvec_iMultiply( rvec_temp, pbond_jk->rel_box, rvec_temp );
                        rvec_Add( ext_press_l, rvec_temp );
#endif
                    }
                }
            }
        }

        Set_End_Index( pi, num_thb_intrs, &thb_list );
    }

#if !defined(CUDA_ACCUM_ATOMIC)
    rvec_Add( workspace.f[j], f_j_l );
    e_ang_g[j] = e_ang_l;
    e_coa_g[j] = e_coa_l;
    e_pen_g[j] = e_pen_l;
    rvec_Copy( ext_press_g[j], ext_press_l );
#else
    atomic_rvecAdd( workspace.f[j], f_j_l );
    atomicAdd( (double *) e_ang_g, (double) e_ang_l );
    atomicAdd( (double *) e_coa_g, (double) e_coa_l );
    atomicAdd( (double *) e_pen_g, (double) e_pen_l );
    atomic_rvecAdd( *ext_press_g, ext_press_l );
#endif
}


#if !defined(CUDA_ACCUM_ATOMIC)
CUDA_GLOBAL void k_valence_angles_part2( reax_atom *atoms,
        control_params *control, storage workspace,
        reax_list bond_list, int N )
{
    int i, pj;
    bond_data *pbond;
    bond_data *sym_index_bond;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        pbond = &bond_list.bond_list[pj];
        sym_index_bond = &bond_list.bond_list[ pbond->sym_index ];

        workspace.CdDelta[i] += sym_index_bond->va_CdDelta;

        rvec_Add( workspace.f[i], sym_index_bond->va_f );
    }
}
#endif


/* Estimate the num. of three-body interactions */
CUDA_GLOBAL void k_estimate_valence_angles( reax_atom *my_atoms,
        control_params *control, reax_list bond_list, int n, int N, int *count )
{
    int j, pi, pk, start_j, end_j, num_thb_intrs;
    real BOA_ij, BOA_jk;
    bond_data *pbond_ij, *pbond_jk;
    bond_order_data *bo_ij, *bo_jk;

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= N )
    {
        return;
    }

    start_j = Start_Index( j, &bond_list );
    end_j = End_Index( j, &bond_list );

    for ( pi = start_j; pi < end_j; ++pi )
    {
        num_thb_intrs = 0;

        pbond_ij = &bond_list.bond_list[pi];
        bo_ij = &pbond_ij->bo_data;
        BOA_ij = bo_ij->BO - control->thb_cut;

        if ( BOA_ij >= 0.0 && (j < n || pbond_ij->nbr < n) )
        {
            for ( pk = start_j; pk < end_j; ++pk )
            {
                if ( pk == pi )
                {
                    continue;
                }

                pbond_jk = &bond_list.bond_list[pk];
                bo_jk = &pbond_jk->bo_data;
                BOA_jk = bo_jk->BO - control->thb_cut;

                if ( BOA_jk < 0.0 )
                {
                    continue;
                }

                ++num_thb_intrs;
            }
        }

        count[ pi ] = num_thb_intrs;
    }
}


/* Estimate the num. of three-body interactions */
CUDA_GLOBAL void k_estimate_valence_angles_opt( reax_atom *my_atoms,
        control_params *control, reax_list bond_list, int n, int N, int *count )
{
    extern __shared__ cub::WarpReduce<int>::TempStorage temp_i2[];
    int j, pi, pk, start_j, end_j, thread_id, lane_id, itr;
    int num_thb_intrs;
    real BOA_ij, BOA_jk;
    bond_data *pbond_ij, *pbond_jk;
    bond_order_data *bo_ij, *bo_jk;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    j = thread_id / warpSize;

    if ( j >= N )
    {
        return;
    }

    lane_id = thread_id % warpSize;
    start_j = Start_Index( j, &bond_list );
    end_j = End_Index( j, &bond_list );

    for ( pi = start_j; pi < end_j; ++pi )
    {
        num_thb_intrs = 0;

        pbond_ij = &bond_list.bond_list[pi];
        bo_ij = &pbond_ij->bo_data;
        BOA_ij = bo_ij->BO - control->thb_cut;

        if ( BOA_ij >= 0.0 && (j < n || pbond_ij->nbr < n) )
        {
            for ( itr = 0, pk = start_j + lane_id; itr < (end_j - start_j + warpSize - 1) / warpSize; ++itr )
            {
                pbond_jk = &bond_list.bond_list[pk];
                bo_jk = &pbond_jk->bo_data;
                BOA_jk = bo_jk->BO - control->thb_cut;

                if ( BOA_jk >= 0.0 )
                {
                    ++num_thb_intrs;
                }

                pk += warpSize;
            }

            num_thb_intrs = cub::WarpReduce<int>(temp_i2[j % (blockDim.x / warpSize)]).Sum(num_thb_intrs);
        }

        if ( lane_id == 0 )
        {
            count[ pi ] = num_thb_intrs;
        }
    }
}


static int Cuda_Estimate_Storage_Three_Body( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, reax_list **lists, int *thbody )
{
    int blocks, ret;

    ret = SUCCESS;

    cuda_memset( thbody, 0, system->total_bonds * sizeof(int),
            "Cuda_Estimate_Storage_Three_Body::thbody" );

//    k_estimate_valence_angles <<< control->blocks_n, control->block_size_n, 0, control->streams[0] >>>
//        ( system->d_my_atoms, (control_params *)control->d_control_params, 
//          *(lists[BONDS]), system->n, system->N, thbody );
//    cudaCheckError( );

    blocks = system->N * 32 / DEF_BLOCK_SIZE
        + (system->N * 32 % DEF_BLOCK_SIZE == 0 ? 0 : 1);

    k_estimate_valence_angles_opt <<< blocks, DEF_BLOCK_SIZE,
                              sizeof(cub::WarpReduce<int>::TempStorage) * (DEF_BLOCK_SIZE / 32),
                              control->streams[0] >>>
        ( system->d_my_atoms, (control_params *)control->d_control_params, 
          *(lists[BONDS]), system->n, system->N, thbody );
    cudaCheckError( );

    Cuda_Reduction_Sum( thbody, system->d_total_thbodies, system->total_bonds,
           control->streams[0] );

    sCudaMemcpyAsync( &system->total_thbodies, system->d_total_thbodies,
            sizeof(int), cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->streams[0] );

    if ( data->step - data->prev_steps == 0 )
    {
        system->total_thbodies = MAX( (int) (system->total_thbodies * SAFE_ZONE), MIN_3BODIES );
        system->total_thbodies_indices = system->total_bonds;

        Cuda_Make_List( system->total_thbodies_indices, system->total_thbodies,
                TYP_THREE_BODY, lists[THREE_BODIES] );
    }

    if ( system->total_thbodies > lists[THREE_BODIES]->max_intrs
            || system->total_bonds > lists[THREE_BODIES]->n )
    {
        if ( system->total_thbodies > lists[THREE_BODIES]->max_intrs )
        {
            system->total_thbodies = MAX( (int) (lists[THREE_BODIES]->max_intrs * SAFE_ZONE),
                    system->total_thbodies );
        }
        if ( system->total_bonds > lists[THREE_BODIES]->n )
        {
            system->total_thbodies_indices = MAX( (int) (lists[THREE_BODIES]->n * SAFE_ZONE),
                    system->total_bonds );
        }

        workspace->d_workspace->realloc.thbody = TRUE;
        ret = FAILURE;
    }

    return ret;
}


/* Initialize indices for three body list post reallocation
 *
 * indices: list indices
 * entries: num. of entries in list */
static void Cuda_Init_Three_Body_Indices( control_params *control, int *indices,
        int entries, reax_list **lists )
{
    reax_list *thbody;

    thbody = lists[THREE_BODIES];

    Cuda_Scan_Excl_Sum( indices, thbody->index, entries, control->streams[0] );
}


int Cuda_Compute_Valence_Angles( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control )
{
    int ret, *thbody, blocks;
    size_t s;
#if !defined(CUDA_ACCUM_ATOMIC)
    int update_energy;
    real *spad;
    rvec *rvec_spad;
#endif

#if !defined(CUDA_ACCUM_ATOMIC)
    s = MAX( sizeof(int) * system->total_bonds,
            (sizeof(real) * 3 + sizeof(rvec)) * system->N + sizeof(rvec) * control->blocks_n ),
#else
    s = sizeof(int) * system->total_bonds;
#endif

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            s, "Cuda_Compute_Valence_Angles::workspace->scratch" );

    thbody = (int *) workspace->scratch;
#if !defined(CUDA_ACCUM_ATOMIC)
    spad = (real *) workspace->scratch;
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#endif

    ret = Cuda_Estimate_Storage_Three_Body( system, control, data, workspace,
            lists, thbody );

    if ( ret == SUCCESS )
    {
        Cuda_Init_Three_Body_Indices( control, thbody, system->total_thbodies_indices, lists );

#if defined(CUDA_ACCUM_ATOMIC)
        cuda_memset( &((simulation_data *)data->d_simulation_data)->my_en.e_ang,
                0, sizeof(real), "Cuda_Compute_Valence_Angles::e_ang" );
        cuda_memset( &((simulation_data *)data->d_simulation_data)->my_en.e_pen,
                0, sizeof(real), "Cuda_Compute_Valence_Angles::e_pen" );
        cuda_memset( &((simulation_data *)data->d_simulation_data)->my_en.e_coa,
                0, sizeof(real), "Cuda_Compute_Valence_Angles::e_coa" );
        cuda_memset( &((simulation_data *)data->d_simulation_data)->my_ext_press,
                0, sizeof(rvec), "Cuda_Compute_Valence_Angles::my_ext_press" );
#endif

        if ( control->virial == 1 )
        {
            k_valence_angles_virial_part1 <<< control->blocks_n, control->block_size_n,
                                          0, control->streams[0] >>>
                ( system->d_my_atoms, system->reax_param.d_gp,
                  system->reax_param.d_sbp, system->reax_param.d_thbp, 
                  (control_params *) control->d_control_params,
                  *(workspace->d_workspace), *(lists[BONDS]), *(lists[THREE_BODIES]),
                  system->n, system->N, system->reax_param.num_atom_types, 
#if !defined(CUDA_ACCUM_ATOMIC)
                  spad, &spad[system->N], &spad[2 * system->N], (rvec *) (&spad[3 * system->N])
#else
                  &((simulation_data *)data->d_simulation_data)->my_en.e_ang,
                  &((simulation_data *)data->d_simulation_data)->my_en.e_pen,
                  &((simulation_data *)data->d_simulation_data)->my_en.e_coa,
                  &((simulation_data *)data->d_simulation_data)->my_ext_press
#endif
                );
        }
        else
        {
//            k_valence_angles_part1 <<< control->blocks_n, control->block_size_n,
//                                   0, control->streams[0] >>>
//                ( system->d_my_atoms, system->reax_param.d_gp,
//                  system->reax_param.d_sbp, system->reax_param.d_thbp, 
//                  (control_params *) control->d_control_params,
//                  *(workspace->d_workspace), *(lists[BONDS]), *(lists[THREE_BODIES]),
//                  system->n, system->N, system->reax_param.num_atom_types, 
//#if !defined(CUDA_ACCUM_ATOMIC)
//                  spad, &spad[system->N], &spad[2 * system->N]
//#else
//                  &((simulation_data *)data->d_simulation_data)->my_en.e_ang,
//                  &((simulation_data *)data->d_simulation_data)->my_en.e_pen,
//                  &((simulation_data *)data->d_simulation_data)->my_en.e_coa
//#endif
//                );

            blocks = system->N * 32 / DEF_BLOCK_SIZE
                + (system->N * 32 % DEF_BLOCK_SIZE == 0 ? 0 : 1);

            k_valence_angles_part1_opt <<< blocks, DEF_BLOCK_SIZE,
                                       (sizeof(cub::WarpScan<int>::TempStorage)
                                        + sizeof(cub::WarpReduce<double>::TempStorage)) * (DEF_BLOCK_SIZE / 32),
                                       control->streams[0] >>>
                ( system->d_my_atoms, system->reax_param.d_gp,
                  system->reax_param.d_sbp, system->reax_param.d_thbp, 
                  (control_params *) control->d_control_params,
                  *(workspace->d_workspace), *(lists[BONDS]), *(lists[THREE_BODIES]),
                  system->n, system->N, system->reax_param.num_atom_types, 
#if !defined(CUDA_ACCUM_ATOMIC)
                  spad, &spad[system->N], &spad[2 * system->N]
#else
                  &((simulation_data *)data->d_simulation_data)->my_en.e_ang,
                  &((simulation_data *)data->d_simulation_data)->my_en.e_pen,
                  &((simulation_data *)data->d_simulation_data)->my_en.e_coa
#endif
                );
        }
        cudaCheckError( );

#if !defined(CUDA_ACCUM_ATOMIC)
        if ( update_energy == TRUE )
        {
            Cuda_Reduction_Sum( spad,
                    &((simulation_data *)data->d_simulation_data)->my_en.e_ang,
                    system->N, control->streams[0] );

            Cuda_Reduction_Sum( &spad[system->N],
                    &((simulation_data *)data->d_simulation_data)->my_en.e_pen,
                    system->N, control->streams[0] );

            Cuda_Reduction_Sum( &spad[2 * system->N],
                    &((simulation_data *)data->d_simulation_data)->my_en.e_coa,
                    system->N, control->streams[0] );
        }

        if ( control->virial == 1 )
        {
            rvec_spad = (rvec *) (&spad[3 * system->N]);

            k_reduction_rvec <<< control->blocks_n, control->block_size_n,
                             sizeof(rvec) * (control->block_size_n / 32),
                             control->streams[0] >>>
                ( rvec_spad, &rvec_spad[system->N], system->N );
            cudaCheckError( );

            k_reduction_rvec <<< 1, control->blocks_pow_2_n,
                             sizeof(rvec) * (control->blocks_pow_2_n / 32),
                             control->streams[0] >>>
                ( &rvec_spad[system->N],
                  &((simulation_data *)data->d_simulation_data)->my_ext_press,
                  control->blocks_n );
            cudaCheckError( );
//            Cuda_Reduction_Sum( rvec_spad,
//                    &((simulation_data *)data->d_simulation_data)->my_ext_press,
//                    system->N, control->streams[0] );
        }
#endif

#if !defined(CUDA_ACCUM_ATOMIC)
        k_valence_angles_part2 <<< control->blocks_n, control->block_size_n,
                               0, control->streams[0] >>>
            ( system->d_my_atoms, (control_params *) control->d_control_params,
              *(workspace->d_workspace), *(lists[BONDS]), system->N );
        cudaCheckError( );
#endif
    }

    return ret;
}
