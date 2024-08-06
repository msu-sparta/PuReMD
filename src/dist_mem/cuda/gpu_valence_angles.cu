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

#include "gpu_valence_angles.h"

#include "gpu_helpers.h"
#include "gpu_list.h"
#include "gpu_reduction.h"
#include "gpu_utils.h"

#include "../index_utils.h"
#include "../vector.h"

#include <cub/util_ptx.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_reduce.cuh>


#define FULL_WARP_MASK (0xFFFFFFFF)


/**
 * Default product (multiplication) functor
 */
struct Prod
{
    template <typename T>
    GPU_HOST_DEVICE __forceinline__ T operator()(const T &a, const T &b) const
    {
        return a * b;
    }
};


/* Compute 3-body interactions, in which the main role is played by
   atom j, which sits in the middle of the other two atoms i and k. */
GPU_GLOBAL void k_valence_angles_part1( reax_atom const * const my_atoms,
        real const * const gp_l, single_body_parameters const * const sbp,
        two_body_parameters const * const tbp, three_body_header const * const thbh,
        real thb_cut, real const * const total_bond_order, real const * const Delta_boc,
        real const * const Delta, real const * const dDelta_lp, real const * const nlp,
        real const * const vlpex, real * const CdDelta, rvec * const f,
        const reax_list bond_list, reax_list thb_list, int n, int N, int num_atom_types,
        real * const e_ang_g, real * const e_pen_g, real * const e_coa_g )
{
    int i, j, pi, k, pk, t;
    int type_i, type_j, type_k;
    int start_j, end_j;
    int cnt, num_thb_intrs, thbh_ijk;
    real temp, temp_bo_jt, pBOjt7;
    real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    real CEpen1, CEpen2, CEpen3;
    real e_ang_, e_coa, e_coa_, e_pen, e_pen_;
    real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    real Cf7ij, Cf7jk, Cf8j, Cf9j;
    real f7_ij, f7_jk, f8_Dj, f9_Dj;
    real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
    real BOA_ij, BOA_jk;
    real Cdbo_ij, CdDelta_i, CdDelta_j;
    rvec f_i, f_j;
#define BL (bond_list.bond_list_gpu)
#define TBL (thb_list.three_body_list_gpu)

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= N )
    {
        return;
    }

    type_j = my_atoms[j].type;

    if ( sbp[type_j].thbp_cnt_j > 0 )
    {
        const real p_pen2 = gp_l[19];
        const real p_pen3 = gp_l[20];
        const real p_pen4 = gp_l[21];
        const real p_coa2 = gp_l[2];
        const real p_coa3 = gp_l[38];
        const real p_coa4 = gp_l[30];
        const real p_val6 = gp_l[14];
        const real p_val8 = gp_l[33];
        const real p_val9 = gp_l[16];
        const real p_val10 = gp_l[17];
        e_ang_ = 0.0;
        e_coa_ = 0.0;
        e_pen_ = 0.0;
        const real Delta_boc_j = Delta_boc[j];
        const real Delta_j = Delta[j];
        const real dDelta_lp_j = dDelta_lp[j];
        const real nlp_j = nlp[j];
        const real vlpex_j = vlpex[j];
        CdDelta_j = 0.0;
        rvec_MakeZero( f_j );

        start_j = Start_Index( j, &bond_list );
        end_j = End_Index( j, &bond_list );
        const real p_val3 = sbp[type_j].p_val3;
        const real p_val5 = sbp[type_j].p_val5;

        /* sum of pi and pi-pi BO terms for all neighbors of atom j,
         * used in determining the equilibrium angle between i-j-k */
        SBOp = 0.0;
        /* product of e^{-BO_j^8} terms for all neighbors of atom j,
         * used in determining the equilibrium angle between i-j-k */
        prod_SBO = 1.0;

        for ( t = start_j; t < end_j; ++t )
        {
            SBOp += BL.BO_pi[t] + BL.BO_pi2[t];
            temp = SQR( BL.BO[t] );
            temp *= temp;
            temp *= temp;
            prod_SBO *= EXP( -temp );
        }

        /* modifications to match Adri's code - 09/01/09 */
        if ( vlpex_j >= 0.0 )
        {
            vlpadj = 0.0;
            dSBO2 = prod_SBO - 1.0;
        }
        else
        {
            vlpadj = nlp_j;
            dSBO2 = (prod_SBO - 1.0) * (1.0 - p_val8 * dDelta_lp_j);
        }

        SBO = SBOp + (1.0 - prod_SBO) * (-Delta_boc_j - p_val8 * vlpadj);
        dSBO1 = -8.0 * prod_SBO * ( Delta_boc_j + p_val8 * vlpadj );

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

        expval6 = EXP( p_val6 * Delta_boc_j );

        for ( pi = start_j; pi < end_j; ++pi )
        {
            i = BL.nbr[pi];
            type_i = my_atoms[i].type;
            num_thb_intrs = Start_Index( pi, &thb_list );

            if ( tbp[index_tbp(type_i, type_j, num_atom_types)].thbp_cnt_ij > 0 )
            {
                BOA_ij = BL.BO[pi] - thb_cut;

                if ( BOA_ij >= 0.0 && (j < n || i < n) )
                {
                    const real total_bond_order_i = total_bond_order[i];
                    Cdbo_ij = 0.0;
                    CdDelta_i = 0.0;
                    rvec_MakeZero( f_i );

                    /* compute _ALL_ 3-body intrs */
                    for ( pk = start_j; pk < end_j; ++pk )
                    {
                        if ( pk == pi )
                        {
                            continue;
                        }

                        BOA_jk = BL.BO[pk] - thb_cut;

                        if ( BOA_jk < 0.0 )
                        {
                            continue;
                        }

                        k = BL.nbr[pk];
                        type_k = my_atoms[k].type;

                        Calculate_Theta( BL.dvec[pi], BL.d[pi], BL.dvec[pk], BL.d[pk],
                                &theta, &cos_theta );

                        Calculate_dCos_Theta( BL.dvec[pi], BL.d[pi], BL.dvec[pk], BL.d[pk],
                                &TBL.dcos_di[num_thb_intrs], &TBL.dcos_dj[num_thb_intrs],
                                &TBL.dcos_dk[num_thb_intrs] );

                        TBL.thb[num_thb_intrs] = k;
                        TBL.pthb[num_thb_intrs] = pk;
                        TBL.theta[num_thb_intrs] = theta;

                        sin_theta = SIN( theta );
                        if ( sin_theta < 1.0e-5 )
                        {
                            sin_theta = 1.0e-5;
                        }

                        ++num_thb_intrs;

                        /* Fortran ReaxFF code hard-codes the constant below
                         * as of 2019-02-27, so use that for now */
                        if ( j >= n || BOA_jk < 0.0 || BL.BO[pi] * BL.BO[pk] < 0.00001 )
//                        if ( j >= n || BOA_jk < 0.0 || BL.BO[pi] * BL.BO[pk] < SQR(thb_cut) )
                        {
                            continue;
                        }

                        thbh_ijk = index_thbp(type_i, type_j, type_k, num_atom_types);

                        for ( cnt = 0; cnt < thbh[thbh_ijk].cnt; ++cnt )
                        {
                            /* valence angle does not exist in the force field */
                            if ( FABS(thbh[thbh_ijk].prm[cnt].p_val1) < 0.001 )
                            {
                                continue;
                            }

                            three_body_parameters const * const thbp = &thbh[thbh_ijk].prm[cnt];

                            /* calculate valence angle energy */
                            const real p_val1 = thbp->p_val1;
                            const real p_val2 = thbp->p_val2;
                            const real p_val4 = thbp->p_val4;
                            const real p_val7 = thbp->p_val7;
                            theta_00 = thbp->theta_00;

                            exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                            f7_ij = 1.0 - exp3ij;
                            Cf7ij = p_val3 * p_val4
                                * POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                            exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                            f7_jk = 1.0 - exp3jk;
                            Cf7jk = p_val3 * p_val4
                                * POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                            expval7 = EXP( -p_val7 * Delta_boc_j );
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
                                e_ang_ += f7_ij * f7_jk * f8_Dj * expval12theta;
                            }

                            /* calculate penalty for double bonds in valency angles */
                            const real p_pen1 = thbp->p_pen1;

                            exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                            exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                            exp_pen3 = EXP( -p_pen3 * Delta_j );
                            exp_pen4 = EXP(  p_pen4 * Delta_j );
                            trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                            f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                            Cf9j = (-p_pen3 * exp_pen3 * trm_pen34
                                    - (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3
                                        + p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                            e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                            if ( pk < pi )
                            {
                                e_pen_ += e_pen;
                            }

                            CEpen1 = e_pen * Cf9j / f9_Dj;
                            temp = -2.0 * p_pen2 * e_pen;
                            CEpen2 = temp * (BOA_ij - 2.0);
                            CEpen3 = temp * (BOA_jk - 2.0);

                            /* calculate valency angle conjugation energy */
                            const real p_coa1 = thbp->p_coa1;

                            exp_coa2 = EXP( p_coa2 * Delta_boc_j );
                            e_coa = p_coa1
                                * EXP( -p_coa4 * SQR(BOA_ij - 1.5) )
                                * EXP( -p_coa4 * SQR(BOA_jk - 1.5) )
                                * EXP( -p_coa3 * SQR(total_bond_order_i - BOA_ij) )
                                * EXP( -p_coa3 * SQR(total_bond_order[k] - BOA_jk) )
                                / (1.0 + exp_coa2);

                            if ( pk < pi )
                            {
                                e_coa_ += e_coa;
                            }

                            CEcoa1 = -2.0 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                            CEcoa2 = -2.0 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                            CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1.0 + exp_coa2);
                            CEcoa4 = -2.0 * p_coa3 * (total_bond_order_i - BOA_ij) * e_coa;
                            CEcoa5 = -2.0 * p_coa3 * (total_bond_order[k] - BOA_jk) * e_coa;

                            /* calculate force contributions */
                            if ( pk < pi )
                            {
                                Cdbo_ij += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
#if defined(GPU_STREAM_SINGLE_ACCUM)
                                atomicAdd( &BL.Cdbo[pk], CEval2 + CEpen3 + (CEcoa2 - CEcoa5) );
#else
                                BL.Cdbo[pk] += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
#endif
                                CdDelta_j += (CEval3 + CEval7) + CEpen1 + CEcoa3;
                                CdDelta_i += CEcoa4;
#if defined(GPU_KERNEL_ATOMIC)
                                atomicAdd( &CdDelta[k], CEcoa5 );
#else
                                BL.CdDelta_val[pk] += CEcoa5;
#endif

                                for ( t = start_j; t < end_j; ++t )
                                {
                                    temp_bo_jt = BL.BO[t];
                                    temp = CUBE( temp_bo_jt );
                                    pBOjt7 = temp * temp * temp_bo_jt;

#if defined(GPU_STREAM_SINGLE_ACCUM)
                                    atomicAdd( &BL.Cdbo[t], CEval6 * pBOjt7 );
                                    atomicAdd( &BL.Cdbopi[t], CEval5 );
                                    atomicAdd( &BL.Cdbopi2[t], CEval5 );
#else
                                    BL.Cdbo[t] += CEval6 * pBOjt7;
                                    BL.Cdbopi[t] = CEval5;
                                    BL.Cdbopi2[t] = CEval5;
#endif
                                }

                                rvec_ScaledAdd( f_i, CEval8, TBL.dcos_di[num_thb_intrs] );
                                rvec_ScaledAdd( f_j, CEval8, TBL.dcos_dj[num_thb_intrs] );
#if defined(GPU_KERNEL_ATOMIC)
                                atomic_rvecScaledAdd( f[k], CEval8, TBL.dcos_dk[num_thb_intrs] );
#else
                                rvec_ScaledAdd( BL.f_val[pk], CEval8, TBL.dcos_dk[num_thb_intrs] );
#endif
                            }
                        }
                    }

#if defined(GPU_STREAM_SINGLE_ACCUM)
                    atomicAdd( &BL.Cdbo[pi], Cdbo_ij );
#else
                    BL.Cdbo[pi] += Cdbo_ij;
#endif
#if defined(GPU_KERNEL_ATOMIC)
                    atomicAdd( &CdDelta[i], CdDelta_i );
                    atomic_rvecAdd( f[i], f_i );
#else
                    BL.CdDelta_val[pi] += CdDelta_i;
                    rvec_Add( BL.f_val[pi], f_i );
#endif
                }
            }

            Set_End_Index( pi, num_thb_intrs, &thb_list );
        }

#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
        atomic_rvecAdd( f[j], f_j );
        atomicAdd( &CdDelta[j], CdDelta_j );
#else
        rvec_Add( f[j], f_j );
        CdDelta[j] += CdDelta_j;
#endif
#if defined(GPU_ATOMIC_EV)
        atomicAdd( (double *) e_ang_g, (double) e_ang_ );
        atomicAdd( (double *) e_coa_g, (double) e_coa_ );
        atomicAdd( (double *) e_pen_g, (double) e_pen_ );
#else
        e_ang_g[j] = e_ang_;
        e_coa_g[j] = e_coa_;
        e_pen_g[j] = e_pen_;
#endif
    }

#undef BL
#undef TBL
}


/* Compute 3-body interactions, in which the main role is played by
   atom j, which sits in the middle of the other two atoms i and k. */
GPU_GLOBAL void k_valence_angles_part1_opt( reax_atom const * const my_atoms,
        real const * const gp_l, single_body_parameters const * const sbp,
        two_body_parameters const * const tbp, three_body_header const * const thbh,
        real thb_cut, real const * const total_bond_order, real const * const Delta_boc,
        real const * const Delta, real const * const dDelta_lp, real const * const nlp,
        real const * const vlpex, real * const CdDelta, rvec * const f,
        const reax_list bond_list, reax_list thb_list, int n, int N,
        int num_atom_types, real * const e_ang_g, real * const e_pen_g, real * const e_coa_g )
{
    extern __shared__ cub::WarpScan<int>::TempStorage temp_i[];
    cub::WarpReduce<double>::TempStorage *temp_d;
    int i, j, pi, k, pk, t, thread_id, warp_id, lane_id, itr;
    int type_i, type_j, type_k;
    int start_j, end_j;
    int cnt, num_thb_intrs, offset, flag, thbh_ijk;
    real temp, temp_bo_jt, pBOjt7;
    real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    real CEpen1, CEpen2, CEpen3;
    real e_ang_, e_coa, e_coa_, e_pen, e_pen_;
    real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    real Cf7ij, Cf7jk, Cf8j, Cf9j;
    real f7_ij, f7_jk, f8_Dj, f9_Dj;
    real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
    real BOA_ij, BOA_jk;
    real Cdbo_ij, CdDelta_i, CdDelta_j;
    rvec f_i, f_j;
#define BL (bond_list.bond_list_gpu)
#define TBL (thb_list.three_body_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    j = thread_id / warpSize;

    if ( j >= N )
    {
        return;
    }

    type_j = my_atoms[j].type;

    if ( sbp[type_j].thbp_cnt_j > 0 )
    {
        temp_d = (cub::WarpReduce<double>::TempStorage *) &temp_i[blockDim.x / warpSize];
        warp_id = threadIdx.x / warpSize;
        lane_id = thread_id % warpSize;
        const real p_pen2 = gp_l[19];
        const real p_pen3 = gp_l[20];
        const real p_pen4 = gp_l[21];
        const real p_coa2 = gp_l[2];
        const real p_coa3 = gp_l[38];
        const real p_coa4 = gp_l[30];
        const real p_val6 = gp_l[14];
        const real p_val8 = gp_l[33];
        const real p_val9 = gp_l[16];
        const real p_val10 = gp_l[17];
        e_ang_ = 0.0;
        e_coa_ = 0.0;
        e_pen_ = 0.0;
        const real Delta_boc_j = Delta_boc[j];
        const real Delta_j = Delta[j];
        const real dDelta_lp_j = dDelta_lp[j];
        const real nlp_j = nlp[j];
        const real vlpex_j = vlpex[j];
        CdDelta_j = 0.0;
        rvec_MakeZero( f_j );

        start_j = Start_Index( j, &bond_list );
        end_j = End_Index( j, &bond_list );
        const real p_val3 = sbp[type_j].p_val3;
        const real p_val5 = sbp[type_j].p_val5;

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
                SBOp += BL.BO_pi[t] + BL.BO_pi2[t];
                temp = SQR( BL.BO[t] );
                temp *= temp;
                temp *= temp;
                prod_SBO *= EXP( -temp );
            }

            t += warpSize;
        }

        SBOp = cub::WarpReduce<double>(temp_d[warp_id]).Sum(SBOp);
        prod_SBO = cub::WarpReduce<double>(temp_d[warp_id]).Reduce(prod_SBO, Prod());

        /* broadcast redux results from lane 0 */
        SBOp = cub::ShuffleIndex<WARP_SIZE>( SBOp, 0, FULL_WARP_MASK );
        prod_SBO = cub::ShuffleIndex<WARP_SIZE>( prod_SBO, 0, FULL_WARP_MASK );

        /* modifications to match Adri's code - 09/01/09 */
        if ( vlpex_j >= 0.0 )
        {
            vlpadj = 0.0;
            dSBO2 = prod_SBO - 1.0;
        }
        else
        {
            vlpadj = nlp_j;
            dSBO2 = (prod_SBO - 1.0) * (1.0 - p_val8 * dDelta_lp_j);
        }

        SBO = SBOp + (1.0 - prod_SBO) * (-Delta_boc_j - p_val8 * vlpadj);
        dSBO1 = -8.0 * prod_SBO * ( Delta_boc_j + p_val8 * vlpadj );

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

        expval6 = EXP( p_val6 * Delta_boc_j );

        for ( pi = start_j; pi < end_j; ++pi )
        {
            i = BL.nbr[pi];
            type_i = my_atoms[i].type;
            num_thb_intrs = Start_Index( pi, &thb_list );

            if ( tbp[index_tbp(type_i, type_j, num_atom_types)].thbp_cnt_ij > 0 )
            {
                BOA_ij = BL.BO[pi] - thb_cut;

                if ( BOA_ij >= 0.0 && (j < n || i < n) )
                {
                    const real total_bond_order_i = total_bond_order[i];
                    Cdbo_ij = 0.0;
                    CdDelta_i = 0.0;
                    rvec_MakeZero( f_i );

                    /* compute _ALL_ 3-body intrs */
                    for ( itr = 0, pk = start_j + lane_id; itr < (end_j - start_j + warpSize - 1) / warpSize; ++itr )
                    {
                        if ( pk != pi && pk < end_j )
                        {
                            BOA_jk = BL.BO[pk] - thb_cut;
            
                            offset = (BOA_jk >= 0.0) ? 1 : 0;
                        }
                        else
                        {
                            offset = 0;
                        }

                        flag = (offset == 1) ? TRUE : FALSE;
                        cub::WarpScan<int>(temp_i[warp_id]).ExclusiveSum(offset, offset);

                        if ( flag == TRUE )
                        {
                            k = BL.nbr[pk];
                            type_k = my_atoms[k].type;

                            Calculate_Theta( BL.dvec[pi], BL.d[pi], BL.dvec[pk], BL.d[pk],
                                    &theta, &cos_theta );

                            Calculate_dCos_Theta( BL.dvec[pi], BL.d[pi], BL.dvec[pk], BL.d[pk],
                                    &TBL.dcos_di[num_thb_intrs + offset],
                                    &TBL.dcos_dj[num_thb_intrs + offset],
                                    &TBL.dcos_dk[num_thb_intrs + offset] );

                            TBL.thb[num_thb_intrs + offset] = k;
                            TBL.pthb[num_thb_intrs + offset] = pk;
                            TBL.theta[num_thb_intrs + offset] = theta;

                            sin_theta = SIN( theta );
                            if ( sin_theta < 1.0e-5 )
                            {
                                sin_theta = 1.0e-5;
                            }

                            /* Fortran ReaxFF code hard-codes the constant below
                             * as of 2019-02-27, so use that for now */
                            if ( j < n && BOA_jk >= 0.0 && BL.BO[pi] * BL.BO[pk] >= 0.00001 )
//                            if ( j < n && BOA_jk >= 0.0 && BL.BO[pi] * BL.BO[pk] >= SQR(thb_cut) )
                            {
                                thbh_ijk = index_thbp(type_i, type_j, type_k, num_atom_types);

                                for ( cnt = 0; cnt < thbh[thbh_ijk].cnt; ++cnt )
                                {
                                    /* valence angle does not exist in the force field */
                                    if ( FABS(thbh[thbh_ijk].prm[cnt].p_val1) < 0.001 )
                                    {
                                        continue;
                                    }

                                    three_body_parameters const * const thbp = &thbh[thbh_ijk].prm[cnt];

                                    /* calculate valence angle energy */
                                    const real p_val1 = thbp->p_val1;
                                    const real p_val2 = thbp->p_val2;
                                    const real p_val4 = thbp->p_val4;
                                    const real p_val7 = thbp->p_val7;
                                    theta_00 = thbp->theta_00;

                                    exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                                    f7_ij = 1.0 - exp3ij;
                                    Cf7ij = p_val3 * p_val4
                                        * POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                                    exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                                    f7_jk = 1.0 - exp3jk;
                                    Cf7jk = p_val3 * p_val4
                                        * POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                                    expval7 = EXP( -p_val7 * Delta_boc_j );
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
                                        e_ang_ += f7_ij * f7_jk * f8_Dj * expval12theta;
                                    }

                                    /* calculate penalty for double bonds in valency angles */
                                    const real p_pen1 = thbp->p_pen1;

                                    exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                                    exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                                    exp_pen3 = EXP( -p_pen3 * Delta_j );
                                    exp_pen4 = EXP(  p_pen4 * Delta_j );
                                    trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                                    f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                                    Cf9j = (-p_pen3 * exp_pen3 * trm_pen34
                                            - (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3
                                                + p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                                    e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                                    if ( pk < pi )
                                    {
                                        e_pen_ += e_pen;
                                    }

                                    CEpen1 = e_pen * Cf9j / f9_Dj;
                                    temp = -2.0 * p_pen2 * e_pen;
                                    CEpen2 = temp * (BOA_ij - 2.0);
                                    CEpen3 = temp * (BOA_jk - 2.0);

                                    /* calculate valency angle conjugation energy */
                                    const real p_coa1 = thbp->p_coa1;

                                    exp_coa2 = EXP( p_coa2 * Delta_boc_j );
                                    e_coa = p_coa1
                                        * EXP( -p_coa4 * SQR(BOA_ij - 1.5) )
                                        * EXP( -p_coa4 * SQR(BOA_jk - 1.5) )
                                        * EXP( -p_coa3 * SQR(total_bond_order_i - BOA_ij) )
                                        * EXP( -p_coa3 * SQR(total_bond_order[k] - BOA_jk) )
                                        / (1.0 + exp_coa2);

                                    if ( pk < pi )
                                    {
                                        e_coa_ += e_coa;
                                    }

                                    CEcoa1 = -2.0 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                                    CEcoa2 = -2.0 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                                    CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1.0 + exp_coa2);
                                    CEcoa4 = -2.0 * p_coa3 * (total_bond_order_i - BOA_ij) * e_coa;
                                    CEcoa5 = -2.0 * p_coa3 * (total_bond_order[k] - BOA_jk) * e_coa;

                                    /* calculate force contributions */
                                    if ( pk < pi )
                                    {
                                        Cdbo_ij += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
#if defined(GPU_STREAM_SINGLE_ACCUM)
                                        atomicAdd( &BL.Cdbo[pk], CEval2 + CEpen3 + (CEcoa2 - CEcoa5) );
#else
                                        BL.Cdbo[pk] += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
#endif
                                        CdDelta_j += (CEval3 + CEval7) + CEpen1 + CEcoa3;
                                        CdDelta_i += CEcoa4;
#if defined(GPU_KERNEL_ATOMIC)
                                        atomicAdd( &CdDelta[k], CEcoa5 );
#else
                                        BL.CdDelta_val[pk] += CEcoa5;
#endif

                                        for ( t = start_j; t < end_j; ++t )
                                        {
                                            temp_bo_jt = BL.BO[t];
                                            temp = CUBE( temp_bo_jt );
                                            pBOjt7 = temp * temp * temp_bo_jt;

#if defined(GPU_STREAM_SINGLE_ACCUM)
                                            atomicAdd( &BL.Cdbo[t], CEval6 * pBOjt7 );
                                            atomicAdd( &BL.Cdbopi[t], CEval5 );
                                            atomicAdd( &BL.Cdbopi2[t], CEval5 );
#else
                                            BL.Cdbo[t] += CEval6 * pBOjt7;
                                            BL.Cdbopi[t] = CEval5;
                                            BL.Cdbopi2[t] = CEval5;
#endif
                                        }

                                        rvec_ScaledAdd( f_i, CEval8, TBL.dcos_di[num_thb_intrs + offset] );
                                        rvec_ScaledAdd( f_j, CEval8, TBL.dcos_dj[num_thb_intrs + offset] );
#if defined(GPU_KERNEL_ATOMIC)
                                        atomic_rvecScaledAdd( f[k], CEval8, TBL.dcos_dk[num_thb_intrs + offset] );
#else
                                        rvec_ScaledAdd( BL.f_val[pk], CEval8, TBL.dcos_dk[num_thb_intrs + offset] );
#endif
                                    }
                                }
                            }
                        }

                        /* get num_thb_intrs from thread in last lane */
                        num_thb_intrs = num_thb_intrs + offset + (flag == TRUE ? 1 : 0);
                        num_thb_intrs = cub::ShuffleIndex<WARP_SIZE>( num_thb_intrs, warpSize - 1, FULL_WARP_MASK );

                        pk += warpSize;
                    }

                    Cdbo_ij = cub::WarpReduce<double>(temp_d[warp_id]).Sum(Cdbo_ij);
                    CdDelta_i = cub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_i);
                    f_i[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[0]);
                    f_i[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[1]);
                    f_i[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[2]);

                    if ( lane_id == 0 )
                    {
#if defined(GPU_STREAM_SINGLE_ACCUM)
                        atomicAdd( &BL.Cdbo[pi], Cdbo_ij );
#else
                        BL.Cdbo[pi] += Cdbo_ij;
#endif
#if defined(GPU_KERNEL_ATOMIC)
                        atomicAdd( &CdDelta[i], CdDelta_i );
                        atomic_rvecAdd( f[i], f_i );
#else
                        BL.CdDelta_val[pi] += CdDelta_i;
                        rvec_Add( BL.f_val[pi], f_i );
#endif
                    }
                }
            }

            if ( lane_id == 0 )
            {
                Set_End_Index( pi, num_thb_intrs, &thb_list );
            }
        }

        CdDelta_j = cub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_j);
        f_j[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[0]);
        f_j[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[1]);
        f_j[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[2]);
        e_ang_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_ang_);
        e_coa_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_coa_);
        e_pen_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_pen_);

        if ( lane_id == 0 )
        {
#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
            atomic_rvecAdd( f[j], f_j );
            atomicAdd( &CdDelta[j], CdDelta_j );
#else
            rvec_Add( f[j], f_j );
            CdDelta[j] += CdDelta_j;
#endif
#if defined(GPU_ATOMIC_EV)
            atomicAdd( (double *) e_ang_g, (double) e_ang_ );
            atomicAdd( (double *) e_coa_g, (double) e_coa_ );
            atomicAdd( (double *) e_pen_g, (double) e_pen_ );
#else
            e_ang_g[j] = e_ang_;
            e_coa_g[j] = e_coa_;
            e_pen_g[j] = e_pen_;
#endif
        }
    }

#undef BL
#undef TBL
}


/* Compute 3-body interactions (no caching), in which the main role is played by
   atom j, which sits in the middle of the other two atoms i and k. */
GPU_GLOBAL void k_valence_angles_part1_no_cache_opt( reax_atom const * const my_atoms,
        real const * const gp_l, single_body_parameters const * const sbp,
        two_body_parameters const * const tbp, three_body_header const * const thbh,
        real thb_cut, real const * const total_bond_order, real const * const Delta_boc,
        real const * const Delta, real const * const dDelta_lp, real const * const nlp,
        real const * const vlpex, real * const CdDelta, rvec * const f,
        reax_list bond_list, int n, int N, int num_atom_types, real * const e_ang_g,
        real * const e_pen_g, real * const e_coa_g )
{
    extern __shared__ cub::WarpScan<int>::TempStorage temp_i[];
    cub::WarpReduce<double>::TempStorage *temp_d;
    int i, j, pi, k, pk, t, thread_id, warp_id, lane_id, itr;
    int type_i, type_j, type_k;
    int start_j, end_j;
    int cnt, offset, flag, thbh_ijk;
    real temp, temp_bo_jt, pBOjt7;
    real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    real CEpen1, CEpen2, CEpen3;
    real e_ang_, e_coa, e_coa_, e_pen, e_pen_;
    real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    real Cf7ij, Cf7jk, Cf8j, Cf9j;
    real f7_ij, f7_jk, f8_Dj, f9_Dj;
    real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
    real BOA_ij, BOA_jk;
    real Cdbo_ij, CdDelta_i, CdDelta_j;
    rvec f_i, f_j, dcos_di, dcos_dj, dcos_dk;
#define BL (bond_list.bond_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    j = thread_id / warpSize;

    if ( j >= N )
    {
        return;
    }

    type_j = my_atoms[j].type;

    if ( sbp[type_j].thbp_cnt_j > 0 )
    {
        temp_d = (cub::WarpReduce<double>::TempStorage *) &temp_i[blockDim.x / warpSize];
        warp_id = threadIdx.x / warpSize;
        lane_id = thread_id % warpSize;
        const real p_pen2 = gp_l[19];
        const real p_pen3 = gp_l[20];
        const real p_pen4 = gp_l[21];
        const real p_coa2 = gp_l[2];
        const real p_coa3 = gp_l[38];
        const real p_coa4 = gp_l[30];
        const real p_val6 = gp_l[14];
        const real p_val8 = gp_l[33];
        const real p_val9 = gp_l[16];
        const real p_val10 = gp_l[17];
        e_ang_ = 0.0;
        e_coa_ = 0.0;
        e_pen_ = 0.0;
        const real Delta_boc_j = Delta_boc[j];
        const real Delta_j = Delta[j];
        const real dDelta_lp_j = dDelta_lp[j];
        const real nlp_j = nlp[j];
        const real vlpex_j = vlpex[j];
        CdDelta_j = 0.0;
        rvec_MakeZero( f_j );

        start_j = Start_Index( j, &bond_list );
        end_j = End_Index( j, &bond_list );
        const real p_val3 = sbp[type_j].p_val3;
        const real p_val5 = sbp[type_j].p_val5;

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
                SBOp += BL.BO_pi[t] + BL.BO_pi2[t];
                temp = SQR( BL.BO[t] );
                temp *= temp;
                temp *= temp;
                prod_SBO *= EXP( -temp );
            }

            t += warpSize;
        }

        SBOp = cub::WarpReduce<double>(temp_d[warp_id]).Sum(SBOp);
        prod_SBO = cub::WarpReduce<double>(temp_d[warp_id]).Reduce(prod_SBO, Prod());

        /* broadcast redux results from lane 0 */
        SBOp = cub::ShuffleIndex<WARP_SIZE>( SBOp, 0, FULL_WARP_MASK );
        prod_SBO = cub::ShuffleIndex<WARP_SIZE>( prod_SBO, 0, FULL_WARP_MASK );

        /* modifications to match Adri's code - 09/01/09 */
        if ( vlpex_j >= 0.0 )
        {
            vlpadj = 0.0;
            dSBO2 = prod_SBO - 1.0;
        }
        else
        {
            vlpadj = nlp_j;
            dSBO2 = (prod_SBO - 1.0) * (1.0 - p_val8 * dDelta_lp_j);
        }

        SBO = SBOp + (1.0 - prod_SBO) * (-Delta_boc_j - p_val8 * vlpadj);
        dSBO1 = -8.0 * prod_SBO * ( Delta_boc_j + p_val8 * vlpadj );

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

        expval6 = EXP( p_val6 * Delta_boc_j );

        for ( pi = start_j; pi < end_j; ++pi )
        {
            i = BL.nbr[pi];
            type_i = my_atoms[i].type;

            if ( tbp[index_tbp(type_i, type_j, num_atom_types)].thbp_cnt_ij > 0 )
            {
                BOA_ij = BL.BO[pi] - thb_cut;

                if ( BOA_ij >= 0.0 && (j < n || i < n) )
                {
                    const real total_bond_order_i = total_bond_order[i];
                    Cdbo_ij = 0.0;
                    CdDelta_i = 0.0;
                    rvec_MakeZero( f_i );

                    /* compute _ALL_ 3-body intrs */
                    for ( itr = 0, pk = start_j + lane_id; itr < (end_j - start_j + warpSize - 1) / warpSize; ++itr )
                    {
                        if ( pk != pi && pk < end_j )
                        {
                            BOA_jk = BL.BO[pk] - thb_cut;
            
                            offset = (BOA_jk >= 0.0) ? 1 : 0;
                        }
                        else
                        {
                            offset = 0;
                        }

                        flag = (offset == 1) ? TRUE : FALSE;
                        cub::WarpScan<int>(temp_i[warp_id]).ExclusiveSum(offset, offset);

                        if ( flag == TRUE )
                        {
                            k = BL.nbr[pk];
                            type_k = my_atoms[k].type;

                            Calculate_Theta( BL.dvec[pi], BL.d[pi], BL.dvec[pk], BL.d[pk],
                                    &theta, &cos_theta );

                            Calculate_dCos_Theta( BL.dvec[pi], BL.d[pi], BL.dvec[pk], BL.d[pk],
                                    &dcos_di, &dcos_dj, &dcos_dk );

                            sin_theta = SIN( theta );
                            if ( sin_theta < 1.0e-5 )
                            {
                                sin_theta = 1.0e-5;
                            }

                            /* Fortran ReaxFF code hard-codes the constant below
                             * as of 2019-02-27, so use that for now */
                            if ( j < n && BOA_jk >= 0.0 && BL.BO[pi] * BL.BO[pk] >= 0.00001 )
//                            if ( j < n && BOA_jk >= 0.0 && BL.BO[pi] * BL.BO[pk] >= SQR(thb_cut) )
                            {
                                thbh_ijk = index_thbp(type_i, type_j, type_k, num_atom_types);

                                for ( cnt = 0; cnt < thbh[thbh_ijk].cnt; ++cnt )
                                {
                                    /* valence angle does not exist in the force field */
                                    if ( FABS(thbh[thbh_ijk].prm[cnt].p_val1) < 0.001 )
                                    {
                                        continue;
                                    }

                                    three_body_parameters const * const thbp = &thbh[thbh_ijk].prm[cnt];

                                    /* calculate valence angle energy */
                                    const real p_val1 = thbp->p_val1;
                                    const real p_val2 = thbp->p_val2;
                                    const real p_val4 = thbp->p_val4;
                                    const real p_val7 = thbp->p_val7;
                                    theta_00 = thbp->theta_00;

                                    exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                                    f7_ij = 1.0 - exp3ij;
                                    Cf7ij = p_val3 * p_val4
                                        * POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                                    exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                                    f7_jk = 1.0 - exp3jk;
                                    Cf7jk = p_val3 * p_val4
                                        * POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                                    expval7 = EXP( -p_val7 * Delta_boc_j );
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
                                        e_ang_ += f7_ij * f7_jk * f8_Dj * expval12theta;
                                    }

                                    /* calculate penalty for double bonds in valency angles */
                                    const real p_pen1 = thbp->p_pen1;

                                    exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                                    exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                                    exp_pen3 = EXP( -p_pen3 * Delta_j );
                                    exp_pen4 = EXP(  p_pen4 * Delta_j );
                                    trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                                    f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                                    Cf9j = (-p_pen3 * exp_pen3 * trm_pen34
                                            - (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3
                                                + p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                                    e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                                    if ( pk < pi )
                                    {
                                        e_pen_ += e_pen;
                                    }

                                    CEpen1 = e_pen * Cf9j / f9_Dj;
                                    temp = -2.0 * p_pen2 * e_pen;
                                    CEpen2 = temp * (BOA_ij - 2.0);
                                    CEpen3 = temp * (BOA_jk - 2.0);

                                    /* calculate valency angle conjugation energy */
                                    const real p_coa1 = thbp->p_coa1;

                                    exp_coa2 = EXP( p_coa2 * Delta_boc_j );
                                    e_coa = p_coa1
                                        * EXP( -p_coa4 * SQR(BOA_ij - 1.5) )
                                        * EXP( -p_coa4 * SQR(BOA_jk - 1.5) )
                                        * EXP( -p_coa3 * SQR(total_bond_order_i - BOA_ij) )
                                        * EXP( -p_coa3 * SQR(total_bond_order[k] - BOA_jk) )
                                        / (1.0 + exp_coa2);

                                    if ( pk < pi )
                                    {
                                        e_coa_ += e_coa;
                                    }

                                    CEcoa1 = -2.0 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                                    CEcoa2 = -2.0 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                                    CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1.0 + exp_coa2);
                                    CEcoa4 = -2.0 * p_coa3 * (total_bond_order_i - BOA_ij) * e_coa;
                                    CEcoa5 = -2.0 * p_coa3 * (total_bond_order[k] - BOA_jk) * e_coa;

                                    /* calculate force contributions */
                                    if ( pk < pi )
                                    {
                                        Cdbo_ij += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
#if defined(GPU_STREAM_SINGLE_ACCUM)
                                        atomicAdd( &BL.Cdbo[pk], CEval2 + CEpen3 + (CEcoa2 - CEcoa5) );
#else
                                        BL.Cdbo[pk] += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
#endif
                                        CdDelta_j += (CEval3 + CEval7) + CEpen1 + CEcoa3;
                                        CdDelta_i += CEcoa4;
#if defined(GPU_KERNEL_ATOMIC)
                                        atomicAdd( &CdDelta[k], CEcoa5 );
#else
                                        BL.CdDelta_val[pk] += CEcoa5;
#endif

                                        for ( t = start_j; t < end_j; ++t )
                                        {
                                            temp_bo_jt = BL.BO[t];
                                            temp = CUBE( temp_bo_jt );
                                            pBOjt7 = temp * temp * temp_bo_jt;

#if defined(GPU_STREAM_SINGLE_ACCUM)
                                            atomicAdd( &BL.Cdbo[t], CEval6 * pBOjt7 );
                                            atomicAdd( &BL.Cdbopi[t], CEval5 );
                                            atomicAdd( &BL.Cdbopi2[t], CEval5 );
#else
                                            BL.Cdbo[t] += CEval6 * pBOjt7;
                                            BL.Cdbopi[t] = CEval5;
                                            BL.Cdbopi2[t] = CEval5;
#endif
                                        }

                                        rvec_ScaledAdd( f_i, CEval8, dcos_di );
                                        rvec_ScaledAdd( f_j, CEval8, dcos_dj );
#if defined(GPU_KERNEL_ATOMIC)
                                        atomic_rvecScaledAdd( f[k], CEval8, dcos_dk );
#else
                                        rvec_ScaledAdd( BL.f_val[pk], CEval8, dcos_dk );
#endif
                                    }
                                }
                            }
                        }

                        pk += warpSize;
                    }

                    Cdbo_ij = cub::WarpReduce<double>(temp_d[warp_id]).Sum(Cdbo_ij);
                    CdDelta_i = cub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_i);
                    f_i[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[0]);
                    f_i[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[1]);
                    f_i[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_i[2]);

                    if ( lane_id == 0 )
                    {
#if defined(GPU_STREAM_SINGLE_ACCUM)
                        atomicAdd( &BL.Cdbo[pi], Cdbo_ij );
#else
                        BL.Cdbo[pi] += Cdbo_ij;
#endif
#if defined(GPU_KERNEL_ATOMIC)
                        atomicAdd( &CdDelta[i], CdDelta_i );
                        atomic_rvecAdd( f[i], f_i );
#else
                        BL.CdDelta_val[pi] += CdDelta_i;
                        rvec_Add( BL.f_val[pi], f_i );
#endif
                    }
                }
            }
        }

        CdDelta_j = cub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_j);
        f_j[0] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[0]);
        f_j[1] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[1]);
        f_j[2] = cub::WarpReduce<double>(temp_d[warp_id]).Sum(f_j[2]);
        e_ang_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_ang_);
        e_coa_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_coa_);
        e_pen_ = cub::WarpReduce<double>(temp_d[warp_id]).Sum(e_pen_);

        if ( lane_id == 0 )
        {
#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
            atomic_rvecAdd( f[j], f_j );
            atomicAdd( &CdDelta[j], CdDelta_j );
#else
            rvec_Add( f[j], f_j );
            CdDelta[j] += CdDelta_j;
#endif
#if defined(GPU_ATOMIC_EV)
            atomicAdd( (double *) e_ang_g, (double) e_ang_ );
            atomicAdd( (double *) e_coa_g, (double) e_coa_ );
            atomicAdd( (double *) e_pen_g, (double) e_pen_ );
#else
            e_ang_g[j] = e_ang_;
            e_coa_g[j] = e_coa_;
            e_pen_g[j] = e_pen_;
#endif
        }
    }

#undef BL
}


/* Compute 3-body interactions, in which the main role is played by
   atom j, which sits in the middle of the other two atoms i and k. */
GPU_GLOBAL void k_valence_angles_virial_part1( reax_atom const * const my_atoms,
        real const * const gp_l, single_body_parameters const * const sbp,
        three_body_header const * const thbh, real thb_cut,
        real const * const total_bond_order, real const * const Delta_boc, real const * const Delta,
        real const * const dDelta_lp, real const * const nlp, real const * const vlpex,
        real * const CdDelta, rvec * const f, const reax_list bond_list,
        reax_list thb_list, int n, int N, int num_atom_types, real * const e_ang_g,
        real * const e_pen_g, real * const e_coa_g, rvec * const ext_press_g )
{
    int i, j, pi, k, pk, t;
    int type_i, type_j, type_k;
    int start_j, end_j;
    int cnt, num_thb_intrs, thbh_ijk;
    real temp, temp_bo_jt, pBOjt7;
    real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    real CEpen1, CEpen2, CEpen3;
    real e_ang_, e_coa, e_coa_, e_pen, e_pen_;
    real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    real Cf7ij, Cf7jk, Cf8j, Cf9j;
    real f7_ij, f7_jk, f8_Dj, f9_Dj;
    real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
    real BOA_ij, BOA_jk;
    real Cdbo_ij, CdDelta_i, CdDelta_j;
    rvec rvec_temp, f_i, f_j, ext_press;
#define BL (bond_list.bond_list_gpu)
#define TBL (thb_list.three_body_list_gpu)

    j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( j >= N )
    {
        return;
    }

    const real p_pen2 = gp_l[19];
    const real p_pen3 = gp_l[20];
    const real p_pen4 = gp_l[21];
    const real p_coa2 = gp_l[2];
    const real p_coa3 = gp_l[38];
    const real p_coa4 = gp_l[30];
    const real p_val6 = gp_l[14];
    const real p_val8 = gp_l[33];
    const real p_val9 = gp_l[16];
    const real p_val10 = gp_l[17];
    e_ang_ = 0.0;
    e_coa_ = 0.0;
    e_pen_ = 0.0;
    const real Delta_boc_j = Delta_boc[j];
    const real Delta_j = Delta[j];
    const real dDelta_lp_j = dDelta_lp[j];
    const real nlp_j = nlp[j];
    const real vlpex_j = vlpex[j];
    CdDelta_j = 0.0;
    rvec_MakeZero( f_j );
    rvec_MakeZero( ext_press );

    type_j = my_atoms[j].type;
    start_j = Start_Index( j, &bond_list );
    end_j = End_Index( j, &bond_list );
    const real p_val3 = sbp[type_j].p_val3;
    const real p_val5 = sbp[type_j].p_val5;

    /* sum of pi and pi-pi BO terms for all neighbors of atom j,
     * used in determining the equilibrium angle between i-j-k */
    SBOp = 0.0;
    /* product of e^{-BO_j^8} terms for all neighbors of atom j,
     * used in determining the equilibrium angle between i-j-k */
    prod_SBO = 1.0;

    for ( t = start_j; t < end_j; ++t )
    {
        SBOp += BL.BO_pi[t] + BL.BO_pi2[t];
        temp = SQR( BL.BO[t] );
        temp *= temp;
        temp *= temp;
        prod_SBO *= EXP( -temp );
    }

    /* modifications to match Adri's code - 09/01/09 */
    if ( vlpex_j >= 0.0 )
    {
        vlpadj = 0.0;
        dSBO2 = prod_SBO - 1.0;
    }
    else
    {
        vlpadj = nlp_j;
        dSBO2 = (prod_SBO - 1.0) * (1.0 - p_val8 * dDelta_lp_j);
    }

    SBO = SBOp + (1.0 - prod_SBO) * (-Delta_boc_j - p_val8 * vlpadj);
    dSBO1 = -8.0 * prod_SBO * ( Delta_boc_j + p_val8 * vlpadj );

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

    expval6 = EXP( p_val6 * Delta_boc_j );

    for ( pi = start_j; pi < end_j; ++pi )
    {
        num_thb_intrs = Start_Index( pi, &thb_list );
        BOA_ij = BL.BO[pi] - thb_cut;

        if ( BOA_ij >= 0.0 && (j < n || BL.nbr[pi] < n) )
        {
            i = BL.nbr[pi];
            type_i = my_atoms[i].type;
            const real total_bond_order_i = total_bond_order[i];
            Cdbo_ij = 0.0;
            CdDelta_i = 0.0;
            rvec_MakeZero( f_i );

            /* compute _ALL_ 3-body intrs */
            for ( pk = start_j; pk < end_j; ++pk )
            {
                if ( pk == pi )
                {
                    continue;
                }

                BOA_jk = BL.BO[pk] - thb_cut;

                if ( BOA_jk < 0.0 )
                {
                    continue;
                }

                k = BL.nbr[pk];
                type_k = my_atoms[k].type;

                Calculate_Theta( BL.dvec[pi], BL.d[pi], BL.dvec[pk], BL.d[pk],
                        &theta, &cos_theta );

                Calculate_dCos_Theta( BL.dvec[pi], BL.d[pi], BL.dvec[pk], BL.d[pk],
                        &TBL.dcos_di[num_thb_intrs], &TBL.dcos_dj[num_thb_intrs],
                        &TBL.dcos_dk[num_thb_intrs] );

                TBL.thb[num_thb_intrs] = k;
                TBL.pthb[num_thb_intrs] = pk;
                TBL.theta[num_thb_intrs] = theta;

                sin_theta = SIN( theta );
                if ( sin_theta < 1.0e-5 )
                {
                    sin_theta = 1.0e-5;
                }

                ++num_thb_intrs;

                /* Fortran ReaxFF code hard-codes the constant below
                 * as of 2019-02-27, so use that for now */
                if ( j >= n || BOA_jk < 0.0 || (BL.BO[pi] * BL.BO[pk]) < 0.00001 )
//                if ( j >= n || BOA_jk < 0.0 || (BL.BO[pi] * BL.BO[pk]) < SQR(thb_cut) )
                {
                    continue;
                }

                thbh_ijk = index_thbp(type_i, type_j, type_k, num_atom_types);

                for ( cnt = 0; cnt < thbh[thbh_ijk].cnt; ++cnt )
                {
                    /* valence angle does not exist in the force field */
                    if ( FABS(thbh[thbh_ijk].prm[cnt].p_val1) < 0.001 )
                    {
                        continue;
                    }

                    three_body_parameters const * const thbp = &thbh[thbh_ijk].prm[cnt];

                    /* calculate valence angle energy */
                    const real p_val1 = thbp->p_val1;
                    const real p_val2 = thbp->p_val2;
                    const real p_val4 = thbp->p_val4;
                    const real p_val7 = thbp->p_val7;
                    theta_00 = thbp->theta_00;

                    exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                    f7_ij = 1.0 - exp3ij;
                    Cf7ij = p_val3 * p_val4
                        * POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                    exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                    f7_jk = 1.0 - exp3jk;
                    Cf7jk = p_val3 * p_val4
                        * POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                    expval7 = EXP( -p_val7 * Delta_boc_j );
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
                        e_ang_ += f7_ij * f7_jk * f8_Dj * expval12theta;
                    }

                    /* calculate penalty for double bonds in valency angles */
                    const real p_pen1 = thbp->p_pen1;

                    exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                    exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                    exp_pen3 = EXP( -p_pen3 * Delta_j );
                    exp_pen4 = EXP(  p_pen4 * Delta_j );
                    trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                    f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                    Cf9j = (-p_pen3 * exp_pen3 * trm_pen34
                            - (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3
                                + p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                    e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                    if ( pk < pi )
                    {
                        e_pen_ += e_pen;
                    }

                    CEpen1 = e_pen * Cf9j / f9_Dj;
                    temp = -2.0 * p_pen2 * e_pen;
                    CEpen2 = temp * (BOA_ij - 2.0);
                    CEpen3 = temp * (BOA_jk - 2.0);

                    /* calculate valency angle conjugation energy */
                    const real p_coa1 = thbp->p_coa1;

                    exp_coa2 = EXP( p_coa2 * Delta_boc_j );
                    e_coa = p_coa1
                        * EXP( -p_coa4 * SQR(BOA_ij - 1.5) )
                        * EXP( -p_coa4 * SQR(BOA_jk - 1.5) )
                        * EXP( -p_coa3 * SQR(total_bond_order_i - BOA_ij) )
                        * EXP( -p_coa3 * SQR(total_bond_order[k] - BOA_jk) )
                        / (1.0 + exp_coa2);

                    if ( pk < pi )
                    {
                        e_coa_ += e_coa;
                    }

                    CEcoa1 = -2.0 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                    CEcoa2 = -2.0 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                    CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1.0 + exp_coa2);
                    CEcoa4 = -2.0 * p_coa3 * (total_bond_order_i - BOA_ij) * e_coa;
                    CEcoa5 = -2.0 * p_coa3 * (total_bond_order[k] - BOA_jk) * e_coa;

                    /* calculate force contributions */
                    if ( pk < pi )
                    {
                        Cdbo_ij += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
#if defined(GPU_STREAM_SINGLE_ACCUM)
                        atomicAdd( &BL.Cdbo[pk], CEval2 + CEpen3 + (CEcoa2 - CEcoa5) );
#else
                        BL.Cdbo[pk] += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
#endif
                        CdDelta_j += (CEval3 + CEval7) + CEpen1 + CEcoa3;
                        CdDelta_i += CEcoa4;
#if defined(GPU_KERNEL_ATOMIC)
                        atomicAdd( &CdDelta[k], CEcoa5 );
#else
                        BL.CdDelta_val[pk] += CEcoa5;
#endif

                        for ( t = start_j; t < end_j; ++t )
                        {
                            temp_bo_jt = BL.BO[t];
                            temp = CUBE( temp_bo_jt );
                            pBOjt7 = temp * temp * temp_bo_jt;

#if defined(GPU_STREAM_SINGLE_ACCUM)
                            atomicAdd( &BL.Cdbo[t], CEval6 * pBOjt7 );
                            atomicAdd( &BL.Cdbopi[t], CEval5 );
                            atomicAdd( &BL.Cdbopi2[t], CEval5 );
#else
                            BL.Cdbo[t] += CEval6 * pBOjt7;
                            BL.Cdbopi[t] = CEval5;
                            BL.Cdbopi2[t] = CEval5;
#endif
                        }

                        /* terms not related to bond order derivatives are
                         * added directly into forces and pressure vector/tensor */
                        rvec_Scale( rvec_temp, CEval8, TBL.dcos_di[num_thb_intrs] );
                        rvec_Add( f_i, rvec_temp );
                        rvec_iMultiply( rvec_temp, BL.rel_box[pi], rvec_temp );
                        rvec_Add( ext_press, rvec_temp );

                        rvec_ScaledAdd( f_j, CEval8, TBL.dcos_dj[num_thb_intrs] );

                        rvec_Scale( rvec_temp, CEval8, TBL.dcos_dk[num_thb_intrs] );
#if defined(GPU_KERNEL_ATOMIC)
                        atomic_rvecAdd( f[k], rvec_temp );
#else
                        rvec_Add( BL.f_val[pk], rvec_temp );
#endif
                        rvec_iMultiply( rvec_temp, BL.rel_box[pk], rvec_temp );
                        rvec_Add( ext_press, rvec_temp );
                    }
                }
            }

#if defined(GPU_STREAM_SINGLE_ACCUM)
            atomicAdd( &BL.Cdbo[pi], Cdbo_ij );
#else
            BL.Cdbo[pi] += Cdbo_ij;
#endif
#if defined(GPU_KERNEL_ATOMIC)
            atomicAdd( &CdDelta[i], CdDelta_i );
            atomic_rvecAdd( f[i], f_i );
#else
            BL.CdDelta_val[pi] += CdDelta_i;
            rvec_Add( BL.f_val[pi], f_i );
#endif
        }

        Set_End_Index( pi, num_thb_intrs, &thb_list );
    }

#if defined(GPU_KERNEL_ATOMIC) || defined(GPU_STREAM_SINGLE_ACCUM)
    atomic_rvecAdd( f[j], f_j );
    atomicAdd( &CdDelta[j], CdDelta_j );
#else
    rvec_Add( f[j], f_j );
    CdDelta[j] += CdDelta_j;
#endif
#if defined(GPU_ATOMIC_EV)
    atomicAdd( (double *) e_ang_g, (double) e_ang_ );
    atomicAdd( (double *) e_coa_g, (double) e_coa_ );
    atomicAdd( (double *) e_pen_g, (double) e_pen_ );
    atomic_rvecAdd( *ext_press_g, ext_press );
#else
    e_ang_g[j] = e_ang_;
    e_coa_g[j] = e_coa_;
    e_pen_g[j] = e_pen_;
    rvec_Copy( ext_press_g[j], ext_press );
#endif

#undef BL
#undef TBL
}


#if !defined(GPU_KERNEL_ATOMIC)
GPU_GLOBAL void k_valence_angles_part2( real * const CdDelta, rvec * const f,
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
        CdDelta_i += BL.CdDelta_val[BL.sym_index[pj]];
        rvec_Add( f_i, BL.f_val[BL.sym_index[pj]] );
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


/* Estimate the num. of three-body interactions */
GPU_GLOBAL void k_estimate_valence_angles( reax_atom const * const my_atoms,
        real thb_cut, reax_list bond_list, int n, int N, int * const count )
{
    int j, pi, pk, start_j, end_j, num_thb_intrs;
    real BOA_ij, BOA_jk;
#define BL (bond_list.bond_list_gpu)

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

        BOA_ij = BL.BO[pi] - thb_cut;

        if ( BOA_ij >= 0.0 && (j < n || BL.nbr[pi] < n) )
        {
            for ( pk = start_j; pk < end_j; ++pk )
            {
                if ( pk == pi )
                {
                    continue;
                }

                BOA_jk = BL.BO[pk] - thb_cut;

                if ( BOA_jk < 0.0 )
                {
                    continue;
                }

                ++num_thb_intrs;
            }
        }

        /* round up to the nearest multiple of warp size to ensure that reads along
         * rows can be coalesced */
        count[pi] = (num_thb_intrs + warpSize - 1) / warpSize * warpSize;
    }

#undef BL
}


/* Estimate the num. of three-body interactions */
GPU_GLOBAL void k_estimate_valence_angles_opt( reax_atom const * const my_atoms,
        real thb_cut, reax_list bond_list, int n, int N, int * const count )
{
    extern __shared__ cub::WarpReduce<int>::TempStorage temp_i2[];
    int j, pi, pk, start_j, end_j, thread_id, warp_id, lane_id, itr;
    int num_thb_intrs;
    real BOA_ij, BOA_jk;
#define BL (bond_list.bond_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    j = thread_id / warpSize;

    if ( j >= N )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize;
    start_j = Start_Index( j, &bond_list );
    end_j = End_Index( j, &bond_list );

    for ( pi = start_j; pi < end_j; ++pi )
    {
        num_thb_intrs = 0;

        BOA_ij = BL.BO[pi] - thb_cut;

        if ( BOA_ij >= 0.0 && (j < n || BL.nbr[pi] < n) )
        {
            for ( itr = 0, pk = start_j + lane_id; itr < (end_j - start_j + warpSize - 1) / warpSize; ++itr )
            {
                if ( pk < end_j )
                {
                    BOA_jk = BL.BO[pk] - thb_cut;
    
                    if ( BOA_jk >= 0.0 )
                    {
                        ++num_thb_intrs;
                    }
                }

                pk += warpSize;
            }

            num_thb_intrs = cub::WarpReduce<int>(temp_i2[warp_id]).Sum(num_thb_intrs);
        }

        if ( lane_id == 0 )
        {
            /* round up to the nearest multiple of warp size to ensure that reads along
             * rows can be coalesced */
            count[pi] = (num_thb_intrs + warpSize - 1) / warpSize * warpSize;
        }
    }

#undef BL
}


static int GPU_Estimate_Storage_Three_Body( reax_system * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list **lists, int * const thbody )
{
    int ret;

    ret = SUCCESS;

    sCudaMemsetAsync( thbody, 0, sizeof(int) * system->total_bonds,
            control->gpu_streams[3], __FILE__, __LINE__ );

//    k_estimate_valence_angles <<< control->blocks_N, control->gpu_block_size,
//                              0, control->gpu_streams[3] >>>
//        ( system->d_my_atoms, control->thb_cut, *(lists[BONDS]), system->n, system->N, thbody );
//    cudaCheckError( );

    k_estimate_valence_angles_opt <<< control->blocks_warp_N, control->gpu_block_size,
                              sizeof(cub::WarpReduce<int>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                              control->gpu_streams[3] >>>
        ( system->d_my_atoms, control->thb_cut, *(lists[BONDS]), system->n, system->N, thbody );
    cudaCheckError( );

    GPU_Reduction_Sum( thbody, system->d_total_thbodies, system->total_bonds,
           3, control->gpu_streams[3] );

    sCudaMemcpyAsync( &system->total_thbodies, system->d_total_thbodies,
            sizeof(int), cudaMemcpyDeviceToHost, control->gpu_streams[3], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->gpu_streams[3] );

    if ( data->step - data->prev_steps == 0 )
    {
        system->total_thbodies = MAX( (int) (system->total_thbodies * SAFE_ZONE), MIN_3BODIES );
        system->total_thbodies_indices = system->total_bonds;

        GPU_Make_List( system->total_thbodies_indices, system->total_thbodies,
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

        workspace->realloc[RE_THBODY] = TRUE;
        ret = FAILURE;
    }

    return ret;
}


/* Initialize indices for three body list post reallocation
 *
 * indices: list indices
 * entries: num. of entries in list */
static void GPU_Init_Three_Body_Indices( control_params const * const control,
        int * const indices, int entries, reax_list **lists )
{
    reax_list *thbody;

    thbody = lists[THREE_BODIES];

    GPU_Scan_Excl_Sum( indices, thbody->index, entries, 3, control->gpu_streams[3] );
}


int GPU_Compute_Valence_Angles( reax_system * const system,
        control_params const * const control, 
        simulation_data * const data, storage * const workspace, 
        reax_list **lists, output_controls const * const out_control )
{
    int ret, *thbody;
    size_t s;
#if !defined(GPU_ATOMIC_EV)
    int update_energy;
    real *spad;
    rvec *rvec_spad;
#endif

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->gpu_time_events[TE_VALENCE_START], control->gpu_streams[3] );
#endif

#if defined(GPU_ATOMIC_EV)
    s = sizeof(int) * system->total_bonds;
#else
    s = MAX( sizeof(int) * system->total_bonds,
            (sizeof(real) * 3 + sizeof(rvec)) * system->N ),
#endif

    sCudaCheckMalloc( &workspace->d_workspace->scratch[3],
            &workspace->d_workspace->scratch_size[3], s, __FILE__, __LINE__ );

    thbody = (int *) workspace->d_workspace->scratch[3];
#if !defined(GPU_ATOMIC_EV)
    spad = (real *) workspace->d_workspace->scratch[3];
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#endif

    cudaStreamWaitEvent( control->gpu_streams[3], control->gpu_stream_events[SE_BOND_ORDER_DONE], 0 );

    ret = GPU_Estimate_Storage_Three_Body( system, control, data, workspace,
            lists, thbody );

    if ( ret == SUCCESS )
    {
        GPU_Init_Three_Body_Indices( control, thbody, system->total_thbodies_indices, lists );

#if defined(GPU_ATOMIC_EV)
        sCudaMemsetAsync( &data->d_my_en[E_ANG], 0, sizeof(real) * 3,
                control->gpu_streams[3], __FILE__, __LINE__ );
        if ( control->virial == 1 )
        {
            sCudaMemsetAsync( &data->d_my_ext_press,
                    0, sizeof(rvec), control->gpu_streams[3], __FILE__, __LINE__ );
        }
#endif

        if ( control->virial == 0 )
        {
//            k_valence_angles_part1 <<< control->blocks_N, control->gpu_block_size,
//                                   0, control->gpu_streams[3] >>>
//                ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp,
//                  system->reax_param.d_tbp, system->reax_param.d_thbp, control->thb_cut,
//                  workspace->d_workspace->total_bond_order, workspace->d_workspace->Delta_boc,
//                  workspace->d_workspace->Delta, workspace->d_workspace->dDelta_lp,
//                  workspace->d_workspace->nlp, workspace->d_workspace->vlpex,
//                  workspace->d_workspace->CdDelta, workspace->d_workspace->f,
//                  *(lists[BONDS]), *(lists[THREE_BODIES]),
//                  system->n, system->N, system->reax_param.num_atom_types, 
//#if defined(GPU_ATOMIC_EV)
//                  &data->d_my_en[E_ANG], &data->d_my_en[E_PEN], &data->d_my_en[E_COA]
//#else
//                  spad, &spad[system->N], &spad[2 * system->N]
//#endif
//                );

            k_valence_angles_part1_opt <<< control->blocks_warp_N, control->gpu_block_size,
                                       (sizeof(cub::WarpScan<int>::TempStorage)
                                        + sizeof(cub::WarpReduce<double>::TempStorage)) * (control->gpu_block_size / WARP_SIZE),
                                       control->gpu_streams[3] >>>
                ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp,
                  system->reax_param.d_tbp, system->reax_param.d_thbp, control->thb_cut,
                  workspace->d_workspace->total_bond_order, workspace->d_workspace->Delta_boc,
                  workspace->d_workspace->Delta, workspace->d_workspace->dDelta_lp,
                  workspace->d_workspace->nlp, workspace->d_workspace->vlpex,
                  workspace->d_workspace->CdDelta, workspace->d_workspace->f,
                  *(lists[BONDS]), *(lists[THREE_BODIES]), system->n, system->N,
                  system->reax_param.num_atom_types, 
#if defined(GPU_ATOMIC_EV)
                  &data->d_my_en[E_ANG], &data->d_my_en[E_PEN], &data->d_my_en[E_COA]
#else
                  spad, &spad[system->N], &spad[2 * system->N]
#endif
                );

//            k_valence_angles_part1_no_cache_opt <<< control->blocks_warp_N, control->gpu_block_size,
//                                       (sizeof(cub::WarpScan<int>::TempStorage)
//                                        + sizeof(cub::WarpReduce<double>::TempStorage)) * (control->gpu_block_size / WARP_SIZE),
//                                       control->gpu_streams[3] >>>
//                ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp,
//                  system->reax_param.d_tbp, system->reax_param.d_thbp, control->thb_cut,
//                  workspace->d_workspace->total_bond_order, workspace->d_workspace->Delta_boc,
//                  workspace->d_workspace->Delta, workspace->d_workspace->dDelta_lp,
//                  workspace->d_workspace->nlp, workspace->d_workspace->vlpex,
//                  workspace->d_workspace->CdDelta, workspace->d_workspace->f,
//                  *(lists[BONDS]), system->n, system->N,
//                  system->reax_param.num_atom_types, 
//#if defined(GPU_ATOMIC_EV)
//                  &data->d_my_en[E_ANG], &data->d_my_en[E_PEN], &data->d_my_en[E_COA]
//#else
//                  spad, &spad[system->N], &spad[2 * system->N]
//#endif
//                );
        }
        else if ( control->virial == 1 )
        {
            k_valence_angles_virial_part1 <<< control->blocks_N, control->gpu_block_size,
                                          0, control->gpu_streams[3] >>>
                ( system->d_my_atoms, system->reax_param.gp.d_l,
                  system->reax_param.d_sbp, system->reax_param.d_thbp, control->thb_cut,
                  workspace->d_workspace->total_bond_order, workspace->d_workspace->Delta_boc,
                  workspace->d_workspace->Delta, workspace->d_workspace->dDelta_lp,
                  workspace->d_workspace->nlp, workspace->d_workspace->vlpex,
                  workspace->d_workspace->CdDelta, workspace->d_workspace->f,
                  *(lists[BONDS]), *(lists[THREE_BODIES]), system->n, system->N,
                  system->reax_param.num_atom_types, 
#if defined(GPU_ATOMIC_EV)
                  &data->d_my_en[E_ANG], &data->d_my_en[E_PEN], &data->d_my_en[E_COA],
                  &data->d_my_ext_press
#else
                  spad, &spad[system->N], &spad[2 * system->N], (rvec *) (&spad[3 * system->N])
#endif
                );
        }
        cudaCheckError( );

#if !defined(GPU_ATOMIC_EV)
        if ( update_energy == TRUE )
        {
            GPU_Reduction_Sum( spad, &data->d_my_en[E_ANG], system->N, 3,
                    control->gpu_streams[3] );

            GPU_Reduction_Sum( &spad[system->N], &data->d_my_en[E_PEN], system->N, 3,
                    control->gpu_streams[3] );

            GPU_Reduction_Sum( &spad[2 * system->N], &data->d_my_en[E_COA], system->N, 3,
                    control->gpu_streams[3] );

            if ( control->virial == 1 )
            {
                rvec_spad = (rvec *) (&spad[3 * system->N]);

                GPU_Reduction_Sum( rvec_spad, &data->d_my_ext_press,
                        system->N, 3, control->gpu_streams[3] );
            }
        }
#endif

#if !defined(GPU_KERNEL_ATOMIC)
        k_valence_angles_part2 <<< control->blocks_N, control->gpu_block_size,
                               0, control->gpu_streams[3] >>>
            ( workspace->d_workspace->CdDelta, workspace->d_workspace->f,
              *(lists[BONDS]), system->N );
        cudaCheckError( );
#endif
    }

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->gpu_time_events[TE_VALENCE_STOP], control->gpu_streams[3] );
#endif

    return ret;
}
