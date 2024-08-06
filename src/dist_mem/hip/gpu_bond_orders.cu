#include "hip/hip_runtime.h"

#include "gpu_bond_orders.h"

#include "gpu_helpers.h"
#include "gpu_list.h"
#include "gpu_utils.h"
#include "gpu_reduction.h"

#include "../index_utils.h"
#include "../bond_orders.h"

#include <hipcub/warp/warp_reduce.hpp>


GPU_DEVICE void GPU_Add_dBond_to_Forces_NPT( int i, int pj,
        rvec const * const dDeltap_self, real const * const CdDelta,
        rvec * const f, reax_list * const bond_list, rvec * const f_i,
        rvec data_ext_press )
{
    int k, j, pk, sym_index;
    real C1dbo, C2dbo, C3dbo;
    real C1dbopi, C2dbopi, C3dbopi, C4dbopi;
    real C1dbopi2, C2dbopi2, C3dbopi2, C4dbopi2;
    real C1dDelta, C2dDelta, C3dDelta;
    ivec rel_box;
    rvec temp, ext_press;
#define BL (bond_list->bond_list_gpu)

    j = BL.nbr[pj];
    sym_index = BL.sym_index[pj];

    C1dbo = BL.C1dbo[pj] * (BL.Cdbo[pj] + BL.Cdbo[sym_index]);
    C2dbo = BL.C2dbo[pj] * (BL.Cdbo[pj] + BL.Cdbo[sym_index]);
    C3dbo = BL.C3dbo[pj] * (BL.Cdbo[pj] + BL.Cdbo[sym_index]);

    C1dbopi = BL.C1dbopi[pj] * (BL.Cdbopi[pj] + BL.Cdbopi[sym_index]);
    C2dbopi = BL.C2dbopi[pj] * (BL.Cdbopi[pj] + BL.Cdbopi[sym_index]);
    C3dbopi = BL.C3dbopi[pj] * (BL.Cdbopi[pj] + BL.Cdbopi[sym_index]);
    C4dbopi = BL.C4dbopi[pj] * (BL.Cdbopi[pj] + BL.Cdbopi[sym_index]);

    C1dbopi2 = BL.C1dbopi2[pj] * (BL.Cdbopi2[pj] + BL.Cdbopi2[sym_index]);
    C2dbopi2 = BL.C2dbopi2[pj] * (BL.Cdbopi2[pj] + BL.Cdbopi2[sym_index]);
    C3dbopi2 = BL.C3dbopi2[pj] * (BL.Cdbopi2[pj] + BL.Cdbopi2[sym_index]);
    C4dbopi2 = BL.C4dbopi2[pj] * (BL.Cdbopi2[pj] + BL.Cdbopi2[sym_index]);

    C1dDelta = BL.C1dbo[pj] * (CdDelta[i] + CdDelta[j]);
    C2dDelta = BL.C2dbo[pj] * (CdDelta[i] + CdDelta[j]);
    C3dDelta = BL.C3dbo[pj] * (CdDelta[i] + CdDelta[j]);

    /************************************
    * forces related to atom i          *
    * first neighbors of atom i         *
    ************************************/
    for ( pk = Start_Index(i, bond_list); pk < End_Index(i, bond_list); ++pk )
    {
        /* 2nd, dBO; 2nd, dDelta; 3rd, dBOpi; 3rd, dBOpi2 */
        rvec_Scale( temp, -(C2dbo + C2dDelta + C3dbopi + C3dbopi2), BL.dBOp[pk] );

        /* force */
#if defined(GPU_KERNEL_ATOMIC)
        k = BL.nbr[pk];
        atomic_rvecAdd( f[k], temp );
#else
        atomic_rvecAdd( BL.f_bo[pk], temp );
#endif

        /* pressure */
        rvec_iMultiply( ext_press, BL.rel_box[pk], temp );
        rvec_Add( data_ext_press, ext_press );
    }

    /* then atom i itself */
    /* 1st, dBO; 1st, dBO; 2nd, dBOpi; 2nd, dBO_pi2 */
    rvec_Scale( temp, C1dbo + C1dDelta + C2dbopi + C2dbopi2, BL.dBOp[pj] );

    /* 2nd, dBO; 2nd, dBO; 3rd, dBOpi; 3rd, dBO_pi2 */
    rvec_ScaledAdd( temp, C2dbo + C2dDelta + C3dbopi + C3dbopi2, dDeltap_self[i] );

    /* 1st, dBOpi */
    rvec_ScaledAdd( temp, C1dbopi, BL.dln_BOp_pi[pj] );

    /* 1st, dBO_pi2 */
    rvec_ScaledAdd( temp, C1dbopi2, BL.dln_BOp_pi2[pj] );

    /* force */
    rvec_Add( *f_i, temp );
    /* ext pressure due to i is dropped, counting force on j will be enough */

    /******************************************************
     * forces and pressure related to atom j               *
     * first neighbors of atom j                           *
     ******************************************************/
    for ( pk = Start_Index(j, bond_list); pk < End_Index(j, bond_list); ++pk )
    {
        k = BL.nbr[pk];

        /* 3rd, dBO; 3rd, dDelta; 4th, dBOpi; 4th, dBOpi2 */
        rvec_Scale( temp, -(C3dbo + C3dDelta + C4dbopi + C4dbopi2), BL.dBOp[pk] );

        /* force */
#if defined(GPU_KERNEL_ATOMIC)
        atomic_rvecAdd( f[k], temp );
#else
        atomic_rvecAdd( BL.f_bo[pk], temp );
#endif

        /* pressure */
        if ( k != i )
        {
            ivec_Sum( rel_box, BL.rel_box[pk], BL.rel_box[pj] ); //rel_box(k, i)
            rvec_iMultiply( ext_press, rel_box, temp );
            rvec_Add( data_ext_press, ext_press );
        }
    }

    /* then atom j itself */
    /* 1st, dBO; 1st, dBO; 2nd, dBOpi; 2nd, dBOpi2 */
    rvec_Scale( temp, -(C1dbo + C1dDelta + C2dbopi + C2dbopi2), BL.dBOp[pj] );

    /* 2nd, dBO; 2nd, dBO; 3rd, dBOpi; 3rd, dBOpi2 */
    rvec_ScaledAdd( temp, C3dbo + C3dDelta + C4dbopi + C4dbopi2, dDeltap_self[j] );

    /* 1st, dBOpi */
    rvec_ScaledAdd( temp, -C1dbopi, BL.dln_BOp_pi[pj] );

    /* 1st, dBOpi2 */
    rvec_ScaledAdd( temp, -C1dbopi2, BL.dln_BOp_pi2[pj] );

    /* force */
#if defined(GPU_KERNEL_ATOMIC)
    atomic_rvecAdd( f[j], temp );
#else
    atomic_rvecAdd( BL.f_bo[pj], temp );
#endif
    /* pressure */
    rvec_iMultiply( ext_press, BL.rel_box[pj], temp );
    rvec_Add( data_ext_press, ext_press );

#undef BL
}


GPU_DEVICE void GPU_Add_dBond_to_Forces( int i, int pj,
        rvec const * const dDeltap_self, real const * const CdDelta,
        rvec * const f, reax_list * const bond_list, rvec * const f_i )
{
    int j, pk, sym_index;
#if defined(GPU_KERNEL_ATOMIC)
    int k;
#endif
    real C1dbo, C2dbo, C3dbo;
    real C1dbopi, C2dbopi, C3dbopi, C4dbopi;
    real C1dbopi2, C2dbopi2, C3dbopi2, C4dbopi2;
    real C1dDelta, C2dDelta, C3dDelta;
    rvec temp;
#define BL (bond_list->bond_list_gpu)

    j = BL.nbr[pj];
    sym_index = BL.sym_index[pj];

    C1dbo = BL.C1dbo[pj] * (BL.Cdbo[pj] + BL.Cdbo[sym_index]);
    C2dbo = BL.C2dbo[pj] * (BL.Cdbo[pj] + BL.Cdbo[sym_index]);
    C3dbo = BL.C3dbo[pj] * (BL.Cdbo[pj] + BL.Cdbo[sym_index]);

    C1dbopi = BL.C1dbopi[pj] * (BL.Cdbopi[pj] + BL.Cdbopi[sym_index]);
    C2dbopi = BL.C2dbopi[pj] * (BL.Cdbopi[pj] + BL.Cdbopi[sym_index]);
    C3dbopi = BL.C3dbopi[pj] * (BL.Cdbopi[pj] + BL.Cdbopi[sym_index]);
    C4dbopi = BL.C4dbopi[pj] * (BL.Cdbopi[pj] + BL.Cdbopi[sym_index]);

    C1dbopi2 = BL.C1dbopi2[pj] * (BL.Cdbopi2[pj] + BL.Cdbopi2[sym_index]);
    C2dbopi2 = BL.C2dbopi2[pj] * (BL.Cdbopi2[pj] + BL.Cdbopi2[sym_index]);
    C3dbopi2 = BL.C3dbopi2[pj] * (BL.Cdbopi2[pj] + BL.Cdbopi2[sym_index]);
    C4dbopi2 = BL.C4dbopi2[pj] * (BL.Cdbopi2[pj] + BL.Cdbopi2[sym_index]);

    C1dDelta = BL.C1dbo[pj] * (CdDelta[i] + CdDelta[j]);
    C2dDelta = BL.C2dbo[pj] * (CdDelta[i] + CdDelta[j]);
    C3dDelta = BL.C3dbo[pj] * (CdDelta[i] + CdDelta[j]);

    for ( pk = Start_Index(i, bond_list); pk < End_Index(i, bond_list); ++pk )
    {
        /* 2nd, dBO, dDelta, dBOpi, dBOpi2 */
        rvec_Scale( temp, -(C2dbo + C2dDelta + C3dbopi + C3dbopi2), BL.dBOp[pk] );

#if defined(GPU_KERNEL_ATOMIC)
        k = BL.nbr[pk];
        atomic_rvecAdd( f[k], temp );
#else
        atomic_rvecAdd( BL.f_bo[pk], temp );
#endif
    }

    /* 1st, dBO; 1st, dBO; 2nd dBOpi; 2nd dBO_pi2 */
    rvec_Scale( temp, C1dbo + C1dDelta + C2dbopi + C2dbopi2, BL.dBOp[pj] );

    /* 2nd, dBO; 2nd, dBO; 3rd, dBOpi; 3rd, dBO_pi2 */
    rvec_ScaledAdd( temp, C2dbo + C2dDelta + C3dbopi + C3dbopi2, dDeltap_self[i] );

    /* 1st, dBOpi */
    rvec_ScaledAdd( temp, C1dbopi, BL.dln_BOp_pi[pj] );

    /* 1st, dBO_pi2 */
    rvec_ScaledAdd( temp, C1dbopi2, BL.dln_BOp_pi2[pj] );

    rvec_Add( *f_i, temp );

    for ( pk = Start_Index(j, bond_list); pk < End_Index(j, bond_list); ++pk )
    {
        /* 3rd, dBO; 3rd, dDelta; 4th, dBOpi; 4th, dBOpi2 */
        rvec_Scale( temp, -(C3dbo + C3dDelta + C4dbopi + C4dbopi2), BL.dBOp[pk] );

#if defined(GPU_KERNEL_ATOMIC)
        k = BL.nbr[pk];
        atomic_rvecAdd( f[k], temp );
#else
        atomic_rvecAdd( BL.f_bo[pk], temp );
#endif
    }

    /* 1st, dBO; 1st, dBO; 2nd, dBOpi; 2nd, dBOpi2 */
    rvec_Scale( temp, -(C1dbo + C1dDelta + C2dbopi + C2dbopi2), BL.dBOp[pj] );

    /* 2nd, dBO; 2nd, dBO; 3rd, dBOpi; 3rd, dBOpi2 */
    rvec_ScaledAdd( temp, C3dbo + C3dDelta + C4dbopi + C4dbopi2, dDeltap_self[j] );

    /* 1st, dBOpi */
    rvec_ScaledAdd( temp, -C1dbopi, BL.dln_BOp_pi[pj] );

    /* 1st, dBOpi2 */
    rvec_ScaledAdd( temp, -C1dbopi2, BL.dln_BOp_pi2[pj] );

#if defined(GPU_KERNEL_ATOMIC)
    atomic_rvecAdd( f[j], temp );
#else
    atomic_rvecAdd( BL.f_bo[pj], temp );
#endif

#undef BL
}


/* Initialize arrays */
GPU_GLOBAL void k_bond_order_part1( reax_atom const * const my_atoms, 
        single_body_parameters const * const sbp, real * const total_bond_order,
        real * const Deltap, real * const Deltap_boc,
        int N )
{
    int i, type_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    /* Calculate Deltaprime, Deltaprime_boc values */
    type_i = my_atoms[i].type;
    Deltap[i] = total_bond_order[i] - sbp[type_i].valency;
    Deltap_boc[i] = total_bond_order[i] - sbp[type_i].valency_val;
    total_bond_order[i] = 0.0; 
}


/* Main BO calculations */
GPU_GLOBAL void k_bond_order_part2( reax_atom const * const my_atoms, real const * const gp_l,
        single_body_parameters const * const sbp, two_body_parameters const * const tbp,
        real * const total_bond_order, real const * const Deltap, real const * const Deltap_boc,
        reax_list bond_list, int num_atom_types, int N )
{
    int i, j, pj, type_i, type_j;
    int start_i, end_i, tbp_ij;
    real f1, f2, f3, f4, f5, f4f5, exp_f4, exp_f5;
    real exp_p1i, exp_p2i, exp_p1j, exp_p2j;
    real temp, u1_ij, u1_ji, Cf1A_ij, Cf1B_ij, Cf1_ij, Cf1_ji;
    real Cf45_ij, Cf45_ji;
    real A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji;
    real total_bond_order_i;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    const real p_boc1 = gp_l[0];
    const real p_boc2 = gp_l[1];

    /* Corrected Bond Order calculations */
    type_i = my_atoms[i].type;
    const real val_i = sbp[type_i].valency;
    const real Deltap_i = Deltap[i];
    const real Deltap_boc_i = Deltap_boc[i];
    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );
    total_bond_order_i = 0.0;

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = BL.nbr[pj];
        type_j = my_atoms[j].type;

        //if ( i < j || bond_mark[j] > 3 )
        if ( i < j )
        {
            tbp_ij = index_tbp(type_i, type_j, num_atom_types);

            if ( tbp[tbp_ij].ovc < 0.001 && tbp[tbp_ij].v13cor < 0.001 )
            {
                /* There is no correction to bond orders nor to derivatives of
                 * bond order prime! So we leave bond orders unchanged and
                 * set derivative of bond order coefficients s.t.
                 * dBO = dBOp & dBOxx = dBOxxp in Add_dBO_to_Forces */
                BL.C1dbo[pj] = 1.0;
                BL.C2dbo[pj] = 0.0;
                BL.C3dbo[pj] = 0.0;

                BL.C1dbopi[pj] = 1.0;
                BL.C2dbopi[pj] = 0.0;
                BL.C3dbopi[pj] = 0.0;
                BL.C4dbopi[pj] = 0.0;

                BL.C1dbopi2[pj] = 1.0;
                BL.C2dbopi2[pj] = 0.0;
                BL.C3dbopi2[pj] = 0.0;
                BL.C4dbopi2[pj] = 0.0;
            }
            else
            {
                const real val_j = sbp[type_j].valency;

                /* on page 1 */
                if ( tbp[tbp_ij].ovc >= 0.001 )
                {
                    /* Correction for overcoordination */
                    exp_p1i = EXP( -p_boc1 * Deltap_i );
                    exp_p2i = EXP( -p_boc2 * Deltap_i );
                    exp_p1j = EXP( -p_boc1 * Deltap[j] );
                    exp_p2j = EXP( -p_boc2 * Deltap[j] );

                    f2 = exp_p1i + exp_p1j;
                    f3 = -1.0 / p_boc2 * LOG( 0.5 * ( exp_p2i  + exp_p2j ) );
                    f1 = 0.5 * ( ( val_i + f2 ) / ( val_i + f2 + f3 )
                            + ( val_j + f2 ) / ( val_j + f2 + f3 ) );

                    /* Now come the derivates */
                    /* Bond Order pages 5-7, derivative of f1 */
                    temp = f2 + f3;
                    u1_ij = val_i + temp;
                    u1_ji = val_j + temp;
                    Cf1A_ij = 0.5 * f3 * (1.0 / SQR( u1_ij ) + 1.0 / SQR( u1_ji ));
                    Cf1B_ij = -0.5 * (( u1_ij - f3 ) / SQR( u1_ij )
                            + ( u1_ji - f3 ) / SQR( u1_ji ));

                    //Cf1_ij = -Cf1A_ij * p_boc1 * exp_p1i +
                    //          Cf1B_ij * exp_p2i / ( exp_p2i + exp_p2j );
                    Cf1_ij = 0.50 * ( -p_boc1 * exp_p1i / u1_ij -
                            ((val_i + f2) / SQR(u1_ij)) * ( -p_boc1 * exp_p1i +
                            exp_p2i / ( exp_p2i + exp_p2j ) ) + -p_boc1 * exp_p1i / u1_ji -
                            ((val_j + f2) / SQR(u1_ji)) * ( -p_boc1 * exp_p1i +
                            exp_p2i / ( exp_p2i + exp_p2j ) ));

                    Cf1_ji = -Cf1A_ij * p_boc1 * exp_p1j
                        + Cf1B_ij * exp_p2j / ( exp_p2i + exp_p2j );
                }
                else
                {
                    /* No overcoordination correction! */
                    f1 = 1.0;
                    Cf1_ij = 0.0;
                    Cf1_ji = 0.0;
                }

                if ( tbp[tbp_ij].v13cor >= 0.001 )
                {
                    /* Correction for 1-3 bond orders */
                    exp_f4 = EXP( -tbp[tbp_ij].p_boc3 * (tbp[tbp_ij].p_boc4 * SQR( BL.BO[pj] ) - Deltap_boc_i)
                            + tbp[tbp_ij].p_boc5 );
                    exp_f5 = EXP( -tbp[tbp_ij].p_boc3 * (tbp[tbp_ij].p_boc4 * SQR( BL.BO[pj] ) - Deltap_boc[j])
                            + tbp[tbp_ij].p_boc5 );

                    f4 = 1.0 / (1.0 + exp_f4);
                    f5 = 1.0 / (1.0 + exp_f5);
                    f4f5 = f4 * f5;

                    /* Bond Order pages 8-9, derivative of f4 and f5 */
//                    temp = tbp[tbp_ij].p_boc5
//                        - tbp[tbp_ij].p_boc3 * tbp[tbp_ij].p_boc4 * SQR( BL.BO[pj] );
//                    u_ij = temp + tbp[tbp_ij].p_boc3 * Deltap_boc_i;
//                    u_ji = temp + tbp[tbp_ij].p_boc3 * Deltap_boc[j];
//                    Cf45_ij = Cf45( u_ij, u_ji ) / f4f5;
//                    Cf45_ji = Cf45( u_ji, u_ij ) / f4f5;
                    Cf45_ij = -f4 * exp_f4;
                    Cf45_ji = -f5 * exp_f5;
                }
                else
                {
                    f4 = 1.0;
                    f5 = 1.0;
                    f4f5 = 1.0;
                    Cf45_ij = 0.0;
                    Cf45_ji = 0.0;
                }

                /* Bond Order page 10, derivative of total bond order */
                A0_ij = f1 * f4f5;
                A1_ij = -2.0 * tbp[tbp_ij].p_boc3 * tbp[tbp_ij].p_boc4 * BL.BO[pj]
                    * (Cf45_ij + Cf45_ji);
                A2_ij = Cf1_ij / f1 + tbp[tbp_ij].p_boc3 * Cf45_ij;
                A2_ji = Cf1_ji / f1 + tbp[tbp_ij].p_boc3 * Cf45_ji;
                A3_ij = A2_ij + Cf1_ij / f1;
                A3_ji = A2_ji + Cf1_ji / f1;

                /* find corrected bond order values and their deriv coefs */
                BL.BO[pj] = BL.BO[pj] * A0_ij;
                BL.BO_pi[pj] = BL.BO_pi[pj] * A0_ij * f1;
                BL.BO_pi2[pj] = BL.BO_pi2[pj] * A0_ij * f1;
                BL.BO_s[pj] = BL.BO[pj] - (BL.BO_pi[pj] + BL.BO_pi2[pj]);

                BL.C1dbo[pj] = A0_ij + BL.BO[pj] * A1_ij;
                BL.C2dbo[pj] = BL.BO[pj] * A2_ij;
                BL.C3dbo[pj] = BL.BO[pj] * A2_ji;

                BL.C1dbopi[pj] = f1 * f1 * f4 * f5;
                BL.C2dbopi[pj] = BL.BO_pi[pj] * A1_ij;
                BL.C3dbopi[pj] = BL.BO_pi[pj] * A3_ij;
                BL.C4dbopi[pj] = BL.BO_pi[pj] * A3_ji;

                BL.C1dbopi2[pj] = f1 * f1 * f4 * f5;
                BL.C2dbopi2[pj] = BL.BO_pi2[pj] * A1_ij;
                BL.C3dbopi2[pj] = BL.BO_pi2[pj] * A3_ij;
                BL.C4dbopi2[pj] = BL.BO_pi2[pj] * A3_ji;
            }

            /* neglect weak bonds */
            if ( BL.BO[pj] < 1.0e-10 )
            {
                BL.BO[pj] = 0.0;
            }
            if ( BL.BO_s[pj] < 1.0e-10 )
            {
                BL.BO_s[pj] = 0.0;
            }
            if ( BL.BO_pi[pj] < 1.0e-10 )
            {
                BL.BO_pi[pj] = 0.0;
            }
            if ( BL.BO_pi2[pj] < 1.0e-10 )
            {
                BL.BO_pi2[pj] = 0.0;
            }

            /* now keeps total_BO */
            total_bond_order_i += BL.BO[pj];
        }

        /* NOTE: handle sym_index later in k_bond_order_part3 */
    }

    total_bond_order[i] += total_bond_order_i;

#undef BL
}


/* Main BO calculations */
GPU_GLOBAL void k_bond_order_part2_opt( reax_atom const * const my_atoms, real const * const gp_l, 
        single_body_parameters const * const sbp, two_body_parameters const * const tbp, 
        real * const total_bond_order, real const * const Deltap, real const * const Deltap_boc,
        reax_list bond_list, int num_atom_types, int N )
{
    extern __shared__ hipcub::WarpReduce<double>::TempStorage temp2[];
    int i, j, pj, type_i, type_j, thread_id, warp_id, lane_id, itr;
    int start_i, end_i, tbp_ij;
    real f1, f2, f3, f4, f5, f4f5, exp_f4, exp_f5;
    real exp_p1i, exp_p2i, exp_p1j, exp_p2j;
    real temp, u1_ij, u1_ji, Cf1A_ij, Cf1B_ij, Cf1_ij, Cf1_ji;
    real Cf45_ij, Cf45_ji;
    real A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji;
    real total_bond_order_i;
#define BL (bond_list.bond_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    i = thread_id / warpSize;

    if ( i >= N )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize;

    const real p_boc1 = gp_l[0];
    const real p_boc2 = gp_l[1];

    /* Corrected Bond Order calculations */
    type_i = my_atoms[i].type;
    const real val_i = sbp[type_i].valency;
    const real Deltap_i = Deltap[i];
    const real Deltap_boc_i = Deltap_boc[i];
    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );
    total_bond_order_i = 0.0;

    for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
    {
        if ( pj < end_i )
        {
            j = BL.nbr[pj];
            type_j = my_atoms[j].type;

            //if ( i < j || bond_mark[j] > 3 )
            if ( i < j )
            {
                tbp_ij = index_tbp(type_i, type_j, num_atom_types);

                if ( tbp[tbp_ij].ovc < 0.001 && tbp[tbp_ij].v13cor < 0.001 )
                {
                    /* There is no correction to bond orders nor to derivatives of
                     * bond order prime! So we leave bond orders unchanged and
                     * set derivative of bond order coefficients s.t.
                     * dBO = dBOp & dBOxx = dBOxxp in Add_dBO_to_Forces */
                    BL.C1dbo[pj] = 1.0;
                    BL.C2dbo[pj] = 0.0;
                    BL.C3dbo[pj] = 0.0;

                    BL.C1dbopi[pj] = 1.0;
                    BL.C2dbopi[pj] = 0.0;
                    BL.C3dbopi[pj] = 0.0;
                    BL.C4dbopi[pj] = 0.0;

                    BL.C1dbopi2[pj] = 1.0;
                    BL.C2dbopi2[pj] = 0.0;
                    BL.C3dbopi2[pj] = 0.0;
                    BL.C4dbopi2[pj] = 0.0;
                }
                else
                {
                    const real val_j = sbp[type_j].valency;

                    /* on page 1 */
                    if ( tbp[tbp_ij].ovc >= 0.001 )
                    {
                        /* Correction for overcoordination */
                        exp_p1i = EXP( -p_boc1 * Deltap_i );
                        exp_p2i = EXP( -p_boc2 * Deltap_i );
                        exp_p1j = EXP( -p_boc1 * Deltap[j] );
                        exp_p2j = EXP( -p_boc2 * Deltap[j] );

                        f2 = exp_p1i + exp_p1j;
                        f3 = -1.0 / p_boc2 * LOG( 0.5 * ( exp_p2i  + exp_p2j ) );
                        f1 = 0.5 * ( ( val_i + f2 ) / ( val_i + f2 + f3 )
                                + ( val_j + f2 ) / ( val_j + f2 + f3 ) );

                        /* Now come the derivates */
                        /* Bond Order pages 5-7, derivative of f1 */
                        temp = f2 + f3;
                        u1_ij = val_i + temp;
                        u1_ji = val_j + temp;
                        Cf1A_ij = 0.5 * f3 * (1.0 / SQR( u1_ij ) + 1.0 / SQR( u1_ji ));
                        Cf1B_ij = -0.5 * (( u1_ij - f3 ) / SQR( u1_ij )
                                + ( u1_ji - f3 ) / SQR( u1_ji ));

                        //Cf1_ij = -Cf1A_ij * p_boc1 * exp_p1i +
                        //          Cf1B_ij * exp_p2i / ( exp_p2i + exp_p2j );
                        Cf1_ij = 0.50 * ( -p_boc1 * exp_p1i / u1_ij -
                                ((val_i + f2) / SQR(u1_ij)) * ( -p_boc1 * exp_p1i +
                                exp_p2i / ( exp_p2i + exp_p2j ) ) + -p_boc1 * exp_p1i / u1_ji -
                                ((val_j + f2) / SQR(u1_ji)) * ( -p_boc1 * exp_p1i +
                                exp_p2i / ( exp_p2i + exp_p2j ) ));

                        Cf1_ji = -Cf1A_ij * p_boc1 * exp_p1j
                            + Cf1B_ij * exp_p2j / ( exp_p2i + exp_p2j );
                    }
                    else
                    {
                        /* No overcoordination correction! */
                        f1 = 1.0;
                        Cf1_ij = 0.0;
                        Cf1_ji = 0.0;
                    }

                    if ( tbp[tbp_ij].v13cor >= 0.001 )
                    {
                        /* Correction for 1-3 bond orders */
                        exp_f4 = EXP( -tbp[tbp_ij].p_boc3 * (tbp[tbp_ij].p_boc4 * SQR( BL.BO[pj] ) - Deltap_boc_i)
                                + tbp[tbp_ij].p_boc5 );
                        exp_f5 = EXP( -tbp[tbp_ij].p_boc3 * (tbp[tbp_ij].p_boc4 * SQR( BL.BO[pj] ) - Deltap_boc[j])
                                + tbp[tbp_ij].p_boc5 );

                        f4 = 1.0 / (1.0 + exp_f4);
                        f5 = 1.0 / (1.0 + exp_f5);
                        f4f5 = f4 * f5;

                        /* Bond Order pages 8-9, derivative of f4 and f5 */
//                        temp = tbp[tbp_ij].p_boc5
//                            - tbp[tbp_ij].p_boc3 * tbp[tbp_ij].p_boc4 * SQR( BL.BO[pj] );
//                        u_ij = temp + tbp[tbp_ij].p_boc3 * Deltap_boc_i;
//                        u_ji = temp + tbp[tbp_ij].p_boc3 * Deltap_boc[j];
//                        Cf45_ij = Cf45( u_ij, u_ji ) / f4f5;
//                        Cf45_ji = Cf45( u_ji, u_ij ) / f4f5;
                        Cf45_ij = -f4 * exp_f4;
                        Cf45_ji = -f5 * exp_f5;
                    }
                    else
                    {
                        f4 = 1.0;
                        f5 = 1.0;
                        f4f5 = 1.0;
                        Cf45_ij = 0.0;
                        Cf45_ji = 0.0;
                    }

                    /* Bond Order page 10, derivative of total bond order */
                    A0_ij = f1 * f4f5;
                    A1_ij = -2.0 * tbp[tbp_ij].p_boc3 * tbp[tbp_ij].p_boc4 * BL.BO[pj]
                        * (Cf45_ij + Cf45_ji);
                    A2_ij = Cf1_ij / f1 + tbp[tbp_ij].p_boc3 * Cf45_ij;
                    A2_ji = Cf1_ji / f1 + tbp[tbp_ij].p_boc3 * Cf45_ji;
                    A3_ij = A2_ij + Cf1_ij / f1;
                    A3_ji = A2_ji + Cf1_ji / f1;

                    /* find corrected bond order values and their deriv coefs */
                    BL.BO[pj] = BL.BO[pj] * A0_ij;
                    BL.BO_pi[pj] = BL.BO_pi[pj] * A0_ij * f1;
                    BL.BO_pi2[pj] = BL.BO_pi2[pj] * A0_ij * f1;
                    BL.BO_s[pj] = BL.BO[pj] - (BL.BO_pi[pj] + BL.BO_pi2[pj]);

                    BL.C1dbo[pj] = A0_ij + BL.BO[pj] * A1_ij;
                    BL.C2dbo[pj] = BL.BO[pj] * A2_ij;
                    BL.C3dbo[pj] = BL.BO[pj] * A2_ji;

                    BL.C1dbopi[pj] = f1 * f1 * f4 * f5;
                    BL.C2dbopi[pj] = BL.BO_pi[pj] * A1_ij;
                    BL.C3dbopi[pj] = BL.BO_pi[pj] * A3_ij;
                    BL.C4dbopi[pj] = BL.BO_pi[pj] * A3_ji;

                    BL.C1dbopi2[pj] = f1 * f1 * f4 * f5;
                    BL.C2dbopi2[pj] = BL.BO_pi2[pj] * A1_ij;
                    BL.C3dbopi2[pj] = BL.BO_pi2[pj] * A3_ij;
                    BL.C4dbopi2[pj] = BL.BO_pi2[pj] * A3_ji;
                }

                /* neglect weak bonds */
                if ( BL.BO[pj] < 1.0e-10 )
                {
                    BL.BO[pj] = 0.0;
                }
                if ( BL.BO_s[pj] < 1.0e-10 )
                {
                    BL.BO_s[pj] = 0.0;
                }
                if ( BL.BO_pi[pj] < 1.0e-10 )
                {
                    BL.BO_pi[pj] = 0.0;
                }
                if ( BL.BO_pi2[pj] < 1.0e-10 )
                {
                    BL.BO_pi2[pj] = 0.0;
                }

                /* now keeps total_BO */
                total_bond_order_i += BL.BO[pj];
            }

            /* NOTE: handle sym_index later in k_bond_order_part3 */
        }

        pj += warpSize;
    }

    total_bond_order_i = hipcub::WarpReduce<double>(temp2[warp_id]).Sum(total_bond_order_i);

    if ( lane_id == 0 )
    {
        total_bond_order[i] += total_bond_order_i;
    }

#undef BL
}


/* Compute sym_index */
GPU_GLOBAL void k_bond_order_part3( real * const total_bond_order, reax_list bond_list, int N )
{
    int i, j, pj, start_i, end_i, sym_index;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = BL.nbr[pj];

        //if ( i >= j && bond_mark[i] <= 3 )
        if ( i >= j )
        {
            /* We only need to update bond orders from bo_ji
             * everything else is set in uncorrected_bo calculations */
            sym_index = BL.sym_index[pj];

            BL.BO[pj] = BL.BO[sym_index];
            BL.BO_s[pj] = BL.BO_s[sym_index];
            BL.BO_pi[pj] = BL.BO_pi[sym_index];
            BL.BO_pi2[pj] = BL.BO_pi2[sym_index];

            /* now keeps total_BO */
            total_bond_order[i] += BL.BO[pj];
        }
    }

#undef BL
}


/* Calculate helper variables */
GPU_GLOBAL void k_bond_order_part4( reax_atom const * const my_atoms,
        real const * const gp_l, single_body_parameters const * const sbp,
        real const * const total_bond_order,
        real * const Delta, real * const Delta_lp, real * const Delta_lp_temp,
        real * const Delta_e, real * const Delta_boc, real * const dDelta_lp,
        real * const dDelta_lp_temp, real * const nlp, real * const nlp_temp,
        real * const Clp, real * const vlpex, int N )
{
    int i, type_i;
    real explp1;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    const real p_lp1 = gp_l[15];

    /* Calculate some helper variables that are  used at many places
     * throughout force calculations */
    type_i = my_atoms[i].type;

    Delta[i] = total_bond_order[i] - sbp[type_i].valency;
    Delta_e[i] = total_bond_order[i] - sbp[type_i].valency_e;
    Delta_boc[i] = total_bond_order[i] - sbp[type_i].valency_boc;

    vlpex[i] = Delta_e[i] - 2.0 * (int)(Delta_e[i] / 2.0);
    explp1 = EXP(-p_lp1 * SQR(2.0 + vlpex[i]));
    nlp[i] = explp1 - (int)(Delta_e[i] / 2.0);
    Delta_lp[i] = sbp[type_i].nlp_opt - nlp[i];
    Clp[i] = 2.0 * p_lp1 * explp1 * (2.0 + vlpex[i]);
    /* Adri uses different dDelta_lp values than the ones in notes... */
    dDelta_lp[i] = Clp[i];
//    dDelta_lp[i] = Clp[i] + (0.5 - Clp[i]) * ((FABS(Delta_e[i] / 2.0
//                    - (int) (Delta_e[i] / 2.0)) < 0.1) ? 1 : 0 );

    if ( sbp[type_i].mass > 21.0 )
    {
        nlp_temp[i] = 0.5 * (sbp[type_i].valency_e - sbp[type_i].valency);
        Delta_lp_temp[i] = sbp[type_i].nlp_opt - nlp_temp[i];
        dDelta_lp_temp[i] = 0.0;
    }
    else
    {
        nlp_temp[i] = nlp[i];
        Delta_lp_temp[i] = sbp[type_i].nlp_opt - nlp_temp[i];
        dDelta_lp_temp[i] = Clp[i];
    }
}


GPU_GLOBAL void k_total_forces_part1( rvec const * const dDeltap_self,
        real const * const CdDelta, rvec * const f, reax_list bond_list,
        int N )
{
    int i, pj;
    rvec f_i;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    rvec_MakeZero( f_i );

    for ( pj = Start_Index( i, &bond_list ); pj < End_Index( i, &bond_list ); ++pj )
    {
        if ( i < BL.nbr[pj] )
        {
            GPU_Add_dBond_to_Forces( i, pj, dDeltap_self, CdDelta, f, &bond_list, &f_i );
        }
    }

    atomic_rvecAdd( f[i], f_i );

#undef BL
}


GPU_GLOBAL void k_total_forces_part1_opt( rvec const * const dDeltap_self,
        real const * const CdDelta, rvec * const f, reax_list bond_list,
        int N )
{
    extern __shared__ hipcub::WarpReduce<double>::TempStorage temp1[];
    int i, pj, start_i, end_i, thread_id, warp_id, lane_id, itr;
    rvec f_i;
#define BL (bond_list.bond_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    i = thread_id / warpSize;

    if ( i >= N )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize;
    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );
    rvec_MakeZero( f_i );

    for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
    {
        if ( pj < end_i && i < BL.nbr[pj] )
        {
            GPU_Add_dBond_to_Forces( i, pj, dDeltap_self, CdDelta, f, &bond_list, &f_i );
        }

        pj += warpSize;
    }

    f_i[0] = hipcub::WarpReduce<double>(temp1[warp_id]).Sum(f_i[0]);
    f_i[1] = hipcub::WarpReduce<double>(temp1[warp_id]).Sum(f_i[1]);
    f_i[2] = hipcub::WarpReduce<double>(temp1[warp_id]).Sum(f_i[2]);

    if ( lane_id == 0 )
    {
        atomic_rvecAdd( f[i], f_i );
    }

#undef BL
}


GPU_GLOBAL void k_total_forces_virial_part1( rvec const * const dDeltap_self,
        real const * const CdDelta, rvec * const f, reax_list bond_list,
        rvec * const data_ext_press, int N )
{
    int i, pj;
    rvec f_i;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    rvec_MakeZero( f_i );

    for ( pj = Start_Index( i, &bond_list ); pj < End_Index( i, &bond_list ); ++pj )
    {
        if ( i < BL.nbr[pj] )
        {
            GPU_Add_dBond_to_Forces_NPT( i, pj, dDeltap_self, CdDelta,
                    f, &bond_list, &f_i, data_ext_press[i] );
        }
    }

    atomic_rvecAdd( f[i], f_i );

#undef BL
}


#if !defined(GPU_KERNEL_ATOMIC)
GPU_GLOBAL void k_total_forces_part1_2( reax_list bond_list, rvec * const f, int N )
{
    int i, pk;
    rvec f_i;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    rvec_MakeZero( f_i );

    for ( pk = Start_Index( i, &bond_list ); pk < End_Index( i, &bond_list ); ++pk )
    {
        rvec_Add( f_i, BL.f_bo[BL.sym_index[pk]] );
    }

    rvec_Add( f[i], f_i );

#undef BL
}
#endif


GPU_GLOBAL void k_total_forces_part2( reax_atom * const my_atoms, int n,
        rvec const * const f )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    rvec_Copy( my_atoms[i].f, f[i] );
}


#if !defined(GPU_STREAM_SINGLE_ACCUM)
GPU_GLOBAL void k_reduction_stream_part1( real const * const CdDelta_bonds,
        real const * const CdDelta_multi, real const * const CdDelta_tor,
        real * const CdDelta, rvec const * const f_hb,
#if defined(FUSED_VDW_COULOMB)
        rvec const * const f_vdw_clmb,
#else
        rvec const * const f_vdw, rvec const * const f_clmb,
#endif
        rvec const * const f_tor, rvec * const f, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    CdDelta[i] += CdDelta_bonds[i] + CdDelta_multi[i] + CdDelta_tor[i];

#if defined(FUSED_VDW_COULOMB)
    f[i][0] += f_hb[i][0] + f_vdw_clmb[i][0] + f_tor[i][0];
    f[i][1] += f_hb[i][1] + f_vdw_clmb[i][1] + f_tor[i][1];
    f[i][2] += f_hb[i][2] + f_vdw_clmb[i][2] + f_tor[i][2];
#else
    f[i][0] += f_hb[i][0] + f_vdw[i][0] + f_clmb[i][0] + f_tor[i][0];
    f[i][1] += f_hb[i][1] + f_vdw[i][1] + f_clmb[i][1] + f_tor[i][1];
    f[i][2] += f_hb[i][2] + f_vdw[i][2] + f_clmb[i][2] + f_tor[i][2];
#endif
}


GPU_GLOBAL void k_reduction_stream_part2( reax_list bond_list, int N )
{
    int i, pj;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    for ( pj = Start_Index( i, &bond_list ); pj < End_Index( i, &bond_list ); ++pj )
    {
        BL.Cdbo[pj] += BL.Cdbo_bonds[pj] + BL.Cdbo_multi[pj]
            + BL.Cdbo_hbonds[pj] + BL.Cdbo_tor[pj];
        BL.Cdbopi[pj] += BL.Cdbopi_bonds[pj] + BL.Cdbopi_multi[pj]
            + BL.Cdbopi_tor[pj];
        BL.Cdbopi2[pj] += BL.Cdbopi2_bonds[pj] + BL.Cdbopi2_multi[pj];
    }

#undef BL
}


GPU_GLOBAL void k_reduction_stream_part2_opt( reax_list bond_list, int N )
{
    int i, pj, start_i, end_i, thread_id, lane_id, itr;
#define BL (bond_list.bond_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    i = thread_id / warpSize;

    if ( i >= N )
    {
        return;
    }

    lane_id = thread_id % warpSize;
    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );

    for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
    {
        if ( pj < end_i )
        {
            BL.Cdbo[pj] += BL.Cdbo_bonds[pj] + BL.Cdbo_multi[pj]
                + BL.Cdbo_hbonds[pj] + BL.Cdbo_tor[pj];
            BL.Cdbopi[pj] += BL.Cdbopi_bonds[pj] + BL.Cdbopi_multi[pj]
                + BL.Cdbopi_tor[pj];
            BL.Cdbopi2[pj] += BL.Cdbopi2_bonds[pj] + BL.Cdbopi2_multi[pj];
        }

        pj += warpSize;
    }

#undef BL
}
#endif


void GPU_Compute_Bond_Orders( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list ** const lists,
        output_controls const * const out_control )
{
#if defined(LOG_PERFORMANCE)
    hipEventRecord( control->gpu_time_events[TE_BOND_ORDER_START], control->gpu_streams[0] );
#endif

    hipStreamWaitEvent( control->gpu_streams[0], control->gpu_stream_events[SE_INIT_BOND_DONE], 0 );

    k_bond_order_part1 <<< control->blocks_N, control->gpu_block_size,
                       0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          workspace->d_workspace->total_bond_order, workspace->d_workspace->Deltap,
          workspace->d_workspace->Deltap_boc, system->N );
    hipCheckError( );

//    k_bond_order_part2 <<< control->blocks_N, control->gpu_block_size,
//                       0, control->gpu_streams[0] >>>
//        ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp, 
//          system->reax_param.d_tbp, workspace->d_workspace->total_bond_order,
//          workspace->d_workspace->Deltap, workspace->d_workspace->Deltap_boc,
//          *(lists[BONDS]), system->reax_param.num_atom_types, system->N );
//    hipCheckError( );

    k_bond_order_part2_opt <<< control->blocks_warp_N, control->gpu_block_size,
                       sizeof(hipcub::WarpReduce<double>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                       control->gpu_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp, 
          system->reax_param.d_tbp, workspace->d_workspace->total_bond_order,
          workspace->d_workspace->Deltap, workspace->d_workspace->Deltap_boc,
          *(lists[BONDS]), system->reax_param.num_atom_types, system->N );
    hipCheckError( );

    k_bond_order_part3 <<< control->blocks_N, control->gpu_block_size,
                       0, control->gpu_streams[0] >>>
        ( workspace->d_workspace->total_bond_order, *(lists[BONDS]), system->N );
    hipCheckError( );

    k_bond_order_part4 <<< control->blocks_N, control->gpu_block_size,
                       0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.gp.d_l, system->reax_param.d_sbp,
          workspace->d_workspace->total_bond_order, workspace->d_workspace->Delta,
          workspace->d_workspace->Delta_lp, workspace->d_workspace->Delta_lp_temp,
          workspace->d_workspace->Delta_e, workspace->d_workspace->Delta_boc,
          workspace->d_workspace->dDelta_lp, workspace->d_workspace->dDelta_lp_temp,
          workspace->d_workspace->nlp, workspace->d_workspace->nlp_temp,
          workspace->d_workspace->Clp, workspace->d_workspace->vlpex,
         system->N );
    hipCheckError( );

    hipEventRecord( control->gpu_stream_events[SE_BOND_ORDER_DONE], control->gpu_streams[0] );

#if defined(LOG_PERFORMANCE)
    hipEventRecord( control->gpu_time_events[TE_BOND_ORDER_STOP], control->gpu_streams[0] );
#endif
}


void GPU_Total_Forces_Part1( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list ** const lists )
{
    rvec *spad_rvec;


#if !defined(GPU_STREAM_SINGLE_ACCUM)
    k_reduction_stream_part1 <<< control->blocks_N, control->gpu_block_size,
                         0, control->gpu_streams[0] >>>
        ( workspace->d_workspace->CdDelta_bonds, workspace->d_workspace->CdDelta_multi,
          workspace->d_workspace->CdDelta_tor, workspace->d_workspace->CdDelta,
          workspace->d_workspace->f_hb,
#if defined(FUSED_VDW_COULOMB)
          workspace->d_workspace->f_vdw_clmb,
#else
          workspace->d_workspace->f_vdw, workspace->d_workspace->f_clmb,
#endif
          workspace->d_workspace->f_tor, workspace->d_workspace->f, system->N );
    hipCheckError( );

//    k_reduction_stream_part2 <<< control->blocks_N, control->gpu_block_size,
//                         0, control->gpu_streams[0] >>>
//        ( *(lists[BONDS]), system->N );
//    hipCheckError( );

    k_reduction_stream_part2_opt <<< control->blocks_warp_N, control->gpu_block_size,
                         0, control->gpu_streams[0] >>>
        ( *(lists[BONDS]), system->N );
    hipCheckError( );
#endif

    if ( control->virial == 0 )
    {
//        k_total_forces_part1 <<< control->blocks_N, control->gpu_block_size,
//                             0, control->gpu_streams[0] >>>
//            ( workspace->d_workspace->dDeltap_self, workspace->d_workspace->CdDelta,
//              *workspace->d_workspace->f, *(lists[BONDS]), system->N );
//        hipCheckError( );

        k_total_forces_part1_opt <<< control->blocks_warp_N, control->gpu_block_size,
                                 sizeof(hipcub::WarpReduce<double>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                                 control->gpu_streams[0] >>>
            ( workspace->d_workspace->dDeltap_self, workspace->d_workspace->CdDelta,
              workspace->d_workspace->f, *(lists[BONDS]), system->N );
        hipCheckError( );
    }
    else
    {
        sHipCheckMalloc( &workspace->d_workspace->scratch[0],
                &workspace->d_workspace->scratch_size[0],
                sizeof(rvec) * 2 * system->N, __FILE__, __LINE__ );
        spad_rvec = (rvec *) workspace->d_workspace->scratch[0];
        sHipMemsetAsync( spad_rvec, 0, sizeof(rvec) * 2 * system->N,
                control->gpu_streams[0], __FILE__, __LINE__ );
        hipStreamSynchronize( control->gpu_streams[0] );

        k_total_forces_virial_part1 <<< control->blocks_N, control->gpu_block_size,
                                    0, control->gpu_streams[0] >>>
            ( workspace->d_workspace->dDeltap_self, workspace->d_workspace->CdDelta,
              workspace->d_workspace->f, *(lists[BONDS]), spad_rvec, system->N );
        hipCheckError( );

        GPU_Reduction_Sum( spad_rvec, &data->d_my_ext_press,
                system->N, 0, control->gpu_streams[0] );
    }

#if !defined(GPU_KERNEL_ATOMIC)
    /* post processing for the atomic forces */
    k_total_forces_part1_2 <<< control->blocks_N, control->gpu_block_size,
                           0, control->gpu_streams[0] >>>
        ( *(lists[BONDS]), workspace->d_workspace->f, system->N );
    hipCheckError( ); 
#endif
}


void GPU_Total_Forces_Part2( reax_system * const system,
        control_params const * const control, storage * const workspace )
{
    k_total_forces_part2 <<< control->blocks_n, control->gpu_block_size,
                         0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, system->n, workspace->d_workspace->f );
    hipCheckError( ); 
}
