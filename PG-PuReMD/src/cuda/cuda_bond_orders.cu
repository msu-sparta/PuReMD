
#include "cuda_bond_orders.h"

#include "cuda_list.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"
#if defined(CUDA_ACCUM_ATOMIC)
#include "cuda_helpers.h"
#endif

#include "../index_utils.h"
#include "../bond_orders.h"


CUDA_DEVICE void Cuda_Add_dBond_to_Forces_NPT( int i, int pj,
        simulation_data *data, storage *workspace, reax_list *bond_list,
        rvec data_ext_press )
{
    bond_data *nbr_j, *nbr_k;
    bond_order_data *bo_ij, *bo_ji;
    dbond_coefficients coef;
    rvec temp, ext_press;
    ivec rel_box;
    int pk, k, j;

    nbr_j = &bond_list->bond_list[pj];
    j = nbr_j->nbr;
    if ( i < j )
    {
        bo_ij = &nbr_j->bo_data;
        bo_ji = &bond_list->bond_list[ nbr_j->sym_index ].bo_data;
    }
    else
    {
        bo_ij = &bond_list->bond_list[ nbr_j->sym_index ].bo_data;
        bo_ji = &nbr_j->bo_data;
    }

    coef.C1dbo = bo_ij->C1dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C2dbo = bo_ij->C2dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C3dbo = bo_ij->C3dbo * (bo_ij->Cdbo + bo_ji->Cdbo);

    coef.C1dbopi = bo_ij->C1dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C2dbopi = bo_ij->C2dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C3dbopi = bo_ij->C3dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C4dbopi = bo_ij->C4dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);

    coef.C1dbopi2 = bo_ij->C1dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C2dbopi2 = bo_ij->C2dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C3dbopi2 = bo_ij->C3dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C4dbopi2 = bo_ij->C4dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);

    coef.C1dDelta = bo_ij->C1dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);
    coef.C2dDelta = bo_ij->C2dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);
    coef.C3dDelta = bo_ij->C3dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);

    /************************************
    * forces related to atom i          *
    * first neighbors of atom i         *
    ************************************/
    if ( i < j )
    {
        for ( pk = Start_Index(i, bond_list); pk < End_Index(i, bond_list); ++pk )
        {
            nbr_k = &bond_list->bond_list[pk];
            k = nbr_k->nbr;

#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_MakeZero( nbr_k->tf_f );
#endif

            /* 2nd, dBO */
            rvec_Scale( temp, -coef.C2dbo, nbr_k->bo_data.dBOp );
            /* dDelta */
            rvec_ScaledAdd( temp, -coef.C2dDelta, nbr_k->bo_data.dBOp );
            /* 3rd, dBOpi */
            rvec_ScaledAdd( temp, -coef.C3dbopi, nbr_k->bo_data.dBOp );
            /* 3rd, dBOpi2 */
            rvec_ScaledAdd( temp, -coef.C3dbopi2, nbr_k->bo_data.dBOp );

            /* force */
#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_Add( nbr_k->tf_f, temp );
#else
            atomic_rvecAdd( workspace->f[k], temp );
#endif
            /* pressure */
            rvec_iMultiply( ext_press, nbr_k->rel_box, temp );
            rvec_Add( data_ext_press, ext_press );
        }

        /* then atom i itself */
        /* 1st, dBO */
        rvec_Scale( temp, coef.C1dbo, bo_ij->dBOp );
        /* 2nd, dBO */
        rvec_ScaledAdd( temp, coef.C2dbo, workspace->dDeltap_self[i] );

        /* 1st, dBO */
        rvec_ScaledAdd( temp, coef.C1dDelta, bo_ij->dBOp );
        /* 2nd, dBO */
        rvec_ScaledAdd( temp, coef.C2dDelta, workspace->dDeltap_self[i] );

        /* 1st, dBOpi */
        rvec_ScaledAdd( temp, coef.C1dbopi, bo_ij->dln_BOp_pi );
        /* 2nd, dBOpi */
        rvec_ScaledAdd( temp, coef.C2dbopi, bo_ij->dBOp );
        /* 3rd, dBOpi */
        rvec_ScaledAdd( temp, coef.C3dbopi, workspace->dDeltap_self[i] );

        /* 1st, dBO_pi2 */
        rvec_ScaledAdd( temp, coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
        /* 2nd, dBO_pi2 */
        rvec_ScaledAdd( temp, coef.C2dbopi2, bo_ij->dBOp );
        /* 3rd, dBO_pi2 */
        rvec_ScaledAdd( temp, coef.C3dbopi2, workspace->dDeltap_self[i] );

        /* force */
        rvec_Add( workspace->f[i], temp );
        /* ext pressure due to i is dropped, counting force on j will be enough */
    }
    else
    {
        /******************************************************
         * forces and pressure related to atom j               * 
         * first neighbors of atom j                           *
         ******************************************************/
        for ( pk = Start_Index(j, bond_list); pk < End_Index(j, bond_list); ++pk )
        {
            nbr_k = &bond_list->bond_list[pk];
            k = nbr_k->nbr;

#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_MakeZero( nbr_k->tf_f );
#endif

            /* 3rd, dBO */
            rvec_Scale( temp, -coef.C3dbo, nbr_k->bo_data.dBOp );
            /* dDelta */
            rvec_ScaledAdd( temp, -coef.C3dDelta, nbr_k->bo_data.dBOp );
            /* 4th, dBOpi */
            rvec_ScaledAdd( temp, -coef.C4dbopi, nbr_k->bo_data.dBOp );
            /* 4th, dBOpi2 */
            rvec_ScaledAdd( temp, -coef.C4dbopi2, nbr_k->bo_data.dBOp );

            /* force */
#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_Add( nbr_k->tf_f, temp );
#else
            atomic_rvecAdd( workspace->f[k], temp );
#endif
            /* pressure */
            if ( k != i )
            {
                ivec_Sum( rel_box, nbr_k->rel_box, nbr_j->rel_box ); //rel_box(k, i)
                rvec_iMultiply( ext_press, rel_box, temp );
                rvec_Add( data_ext_press, ext_press );
            }
        }

        /* then atom j itself */
        /* 1st, dBO */
        rvec_Scale( temp, -coef.C1dbo, bo_ij->dBOp );
        /* 2nd, dBO */
        rvec_ScaledAdd( temp, coef.C3dbo, workspace->dDeltap_self[j] );

        /* 1st, dBO */
        rvec_ScaledAdd( temp, -coef.C1dDelta, bo_ij->dBOp );
        /* 2nd, dBO */
        rvec_ScaledAdd( temp, coef.C3dDelta, workspace->dDeltap_self[j] );

        /* 1st, dBOpi */
        rvec_ScaledAdd( temp, -coef.C1dbopi, bo_ij->dln_BOp_pi );
        /* 2nd, dBOpi */
        rvec_ScaledAdd( temp, -coef.C2dbopi, bo_ij->dBOp );
        /* 3rd, dBOpi */
        rvec_ScaledAdd( temp, coef.C4dbopi, workspace->dDeltap_self[j] );

        /* 1st, dBOpi2 */
        rvec_ScaledAdd( temp, -coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
        /* 2nd, dBOpi2 */
        rvec_ScaledAdd( temp, -coef.C2dbopi2, bo_ij->dBOp );
        /* 3rd, dBOpi2 */
        rvec_ScaledAdd( temp, coef.C4dbopi2, workspace->dDeltap_self[j] );

        /* force */
#if !defined(CUDA_ACCUM_ATOMIC)
        rvec_Add( workspace->f[j], temp );
#else
        atomic_rvecAdd( workspace->f[j], temp );
#endif
        /* pressure */
        rvec_iMultiply( ext_press, nbr_j->rel_box, temp );
        rvec_Add( data->my_ext_press, ext_press );
    }
}


CUDA_DEVICE void Cuda_Add_dBond_to_Forces( int i, int pj,
        storage *workspace, reax_list *bond_list )
{
    bond_data *nbr_j, *nbr_k;
    bond_order_data *bo_ij, *bo_ji;
    dbond_coefficients coef;
    int pk, j;
    rvec temp;

    nbr_j = &bond_list->bond_list[pj];
    j = nbr_j->nbr;
    if ( i < j )
    {
        bo_ij = &nbr_j->bo_data;
        bo_ji = &bond_list->bond_list[ nbr_j->sym_index ].bo_data;
    }
    else
    {
        bo_ij = &bond_list->bond_list[ nbr_j->sym_index ].bo_data;
        bo_ji = &nbr_j->bo_data;
    }

    coef.C1dbo = bo_ij->C1dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C2dbo = bo_ij->C2dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C3dbo = bo_ij->C3dbo * (bo_ij->Cdbo + bo_ji->Cdbo);

    coef.C1dbopi = bo_ij->C1dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C2dbopi = bo_ij->C2dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C3dbopi = bo_ij->C3dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C4dbopi = bo_ij->C4dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);

    coef.C1dbopi2 = bo_ij->C1dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C2dbopi2 = bo_ij->C2dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C3dbopi2 = bo_ij->C3dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C4dbopi2 = bo_ij->C4dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);

    coef.C1dDelta = bo_ij->C1dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);
    coef.C2dDelta = bo_ij->C2dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);
    coef.C3dDelta = bo_ij->C3dbo * (workspace->CdDelta[i] + workspace->CdDelta[j]);

    if ( i < j )
    {
        for ( pk = Start_Index(i, bond_list); pk < End_Index(i, bond_list); ++pk )
        {
            nbr_k = &bond_list->bond_list[pk];

            /* 2nd, dBO */
            rvec_Scale( temp, -coef.C2dbo, nbr_k->bo_data.dBOp );
            /* dDelta */
            rvec_ScaledAdd( temp, -coef.C2dDelta, nbr_k->bo_data.dBOp );
            /* 3rd, dBOpi */
            rvec_ScaledAdd( temp, -coef.C3dbopi, nbr_k->bo_data.dBOp );
            /* 3rd, dBOpi2 */
            rvec_ScaledAdd( temp, -coef.C3dbopi2, nbr_k->bo_data.dBOp );

#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_Add( nbr_k->tf_f, temp );
#else
            atomic_rvecAdd( workspace->f[nbr_k->nbr], temp );
#endif
        }

        /* 1st, dBO */
        rvec_Scale( temp, coef.C1dbo, bo_ij->dBOp );
        /* 2nd, dBO */
        rvec_ScaledAdd( temp, coef.C2dbo, workspace->dDeltap_self[i] );

        /* 1st, dBO */
        rvec_ScaledAdd( temp, coef.C1dDelta, bo_ij->dBOp );
        /* 2nd, dBO */
        rvec_ScaledAdd( temp, coef.C2dDelta, workspace->dDeltap_self[i] );

        /* 1st, dBOpi */
        rvec_ScaledAdd( temp, coef.C1dbopi, bo_ij->dln_BOp_pi );
        /* 2nd, dBOpi */
        rvec_ScaledAdd( temp, coef.C2dbopi, bo_ij->dBOp );
        /* 3rd, dBOpi */
        rvec_ScaledAdd( temp, coef.C3dbopi, workspace->dDeltap_self[i] );

        /* 1st, dBO_pi2 */
        rvec_ScaledAdd( temp, coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
        /* 2nd, dBO_pi2 */
        rvec_ScaledAdd( temp, coef.C2dbopi2, bo_ij->dBOp );
        /* 3rd, dBO_pi2 */
        rvec_ScaledAdd( temp, coef.C3dbopi2, workspace->dDeltap_self[i] );

#if !defined(CUDA_ACCUM_ATOMIC)
        rvec_Add( workspace->f[i], temp );
#else
        atomic_rvecAdd( workspace->f[i], temp );
#endif
    }
    else
    {
        for ( pk = Start_Index(i, bond_list); pk < End_Index(i, bond_list); ++pk )
        {
            nbr_k = &bond_list->bond_list[pk];

            /* 3rd, dBO */
            rvec_Scale( temp, -coef.C3dbo, nbr_k->bo_data.dBOp );
            /* dDelta */
            rvec_ScaledAdd( temp, -coef.C3dDelta, nbr_k->bo_data.dBOp );
            /* 4th, dBOpi */
            rvec_ScaledAdd( temp, -coef.C4dbopi, nbr_k->bo_data.dBOp );
            /* 4th, dBOpi2 */
            rvec_ScaledAdd( temp, -coef.C4dbopi2, nbr_k->bo_data.dBOp );

#if !defined(CUDA_ACCUM_ATOMIC)
            rvec_Add( nbr_k->tf_f, temp );
#else
            atomic_rvecAdd( workspace->f[nbr_k->nbr], temp );
#endif
        }

        /* 1st, dBO */
        rvec_Scale( temp, -coef.C1dbo, bo_ij->dBOp );
        /* 2nd, dBO */
        rvec_ScaledAdd( temp, coef.C3dbo, workspace->dDeltap_self[i] );

        /* 1st, dBO */
        rvec_ScaledAdd( temp, -coef.C1dDelta, bo_ij->dBOp );
        /* 2nd, dBO */
        rvec_ScaledAdd( temp, coef.C3dDelta, workspace->dDeltap_self[i] );

        /* 1st, dBOpi */
        rvec_ScaledAdd( temp, -coef.C1dbopi, bo_ij->dln_BOp_pi );
        /* 2nd, dBOpi */
        rvec_ScaledAdd( temp, -coef.C2dbopi, bo_ij->dBOp );
        /* 3rd, dBOpi */
        rvec_ScaledAdd( temp, coef.C4dbopi, workspace->dDeltap_self[i] );

        /* 1st, dBOpi2 */
        rvec_ScaledAdd( temp, -coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
        /* 2nd, dBOpi2 */
        rvec_ScaledAdd( temp, -coef.C2dbopi2, bo_ij->dBOp );
        /* 3rd, dBOpi2 */
        rvec_ScaledAdd( temp, coef.C4dbopi2, workspace->dDeltap_self[i] );

#if !defined(CUDA_ACCUM_ATOMIC)
        rvec_Add( workspace->f[i], temp );
#else
        atomic_rvecAdd( workspace->f[i], temp );
#endif
    }
}


/* Initialize arrays */
CUDA_GLOBAL void k_bond_order_part1( reax_atom *my_atoms, 
        single_body_parameters *sbp, storage workspace, int N )
{
    int i, type_i;
    single_body_parameters *sbp_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    /* Calculate Deltaprime, Deltaprime_boc values */
    type_i = my_atoms[i].type;
    sbp_i = &sbp[type_i];
    workspace.Deltap[i] = workspace.total_bond_order[i] - sbp_i->valency;
    workspace.Deltap_boc[i] = workspace.total_bond_order[i]
        - sbp_i->valency_val;
    workspace.total_bond_order[i] = 0.0; 
}


/* Main BO calculations */
CUDA_GLOBAL void k_bond_order_part2( reax_atom *my_atoms, global_parameters gp, 
        single_body_parameters *sbp, two_body_parameters *tbp, 
        storage workspace, reax_list bond_list, int num_atom_types, int N )
{
    int i, j, pj, type_i, type_j;
    int start_i, end_i;
    real val_i, Deltap_i, Deltap_boc_i;
    real val_j, Deltap_j, Deltap_boc_j;
    real f1, f2, f3, f4, f5, f4f5, exp_f4, exp_f5;
    real exp_p1i, exp_p2i, exp_p1j, exp_p2j;
    real temp, u1_ij, u1_ji, Cf1A_ij, Cf1B_ij, Cf1_ij, Cf1_ji;
    real Cf45_ij, Cf45_ji;
    real A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji;
    real p_boc1, p_boc2;
    single_body_parameters *sbp_i;
    two_body_parameters *twbp;
    bond_order_data *bo_ij;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    p_boc1 = gp.l[0];
    p_boc2 = gp.l[1];

    /* Corrected Bond Order calculations */
    type_i = my_atoms[i].type;
    sbp_i = &sbp[type_i];
    val_i = sbp_i->valency;
    Deltap_i = workspace.Deltap[i];
    Deltap_boc_i = workspace.Deltap_boc[i];
    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = bond_list.bond_list[pj].nbr;
        type_j = my_atoms[j].type;
        bo_ij = &bond_list.bond_list[pj].bo_data;

        //if ( i < j || workspace.bond_mark[j] > 3 )
        if ( i < j )
        {
            twbp = &tbp[ index_tbp(type_i, type_j, num_atom_types) ];

            if ( twbp->ovc < 0.001 && twbp->v13cor < 0.001 )
            {
                /* There is no correction to bond orders nor to derivatives of
                 * bond order prime! So we leave bond orders unchanged and
                 * set derivative of bond order coefficients s.t.
                 * dBO = dBOp & dBOxx = dBOxxp in Add_dBO_to_Forces */
                bo_ij->C1dbo = 1.0;
                bo_ij->C2dbo = 0.0;
                bo_ij->C3dbo = 0.0;

                bo_ij->C1dbopi = 1.0;
                bo_ij->C2dbopi = 0.0;
                bo_ij->C3dbopi = 0.0;
                bo_ij->C4dbopi = 0.0;

                bo_ij->C1dbopi2 = 1.0;
                bo_ij->C2dbopi2 = 0.0;
                bo_ij->C3dbopi2 = 0.0;
                bo_ij->C4dbopi2 = 0.0;
            }
            else
            {
                val_j = sbp[type_j].valency;
                Deltap_j = workspace.Deltap[j];
                Deltap_boc_j = workspace.Deltap_boc[j];

                /* on page 1 */
                if ( twbp->ovc >= 0.001 )
                {
                    /* Correction for overcoordination */
                    exp_p1i = EXP( -p_boc1 * Deltap_i );
                    exp_p2i = EXP( -p_boc2 * Deltap_i );
                    exp_p1j = EXP( -p_boc1 * Deltap_j );
                    exp_p2j = EXP( -p_boc2 * Deltap_j );

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

                if ( twbp->v13cor >= 0.001 )
                {
                    /* Correction for 1-3 bond orders */
                    exp_f4 = EXP( -twbp->p_boc3 * (twbp->p_boc4 * SQR( bo_ij->BO ) - Deltap_boc_i)
                            + twbp->p_boc5 );
                    exp_f5 = EXP( -twbp->p_boc3 * (twbp->p_boc4 * SQR( bo_ij->BO ) - Deltap_boc_j)
                            + twbp->p_boc5 );

                    f4 = 1.0 / (1.0 + exp_f4);
                    f5 = 1.0 / (1.0 + exp_f5);
                    f4f5 = f4 * f5;

                    /* Bond Order pages 8-9, derivative of f4 and f5 */
//                    temp = twbp->p_boc5
//                        - twbp->p_boc3 * twbp->p_boc4 * SQR( bo_ij->BO );
//                    u_ij = temp + twbp->p_boc3 * Deltap_boc_i;
//                    u_ji = temp + twbp->p_boc3 * Deltap_boc_j;
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
                A1_ij = -2.0 * twbp->p_boc3 * twbp->p_boc4 * bo_ij->BO
                    * (Cf45_ij + Cf45_ji);
                A2_ij = Cf1_ij / f1 + twbp->p_boc3 * Cf45_ij;
                A2_ji = Cf1_ji / f1 + twbp->p_boc3 * Cf45_ji;
                A3_ij = A2_ij + Cf1_ij / f1;
                A3_ji = A2_ji + Cf1_ji / f1;

                /* find corrected bond order values and their deriv coefs */
                bo_ij->BO = bo_ij->BO * A0_ij;
                bo_ij->BO_pi = bo_ij->BO_pi * A0_ij * f1;
                bo_ij->BO_pi2 = bo_ij->BO_pi2 * A0_ij * f1;
                bo_ij->BO_s = bo_ij->BO - ( bo_ij->BO_pi + bo_ij->BO_pi2 );

                bo_ij->C1dbo = A0_ij + bo_ij->BO * A1_ij;
                bo_ij->C2dbo = bo_ij->BO * A2_ij;
                bo_ij->C3dbo = bo_ij->BO * A2_ji;

                bo_ij->C1dbopi = f1 * f1 * f4 * f5;
                bo_ij->C2dbopi = bo_ij->BO_pi * A1_ij;
                bo_ij->C3dbopi = bo_ij->BO_pi * A3_ij;
                bo_ij->C4dbopi = bo_ij->BO_pi * A3_ji;

                bo_ij->C1dbopi2 = f1 * f1 * f4 * f5;
                bo_ij->C2dbopi2 = bo_ij->BO_pi2 * A1_ij;
                bo_ij->C3dbopi2 = bo_ij->BO_pi2 * A3_ij;
                bo_ij->C4dbopi2 = bo_ij->BO_pi2 * A3_ji;
            }

            /* neglect weak bonds */
            if ( bo_ij->BO < 1.0e-10 )
            {
                bo_ij->BO = 0.0;
            }
            if ( bo_ij->BO_s < 1.0e-10 )
            {
                bo_ij->BO_s = 0.0;
            }
            if ( bo_ij->BO_pi < 1.0e-10 )
            {
                bo_ij->BO_pi = 0.0;
            }
            if ( bo_ij->BO_pi2 < 1.0e-10 )
            {
                bo_ij->BO_pi2 = 0.0;
            }

            /* now keeps total_BO */
            workspace.total_bond_order[i] += bo_ij->BO;

            /* NOTE: handle sym_index later in Cuda_Calculate_BO_Part3 */
        }
    }
}


/* Compute sym_index */
CUDA_GLOBAL void k_bond_order_part3( storage workspace, reax_list bond_list, int N )
{
    int i, j, pj;
    int start_i, end_i;
    int sym_index;
    bond_order_data *bo_ij, *bo_ji;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = bond_list.bond_list[pj].nbr;
        bo_ij = &bond_list.bond_list[pj].bo_data;

        //if ( i >= j || workspace.bond_mark[i] <= 3 )
        if ( i >= j )
        {
            /* We only need to update bond orders from bo_ji
             * everything else is set in uncorrected_bo calculations */
            sym_index = bond_list.bond_list[pj].sym_index;

            bo_ji = &bond_list.bond_list[ sym_index ].bo_data;
            bo_ij->BO = bo_ji->BO;
            bo_ij->BO_s = bo_ji->BO_s;
            bo_ij->BO_pi = bo_ji->BO_pi;
            bo_ij->BO_pi2 = bo_ji->BO_pi2;

            /* now keeps total_BO */
            workspace.total_bond_order[i] += bo_ij->BO;
        }
    }
}


/* Calculate helper variables */
CUDA_GLOBAL void k_bond_order_part4( reax_atom *my_atoms,
        global_parameters gp, single_body_parameters *sbp,
        storage workspace, int N )
{
    int i, type_i;
    real explp1, p_lp1;
    single_body_parameters *sbp_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    p_lp1 = gp.l[15];

    /* Calculate some helper variables that are  used at many places
     * throughout force calculations */
    type_i = my_atoms[i].type;
    sbp_i = &sbp[ type_i ];

    workspace.Delta[i] = workspace.total_bond_order[i] - sbp_i->valency;
    workspace.Delta_e[i] = workspace.total_bond_order[i] - sbp_i->valency_e;
    workspace.Delta_boc[i] = workspace.total_bond_order[i]
        - sbp_i->valency_boc;

    workspace.vlpex[i] = workspace.Delta_e[i]
        - 2.0 * (int)(workspace.Delta_e[i] / 2.0);
    explp1 = EXP(-p_lp1 * SQR(2.0 + workspace.vlpex[i]));
    workspace.nlp[i] = explp1 - (int)(workspace.Delta_e[i] / 2.0);
    workspace.Delta_lp[i] = sbp_i->nlp_opt - workspace.nlp[i];
    workspace.Clp[i] = 2.0 * p_lp1 * explp1 * (2.0 + workspace.vlpex[i]);
    /* Adri uses different dDelta_lp values than the ones in notes... */
    workspace.dDelta_lp[i] = workspace.Clp[i];
//    workspace.dDelta_lp[i] = workspace.Clp[i] + (0.5 - workspace.Clp[i])
//        * ((FABS(workspace.Delta_e[i] / 2.0
//                        - (int)(workspace.Delta_e[i] / 2.0)) < 0.1) ? 1 : 0 );

    if ( sbp_i->mass > 21.0 )
    {
        workspace.nlp_temp[i] = 0.5 * (sbp_i->valency_e - sbp_i->valency);
        workspace.Delta_lp_temp[i] = sbp_i->nlp_opt - workspace.nlp_temp[i];
        workspace.dDelta_lp_temp[i] = 0.0;
    }
    else
    {
        workspace.nlp_temp[i] = workspace.nlp[i];
        workspace.Delta_lp_temp[i] = sbp_i->nlp_opt - workspace.nlp_temp[i];
        workspace.dDelta_lp_temp[i] = workspace.Clp[i];
    }
}


CUDA_GLOBAL void k_total_forces_part1( storage workspace, reax_list bond_list, 
        control_params *control, simulation_data *data, rvec *data_ext_press,
        int N )
{
    int i, pj;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    if ( control->virial == 0 )
    {
        for ( pj = Start_Index( i, &bond_list ); pj < End_Index( i, &bond_list ); ++pj )
        {
            Cuda_Add_dBond_to_Forces( i, pj, &workspace, &bond_list );
        }
    }
    else 
    {
        for ( pj = Start_Index( i, &bond_list ); pj < End_Index( i, &bond_list ); ++pj )
        {
            Cuda_Add_dBond_to_Forces_NPT( i, pj, data, &workspace, &bond_list,
                    data_ext_press[i] );
        }
    }
}


#if !defined(CUDA_ACCUM_ATOMIC)
CUDA_GLOBAL void k_total_forces_part1_2( reax_atom *my_atoms, reax_list bond_list,
        storage workspace, int N )
{
    int i, pk;
    bond_data *nbr_k, *nbr_k_sym;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    for ( pk = Start_Index( i, &bond_list ); pk < End_Index( i, &bond_list ); ++pk )
    {
        nbr_k = &bond_list.bond_list[pk];
        nbr_k_sym = &bond_list.bond_list[nbr_k->sym_index];

        rvec_Add( workspace.f[i], nbr_k_sym->tf_f );
    }
}
#endif


CUDA_GLOBAL void k_total_forces_part2( reax_atom *my_atoms, int n, 
        storage workspace )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    rvec_Copy( my_atoms[i].f, workspace.f[i] );
}


void Cuda_Compute_Bond_Orders( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control )
{
    k_bond_order_part1 <<< control->blocks_n, control->block_size_n >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          *(workspace->d_workspace), system->N );
    cudaCheckError( );

    k_bond_order_part2 <<< control->blocks_n, control->block_size_n >>>
        ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, 
          system->reax_param.d_tbp, *(workspace->d_workspace), 
          *(lists[BONDS]),
          system->reax_param.num_atom_types, system->N );
    cudaCheckError( );

    k_bond_order_part3 <<< control->blocks_n, control->block_size_n >>>
        ( *(workspace->d_workspace), *(lists[BONDS]), system->N );
    cudaCheckError( );

    k_bond_order_part4 <<< control->blocks_n, control->block_size_n >>>
        ( system->d_my_atoms, system->reax_param.d_gp, system->reax_param.d_sbp, 
         *(workspace->d_workspace), system->N );
    cudaCheckError( );
}


void Cuda_Total_Forces_Part1( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, reax_list **lists )
{
    int blocks;
    rvec *spad_rvec;

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(rvec) * 2 * system->N,
            "Cuda_Total_Forces_Part1::workspace->scratch" );
    spad_rvec = (rvec *) workspace->scratch;
    cuda_memset( spad_rvec, 0, sizeof(rvec) * 2 * system->N,
            "total_forces:ext_press" );

    blocks = system->N / DEF_BLOCK_SIZE
        + ((system->N % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_total_forces_part1 <<< blocks, DEF_BLOCK_SIZE >>>
        ( *(workspace->d_workspace), *(lists[BONDS]), 
          (control_params *) control->d_control_params, 
          (simulation_data *)data->d_simulation_data, 
          spad_rvec, system->N );
    cudaCheckError( );

    if ( control->virial != 0 )
    {
        /* reduction for ext_press */
        k_reduction_rvec <<< blocks, DEF_BLOCK_SIZE, sizeof(rvec) * (DEF_BLOCK_SIZE / 32) >>> 
            ( spad_rvec, &spad_rvec[system->N], system->N );
        cudaCheckError( ); 

        k_reduction_rvec <<< 1, ((blocks + 31) / 32) * 32, sizeof(rvec) * ((blocks + 31) / 32) >>>
            ( &spad_rvec[system->N], &((simulation_data *)data->d_simulation_data)->my_ext_press, blocks );
        cudaCheckError( ); 
    }

#if !defined(CUDA_ACCUM_ATOMIC)
    /* post processing for the atomic forces */
    k_total_forces_part1_2  <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, *(lists[BONDS]), *(workspace->d_workspace), system->N );
    cudaCheckError( ); 
#endif
}


void Cuda_Total_Forces_Part2( reax_system *system, storage *workspace )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_total_forces_part2 <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->n, *(workspace->d_workspace) );
    cudaCheckError( ); 
}
