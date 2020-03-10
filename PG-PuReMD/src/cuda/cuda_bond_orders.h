
#ifndef __CUDA_BOND_ORDERS_H__
#define __CUDA_BOND_ORDERS_H__

#include "../reax_types.h"

#include "../vector.h"

extern "C" {

void Cuda_Total_Forces( reax_system *, control_params *,
        simulation_data *, storage *, reax_list ** );

void Cuda_Total_Forces_PURE( reax_system *, storage * );

}


/* Compute the bond order term between atoms i and j,
 * and if this term exceeds the cutoff bo_cut, then adds
 * BOTH atoms the bonds list (i.e., compute term once
 * and copy to avoid redundant computation) */
CUDA_DEVICE static inline int Cuda_BOp( reax_list bond_list, real bo_cut,
        int i, int btop_i, int j, ivec *rel_box,
        real d, rvec *dvec, int format, single_body_parameters *sbp_i,
        single_body_parameters *sbp_j, two_body_parameters *twbp,
        rvec *dDeltap_self, real *total_bond_order )
{
    int btop_j, ret;
    real r2, C12, C34, C56;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
    real BO, BO_s, BO_pi, BO_pi2;
    bond_data *ibond, *jbond;
    bond_order_data *bo_ij, *bo_ji;

    ret = FALSE;
    r2 = SQR( d );

    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0 )
    {
        C12 = twbp->p_bo1 * POW( d / twbp->r_s, twbp->p_bo2 );
        BO_s = (1.0 + bo_cut) * exp( C12 );
    }
    else
    {
        C12 = 0.0;
        BO_s = 0.0;
    }

    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0 )
    {
        C34 = twbp->p_bo3 * POW( d / twbp->r_p, twbp->p_bo4 );
        BO_pi = exp( C34 );
    }
    else
    {
        C34 = 0.0;
        BO_pi = 0.0;
    }

    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0 )
    {
        C56 = twbp->p_bo5 * POW( d / twbp->r_pp, twbp->p_bo6 );
        BO_pi2 = exp( C56 );
    }
    else
    {
        C56 = 0.0;
        BO_pi2 = 0.0;
    }

    /* Initially BO values are the uncorrected ones, page 1 */
    BO = BO_s + BO_pi + BO_pi2;

    if ( BO >= bo_cut )
    {
        /* Bond Order page2-3, derivative of total bond order prime */
        Cln_BOp_s = twbp->p_bo2 * C12 / r2;
        Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
        Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

        /****** bonds i-j and j-i ******/
        if ( i < j )
        {
            /****** bond i-j ONLY ******/
            ibond = &bond_list.bond_list[btop_i];
            ibond->nbr = j;
            ibond->d = d;

            rvec_Copy( ibond->dvec, *dvec );
            ivec_Copy( ibond->rel_box, *rel_box );

            //ibond->dbond_index = btop_i;
            //ibond->sym_index = btop_j;

            bo_ij = &ibond->bo_data;
            bo_ij->BO = BO;
            bo_ij->BO_s = BO_s;
            bo_ij->BO_pi = BO_pi;
            bo_ij->BO_pi2 = BO_pi2;

            /* Only dln_BOp_xx wrt. dr_i is stored here, note that
             * dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
            rvec_Scale( bo_ij->dln_BOp_s, -1.0 * bo_ij->BO_s * Cln_BOp_s, ibond->dvec );
            rvec_Scale( bo_ij->dln_BOp_pi, -1.0 * bo_ij->BO_pi * Cln_BOp_pi, ibond->dvec );
            rvec_Scale( bo_ij->dln_BOp_pi2, -1.0 * bo_ij->BO_pi2 * Cln_BOp_pi2, ibond->dvec );

            /* Only dBOp wrt. dr_i is stored here, note that
             * dBOp/dr_i = -dBOp/dr_j and all others are 0 */
            rvec_Scale( bo_ij->dBOp, -(bo_ij->BO_s * Cln_BOp_s
                        + bo_ij->BO_pi * Cln_BOp_pi
                        + bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );

            rvec_Add( dDeltap_self[i], bo_ij->dBOp );

            bo_ij->BO_s -= bo_cut;
            bo_ij->BO -= bo_cut;
            /* currently total_BOp */
            total_bond_order[i] += bo_ij->BO; 
            bo_ij->Cdbo = 0.0;
            bo_ij->Cdbopi = 0.0;
            bo_ij->Cdbopi2 = 0.0;

            /* CUDA-specific */
            ibond->ae_CdDelta = 0.0;
            ibond->va_CdDelta = 0.0;
            rvec_MakeZero( ibond->va_f );
            ibond->ta_CdDelta = 0.0;
            ibond->ta_Cdbo = 0.0;
            rvec_MakeZero( ibond->ta_f );
            rvec_MakeZero( ibond->hb_f );
            rvec_MakeZero( ibond->tf_f );
        }
        else
        {
            /****** bond j-i ONLY ******/
            //btop_j = End_Index( j, &bond_list );
            btop_j = btop_i;
            jbond = &bond_list.bond_list[btop_j];

            jbond->nbr = j;
            jbond->d = d;
            rvec_Scale( jbond->dvec, -1.0, *dvec );
            ivec_Scale( jbond->rel_box, -1.0, *rel_box );
            //jbond->dbond_index = btop_i;
            //jbond->sym_index = btop_i;

            //Set_End_Index( j, btop_j + 1, &bond_list );

            bo_ji = &jbond->bo_data;
            bo_ji->BO = BO;
            bo_ji->BO_s = BO_s;
            bo_ji->BO_pi = BO_pi;
            bo_ji->BO_pi2 = BO_pi2;

            /* Only dln_BOp_xx wrt. dr_i is stored here, note that
             * dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
            rvec_Scale( bo_ji->dln_BOp_s, BO_s * Cln_BOp_s, *dvec );
            rvec_Scale( bo_ji->dln_BOp_pi, BO_pi * Cln_BOp_pi, *dvec );
            rvec_Scale( bo_ji->dln_BOp_pi2, BO_pi2 * Cln_BOp_pi2, *dvec );

            /* Only dBOp wrt. dr_i is stored here, note that
             * dBOp/dr_i = -dBOp/dr_j and all others are 0 */
            rvec_Scale( bo_ji->dBOp, BO_s * Cln_BOp_s
                        + BO_pi * Cln_BOp_pi
                        + BO_pi2 * Cln_BOp_pi2, *dvec );

            rvec_Add( dDeltap_self[i], bo_ji->dBOp );

            bo_ji->BO_s -= bo_cut;
            bo_ji->BO -= bo_cut;

            /* currently total_BOp */
            total_bond_order[i] += bo_ji->BO;
            bo_ji->Cdbo = 0.0;
            bo_ji->Cdbopi = 0.0;
            bo_ji->Cdbopi2 = 0.0;

            /* CUDA-specific */
            jbond->ae_CdDelta = 0.0;
            jbond->va_CdDelta = 0.0;
            rvec_MakeZero( jbond->va_f );
            jbond->ta_CdDelta = 0.0;
            jbond->ta_Cdbo = 0.0;
            rvec_MakeZero( jbond->ta_f );
            rvec_MakeZero( jbond->hb_f );
            rvec_MakeZero( jbond->tf_f );
        }

        ret = TRUE;
    }

    return ret;
}


CUDA_GLOBAL void Cuda_BO_Part1( reax_atom *,
        single_body_parameters *, storage , int );

CUDA_GLOBAL void Cuda_BO_Part2( reax_atom *, global_parameters,
        single_body_parameters *, two_body_parameters *,
        storage, reax_list, int, int );

CUDA_GLOBAL void Cuda_BO_Part3( storage, reax_list, int );

CUDA_GLOBAL void Cuda_BO_Part4( reax_atom *, global_parameters,
        single_body_parameters *, storage, int );

#endif
