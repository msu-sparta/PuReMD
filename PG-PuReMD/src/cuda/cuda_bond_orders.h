
#ifndef __CUDA_BOND_ORDERS_H__
#define __CUDA_BOND_ORDERS_H__

#include "../reax_types.h"

#include "cuda_helpers.h"

#include "../vector.h"


void Cuda_Compute_Bond_Orders( reax_system * const, control_params * const, 
        simulation_data * const, storage * const, reax_list ** const,
        output_controls * const );

void Cuda_Total_Forces_Part1( reax_system * const, control_params * const,
        simulation_data * const, storage *, reax_list ** const );

void Cuda_Total_Forces_Part2( reax_system * const, control_params * const,
        storage * const );


/* Compute the bond order term between atoms i and j,
 * and if this term exceeds the cutoff bo_cut, then adds
 * BOTH atoms the bonds list (i.e., compute term once
 * and copy to avoid redundant computation) */
CUDA_DEVICE static inline void Cuda_Compute_BOp( reax_list bond_list, real bo_cut,
        int i, int btop_i, int j, real C12, real C34, real C56, real BO_s,
        real BO_pi, real BO_pi2, real BO, ivec *rel_box, real d, rvec *dvec,
        int format, two_body_parameters *twbp, rvec *dDeltap_self,
        real *total_bond_order )
{
    real r2;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
    bond_data *ibond;
    bond_order_data *bo_ij;

    r2 = SQR( d );

    /****** bond i-j ONLY ******/
    ibond = &bond_list.bond_list[btop_i];
    ibond->nbr = j;
    ibond->d = d;

    rvec_Copy( ibond->dvec, *dvec );
    ivec_Copy( ibond->rel_box, *rel_box );

    bo_ij = &ibond->bo_data;
    bo_ij->BO = BO;
    bo_ij->BO_s = BO_s;
    bo_ij->BO_pi = BO_pi;
    bo_ij->BO_pi2 = BO_pi2;

    /* Bond Order page2-3, derivative of total bond order prime */
    Cln_BOp_s = twbp->p_bo2 * C12 / r2;
    Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
    Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

    /* Only dln_BOp_xx wrt. dr_i is stored here, note that
     * dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
    rvec_Scale( bo_ij->dln_BOp_s, -1.0 * bo_ij->BO_s * Cln_BOp_s, ibond->dvec );
    rvec_Scale( bo_ij->dln_BOp_pi, -1.0 * bo_ij->BO_pi * Cln_BOp_pi, ibond->dvec );
    rvec_Scale( bo_ij->dln_BOp_pi2, -1.0 * bo_ij->BO_pi2 * Cln_BOp_pi2, ibond->dvec );

    /* Only dBOp wrt. dr_i is stored here, note that
     * dBOp/dr_i = -dBOp/dr_j and all others are 0 */
    rvec_Scale( bo_ij->dBOp, -1.0 * (bo_ij->BO_s * Cln_BOp_s
                + bo_ij->BO_pi * Cln_BOp_pi
                + bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );

//    rvec_Add( dDeltap_self[i], bo_ij->dBOp );
    atomic_rvecAdd( dDeltap_self[i], bo_ij->dBOp );

    bo_ij->BO_s -= bo_cut;
    bo_ij->BO -= bo_cut;
    /* currently total_BOp */
//    total_bond_order[i] += bo_ij->BO; 
    atomicAdd( (double *) &total_bond_order[i], bo_ij->BO ); 
    bo_ij->Cdbo = 0.0;
    bo_ij->Cdbopi = 0.0;
    bo_ij->Cdbopi2 = 0.0;

#if !defined(CUDA_ACCUM_ATOMIC)
    ibond->ae_CdDelta = 0.0;
    ibond->va_CdDelta = 0.0;
    rvec_MakeZero( ibond->va_f );
    ibond->ta_CdDelta = 0.0;
    ibond->ta_Cdbo = 0.0;
    rvec_MakeZero( ibond->ta_f );
    rvec_MakeZero( ibond->hb_f );
    rvec_MakeZero( ibond->tf_f );
#endif
}


#endif
