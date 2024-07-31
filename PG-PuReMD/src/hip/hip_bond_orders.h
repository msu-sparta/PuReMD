
#ifndef __HIP_BOND_ORDERS_H__
#define __HIP_BOND_ORDERS_H__

#include "../reax_types.h"

#include "hip_helpers.h"

#include "../vector.h"


void Hip_Compute_Bond_Orders( reax_system const * const, control_params const * const, 
        simulation_data * const, storage * const, reax_list ** const,
        output_controls const * const );

void Hip_Total_Forces_Part1( reax_system const * const, control_params const * const,
        simulation_data * const, storage *, reax_list ** const );

void Hip_Total_Forces_Part2( reax_system * const, control_params const * const,
        storage * const );


/* Compute the bond order term between atoms i and j,
 * and if this term exceeds the cutoff bo_cut, then add
 * BOTH atoms to the bonds list (i.e., compute term once
 * and copy to avoid redundant computation) */
GPU_DEVICE static inline void Hip_Compute_BOp( reax_list bond_list, real bo_cut,
        int i, int btop_i, int j, real C12, real C34, real C56, real BO_s,
        real BO_pi, real BO_pi2, real BO,
        ivec const * const rel_box, real d, rvec const * const dvec,
        int format, two_body_parameters const * const twbp, rvec dDeltap_self_i,
        real * const total_bond_order_i )
{
    real r2;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
#define BL (bond_list.bond_list_gpu)

    r2 = SQR( d );

    /****** bond i-j ONLY ******/
    BL.nbr[btop_i] = j;
    BL.d[btop_i] = d;

    rvec_Copy( BL.dvec[btop_i], *dvec );
    ivec_Copy( BL.rel_box[btop_i], *rel_box );

    BL.BO[btop_i] = BO;
    BL.BO_s[btop_i] = BO_s;
    BL.BO_pi[btop_i] = BO_pi;
    BL.BO_pi2[btop_i] = BO_pi2;

    /* Bond Order page2-3, derivative of total bond order prime */
    Cln_BOp_s = twbp->p_bo2 * C12 / r2;
    Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
    Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

    /* Only dln_BOp_xx wrt. dr_i is stored here, note that
     * dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
    rvec_Scale( BL.dln_BOp_s[btop_i], -1.0 * BL.BO_s[btop_i] * Cln_BOp_s, BL.dvec[btop_i] );
    rvec_Scale( BL.dln_BOp_pi[btop_i], -1.0 * BL.BO_pi[btop_i] * Cln_BOp_pi, BL.dvec[btop_i] );
    rvec_Scale( BL.dln_BOp_pi2[btop_i], -1.0 * BL.BO_pi2[btop_i] * Cln_BOp_pi2, BL.dvec[btop_i] );

    /* Only dBOp wrt. dr_i is stored here, note that
     * dBOp/dr_i = -dBOp/dr_j and all others are 0 */
    rvec_Scale( BL.dBOp[btop_i], -1.0 * (BL.BO_s[btop_i] * Cln_BOp_s
                + BL.BO_pi[btop_i] * Cln_BOp_pi
                + BL.BO_pi2[btop_i] * Cln_BOp_pi2), BL.dvec[btop_i] );

    rvec_Add( dDeltap_self_i, BL.dBOp[btop_i] );

    BL.BO_s[btop_i] -= bo_cut;
    BL.BO[btop_i] -= bo_cut;
    /* currently total_BOp */
    *total_bond_order_i += BL.BO[btop_i]; 
    BL.Cdbo[btop_i] = 0.0;
    BL.Cdbopi[btop_i] = 0.0;
    BL.Cdbopi2[btop_i] = 0.0;
#if !defined(GPU_KERNEL_ATOMIC)
    BL.CdDelta_multi[btop_i] = 0.0;
    BL.CdDelta_val[btop_i] = 0.0;
    BL.CdDelta_tor[btop_i] = 0.0;
    rvec_MakeZero( BL.f_hb[btop_i] );
    rvec_MakeZero( BL.f_val[btop_i] );
    rvec_MakeZero( BL.f_tor[btop_i] );
    rvec_MakeZero( BL.f_bo[btop_i] );
#endif
#if !defined(GPU_STREAM_SINGLE_ACCUM)
    BL.Cdbo_bonds[btop_i] = 0.0;
    BL.Cdbo_multi[btop_i] = 0.0;
    BL.Cdbo_hbonds[btop_i] = 0.0;
    BL.Cdbo_tor[btop_i] = 0.0;
    BL.Cdbopi_bonds[btop_i] = 0.0;
    BL.Cdbopi_multi[btop_i] = 0.0;
    BL.Cdbopi_tor[btop_i] = 0.0;
    BL.Cdbopi2_bonds[btop_i] = 0.0;
    BL.Cdbopi2_multi[btop_i] = 0.0;
#endif

#undef BL
}


#endif
