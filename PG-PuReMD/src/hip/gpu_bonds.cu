#include "hip/hip_runtime.h"
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

#include "gpu_bonds.h"

#include "gpu_list.h"
#include "gpu_helpers.h"
#if !defined(GPU_ATOMIC_EV)
  #include "gpu_reduction.h"
#endif
#include "gpu_utils.h"

#include "../index_utils.h"

#include <hipcub/warp/warp_reduce.hpp>


GPU_GLOBAL void k_bonds( reax_atom const * const my_atoms, real const * const gp_l, 
        single_body_parameters const * const sbp, two_body_parameters const * const tbp, 
        real const * const total_bond_order, real const * const Delta, real * const CdDelta,
        reax_list bond_list, int n, int num_atom_types, real * const e_bond_g )
{
    int i, j, pj, orig_id_i;
    int start_i, end_i;
    int type_i, type_j;
    real pow_BOs_be2, exp_be12, CEbo, e_bond_;
    real exphu, exphua1, exphub1, exphuov, hulpov;
    real decobdbo, decobdboua, decobdboub;
    real total_bond_order_i, Delta_i, CdDelta_i, Cdbo_ij;
#define BL (bond_list.bond_list_gpu)

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    const real gp3 = gp_l[3];
    const real gp4 = gp_l[4];
    const real gp7 = gp_l[7];
    const real gp10 = gp_l[10];
    e_bond_ = 0.0;
    orig_id_i = my_atoms[i].orig_id;
    total_bond_order_i = total_bond_order[i];
    Delta_i = Delta[i];
    CdDelta_i = 0.0;

    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );
    type_i = my_atoms[i].type;
    const int flag_i_C = GPU_strncmp( sbp[type_i].name, "C", sizeof(sbp[type_i].name) );
    const int flag_i_O = GPU_strncmp( sbp[type_i].name, "O", sizeof(sbp[type_i].name) );

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = BL.nbr[pj];

        if ( orig_id_i <= my_atoms[j].orig_id )
        {
            type_j = my_atoms[j].type;
            two_body_parameters const * const twbp = &tbp[ index_tbp(type_i, type_j, num_atom_types) ];

            pow_BOs_be2 = POW( BL.BO_s[pj], twbp->p_be2 );
            exp_be12 = EXP( twbp->p_be1 * ( 1.0 - pow_BOs_be2 ) );
            CEbo = -twbp->De_s * exp_be12
                * (1.0 - twbp->p_be1 * twbp->p_be2 * pow_BOs_be2);

            /* calculate bond energy */
            e_bond_ += -twbp->De_s * BL.BO_s[pj] * exp_be12
                - twbp->De_p * BL.BO_pi[pj]
                - twbp->De_pp * BL.BO_pi2[pj];

            /* calculate derivatives of bond orders */
            Cdbo_ij = CEbo;
#if defined(GPU_STREAM_SINGLE_ACCUM)
            atomicAdd( &BL.Cdbopi[pj], -1.0 * (CEbo + twbp->De_p) );
            atomicAdd( &BL.Cdbopi2[pj], -1.0 * (CEbo + twbp->De_pp) );
#else
            BL.Cdbopi_bonds[pj] = -1.0 * (CEbo + twbp->De_p);
            BL.Cdbopi2_bonds[pj] = -1.0 * (CEbo + twbp->De_pp);
#endif

            /* Stabilisation terminal triple bond */
            if ( BL.BO[pj] >= 1.00 )
            {
                if ( (flag_i_C == 0 && GPU_strncmp( sbp[type_j].name, "O", sizeof(sbp[type_j].name) ) == 0)
                        || (flag_i_O == 0 && GPU_strncmp( sbp[type_j].name, "C", sizeof(sbp[type_j].name) ) == 0) )
                {
                    //ba = SQR( BL.BO[pj] - 2.5 );
                    exphu = EXP( -gp7 * SQR(BL.BO[pj] - 2.5) );
                    //oboa = abo(j1) - boa;
                    //obob = abo(j2) - boa;
                    exphua1 = EXP(-gp3 * (total_bond_order_i - BL.BO[pj]));
                    exphub1 = EXP(-gp3 * (total_bond_order[j] - BL.BO[pj]));
                    //ovoab = abo(j1) - aval(it1) + abo(j2) - aval(it2);
                    exphuov = EXP(gp4 * (Delta_i + Delta[j]));
                    hulpov = 1.0 / (1.0 + 25.0 * exphuov);

                    e_bond_ += gp10 * exphu * hulpov * (exphua1 + exphub1);

                    decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1)
                        * ( gp3 - 2.0 * gp7 * (BL.BO[pj] - 2.5) );
                    decobdboua = -gp10 * exphu * hulpov
                        * (gp3 * exphua1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));
                    decobdboub = -gp10 * exphu * hulpov
                        * (gp3 * exphub1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));

                    Cdbo_ij += decobdbo;
                    CdDelta_i += decobdboua;
                    atomicAdd( &CdDelta[j], decobdboub );
                }
            }

#if defined(GPU_STREAM_SINGLE_ACCUM)
            atomicAdd( &BL.Cdbo[pj], Cdbo_ij );
#else
            BL.Cdbo_bonds[pj] = Cdbo_ij;
#endif
        }
    }

    atomicAdd( &CdDelta[i], CdDelta_i );

#if defined(GPU_ATOMIC_EV)
    atomicAdd( (double *) e_bond_g, (double) e_bond_ );
#else
    e_bond_g[i] = e_bond_;
#endif

#undef BL
}


GPU_GLOBAL void k_bonds_opt( reax_atom const * const my_atoms, real const * const gp_l, 
        single_body_parameters const * const sbp, two_body_parameters const * const tbp, 
        real const * const total_bond_order, real const * const Delta, real * const CdDelta,
        reax_list bond_list, int n, int num_atom_types, real *e_bond_g )
{
    extern __shared__ hipcub::WarpReduce<double>::TempStorage temp_d[];
    int i, j, pj, orig_id_i, thread_id, warp_id, lane_id, itr;
    int start_i, end_i;
    int type_i, type_j;
    real pow_BOs_be2, exp_be12, CEbo, e_bond_;
    real exphu, exphua1, exphub1, exphuov, hulpov;
    real decobdbo, decobdboua, decobdboub;
    real total_bond_order_i, Delta_i, CdDelta_i, Cdbo_ij;
#define BL (bond_list.bond_list_gpu)

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    /* all threads within a warp are assigned the interactions
     * for a unique atom */
    i = thread_id / warpSize;

    if ( i >= n )
    {
        return;
    }

    warp_id = threadIdx.x / warpSize;
    lane_id = thread_id % warpSize;
    const real gp3 = gp_l[3];
    const real gp4 = gp_l[4];
    const real gp7 = gp_l[7];
    const real gp10 = gp_l[10];
    e_bond_ = 0.0;
    orig_id_i = my_atoms[i].orig_id;
    total_bond_order_i = total_bond_order[i];
    Delta_i = Delta[i];
    CdDelta_i = 0.0;

    start_i = Start_Index( i, &bond_list );
    end_i = End_Index( i, &bond_list );
    type_i = my_atoms[i].type;
    const int flag_i_C = GPU_strncmp( sbp[type_i].name, "C", sizeof(sbp[type_i].name) );
    const int flag_i_O = GPU_strncmp( sbp[type_i].name, "O", sizeof(sbp[type_i].name) );

    for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
    {
        if ( pj < end_i )
        {
            j = BL.nbr[pj];

            if ( orig_id_i <= my_atoms[j].orig_id )
            {
                type_j = my_atoms[j].type;
                two_body_parameters const * const twbp = &tbp[ index_tbp(type_i, type_j, num_atom_types) ];

                pow_BOs_be2 = POW( BL.BO_s[pj], twbp->p_be2 );
                exp_be12 = EXP( twbp->p_be1 * ( 1.0 - pow_BOs_be2 ) );
                CEbo = -twbp->De_s * exp_be12
                    * (1.0 - twbp->p_be1 * twbp->p_be2 * pow_BOs_be2);

                /* calculate bond energy */
                e_bond_ += -twbp->De_s * BL.BO_s[pj] * exp_be12
                    - twbp->De_p * BL.BO_pi[pj]
                    - twbp->De_pp * BL.BO_pi2[pj];

                /* calculate derivatives of bond orders */
                Cdbo_ij = CEbo;
#if defined(GPU_STREAM_SINGLE_ACCUM)
                atomicAdd( &BL.Cdbopi[pj], -1.0 * (CEbo + twbp->De_p) );
                atomicAdd( &BL.Cdbopi2[pj], -1.0 * (CEbo + twbp->De_pp) );
#else
                BL.Cdbopi_bonds[pj] = -1.0 * (CEbo + twbp->De_p);
                BL.Cdbopi2_bonds[pj] = -1.0 * (CEbo + twbp->De_pp);
#endif

                /* Stabilisation terminal triple bond */
                if ( BL.BO[pj] >= 1.00 )
                {
                    if ( (flag_i_C == 0 && GPU_strncmp( sbp[type_j].name, "O", sizeof(sbp[type_j].name) ) == 0)
                            || (flag_i_O == 0 && GPU_strncmp( sbp[type_j].name, "C", sizeof(sbp[type_j].name) ) == 0) )
                    {
                        //ba = SQR( BL.BO[pj] - 2.5 );
                        exphu = EXP( -gp7 * SQR(BL.BO[pj] - 2.5) );
                        //oboa = abo(j1) - boa;
                        //obob = abo(j2) - boa;
                        exphua1 = EXP(-gp3 * (total_bond_order_i - BL.BO[pj]));
                        exphub1 = EXP(-gp3 * (total_bond_order[j] - BL.BO[pj]));
                        //ovoab = abo(j1) - aval(it1) + abo(j2) - aval(it2);
                        exphuov = EXP(gp4 * (Delta_i + Delta[j]));
                        hulpov = 1.0 / (1.0 + 25.0 * exphuov);

                        e_bond_ += gp10 * exphu * hulpov * (exphua1 + exphub1);

                        decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1)
                            * ( gp3 - 2.0 * gp7 * (BL.BO[pj] - 2.5) );
                        decobdboua = -gp10 * exphu * hulpov
                            * (gp3 * exphua1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));
                        decobdboub = -gp10 * exphu * hulpov
                            * (gp3 * exphub1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));

                        Cdbo_ij += decobdbo;
                        CdDelta_i += decobdboua;
                        atomicAdd( &CdDelta[j], decobdboub );
                    }
                }

#if defined(GPU_STREAM_SINGLE_ACCUM)
                atomicAdd( &BL.Cdbo[pj], Cdbo_ij );
#else
                BL.Cdbo_bonds[pj] = Cdbo_ij;
#endif
            }
        }

        pj += warpSize;
    }

    CdDelta_i = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_i);
    e_bond_ = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(e_bond_);

    if ( lane_id == 0 )
    {
        atomicAdd( &CdDelta[i], CdDelta_i );

#if defined(GPU_ATOMIC_EV)
        atomicAdd( (double *) e_bond_g, (double) e_bond_ );
#else
        e_bond_g[i] = e_bond_;
#endif
    }

#undef BL
}


void GPU_Compute_Bonds( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list **lists,
        output_controls const * const out_control )
{
#if !defined(GPU_ATOMIC_EV)
    int update_energy;
    real *spad;
#endif

#if defined(LOG_PERFORMANCE)
    hipEventRecord( control->gpu_time_events[TE_BONDS_START], control->gpu_streams[1] );
#endif

#if defined(GPU_ATOMIC_EV)
    sHipMemsetAsync( &data->d_my_en[E_BOND], 0, sizeof(real),
            control->gpu_streams[1], __FILE__, __LINE__ );
#else
    sHipCheckMalloc( &workspace->d_workspace->scratch[1],
            &workspace->d_workspace->scratch_size[1],
            sizeof(real) * system->n, __FILE__, __LINE__ );

    spad = (real *) workspace->d_workspace->scratch[1];
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#endif

    hipStreamWaitEvent( control->gpu_streams[1], control->gpu_stream_events[SE_BOND_ORDER_DONE], 0 );

//    k_bonds <<< control->blocks_n, control->gpu_block_size, 0, control->gpu_streams[1] >>>
//        ( system->d_my_atoms, system->reax_param.gp.d_l,
//          system->reax_param.d_sbp, system->reax_param.d_tbp,
//          workspace->d_workspace->total_bond_order, workspace->d_workspace->Delta, 
//#if defined(GPU_STREAM_SINGLE_ACCUM)
//          workspace->d_workspace->CdDelta,
//#else
//          workspace->d_workspace->CdDelta_bonds,
//#endif
//          *(lists[BONDS]), system->n, system->reax_param.num_atom_types,
//#if defined(GPU_ATOMIC_EV)
//          &data->d_my_en[E_BOND]
//#else
//          spad
//#endif
//        );
//    hipCheckError( );

    k_bonds_opt <<< control->blocks_warp_n, control->gpu_block_size,
                sizeof(hipcub::WarpReduce<double>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                control->gpu_streams[1] >>>
        ( system->d_my_atoms, system->reax_param.gp.d_l,
          system->reax_param.d_sbp, system->reax_param.d_tbp,
          workspace->d_workspace->total_bond_order, workspace->d_workspace->Delta, 
#if defined(GPU_STREAM_SINGLE_ACCUM)
          workspace->d_workspace->CdDelta,
#else
          workspace->d_workspace->CdDelta_bonds,
#endif
          *(lists[BONDS]), system->n, system->reax_param.num_atom_types,
#if defined(GPU_ATOMIC_EV)
          &data->d_my_en[E_BOND]
#else
          spad
#endif
        );
    hipCheckError( );

#if !defined(GPU_ATOMIC_EV)
    if ( update_energy == TRUE )
    {
        GPU_Reduction_Sum( spad, &data->d_my_en[E_BOND],
                system->n, 1, control->gpu_streams[1] );
    }
#endif

#if defined(LOG_PERFORMANCE)
    hipEventRecord( control->gpu_time_events[TE_BONDS_STOP], control->gpu_streams[1] );
#endif
}
