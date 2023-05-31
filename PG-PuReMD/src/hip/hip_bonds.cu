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

#include "hip_bonds.h"

#include "hip_list.h"
#include "hip_helpers.h"
#include "hip_reduction.h"
#include "hip_utils.h"

#include "../index_utils.h"

#include <hipcub/warp/warp_reduce.hpp>


GPU_GLOBAL void k_bonds( reax_atom *my_atoms, global_parameters gp, 
        single_body_parameters *sbp, two_body_parameters *tbp, 
        storage p_workspace, reax_list p_bond_list, int n, int num_atom_types, 
        real *e_bond_g )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    real pow_BOs_be2, exp_be12, CEbo, e_bond_;
    real gp3, gp4, gp7, gp10;
    real exphu, exphua1, exphub1, exphuov, hulpov;
    real decobdbo, decobdboua, decobdboub;
    real CdDelta_i;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    bond_order_data *bo_ij;
    reax_list *bond_list;
    storage *workspace;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    bond_list = &p_bond_list;
    workspace = &p_workspace;
    gp3 = gp.l[3];
    gp4 = gp.l[4];
    gp7 = gp.l[7];
    gp10 = gp.l[10];
    e_bond_ = 0.0;
    CdDelta_i = 0.0;

    start_i = Start_Index( i, bond_list );
    end_i = End_Index( i, bond_list );

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = bond_list->bond_list[pj].nbr;

        if ( my_atoms[i].orig_id <= my_atoms[j].orig_id )
        {
            type_i = my_atoms[i].type;
            type_j = my_atoms[j].type;
            sbp_i = &sbp[type_i];
            sbp_j = &sbp[type_j];
            twbp = &tbp[ index_tbp(type_i,type_j, num_atom_types) ];
            bo_ij = &bond_list->bond_list[pj].bo_data;

            pow_BOs_be2 = POW( bo_ij->BO_s, twbp->p_be2 );
            exp_be12 = EXP( twbp->p_be1 * ( 1.0 - pow_BOs_be2 ) );
            CEbo = -twbp->De_s * exp_be12
                * (1.0 - twbp->p_be1 * twbp->p_be2 * pow_BOs_be2);

            /* calculate bond energy */
            e_bond_ += -twbp->De_s * bo_ij->BO_s * exp_be12
                - twbp->De_p * bo_ij->BO_pi
                - twbp->De_pp * bo_ij->BO_pi2;

            /* calculate derivatives of bond orders */
            atomicAdd( &bo_ij->Cdbo, CEbo );
            atomicAdd( &bo_ij->Cdbopi, -1.0 * (CEbo + twbp->De_p) );
            atomicAdd( &bo_ij->Cdbopi2, -1.0 * (CEbo + twbp->De_pp) );

            /* Stabilisation terminal triple bond */
            if ( bo_ij->BO >= 1.00 )
            {
                if ( (Hip_strncmp( sbp_i->name, "C", sizeof(sbp_i->name) ) == 0
                            && Hip_strncmp( sbp_j->name, "O", sizeof(sbp_j->name) ) == 0)
                        || (Hip_strncmp( sbp_i->name, "O", sizeof(sbp_i->name) ) == 0
                            && Hip_strncmp( sbp_j->name, "C", sizeof(sbp_j->name) ) == 0) )
                {
                    //ba = SQR( bo_ij->BO - 2.5 );
                    exphu = EXP( -gp7 * SQR(bo_ij->BO - 2.5) );
                    //oboa = abo(j1) - boa;
                    //obob = abo(j2) - boa;
                    exphua1 = EXP(-gp3 * (workspace->total_bond_order[i] - bo_ij->BO));
                    exphub1 = EXP(-gp3 * (workspace->total_bond_order[j] - bo_ij->BO));
                    //ovoab = abo(j1) - aval(it1) + abo(j2) - aval(it2);
                    exphuov = EXP(gp4 * (workspace->Delta[i] + workspace->Delta[j]));
                    hulpov = 1.0 / (1.0 + 25.0 * exphuov);

                    e_bond_ += gp10 * exphu * hulpov * (exphua1 + exphub1);

                    decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1)
                        * ( gp3 - 2.0 * gp7 * (bo_ij->BO - 2.5) );
                    decobdboua = -gp10 * exphu * hulpov
                        * (gp3 * exphua1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));
                    decobdboub = -gp10 * exphu * hulpov
                        * (gp3 * exphub1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));

                    atomicAdd( &bo_ij->Cdbo, decobdbo );
                    CdDelta_i += decobdboua;
                    atomicAdd( &workspace->CdDelta[j], decobdboub );
                }
            }
        }
    }

    atomicAdd( &workspace->CdDelta[i], CdDelta_i );

#if !defined(GPU_ACCUM_ATOMIC)
    e_bond_g[i] = e_bond_;
#else
    atomicAdd( (double *) e_bond_g, (double) e_bond_ );
#endif
}


GPU_GLOBAL void k_bonds_opt( reax_atom *my_atoms, global_parameters gp, 
        single_body_parameters *sbp, two_body_parameters *tbp, 
        storage p_workspace, reax_list p_bond_list, int n, int num_atom_types, 
        real *e_bond_g )
{
    extern __shared__ hipcub::WarpReduce<double>::TempStorage temp_d[];
    int i, j, pj, thread_id, warp_id, lane_id, itr;;
    int start_i, end_i;
    int type_i, type_j;
    real pow_BOs_be2, exp_be12, CEbo, e_bond_;
    real gp3, gp4, gp7, gp10;
    real exphu, exphua1, exphub1, exphuov, hulpov;
    real decobdbo, decobdboua, decobdboub;
    real CdDelta_i;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    bond_order_data *bo_ij;
    reax_list *bond_list;
    storage *workspace;

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
    bond_list = &p_bond_list;
    workspace = &p_workspace;
    gp3 = gp.l[3];
    gp4 = gp.l[4];
    gp7 = gp.l[7];
    gp10 = gp.l[10];
    e_bond_ = 0.0;
    CdDelta_i = 0.0;

    start_i = Start_Index( i, bond_list );
    end_i = End_Index( i, bond_list );

    for ( itr = 0, pj = start_i + lane_id; itr < (end_i - start_i + warpSize - 1) / warpSize; ++itr )
    {
        if ( pj < end_i )
        {
            j = bond_list->bond_list[pj].nbr;

            if ( my_atoms[i].orig_id <= my_atoms[j].orig_id )
            {
                type_i = my_atoms[i].type;
                type_j = my_atoms[j].type;
                sbp_i = &sbp[type_i];
                sbp_j = &sbp[type_j];
                twbp = &tbp[ index_tbp(type_i,type_j, num_atom_types) ];
                bo_ij = &bond_list->bond_list[pj].bo_data;

                pow_BOs_be2 = POW( bo_ij->BO_s, twbp->p_be2 );
                exp_be12 = EXP( twbp->p_be1 * ( 1.0 - pow_BOs_be2 ) );
                CEbo = -twbp->De_s * exp_be12
                    * (1.0 - twbp->p_be1 * twbp->p_be2 * pow_BOs_be2);

                /* calculate bond energy */
                e_bond_ += -twbp->De_s * bo_ij->BO_s * exp_be12
                    - twbp->De_p * bo_ij->BO_pi
                    - twbp->De_pp * bo_ij->BO_pi2;

                /* calculate derivatives of bond orders */
                atomicAdd( &bo_ij->Cdbo, CEbo );
                atomicAdd( &bo_ij->Cdbopi, -1.0 * (CEbo + twbp->De_p) );
                atomicAdd( &bo_ij->Cdbopi2, -1.0 * (CEbo + twbp->De_pp) );

                /* Stabilisation terminal triple bond */
                if ( bo_ij->BO >= 1.00 )
                {
                    if ( (Hip_strncmp( sbp_i->name, "C", sizeof(sbp_i->name) ) == 0
                                && Hip_strncmp( sbp_j->name, "O", sizeof(sbp_j->name) ) == 0)
                            || (Hip_strncmp( sbp_i->name, "O", sizeof(sbp_i->name) ) == 0
                                && Hip_strncmp( sbp_j->name, "C", sizeof(sbp_j->name) ) == 0) )
                    {
                        //ba = SQR( bo_ij->BO - 2.5 );
                        exphu = EXP( -gp7 * SQR(bo_ij->BO - 2.5) );
                        //oboa = abo(j1) - boa;
                        //obob = abo(j2) - boa;
                        exphua1 = EXP(-gp3 * (workspace->total_bond_order[i] - bo_ij->BO));
                        exphub1 = EXP(-gp3 * (workspace->total_bond_order[j] - bo_ij->BO));
                        //ovoab = abo(j1) - aval(it1) + abo(j2) - aval(it2);
                        exphuov = EXP(gp4 * (workspace->Delta[i] + workspace->Delta[j]));
                        hulpov = 1.0 / (1.0 + 25.0 * exphuov);

                        e_bond_ += gp10 * exphu * hulpov * (exphua1 + exphub1);

                        decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1)
                            * ( gp3 - 2.0 * gp7 * (bo_ij->BO - 2.5) );
                        decobdboua = -gp10 * exphu * hulpov
                            * (gp3 * exphua1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));
                        decobdboub = -gp10 * exphu * hulpov
                            * (gp3 * exphub1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));

                        atomicAdd( &bo_ij->Cdbo, decobdbo );
                        CdDelta_i += decobdboua;
                        atomicAdd( &workspace->CdDelta[j], decobdboub );
                    }
                }
            }
        }

        pj += warpSize;
    }

    CdDelta_i = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(CdDelta_i);
    e_bond_ = hipcub::WarpReduce<double>(temp_d[warp_id]).Sum(e_bond_);

    if ( lane_id == 0 )
    {
        atomicAdd( &workspace->CdDelta[i], CdDelta_i );

#if !defined(GPU_ACCUM_ATOMIC)
        e_bond_g[i] = e_bond_;
#else
        atomicAdd( (double *) e_bond_g, (double) e_bond_ );
#endif
    }
}


void Hip_Compute_Bonds( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list **lists,
        output_controls const * const out_control )
{
#if !defined(GPU_ACCUM_ATOMIC)
    int update_energy;
    real *spad;
#endif

#if defined(LOG_PERFORMANCE)
    hipEventRecord( control->hip_time_events[TE_BONDS_START], control->hip_streams[1] );
#endif

#if !defined(GPU_ACCUM_ATOMIC)
    sHipCheckMalloc( &workspace->scratch[1], &workspace->scratch_size[1],
            sizeof(real) * system->n, __FILE__, __LINE__ );

    spad = (real *) workspace->scratch[1];
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#else
    sHipMemsetAsync( &data->d_my_en->e_bond,
            0, sizeof(real), control->hip_streams[1], __FILE__, __LINE__ );
#endif

    hipStreamWaitEvent( control->hip_streams[1], control->hip_stream_events[SE_BOND_ORDER_DONE], 0 );

//    k_bonds <<< control->blocks_n, control->gpu_block_size, 0, control->hip_streams[1] >>>
//        ( system->d_my_atoms, system->reax_param.d_gp,
//          system->reax_param.d_sbp, system->reax_param.d_tbp,
//          *(workspace->d_workspace), *(lists[BONDS]), 
//          system->n, system->reax_param.num_atom_types,
//#if !defined(GPU_ACCUM_ATOMIC)
//          spad
//#else
//          &data->d_my_en->e_bond
//#endif
//        );
//    hipCheckError( );

    k_bonds_opt <<< control->blocks_warp_n, control->gpu_block_size,
                sizeof(hipcub::WarpReduce<double>::TempStorage) * (control->gpu_block_size / WARP_SIZE),
                control->hip_streams[1] >>>
        ( system->d_my_atoms, system->reax_param.d_gp,
          system->reax_param.d_sbp, system->reax_param.d_tbp,
          *(workspace->d_workspace), *(lists[BONDS]), 
          system->n, system->reax_param.num_atom_types,
#if !defined(GPU_ACCUM_ATOMIC)
          spad
#else
          &data->d_my_en->e_bond
#endif
        );
    hipCheckError( );

#if !defined(GPU_ACCUM_ATOMIC)
    if ( update_energy == TRUE )
    {
        Hip_Reduction_Sum( spad, &data->d_my_en->e_bond,
                system->n, 1, control->hip_streams[1] );
    }
#endif

#if defined(LOG_PERFORMANCE)
    hipEventRecord( control->hip_time_events[TE_BONDS_STOP], control->hip_streams[1] );
#endif
}
