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

#include "cuda_bonds.h"

#include "cuda_list.h"
#include "cuda_helpers.h"
#include "cuda_reduction.h"
#include "cuda_utils.h"

#include "../index_utils.h"


CUDA_GLOBAL void k_bonds( reax_atom *my_atoms, global_parameters gp, 
        single_body_parameters *sbp, two_body_parameters *tbp, 
        storage p_workspace, reax_list p_bond_list, int n, int num_atom_types, 
        real *e_bond_g )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    real pow_BOs_be2, exp_be12, CEbo, e_bond_l;
    real gp3, gp4, gp7, gp10, gp37;
    real exphu, exphua1, exphub1, exphuov, hulpov;
    real decobdbo, decobdboua, decobdboub;
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
    gp37 = (int) gp.l[37];
    e_bond_l = 0.0;

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
            e_bond_l += -twbp->De_s * bo_ij->BO_s * exp_be12
                - twbp->De_p * bo_ij->BO_pi
                - twbp->De_pp * bo_ij->BO_pi2;

            /* calculate derivatives of bond orders */
            bo_ij->Cdbo += CEbo;
            bo_ij->Cdbopi -= CEbo + twbp->De_p;
            bo_ij->Cdbopi2 -= CEbo + twbp->De_pp;

            /* Stabilisation terminal triple bond */
            if ( bo_ij->BO >= 1.00 )
            {
                if ( gp37 == 2
                        || ( (Cuda_strncmp( sbp_i->name, "C", sizeof(sbp_i->name) ) == 0
                            && Cuda_strncmp( sbp_j->name, "O", sizeof(sbp_j->name) ) == 0)
                        || (Cuda_strncmp( sbp_i->name, "O", sizeof(sbp_i->name) ) == 0
                            && Cuda_strncmp( sbp_j->name, "C", sizeof(sbp_j->name) ) == 0) ) )
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

                    e_bond_l += gp10 * exphu * hulpov * (exphua1 + exphub1);

                    decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1)
                        * ( gp3 - 2.0 * gp7 * (bo_ij->BO - 2.5) );
                    decobdboua = -gp10 * exphu * hulpov
                        * (gp3 * exphua1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));
                    decobdboub = -gp10 * exphu * hulpov
                        * (gp3 * exphub1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));

                    bo_ij->Cdbo += decobdbo;
                    workspace->CdDelta[i] += decobdboua;
                    workspace->CdDelta[j] += decobdboub;
                }
            }
        }
    }

#if !defined(CUDA_ACCUM_ATOMIC)
    e_bond_g[i] = e_bond_l;
#else
    atomicAdd( (double *) e_bond_g, (double) e_bond_l );
#endif
}


void Cuda_Compute_Bonds( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control )
{
#if !defined(CUDA_ACCUM_ATOMIC)
    int update_energy;
    real *spad;

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(real) * system->n,
            "Cuda_Compute_Bonds::workspace->scratch" );

    spad = (real *) workspace->scratch;
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#else
    cuda_memset( &((simulation_data *)data->d_simulation_data)->my_en.e_bond,
            0, sizeof(real), "Cuda_Compute_Bonds::e_bond" );
#endif

    k_bonds <<< control->blocks, control->block_size,
            sizeof(real) * control->block_size >>>
        ( system->d_my_atoms, system->reax_param.d_gp,
          system->reax_param.d_sbp, system->reax_param.d_tbp,
          *(workspace->d_workspace), *(lists[BONDS]), 
          system->n, system->reax_param.num_atom_types,
#if !defined(CUDA_ACCUM_ATOMIC)
          spad
#else
          &((simulation_data *)data->d_simulation_data)->my_en.e_bond
#endif
        );
    cudaCheckError( );

#if !defined(CUDA_ACCUM_ATOMIC)
    if ( update_energy == TRUE )
    {
        Cuda_Reduction_Sum( spad, &((simulation_data *)data->d_simulation_data)->my_en.e_bond,
                system->n );
    }
#endif
}
