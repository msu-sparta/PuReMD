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

#include "cuda_multi_body.h"

#include "cuda_helpers.h"
#include "cuda_list.h"
#include "cuda_utils.h"

#include "../index_utils.h"


CUDA_GLOBAL void k_atom_energy_part1( reax_atom *my_atoms, global_parameters gp, 
        single_body_parameters *sbp, two_body_parameters *tbp, 
        storage workspace, reax_list bond_list, int n, int num_atom_types,
        real *e_lp_g, real *e_ov_g, real *e_un_g )
{
    int i, j, pj, type_i, type_j;
    real Delta_lpcorr, dfvl;
    real expvd2, inv_expvd2, dElp, CElp, DlpVi;
    real Di, vov3, deahu2dbo, deahu2dsbo;
    real CEover1, CEover2, CEover3, CEover4, e_lp_l;
    real exp_ovun1, exp_ovun2, sum_ovun1, sum_ovun2;
    real exp_ovun2n, exp_ovun6, exp_ovun8;
    real inv_exp_ovun1, inv_exp_ovun2, inv_exp_ovun2n, inv_exp_ovun8;
    real e_un, CEunder1, CEunder2, CEunder3, CEunder4;
    real p_lp2, p_lp3;
    real p_ovun2, p_ovun3, p_ovun4, p_ovun5, p_ovun6, p_ovun7, p_ovun8;
    single_body_parameters *sbp_i;
    two_body_parameters *twbp;
    bond_data *pbond_ij;
    bond_order_data *bo_ij; 

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    p_lp3 = gp.l[5];
    p_ovun3 = gp.l[32];
    p_ovun4 = gp.l[31];
    p_ovun6 = gp.l[6];
    p_ovun7 = gp.l[8];
    p_ovun8 = gp.l[9];

    type_i = my_atoms[i].type;
    sbp_i = &sbp[ type_i ];

    /* lone-pair Energy */
    p_lp2 = sbp_i->p_lp2;      
    expvd2 = EXP( -75.0 * workspace.Delta_lp[i] );
    inv_expvd2 = 1.0 / (1.0 + expvd2 );

    /* calculate the energy */
    e_lp_l = p_lp2 * workspace.Delta_lp[i] * inv_expvd2;

    dElp = p_lp2 * inv_expvd2 + 75.0 * p_lp2 * workspace.Delta_lp[i]
        * expvd2 * SQR(inv_expvd2);
    CElp = dElp * workspace.dDelta_lp[i];

    workspace.CdDelta[i] += CElp;  // lp - 1st term  

    /* correction for C2 */
    if ( gp.l[5] > 0.001
            && Cuda_strncmp( sbp[ type_i ].name, "C",
                sizeof(sbp[ type_i ].name) ) == 0 )
    {
        for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
        {
            if ( my_atoms[i].orig_id < 
                    my_atoms[bond_list.bond_list[pj].nbr].orig_id )
            {
                j = bond_list.bond_list[pj].nbr;
                type_j = my_atoms[j].type;

                if ( Cuda_strncmp( sbp[ type_j ].name, "C",
                            sizeof(sbp[ type_j ].name) ) == 0 )
                {
                    twbp = &tbp[ index_tbp(type_i,type_j, num_atom_types) ];
                    bo_ij = &bond_list.bond_list[pj].bo_data;
                    Di = workspace.Delta[i];
                    vov3 = bo_ij->BO - Di - 0.040 * POW( Di, 4.0 );

                    if ( vov3 > 3.0 )
                    {
                        e_lp_l += p_lp3 * SQR( vov3 - 3.0 );

                        deahu2dbo = 2.0 * p_lp3 * (vov3 - 3.0);
                        deahu2dsbo = 2.0 * p_lp3 * (vov3 - 3.0)
                            * (-1.0 - 0.16 * POW(Di, 3.0));

                        bo_ij->Cdbo += deahu2dbo;
                        workspace.CdDelta[i] += deahu2dsbo;
                    }
                }    
            }
        }
    }

#if !defined(CUDA_ACCUM_ATOMIC)
    e_lp_g[i] = e_lp_l;
#else
    atomicAdd( (double *) e_lp_g, (double) e_lp_l );
#endif

    /* over-coordination energy */
    if ( sbp_i->mass > 21.0 ) 
    {
        dfvl = 0.0;
    }
    /* only for 1st-row elements */
    else
    {
        dfvl = 1.0;
    }

    p_ovun2 = sbp_i->p_ovun2;
    sum_ovun1 = 0.0;
    sum_ovun2 = 0.0;

    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        j = bond_list.bond_list[pj].nbr;
        type_j = my_atoms[j].type;
        bo_ij = &bond_list.bond_list[pj].bo_data;
        twbp = &tbp[ index_tbp(type_i, type_j, num_atom_types) ];

        sum_ovun1 += twbp->p_ovun1 * twbp->De_s * bo_ij->BO;
        sum_ovun2 += (workspace.Delta[j] - dfvl * workspace.Delta_lp_temp[j])
            * ( bo_ij->BO_pi + bo_ij->BO_pi2 );
    }

    exp_ovun1 = p_ovun3 * EXP( p_ovun4 * sum_ovun2 );
    inv_exp_ovun1 = 1.0 / (1.0 + exp_ovun1);
    Delta_lpcorr  = workspace.Delta[i]
        - (dfvl * workspace.Delta_lp_temp[i]) * inv_exp_ovun1;

    exp_ovun2 = EXP( p_ovun2 * Delta_lpcorr );
    inv_exp_ovun2 = 1.0 / (1.0 + exp_ovun2);

    DlpVi = 1.0 / (Delta_lpcorr + sbp_i->valency + 1.0e-8);
    CEover1 = Delta_lpcorr * DlpVi * inv_exp_ovun2;

#if !defined(CUDA_ACCUM_ATOMIC)
    e_ov_g[i] = sum_ovun1 * CEover1;
#else
    atomicAdd( (double *) e_ov_g, (double) (sum_ovun1 * CEover1) );
#endif

    CEover2 = sum_ovun1 * DlpVi * inv_exp_ovun2 * (1.0 - Delta_lpcorr
            * ( DlpVi + p_ovun2 * exp_ovun2 * inv_exp_ovun2 ));

    CEover3 = CEover2 * (1.0 - dfvl * workspace.dDelta_lp[i] * inv_exp_ovun1 );

    CEover4 = CEover2 * (dfvl * workspace.Delta_lp_temp[i])
        * p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1);

    /* under-coordination potential */
    p_ovun2 = sbp_i->p_ovun2;
    p_ovun5 = sbp_i->p_ovun5;

    exp_ovun2n = 1.0 / exp_ovun2;
    exp_ovun6 = EXP( p_ovun6 * Delta_lpcorr );
    exp_ovun8 = p_ovun7 * EXP(p_ovun8 * sum_ovun2);
    inv_exp_ovun2n = 1.0 / (1.0 + exp_ovun2n);
    inv_exp_ovun8 = 1.0 / (1.0 + exp_ovun8);

    e_un = -p_ovun5 * (1.0 - exp_ovun6) * inv_exp_ovun2n * inv_exp_ovun8;
#if !defined(CUDA_ACCUM_ATOMIC)
    e_un_g[i] = e_un;
#else
    atomicAdd( (double *) e_un_g, (double) e_un );
#endif

    CEunder1 = inv_exp_ovun2n * ( p_ovun5 * p_ovun6 * exp_ovun6 * inv_exp_ovun8
            + p_ovun2 * e_un * exp_ovun2n );
    CEunder2 = -e_un * p_ovun8 * exp_ovun8 * inv_exp_ovun8;
    CEunder3 = CEunder1 * (1.0 - dfvl * workspace.dDelta_lp[i] * inv_exp_ovun1);
    CEunder4 = CEunder1 * (dfvl * workspace.Delta_lp_temp[i])
        * p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1) + CEunder2;

    /* forces */
    workspace.CdDelta[i] += CEover3;   // OvCoor - 2nd term
    workspace.CdDelta[i] += CEunder3;  // UnCoor - 1st term

    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        pbond_ij = &bond_list.bond_list[pj];
        j = pbond_ij->nbr;
        bo_ij = &pbond_ij->bo_data;
        twbp  = &tbp[
            index_tbp(my_atoms[i].type, my_atoms[j].type, 
                    num_atom_types) ];

        bo_ij->Cdbo += CEover1 * twbp->p_ovun1 * twbp->De_s;// OvCoor-1st 
#if !defined(CUDA_ACCUM_ATOMIC)
        pbond_ij->ae_CdDelta += CEover4 * (1.0 - dfvl * workspace.dDelta_lp[j])
            * (bo_ij->BO_pi + bo_ij->BO_pi2); // OvCoor-3a
#else
        atomicAdd( &workspace.CdDelta[j], CEover4 * (1.0 - dfvl * workspace.dDelta_lp[j])
            * (bo_ij->BO_pi + bo_ij->BO_pi2) );
#endif
        bo_ij->Cdbopi += CEover4 * (workspace.Delta[j] - dfvl
                * workspace.Delta_lp_temp[j]); // OvCoor-3b
        bo_ij->Cdbopi2 += CEover4 * (workspace.Delta[j] - dfvl
                * workspace.Delta_lp_temp[j]);  // OvCoor-3b

#if !defined(CUDA_ACCUM_ATOMIC)
        pbond_ij->ae_CdDelta += CEunder4 * (1.0 - dfvl * workspace.dDelta_lp[j])
            * (bo_ij->BO_pi + bo_ij->BO_pi2);   // UnCoor - 2a
#else
        atomicAdd( &workspace.CdDelta[j], CEunder4 * (1.0 - dfvl * workspace.dDelta_lp[j])
            * (bo_ij->BO_pi + bo_ij->BO_pi2) );
#endif
        bo_ij->Cdbopi += CEunder4 * (workspace.Delta[j] - dfvl
                * workspace.Delta_lp_temp[j]);  // UnCoor-2b
        bo_ij->Cdbopi2 += CEunder4 * (workspace.Delta[j] - dfvl
                * workspace.Delta_lp_temp[j]);  // UnCoor-2b
    }
}


#if !defined(CUDA_ACCUM_ATOMIC)
/* Traverse bond list and accumulate lone pair contributions from bonded neighbors */
CUDA_GLOBAL void k_atom_energy_part2( reax_list bond_list, 
        storage workspace, int n )
{
    int i, pj;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    for ( pj = Start_Index(i, &bond_list); pj < End_Index(i, &bond_list); ++pj )
    {
        workspace.CdDelta[i] +=
            bond_list.bond_list[ bond_list.bond_list[pj].sym_index ].ae_CdDelta;
    }
}
#endif


void Cuda_Compute_Atom_Energy( reax_system *system, control_params *control, 
        simulation_data *data, storage *workspace, 
        reax_list **lists, output_controls *out_control )
{
#if !defined(CUDA_ACCUM_ATOMIC)
    int update_energy;
    real *spad;

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(real) * 3 * system->n,
            "Cuda_Compute_Atom_Energy::workspace->scratch" );

    spad = (real *) workspace->scratch;
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
#else
    cuda_memset( &((simulation_data *)data->d_simulation_data)->my_en.e_lp,
            0, sizeof(real), "Cuda_Compute_Atom_Energy::e_lp" );
    cuda_memset( &((simulation_data *)data->d_simulation_data)->my_en.e_ov,
            0, sizeof(real), "Cuda_Compute_Atom_Energy::e_ov" );
    cuda_memset( &((simulation_data *)data->d_simulation_data)->my_en.e_un,
            0, sizeof(real), "Cuda_Compute_Atom_Energy::e_un" );
#endif

    k_atom_energy_part1 <<< control->blocks, control->block_size >>>
        ( system->d_my_atoms, system->reax_param.d_gp,
          system->reax_param.d_sbp, system->reax_param.d_tbp, *(workspace->d_workspace),
          *(lists[BONDS]), system->n, system->reax_param.num_atom_types,
#if !defined(CUDA_ACCUM_ATOMIC)
          spad, &spad[system->n], &spad[2 * system->n]
#else
          &((simulation_data *)data->d_simulation_data)->my_en.e_lp,
          &((simulation_data *)data->d_simulation_data)->my_en.e_ov,
          &((simulation_data *)data->d_simulation_data)->my_en.e_un
#endif
         );
    cudaCheckError( );

#if !defined(CUDA_ACCUM_ATOMIC)
    k_atom_energy_part2 <<< control->blocks, control->block_size >>>
        ( *(lists[BONDS]), *(workspace->d_workspace), system->n );
    cudaCheckError( );

    if ( update_energy == TRUE )
    {
        Cuda_Reduction_Sum( spad,
                &((simulation_data *)data->d_simulation_data)->my_en.e_lp,
                system->n );

        Cuda_Reduction_Sum( &spad[system->n],
                &((simulation_data *)data->d_simulation_data)->my_en.e_ov,
                system->n );

        Cuda_Reduction_Sum( &spad[2 * system->n],
                &((simulation_data *)data->d_simulation_data)->my_en.e_un,
                system->n );
    }
#endif
}
