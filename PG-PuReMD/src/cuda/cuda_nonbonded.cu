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

#include "cuda_nonbonded.h"

#include "cuda_helpers.h"
#include "cuda_list.h"
#include "cuda_reduction.h"
#include "cuda_utils.h"

#include "../index_utils.h"
#include "../vector.h"

#include <cub/warp/warp_reduce.cuh>


CUDA_GLOBAL void k_compute_polarization_energy( reax_atom const * const my_atoms, 
        single_body_parameters const * const sbp, int n, real * const e_pol_g )
{
    int i, type_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    type_i = my_atoms[i].type;

#if !defined(CUDA_ACCUM_ATOMIC)
    e_pol_g[i] = KCALpMOL_to_EV * (sbp[type_i].chi * my_atoms[i].q
            + (sbp[type_i].eta / 2.0) * SQR(my_atoms[i].q));
#else
    atomicAdd( (double *) e_pol_g, (double) (KCALpMOL_to_EV * (sbp[type_i].chi
                    * my_atoms[i].q + (sbp[type_i].eta / 2.0) * SQR(my_atoms[i].q))) );
#endif
}


/* Compute energies and forces due to van der Waals and Coulomb interactions
 * where the far neighbors list is in full format
 *
 * This implementation assigns one thread per atom */
CUDA_GLOBAL void k_vdW_coulomb_energy_full( reax_atom const * const my_atoms,
        two_body_parameters const * const tbp, global_parameters gp,
        control_params const * const control, storage workspace,
        reax_list far_nbr_list, int n, int num_atom_types, 
        real * const e_vdW_g, real * const e_ele_g )
{
    int i, j, pj;
    int start_i, end_i, orig_i, orig_j, tbp_ij;
    real self_coef;
    real p_vdW1, p_vdW1i;
    real powr_vdW1, powgi_vdW1;
    real r_ij, fn13, exp1, exp2, e_base, de_base;
    real Tap, dTap, dfn13, CEvd, CEclmb;
    real dr3gamij_1, dr3gamij_3;
    real e_ele_, e_vdW_, e_core, de_core, e_clb, de_clb;
    rvec temp, f_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    p_vdW1 = gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    e_vdW_ = 0.0;
    e_ele_ = 0.0;
    rvec_MakeZero( f_i );

    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    orig_i = my_atoms[i].orig_id;

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        orig_j = my_atoms[j].orig_id;

        if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut 
                && orig_i < orig_j )
        {
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            tbp_ij = index_tbp(my_atoms[i].type, my_atoms[j].type, num_atom_types);

            /* i == j: self-interaction from periodic image,
             * important for supporting small boxes! */
            self_coef = (orig_i == orig_j) ? 0.5 : 1.0;

            /* Calculate Taper and its derivative */
            Tap = workspace.Tap[7] * r_ij
                + workspace.Tap[6];
            Tap = Tap * r_ij + workspace.Tap[5];
            Tap = Tap * r_ij + workspace.Tap[4];
            Tap = Tap * r_ij + workspace.Tap[3];
            Tap = Tap * r_ij + workspace.Tap[2];
            Tap = Tap * r_ij + workspace.Tap[1];
            Tap = Tap * r_ij + workspace.Tap[0];

            dTap = 7.0 * workspace.Tap[7] * r_ij
                + 6.0 * workspace.Tap[6];
            dTap = dTap * r_ij + 5.0 * workspace.Tap[5];
            dTap = dTap * r_ij + 4.0 * workspace.Tap[4];
            dTap = dTap * r_ij + 3.0 * workspace.Tap[3];
            dTap = dTap * r_ij + 2.0 * workspace.Tap[2];
            dTap = dTap * r_ij + workspace.Tap[1];

            /* vdWaals Calculations */
            if ( gp.vdw_type == 1 || gp.vdw_type == 3 )
            {
                /* shielding */
                powr_vdW1 = POW( r_ij, p_vdW1 );
                powgi_vdW1 = POW( 1.0 / tbp[tbp_ij].gamma_w, p_vdW1 );

                fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                exp1 = EXP( tbp[tbp_ij].alpha * (1.0 - fn13 / tbp[tbp_ij].r_vdW) );
                exp2 = EXP( 0.5 * tbp[tbp_ij].alpha * (1.0 - fn13 / tbp[tbp_ij].r_vdW) );
                e_base = tbp[tbp_ij].D * (exp1 - 2.0 * exp2);

                e_vdW_ += self_coef * (e_base * Tap);

                dfn13 = POW( r_ij, p_vdW1 - 1.0 )
                    * POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0 );
                de_base = (tbp[tbp_ij].D * tbp[tbp_ij].alpha / tbp[tbp_ij].r_vdW) * (exp2 - exp1) * dfn13;
            }
            /* no shielding */
            else
            {
                exp1 = EXP( tbp[tbp_ij].alpha * (1.0 - r_ij / tbp[tbp_ij].r_vdW) );
                exp2 = EXP( 0.5 * tbp[tbp_ij].alpha * (1.0 - r_ij / tbp[tbp_ij].r_vdW) );
                e_base = tbp[tbp_ij].D * (exp1 - 2.0 * exp2);

                e_vdW_ += self_coef * (e_base * Tap);

                de_base = (tbp[tbp_ij].D * tbp[tbp_ij].alpha / tbp[tbp_ij].r_vdW) * (exp2 - exp1);
            }

            /* calculate inner core repulsion */
            if ( gp.vdw_type == 2 || gp.vdw_type == 3 )
            {
                e_core = tbp[tbp_ij].ecore * EXP( tbp[tbp_ij].acore * (1.0 - (r_ij / tbp[tbp_ij].rcore)) );
                e_vdW_ += self_coef * (e_core * Tap);

                de_core = -(tbp[tbp_ij].acore / tbp[tbp_ij].rcore) * e_core;
            }
            else
            {
                e_core = 0.0;
                de_core = 0.0;
            }

            CEvd = self_coef * ( (de_base + de_core) * Tap
                    + (e_base + e_core) * dTap );

            /* Coulomb Calculations */
            dr3gamij_1 = r_ij * r_ij * r_ij
                + POW( tbp[tbp_ij].gamma, -3.0 );
            dr3gamij_3 = POW( dr3gamij_1, 1.0 / 3.0 );
            e_clb = C_ELE * (my_atoms[i].q * my_atoms[j].q) / dr3gamij_3;
            e_ele_ += self_coef * (e_clb * Tap);

            de_clb = -C_ELE * (my_atoms[i].q * my_atoms[j].q)
                    * (r_ij * r_ij) / POW( dr3gamij_1, 4.0 / 3.0 );
            CEclmb = self_coef * (de_clb * Tap + e_clb * dTap);

            rvec_Scale( temp, -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_Add( f_i, temp );
            rvec_Scale( temp, -1.0, temp );
            atomic_rvecAdd( workspace.f[j], temp );
        }
    }

    atomic_rvecAdd( workspace.f[i], f_i );
#if !defined(CUDA_ACCUM_ATOMIC)
    e_vdW_g[i] = e_vdW_;
    e_ele_g[i] = e_ele_;
#else
    atomicAdd( (double *) e_vdW_g, (double) e_vdW_ );
    atomicAdd( (double *) e_ele_g, (double) e_ele_ );
#endif
}


/* Compute virial terms, energies, and forces due to van der Waals and Coulomb interactions
 * where the far neighbors list is in full format
 *
 * This implementation assigns one thread per atom */
CUDA_GLOBAL void k_vdW_coulomb_energy_virial_full( reax_atom const * const my_atoms, 
        two_body_parameters const * const tbp, global_parameters gp,
        control_params const * const control, storage workspace,
        reax_list far_nbr_list, int n, int num_atom_types, 
        real * const e_vdW_g, real * const e_ele_g, rvec * const ext_press_g )
{
    int i, j, pj;
    int start_i, end_i, orig_i, orig_j, tbp_ij;
    real self_coef;
    real p_vdW1, p_vdW1i;
    real powr_vdW1, powgi_vdW1;
    real r_ij, fn13, exp1, exp2, e_base, de_base;
    real Tap, dTap, dfn13, CEvd, CEclmb;
    real dr3gamij_1, dr3gamij_3;
    real e_ele_, e_vdW_, e_core, de_core, e_clb, de_clb;
    rvec temp, f_i, ext_press_;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    p_vdW1 = gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    e_vdW_ = 0.0;
    e_ele_ = 0.0;
    rvec_MakeZero( f_i );
    rvec_MakeZero( ext_press_ );

    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    orig_i = my_atoms[i].orig_id;

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        orig_j = my_atoms[j].orig_id;

        if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut 
                && orig_i < orig_j )
        {
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            tbp_ij = index_tbp(my_atoms[i].type, my_atoms[j].type, num_atom_types);

            /* i == j: self-interaction from periodic image,
             * important for supporting small boxes! */
            self_coef = (orig_i == orig_j) ? 0.5 : 1.0;

            /* Calculate Taper and its derivative */
            Tap = workspace.Tap[7] * r_ij
                + workspace.Tap[6];
            Tap = Tap * r_ij + workspace.Tap[5];
            Tap = Tap * r_ij + workspace.Tap[4];
            Tap = Tap * r_ij + workspace.Tap[3];
            Tap = Tap * r_ij + workspace.Tap[2];
            Tap = Tap * r_ij + workspace.Tap[1];
            Tap = Tap * r_ij + workspace.Tap[0];

            dTap = 7.0 * workspace.Tap[7] * r_ij
                + 6.0 * workspace.Tap[6];
            dTap = dTap * r_ij + 5.0 * workspace.Tap[5];
            dTap = dTap * r_ij + 4.0 * workspace.Tap[4];
            dTap = dTap * r_ij + 3.0 * workspace.Tap[3];
            dTap = dTap * r_ij + 2.0 * workspace.Tap[2];
            dTap = dTap * r_ij + workspace.Tap[1];

            /* vdWaals Calculations */
            if ( gp.vdw_type == 1 || gp.vdw_type == 3 )
            {
                /* shielding */
                powr_vdW1 = POW( r_ij, p_vdW1 );
                powgi_vdW1 = POW( 1.0 / tbp[tbp_ij].gamma_w, p_vdW1 );

                fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                exp1 = EXP( tbp[tbp_ij].alpha * (1.0 - fn13 / tbp[tbp_ij].r_vdW) );
                exp2 = EXP( 0.5 * tbp[tbp_ij].alpha * (1.0 - fn13 / tbp[tbp_ij].r_vdW) );
                e_base = tbp[tbp_ij].D * (exp1 - 2.0 * exp2);

                e_vdW_ += self_coef * (e_base * Tap);

                dfn13 = POW( r_ij, p_vdW1 - 1.0 )
                    * POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0 );
                de_base = (tbp[tbp_ij].D * tbp[tbp_ij].alpha / tbp[tbp_ij].r_vdW) * (exp2 - exp1) * dfn13;
            }
            /* no shielding */
            else
            {
                exp1 = EXP( tbp[tbp_ij].alpha * (1.0 - r_ij / tbp[tbp_ij].r_vdW) );
                exp2 = EXP( 0.5 * tbp[tbp_ij].alpha * (1.0 - r_ij / tbp[tbp_ij].r_vdW) );
                e_base = tbp[tbp_ij].D * (exp1 - 2.0 * exp2);

                e_vdW_ += self_coef * (e_base * Tap);

                de_base = (tbp[tbp_ij].D * tbp[tbp_ij].alpha / tbp[tbp_ij].r_vdW) * (exp2 - exp1);
            }

            /* calculate inner core repulsion */
            if ( gp.vdw_type == 2 || gp.vdw_type == 3 )
            {
                e_core = tbp[tbp_ij].ecore * EXP( tbp[tbp_ij].acore * (1.0 - (r_ij / tbp[tbp_ij].rcore)) );
                e_vdW_ += self_coef * (e_core * Tap);

                de_core = -(tbp[tbp_ij].acore / tbp[tbp_ij].rcore) * e_core;
            }
            else
            {
                e_core = 0.0;
                de_core = 0.0;
            }

            CEvd = self_coef * ( (de_base + de_core) * Tap
                    + (e_base + e_core) * dTap );

            /* Coulomb Calculations */
            dr3gamij_1 = r_ij * r_ij * r_ij
                + POW( tbp[tbp_ij].gamma, -3.0 );
            dr3gamij_3 = POW( dr3gamij_1, 1.0 / 3.0 );
            e_clb = C_ELE * (my_atoms[i].q * my_atoms[j].q) / dr3gamij_3;
            e_ele_ += self_coef * (e_clb * Tap);

            de_clb = -C_ELE * (my_atoms[i].q * my_atoms[j].q)
                    * (r_ij * r_ij) / POW( dr3gamij_1, 4.0 / 3.0 );
            CEclmb = self_coef * (de_clb * Tap + e_clb * dTap);

            /* for pressure coupling, terms not related to bond order 
               derivatives are added directly into pressure vector/tensor */
            rvec_Scale( temp,
                    -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_Add( f_i, temp );
            rvec_Scale( temp, -1.0, temp );
            atomic_rvecAdd( workspace.f[j], temp );

            rvec_iMultiply( temp,
                    far_nbr_list.far_nbr_list.rel_box[pj], temp );
            rvec_Add( ext_press_, temp );
        }
    }

    atomic_rvecAdd( workspace.f[i], f_i );
#if !defined(CUDA_ACCUM_ATOMIC)
    e_vdW_g[i] = e_vdW_;
    e_ele_g[i] = e_ele_;
    rvec_Copy( ext_press_g[j], ext_press_ );
#else
    atomicAdd( (double *) e_vdW_g, (double) e_vdW_ );
    atomicAdd( (double *) e_ele_g, (double) e_ele_ );
    atomic_rvecAdd( *ext_press_g, ext_press_ );
#endif
}


/* Compute energies and forces due to van der Waals and Coulomb interactions
 * where the far neighbors list is in full format
 *
 * This implementation assigns one warp of threads per atom */
CUDA_GLOBAL void k_vdW_coulomb_energy_full_opt( reax_atom const * const my_atoms, 
        two_body_parameters const * const tbp, global_parameters gp,
        control_params const * const control, storage workspace,
        reax_list far_nbr_list, int n, int num_atom_types, 
        real * const e_vdW_g, real * const e_ele_g )
{
    extern __shared__ cub::WarpReduce<double>::TempStorage temp_storage[];
    int i, j, pj;
    int start_i, end_i, orig_i, orig_j, tbp_ij;
    real self_coef;
    real p_vdW1, p_vdW1i;
    real powr_vdW1, powgi_vdW1;
    real r_ij, fn13, exp1, exp2, e_base, de_base;
    real Tap, dTap, dfn13, CEvd, CEclmb;
    real dr3gamij_1, dr3gamij_3;
    real e_vdW_, e_ele_, e_core, de_core, e_clb, de_clb;
    rvec temp, f_i;
    int thread_id, warp_id, lane_id;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id >> 5;

    if ( warp_id >= n )
    {
        return;
    }

    lane_id = thread_id & 0x0000001F; 
    i = warp_id;
    p_vdW1 = gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    e_vdW_ = 0.0;
    e_ele_ = 0.0;
    rvec_MakeZero( f_i );

    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    orig_i = my_atoms[i].orig_id;

    pj = start_i + lane_id;
    while ( pj < end_i )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        orig_j = my_atoms[j].orig_id;

        if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut 
                && orig_i < orig_j )
        {
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            tbp_ij = index_tbp(my_atoms[i].type, my_atoms[j].type, num_atom_types);

            /* i == j: self-interaction from periodic image,
             * important for supporting small boxes! */
            self_coef = (orig_i == orig_j) ? 0.5 : 1.0;

            /* Calculate Taper and its derivative */
            Tap = workspace.Tap[7] * r_ij
                + workspace.Tap[6];
            Tap = Tap * r_ij + workspace.Tap[5];
            Tap = Tap * r_ij + workspace.Tap[4];
            Tap = Tap * r_ij + workspace.Tap[3];
            Tap = Tap * r_ij + workspace.Tap[2];
            Tap = Tap * r_ij + workspace.Tap[1];
            Tap = Tap * r_ij + workspace.Tap[0];

            dTap = 7.0 * workspace.Tap[7] * r_ij
                + 6.0 * workspace.Tap[6];
            dTap = dTap * r_ij + 5.0 * workspace.Tap[5];
            dTap = dTap * r_ij + 4.0 * workspace.Tap[4];
            dTap = dTap * r_ij + 3.0 * workspace.Tap[3];
            dTap = dTap * r_ij + 2.0 * workspace.Tap[2];
            dTap = dTap * r_ij + workspace.Tap[1];

            /* vdWaals Calculations */
            if ( gp.vdw_type == 1 || gp.vdw_type == 3 )
            {
                /* shielding */
                powr_vdW1 = POW( r_ij, p_vdW1 );
                powgi_vdW1 = POW( 1.0 / tbp[tbp_ij].gamma_w, p_vdW1 );

                fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                exp1 = EXP( tbp[tbp_ij].alpha * (1.0 - fn13 / tbp[tbp_ij].r_vdW) );
                exp2 = EXP( 0.5 * tbp[tbp_ij].alpha * (1.0 - fn13 / tbp[tbp_ij].r_vdW) );
                e_base = tbp[tbp_ij].D * (exp1 - 2.0 * exp2);

                e_vdW_ += self_coef * (e_base * Tap);

                dfn13 = POW( r_ij, p_vdW1 - 1.0 )
                    * POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0 );
                de_base = (tbp[tbp_ij].D * tbp[tbp_ij].alpha / tbp[tbp_ij].r_vdW) * (exp2 - exp1) * dfn13;
            }
            /* no shielding */
            else
            {
                exp1 = EXP( tbp[tbp_ij].alpha * (1.0 - r_ij / tbp[tbp_ij].r_vdW) );
                exp2 = EXP( 0.5 * tbp[tbp_ij].alpha * (1.0 - r_ij / tbp[tbp_ij].r_vdW) );
                e_base = tbp[tbp_ij].D * (exp1 - 2.0 * exp2);

                e_vdW_ += self_coef * (e_base * Tap);

                de_base = (tbp[tbp_ij].D * tbp[tbp_ij].alpha / tbp[tbp_ij].r_vdW) * (exp2 - exp1);
            }

            /* calculate inner core repulsion */
            if ( gp.vdw_type == 2 || gp.vdw_type == 3 )
            {
                e_core = tbp[tbp_ij].ecore * EXP( tbp[tbp_ij].acore * (1.0 - (r_ij / tbp[tbp_ij].rcore)) );
                e_vdW_ += self_coef * (e_core * Tap);

                de_core = -(tbp[tbp_ij].acore / tbp[tbp_ij].rcore) * e_core;
            }
            else
            {
                e_core = 0.0;
                de_core = 0.0;
            }

            CEvd = self_coef * ( (de_base + de_core) * Tap
                    + (e_base + e_core) * dTap );

            /* Coulomb Calculations */
            dr3gamij_1 = r_ij * r_ij * r_ij
                + POW( tbp[tbp_ij].gamma, -3.0 );
            dr3gamij_3 = POW( dr3gamij_1, 1.0 / 3.0 );
            e_clb = C_ELE * (my_atoms[i].q * my_atoms[j].q) / dr3gamij_3;
            e_ele_ += self_coef * (e_clb * Tap);

            de_clb = -C_ELE * (my_atoms[i].q * my_atoms[j].q)
                    * (r_ij * r_ij) / POW( dr3gamij_1, 4.0 / 3.0 );
            CEclmb = self_coef * (de_clb * Tap + e_clb * dTap);

            rvec_Scale( temp, -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_Add( f_i, temp );
            rvec_Scale( temp, -1.0, temp );
            atomic_rvecAdd( workspace.f[j], temp );
        }

        pj += warpSize;
    }

    e_vdW_ = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(e_vdW_);
    e_ele_ = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(e_ele_);
    f_i[0] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_i[0]);
    f_i[1] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_i[1]);
    f_i[2] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_i[2]);

    /* first thread within a warp writes warp-level sum to global memory */
    if ( lane_id == 0 )
    {
        atomic_rvecAdd( workspace.f[i], f_i );
#if !defined(CUDA_ACCUM_ATOMIC)
        e_vdW_g[i] = e_vdW_;
        e_ele_g[i] = e_ele_;
#else
        atomicAdd( (double *) e_vdW_g, (double) e_vdW_ );
        atomicAdd( (double *) e_ele_g, (double) e_ele_ );
#endif
    }
}


/* Compute virial terms, energies, and forces due to van der Waals and Coulomb interactions
 * where the far neighbors list is in full format
 *
 * This implementation assigns one warp of threads per atom */
CUDA_GLOBAL void k_vdW_coulomb_energy_virial_full_opt( reax_atom const * const my_atoms,
        two_body_parameters const * const tbp, global_parameters gp,
        control_params const * const control, storage workspace,
        reax_list far_nbr_list, int n, int num_atom_types, 
        real * const e_vdW_g, real * const e_ele_g, rvec * const ext_press_g )
{
    extern __shared__ cub::WarpReduce<double>::TempStorage temp_storage[];
    int i, j, pj;
    int start_i, end_i, orig_i, orig_j, tbp_ij;
    real self_coef;
    real p_vdW1, p_vdW1i;
    real powr_vdW1, powgi_vdW1;
    real r_ij, fn13, exp1, exp2, e_base, de_base;
    real Tap, dTap, dfn13, CEvd, CEclmb;
    real dr3gamij_1, dr3gamij_3;
    real e_vdW_, e_ele_, e_core, de_core, e_clb, de_clb;
    rvec temp, f_i, ext_press_;
    int thread_id, warp_id, lane_id;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id >> 5;

    if ( warp_id >= n )
    {
        return;
    }

    lane_id = thread_id & 0x0000001F; 
    i = warp_id;
    p_vdW1 = gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    e_vdW_ = 0.0;
    e_ele_ = 0.0;
    rvec_MakeZero( f_i );
    rvec_MakeZero( ext_press_ );

    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    orig_i = my_atoms[i].orig_id;

    pj = start_i + lane_id;
    while ( pj < end_i )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        orig_j = my_atoms[j].orig_id;

        if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut 
                && orig_i < orig_j )
        {
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            tbp_ij = index_tbp(my_atoms[i].type, my_atoms[j].type, num_atom_types);

            /* i == j: self-interaction from periodic image,
             * important for supporting small boxes! */
            self_coef = (orig_i == orig_j) ? 0.5 : 1.0;

            /* Calculate Taper and its derivative */
            Tap = workspace.Tap[7] * r_ij
                + workspace.Tap[6];
            Tap = Tap * r_ij + workspace.Tap[5];
            Tap = Tap * r_ij + workspace.Tap[4];
            Tap = Tap * r_ij + workspace.Tap[3];
            Tap = Tap * r_ij + workspace.Tap[2];
            Tap = Tap * r_ij + workspace.Tap[1];
            Tap = Tap * r_ij + workspace.Tap[0];

            dTap = 7.0 * workspace.Tap[7] * r_ij
                + 6.0 * workspace.Tap[6];
            dTap = dTap * r_ij + 5.0 * workspace.Tap[5];
            dTap = dTap * r_ij + 4.0 * workspace.Tap[4];
            dTap = dTap * r_ij + 3.0 * workspace.Tap[3];
            dTap = dTap * r_ij + 2.0 * workspace.Tap[2];
            dTap = dTap * r_ij + workspace.Tap[1];

            /* vdWaals Calculations */
            if ( gp.vdw_type == 1 || gp.vdw_type == 3 )
            {
                /* shielding */
                powr_vdW1 = POW( r_ij, p_vdW1 );
                powgi_vdW1 = POW( 1.0 / tbp[tbp_ij].gamma_w, p_vdW1 );

                fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                exp1 = EXP( tbp[tbp_ij].alpha * (1.0 - fn13 / tbp[tbp_ij].r_vdW) );
                exp2 = EXP( 0.5 * tbp[tbp_ij].alpha * (1.0 - fn13 / tbp[tbp_ij].r_vdW) );
                e_base = tbp[tbp_ij].D * (exp1 - 2.0 * exp2);

                e_vdW_ += self_coef * (e_base * Tap);

                dfn13 = POW( r_ij, p_vdW1 - 1.0 )
                    * POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0 );
                de_base = (tbp[tbp_ij].D * tbp[tbp_ij].alpha / tbp[tbp_ij].r_vdW) * (exp2 - exp1) * dfn13;
            }
            /* no shielding */
            else
            {
                exp1 = EXP( tbp[tbp_ij].alpha * (1.0 - r_ij / tbp[tbp_ij].r_vdW) );
                exp2 = EXP( 0.5 * tbp[tbp_ij].alpha * (1.0 - r_ij / tbp[tbp_ij].r_vdW) );
                e_base = tbp[tbp_ij].D * (exp1 - 2.0 * exp2);

                e_vdW_ += self_coef * (e_base * Tap);

                de_base = (tbp[tbp_ij].D * tbp[tbp_ij].alpha / tbp[tbp_ij].r_vdW) * (exp2 - exp1);
            }

            /* calculate inner core repulsion */
            if ( gp.vdw_type == 2 || gp.vdw_type == 3 )
            {
                e_core = tbp[tbp_ij].ecore * EXP( tbp[tbp_ij].acore * (1.0 - (r_ij / tbp[tbp_ij].rcore)) );
                e_vdW_ += self_coef * (e_core * Tap);

                de_core = -(tbp[tbp_ij].acore / tbp[tbp_ij].rcore) * e_core;
            }
            else
            {
                e_core = 0.0;
                de_core = 0.0;
            }

            CEvd = self_coef * ( (de_base + de_core) * Tap
                    + (e_base + e_core) * dTap );

            /* Coulomb Calculations */
            dr3gamij_1 = r_ij * r_ij * r_ij
                + POW( tbp[tbp_ij].gamma, -3.0 );
            dr3gamij_3 = POW( dr3gamij_1, 1.0 / 3.0 );
            e_clb = C_ELE * (my_atoms[i].q * my_atoms[j].q) / dr3gamij_3;
            e_ele_ += self_coef * (e_clb * Tap);

            de_clb = -C_ELE * (my_atoms[i].q * my_atoms[j].q)
                    * (r_ij * r_ij) / POW( dr3gamij_1, 4.0 / 3.0 );
            CEclmb = self_coef * (de_clb * Tap + e_clb * dTap);

            /* for pressure coupling, terms not related to bond order 
               derivatives are added directly into pressure vector/tensor */
            rvec_Scale( temp,
                    -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_Add( f_i, temp );
            rvec_Scale( temp, -1.0, temp );
            atomic_rvecAdd( workspace.f[j], temp );

            rvec_iMultiply( temp,
                    far_nbr_list.far_nbr_list.rel_box[pj], temp );
            rvec_Add( ext_press_, temp );
        }

        pj += warpSize;
    }

    e_vdW_ = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(e_vdW_);
    e_ele_ = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(e_ele_);
    f_i[0] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_i[0]);
    f_i[1] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_i[1]);
    f_i[2] = cub::WarpReduce<double>(temp_storage[warp_id]).Sum(f_i[2]);

    /* first thread within a warp writes warp-level sum to global memory */
    if ( lane_id == 0 )
    {
        atomic_rvecAdd( workspace.f[i], f_i );
#if !defined(CUDA_ACCUM_ATOMIC)
        e_vdW_g[i] = e_vdW_;
        e_ele_g[i] = e_ele_;
        rvec_Copy( ext_press_g[j], ext_press_ );
#else
        atomicAdd( (double *) e_vdW_g, (double) e_vdW_ );
        atomicAdd( (double *) e_ele_g, (double) e_ele_ );
        atomic_rvecAdd( *ext_press_g, ext_press_ );
#endif
    }
}


/* one thread per atom implementation */
CUDA_GLOBAL void k_vdW_coulomb_energy_tab_full( reax_atom const * const my_atoms, 
        global_parameters gp, control_params const * const control, 
        storage workspace, reax_list far_nbr_list, 
        LR_lookup_table * const t_LR, int n, int num_atom_types, 
        real * const e_vdW_g, real * const e_ele_g, rvec * const ext_press_g )
{
    int i, j, pj, r;
    int type_i, type_j, tmin, tmax;
    int start_i, end_i, orig_i, orig_j;
    real r_ij, self_coef, base, dif;
    real e_vdW_, e_ele_;
    real CEvd, CEclmb;
    rvec temp, f_i, ext_press_;
    LR_lookup_table *t;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    e_ele_ = 0.0;
    e_vdW_ = 0.0;
    rvec_MakeZero( f_i );
    rvec_MakeZero( ext_press_ );

    type_i = my_atoms[i].type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    orig_i = my_atoms[i].orig_id;

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        orig_j = my_atoms[j].orig_id;

        if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut
                && orig_i < orig_j )
        {
            type_j = my_atoms[j].type;
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            self_coef = (i == j) ? 0.5 : 1.0;
            tmin = MIN( type_i, type_j );
            tmax = MAX( type_i, type_j );
            t = &t_LR[ index_lr(tmin, tmax, num_atom_types) ];

            /* Cubic Spline Interpolation */
            r = (int)(r_ij * t->inv_dx);
            if ( r == 0 )
            {
                ++r;
            }
            base = (real)(r + 1) * t->dx;
            dif = r_ij - base;

            e_vdW_ += self_coef * (((t->vdW[r].d * dif + t->vdW[r].c) * dif + t->vdW[r].b)
                * dif + t->vdW[r].a);

            e_ele_ += (((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b)
                * dif + t->ele[r].a) * self_coef * my_atoms[i].q * my_atoms[j].q;

            CEvd = ((t->CEvd[r].d * dif + t->CEvd[r].c) * dif + t->CEvd[r].b)
                * dif + t->CEvd[r].a;
            CEvd *= self_coef;

            CEclmb = ((t->CEclmb[r].d * dif + t->CEclmb[r].c) * dif + t->CEclmb[r].b)
                * dif + t->CEclmb[r].a;
            CEclmb *= self_coef * my_atoms[i].q * my_atoms[j].q;

            if ( control->virial == 0 )
            {
                rvec_ScaledAdd( temp,
                        -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                rvec_Add( f_i, temp );
                rvec_Scale( temp, -1.0, temp );
                atomic_rvecAdd( workspace.f[j], temp );
            }
            /* NPT, iNPT or sNPT */
            else
            {
                /* for pressure coupling, terms not related to bond order derivatives
                   are added directly into pressure vector/tensor */
                rvec_Scale( temp,
                        -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                rvec_Add( f_i, temp );
                rvec_ScaledAdd( temp, -1.0, temp );
                atomic_rvecAdd( workspace.f[j], temp );

                rvec_iMultiply( temp, far_nbr_list.far_nbr_list.rel_box[pj], temp );
                rvec_Add( ext_press_, temp );
            }
        }
    }

    atomic_rvecAdd( workspace.f[i], f_i );
#if !defined(CUDA_ACCUM_ATOMIC)
    __syncthreads( );
    e_vdW_g[i] = e_vdW_;
    e_ele_g[i] = e_ele_;
    if ( control->virial == 1 )
        rvec_Copy( ext_press_g[j], ext_press_ );
#else
    atomicAdd( (double *) e_vdW_g, (double) e_vdW_ );
    atomicAdd( (double *) e_ele_g, (double) e_ele_ );
    if ( control->virial == 1 )
        atomic_rvecAdd( *ext_press_g, ext_press_ );
#endif
}


static void Cuda_Compute_Polarization_Energy( reax_system const * const system,
        control_params const * const control, storage * const workspace,
        simulation_data * const data )
{
    int blocks;
#if !defined(CUDA_ACCUM_ATOMIC)
    real *spad;

    sCudaCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            sizeof(real) * system->n, __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[4];
#else
    sCudaMemsetAsync( &((simulation_data *)data->d_simulation_data)->my_en.e_pol,
            0, sizeof(real), control->streams[4], __FILE__, __LINE__ );
#endif

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_compute_polarization_energy <<< blocks, DEF_BLOCK_SIZE, 0,
                                  control->streams[4] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          system->n,
#if !defined(CUDA_ACCUM_ATOMIC)
          spad
#else
          &((simulation_data *)data->d_simulation_data)->my_en.e_pol
#endif
        );
    cudaCheckError( );

#if !defined(CUDA_ACCUM_ATOMIC)
    Cuda_Reduction_Sum( spad,
            &((simulation_data *)data->d_simulation_data)->my_en.e_pol,
            system->n, 4, control->streams[4] );
#endif
}


void Cuda_Compute_NonBonded_Forces( reax_system const * const system,
        control_params const * const control, simulation_data * const data,
        storage * const workspace, reax_list **lists,
        output_controls const * const out_control )
{
    int update_energy, blocks;
#if !defined(CUDA_ACCUM_ATOMIC)
    size_t s;
    real *spad;
    rvec *spad_rvec;
#endif

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->time_events[TE_VDW_COULOMB_START], control->streams[4] );
#endif

    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;

#if !defined(CUDA_ACCUM_ATOMIC)
    if ( control->virial == 1 )
    {
        s = (sizeof(real) * 2 + sizeof(rvec)) * system->n + sizeof(rvec) * control->blocks;
    }
    else
    {
        s = sizeof(real) * 2 * system->n;
    }
    sCudaCheckMalloc( &workspace->scratch[4], &workspace->scratch_size[4],
            s, __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[4];
#else
    sCudaMemsetAsync( &((simulation_data *)data->d_simulation_data)->my_en.e_vdW,
            0, sizeof(real), control->streams[4], __FILE__, __LINE__ );
    sCudaMemsetAsync( &((simulation_data *)data->d_simulation_data)->my_en.e_ele,
            0, sizeof(real), control->streams[4], __FILE__, __LINE__ );
    if ( control->virial == 1 )
    {
        sCudaMemsetAsync( &((simulation_data *)data->d_simulation_data)->my_ext_press,
                0, sizeof(rvec), control->streams[4], __FILE__, __LINE__ );
    }
#endif

    blocks = system->n * 32 / DEF_BLOCK_SIZE
        + (system->n * 32 % DEF_BLOCK_SIZE == 0 ? 0 : 1);

    if ( control->tabulate == 0 )
    {
        if ( control->virial == 1 )
        {
//            k_vdW_coulomb_energy_virial_full <<< control->blocks, control->block_size,
//                                             0, control->streams[4] >>>
//                ( system->d_my_atoms, system->reax_param.d_tbp, 
//                  system->reax_param.d_gp, (control_params *) control->d_control_params, 
//                  *(workspace->d_workspace), *(lists[FAR_NBRS]), 
//                  system->n, system->reax_param.num_atom_types, 
//#if !defined(CUDA_ACCUM_ATOMIC)
//                  spad, &spad[system->n], (rvec *) (&spad[2 * system->n])
//#else
//                  &((simulation_data *)data->d_simulation_data)->my_en.e_vdW,
//                  &((simulation_data *)data->d_simulation_data)->my_en.e_ele,
//                  &((simulation_data *)data->d_simulation_data)->my_ext_press
//#endif
//            );

            k_vdW_coulomb_energy_virial_full_opt <<< blocks, DEF_BLOCK_SIZE,
                                     sizeof(real) * (DEF_BLOCK_SIZE / 32),
                                     control->streams[4] >>>
                ( system->d_my_atoms, system->reax_param.d_tbp, 
                  system->reax_param.d_gp, (control_params *) control->d_control_params, 
                  *(workspace->d_workspace), *(lists[FAR_NBRS]), 
                  system->n, system->reax_param.num_atom_types, 
#if !defined(CUDA_ACCUM_ATOMIC)
                  spad, &spad[system->n], (rvec *) (&spad[2 * system->n])
#else
                  &((simulation_data *)data->d_simulation_data)->my_en.e_vdW,
                  &((simulation_data *)data->d_simulation_data)->my_en.e_ele,
                  &((simulation_data *)data->d_simulation_data)->my_ext_press
#endif
                );
        }
        else
        {
//            k_vdW_coulomb_energy_full <<< control->blocks, control->block_size,
//                                      0, control->streams[4] >>>
//                ( system->d_my_atoms, system->reax_param.d_tbp, 
//                  system->reax_param.d_gp, (control_params *) control->d_control_params, 
//                  *(workspace->d_workspace), *(lists[FAR_NBRS]), 
//                  system->n, system->reax_param.num_atom_types, 
//#if !defined(CUDA_ACCUM_ATOMIC)
//                  spad, &spad[system->n]
//#else
//                  &((simulation_data *)data->d_simulation_data)->my_en.e_vdW,
//                  &((simulation_data *)data->d_simulation_data)->my_en.e_ele
//#endif
//                );

            k_vdW_coulomb_energy_full_opt <<< blocks, DEF_BLOCK_SIZE,
                                     sizeof(real) * (DEF_BLOCK_SIZE / 32),
                                     control->streams[4] >>>
                ( system->d_my_atoms, system->reax_param.d_tbp, 
                  system->reax_param.d_gp, (control_params *) control->d_control_params, 
                  *(workspace->d_workspace), *(lists[FAR_NBRS]), 
                  system->n, system->reax_param.num_atom_types, 
#if !defined(CUDA_ACCUM_ATOMIC)
                  spad, &spad[system->n]
#else
                  &((simulation_data *)data->d_simulation_data)->my_en.e_vdW,
                  &((simulation_data *)data->d_simulation_data)->my_en.e_ele
#endif
                );
        }
        cudaCheckError( );
    }
    else
    {
        k_vdW_coulomb_energy_tab_full <<< control->blocks, control->block_size,
                                      0, control->streams[4] >>>
            ( system->d_my_atoms, system->reax_param.d_gp, 
              (control_params *) control->d_control_params, 
              *(workspace->d_workspace), *(lists[FAR_NBRS]), 
              workspace->d_LR, system->n, system->reax_param.num_atom_types, 
#if !defined(CUDA_ACCUM_ATOMIC)
              spad, &spad[system->n], (rvec *) (&spad[2 * system->n])
#else
              &((simulation_data *)data->d_simulation_data)->my_en.e_vdW,
              &((simulation_data *)data->d_simulation_data)->my_en.e_ele,
              &((simulation_data *)data->d_simulation_data)->my_ext_press
#endif
            );
        cudaCheckError( );
    }

#if !defined(CUDA_ACCUM_ATOMIC)
    if ( update_energy == TRUE )
    {
        /* reduction for vdw */
        Cuda_Reduction_Sum( spad,
                &((simulation_data *)data->d_simulation_data)->my_en.e_vdW,
                system->n, 4, control->streams[4] );

        /* reduction for ele */
        Cuda_Reduction_Sum( &spad[system->n],
                &((simulation_data *)data->d_simulation_data)->my_en.e_ele,
                system->n, 4, control->streams[4] );
    }

    if ( control->virial == 1 )
    {
        spad_rvec = (rvec *) (&spad[2 * system->n]);

        /* reduction for ext_press */
        k_reduction_rvec <<< control->blocks, control->block_size,
                         sizeof(rvec) * (control->block_size / 32),
                         control->streams[4] >>>
            ( spad_rvec, &spad_rvec[system->n], system->n );
        cudaCheckError( );

        k_reduction_rvec <<< 1, control->blocks_pow_2,
                         sizeof(rvec) * (control->blocks_pow_2 / 32),
                         control->streams[4] >>>
            ( &spad_rvec[system->n],
              &((simulation_data *)data->d_simulation_data)->my_ext_press,
              control->blocks );
        cudaCheckError( );
    }
#endif

    if ( update_energy == TRUE && control->polarization_energy_enabled == TRUE )
    {
        Cuda_Compute_Polarization_Energy( system, control, workspace, data );
    }

#if defined(LOG_PERFORMANCE)
    cudaEventRecord( control->time_events[TE_VDW_COULOMB_STOP], control->streams[4] );
#endif
}
