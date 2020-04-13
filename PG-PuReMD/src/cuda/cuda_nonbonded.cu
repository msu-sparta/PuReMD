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
#include "cuda_utils.h"
#include "cuda_reduction.h"

#include "../index_utils.h"
#include "../vector.h"


/* mask used to determine which threads within a warp participate in operations */
#define FULL_MASK (0xFFFFFFFF)


CUDA_GLOBAL void k_compute_polarization_energy( reax_atom *my_atoms, 
        single_body_parameters *sbp, int n, real *data_e_pol )
{
    int i, type_i;
    real q;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    q = my_atoms[i].q;
    type_i = my_atoms[i].type;

    data_e_pol[i] = KCALpMOL_to_EV * (sbp[type_i].chi * q
            + (sbp[type_i].eta / 2.0) * SQR(q));
}


/* one thread per atom implementation */
CUDA_GLOBAL void k_vdW_coulomb_energy( reax_atom *my_atoms, 
        two_body_parameters *tbp, global_parameters gp, control_params *control, 
        storage workspace, reax_list far_nbr_list, int n, int num_atom_types, 
        real *data_e_vdW, real *data_e_ele, rvec *data_ext_press )
{
    int i, j, pj;
    int start_i, end_i, orig_i, orig_j;
    real self_coef;
    real p_vdW1, p_vdW1i;
    real powr_vdW1, powgi_vdW1;
    real r_ij, fn13, exp1, exp2, e_base, de_base;
    real Tap, dTap, dfn13, CEvd, CEclmb;
    real dr3gamij_1, dr3gamij_3;
    real e_ele, e_vdW, e_core, de_core, e_clb, de_clb;
    rvec temp, ext_press;
    two_body_parameters *twbp;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    p_vdW1 = gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    e_vdW = 0.0;
    e_ele = 0.0;
    rvec_MakeZero( data_ext_press[i] );

    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    orig_i = my_atoms[i].orig_id;

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        orig_j = my_atoms[j].orig_id;

        //TODO: assuming far_nbr_list in FULL_LIST, add conditions for HALF_LIST
        if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut 
                && orig_i < orig_j )
        {
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            twbp = &tbp[
                index_tbp(my_atoms[i].type, my_atoms[j].type, num_atom_types) ];

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
                powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1 );

                fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
                exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
                e_base = twbp->D * (exp1 - 2.0 * exp2);

                e_vdW += self_coef * (e_base * Tap);

                dfn13 = POW( r_ij, p_vdW1 - 1.0 )
                    * POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0 );
                de_base = (twbp->D * twbp->alpha / twbp->r_vdW) * (exp2 - exp1) * dfn13;
            }
            /* no shielding */
            else
            {
                exp1 = EXP( twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
                exp2 = EXP( 0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
                e_base = twbp->D * (exp1 - 2.0 * exp2);

                e_vdW += self_coef * (e_base * Tap);

                de_base = (twbp->D * twbp->alpha / twbp->r_vdW) * (exp2 - exp1);
            }

            /* calculate inner core repulsion */
            if ( gp.vdw_type == 2 || gp.vdw_type == 3 )
            {
                e_core = twbp->ecore * EXP( twbp->acore * (1.0 - (r_ij / twbp->rcore)) );
                e_vdW += self_coef * (e_core * Tap);

                de_core = -(twbp->acore / twbp->rcore) * e_core;
            }
            else
            {
                e_core = 0.0;
                de_core = 0.0;
            }

            CEvd = self_coef * ( (de_base + de_core) * Tap
                    + (e_base + e_core) * dTap );

#if defined(DEBUG_FOCUS)
            printf( "%6d%6d%24.12f%24.12f%24.12f%24.12f\n",
                    i + 1, j + 1, 
                    e_base, de_base, e_core, de_core );
#endif

            /* Coulomb Calculations */
            dr3gamij_1 = r_ij * r_ij * r_ij
                + POW( twbp->gamma, -3.0 );
            dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );
            e_clb = C_ELE * (my_atoms[i].q * my_atoms[j].q) / dr3gamij_3;
            e_ele += self_coef * (e_clb * Tap);

            de_clb = -C_ELE * (my_atoms[i].q * my_atoms[j].q)
                    * (r_ij * r_ij) / POW( dr3gamij_1, 4.0 / 3.0 );
            CEclmb = self_coef * (de_clb * Tap + e_clb * dTap);

#if defined(DEBUG_FOCUS)
            printf( "%6d%6d%24.12f%24.12f\n",
                    i + 1, j + 1, e_clb, de_clb );
#endif

            if ( control->virial == 0 )
            {
                if ( i < j ) 
                {
                    rvec_Scale( temp, -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                    atomic_rvecAdd( workspace.f[i], temp );
                    rvec_Scale( temp, -1.0, temp );
                    atomic_rvecAdd( workspace.f[j], temp );
                }
                else 
                {
                    rvec_Scale( temp, (CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                    atomic_rvecAdd( workspace.f[i], temp );
                    rvec_Scale( temp, -1.0, temp );
                    atomic_rvecAdd( workspace.f[j], temp );
                }
            }
            /* NPT, iNPT or sNPT */
            else
            {
                /* for pressure coupling, terms not related to bond order 
                   derivatives are added directly into pressure vector/tensor */
                if ( i < j ) 
                {
                    rvec_Scale( temp,
                            -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                    atomic_rvecAdd( workspace.f[i], temp );
                    rvec_Scale( temp, -1.0, temp );
                    atomic_rvecAdd( workspace.f[j], temp );
                }
                else 
                {
                    rvec_Scale( temp,
                            (CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                    atomic_rvecAdd( workspace.f[i], temp );
                    rvec_Scale( temp, -1.0, temp );
                    atomic_rvecAdd( workspace.f[j], temp );
                }

                rvec_iMultiply( ext_press,
                        far_nbr_list.far_nbr_list.rel_box[pj], temp );
                rvec_Add( data_ext_press[i], ext_press );
            }

#if defined(TEST_ENERGY)
            // fprintf( out_control->evdw, 
            // "%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f\n", 
            // workspace.Tap[7], workspace.Tap[6], workspace.Tap[5],
            // workspace.Tap[4], workspace.Tap[3], workspace.Tap[2], 
            // workspace.Tap[1], Tap );
            //fprintf( out_control->evdw, "%6d%6d%24.15e%24.15e%24.15e\n",
            fprintf( out_control->evdw, "%6d%6d%12.4f%12.4f%12.4f\n",
                    my_atoms[i].orig_id, my_atoms[j].orig_id, 
                    r_ij, e_vdW, data->my_en.e_vdW );
            //fprintf(out_control->ecou,"%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
            fprintf( out_control->ecou, "%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                    my_atoms[i].orig_id, my_atoms[j].orig_id,
                    r_ij, my_atoms[i].q, my_atoms[j].q, 
                    e_ele, data->my_en.e_ele );
#endif

#if defined(TEST_FORCES)
            rvec_ScaledAdd( workspace.f_vdw[i], -CEvd,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_vdw[j], +CEvd,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_ele[i], -CEclmb,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_ele[j], +CEclmb,
                    far_nbr_list.far_nbr_list.dvec[pj] );
#endif
        }
    }

    __syncthreads( );
    data_e_vdW[i] = e_vdW;
    data_e_ele[i] = e_ele;
}


//TODO: fix issue with atomic forces not being correctly accumulated
/* one warp of threads (32) per atom implementation */
CUDA_GLOBAL void k_vdW_coulomb_energy_opt( reax_atom *my_atoms, 
        two_body_parameters *tbp, global_parameters gp, control_params *control, 
        storage workspace, reax_list far_nbr_list, int n, int num_atom_types, 
        real *data_e_vdW, real *data_e_ele, rvec *data_ext_press )
{
    int i, j, pj;
    int start_i, end_i, orig_i, orig_j;
    real self_coef;
    real p_vdW1, p_vdW1i;
    real powr_vdW1, powgi_vdW1;
    real r_ij, fn13, exp1, exp2, e_base, de_base;
    real Tap, dTap, dfn13, CEvd, CEclmb;
    real dr3gamij_1, dr3gamij_3;
    real e_ele, e_vdW, e_core, de_core, e_clb, de_clb;
    rvec temp, ext_press;
    two_body_parameters *twbp;
    int thread_id, warp_id, lane_id, offset;
    unsigned int mask;
    real e_vdW_s, e_ele_s;
    rvec f_s;

    thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    warp_id = thread_id >> 5;
    lane_id = thread_id & 0x0000001F; 
    mask = __ballot_sync( FULL_MASK, i < n );

    if ( warp_id >= n )
    {
        return;
    }

    i = warp_id;
    e_vdW_s = 0.0;
    e_ele_s = 0.0;
    rvec_MakeZero( f_s );

    p_vdW1 = gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;

    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    orig_i = my_atoms[i].orig_id;

    pj = start_i + lane_id;
    while ( pj < end_i )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        orig_j = my_atoms[j].orig_id;

        //TODO: assuming far_nbr_list in FULL_LIST, add conditions for HALF_LIST
        if ( far_nbr_list.far_nbr_list.d[pj] <= control->nonb_cut 
                && orig_i < orig_j )
        {
            r_ij = far_nbr_list.far_nbr_list.d[pj];
            twbp = &tbp[
                index_tbp(my_atoms[i].type, my_atoms[j].type, num_atom_types) ];

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
                powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1 );

                fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
                exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
                e_base = twbp->D * (exp1 - 2.0 * exp2);

                e_vdW = self_coef * (e_base * Tap);
                e_vdW_s += e_vdW;

                dfn13 = POW( r_ij, p_vdW1 - 1.0 )
                    * POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0 );
                de_base = (twbp->D * twbp->alpha / twbp->r_vdW) * (exp2 - exp1) * dfn13;
            }
            /* no shielding */
            else
            {
                exp1 = EXP( twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
                exp2 = EXP( 0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
                e_base = twbp->D * (exp1 - 2.0 * exp2);

                e_vdW = self_coef * (e_base * Tap);
                e_vdW_s += e_vdW;

                de_base = (twbp->D * twbp->alpha / twbp->r_vdW) * (exp2 - exp1);
            }

            /* calculate inner core repulsion */
            if ( gp.vdw_type == 2 || gp.vdw_type == 3 )
            {
                e_core = twbp->ecore * EXP( twbp->acore * (1.0 - (r_ij / twbp->rcore)) );
                e_vdW += self_coef * (e_core * Tap);
                e_vdW_s += (self_coef * (e_core * Tap));

                de_core = -(twbp->acore / twbp->rcore) * e_core;
            }
            else
            {
                e_core = 0.0;
                de_core = 0.0;
            }

            CEvd = self_coef * ( (de_base + de_core) * Tap
                    + (e_base + e_core) * dTap );

#if defined(DEBUG_FOCUS)
            if ( i < j )
            {
                printf( "%6d%6d%24.12f%24.12f%24.12f%24.12f\n",
                        i + 1, j + 1, 
                        e_base, de_base, e_core, de_core );
            }
            else
            {
                printf( "%6d%6d%24.12f%24.12f%24.12f%24.12f\n",
                        j + 1, i + 1, 
                        e_base, de_base, e_core, de_core );
            }
#endif

            /* Coulomb Calculations */
            dr3gamij_1 = r_ij * r_ij * r_ij
                + POW( twbp->gamma, -3.0 );
            dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );
            e_clb = C_ELE * (my_atoms[i].q * my_atoms[j].q) / dr3gamij_3;
            e_ele = self_coef * (e_clb * Tap);
            e_ele_s += e_ele;

            de_clb = -C_ELE * (my_atoms[i].q * my_atoms[j].q)
                    * (r_ij * r_ij) / POW( dr3gamij_1, 4.0 / 3.0 );
            CEclmb = self_coef * (de_clb * Tap + e_clb * dTap);

#if defined(DEBUG_FOCUS)
            printf( "%6d%6d%24.12f%24.12f\n",
                    i + 1, j + 1, e_clb, de_clb );
#endif

            if ( control->virial == 0 )
            {
                if ( i < j ) 
                {
                    rvec_ScaledAdd( f_s,
                            -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                }
                else 
                {
                    rvec_ScaledAdd( f_s,
                            (CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                }
            }
            /* NPT, iNPT or sNPT */
            else
            {
                /* for pressure coupling, terms not related to bond order 
                   derivatives are added directly into pressure vector/tensor */
                rvec_Scale( temp,
                        (CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );

                if ( i < j ) 
                {
                    rvec_ScaledAdd( f_s, -1.0, temp );
                }
                else 
                {
                    rvec_Add( f_s, temp );
                }

                rvec_iMultiply( ext_press,
                        far_nbr_list.far_nbr_list.rel_box[pj], temp );
                rvec_Add( data_ext_press[i], ext_press );
            }

#if defined(TEST_ENERGY)
            // fprintf( out_control->evdw, 
            // "%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f\n", 
            // workspace.Tap[7], workspace.Tap[6], workspace.Tap[5],
            // workspace.Tap[4], workspace.Tap[3], workspace.Tap[2], 
            // workspace.Tap[1], Tap );
            //fprintf( out_control->evdw, "%6d%6d%24.15e%24.15e%24.15e\n",
            fprintf( out_control->evdw, "%6d%6d%12.4f%12.4f%12.4f\n",
                    my_atoms[i].orig_id, my_atoms[j].orig_id, 
                    r_ij, e_vdW, data->my_en.e_vdW );
            //fprintf(out_control->ecou,"%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
            fprintf( out_control->ecou, "%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                    my_atoms[i].orig_id, my_atoms[j].orig_id,
                    r_ij, my_atoms[i].q, my_atoms[j].q, 
                    e_ele, data->my_en.e_ele );
#endif

#if defined(TEST_FORCES)
            rvec_ScaledAdd( workspace.f_vdw[i], -CEvd,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_vdw[j], +CEvd,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_ele[i], -CEclmb,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_ele[j], +CEclmb,
                    far_nbr_list.far_nbr_list.dvec[pj] );
#endif
        }

        pj += 32;
    }

    /* warp-level sum using registers within a warp */
    for ( offset = 16; offset > 0; offset /= 2 )
    {
        e_vdW_s += __shfl_down_sync( mask, e_vdW_s, offset );
        e_ele_s += __shfl_down_sync( mask, e_ele_s, offset );
        f_s[0] += __shfl_down_sync( mask, f_s[0], offset );
        f_s[1] += __shfl_down_sync( mask, f_s[1], offset );
        f_s[2] += __shfl_down_sync( mask, f_s[2], offset );
    }

    /* first thread within a warp writes warp-level sum to global memory */
    if ( lane_id == 0 )
    {
        data_e_vdW[i] = e_vdW_s;
        data_e_ele[i] = e_ele_s;
        rvec_Add( workspace.f[i], f_s );
    }
}


/* one thread per atom implementation */
CUDA_GLOBAL void k_tabulated_vdW_coulomb_energy( reax_atom *my_atoms, 
        global_parameters gp, control_params *control, 
        storage workspace, reax_list far_nbr_list, 
        LR_lookup_table *t_LR, int n, int num_atom_types, 
        int step, int prev_steps, int energy_update_freq, 
        real *data_e_vdW, real *data_e_ele, rvec *data_ext_press )
{
    int i, j, pj, r, steps, update_freq, update_energies;
    int type_i, type_j, tmin, tmax;
    int start_i, end_i, orig_i, orig_j;
    real r_ij, self_coef, base, dif;
    real e_vdW, e_ele;
    real CEvd, CEclmb;
    rvec temp, ext_press;
    LR_lookup_table *t;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    steps = step - prev_steps;
    update_freq = energy_update_freq;
    update_energies = update_freq > 0 && steps % update_freq == 0;
    e_ele = 0.0;
    e_vdW = 0.0;
    data_e_vdW[i] = 0.0;
    data_e_ele[i] = 0.0;

    type_i = my_atoms[i].type;
    start_i = Start_Index( i, &far_nbr_list );
    end_i = End_Index( i, &far_nbr_list );
    orig_i = my_atoms[i].orig_id;

    for ( pj = start_i; pj < end_i; ++pj )
    {
        j = far_nbr_list.far_nbr_list.nbr[pj];
        orig_j = my_atoms[j].orig_id;

        //TODO: assuming far_nbr_list in FULL_LIST, add conditions for HALF_LIST
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

            if ( update_energies )
            {
                e_vdW = ((t->vdW[r].d * dif + t->vdW[r].c) * dif + t->vdW[r].b)
                    * dif + t->vdW[r].a;
                e_vdW *= self_coef;

                e_ele = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b)
                    * dif + t->ele[r].a;
                e_ele *= self_coef * my_atoms[i].q * my_atoms[j].q;

                data_e_vdW[i] += e_vdW;
                data_e_ele[i] += e_ele;
            }    

            CEvd = ((t->CEvd[r].d * dif + t->CEvd[r].c) * dif + t->CEvd[r].b)
                * dif + t->CEvd[r].a;
            CEvd *= self_coef;

            CEclmb = ((t->CEclmb[r].d * dif + t->CEclmb[r].c) * dif + t->CEclmb[r].b)
                * dif + t->CEclmb[r].a;
            CEclmb *= self_coef * my_atoms[i].q * my_atoms[j].q;

            if ( control->virial == 0 )
            {
                if ( i < j ) 
                {
                    rvec_ScaledAdd( workspace.f[i],
                            -(CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                }
                else 
                {
                    rvec_ScaledAdd( workspace.f[i],
                            (CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );
                }
            }
            /* NPT, iNPT or sNPT */
            else
            {
                /* for pressure coupling, terms not related to bond order derivatives
                   are added directly into pressure vector/tensor */
                rvec_Scale( temp,
                        (CEvd + CEclmb) / r_ij, far_nbr_list.far_nbr_list.dvec[pj] );

                rvec_ScaledAdd( workspace.f[i], -1.0, temp );
                rvec_Add( workspace.f[j], temp );

                rvec_iMultiply( ext_press, far_nbr_list.far_nbr_list.rel_box[pj], temp );
                rvec_Add( data_ext_press[i], ext_press );
            }

#if defined(TEST_ENERGY)
            //fprintf( out_control->evdw, "%6d%6d%24.15e%24.15e%24.15e\n",
            fprintf( out_control->evdw, "%6d%6d%12.4f%12.4f%12.4f\n",
                    my_atoms[i].orig_id, my_atoms[j].orig_id, 
                    r_ij, e_vdW, data->my_en.e_vdW );
            //fprintf(out_control->ecou,"%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
            fprintf( out_control->ecou, "%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                    my_atoms[i].orig_id, my_atoms[j].orig_id,
                    r_ij, my_atoms[i].q, my_atoms[j].q, 
                    e_ele, data->my_en.e_ele );
#endif

#if defined(TEST_FORCES)
            rvec_ScaledAdd( workspace.f_vdw[i], -CEvd,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_vdw[j], +CEvd,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_ele[i], -CEclmb,
                    far_nbr_list.far_nbr_list.dvec[pj] );
            rvec_ScaledAdd( workspace.f_ele[j], +CEclmb,
                    far_nbr_list.far_nbr_list.dvec[pj] );
#endif
        }
    }
}


static void Cuda_Compute_Polarization_Energy( reax_system *system, storage *workspace,
        simulation_data *data )
{
    int blocks;
    real *spad;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    spad = (real *) workspace->scratch;

    k_compute_polarization_energy <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
          system->n, spad );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    Cuda_Reduction_Sum( spad,
            &((simulation_data *)data->d_simulation_data)->my_en.e_pol,
            system->n );
}


void Cuda_NonBonded_Energy( reax_system *system, control_params *control, 
        storage *workspace, simulation_data *data, reax_list **lists,
        output_controls *out_control )
{
    int blocks, rblocks, update_energy;
    rvec *spad_rvec;
    real *spad;

    spad = (real *) workspace->scratch;
    update_energy = (out_control->energy_update_freq > 0
            && data->step % out_control->energy_update_freq == 0) ? TRUE : FALSE;
    rblocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    blocks = (system->n * VDW_KER_THREADS_PER_ATOM / DEF_BLOCK_SIZE) 
        + ((system->n * VDW_KER_THREADS_PER_ATOM % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    if ( control->tabulate == 0 )
    {
        k_vdW_coulomb_energy <<< rblocks, DEF_BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_tbp, 
              system->reax_param.d_gp, (control_params *)control->d_control_params, 
              *(workspace->d_workspace), *(lists[FAR_NBRS]), 
              system->n, system->reax_param.num_atom_types, 
              spad, &spad[system->n], (rvec *)(&spad[2 * system->n]) );
//        k_vdW_coulomb_energy_opt <<< blocks, DEF_BLOCK_SIZE,
//                             (2 * sizeof(real) + sizeof(rvec)) * DEF_BLOCK_SIZE >>>
//            ( system->d_my_atoms, system->reax_param.d_tbp, 
//              system->reax_param.d_gp, (control_params *)control->d_control_params, 
//              *(workspace->d_workspace), *(lists[FAR_NBRS]), 
//              system->n, system->reax_param.num_atom_types, 
//              spad, &spad[system->n], (rvec *)(&spad[2 * system->n]) );
        cudaDeviceSynchronize( );
        cudaCheckError( );
    }
    else
    {
        k_tabulated_vdW_coulomb_energy <<< blocks, DEF_BLOCK_SIZE >>>
            ( system->d_my_atoms, system->reax_param.d_gp, 
              (control_params *)control->d_control_params, 
              *(workspace->d_workspace), *(lists[FAR_NBRS]), 
              workspace->d_LR, system->n,
              system->reax_param.num_atom_types, 
              data->step, data->prev_steps, 
              out_control->energy_update_freq,
              spad, &spad[system->n], (rvec *)(&spad[2 * system->n]));
        cudaDeviceSynchronize( );
        cudaCheckError( );
    }

    if ( update_energy == TRUE )
    {
        /* reduction for vdw */
        Cuda_Reduction_Sum( spad, &((simulation_data *)data->d_simulation_data)->my_en.e_vdW,
                system->n );

        /* reduction for ele */
        Cuda_Reduction_Sum( &spad[system->n], &((simulation_data *)data->d_simulation_data)->my_en.e_ele,
                system->n );
    }

    /* reduction for ext_press */
    spad_rvec = (rvec *) (&spad[2 * system->n]);
    k_reduction_rvec <<< rblocks, DEF_BLOCK_SIZE, sizeof(rvec) * DEF_BLOCK_SIZE >>>
        ( spad_rvec, &spad_rvec[system->n], system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_reduction_rvec <<< 1, control->blocks_pow_2_n, sizeof(rvec) * control->blocks_pow_2_n>>>
        ( &spad_rvec[system->n], &((simulation_data *)data->d_simulation_data)->my_ext_press, rblocks );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    if ( update_energy == TRUE )
    {
        Cuda_Compute_Polarization_Energy( system, workspace, data );
    }
}
