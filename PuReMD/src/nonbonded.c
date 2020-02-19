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

#include "reax_types.h"
#if defined(PURE_REAX)
#include "nonbonded.h"
#include "bond_orders.h"
#include "list.h"
#include "vector.h"
#elif defined(LAMMPS_REAX)
#include "reax_nonbonded.h"
#include "reax_bond_orders.h"
#include "reax_list.h"
#include "reax_vector.h"
#endif


static void Compute_Polarization_Energy( reax_system const * const system,
        simulation_data * const data )
{
    int i, type_i;
    real q;

    data->my_en.e_pol = 0.0;

    for ( i = 0; i < system->n; i++ )
    {
        q = system->my_atoms[i].q;
        type_i = system->my_atoms[i].type;

        data->my_en.e_pol += KCALpMOL_to_EV * (system->reax_param.sbp[type_i].chi * q
                + (system->reax_param.sbp[type_i].eta / 2.0) * SQR(q));
    }
}


void vdW_Coulomb_Energy( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control )
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
    reax_list *far_nbrs;

    far_nbrs = lists[FAR_NBRS];
    p_vdW1 = system->reax_param.gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    e_core = 0.0;
    e_vdW = 0.0;

    for ( i = 0; i < system->n; ++i )
    {
        start_i = Start_Index( i, far_nbrs );
        end_i = End_Index( i, far_nbrs );
        orig_i = system->my_atoms[i].orig_id;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbrs->far_nbr_list.nbr[pj];
            orig_j = system->my_atoms[j].orig_id;

            if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut
                    && ((far_nbrs->format == HALF_LIST && (j < system->n || orig_i < orig_j))
                        || (far_nbrs->format == FULL_LIST && orig_i < orig_j)) )
            {
                r_ij = far_nbrs->far_nbr_list.d[pj];
                twbp = &system->reax_param.tbp[ 
                    system->my_atoms[i].type ][ system->my_atoms[j].type ];
                /* i == j: self-interaction from periodic image,
                 * important for supporting small boxes! */
                self_coef = (orig_i == orig_j) ? 0.5 : 1.0;

                /* Calculate Taper and its derivative */
                Tap = workspace->Tap[7] * r_ij
                    + workspace->Tap[6];
                Tap = Tap * r_ij + workspace->Tap[5];
                Tap = Tap * r_ij + workspace->Tap[4];
                Tap = Tap * r_ij + workspace->Tap[3];
                Tap = Tap * r_ij + workspace->Tap[2];
                Tap = Tap * r_ij + workspace->Tap[1];
                Tap = Tap * r_ij + workspace->Tap[0];

                dTap = 7.0 * workspace->Tap[7] * r_ij
                    + 6.0 * workspace->Tap[6];
                dTap = dTap * r_ij + 5.0 * workspace->Tap[5];
                dTap = dTap * r_ij + 4.0 * workspace->Tap[4];
                dTap = dTap * r_ij + 3.0 * workspace->Tap[3];
                dTap = dTap * r_ij + 2.0 * workspace->Tap[2];
                dTap = dTap * r_ij + workspace->Tap[1];

                /* vdWaals Calculations */
                if ( system->reax_param.gp.vdw_type == 1
                        || system->reax_param.gp.vdw_type == 3 )
                {
                    /* shielding */
                    powr_vdW1 = POW( r_ij, p_vdW1 );
                    powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1 );

                    fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                    exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
                    exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
                    e_base = twbp->D * (exp1 - 2.0 * exp2);

                    e_vdW = self_coef * (e_base * Tap);
                    data->my_en.e_vdW += e_vdW;

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
                    data->my_en.e_vdW += e_vdW;

                    de_base = (twbp->D * twbp->alpha / twbp->r_vdW) * (exp2 - exp1);
                }

                /* calculate inner core repulsion */
                if ( system->reax_param.gp.vdw_type == 2 || system->reax_param.gp.vdw_type == 3 )
                {
                    e_core = twbp->ecore * EXP( twbp->acore * (1.0 - (r_ij / twbp->rcore)) );
                    e_vdW += self_coef * (e_core * Tap);
                    data->my_en.e_vdW += self_coef * (e_core * Tap);

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
                fprintf( stderr, "%6d%6d%24.12f%24.12f%24.12f%24.12f\n",
                        i + 1, j + 1, 
                        e_base, de_base, e_core, de_core ); fflush( stderr );
#endif

                /* Coulomb Calculations */
                dr3gamij_1 = r_ij * r_ij * r_ij
                    + POW( twbp->gamma, -3.0 );
                dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );
                e_clb = C_ELE * (system->my_atoms[i].q * system->my_atoms[j].q) / dr3gamij_3;
                e_ele = self_coef * (e_clb * Tap);
                data->my_en.e_ele += e_ele;

                de_clb = -C_ELE * (system->my_atoms[i].q * system->my_atoms[j].q)
                        * (r_ij * r_ij) / POW( dr3gamij_1, 4.0 / 3.0);
                CEclmb = self_coef * (de_clb * Tap + e_clb * dTap);

#if defined(DEBUG_FOCUS)
                fprintf( stderr, "%6d%6d%24.12f%24.12f\n",
                        i + 1, j + 1, e_clb, de_clb ); fflush( stderr );
#endif

                if ( control->virial == 0 )
                {
                    rvec_ScaledAdd( workspace->f[i],
                            -(CEvd + CEclmb) / r_ij, far_nbrs->far_nbr_list.dvec[pj] );
                    rvec_ScaledAdd( workspace->f[j],
                            (CEvd + CEclmb) / r_ij, far_nbrs->far_nbr_list.dvec[pj] );
                }
                else   /* NPT, iNPT or sNPT */
                {
                    /* for pressure coupling, terms not related to bond order
                       derivatives are added directly into pressure vector/tensor */
                    rvec_Scale( temp,
                            (CEvd + CEclmb) / r_ij, far_nbrs->far_nbr_list.dvec[pj] );

                    rvec_ScaledAdd( workspace->f[i], -1., temp );
                    rvec_Add( workspace->f[j], temp );

                    rvec_iMultiply( ext_press,
                            far_nbrs->far_nbr_list.rel_box[pj], temp );
                    rvec_Add( data->my_ext_press, ext_press );

                    // fprintf( stderr, "nonbonded(%d,%d): rel_box (%f %f %f)
                    //   force(%f %f %f) ext_press (%12.6f %12.6f %12.6f)\n",
                    //   i, j, far_nbrs->far_nbr_list.rel_box[pj][0],
                    //   far_nbrs->far_nbr_list.rel_box[pj][1],
                    //   far_nbrs->far_nbr_list.rel_box[pj][2],
                    //   temp[0], temp[1], temp[2],
                    //   data->ext_press[0], data->ext_press[1], data->ext_press[2] );
                }

#if defined(TEST_ENERGY)
                // fprintf( out_control->evdw,
                // "%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f%12.9f\n",
                // workspace->Tap[7],workspace->Tap[6],workspace->Tap[5],
                // workspace->Tap[4],workspace->Tap[3],workspace->Tap[2],
                // workspace->Tap[1], Tap );
                //fprintf( out_control->evdw, "%6d%6d%24.15e%24.15e%24.15e\n",
                fprintf( out_control->evdw, "%6d%6d%12.4f%12.4f%12.4f\n",
                         system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                         r_ij, e_vdW, data->my_en.e_vdW );
                //fprintf(out_control->ecou,"%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                fprintf( out_control->ecou, "%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                         system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                         r_ij, system->my_atoms[i].q, system->my_atoms[j].q,
                         e_ele, data->my_en.e_ele );
#endif

#if defined(TEST_FORCES)
                rvec_ScaledAdd( workspace->f_vdw[i], -CEvd,
                        far_nbrs->far_nbr_list.dvec[pj] );
                rvec_ScaledAdd( workspace->f_vdw[j], +CEvd,
                        far_nbrs->far_nbr_list.dvec[pj] );
                rvec_ScaledAdd( workspace->f_ele[i], -CEclmb,
                        far_nbrs->far_nbr_list.dvec[pj] );
                rvec_ScaledAdd( workspace->f_ele[j], +CEclmb,
                        far_nbrs->far_nbr_list.dvec[pj] );
#endif
            }
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "nonbonded: ext_press (%12.6f %12.6f %12.6f)\n",
             data->ext_press[0], data->ext_press[1], data->ext_press[2] );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    Compute_Polarization_Energy( system, data );
}



void Tabulated_vdW_Coulomb_Energy( reax_system const * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control )
{
    int i, j, pj, r, steps, update_freq, update_energies;
    int type_i, type_j, tmin, tmax;
    int start_i, end_i, orig_i, orig_j;
    real r_ij, self_coef, base, dif;
    real e_vdW, e_ele;
    real CEvd, CEclmb;
    rvec temp, ext_press;
    reax_list *far_nbrs;
    LR_lookup_table *t;

    far_nbrs = lists[FAR_NBRS];
    steps = data->step - data->prev_steps;
    update_freq = out_control->energy_update_freq;
    update_energies = update_freq > 0 && steps % update_freq == 0;
    e_ele = 0.0;
    e_vdW = 0.0;

    for ( i = 0; i < system->n; ++i )
    {
        type_i = system->my_atoms[i].type;
        start_i = Start_Index(i, far_nbrs);
        end_i = End_Index(i, far_nbrs);
        orig_i = system->my_atoms[i].orig_id;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            j = far_nbrs->far_nbr_list.nbr[pj];
            orig_j = system->my_atoms[j].orig_id;

            if ( far_nbrs->far_nbr_list.d[pj] <= control->nonb_cut
                    && ((far_nbrs->format == HALF_LIST && (j < system->n || orig_i < orig_j))
                        || (far_nbrs->format == FULL_LIST && orig_i < orig_j)) )
            {
                type_j = system->my_atoms[j].type;
                r_ij = far_nbrs->far_nbr_list.d[pj];
                self_coef = (i == j) ? 0.5 : 1.0;
                tmin = MIN( type_i, type_j );
                tmax = MAX( type_i, type_j );
                t = &workspace->LR[tmin][tmax];

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
                    e_ele *= self_coef * system->my_atoms[i].q * system->my_atoms[j].q;

                    data->my_en.e_vdW += e_vdW;
                    data->my_en.e_ele += e_ele;
                }

                CEvd = ((t->CEvd[r].d * dif + t->CEvd[r].c) * dif + t->CEvd[r].b)
                    * dif + t->CEvd[r].a;
                CEvd *= self_coef;

                CEclmb = ((t->CEclmb[r].d * dif + t->CEclmb[r].c) * dif + t->CEclmb[r].b)
                    * dif + t->CEclmb[r].a;
                CEclmb *= self_coef * system->my_atoms[i].q * system->my_atoms[j].q;

                if ( control->virial == 0 )
                {
                    rvec_ScaledAdd( workspace->f[i],
                            -(CEvd + CEclmb) / r_ij, far_nbrs->far_nbr_list.dvec[pj] );
                    rvec_ScaledAdd( workspace->f[j],
                            (CEvd + CEclmb) / r_ij, far_nbrs->far_nbr_list.dvec[pj] );
                }
                /* NPT, iNPT or sNPT */
                else
                {
                    /* for pressure coupling, terms not related to bond order derivatives
                       are added directly into pressure vector/tensor */
                    rvec_Scale( temp,
                            (CEvd + CEclmb) / r_ij, far_nbrs->far_nbr_list.dvec[pj] );

                    rvec_ScaledAdd( workspace->f[i], -1.0, temp );
                    rvec_Add( workspace->f[j], temp );

                    rvec_iMultiply( ext_press, far_nbrs->far_nbr_list.rel_box[pj], temp );
                    rvec_Add( data->my_ext_press, ext_press );
                }

#if defined(TEST_ENERGY)
                //fprintf( out_control->evdw, "%6d%6d%24.15e%24.15e%24.15e\n",
                fprintf( out_control->evdw, "%6d%6d%12.4f%12.4f%12.4f\n",
                         system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                         r_ij, e_vdW, data->my_en.e_vdW );
                //fprintf(out_control->ecou,"%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                fprintf( out_control->ecou, "%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                         system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                         r_ij, system->my_atoms[i].q, system->my_atoms[j].q,
                         e_ele, data->my_en.e_ele );
#endif

#if defined(TEST_FORCES)
                rvec_ScaledAdd( workspace->f_vdw[i], -CEvd,
                        far_nbrs->far_nbr_list.dvec[pj] );
                rvec_ScaledAdd( workspace->f_vdw[j], +CEvd,
                        far_nbrs->far_nbr_list.dvec[pj] );
                rvec_ScaledAdd( workspace->f_ele[i], -CEclmb,
                        far_nbrs->far_nbr_list.dvec[pj] );
                rvec_ScaledAdd( workspace->f_ele[j], +CEclmb,
                        far_nbrs->far_nbr_list.dvec[pj] );
#endif
            }
        }
    }

    Compute_Polarization_Energy( system, data );
}


void LR_vdW_Coulomb( reax_system *system, storage *workspace,
                     int i, int j, real r_ij, LR_data *lr )
{
    real p_vdW1, p_vdW1i;
    real powr_vdW1, powgi_vdW1;
    real tmp, fn13, exp1, exp2, e_base, de_base;
    real Tap, dTap, dfn13;
    real dr3gamij_1, dr3gamij_3;
    real e_core, de_core;
    two_body_parameters *twbp;

    p_vdW1 = system->reax_param.gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    twbp = &system->reax_param.tbp[i][j];
    e_core = 0.0;
    de_core = 0.0;

    /* Calculate Taper and its derivative */
    Tap = workspace->Tap[7] * r_ij
        + workspace->Tap[6];
    Tap = Tap * r_ij + workspace->Tap[5];
    Tap = Tap * r_ij + workspace->Tap[4];
    Tap = Tap * r_ij + workspace->Tap[3];
    Tap = Tap * r_ij + workspace->Tap[2];
    Tap = Tap * r_ij + workspace->Tap[1];
    Tap = Tap * r_ij + workspace->Tap[0];

    dTap = 7.0 * workspace->Tap[7] * r_ij
        + 6.0 * workspace->Tap[6];
    dTap = dTap * r_ij + 5.0 * workspace->Tap[5];
    dTap = dTap * r_ij + 4.0 * workspace->Tap[4];
    dTap = dTap * r_ij + 3.0 * workspace->Tap[3];
    dTap = dTap * r_ij + 2.0 * workspace->Tap[2];
    dTap = dTap * r_ij + workspace->Tap[1];

    /* vdWaals Calculations */
    if ( system->reax_param.gp.vdw_type == 1
            || system->reax_param.gp.vdw_type == 3 )
    {
        /* shielding */
        powr_vdW1 = POW( r_ij, p_vdW1 );
        powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1 );

        fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
        exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
        exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
        e_base = twbp->D * (exp1 - 2.0 * exp2);

        lr->e_vdW = e_base * Tap;

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

        lr->e_vdW = e_base * Tap;

        de_base = (twbp->D * twbp->alpha / twbp->r_vdW) * (exp2 - exp1);
    }

    /* calculate inner core repulsion */
    if ( system->reax_param.gp.vdw_type == 2 || system->reax_param.gp.vdw_type == 3 )
    {
        e_core = twbp->ecore * EXP( twbp->acore * (1.0 - (r_ij / twbp->rcore)) );
        lr->e_vdW += e_core * Tap;

        de_core = -(twbp->acore / twbp->rcore) * e_core;
    }
    else
    {
        e_core = 0.0;
        de_core = 0.0;
    }

    lr->CEvd = (de_base + de_core) * Tap
            + (e_base + e_core) * dTap;

    /* Coulomb calculations */
    dr3gamij_1 = r_ij * r_ij * r_ij
        + POW( twbp->gamma, -3.0 );
    dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

    tmp = Tap / dr3gamij_3;
    lr->H = EV_to_KCALpMOL * tmp;
    lr->e_ele = C_ELE * tmp;

    lr->CEclmb = C_ELE * ( dTap - Tap * r_ij / dr3gamij_1 ) / dr3gamij_3;
}
