/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

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

#include "two_body_interactions.h"

#include "bond_orders.h"
#include "list.h"
#include "lookup.h"
#include "vector.h"


void Bond_Energy( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i;
    real gp3, gp4, gp7, gp10, gp37, ebond_total;
    reax_list *bonds;

    bonds = &(*lists)[BONDS];
    gp3 = system->reaxprm.gp.l[3];
    gp4 = system->reaxprm.gp.l[4];
    gp7 = system->reaxprm.gp.l[7];
    gp10 = system->reaxprm.gp.l[10];
    gp37 = (int) system->reaxprm.gp.l[37];
    ebond_total = 0.0;

#ifdef _OPENMP
//    #pragma omp parallel default(shared) reduction(+: ebond_total)
#endif
    { 
        int j, pj;
        int start_i, end_i;
        int type_i, type_j;
        real ebond, pow_BOs_be2, exp_be12, CEbo;
        real exphu, exphua1, exphub1, exphuov, hulpov, estriph;
        real decobdbo, decobdboua, decobdboub;
        single_body_parameters *sbp_i, *sbp_j;
        two_body_parameters *twbp;
        bond_order_data *bo_ij;

#ifdef _OPENMP
//        #pragma omp for schedule(guided)
#endif
        for ( i = 0; i < system->N; ++i )
        {
            start_i = Start_Index(i, bonds);
            end_i = End_Index(i, bonds);

            for ( pj = start_i; pj < end_i; ++pj )
            {
                if ( i < bonds->select.bond_list[pj].nbr )
                {
                    /* set the pointers */
                    j = bonds->select.bond_list[pj].nbr;
                    type_i = system->atoms[i].type;
                    type_j = system->atoms[j].type;
                    sbp_i = &( system->reaxprm.sbp[type_i] );
                    sbp_j = &( system->reaxprm.sbp[type_j] );
                    twbp = &( system->reaxprm.tbp[type_i][type_j] );
                    bo_ij = &( bonds->select.bond_list[pj].bo_data );

                    /* calculate the constants */
                    pow_BOs_be2 = POW( bo_ij->BO_s, twbp->p_be2 );
                    exp_be12 = EXP( twbp->p_be1 * ( 1.0 - pow_BOs_be2 ) );
                    CEbo = -twbp->De_s * exp_be12 *
                           ( 1.0 - twbp->p_be1 * twbp->p_be2 * pow_BOs_be2 );

                    /* calculate the Bond Energy */
                    ebond = -twbp->De_s * bo_ij->BO_s * exp_be12
                        - twbp->De_p * bo_ij->BO_pi
                        - twbp->De_pp * bo_ij->BO_pi2;
                    ebond_total += ebond;

                    /* calculate derivatives of Bond Orders */
                    bo_ij->Cdbo += CEbo;
                    bo_ij->Cdbopi -= (CEbo + twbp->De_p);
                    bo_ij->Cdbopi2 -= (CEbo + twbp->De_pp);

#ifdef TEST_ENERGY
                    fprintf( out_control->ebond, "%6d%6d%24.15e%24.15e\n",
                             workspace->orig_id[i], workspace->orig_id[j],
                             // i+1, j+1,
                             bo_ij->BO, ebond );
#endif

#ifdef TEST_FORCES
                    Add_dBO( system, lists, i, pj, CEbo, workspace->f_be );
                    Add_dBOpinpi2( system, lists, i, pj,
                                   -(CEbo + twbp->De_p), -(CEbo + twbp->De_pp),
                                   workspace->f_be, workspace->f_be );
#endif

                    /* Stabilisation terminal triple bond */
                    if ( bo_ij->BO >= 1.00 )
                    {
                        if ( gp37 == 2 ||
                                (sbp_i->mass == 12.0000 && sbp_j->mass == 15.9990) ||
                                (sbp_j->mass == 12.0000 && sbp_i->mass == 15.9990) )
                        {
                            //ba = SQR(bo_ij->BO - 2.50);
                            exphu = EXP( -gp7 * SQR(bo_ij->BO - 2.50) );
                            //oboa=abo(j1)-boa;
                            //obob=abo(j2)-boa;
                            exphua1 = EXP(-gp3 * (workspace->total_bond_order[i] - bo_ij->BO));
                            exphub1 = EXP(-gp3 * (workspace->total_bond_order[j] - bo_ij->BO));
                            //ovoab=abo(j1)-aval(it1)+abo(j2)-aval(it2);
                            exphuov = EXP(gp4 * (workspace->Delta[i] + workspace->Delta[j]));
                            hulpov = 1.0 / (1.0 + 25.0 * exphuov);

                            estriph = gp10 * exphu * hulpov * (exphua1 + exphub1);
                            //estrain(j1) = estrain(j1) + 0.50*estriph;
                            //estrain(j2) = estrain(j2) + 0.50*estriph;
                            ebond_total += estriph;

                            decobdbo = gp10 * exphu * hulpov * (exphua1 + exphub1) *
                                ( gp3 - 2.0 * gp7 * (bo_ij->BO - 2.50) );
                            decobdboua = -gp10 * exphu * hulpov *
                                (gp3 * exphua1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));
                            decobdboub = -gp10 * exphu * hulpov *
                                (gp3 * exphub1 + 25.0 * gp4 * exphuov * hulpov * (exphua1 + exphub1));

                            bo_ij->Cdbo += decobdbo;
                            workspace->CdDelta[i] += decobdboua;
                            workspace->CdDelta[j] += decobdboub;

#ifdef TEST_ENERGY
                            fprintf( out_control->ebond,
                                     "%6d%6d%24.15e%24.15e%24.15e%24.15e\n",
                                     workspace->orig_id[i], workspace->orig_id[j],
                                     //i+1, j+1,
                                     estriph, decobdbo, decobdboua, decobdboub );
#endif

#ifdef TEST_FORCES
                            Add_dBO( system, lists, i, pj, decobdbo, workspace->f_be );
                            Add_dDelta( system, lists, i, decobdboua, workspace->f_be );
                            Add_dDelta( system, lists, j, decobdboub, workspace->f_be );
#endif
                        }
                    }
                }
            }
        }
    }

    data->E_BE += ebond_total;
}


void vdW_Coulomb_Energy( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i;
    real p_vdW1, p_vdW1i;
    reax_list *far_nbrs;
    real e_vdW_total, e_ele_total;

    p_vdW1 = system->reaxprm.gp.l[28];
    p_vdW1i = 1.0 / p_vdW1;
    far_nbrs = &(*lists)[FAR_NBRS];
    e_vdW_total = 0.0;
    e_ele_total = 0.0;

#ifdef _OPENMP
    #pragma omp parallel default(shared) reduction(+: e_vdW_total, e_ele_total)
#endif
    {
        int j, pj;
        int start_i, end_i;
        real self_coef;
        real powr_vdW1, powgi_vdW1;
        real tmp, r_ij, fn13, exp1, exp2;
        real Tap, dTap, dfn13, CEvd, CEclmb;
        real dr3gamij_1, dr3gamij_3;
        real e_ele, e_vdW, e_core, de_core;
        rvec temp, ext_press;
        //rtensor temp_rtensor, total_rtensor;
        two_body_parameters *twbp;
        far_neighbor_data *nbr_pj;
#ifdef _OPENMP
        int tid;

        tid = omp_get_thread_num( );
#endif

        e_ele = 0.0;
        e_vdW = 0.0;
        e_core = 0.0;
        de_core = 0.0;

#ifdef _OPENMP
        #pragma omp for schedule(guided)
#endif
        for ( i = 0; i < system->N; ++i )
        {
            start_i = Start_Index( i, far_nbrs );
            end_i = End_Index( i, far_nbrs );

            for ( pj = start_i; pj < end_i; ++pj )
            {
                if ( far_nbrs->select.far_nbr_list[pj].d <= control->r_cut )
                {
                    nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
                    j = nbr_pj->nbr;
                    r_ij = nbr_pj->d;
                    twbp = &(system->reaxprm.tbp[ system->atoms[i].type ]
                             [ system->atoms[j].type ]);
                    self_coef = (i == j) ? 0.5 : 1.0; // for supporting small boxes!

                    /* Calculate Taper and its derivative */
                    // Tap = nbr_pj->Tap;   -- precomputed during compte_H
                    Tap = control->Tap7 * r_ij + control->Tap6;
                    Tap = Tap * r_ij + control->Tap5;
                    Tap = Tap * r_ij + control->Tap4;
                    Tap = Tap * r_ij + control->Tap3;
                    Tap = Tap * r_ij + control->Tap2;
                    Tap = Tap * r_ij + control->Tap1;
                    Tap = Tap * r_ij + control->Tap0;

                    dTap = 7 * control->Tap7 * r_ij + 6 * control->Tap6;
                    dTap = dTap * r_ij + 5 * control->Tap5;
                    dTap = dTap * r_ij + 4 * control->Tap4;
                    dTap = dTap * r_ij + 3 * control->Tap3;
                    dTap = dTap * r_ij + 2 * control->Tap2;
                    dTap += control->Tap1 / r_ij;

                    /* vdWaals Calculations */
                    if ( system->reaxprm.gp.vdw_type == 1 || system->reaxprm.gp.vdw_type == 3 )
                    {
                        /* shielding */
                        powr_vdW1 = POW( r_ij, p_vdW1 );
                        powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1 );

                        fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
                        exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
                        exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

                        e_vdW = self_coef * Tap * twbp->D * (exp1 - 2.0 * exp2);
                        e_vdW_total += e_vdW;

                        dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) *
                            POW( r_ij, p_vdW1 - 2.0 );

                        CEvd = self_coef * ( dTap * twbp->D * (exp1 - 2 * exp2) -
                                Tap * twbp->D * (twbp->alpha / twbp->r_vdW) *
                                (exp1 - exp2) * dfn13 );
                    }
                    /* no shielding */
                    else
                    {
                        exp1 = EXP( twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
                        exp2 = EXP( 0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );

                        e_vdW = self_coef * Tap * twbp->D * (exp1 - 2.0 * exp2);
                        e_vdW_total += e_vdW;

                        CEvd = self_coef * ( dTap * twbp->D * (exp1 - 2.0 * exp2) -
                                Tap * twbp->D * (twbp->alpha / twbp->r_vdW) *
                                (exp1 - exp2) );
                    }

                    if ( system->reaxprm.gp.vdw_type == 2 || system->reaxprm.gp.vdw_type == 3 )
                    {
                        /* innner wall */
                        e_core = twbp->ecore * EXP( twbp->acore * (1.0 - (r_ij / twbp->rcore)) );
                        e_vdW += self_coef * Tap * e_core;
                        e_vdW_total += self_coef * Tap * e_core;

                        de_core = -(twbp->acore / twbp->rcore) * e_core;
                        CEvd += self_coef * ( dTap * e_core + Tap * de_core );
                    }

                    /* Coulomb Calculations */
                    dr3gamij_1 = ( r_ij * r_ij * r_ij + twbp->gamma );
                    dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

                    tmp = Tap / dr3gamij_3;
                    e_ele = self_coef * C_ele * system->atoms[i].q * system->atoms[j].q * tmp;
                    e_ele_total += e_ele;

                    CEclmb = self_coef * C_ele * system->atoms[i].q * system->atoms[j].q *
                             ( dTap -  Tap * r_ij / dr3gamij_1 ) / dr3gamij_3;

                    if ( control->ensemble == NVE || control->ensemble == NVT || control->ensemble == bNVT )
                    {
#ifndef _OPENMP
                        rvec_ScaledAdd( system->atoms[i].f,
                                -(CEvd + CEclmb), nbr_pj->dvec );
                        rvec_ScaledAdd( system->atoms[j].f,
                                +(CEvd + CEclmb), nbr_pj->dvec );
#else
                        rvec_ScaledAdd( workspace->f_local[tid * system->N + i],
                                -(CEvd + CEclmb), nbr_pj->dvec );
                        rvec_ScaledAdd( workspace->f_local[tid * system->N + j],
                                +(CEvd + CEclmb), nbr_pj->dvec );
#endif
                    }
                    /* NPT, iNPT or sNPT */
                    else
                    {
                        /* for pressure coupling, terms not related to bond order
                           derivatives are added directly into pressure vector/tensor */
                        rvec_Scale( temp, CEvd + CEclmb, nbr_pj->dvec );

#ifndef _OPENMP
                        rvec_ScaledAdd( system->atoms[i].f, -1., temp );
                        rvec_Add( system->atoms[j].f, temp );
#else
                        rvec_ScaledAdd( workspace->f_local[tid * system->N + i], -1., temp );
                        rvec_Add( workspace->f_local[tid * system->N + j], temp );
#endif

                        rvec_iMultiply( ext_press, nbr_pj->rel_box, temp );
#ifdef _OPENMP
                        #pragma omp critical (vdW_Coulomb_Energy_ext_press)
#endif
                        {
                            rvec_Add( data->ext_press, ext_press );
                        }

                        /*fprintf( stderr, "nonbonded(%d,%d): rel_box (%f %f %f)",
                          i,j,nbr_pj->rel_box[0],nbr_pj->rel_box[1],nbr_pj->rel_box[2] );

                          fprintf( stderr, "force(%f %f %f)", temp[0], temp[1], temp[2] );

                          fprintf( stderr, "ext_press (%12.6f %12.6f %12.6f)\n",
                          data->ext_press[0], data->ext_press[1], data->ext_press[2] );*/

                        /* This part is intended for a fully-flexible box */
                        /* rvec_OuterProduct( temp_rtensor, nbr_pj->dvec,
                           system->atoms[i].x );
                           rtensor_Scale( total_rtensor,
                           F_C * -(CEvd + CEclmb), temp_rtensor );
                           rvec_OuterProduct( temp_rtensor,
                           nbr_pj->dvec, system->atoms[j].x );
                           rtensor_ScaledAdd( total_rtensor,
                           F_C * +(CEvd + CEclmb), temp_rtensor );

                           if( nbr_pj->imaginary )
                           // This is an external force due to an imaginary nbr
                           rtensor_ScaledAdd( data->flex_bar.P, -1.0, total_rtensor );
                           else
                           // This interaction is completely internal
                           rtensor_Add( data->flex_bar.P, total_rtensor ); */
                    }

#ifdef TEST_ENERGY
                    rvec_MakeZero( temp );
                    rvec_ScaledAdd( temp, +CEvd, nbr_pj->dvec );
                    fprintf( out_control->evdw,
                             "%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                             //i+1, j+1,
                             MIN( workspace->orig_id[i], workspace->orig_id[j] ),
                             MAX( workspace->orig_id[i], workspace->orig_id[j] ),
                             r_ij, e_vdW, temp[0], temp[1], temp[2]/*, e_vdW_total*/ );

                    fprintf( out_control->ecou, "%6d%6d%24.15e%24.15e%24.15e%24.15e\n",
                             MIN( workspace->orig_id[i], workspace->orig_id[j] ),
                             MAX( workspace->orig_id[i], workspace->orig_id[j] ),
                             r_ij, system->atoms[i].q, system->atoms[j].q,
                             e_ele/*, e_ele_total*/ );
#endif

#ifdef TEST_FORCES
                    rvec_ScaledAdd( workspace->f_vdw[i], -CEvd, nbr_pj->dvec );
                    rvec_ScaledAdd( workspace->f_vdw[j], +CEvd, nbr_pj->dvec );
                    rvec_ScaledAdd( workspace->f_ele[i], -CEclmb, nbr_pj->dvec );
                    rvec_ScaledAdd( workspace->f_ele[j], +CEclmb, nbr_pj->dvec );
#endif
                }
            }
        }
    }

    data->E_vdW = e_vdW_total;
    data->E_Ele = e_ele_total;

    // sfclose( fout, "vdW_Coulomb_Energy::fout" );

    // fprintf( stderr, "nonbonded: ext_press (%24.15e %24.15e %24.15e)\n",
    // data->ext_press[0], data->ext_press[1], data->ext_press[2] );
}


void LR_vdW_Coulomb( reax_system *system, control_params *control,
        int i, int j, real r_ij, LR_data *lr )
{
    real p_vdW1 = system->reaxprm.gp.l[28];
    real p_vdW1i = 1.0 / p_vdW1;
    real powr_vdW1, powgi_vdW1;
    real tmp, fn13, exp1, exp2;
    real Tap, dTap, dfn13;
    real dr3gamij_1, dr3gamij_3;
    real e_core, de_core;
    two_body_parameters *twbp;

    twbp = &(system->reaxprm.tbp[i][j]);
    e_core = 0;
    de_core = 0;

    /* calculate taper and its derivative */
    Tap = control->Tap7 * r_ij + control->Tap6;
    Tap = Tap * r_ij + control->Tap5;
    Tap = Tap * r_ij + control->Tap4;
    Tap = Tap * r_ij + control->Tap3;
    Tap = Tap * r_ij + control->Tap2;
    Tap = Tap * r_ij + control->Tap1;
    Tap = Tap * r_ij + control->Tap0;

    dTap = 7 * control->Tap7 * r_ij + 6 * control->Tap6;
    dTap = dTap * r_ij + 5 * control->Tap5;
    dTap = dTap * r_ij + 4 * control->Tap4;
    dTap = dTap * r_ij + 3 * control->Tap3;
    dTap = dTap * r_ij + 2 * control->Tap2;
    dTap += control->Tap1 / r_ij;


    /* vdWaals calculations */
    powr_vdW1 = POW( r_ij, p_vdW1 );
    powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1 );

    fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
    exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
    exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

    lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);
    /* fprintf(stderr,"vdW: Tap:%f, r: %f, f13:%f, D:%f, Energy:%f,\
       Gamma_w:%f, p_vdw: %f, alpha: %f, r_vdw: %f, %lf %lf\n",
       Tap, r_ij, fn13, twbp->D, Tap * twbp->D * (exp1 - 2.0 * exp2),
       powgi_vdW1, p_vdW1, twbp->alpha, twbp->r_vdW, exp1, exp2); */

    dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0 )
        * POW( r_ij, p_vdW1 - 2.0 );

    lr->CEvd = dTap * twbp->D * (exp1 - 2 * exp2) -
        Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;

    /* vdWaals Calculations */
    if ( system->reaxprm.gp.vdw_type == 1 || system->reaxprm.gp.vdw_type == 3 )
    {
        // shielding
        powr_vdW1 = POW( r_ij, p_vdW1 );
        powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1 );

        fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
        exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
        exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

        lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);

        dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) *
            POW( r_ij, p_vdW1 - 2.0 );

        lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
            Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;
    }
    /* no shielding */
    else
    {
        exp1 = EXP( twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
        exp2 = EXP( 0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );

        lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);

        lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
            Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2);
    }

    if ( system->reaxprm.gp.vdw_type == 2 || system->reaxprm.gp.vdw_type == 3 )
    {
        // innner wall
        e_core = twbp->ecore * EXP(twbp->acore * (1.0 - (r_ij / twbp->rcore)));
        lr->e_vdW += Tap * e_core;

        de_core = -(twbp->acore / twbp->rcore) * e_core;
        lr->CEvd += dTap * e_core + Tap * de_core;
    }

    /* Coulomb calculations */
    dr3gamij_1 = ( r_ij * r_ij * r_ij + twbp->gamma );
    dr3gamij_3 = POW( dr3gamij_1 , 1.0 / 3.0 );

    tmp = Tap / dr3gamij_3;
    lr->H = EV_to_KCALpMOL * tmp;
    lr->e_ele = C_ele * tmp;
    /* fprintf( stderr,"i:%d(%d), j:%d(%d), gamma:%f,\
       Tap:%f, dr3gamij_3:%f, qi: %f, qj: %f\n",
       i, system->atoms[i].type, j, system->atoms[j].type,
       twbp->gamma, Tap, dr3gamij_3,
       system->atoms[i].q, system->atoms[j].q ); */

    lr->CEclmb = C_ele * ( dTap -  Tap * r_ij / dr3gamij_1 ) / dr3gamij_3;
    /* fprintf( stdout, "%d %d\t%g\t%g  %g\t%g  %g\t%g  %g\n",
       i+1, j+1, r_ij, e_vdW, CEvd * r_ij,
       system->atoms[i].q, system->atoms[j].q, e_ele, CEclmb * r_ij ); */

    /* fprintf( stderr,"LR_Lookup:%3d%3d%5.3f-%8.5f,%8.5f%8.5f,%8.5f%8.5f\n",
       i, j, r_ij, lr->H, lr->e_vdW, lr->CEvd, lr->e_ele, lr->CEclmb ); */
}


void Tabulated_vdW_Coulomb_Energy( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int steps, update_freq, update_energies;
    reax_list *far_nbrs;
    real e_vdW_total, e_ele_total;

    far_nbrs = &(*lists)[FAR_NBRS];
    steps = data->step - data->prev_steps;
    update_freq = out_control->energy_update_freq;
    update_energies = update_freq > 0 && steps % update_freq == 0;
    e_vdW_total = 0.0;
    e_ele_total = 0.0;

#ifdef _OPENMP
    #pragma omp parallel default(shared) reduction(+: e_vdW_total, e_ele_total)
#endif
    {
        int i, j, pj, r;
        int type_i, type_j, tmin, tmax;
        int start_i, end_i;
        real r_ij, self_coef, base, dif;
        real e_vdW, e_ele;
        real CEvd, CEclmb;
        rvec temp, ext_press;
        far_neighbor_data *nbr_pj;
        LR_lookup_table *t;
#ifdef _OPENMP
        int tid;

        tid = omp_get_thread_num( );

        #pragma omp for schedule(guided)
#endif
        for ( i = 0; i < system->N; ++i )
        {
            type_i = system->atoms[i].type;
            start_i = Start_Index(i, far_nbrs);
            end_i = End_Index(i, far_nbrs);

            for ( pj = start_i; pj < end_i; ++pj )
            {
                if ( far_nbrs->select.far_nbr_list[pj].d <= control->r_cut )
                {
                    nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
                    j = nbr_pj->nbr;
                    type_j = system->atoms[j].type;
                    r_ij = nbr_pj->d;
                    self_coef = (i == j) ? 0.5 : 1.0;
                    tmin = MIN( type_i, type_j );
                    tmax = MAX( type_i, type_j );
                    t = &( workspace->LR[tmin][tmax] );

                    /* Cubic Spline Interpolation */
                    r = (int)(r_ij * t->inv_dx);
                    if ( r == 0 )
                    {
                        ++r;
                    }
                    base = (real)(r + 1) * t->dx;
                    dif = r_ij - base;
                    //fprintf(stderr, "r: %f, i: %d, base: %f, dif: %f\n", r, i, base, dif);

                    if ( update_energies )
                    {
                        e_vdW = ((t->vdW[r].d * dif + t->vdW[r].c) * dif + t->vdW[r].b) * dif +
                                t->vdW[r].a;
                        e_vdW *= self_coef;

                        e_ele = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b) * dif +
                                t->ele[r].a;
                        e_ele *= self_coef * system->atoms[i].q * system->atoms[j].q;

                        e_vdW_total += e_vdW;
                        e_ele_total += e_ele;
                    }

                    CEvd = ((t->CEvd[r].d * dif + t->CEvd[r].c) * dif + t->CEvd[r].b) * dif +
                           t->CEvd[r].a;
                    CEvd *= self_coef;
                    //CEvd = (3*t->vdW[r].d*dif + 2*t->vdW[r].c)*dif + t->vdW[r].b;

                    CEclmb = ((t->CEclmb[r].d * dif + t->CEclmb[r].c) * dif + t->CEclmb[r].b) * dif +
                             t->CEclmb[r].a;
                    CEclmb *= self_coef * system->atoms[i].q * system->atoms[j].q;

                    if ( control->ensemble == NVE || control->ensemble == NVT  || control->ensemble == bNVT)
                    {
#ifndef _OPENMP
                        rvec_ScaledAdd( system->atoms[i].f, -(CEvd + CEclmb), nbr_pj->dvec );
                        rvec_ScaledAdd( system->atoms[j].f, +(CEvd + CEclmb), nbr_pj->dvec );
#else
                        rvec_ScaledAdd( workspace->f_local[tid * system->N + i],
                                -(CEvd + CEclmb), nbr_pj->dvec );
                        rvec_ScaledAdd( workspace->f_local[tid * system->N + j],
                                +(CEvd + CEclmb), nbr_pj->dvec );
#endif
                    }
                    else   // NPT, iNPT or sNPT
                    {
                        /* for pressure coupling, terms not related to bond order
                           derivatives are added directly into pressure vector/tensor */
                        rvec_Scale( temp, CEvd + CEclmb, nbr_pj->dvec );
#ifndef _OPENMP
                        rvec_ScaledAdd( system->atoms[i].f, -1., temp );
                        rvec_Add( system->atoms[j].f, temp );
#else
                        rvec_ScaledAdd( workspace->f_local[tid * system->N + i], -1., temp );
                        rvec_Add( workspace->f_local[tid * system->N + j], temp );
#endif
                        rvec_iMultiply( ext_press, nbr_pj->rel_box, temp );
#ifdef _OPENMP
                        #pragma omp critical (Tabulated_vdW_Coulomb_Energy_ext_press)
#endif
                        {
                        rvec_Add( data->ext_press, ext_press );
                        }
                    }

#ifdef TEST_ENERGY
                    fprintf( out_control->evdw, "%6d%6d%24.15e%24.15e%24.15e\n",
                            workspace->orig_id[i], workspace->orig_id[j],
                            r_ij, e_vdW, data->E_vdW );
                    fprintf( out_control->ecou, "%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                            workspace->orig_id[i], workspace->orig_id[j],
                            r_ij, system->atoms[i].q, system->atoms[j].q,
                            e_ele, data->E_Ele );
#endif

#ifdef TEST_FORCES
                    rvec_ScaledAdd( workspace->f_vdw[i], -CEvd, nbr_pj->dvec );
                    rvec_ScaledAdd( workspace->f_vdw[j], +CEvd, nbr_pj->dvec );
                    rvec_ScaledAdd( workspace->f_ele[i], -CEclmb, nbr_pj->dvec );
                    rvec_ScaledAdd( workspace->f_ele[j], +CEclmb, nbr_pj->dvec );
#endif
                }
            }
        }
    }

    data->E_vdW += e_vdW_total;
    data->E_Ele += e_ele_total;
}


#if defined(OLD)
/* Linear extrapolation */
/*p     = (r_ij * t->inv_dx;
  r     = (int) p;
  prev  = &( t->y[r] );
  next  = &( t->y[r+1] );

  tmp    = p - r;
  e_vdW  = self_coef * (prev->e_vdW + tmp*(next->e_vdW - prev->e_vdW ));
  CEvd   = self_coef * (prev->CEvd  + tmp*(next->CEvd  - prev->CEvd  ));

  e_ele  = self_coef * (prev->e_ele + tmp*(next->e_ele - prev->e_ele ));
  e_ele  = e_ele  * system->atoms[i].q * system->atoms[j].q;
  CEclmb = self_coef * (prev->CEclmb+tmp*(next->CEclmb - prev->CEclmb));
  CEclmb = CEclmb * system->atoms[i].q * system->atoms[j].q;*/
#endif
