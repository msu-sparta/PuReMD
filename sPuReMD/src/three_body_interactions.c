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

#include "three_body_interactions.h"
#include "bond_orders.h"
#include "list.h"
#include "lookup.h"
#include "vector.h"


/* calculates the theta angle between i-j-k */
void Calculate_Theta( rvec dvec_ji, real d_ji, rvec dvec_jk, real d_jk,
        real *theta, real *cos_theta )
{
    (*cos_theta) = rvec_Dot( dvec_ji, dvec_jk ) / ( d_ji * d_jk );
    if ( *cos_theta > 1.0 )
    {
        *cos_theta  = 1.0;
    }
    if ( *cos_theta < -1.0 )
    {
        *cos_theta  = -1.0;
    }

    (*theta) = ACOS( *cos_theta );
}


/* calculates the derivative of the cosine of the angle between i-j-k */
void Calculate_dCos_Theta( rvec dvec_ji, real d_ji, rvec dvec_jk, real d_jk,
        rvec* dcos_theta_di, rvec* dcos_theta_dj, rvec* dcos_theta_dk )
{
    int t;
    real sqr_d_ji, sqr_d_jk, inv_dists, inv_dists3, dot_dvecs, Cdot_inv3;

    sqr_d_ji = SQR( d_ji );
    sqr_d_jk = SQR( d_jk );
    inv_dists = 1.0 / (d_ji * d_jk);
    inv_dists3 = POW( inv_dists, 3 );
    dot_dvecs = rvec_Dot( dvec_ji, dvec_jk );
    Cdot_inv3 = dot_dvecs * inv_dists3;

    for ( t = 0; t < 3; ++t )
    {
        (*dcos_theta_di)[t] = dvec_jk[t] * inv_dists -
            Cdot_inv3 * sqr_d_jk * dvec_ji[t];

        (*dcos_theta_dj)[t] = -(dvec_jk[t] + dvec_ji[t]) * inv_dists +
            Cdot_inv3 * ( sqr_d_jk * dvec_ji[t] + sqr_d_ji * dvec_jk[t] );

        (*dcos_theta_dk)[t] = dvec_ji[t] * inv_dists -
            Cdot_inv3 * sqr_d_ji * dvec_jk[t];
    }

    /*fprintf( stderr,
      "%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e\n",
      dvec_jk[t] * inv_dists*/
}


/* this is a 3-body interaction in which the main role is
   played by j which sits in the middle of the other two. */
void Three_Body_Interactions( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        list **lists, output_controls *out_control )
{
    real *total_bo;
    list *bonds, *thb_intrs;
    bond_data *bond_list;
    three_body_interaction_data *thb_list;
    real p_pen2, p_pen3, p_pen4;
    real p_coa2, p_coa3, p_coa4;
    real p_val6, p_val8, p_val9, p_val10;
    int num_thb_intrs;
    real e_ang_total, e_pen_total, e_coa_total;

    total_bo = workspace->total_bond_order;
    bonds = (*lists) + BONDS;
    bond_list = bonds->select.bond_list;
    thb_intrs = (*lists) + THREE_BODIES;
    thb_list = thb_intrs->select.three_body_list;
    /* global parameters used in these calculations */
    p_pen2 = system->reaxprm.gp.l[19];
    p_pen3 = system->reaxprm.gp.l[20];
    p_pen4 = system->reaxprm.gp.l[21];
    p_coa2 = system->reaxprm.gp.l[2];
    p_coa3 = system->reaxprm.gp.l[38];
    p_coa4 = system->reaxprm.gp.l[30];
    p_val6 = system->reaxprm.gp.l[14];
    p_val8 = system->reaxprm.gp.l[33];
    p_val9 = system->reaxprm.gp.l[16];
    p_val10 = system->reaxprm.gp.l[17];
    num_thb_intrs = 0;
    e_ang_total = 0.0;
    e_pen_total = 0.0;
    e_coa_total = 0.0;

    //TODO: change interaction lists for parallelization
#ifdef _OPENMP
//    #pragma omp parallel default(shared) reduction(+:total_Eang, total_Epen, total_Ecoa, num_thb_intrs) 
#endif
    {
        int i, j, pi, k, pk, t;
        int type_i, type_j, type_k;
        int start_j, end_j, start_pk, end_pk;
        int cnt;
        real temp, temp_bo_jt, pBOjt7;
        real p_val1, p_val2, p_val3, p_val4, p_val5, p_val7;
        real p_pen1;
        real p_coa1;
        real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
        real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
        real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO;
        real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
        real CEpen1, CEpen2, CEpen3;
        real e_ang, e_coa, e_pen;
        real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
        real Cf7ij, Cf7jk, Cf8j, Cf9j;
        real f7_ij, f7_jk, f8_Dj, f9_Dj;
        real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
        real BOA_ij, BOA_jk;
        real vlpadj;
        rvec force, ext_press;
        //rtensor temp_rtensor, total_rtensor;
        three_body_header *thbh;
        three_body_parameters *thbp;
        three_body_interaction_data *p_ijk, *p_kji;
        bond_data *pbond_ij, *pbond_jk, *pbond_jt;
        bond_order_data *bo_ij, *bo_jk, *bo_jt;
        rvec *f_i, *f_j, *f_k;
#ifdef _OPENMP
//        int tid = omp_get_thread_num( );
#endif

        for ( j = 0; j < system->N; ++j )
        {
            // fprintf( out_control->eval, "j: %d\n", j );
            type_j = system->atoms[j].type;
            start_j = Start_Index(j, bonds);
            end_j = End_Index(j, bonds);
//#ifdef _OPENMP
//            f_j = &(workspace->f_local[tid * system->N + j]);
//#else
            f_j = &(system->atoms[j].f);
//#endif

            p_val3 = system->reaxprm.sbp[ type_j ].p_val3;
            p_val5 = system->reaxprm.sbp[ type_j ].p_val5;

            SBOp = 0.0;
            prod_SBO = 1.0;
            for ( t = start_j; t < end_j; ++t )
            {
                bo_jt = &(bond_list[t].bo_data);
                SBOp += (bo_jt->BO_pi + bo_jt->BO_pi2);
                temp = SQR( bo_jt->BO );
                temp *= temp;
                temp *= temp;
                prod_SBO *= EXP( -temp );
            }

            /* modifications to match Adri's code - 09/01/09 */
            if ( workspace->vlpex[j] >= 0.0 )
            {
                vlpadj = 0.0;
                dSBO2 = prod_SBO - 1.0;
            }
            else
            {
                vlpadj = workspace->nlp[j];
                dSBO2 = (prod_SBO - 1.0) * (1.0 - p_val8 * workspace->dDelta_lp[j]);
            }

            SBO = SBOp + (1.0 - prod_SBO) * (-workspace->Delta_boc[j] - p_val8 * vlpadj);
            dSBO1 = -8.0 * prod_SBO * ( workspace->Delta_boc[j] + p_val8 * vlpadj );

            if ( SBO <= 0.0 )
            {
                SBO2 = 0.0;
                CSBO2 = 0.0;
            }
            else if ( SBO > 0.0 && SBO <= 1.0 )
            {
                SBO2 = POW( SBO, p_val9 );
                CSBO2 = p_val9 * POW( SBO, p_val9 - 1.0 );
            }
            else if ( SBO > 1.0 && SBO < 2.0 )
            {
                SBO2 = 2.0 - POW( 2.0 - SBO, p_val9 );
                CSBO2 = p_val9 * POW( 2.0 - SBO, p_val9 - 1.0 );
            }
            else
            {
                SBO2 = 2.0;
                CSBO2 = 0.0;
            }

            expval6 = EXP( p_val6 * workspace->Delta_boc[j] );

            /* unlike 2-body intrs where we enforce i<j, we cannot put any such
               restrictions here. such a restriction would prevent us from producing
               all 4-body intrs correctly */
            for ( pi = start_j; pi < end_j; ++pi )
            {
                Set_Start_Index( pi, num_thb_intrs, thb_intrs );
                pbond_ij = &(bond_list[pi]);
                bo_ij = &(pbond_ij->bo_data);
                BOA_ij = bo_ij->BO - control->thb_cut;

                if ( BOA_ij > 0.0 )
                {
                    i = pbond_ij->nbr;
                    type_i = system->atoms[i].type;
//#ifdef _OPENMP
//                    f_i = &(workspace->f_local[tid * system->N + i]);
//#else
                    f_i = &(system->atoms[i].f);
//#endif

                    /* first copy 3-body intrs from previously computed ones where i>k.
                       IMPORTANT: if it is less costly to compute theta and its
                       derivative, we should definitely re-compute them,
                       instead of copying!
                       in the second for-loop below, we compute only new 3-body intrs
                       where i < k */
                    for ( pk = start_j; pk < pi; ++pk )
                    {
                        // fprintf( out_control->eval, "pk: %d\n", pk );
                        start_pk = Start_Index( pk, thb_intrs );
                        end_pk = End_Index( pk, thb_intrs );

                        for ( t = start_pk; t < end_pk; ++t )
                        {
                            if ( thb_list[t].thb == i )
                            {
                                p_ijk = &(thb_list[num_thb_intrs]);
                                p_kji = &(thb_list[t]);

                                p_ijk->thb = bond_list[pk].nbr;
                                p_ijk->pthb  = pk;
                                p_ijk->theta = p_kji->theta;
                                rvec_Copy( p_ijk->dcos_di, p_kji->dcos_dk );
                                rvec_Copy( p_ijk->dcos_dj, p_kji->dcos_dj );
                                rvec_Copy( p_ijk->dcos_dk, p_kji->dcos_di );

                                ++num_thb_intrs;
                                break;
                            }
                        }
                    }


                    /* and this is the second for loop mentioned above */
                    for ( pk = pi + 1; pk < end_j; ++pk )
                    {
                        pbond_jk = &(bond_list[pk]);
                        bo_jk = &(pbond_jk->bo_data);
                        BOA_jk = bo_jk->BO - control->thb_cut;
                        k = pbond_jk->nbr;
                        type_k = system->atoms[k].type;
                        p_ijk = &( thb_list[num_thb_intrs] );
//#ifdef _OPENMP
//                        f_k = &(workspace->f_local[tid * system->N + k]);
//#else
                        f_k = &(system->atoms[k].f);
//#endif

                        //CHANGE ORIGINAL
                        if ( BOA_jk <= 0 )
                        {
                            continue;
                        }
                        //CHANGE ORIGINAL

                        Calculate_Theta( pbond_ij->dvec, pbond_ij->d,
                                pbond_jk->dvec, pbond_jk->d,
                                &theta, &cos_theta );

                        Calculate_dCos_Theta( pbond_ij->dvec, pbond_ij->d,
                                pbond_jk->dvec, pbond_jk->d,
                                &(p_ijk->dcos_di), &(p_ijk->dcos_dj),
                                &(p_ijk->dcos_dk) );

                        p_ijk->thb = k;
                        p_ijk->pthb = pk;
                        p_ijk->theta = theta;

                        sin_theta = SIN( theta );
                        if ( sin_theta < 1.0e-5 )
                        {
                            sin_theta = 1.0e-5;
                        }

                        ++num_thb_intrs;

                        if ( BOA_jk > 0.0 &&
                                (bo_ij->BO * bo_jk->BO) > SQR(control->thb_cut)/*0*/)
                        {
                            thbh = &( system->reaxprm.thbp[type_i][type_j][type_k] );

                            /* if( workspace->orig_id[i] < workspace->orig_id[k] )
                               fprintf( stdout, "%6d %6d %6d %7.3f %7.3f %7.3f\n",
                               workspace->orig_id[i], workspace->orig_id[j],
                               workspace->orig_id[k], bo_ij->BO, bo_jk->BO, p_ijk->theta );
                               else
                               fprintf( stdout, "%6d %6d %6d %7.3f %7.3f %7.3f\n",
                               workspace->orig_id[k], workspace->orig_id[j],
                               workspace->orig_id[i], bo_jk->BO, bo_ij->BO, p_ijk->theta ); */

                            for ( cnt = 0; cnt < thbh->cnt; ++cnt )
                            {
                                if ( FABS(thbh->prm[cnt].p_val1) > 0.001 )
                                {
                                    thbp = &( thbh->prm[cnt] );

                                    /* ANGLE ENERGY */
                                    p_val1 = thbp->p_val1;
                                    p_val2 = thbp->p_val2;
                                    p_val4 = thbp->p_val4;
                                    p_val7 = thbp->p_val7;
                                    theta_00 = thbp->theta_00;

                                    exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                                    f7_ij = 1.0 - exp3ij;
                                    Cf7ij = p_val3 * p_val4 *
                                        POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                                    exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                                    f7_jk = 1.0 - exp3jk;
                                    Cf7jk = p_val3 * p_val4 *
                                        POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                                    expval7 = EXP( -p_val7 * workspace->Delta_boc[j] );
                                    trm8 = 1.0 + expval6 + expval7;
                                    f8_Dj = p_val5 - ( (p_val5 - 1.0) * (2.0 + expval6) / trm8 );
                                    Cf8j = ( (1.0 - p_val5) / SQR(trm8) ) *
                                        (p_val6 * expval6 * trm8 -
                                         (2.0 + expval6) * ( p_val6 * expval6 - p_val7 * expval7 ));

                                    theta_0 = 180.0 - theta_00 * (1.0 - EXP(-p_val10 * (2.0 - SBO2)));
                                    theta_0 = DEG2RAD( theta_0 );

                                    expval2theta  = EXP(-p_val2 * SQR(theta_0 - theta));
                                    if ( p_val1 >= 0 )
                                    {
                                        expval12theta = p_val1 * (1.0 - expval2theta);
                                    }
                                    /* To avoid linear Me-H-Me angles (6/6/06) */
                                    else
                                    {
                                        expval12theta = p_val1 * -expval2theta;
                                    }

                                    CEval1 = Cf7ij * f7_jk * f8_Dj * expval12theta;
                                    CEval2 = Cf7jk * f7_ij * f8_Dj * expval12theta;
                                    CEval3 = Cf8j  * f7_ij * f7_jk * expval12theta;
                                    CEval4 = -2.0 * p_val1 * p_val2 * f7_ij * f7_jk * f8_Dj *
                                        expval2theta * (theta_0 - theta);

                                    Ctheta_0 = p_val10 * DEG2RAD(theta_00) *
                                        EXP( -p_val10 * (2.0 - SBO2) );

                                    CEval5 = -CEval4 * Ctheta_0 * CSBO2;
                                    CEval6 = CEval5 * dSBO1;
                                    CEval7 = CEval5 * dSBO2;
                                    CEval8 = -CEval4 / sin_theta;

                                    e_ang = f7_ij * f7_jk * f8_Dj * expval12theta;
                                    e_ang_total += e_ang;
                                    /* END ANGLE ENERGY*/

                                    /* PENALTY ENERGY */
                                    p_pen1 = thbp->p_pen1;

                                    exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                                    exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                                    exp_pen3 = EXP( -p_pen3 * workspace->Delta[j] );
                                    exp_pen4 = EXP(  p_pen4 * workspace->Delta[j] );
                                    trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                                    f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                                    Cf9j = (-p_pen3 * exp_pen3 * trm_pen34 -
                                            (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3 +
                                                p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                                    e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                                    e_pen_total += e_pen;

                                    CEpen1 = e_pen * Cf9j / f9_Dj;
                                    temp = -2.0 * p_pen2 * e_pen;
                                    CEpen2 = temp * (BOA_ij - 2.0);
                                    CEpen3 = temp * (BOA_jk - 2.0);
                                    /* END PENALTY ENERGY */

                                    /* COALITION ENERGY */
                                    p_coa1 = thbp->p_coa1;

                                    exp_coa2 = EXP( p_coa2 * workspace->Delta_boc[j] );
                                    e_coa = p_coa1 / (1. + exp_coa2) *
                                        EXP( -p_coa3 * SQR(total_bo[i] - BOA_ij) ) *
                                        EXP( -p_coa3 * SQR(total_bo[k] - BOA_jk) ) *
                                        EXP( -p_coa4 * SQR(BOA_ij - 1.5) ) *
                                        EXP( -p_coa4 * SQR(BOA_jk - 1.5) );
                                    e_coa_total += e_coa;

                                    CEcoa1 = -2 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                                    CEcoa2 = -2 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                                    CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1 + exp_coa2);
                                    CEcoa4 = -2 * p_coa3 * (total_bo[i] - BOA_ij) * e_coa;
                                    CEcoa5 = -2 * p_coa3 * (total_bo[k] - BOA_jk) * e_coa;
                                    /* END COALITION ENERGY */

                                    /* FORCES */
#ifdef _OPENMP
//                                    #pragma omp atomic
#endif
                                    bo_ij->Cdbo += (CEval1 + CEpen2 + (CEcoa1 - CEcoa4));
#ifdef _OPENMP
//                                    #pragma omp atomic
#endif
                                    bo_jk->Cdbo += (CEval2 + CEpen3 + (CEcoa2 - CEcoa5));
#ifdef _OPENMP
//                                    #pragma omp atomic
#endif
                                    workspace->CdDelta[j] += ((CEval3 + CEval7) + CEpen1 + CEcoa3);
#ifdef _OPENMP
//                                    #pragma omp atomic
#endif
                                    workspace->CdDelta[i] += CEcoa4;
#ifdef _OPENMP
//                                    #pragma omp atomic
#endif
                                    workspace->CdDelta[k] += CEcoa5;

                                    for ( t = start_j; t < end_j; ++t )
                                    {
                                        pbond_jt = &( bond_list[t] );
                                        bo_jt = &(pbond_jt->bo_data);
                                        temp_bo_jt = bo_jt->BO;
                                        temp = CUBE( temp_bo_jt );
                                        pBOjt7 = temp * temp * temp_bo_jt;

                                        // fprintf( out_control->eval, "%6d%12.8f\n",
                                        // workspace->orig_id[ bond_list[t].nbr ],
                                        //    (CEval6 * pBOjt7) );

#ifdef _OPENMP
//                                        #pragma omp atomic
#endif
                                        bo_jt->Cdbo += (CEval6 * pBOjt7);
#ifdef _OPENMP
//                                        #pragma omp atomic
#endif
                                        bo_jt->Cdbopi += CEval5;
#ifdef _OPENMP
//                                        #pragma omp atomic
#endif
                                        bo_jt->Cdbopi2 += CEval5;
                                    }


                                    if ( control->ensemble == NVE || control->ensemble == NVT  || control->ensemble == bNVT)
                                    {
                                        rvec_ScaledAdd( *f_i, CEval8, p_ijk->dcos_di );
                                        rvec_ScaledAdd( *f_j, CEval8, p_ijk->dcos_dj );
                                        rvec_ScaledAdd( *f_k, CEval8, p_ijk->dcos_dk );
                                    }
                                    else
                                    {
                                        /* terms not related to bond order derivatives
                                           are added directly into
                                           forces and pressure vector/tensor */
                                        rvec_Scale( force, CEval8, p_ijk->dcos_di );
                                        rvec_Add( *f_i, force );
                                        rvec_iMultiply( ext_press, pbond_ij->rel_box, force );
#ifdef _OPENMP
//                                        #pragma omp critical (Three_Body_Interactions_ext_press)
#endif
                                        {
                                            rvec_Add( data->ext_press, ext_press );
                                        }

                                        rvec_ScaledAdd( *f_j, CEval8, p_ijk->dcos_dj );

                                        rvec_Scale( force, CEval8, p_ijk->dcos_dk );
                                        rvec_Add( *f_k, force );
                                        rvec_iMultiply( ext_press, pbond_jk->rel_box, force );
#ifdef _OPENMP
//                                        #pragma omp critical (Three_Body_Interactions_ext_press)
#endif
                                        {
                                            rvec_Add( data->ext_press, ext_press );
                                        }

                                        /* This part is for a fully-flexible box */
                                        /* rvec_OuterProduct( temp_rtensor,
                                           p_ijk->dcos_di, system->atoms[i].x );
                                           rtensor_Scale( total_rtensor, +CEval8, temp_rtensor );

                                           rvec_OuterProduct( temp_rtensor,
                                           p_ijk->dcos_dj, system->atoms[j].x );
                                           rtensor_ScaledAdd(total_rtensor, CEval8, temp_rtensor);

                                           rvec_OuterProduct( temp_rtensor,
                                           p_ijk->dcos_dk, system->atoms[k].x );
                                           rtensor_ScaledAdd(total_rtensor, CEval8, temp_rtensor);

                                           if( pbond_ij->imaginary || pbond_jk->imaginary )
                                           rtensor_ScaledAdd( data->flex_bar.P,
                                           -1.0, total_rtensor );
                                           else
                                           rtensor_Add( data->flex_bar.P, total_rtensor ); */
                                    }

#ifdef TEST_ENERGY
                                    fprintf( out_control->eval,
                                             //"%6d%6d%6d%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e",
                                             "%6d%6d%6d%23.15e%23.15e%23.15e\n",
                                             i + 1, j + 1, k + 1,
                                             //workspace->orig_id[i]+1,
                                             //workspace->orig_id[j]+1,
                                             //workspace->orig_id[k]+1,
                                             //workspace->Delta_boc[j],
                                             RAD2DEG(theta), /*BOA_ij, BOA_jk, */
                                             e_ang, data->E_Ang );

                                    /*fprintf( out_control->eval,
                                      "%23.15e%23.15e%23.15e%23.15e",
                                      p_val3, p_val4, BOA_ij, BOA_jk );
                                      fprintf( out_control->eval,
                                      "%23.15e%23.15e%23.15e%23.15e",
                                      f7_ij, f7_jk, f8_Dj, expval12theta );
                                      fprintf( out_control->eval,
                                      "%23.15e%23.15e%23.15e%23.15e%23.15e\n",
                                      CEval1, CEval2, CEval3, CEval4, CEval5
                                      //CEval6, CEval7, CEval8  );*/

                                    /*fprintf( out_control->eval,
                                      "%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e%23.15e\n",
                                      -p_ijk->dcos_di[0]/sin_theta,
                                      -p_ijk->dcos_di[1]/sin_theta,
                                      -p_ijk->dcos_di[2]/sin_theta,
                                      -p_ijk->dcos_dj[0]/sin_theta,
                                      -p_ijk->dcos_dj[1]/sin_theta,
                                      -p_ijk->dcos_dj[2]/sin_theta,
                                      -p_ijk->dcos_dk[0]/sin_theta,
                                      -p_ijk->dcos_dk[1]/sin_theta,
                                      -p_ijk->dcos_dk[2]/sin_theta );*/

                                    /* fprintf( out_control->epen,
                                       "%23.15e%23.15e%23.15e\n",
                                       CEpen1, CEpen2, CEpen3 );
                                       fprintf( out_control->epen,
                                       "%6d%6d%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n",
                                       workspace->orig_id[i],  workspace->orig_id[j],
                                       workspace->orig_id[k], RAD2DEG(theta),
                                       BOA_ij, BOA_jk, e_pen, data->E_Pen ); */

                                    fprintf( out_control->ecoa,
                                             "%6d%6d%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n",
                                             workspace->orig_id[i],
                                             workspace->orig_id[j],
                                             workspace->orig_id[k],
                                             RAD2DEG(theta), BOA_ij, BOA_jk,
                                             e_coa, data->E_Coa );
#endif

#ifdef TEST_FORCES
                                    /* angle forces */
                                    Add_dBO( system, lists, j, pi, CEval1, workspace->f_ang );
                                    Add_dBO( system, lists, j, pk, CEval2, workspace->f_ang );
                                    Add_dDelta( system, lists,
                                                j, CEval3 + CEval7, workspace->f_ang );

                                    for ( t = start_j; t < end_j; ++t )
                                    {
                                        pbond_jt = &( bond_list[t] );
                                        bo_jt = &(pbond_jt->bo_data);
                                        temp_bo_jt = bo_jt->BO;
                                        temp = CUBE( temp_bo_jt );
                                        pBOjt7 = temp * temp * temp_bo_jt;

                                        Add_dBO( system, lists, j, t, pBOjt7 * CEval6,
                                                 workspace->f_ang );
                                        Add_dBOpinpi2( system, lists, j, t,
                                                       CEval5, CEval5,
                                                       workspace->f_ang, workspace->f_ang );
                                    }

                                    rvec_ScaledAdd( workspace->f_ang[i], CEval8, p_ijk->dcos_di );
                                    rvec_ScaledAdd( workspace->f_ang[j], CEval8, p_ijk->dcos_dj );
                                    rvec_ScaledAdd( workspace->f_ang[k], CEval8, p_ijk->dcos_dk );
                                    /* end angle forces */

                                    /* penalty forces */
                                    Add_dDelta( system, lists, j, CEpen1, workspace->f_pen );
                                    Add_dBO( system, lists, j, pi, CEpen2, workspace->f_pen );
                                    Add_dBO( system, lists, j, pk, CEpen3, workspace->f_pen );
                                    /* end penalty forces */

                                    /* coalition forces */
                                    Add_dBO( system, lists,
                                             j, pi, CEcoa1 - CEcoa4, workspace->f_coa );
                                    Add_dBO( system, lists,
                                             j, pk, CEcoa2 - CEcoa5, workspace->f_coa );
                                    Add_dDelta( system, lists, j, CEcoa3, workspace->f_coa );
                                    Add_dDelta( system, lists, i, CEcoa4, workspace->f_coa );
                                    Add_dDelta( system, lists, k, CEcoa5, workspace->f_coa );
                                    /* end coalition forces */
#endif
                                }
                            }
                        }
                    }
                }

                Set_End_Index( pi, num_thb_intrs, thb_intrs );
            }
        }
    }

    data->E_Ang += e_ang_total;
    data->E_Pen += e_pen_total;
    data->E_Coa += e_coa_total;

    if ( num_thb_intrs >= thb_intrs->total_intrs * DANGER_ZONE )
    {
        workspace->realloc.num_3body = num_thb_intrs;
        if ( num_thb_intrs > thb_intrs->total_intrs )
        {
            fprintf( stderr, "step%d-ran out of space on angle_list: top=%d, max=%d",
                     data->step, num_thb_intrs, thb_intrs->total_intrs );
            exit( INSUFFICIENT_MEMORY );
        }
    }

    //fprintf( stderr,"%d: Number of angle interactions: %d\n",
    // data->step, num_thb_intrs );
#ifdef TEST_ENERGY
    fprintf( stderr, "Number of angle interactions: %d\n", num_thb_intrs );

    fprintf( stderr, "Angle Energy:%g\t Penalty Energy:%g\t Coalition Energy:%g\n",
             data->E_Ang, data->E_Pen, data->E_Coa );

    fprintf( stderr, "3body: ext_press (%23.15e %23.15e %23.15e)\n",
             data->ext_press[0], data->ext_press[1], data->ext_press[2] );
#endif
}


void Hydrogen_Bonds( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        list **lists, output_controls *out_control )
{
    real e_hb_total;

    e_hb_total = 0.0;

#ifdef _OPENMP
    #pragma omp parallel default(shared) reduction(+: e_hb_total)
#endif
    {
        int i, j, k, pi, pk, itr, top;
        int type_i, type_j, type_k;
        int start_j, end_j, hb_start_j, hb_end_j;
        int hblist[MAX_BONDS];
#ifdef TEST_FORCES
        int num_hb_intrs;
#endif
        real r_ij, r_jk, theta, cos_theta, sin_xhz4, cos_xhz1, sin_theta2;
        real e_hb, exp_hb2, exp_hb3, CEhb1, CEhb2, CEhb3;
        rvec dcos_theta_di, dcos_theta_dj, dcos_theta_dk;
        rvec dvec_jk, force, ext_press;
        ivec rel_jk;
        //rtensor temp_rtensor, total_rtensor;
        hbond_parameters *hbp;
        bond_order_data *bo_ij;
        bond_data *pbond_ij;
        far_neighbor_data *nbr_jk;
        list *bonds, *hbonds;
        bond_data *bond_list;
        hbond_data *hbond_list;
        rvec *f_i, *f_j, *f_k;
#ifdef _OPENMP
        int tid = omp_get_thread_num( );
#endif

#ifdef TEST_FORCES
        num_hb_intrs = 0;
#endif
        bonds = (*lists) + BONDS;
        bond_list = bonds->select.bond_list;
        hbonds = (*lists) + HBONDS;
        hbond_list = hbonds->select.hbond_list;

        /* loops below discover the Hydrogen bonds between i-j-k triplets.
           here j is H atom and there has to be some bond between i and j.
           Hydrogen bond is between j and k.
           so in this function i->X, j->H, k->Z when we map
           variables onto the ones in the handout.*/
#ifdef _OPENMP
        #pragma omp for schedule(guided)
#endif
        for ( j = 0; j < system->N; ++j )
        {
            /* j must be H */
            if ( system->reaxprm.sbp[system->atoms[j].type].p_hbond == 1 )
            {
                /* set j's variables */
                type_j = system->atoms[j].type;
                start_j = Start_Index( j, bonds );
                end_j = End_Index( j, bonds );
                hb_start_j = Start_Index( workspace->hbond_index[j], hbonds );
                hb_end_j = End_Index( workspace->hbond_index[j], hbonds );
#ifdef _OPENMP
                f_j = &(workspace->f_local[tid * system->N + j]);
#else
                f_j = &(system->atoms[j].f);
#endif

                top = 0;
                for ( pi = start_j; pi < end_j; ++pi )
                {
                    pbond_ij = &( bond_list[pi] );
                    i = pbond_ij->nbr;
                    bo_ij = &(pbond_ij->bo_data);
                    type_i = system->atoms[i].type;

                    if ( system->reaxprm.sbp[type_i].p_hbond == 2 &&
                            bo_ij->BO >= HB_THRESHOLD )
                    {
                        hblist[top++] = pi;
                    }
                }

                for ( pk = hb_start_j; pk < hb_end_j; ++pk )
                {
                    /* set k's varibles */
                    k = hbond_list[pk].nbr;
                    type_k = system->atoms[k].type;
                    nbr_jk = hbond_list[pk].ptr;
                    r_jk = nbr_jk->d;
                    rvec_Scale( dvec_jk, hbond_list[pk].scl, nbr_jk->dvec );
#ifdef _OPENMP
                    f_k = &(workspace->f_local[tid * system->N + k]);
#else
                    f_k = &(system->atoms[k].f);
#endif

                    for ( itr = 0; itr < top; ++itr )
                    {
                        pi = hblist[itr];
                        pbond_ij = &( bond_list[pi] );
                        i = pbond_ij->nbr;

                        if ( i != k )
                        {
                            bo_ij = &(pbond_ij->bo_data);
                            type_i = system->atoms[i].type;
                            r_ij = pbond_ij->d;
                            hbp = &(system->reaxprm.hbp[ type_i ][ type_j ][ type_k ]);
#ifdef _OPENMP
                            f_i = &(workspace->f_local[tid * system->N + i]);
#else
                            f_i = &(system->atoms[i].f);
#endif

#ifdef TEST_FORCES
                            ++num_hb_intrs;
#endif

                            Calculate_Theta( pbond_ij->dvec, pbond_ij->d, dvec_jk, r_jk,
                                    &theta, &cos_theta );
                            /* the derivative of cos(theta) */
                            Calculate_dCos_Theta( pbond_ij->dvec, pbond_ij->d, dvec_jk, r_jk,
                                    &dcos_theta_di, &dcos_theta_dj, &dcos_theta_dk );

                            /* hydrogen bond energy */
                            sin_theta2 = SIN( theta / 2.0 );
                            sin_xhz4 = SQR( sin_theta2 );
                            sin_xhz4 *= sin_xhz4;
                            cos_xhz1 = ( 1.0 - cos_theta );
                            exp_hb2 = EXP( -hbp->p_hb2 * bo_ij->BO );
                            exp_hb3 = EXP( -hbp->p_hb3 * ( hbp->r0_hb / r_jk +
                                        r_jk / hbp->r0_hb - 2.0 ) );

                            e_hb = hbp->p_hb1 * (1.0 - exp_hb2) * exp_hb3 * sin_xhz4;
                            e_hb_total += e_hb;

                            CEhb1 = hbp->p_hb1 * hbp->p_hb2 * exp_hb2 * exp_hb3 * sin_xhz4;
                            CEhb2 = -hbp->p_hb1 / 2.0 * (1.0 - exp_hb2) * exp_hb3 * cos_xhz1;
                            CEhb3 = -hbp->p_hb3 * e_hb * (-hbp->r0_hb / SQR( r_jk ) +
                                    1.0 / hbp->r0_hb);

                            /* hydrogen bond forces */
                            /* dbo term,
                             * note: safe to update across threads as this points
                             * to the bond_order_data struct inside atom j's list,
                             * and threads are partitioned across all j's */
#ifdef _OPENMP
                            #pragma omp atomic
#endif
                            bo_ij->Cdbo += CEhb1;

                            if ( control->ensemble == NVE || control->ensemble == NVT  || control->ensemble == bNVT)
                            {
                                /* dcos terms */
                                rvec_ScaledAdd( *f_i, +CEhb2, dcos_theta_di );
                                rvec_ScaledAdd( *f_j, +CEhb2, dcos_theta_dj );
                                rvec_ScaledAdd( *f_k, +CEhb2, dcos_theta_dk );

                                /* dr terms */
                                rvec_ScaledAdd( *f_j, -CEhb3 / r_jk, dvec_jk );
                                rvec_ScaledAdd( *f_k, +CEhb3 / r_jk, dvec_jk );
                            }
                            else
                            {
                                /* for pressure coupling, terms that are not related
                                   to bond order derivatives are added directly into
                                   pressure vector/tensor */

                                /* dcos terms */
                                rvec_Scale( force, +CEhb2, dcos_theta_di );
                                rvec_Add( *f_i, force );
                                rvec_iMultiply( ext_press, pbond_ij->rel_box, force );
#ifdef _OPENMP
                                #pragma omp critical (Hydrogen_Bonds_ext_press)
#endif
                                {
                                    rvec_ScaledAdd( data->ext_press, 1.0, ext_press );
                                }

                                rvec_ScaledAdd( *f_j, +CEhb2, dcos_theta_dj );

                                ivec_Scale( rel_jk, hbond_list[pk].scl, nbr_jk->rel_box );
                                rvec_Scale( force, +CEhb2, dcos_theta_dk );
                                rvec_Add( *f_k, force );
                                rvec_iMultiply( ext_press, rel_jk, force );
#ifdef _OPENMP
                                #pragma omp critical (Hydrogen_Bonds_ext_press)
#endif
                                {
                                    rvec_ScaledAdd( data->ext_press, 1.0, ext_press );
                                }

                                /* dr terms */
                                rvec_ScaledAdd( *f_j, -CEhb3 / r_jk, dvec_jk );

                                rvec_Scale( force, CEhb3 / r_jk, dvec_jk );
                                rvec_Add( *f_k, force );
                                rvec_iMultiply( ext_press, rel_jk, force );
#ifdef _OPENMP
                                #pragma omp critical (Hydrogen_Bonds_ext_press)
#endif
                                {
                                    rvec_ScaledAdd( data->ext_press, 1.0, ext_press );
                                }

                                /* This part is intended for a fully-flexible box */
                                /* rvec_OuterProduct( temp_rtensor,
                                   dcos_theta_di, system->atoms[i].x );
                                   rtensor_Scale( total_rtensor, -CEhb2, temp_rtensor );

                                   rvec_ScaledSum( temp_rvec, -CEhb2, dcos_theta_dj,
                                   -CEhb3/r_jk, pbond_jk->dvec );
                                   rvec_OuterProduct( temp_rtensor,
                                   temp_rvec, system->atoms[j].x );
                                   rtensor_Add( total_rtensor, temp_rtensor );

                                   rvec_ScaledSum( temp_rvec, -CEhb2, dcos_theta_dk,
                                   +CEhb3/r_jk, pbond_jk->dvec );
                                   rvec_OuterProduct( temp_rtensor,
                                   temp_rvec, system->atoms[k].x );
                                   rtensor_Add( total_rtensor, temp_rtensor );

                                   if( pbond_ij->imaginary || pbond_jk->imaginary )
                                   rtensor_ScaledAdd( data->flex_bar.P, -1.0, total_rtensor );
                                   else
                                   rtensor_Add( data->flex_bar.P, total_rtensor ); */
                            }

#ifdef TEST_ENERGY
                            /*fprintf( out_control->ehb,
                              "%23.15e%23.15e%23.15e\n%23.15e%23.15e%23.15e\n%23.15e%23.15e%23.15e\n",
                              dcos_theta_di[0], dcos_theta_di[1], dcos_theta_di[2],
                              dcos_theta_dj[0], dcos_theta_dj[1], dcos_theta_dj[2],
                              dcos_theta_dk[0], dcos_theta_dk[1], dcos_theta_dk[2]);
                              fprintf( out_control->ehb, "%23.15e%23.15e%23.15e\n",
                              CEhb1, CEhb2, CEhb3 ); */
                            fprintf( stderr, //out_control->ehb,
                                     "%6d%6d%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n",
                                     workspace->orig_id[i],
                                     workspace->orig_id[j],
                                     workspace->orig_id[k],
                                     r_jk, theta, bo_ij->BO, e_hb, data->E_HB );
#endif

#ifdef TEST_FORCES
                            /* dbo term */
                            Add_dBO( system, lists, j, pi, +CEhb1, workspace->f_hb );
                            /* dcos terms */
                            rvec_ScaledAdd( workspace->f_hb[i], +CEhb2, dcos_theta_di );
                            rvec_ScaledAdd( workspace->f_hb[j], +CEhb2, dcos_theta_dj );
                            rvec_ScaledAdd( workspace->f_hb[k], +CEhb2, dcos_theta_dk );
                            /* dr terms */
                            rvec_ScaledAdd( workspace->f_hb[j], -CEhb3 / r_jk, dvec_jk );
                            rvec_ScaledAdd( workspace->f_hb[k], +CEhb3 / r_jk, dvec_jk );
#endif
                        }
                    }
                }
            }
        }
    }

    data->E_HB += e_hb_total;

#ifdef TEST_FORCES
    fprintf( stderr, "Number of hydrogen bonds: %d\n", num_hb_intrs );
    fprintf( stderr, "Hydrogen Bond Energy: %g\n", data->E_HB );
#endif
}
