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
#include "valence_angles.h"

#include "bond_orders.h"
#include "list.h"
#include "vector.h"
#elif defined(LAMMPS_REAX)
#include "reax_valence_angles.h"

#include "reax_bond_orders.h"
#include "reax_list.h"
#include "reax_vector.h"
#endif


/* calculates the theta angle between i-j-k */
void Calculate_Theta( rvec dvec_ji, real d_ji, rvec dvec_jk, real d_jk,
        real *theta, real *cos_theta )
{
    assert( d_ji > 0.0 );
    assert( d_jk > 0.0 );

    *cos_theta = rvec_Dot( dvec_ji, dvec_jk ) / ( d_ji * d_jk );

    if ( *cos_theta > 1.0 )
    {
        *cos_theta = 1.0;
    }
    if ( *cos_theta < -1.0 )
    {
        *cos_theta = -1.0;
    }

    *theta = ACOS( *cos_theta );
}


/* calculates the derivative of the cosine of the angle between i-j-k */
void Calculate_dCos_Theta( rvec dvec_ji, real d_ji, rvec dvec_jk, real d_jk,
        rvec* dcos_theta_di, rvec* dcos_theta_dj, rvec* dcos_theta_dk )
{
    int t;
    real sqr_d_ji, sqr_d_jk, inv_dists, inv_dists3, dot_dvecs, Cdot_inv3;

    assert( d_ji > 0.0 );
    assert( d_jk > 0.0 );

    sqr_d_ji = SQR( d_ji );
    sqr_d_jk = SQR( d_jk );
    inv_dists = 1.0 / (d_ji * d_jk);
    inv_dists3 = POW( inv_dists, 3 );
    dot_dvecs = rvec_Dot( dvec_ji, dvec_jk );
    Cdot_inv3 = dot_dvecs * inv_dists3;

    for ( t = 0; t < 3; ++t )
    {
        (*dcos_theta_di)[t] = dvec_jk[t] * inv_dists
            - Cdot_inv3 * sqr_d_jk * dvec_ji[t];

        (*dcos_theta_dj)[t] = -(dvec_jk[t] + dvec_ji[t]) * inv_dists
            + Cdot_inv3 * ( sqr_d_jk * dvec_ji[t] + sqr_d_ji * dvec_jk[t] );

        (*dcos_theta_dk)[t] = dvec_ji[t] * inv_dists
            - Cdot_inv3 * sqr_d_ji * dvec_jk[t];
    }
}


/* this is a 3-body interaction in which the main role is
   played by j which sits in the middle of the other two. */
void Valence_Angles( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, j, pi, k, pk, t;
    int type_i, type_j, type_k;
    int start_j, end_j, start_pk, end_pk;
    int cnt, x, num_thb_intrs;
    real temp, temp_bo_jt, pBOjt7;
    real p_val1, p_val2, p_val3, p_val4, p_val5;
    real p_val6, p_val7, p_val8, p_val9, p_val10;
    real p_pen1, p_pen2, p_pen3, p_pen4;
    real p_coa1, p_coa2, p_coa3, p_coa4;
    real trm8, expval6, expval7, expval2theta, expval12theta, exp3ij, exp3jk;
    real exp_pen2ij, exp_pen2jk, exp_pen3, exp_pen4, trm_pen34, exp_coa2;
    real dSBO1, dSBO2, SBO, SBO2, CSBO2, SBOp, prod_SBO, vlpadj;
    real CEval1, CEval2, CEval3, CEval4, CEval5, CEval6, CEval7, CEval8;
    real CEpen1, CEpen2, CEpen3;
    real e_ang, e_coa, e_pen;
    real CEcoa1, CEcoa2, CEcoa3, CEcoa4, CEcoa5;
    real Cf7ij, Cf7jk, Cf8j, Cf9j;
    real f7_ij, f7_jk, f8_Dj, f9_Dj;
    real Ctheta_0, theta_0, theta_00, theta, cos_theta, sin_theta;
    real r_ij, r_jk;
    real BOA_ij, BOA_jk;
    rvec force, ext_press;
    three_body_header *thbh;
    three_body_parameters *thbp;
    three_body_interaction_data *p_ijk, *p_kji;
    bond_data *pbond_ij, *pbond_jk, *pbond_jt;
    bond_order_data *bo_ij, *bo_jk, *bo_jt;
    reax_list *bonds, *thb_intrs;

    bonds = lists[BONDS];
    thb_intrs = lists[THREE_BODIES];
    p_pen2 = system->reax_param.gp.l[19];
    p_pen3 = system->reax_param.gp.l[20];
    p_pen4 = system->reax_param.gp.l[21];
    p_coa2 = system->reax_param.gp.l[2];
    p_coa3 = system->reax_param.gp.l[38];
    p_coa4 = system->reax_param.gp.l[30];
    p_val6 = system->reax_param.gp.l[14];
    p_val8 = system->reax_param.gp.l[33];
    p_val9 = system->reax_param.gp.l[16];
    p_val10 = system->reax_param.gp.l[17];
    num_thb_intrs = 0;

    for ( x = 0; x < thb_intrs->n; ++x )
    {
        Set_Start_Index( x, 0, thb_intrs );
    }
    for ( x = 0; x < thb_intrs->n; ++x )
    {
        Set_End_Index( x, 0, thb_intrs );
    }

    for ( j = 0; j < system->N; ++j )
    {
        type_j = system->my_atoms[j].type;
        start_j = Start_Index( j, bonds );
        end_j = End_Index( j, bonds );

        p_val3 = system->reax_param.sbp[ type_j ].p_val3;
        p_val5 = system->reax_param.sbp[ type_j ].p_val5;

        /* sum of pi and pi-pi BO terms for all neighbors of atom j,
         * used in determining the equilibrium angle between i-j-k */
        SBOp = 0.0;
        /* product of e^{-BO_j^8} terms for all neighbors of atom j,
         * used in determining the equilibrium angle between i-j-k */
        prod_SBO = 1.0;

        for ( t = start_j; t < end_j; ++t )
        {
            bo_jt = &bonds->bond_list[t].bo_data;
            SBOp += bo_jt->BO_pi + bo_jt->BO_pi2;
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
         * restrictions here. such a restriction would prevent us from producing
         * all 4-body intrs correctly */
        for ( pi = start_j; pi < end_j; ++pi )
        {
            Set_Start_Index( pi, num_thb_intrs, thb_intrs );
            pbond_ij = &bonds->bond_list[pi];
            bo_ij = &pbond_ij->bo_data;
            BOA_ij = bo_ij->BO - control->thb_cut;

            if ( BOA_ij >= 0.0 && (j < system->n || pbond_ij->nbr < system->n) )
            {
                i = pbond_ij->nbr;
                r_ij = pbond_ij->d;
                type_i = system->my_atoms[i].type;

                /* first copy 3-body intrs from previously computed ones where i > k.
                 * IMPORTANT: if it is less costly to compute theta and its
                 * derivative, we should definitely re-compute them,
                 * instead of copying!
                 * in the second for-loop below, we compute only new 3-body intrs
                 * where i < k */
                for ( pk = start_j; pk < pi; ++pk )
                {
                    start_pk = Start_Index( pk, thb_intrs );
                    end_pk = End_Index( pk, thb_intrs );

                    for ( t = start_pk; t < end_pk; ++t )
                    {
                        if ( thb_intrs->three_body_list[t].thb == i )
                        {
                            p_ijk = &thb_intrs->three_body_list[num_thb_intrs];
                            p_kji = &thb_intrs->three_body_list[t];

                            p_ijk->thb = bonds->bond_list[pk].nbr;
                            p_ijk->pthb = pk;
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
                    pbond_jk = &bonds->bond_list[pk];
                    bo_jk = &pbond_jk->bo_data;
                    BOA_jk = bo_jk->BO - control->thb_cut;

                    if ( BOA_jk < 0.0 )
                    {
                        continue;
                    }

                    k = pbond_jk->nbr;
                    type_k = system->my_atoms[k].type;
                    p_ijk = &thb_intrs->three_body_list[num_thb_intrs];

                    Calculate_Theta( pbond_ij->dvec, pbond_ij->d,
                            pbond_jk->dvec, pbond_jk->d,
                            &theta, &cos_theta );

                    Calculate_dCos_Theta( pbond_ij->dvec, pbond_ij->d,
                            pbond_jk->dvec, pbond_jk->d,
                            &p_ijk->dcos_di, &p_ijk->dcos_dj,
                            &p_ijk->dcos_dk );

                    p_ijk->thb = k;
                    p_ijk->pthb = pk;
                    p_ijk->theta = theta;

                    sin_theta = SIN( theta );
                    if ( sin_theta < 1.0e-5 )
                    {
                        sin_theta = 1.0e-5;
                    }

                    ++num_thb_intrs;


                    /* Fortran ReaxFF code hard-codes the constant below
                     * as of 2019-02-27, so use that for now */
                    if ( j < system->n && BOA_jk >= 0.0 && (bo_ij->BO * bo_jk->BO) >= 0.00001 )
//                    if ( j < system->n && BOA_jk >= 0.0 && (bo_ij->BO * bo_jk->BO) > SQR(control->thb_cut) )
                    {
                        r_jk = pbond_jk->d;
                        thbh = &system->reax_param.thbp[ type_i ][ type_j ][ type_k ];

                        for ( cnt = 0; cnt < thbh->cnt; ++cnt )
                        {
                            /* valence angle does not exist in the force field */
                            if ( FABS(thbh->prm[cnt].p_val1) < 0.001 )
                            {
                                continue;
                            }

                            thbp = &thbh->prm[cnt];

                            /* calculate valence angle energy */
                            p_val1 = thbp->p_val1;
                            p_val2 = thbp->p_val2;
                            p_val4 = thbp->p_val4;
                            p_val7 = thbp->p_val7;
                            theta_00 = thbp->theta_00;

                            exp3ij = EXP( -p_val3 * POW( BOA_ij, p_val4 ) );
                            f7_ij = 1.0 - exp3ij;
                            Cf7ij = p_val3 * p_val4
                                * POW( BOA_ij, p_val4 - 1.0 ) * exp3ij;

                            exp3jk = EXP( -p_val3 * POW( BOA_jk, p_val4 ) );
                            f7_jk = 1.0 - exp3jk;
                            Cf7jk = p_val3 * p_val4 *
                                POW( BOA_jk, p_val4 - 1.0 ) * exp3jk;

                            expval7 = EXP( -p_val7 * workspace->Delta_boc[j] );
                            trm8 = 1.0 + expval6 + expval7;
                            f8_Dj = p_val5 - (p_val5 - 1.0) * (2.0 + expval6) / trm8;
                            Cf8j = ( (1.0 - p_val5) / SQR(trm8) )
                                * (p_val6 * expval6 * trm8
                                - (2.0 + expval6) * ( p_val6 * expval6 - p_val7 * expval7 ));

                            theta_0 = 180.0 - theta_00 * (1.0 - EXP(-p_val10 * (2.0 - SBO2)));
                            theta_0 = DEG2RAD( theta_0 );

                            expval2theta = p_val1 * EXP(-p_val2 * SQR(theta_0 - theta));
                            if ( p_val1 >= 0 )
                            {
                                expval12theta = p_val1 - expval2theta;
                            }
                            /* To avoid linear Me-H-Me angles (6/6/06) */
                            else
                            {
                                expval12theta = -expval2theta;
                            }

                            CEval1 = Cf7ij * f7_jk * f8_Dj * expval12theta;
                            CEval2 = Cf7jk * f7_ij * f8_Dj * expval12theta;
                            CEval3 = Cf8j * f7_ij * f7_jk * expval12theta;
                            CEval4 = 2.0 * p_val2 * f7_ij * f7_jk * f8_Dj
                                * expval2theta * (theta_0 - theta);

                            Ctheta_0 = p_val10 * DEG2RAD(theta_00)
                                * EXP( -p_val10 * (2.0 - SBO2) );

                            CEval5 = CEval4 * Ctheta_0 * CSBO2;
                            CEval6 = CEval5 * dSBO1;
                            CEval7 = CEval5 * dSBO2;
                            CEval8 = CEval4 / sin_theta;

                            e_ang = f7_ij * f7_jk * f8_Dj * expval12theta;
                            data->my_en.e_ang += e_ang;

                            /* calculate penalty for double bonds in valency angles */
                            p_pen1 = thbp->p_pen1;

                            exp_pen2ij = EXP( -p_pen2 * SQR( BOA_ij - 2.0 ) );
                            exp_pen2jk = EXP( -p_pen2 * SQR( BOA_jk - 2.0 ) );
                            exp_pen3 = EXP( -p_pen3 * workspace->Delta[j] );
                            exp_pen4 = EXP(  p_pen4 * workspace->Delta[j] );
                            trm_pen34 = 1.0 + exp_pen3 + exp_pen4;
                            f9_Dj = ( 2.0 + exp_pen3 ) / trm_pen34;
                            Cf9j = (-p_pen3 * exp_pen3 * trm_pen34
                                    - (2.0 + exp_pen3) * ( -p_pen3 * exp_pen3
                                        + p_pen4 * exp_pen4 )) / SQR( trm_pen34 );

                            e_pen = p_pen1 * f9_Dj * exp_pen2ij * exp_pen2jk;
                            data->my_en.e_pen += e_pen;

                            CEpen1 = e_pen * Cf9j / f9_Dj;
                            temp = -2.0 * p_pen2 * e_pen;
                            CEpen2 = temp * (BOA_ij - 2.0);
                            CEpen3 = temp * (BOA_jk - 2.0);

                            /* calculate valency angle conjugation energy */
                            p_coa1 = thbp->p_coa1;

                            exp_coa2 = EXP( p_coa2 * workspace->Delta_boc[j] );
                            e_coa = p_coa1
                                * EXP( -p_coa4 * SQR(BOA_ij - 1.5) )
                                * EXP( -p_coa4 * SQR(BOA_jk - 1.5) )
                                * EXP( -p_coa3 * SQR(workspace->total_bond_order[i] - BOA_ij) )
                                * EXP( -p_coa3 * SQR(workspace->total_bond_order[k] - BOA_jk) )
                                / (1.0 + exp_coa2);
                            data->my_en.e_coa += e_coa;

                            CEcoa1 = -2.0 * p_coa4 * (BOA_ij - 1.5) * e_coa;
                            CEcoa2 = -2.0 * p_coa4 * (BOA_jk - 1.5) * e_coa;
                            CEcoa3 = -p_coa2 * exp_coa2 * e_coa / (1.0 + exp_coa2);
                            CEcoa4 = -2.0 * p_coa3 * (workspace->total_bond_order[i] - BOA_ij) * e_coa;
                            CEcoa5 = -2.0 * p_coa3 * (workspace->total_bond_order[k] - BOA_jk) * e_coa;

                            /* calculate force contributions */
                            bo_ij->Cdbo += CEval1 + CEpen2 + (CEcoa1 - CEcoa4);
                            bo_jk->Cdbo += CEval2 + CEpen3 + (CEcoa2 - CEcoa5);
                            workspace->CdDelta[j] += ((CEval3 + CEval7) + CEpen1 + CEcoa3);
                            workspace->CdDelta[i] += CEcoa4;
                            workspace->CdDelta[k] += CEcoa5;

                            for ( t = start_j; t < end_j; ++t )
                            {
                                pbond_jt = &bonds->bond_list[t];
                                bo_jt = &pbond_jt->bo_data;
                                temp_bo_jt = bo_jt->BO;
                                temp = CUBE( temp_bo_jt );
                                pBOjt7 = temp * temp * temp_bo_jt;

                                bo_jt->Cdbo += CEval6 * pBOjt7;
                                bo_jt->Cdbopi += CEval5;
                                bo_jt->Cdbopi2 += CEval5;
                            }

                            if ( control->virial == 0 )
                            {
                                rvec_ScaledAdd( workspace->f[i], CEval8, p_ijk->dcos_di );
                                rvec_ScaledAdd( workspace->f[j], CEval8, p_ijk->dcos_dj );
                                rvec_ScaledAdd( workspace->f[k], CEval8, p_ijk->dcos_dk );
                            }
                            else
                            {
                                /* terms not related to bond order derivatives are
                                   added directly into forces and pressure vector/tensor */
                                rvec_Scale( force, CEval8, p_ijk->dcos_di );
                                rvec_Add( workspace->f[i], force );
                                rvec_iMultiply( ext_press, pbond_ij->rel_box, force );
                                rvec_Add( data->my_ext_press, ext_press );

                                rvec_ScaledAdd( workspace->f[j], CEval8, p_ijk->dcos_dj );

                                rvec_Scale( force, CEval8, p_ijk->dcos_dk );
                                rvec_Add( workspace->f[k], force );
                                rvec_iMultiply( ext_press, pbond_jk->rel_box, force );
                                rvec_Add( data->my_ext_press, ext_press );
                            }

#if defined(TEST_ENERGY)
                            /*fprintf( out_control->eval, "%12.8f%12.8f%12.8f%12.8f\n",
                              p_val3, p_val4, BOA_ij, BOA_jk );
                            fprintf(out_control->eval, "%13.8f%13.8f%13.8f%13.8f%13.8f\n",
                                workspace->Delta_e[j], workspace->vlpex[j],
                                dSBO1, dSBO2, vlpadj );
                            fprintf( out_control->eval, "%12.8f%12.8f%12.8f%12.8f\n",
                                 f7_ij, f7_jk, f8_Dj, expval12theta );
                            fprintf( out_control->eval,
                                 "%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f%12.8f\n",
                                 CEval1, CEval2, CEval3, CEval4,
                                 CEval5, CEval6, CEval7, CEval8 );

                            fprintf( out_control->eval,
                            "%12.8f%12.8f%12.8f\n%12.8f%12.8f%12.8f\n%12.8f%12.8f%12.8f\n",
                               p_ijk->dcos_di[0]/sin_theta, p_ijk->dcos_di[1]/sin_theta,
                               p_ijk->dcos_di[2]/sin_theta,
                               p_ijk->dcos_dj[0]/sin_theta, p_ijk->dcos_dj[1]/sin_theta,
                               p_ijk->dcos_dj[2]/sin_theta,
                               p_ijk->dcos_dk[0]/sin_theta, p_ijk->dcos_dk[1]/sin_theta,
                               p_ijk->dcos_dk[2]/sin_theta);

                            fprintf( out_control->eval,
                                 "%6d%6d%6d%15.8f%15.8f\n",
                                 system->my_atoms[i].orig_id,
                                 system->my_atoms[j].orig_id,
                                 system->my_atoms[k].orig_id,
                                 RAD2DEG(theta), e_ang );*/

                            fprintf( out_control->eval,
                                     //"%6d%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                                     "%6d%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                                     system->my_atoms[i].orig_id,
                                     system->my_atoms[j].orig_id,
                                     system->my_atoms[k].orig_id,
                                     RAD2DEG(theta), theta_0, BOA_ij, BOA_jk,
                                     e_ang, data->my_en.e_ang );

                            fprintf( out_control->epen,
                                     //"%6d%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                                     "%6d%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                                     system->my_atoms[i].orig_id,
                                     system->my_atoms[j].orig_id,
                                     system->my_atoms[k].orig_id,
                                     RAD2DEG(theta), BOA_ij, BOA_jk, e_pen,
                                     data->my_en.e_pen );

                            fprintf( out_control->ecoa,
                                     //"%6d%6d%6d%24.15e%24.15e%24.15e%24.15e%24.15e\n",
                                     "%6d%6d%6d%12.4f%12.4f%12.4f%12.4f%12.4f\n",
                                     system->my_atoms[i].orig_id,
                                     system->my_atoms[j].orig_id,
                                     system->my_atoms[k].orig_id,
                                     RAD2DEG(theta), BOA_ij, BOA_jk,
                                     e_coa, data->my_en.e_coa );
#endif

#if defined(TEST_FORCES)
                            /* angle forces */
                            Add_dBO( system, lists, j, pi, CEval1, workspace->f_ang );
                            Add_dBO( system, lists, j, pk, CEval2, workspace->f_ang );
                            Add_dDelta( system, lists, j,
                                        CEval3 + CEval7, workspace->f_ang );

                            for ( t = start_j; t < end_j; ++t )
                            {
                                pbond_jt = &bonds->bond_list[t];
                                bo_jt = &pbond_jt->bo_data;
                                temp_bo_jt = bo_jt->BO;
                                temp = CUBE( temp_bo_jt );
                                pBOjt7 = temp * temp * temp_bo_jt;

                                Add_dBO( system, lists, j, t, pBOjt7 * CEval6,
                                         workspace->f_ang );
                                Add_dBOpinpi2( system, lists, j, t, CEval5, CEval5,
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
                            Add_dBO( system, lists, j, pi, CEcoa1 - CEcoa4,
                                     workspace->f_coa );
                            Add_dBO( system, lists, j, pk, CEcoa2 - CEcoa5,
                                     workspace->f_coa );
                            Add_dDelta( system, lists, j, CEcoa3, workspace->f_coa );
                            Add_dDelta( system, lists, i, CEcoa4, workspace->f_coa );
                            Add_dDelta( system, lists, k, CEcoa5, workspace->f_coa );
                            /* end coalition forces */
#endif
                        }
                    }
                }
            }

            Set_End_Index( pi, num_thb_intrs, thb_intrs );
        }
    }

    if ( num_thb_intrs >= thb_intrs->num_intrs * DANGER_ZONE )
    {
        workspace->realloc.num_3body = num_thb_intrs;
        if ( num_thb_intrs > thb_intrs->num_intrs )
        {
            fprintf( stderr, "step%d-ran out of space on angle_list: top=%d, max=%d",
                     data->step, num_thb_intrs, thb_intrs->num_intrs );
            MPI_Abort( MPI_COMM_WORLD, INSUFFICIENT_MEMORY );
        }
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "Number of angle interactions: %d\n", num_thb_intrs );
    fprintf( stderr,
             "Angle Energy: %g\t Penalty Energy: %g\t Coalition Energy: %g\t\n",
             data->my_en.e_ang, data->my_en.e_pen, data->my_en.e_coa );

    fprintf( stderr, "3body: ext_press (%12.6f %12.6f %12.6f)\n",
             data->ext_press[0], data->ext_press[1], data->ext_press[2] );
#endif
}
