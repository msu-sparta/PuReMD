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

#include "multi_body.h"

#include "bond_orders.h"
#include "list.h"
#include "lookup.h"
#include "vector.h"


void Atom_Energy( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    real e_lp, e_ov, e_un;
    real p_lp3;
    real p_ovun3, p_ovun4, p_ovun6, p_ovun7, p_ovun8;
    reax_list *bonds;

    bonds = lists[BONDS];
    p_lp3 = system->reax_param.gp.l[5];
    p_ovun3 = system->reax_param.gp.l[32];
    p_ovun4 = system->reax_param.gp.l[31];
    p_ovun6 = system->reax_param.gp.l[6];
    p_ovun7 = system->reax_param.gp.l[8];
    p_ovun8 = system->reax_param.gp.l[9];
    e_lp = 0.0;
    e_ov = 0.0;
    e_un = 0.0;

#if defined(_OPENMP)
    #pragma omp parallel default(shared) reduction(+: e_lp, e_ov, e_un)
#endif
    {
        uint32_t i, j, pj, type_i, type_j;
        real Delta_lpcorr, dfvl;
        real expvd2, inv_expvd2, dElp, CElp, DlpVi;
        real CdDelta_i;
        real Di, vov3, deahu2dbo, deahu2dsbo;
        real CEover1, CEover2, CEover3, CEover4;
        real exp_ovun1, exp_ovun2, sum_ovun1, sum_ovun2;
        real exp_ovun2n, exp_ovun6, exp_ovun8;
        real inv_exp_ovun1, inv_exp_ovun2, inv_exp_ovun2n, inv_exp_ovun8;
        real e_unh, CEunder1, CEunder2, CEunder3, CEunder4;
        real p_lp2;
        real p_ovun2, p_ovun5;
        single_body_parameters *sbp_i;
        two_body_parameters *twbp;
        bond_data *pbond;
        bond_order_data *bo_ij;

#if defined(_OPENMP)
        #pragma omp for
#endif
        for ( i = 0; i < system->N; ++i ) {
#if defined(QMMM)
            if ( system->atoms[i].qmmm_mask == TRUE ) {
#endif
            /* set the parameter pointer */
            type_i = system->atoms[i].type;
            sbp_i = &system->reax_param.sbp[ type_i ];

            /* lone-pair Energy */
            p_lp2 = sbp_i->p_lp2;
            expvd2 = EXP( -75.0 * workspace->Delta_lp[i] );
            inv_expvd2 = 1.0 / (1.0 + expvd2);

            /* calculate the energy */
            e_lp += p_lp2 * workspace->Delta_lp[i] * inv_expvd2;

            dElp = p_lp2 * inv_expvd2
                + 75.0 * p_lp2 * workspace->Delta_lp[i] * expvd2 * SQR(inv_expvd2);
            CElp = dElp * workspace->dDelta_lp[i];

            // lp - 1st term
            CdDelta_i = CElp;

#if defined(TEST_FORCES)
            Add_dDelta( system, lists, i, CElp, workspace->f_lp );  // lp - 1st term
#endif

            /* correction for C2 */
            if ( system->reax_param.gp.l[5] > 0.001
                    && strncmp( system->reax_param.sbp[type_i].name, "C",
                        sizeof(system->reax_param.sbp[type_i].name) ) == 0 ) {
                for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
                    if ( i < bonds->bond_list[pj].nbr ) {
                        j = bonds->bond_list[pj].nbr;
                        type_j = system->atoms[j].type;

                        if ( strncmp( system->reax_param.sbp[type_j].name, "C",
                                    sizeof(system->reax_param.sbp[type_j].name) ) == 0 ) {
                            twbp = &system->reax_param.tbp[type_i][type_j];
                            bo_ij = &bonds->bond_list[pj].bo_data;
                            Di = workspace->Delta[i];
                            vov3 = bo_ij->BO - Di - 0.04 * FOURTH( Di );

                            if ( vov3 > 3.0 ) {
                                e_lp += p_lp3 * SQR(vov3 - 3.0);

                                deahu2dbo = 2.0 * p_lp3 * (vov3 - 3.0);
                                deahu2dsbo = 2.0 * p_lp3 * (vov3 - 3.0)
                                    * (-1.0 - 0.16 * CUBE( Di ));

                                bo_ij->Cdbo += deahu2dbo;
                                CdDelta_i += deahu2dsbo;

#if defined(TEST_FORCES)
                                Add_dBO( system, lists, i, pj, deahu2dbo, workspace->f_lp );
                                Add_dDelta( system, lists, i, deahu2dsbo, workspace->f_lp );
#endif
                            }
                        }

                    }
                }
            }

            workspace->CdDelta[i] += CdDelta_i;
#if defined(QMMM)
            }
#endif
        }

#if defined(_OPENMP)
        #pragma omp barrier
        #pragma omp for
#endif
        for ( i = 0; i < system->N; ++i ) {
#if defined(QMMM)
            if ( system->atoms[i].qmmm_mask == TRUE ) {
#endif
            type_i = system->atoms[i].type;
            sbp_i = &system->reax_param.sbp[ type_i ];

            /* over-coordination energy */
            if ( sbp_i->mass > 21.0 ) {
                dfvl = 0.0;
            } else {
                dfvl = 1.0; // only for 1st-row elements
            }

            p_ovun2 = sbp_i->p_ovun2;
            sum_ovun1 = 0.0;
            sum_ovun2 = 0.0;

            for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
                j = bonds->bond_list[pj].nbr;
                type_j = system->atoms[j].type;
                bo_ij = &bonds->bond_list[pj].bo_data;
                twbp = &system->reax_param.tbp[ type_i ][ type_j ];

                sum_ovun1 += twbp->p_ovun1 * twbp->De_s * bo_ij->BO;
                sum_ovun2 += (workspace->Delta[j] - dfvl * workspace->Delta_lp_temp[j])
                    * ( bo_ij->BO_pi + bo_ij->BO_pi2 );
            }

            exp_ovun1 = p_ovun3 * EXP( p_ovun4 * sum_ovun2 );
            inv_exp_ovun1 = 1.0 / (1.0 + exp_ovun1);
            Delta_lpcorr  = workspace->Delta[i]
                - (dfvl * workspace->Delta_lp_temp[i]) * inv_exp_ovun1;

            exp_ovun2 = EXP( p_ovun2 * Delta_lpcorr );
            inv_exp_ovun2 = 1.0 / (1.0 + exp_ovun2);

            DlpVi = 1.0 / (Delta_lpcorr + sbp_i->valency + 1.0e-8);
            CEover1 = Delta_lpcorr * DlpVi * inv_exp_ovun2;

            e_ov += sum_ovun1 * CEover1;

            CEover2 = sum_ovun1 * DlpVi * inv_exp_ovun2
                * (1.0 - Delta_lpcorr * (DlpVi + p_ovun2 * exp_ovun2 * inv_exp_ovun2));

            CEover3 = CEover2 * (1.0 - dfvl * workspace->dDelta_lp[i] * inv_exp_ovun1);

            CEover4 = CEover2 * (dfvl * workspace->Delta_lp_temp[i])
                * p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1);

            /* under-coordination potential */
            p_ovun2 = sbp_i->p_ovun2;
            p_ovun5 = sbp_i->p_ovun5;

            exp_ovun2n = 1.0 / exp_ovun2;
            exp_ovun6 = EXP( p_ovun6 * Delta_lpcorr );
            exp_ovun8 = p_ovun7 * EXP(p_ovun8 * sum_ovun2);
            inv_exp_ovun2n = 1.0 / (1.0 + exp_ovun2n);
            inv_exp_ovun8 = 1.0 / (1.0 + exp_ovun8);

            e_unh = -p_ovun5 * (1.0 - exp_ovun6) * inv_exp_ovun2n * inv_exp_ovun8;
            e_un += e_unh;

            CEunder1 = inv_exp_ovun2n * (p_ovun5 * p_ovun6 * exp_ovun6 * inv_exp_ovun8
                    + p_ovun2 * e_unh * exp_ovun2n);
            CEunder2 = -e_unh * p_ovun8 * exp_ovun8 * inv_exp_ovun8;
            CEunder3 = CEunder1 * (1.0 - dfvl * workspace->dDelta_lp[i] * inv_exp_ovun1);
            CEunder4 = CEunder1 * (dfvl * workspace->Delta_lp_temp[i])
                * p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1) + CEunder2;

            /* forces */
            // OvCoor - 2nd term, UnCoor - 1st term
            CdDelta_i = CEover3 + CEunder3;

#if defined(TEST_FORCES)
            Add_dDelta( system, lists, i, CEover3, workspace->f_ov );  // OvCoor - 2nd
            Add_dDelta( system, lists, i, CEunder3, workspace->f_un ); // UnCoor - 1st
#endif

            for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
                pbond = &bonds->bond_list[pj];
                j = pbond->nbr;
                type_j = system->atoms[j].type;
                bo_ij = &pbond->bo_data;
                twbp = &system->reax_param.tbp[ type_i ][ type_j ];

                // OvCoor - 1st
                bo_ij->Cdbo += CEover1 * twbp->p_ovun1 * twbp->De_s;
                // OvCoor - 3a, UnCoor - 2a
#if defined(_OPENMP)
                #pragma omp atomic
#endif
                workspace->CdDelta[j] += (CEover4 + CEunder4) * (1.0 - dfvl * workspace->dDelta_lp[j])
                    * (bo_ij->BO_pi + bo_ij->BO_pi2);
                // OvCoor-3b, UnCoor-2b
                bo_ij->Cdbopi += (CEover4 + CEunder4)
                    * (workspace->Delta[j] - dfvl * workspace->Delta_lp_temp[j]);
                // OvCoor-3b, UnCoor-2b
                bo_ij->Cdbopi2 += (CEover4 + CEunder4)
                    * (workspace->Delta[j] - dfvl * workspace->Delta_lp_temp[j]);

#if defined(TEST_FORCES)
                Add_dBO( system, lists, i, pj, CEover1 * twbp->p_ovun1 * twbp->De_s,
                         workspace->f_ov ); // OvCoor - 1st term

                Add_dDelta( system, lists, j,
                            CEover4 * (1.0 - dfvl * workspace->dDelta_lp[j]) *
                            (bo_ij->BO_pi + bo_ij->BO_pi2), workspace->f_ov ); //OvCoor3a

                Add_dBOpinpi2( system, lists, i, pj,
                               CEover4 * (workspace->Delta[j] -
                                          dfvl * workspace->Delta_lp_temp[j]),
                               CEover4 * (workspace->Delta[j] -
                                          dfvl * workspace->Delta_lp_temp[j]),
                               workspace->f_ov, workspace->f_ov ); // OvCoor - 3b

                Add_dDelta( system, lists, j,
                            CEunder4 * (1.0 - dfvl * workspace->dDelta_lp[j]) *
                            (bo_ij->BO_pi + bo_ij->BO_pi2),
                            workspace->f_un ); // UnCoor - 2a

                Add_dBOpinpi2( system, lists, i, pj,
                               CEunder4 * (workspace->Delta[j] -
                                           dfvl * workspace->Delta_lp_temp[j]),
                               CEunder4 * (workspace->Delta[j] -
                                           dfvl * workspace->Delta_lp_temp[j]),
                               workspace->f_un, workspace->f_un ); // UnCoor - 2b
#endif
            }

#if defined(_OPENMP)
            #pragma omp atomic
#endif
            workspace->CdDelta[i] += CdDelta_i;
#if defined(QMMM)
            }
#endif
        }
    }
        
    data->E_Lp = e_lp;
    data->E_Ov = e_ov;
    data->E_Un = e_un;
}
