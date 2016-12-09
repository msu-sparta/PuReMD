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

#include "forces.h"

#include "box.h"
#include "bond_orders.h"
#include "single_body_interactions.h"
#include "two_body_interactions.h"
#include "three_body_interactions.h"
#include "four_body_interactions.h"
#include "index_utils.h"
#include "list.h"
#include "print_utils.h"
#include "qeq.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


void Dummy_Interaction( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, list **lists,
        output_controls *out_control )
{
}


void Init_Bonded_Force_Functions( control_params *control )
{
    Interaction_Functions[0] = Calculate_Bond_Orders;
    Interaction_Functions[1] = Bond_Energy;  //*/Dummy_Interaction;
    Interaction_Functions[2] = LonePair_OverUnder_Coordination_Energy;
    //*/Dummy_Interaction;
    Interaction_Functions[3] = Three_Body_Interactions; //*/Dummy_Interaction;
    Interaction_Functions[4] = Four_Body_Interactions;  //*/Dummy_Interaction;
    if ( control->hb_cut > 0 )
        Interaction_Functions[5] = Hydrogen_Bonds; //*/Dummy_Interaction;
    else Interaction_Functions[5] = Dummy_Interaction;
    Interaction_Functions[6] = Dummy_Interaction; //empty
    Interaction_Functions[7] = Dummy_Interaction; //empty
    Interaction_Functions[8] = Dummy_Interaction; //empty
    Interaction_Functions[9] = Dummy_Interaction; //empty
}


void Compute_Bonded_Forces( reax_system *system, control_params *control,
                            simulation_data *data, static_storage *workspace,
                            list **lists, output_controls *out_control )
{

    int i;
    // real t_start, t_end, t_elapsed;

#ifdef TEST_ENERGY
    /* Mark beginning of a new timestep in each energy file */
    fprintf( out_control->ebond, "step: %d\n%6s%6s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "bo", "ebond", "total" );
    fprintf( out_control->elp, "step: %d\n%6s%12s%12s%12s\n",
             data->step, "atom", "nlp", "elp", "total" );
    fprintf( out_control->eov, "step: %d\n%6s%12s%12s\n",
             data->step, "atom", "eov", "total" );
    fprintf( out_control->eun, "step: %d\n%6s%12s%12s\n",
             data->step, "atom", "eun", "total" );
    fprintf( out_control->eval, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3",
             "angle", "bo(12)", "bo(23)", "eval", "epen", "total" );
    fprintf( out_control->epen, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3",
             "angle", "bo(12)", "bo(23)", "epen", "total" );
    fprintf( out_control->ecoa, "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3",
             "angle", "bo(12)", "bo(23)", "ecoa", "total" );
    fprintf( out_control->ehb,  "step: %d\n%6s%6s%6s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3",
             "r(23)", "angle", "bo(12)", "ehb", "total" );
    fprintf( out_control->etor, "step: %d\n%6s%6s%6s%6s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3", "atom4",
             "phi", "bo(23)", "etor", "total" );
    fprintf( out_control->econ, "step:%d\n%6s%6s%6s%6s%12s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "atom3", "atom4",
             "phi", "bo(12)", "bo(23)", "bo(34)", "econ", "total" );
#endif

    /* Implement all the function calls as function pointers */
    for ( i = 0; i < NO_OF_INTERACTIONS; i++ )
    {
        (Interaction_Functions[i])(system, control, data, workspace,
                                   lists, out_control);
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "f%d-", i );
#endif
#ifdef TEST_FORCES
        (Print_Interactions[i])(system, control, data, workspace,
                                lists, out_control);
#endif
    }
}


void Compute_NonBonded_Forces( reax_system *system, control_params *control,
                               simulation_data *data, static_storage *workspace,
                               list** lists, output_controls *out_control )
{
    real t_start, t_elapsed;
#ifdef TEST_ENERGY
    fprintf( out_control->evdw, "step: %d\n%6s%6s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "r12", "evdw", "total" );
    fprintf( out_control->ecou, "step: %d\n%6s%6s%12s%12s%12s%12s%12s\n",
             data->step, "atom1", "atom2", "r12", "q1", "q2", "ecou", "total" );
#endif

    t_start = Get_Time( );
    QEq( system, control, data, workspace, lists[FAR_NBRS], out_control );
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.QEq += t_elapsed;
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "qeq - " );
#endif

    if ( control->tabulate == 0)
    {
        vdW_Coulomb_Energy( system, control, data, workspace, lists, out_control );
    }
    else
    {
        Tabulated_vdW_Coulomb_Energy( system, control, data, workspace,
                                      lists, out_control );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "nonb forces - " );
#endif

#ifdef TEST_FORCES
    Print_vdW_Coulomb_Forces( system, control, data, workspace,
                              lists, out_control );
#endif
}


/* This version of Compute_Total_Force computes forces from coefficients
   accumulated by all interaction functions. Saves enormous time & space! */
void Compute_Total_Force( reax_system *system, control_params *control,
                          simulation_data *data, static_storage *workspace,
                          list **lists )
{
    int i, pj;
    list *bonds = (*lists) + BONDS;

    for ( i = 0; i < system->N; ++i )
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
            if ( i < bonds->select.bond_list[pj].nbr )
            {
                if ( control->ensemble == NVE || control->ensemble == NVT || control->ensemble == bNVT)
                    Add_dBond_to_Forces( i, pj, system, data, workspace, lists );
                else
                    Add_dBond_to_Forces_NPT( i, pj, system, data, workspace, lists );
            }
}


void Validate_Lists( static_storage *workspace, list **lists, int step, int n,
                     int Hmax, int Htop, int num_bonds, int num_hbonds )
{
    int i, flag;
    list *bonds, *hbonds;

    bonds = *lists + BONDS;
    hbonds = *lists + HBONDS;

    /* far neighbors */
    if ( Htop > Hmax * DANGER_ZONE )
    {
        workspace->realloc.Htop = Htop;
        if ( Htop > Hmax )
        {
            fprintf( stderr,
                     "step%d - ran out of space on H matrix: Htop=%d, max = %d",
                     step, Htop, Hmax );
            exit( INSUFFICIENT_MEMORY );
        }
    }

    /* bond list */
    flag = -1;
    workspace->realloc.num_bonds = num_bonds;
    for ( i = 0; i < n - 1; ++i )
        if ( End_Index(i, bonds) >= Start_Index(i + 1, bonds) - 2 )
        {
            workspace->realloc.bonds = 1;
            if ( End_Index(i, bonds) > Start_Index(i + 1, bonds) )
                flag = i;
        }

    if ( flag > -1 )
    {
        fprintf( stderr, "step%d-bondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                 step, flag, End_Index(flag, bonds), Start_Index(flag + 1, bonds) );
        exit( INSUFFICIENT_MEMORY );
    }

    if ( End_Index(i, bonds) >= bonds->num_intrs - 2 )
    {
        workspace->realloc.bonds = 1;

        if ( End_Index(i, bonds) > bonds->num_intrs )
        {
            fprintf( stderr, "step%d-bondchk failed: i=%d end(i)=%d bond_end=%d\n",
                     step, flag, End_Index(i, bonds), bonds->num_intrs );
            exit( INSUFFICIENT_MEMORY );
        }
    }


    /* hbonds list */
    if ( workspace->num_H > 0 )
    {
        flag = -1;
        workspace->realloc.num_hbonds = num_hbonds;
        for ( i = 0; i < workspace->num_H - 1; ++i )
            if ( Num_Entries(i, hbonds) >=
                    (Start_Index(i + 1, hbonds) - Start_Index(i, hbonds)) * DANGER_ZONE )
            {
                workspace->realloc.hbonds = 1;
                if ( End_Index(i, hbonds) > Start_Index(i + 1, hbonds) )
                    flag = i;
            }

        if ( flag > -1 )
        {
            fprintf( stderr, "step%d-hbondchk failed: i=%d end(i)=%d str(i+1)=%d\n",
                     step, flag, End_Index(flag, hbonds), Start_Index(flag + 1, hbonds) );
            exit( INSUFFICIENT_MEMORY );
        }

        if ( Num_Entries(i, hbonds) >=
                (hbonds->num_intrs - Start_Index(i, hbonds)) * DANGER_ZONE )
        {
            workspace->realloc.hbonds = 1;

            if ( End_Index(i, hbonds) > hbonds->num_intrs )
            {
                fprintf( stderr, "step%d-hbondchk failed: i=%d end(i)=%d hbondend=%d\n",
                         step, flag, End_Index(i, hbonds), hbonds->num_intrs );
                exit( INSUFFICIENT_MEMORY );
            }
        }
    }
}


void Init_Forces( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, list **lists,
        output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop, H_sp_top, btop_i, btop_j, num_bonds, num_hbonds;
    int ihb, jhb, ihb_top, jhb_top;
    int flag, flag_sp;
    real r_ij, r2, self_coef;
    real dr3gamij_1, dr3gamij_3, Tap;
    //real val, dif, base;
    real C12, C34, C56;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
    real BO, BO_s, BO_pi, BO_pi2;
    real p_boc1, p_boc2;
    sparse_matrix *H, *H_sp;
    list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    //LR_lookup_table *t;
    reax_atom *atom_i, *atom_j;
    bond_data *ibond, *jbond;
    bond_order_data *bo_ij, *bo_ji;

    far_nbrs = *lists + FAR_NBRS;
    bonds = *lists + BONDS;
    hbonds = *lists + HBONDS;

    H = workspace->H;
    H_sp = workspace->H_sp;
    Htop = 0;
    H_sp_top = 0;
    num_bonds = 0;
    num_hbonds = 0;
    btop_i = btop_j = 0;
    p_boc1 = system->reaxprm.gp.l[0];
    p_boc2 = system->reaxprm.gp.l[1];

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &(system->atoms[i]);
        type_i  = atom_i->type;
        start_i = Start_Index( i, far_nbrs );
        end_i   = End_Index( i, far_nbrs );
        H->start[i] = Htop;
        H_sp->start[i] = H_sp_top;
        btop_i = End_Index( i, bonds );
        sbp_i = &(system->reaxprm.sbp[type_i]);
        ihb = ihb_top = -1;
        if ( control->hb_cut > 0 && (ihb = sbp_i->p_hbond) == 1 )
        {
            ihb_top = End_Index( workspace->hbond_index[i], hbonds );
        }

        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j = nbr_pj->nbr;
            atom_j = &(system->atoms[j]);

            flag = 0;
            flag_sp = 0;
            if ((data->step - data->prev_steps) % control->reneighbor == 0)
            {
                if ( nbr_pj->d <= control->r_cut )
                {
                    flag = 1;
                    if ( nbr_pj->d <= control->r_sp_cut )
                    {
                        flag_sp = 1;
                    }
                }
                else
                {
                    flag = 0;
                    flag_sp = 0;
                }
            }
            else if ((nbr_pj->d = Sq_Distance_on_T3(atom_i->x, atom_j->x, &(system->box),
                                                    nbr_pj->dvec)) <= SQR(control->r_cut))
            {
                if ( nbr_pj->d <= SQR(control->r_sp_cut))
                {
                    flag_sp = 1;
                }
                nbr_pj->d = SQRT( nbr_pj->d );
                flag = 1;
            }

            if ( flag )
            {
                type_j = system->atoms[j].type;
                r_ij = nbr_pj->d;
                sbp_j = &(system->reaxprm.sbp[type_j]);
                twbp = &(system->reaxprm.tbp[ index_tbp(type_i,type_j,system->reaxprm.num_atom_types) ]);
                self_coef = (i == j) ? 0.5 : 1.0;

                /* H matrix entry */
                Tap = control->Tap7 * r_ij + control->Tap6;
                Tap = Tap * r_ij + control->Tap5;
                Tap = Tap * r_ij + control->Tap4;
                Tap = Tap * r_ij + control->Tap3;
                Tap = Tap * r_ij + control->Tap2;
                Tap = Tap * r_ij + control->Tap1;
                Tap = Tap * r_ij + control->Tap0;

                dr3gamij_1 = ( r_ij * r_ij * r_ij + twbp->gamma );
                dr3gamij_3 = POW( dr3gamij_1 , 0.33333333333333 );

                H->j[Htop] = j;
                H->val[Htop] = self_coef * Tap * EV_to_KCALpMOL / dr3gamij_3;
                ++Htop;

                /* H_sp matrix entry */
                if ( flag_sp )
                {
                    H_sp->j[H_sp_top] = j;
                    H_sp->val[H_sp_top] = H->val[Htop - 1];
                    ++H_sp_top;
                }

                /* hydrogen bond lists */
                if ( control->hb_cut > 0 && (ihb == 1 || ihb == 2) &&
                        nbr_pj->d <= control->hb_cut )
                {
                    // fprintf( stderr, "%d %d\n", atom1, atom2 );
                    jhb = sbp_j->p_hbond;
                    if ( ihb == 1 && jhb == 2 )
                    {
                        hbonds->select.hbond_list[ihb_top].nbr = j;
                        hbonds->select.hbond_list[ihb_top].scl = 1;
                        hbonds->select.hbond_list[ihb_top].ptr = nbr_pj;
                        ++ihb_top;
                        ++num_hbonds;
                    }
                    else if ( ihb == 2 && jhb == 1 )
                    {
                        jhb_top = End_Index( workspace->hbond_index[j], hbonds );
                        hbonds->select.hbond_list[jhb_top].nbr = i;
                        hbonds->select.hbond_list[jhb_top].scl = -1;
                        hbonds->select.hbond_list[jhb_top].ptr = nbr_pj;
                        Set_End_Index( workspace->hbond_index[j], jhb_top + 1, hbonds );
                        ++num_hbonds;
                    }
                }

                /* uncorrected bond orders */
                if ( far_nbrs->select.far_nbr_list[pj].d <= control->nbr_cut )
                {
                    r2 = SQR(r_ij);

                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0)
                    {
                        C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                    }
                    else BO_s = C12 = 0.0;

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0)
                    {
                        C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = EXP( C34 );
                    }
                    else BO_pi = C34 = 0.0;

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0)
                    {
                        C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = EXP( C56 );
                    }
                    else BO_pi2 = C56 = 0.0;

                    /* Initially BO values are the uncorrected ones, page 1 */
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        num_bonds += 2;
                        /****** bonds i-j and j-i ******/
                        ibond = &( bonds->select.bond_list[btop_i] );
                        btop_j = End_Index( j, bonds );
                        jbond = &(bonds->select.bond_list[btop_j]);

                        ibond->nbr = j;
                        jbond->nbr = i;
                        ibond->d = r_ij;
                        jbond->d = r_ij;
                        rvec_Copy( ibond->dvec, nbr_pj->dvec );
                        rvec_Scale( jbond->dvec, -1, nbr_pj->dvec );
                        ivec_Copy( ibond->rel_box, nbr_pj->rel_box );
                        ivec_Scale( jbond->rel_box, -1, nbr_pj->rel_box );
                        ibond->dbond_index = btop_i;
                        jbond->dbond_index = btop_i;
                        ibond->sym_index = btop_j;
                        jbond->sym_index = btop_i;
                        ++btop_i;
                        Set_End_Index( j, btop_j + 1, bonds );

                        bo_ij = &( ibond->bo_data );
                        bo_ji = &( jbond->bo_data );
                        bo_ji->BO = bo_ij->BO = BO;
                        bo_ji->BO_s = bo_ij->BO_s = BO_s;
                        bo_ji->BO_pi = bo_ij->BO_pi = BO_pi;
                        bo_ji->BO_pi2 = bo_ij->BO_pi2 = BO_pi2;

                        /* Bond Order page2-3, derivative of total bond order prime */
                        Cln_BOp_s = twbp->p_bo2 * C12 / r2;
                        Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
                        Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

                        /* Only dln_BOp_xx wrt. dr_i is stored here, note that
                           dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
                        rvec_Scale(bo_ij->dln_BOp_s, -bo_ij->BO_s * Cln_BOp_s, ibond->dvec);
                        rvec_Scale(bo_ij->dln_BOp_pi, -bo_ij->BO_pi * Cln_BOp_pi, ibond->dvec);
                        rvec_Scale(bo_ij->dln_BOp_pi2,
                                   -bo_ij->BO_pi2 * Cln_BOp_pi2, ibond->dvec);
                        rvec_Scale(bo_ji->dln_BOp_s, -1., bo_ij->dln_BOp_s);
                        rvec_Scale(bo_ji->dln_BOp_pi, -1., bo_ij->dln_BOp_pi );
                        rvec_Scale(bo_ji->dln_BOp_pi2, -1., bo_ij->dln_BOp_pi2 );

                        /* Only dBOp wrt. dr_i is stored here, note that
                           dBOp/dr_i = -dBOp/dr_j and all others are 0 */
                        rvec_Scale( bo_ij->dBOp,
                                    -(bo_ij->BO_s * Cln_BOp_s +
                                      bo_ij->BO_pi * Cln_BOp_pi +
                                      bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );
                        rvec_Scale( bo_ji->dBOp, -1., bo_ij->dBOp );

                        rvec_Add( workspace->dDeltap_self[i], bo_ij->dBOp );
                        rvec_Add( workspace->dDeltap_self[j], bo_ji->dBOp );

                        bo_ij->BO_s -= control->bo_cut;
                        bo_ij->BO -= control->bo_cut;
                        bo_ji->BO_s -= control->bo_cut;
                        bo_ji->BO -= control->bo_cut;
                        workspace->total_bond_order[i] += bo_ij->BO; //currently total_BOp
                        workspace->total_bond_order[j] += bo_ji->BO; //currently total_BOp
                        bo_ij->Cdbo = bo_ij->Cdbopi = bo_ij->Cdbopi2 = 0.0;
                        bo_ji->Cdbo = bo_ji->Cdbopi = bo_ji->Cdbopi2 = 0.0;

                        /*fprintf( stderr, "%d %d %g %g %g\n",
                          i+1, j+1, bo_ij->BO, bo_ij->BO_pi, bo_ij->BO_pi2 );*/

                        /*fprintf( stderr, "Cln_BOp_s: %f, pbo2: %f, C12:%f\n",
                          Cln_BOp_s, twbp->p_bo2, C12 );
                          fprintf( stderr, "Cln_BOp_pi: %f, pbo4: %f, C34:%f\n",
                          Cln_BOp_pi, twbp->p_bo4, C34 );
                          fprintf( stderr, "Cln_BOp_pi2: %f, pbo6: %f, C56:%f\n",
                          Cln_BOp_pi2, twbp->p_bo6, C56 );*/
                        /*fprintf(stderr, "pbo1: %f, pbo2:%f\n", twbp->p_bo1, twbp->p_bo2);
                          fprintf(stderr, "pbo3: %f, pbo4:%f\n", twbp->p_bo3, twbp->p_bo4);
                          fprintf(stderr, "pbo5: %f, pbo6:%f\n", twbp->p_bo5, twbp->p_bo6);
                          fprintf( stderr, "r_s: %f, r_p: %f, r_pp: %f\n",
                          twbp->r_s, twbp->r_p, twbp->r_pp );
                          fprintf( stderr, "C12: %g, C34:%g, C56:%g\n", C12, C34, C56 );*/

                        /*fprintf( stderr, "\tfactors: %g %g %g\n",
                          -(bo_ij->BO_s * Cln_BOp_s + bo_ij->BO_pi * Cln_BOp_pi +
                          bo_ij->BO_pi2 * Cln_BOp_pp),
                          -bo_ij->BO_pi * Cln_BOp_pi, -bo_ij->BO_pi2 * Cln_BOp_pi2 );*/
                        /*fprintf( stderr, "dBOpi:\t[%g, %g, %g]\n",
                          bo_ij->dBOp[0], bo_ij->dBOp[1], bo_ij->dBOp[2] );
                          fprintf( stderr, "dBOpi:\t[%g, %g, %g]\n",
                          bo_ij->dln_BOp_pi[0], bo_ij->dln_BOp_pi[1],
                          bo_ij->dln_BOp_pi[2] );
                          fprintf( stderr, "dBOpi2:\t[%g, %g, %g]\n\n",
                          bo_ij->dln_BOp_pi2[0], bo_ij->dln_BOp_pi2[1],
                          bo_ij->dln_BOp_pi2[2] );*/

                        Set_End_Index( j, btop_j + 1, bonds );
                    }
                }
            }
        }

        /* diagonal entry */
        H->j[Htop] = i;
        H->val[Htop] = system->reaxprm.sbp[type_i].eta;
        ++Htop;

        /* diagonal entry */
        H_sp->j[H_sp_top] = i;
        H_sp->val[H_sp_top] = H->val[Htop - 1];
        ++H_sp_top;

        Set_End_Index( i, btop_i, bonds );
        if ( ihb == 1 )
        {
            Set_End_Index( workspace->hbond_index[i], ihb_top, hbonds );
        }

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "%d bonds start: %d, end: %d\n",
             i, Start_Index( i, bonds ), End_Index( i, bonds ) );
#endif
    }

#if defined(DEBUG_FOCUS)
    printf( "Htop = %d\n", Htop );
    printf( "H_sp_top = %d\n", H_sp_top );
#endif

    // mark the end of j list
    H->start[i] = Htop;
    H_sp->start[i] = H_sp_top;
    /* validate lists - decide if reallocation is required! */
    Validate_Lists( workspace, lists, data->step, system->N, H->m,
            Htop, num_bonds, num_hbonds );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "step%d: Htop = %d, num_bonds = %d, num_hbonds = %d\n",
             data->step, Htop, num_bonds, num_hbonds );

#endif
}


void Init_Forces_Tab( reax_system *system, control_params *control,
                      simulation_data *data, static_storage *workspace,
                      list **lists, output_controls *out_control )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int Htop, H_sp_top, btop_i, btop_j, num_bonds, num_hbonds;
    int tmin, tmax, r;
    int ihb, jhb, ihb_top, jhb_top;
    int flag, flag_sp;
    real r_ij, r2, self_coef;
    real val, dif, base;
    real C12, C34, C56;
    real Cln_BOp_s, Cln_BOp_pi, Cln_BOp_pi2;
    real BO, BO_s, BO_pi, BO_pi2;
    real p_boc1, p_boc2;
    sparse_matrix *H, *H_sp;
    list *far_nbrs, *bonds, *hbonds;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    LR_lookup_table *t;
    reax_atom *atom_i, *atom_j;
    bond_data *ibond, *jbond;
    bond_order_data *bo_ij, *bo_ji;

    far_nbrs = *lists + FAR_NBRS;
    bonds = *lists + BONDS;
    hbonds = *lists + HBONDS;

    H = workspace->H;
    H_sp = workspace->H_sp;
    Htop = 0;
    H_sp_top = 0;
    num_bonds = 0;
    num_hbonds = 0;
    btop_i = btop_j = 0;
    p_boc1 = system->reaxprm.gp.l[0];
    p_boc2 = system->reaxprm.gp.l[1];

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &(system->atoms[i]);
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);
        H->start[i] = Htop;
        H_sp->start[i] = H_sp_top;
        btop_i = End_Index( i, bonds );
        sbp_i = &(system->reaxprm.sbp[type_i]);
        ihb = ihb_top = -1;
        if ( control->hb_cut > 0 && (ihb = sbp_i->p_hbond) == 1 )
            ihb_top = End_Index( workspace->hbond_index[i], hbonds );

        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j = nbr_pj->nbr;
            atom_j = &(system->atoms[j]);

            flag = 0;
            flag_sp = 0;
            if ((data->step - data->prev_steps) % control->reneighbor == 0)
            {
                if (nbr_pj->d <= control->r_cut)
                {
                    flag = 1;
                    if ( nbr_pj->d <= control->r_sp_cut )
                    {
                        flag_sp = 1;
                    }
                }
                else
                {
                    flag = 0;
                    flag_sp = 0;
                }
            }
            else if ((nbr_pj->d = Sq_Distance_on_T3(atom_i->x, atom_j->x, &(system->box),
                                                    nbr_pj->dvec)) <= SQR(control->r_cut))
            {
                if ( nbr_pj->d <= SQR(control->r_sp_cut))
                {
                    flag_sp = 1;
                }
                nbr_pj->d = sqrt(nbr_pj->d);
                flag = 1;
            }

            if ( flag )
            {
                type_j = system->atoms[j].type;
                r_ij = nbr_pj->d;
                sbp_j = &(system->reaxprm.sbp[type_j]);
                twbp = &(system->reaxprm.tbp[ index_tbp(type_i,type_j,system->reaxprm.num_atom_types) ]);
                self_coef = (i == j) ? 0.5 : 1.0;
                tmin  = MIN( type_i, type_j );
                tmax  = MAX( type_i, type_j );
                t = &( LR[ index_lr (tmin,tmax,system->reaxprm.num_atom_types) ] );      

                /* cubic spline interpolation */
                r = (int)(r_ij * t->inv_dx);
                if ( r == 0 )  ++r;
                base = (real)(r + 1) * t->dx;
                dif = r_ij - base;
                val = ((t->ele[r].d * dif + t->ele[r].c) * dif + t->ele[r].b) * dif +
                      t->ele[r].a;
                val *= EV_to_KCALpMOL / C_ele;

                H->j[Htop] = j;
                H->val[Htop] = self_coef * val;
                ++Htop;

                /* H_sp matrix entry */
                if ( flag_sp )
                {
                    H_sp->j[H_sp_top] = j;
                    H_sp->val[H_sp_top] = H->val[Htop - 1];
                    ++H_sp_top;
                }

                /* hydrogen bond lists */
                if ( control->hb_cut > 0 && (ihb == 1 || ihb == 2) &&
                        nbr_pj->d <= control->hb_cut )
                {
                    // fprintf( stderr, "%d %d\n", atom1, atom2 );
                    jhb = sbp_j->p_hbond;
                    if ( ihb == 1 && jhb == 2 )
                    {
                        hbonds->select.hbond_list[ihb_top].nbr = j;
                        hbonds->select.hbond_list[ihb_top].scl = 1;
                        hbonds->select.hbond_list[ihb_top].ptr = nbr_pj;
                        ++ihb_top;
                        ++num_hbonds;
                    }
                    else if ( ihb == 2 && jhb == 1 )
                    {
                        jhb_top = End_Index( workspace->hbond_index[j], hbonds );
                        hbonds->select.hbond_list[jhb_top].nbr = i;
                        hbonds->select.hbond_list[jhb_top].scl = -1;
                        hbonds->select.hbond_list[jhb_top].ptr = nbr_pj;
                        Set_End_Index( workspace->hbond_index[j], jhb_top + 1, hbonds );
                        ++num_hbonds;
                    }
                }

                /* uncorrected bond orders */
                if ( far_nbrs->select.far_nbr_list[pj].d <= control->nbr_cut )
                {
                    r2 = SQR(r_ij);

                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0)
                    {
                        C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                    }
                    else BO_s = C12 = 0.0;

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0)
                    {
                        C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = EXP( C34 );
                    }
                    else BO_pi = C34 = 0.0;

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0)
                    {
                        C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = EXP( C56 );
                    }
                    else BO_pi2 = C56 = 0.0;

                    /* Initially BO values are the uncorrected ones, page 1 */
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        num_bonds += 2;
                        /****** bonds i-j and j-i ******/
                        ibond = &( bonds->select.bond_list[btop_i] );
                        btop_j = End_Index( j, bonds );
                        jbond = &(bonds->select.bond_list[btop_j]);

                        ibond->nbr = j;
                        jbond->nbr = i;
                        ibond->d = r_ij;
                        jbond->d = r_ij;
                        rvec_Copy( ibond->dvec, nbr_pj->dvec );
                        //fprintf (stderr, " %f - %f - %f \n", nbr_pj->dvec[0], nbr_pj->dvec[1], nbr_pj->dvec[2]);
                        rvec_Scale( jbond->dvec, -1, nbr_pj->dvec );
                        ivec_Copy( ibond->rel_box, nbr_pj->rel_box );
                        ivec_Scale( jbond->rel_box, -1, nbr_pj->rel_box );
                        ibond->dbond_index = btop_i;
                        jbond->dbond_index = btop_i;
                        ibond->sym_index = btop_j;
                        jbond->sym_index = btop_i;
                        ++btop_i;
                        Set_End_Index( j, btop_j + 1, bonds );

                        bo_ij = &( ibond->bo_data );
                        bo_ji = &( jbond->bo_data );
                        bo_ji->BO = bo_ij->BO = BO;
                        bo_ji->BO_s = bo_ij->BO_s = BO_s;
                        bo_ji->BO_pi = bo_ij->BO_pi = BO_pi;
                        bo_ji->BO_pi2 = bo_ij->BO_pi2 = BO_pi2;

                        /* Bond Order page2-3, derivative of total bond order prime */
                        Cln_BOp_s = twbp->p_bo2 * C12 / r2;
                        Cln_BOp_pi = twbp->p_bo4 * C34 / r2;
                        Cln_BOp_pi2 = twbp->p_bo6 * C56 / r2;

                        /* Only dln_BOp_xx wrt. dr_i is stored here, note that
                           dln_BOp_xx/dr_i = -dln_BOp_xx/dr_j and all others are 0 */
                        rvec_Scale(bo_ij->dln_BOp_s, -bo_ij->BO_s * Cln_BOp_s, ibond->dvec);
                        rvec_Scale(bo_ij->dln_BOp_pi, -bo_ij->BO_pi * Cln_BOp_pi, ibond->dvec);
                        rvec_Scale(bo_ij->dln_BOp_pi2,
                                   -bo_ij->BO_pi2 * Cln_BOp_pi2, ibond->dvec);
                        rvec_Scale(bo_ji->dln_BOp_s, -1., bo_ij->dln_BOp_s);
                        rvec_Scale(bo_ji->dln_BOp_pi, -1., bo_ij->dln_BOp_pi );
                        rvec_Scale(bo_ji->dln_BOp_pi2, -1., bo_ij->dln_BOp_pi2 );

                        /* Only dBOp wrt. dr_i is stored here, note that
                           dBOp/dr_i = -dBOp/dr_j and all others are 0 */
                        rvec_Scale( bo_ij->dBOp,
                                    -(bo_ij->BO_s * Cln_BOp_s +
                                      bo_ij->BO_pi * Cln_BOp_pi +
                                      bo_ij->BO_pi2 * Cln_BOp_pi2), ibond->dvec );
                        rvec_Scale( bo_ji->dBOp, -1., bo_ij->dBOp );

                        rvec_Add( workspace->dDeltap_self[i], bo_ij->dBOp );
                        rvec_Add( workspace->dDeltap_self[j], bo_ji->dBOp );

                        bo_ij->BO_s -= control->bo_cut;
                        bo_ij->BO -= control->bo_cut;
                        bo_ji->BO_s -= control->bo_cut;
                        bo_ji->BO -= control->bo_cut;
                        workspace->total_bond_order[i] += bo_ij->BO; //currently total_BOp
                        workspace->total_bond_order[j] += bo_ji->BO; //currently total_BOp
                        bo_ij->Cdbo = bo_ij->Cdbopi = bo_ij->Cdbopi2 = 0.0;
                        bo_ji->Cdbo = bo_ji->Cdbopi = bo_ji->Cdbopi2 = 0.0;

                        Set_End_Index( j, btop_j + 1, bonds );
                    }
                }
            }
        }

        /* diagonal entry */
        H->j[Htop] = i;
        H->val[Htop] = system->reaxprm.sbp[type_i].eta;
        ++Htop;

        /* diagonal entry */
        H_sp->j[H_sp_top] = i;
        H_sp->val[H_sp_top] = H->val[Htop - 1];
        ++H_sp_top;

        Set_End_Index( i, btop_i, bonds );
        if ( ihb == 1 )
            Set_End_Index( workspace->hbond_index[i], ihb_top, hbonds );
    }

    // mark the end of j list
    H->start[i] = Htop;
    H_sp->start[i] = H_sp_top;
    /* validate lists - decide if reallocation is required! */
    Validate_Lists( workspace, lists,
                    data->step, system->N, H->m, Htop, num_bonds, num_hbonds );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "step%d: Htop = %d, num_bonds = %d, num_hbonds = %d\n",
             data->step, Htop, num_bonds, num_hbonds );
    //Print_Bonds( system, bonds, "sbonds.out" );
    //Print_Bond_List2( system, bonds, "sbonds.out" );
    //Print_Sparse_Matrix2( H, "H.out" );
#endif
}


void Estimate_Storage_Sizes( reax_system *system, control_params *control,
                             list **lists, int *Htop, int *hb_top,
                             int *bond_top, int *num_3body )
{
    int i, j, pj;
    int start_i, end_i;
    int type_i, type_j;
    int ihb, jhb;
    real r_ij, r2;
    real C12, C34, C56;
    real BO, BO_s, BO_pi, BO_pi2;
    real p_boc1, p_boc2;
    list *far_nbrs;
    single_body_parameters *sbp_i, *sbp_j;
    two_body_parameters *twbp;
    far_neighbor_data *nbr_pj;
    reax_atom *atom_i, *atom_j;

    far_nbrs = *lists + FAR_NBRS;
    p_boc1 = system->reaxprm.gp.l[0];
    p_boc2 = system->reaxprm.gp.l[1];

    for ( i = 0; i < system->N; ++i )
    {
        atom_i = &(system->atoms[i]);
        type_i  = atom_i->type;
        start_i = Start_Index(i, far_nbrs);
        end_i   = End_Index(i, far_nbrs);
        sbp_i = &(system->reaxprm.sbp[type_i]);
        ihb = sbp_i->p_hbond;

        for ( pj = start_i; pj < end_i; ++pj )
        {
            nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
            j = nbr_pj->nbr;
            atom_j = &(system->atoms[j]);
            type_j = atom_j->type;
            sbp_j = &(system->reaxprm.sbp[type_j]);
            twbp = &(system->reaxprm.tbp[ index_tbp(type_i,type_j,system->reaxprm.num_atom_types) ]);

            if ( nbr_pj->d <= control->r_cut )
            {
                ++(*Htop);

                /* hydrogen bond lists */
                if ( control->hb_cut > 0.1 && (ihb == 1 || ihb == 2) &&
                        nbr_pj->d <= control->hb_cut )
                {
                    jhb = sbp_j->p_hbond;
                    if ( ihb == 1 && jhb == 2 )
                        ++hb_top[i];
                    else if ( ihb == 2 && jhb == 1 )
                        ++hb_top[j];
                }

                /* uncorrected bond orders */
                if ( nbr_pj->d <= control->nbr_cut )
                {
                    r_ij = nbr_pj->d;
                    r2 = SQR(r_ij);

                    if ( sbp_i->r_s > 0.0 && sbp_j->r_s > 0.0)
                    {
                        C12 = twbp->p_bo1 * POW( r_ij / twbp->r_s, twbp->p_bo2 );
                        BO_s = (1.0 + control->bo_cut) * EXP( C12 );
                    }
                    else BO_s = C12 = 0.0;

                    if ( sbp_i->r_pi > 0.0 && sbp_j->r_pi > 0.0)
                    {
                        C34 = twbp->p_bo3 * POW( r_ij / twbp->r_p, twbp->p_bo4 );
                        BO_pi = EXP( C34 );
                    }
                    else BO_pi = C34 = 0.0;

                    if ( sbp_i->r_pi_pi > 0.0 && sbp_j->r_pi_pi > 0.0)
                    {
                        C56 = twbp->p_bo5 * POW( r_ij / twbp->r_pp, twbp->p_bo6 );
                        BO_pi2 = EXP( C56 );
                    }
                    else BO_pi2 = C56 = 0.0;

                    /* Initially BO values are the uncorrected ones, page 1 */
                    BO = BO_s + BO_pi + BO_pi2;

                    if ( BO >= control->bo_cut )
                    {
                        ++bond_top[i];
                        ++bond_top[j];
                    }
                }
            }
        }
    }

    *Htop += system->N;
    *Htop *= SAFE_ZONE;
    for ( i = 0; i < system->N; ++i )
    {
        hb_top[i] = MAX( hb_top[i] * SAFE_HBONDS, MIN_HBONDS );
        *num_3body += SQR(bond_top[i]);
        bond_top[i] = MAX( bond_top[i] * 2, MIN_BONDS );
    }
    *num_3body *= SAFE_ZONE;
}


void Compute_Forces( reax_system *system, control_params *control,
                     simulation_data *data, static_storage *workspace,
                     list** lists, output_controls *out_control )
{
    real t_start, t_elapsed;

    t_start = Get_Time( );
    if ( !control->tabulate )
    {
        Init_Forces( system, control, data, workspace, lists, out_control );
    }
    else
    {
        Init_Forces_Tab( system, control, data, workspace, lists, out_control );
    }
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.init_forces += t_elapsed;
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "init_forces - ");
#endif

    t_start = Get_Time( );
    Compute_Bonded_Forces( system, control, data, workspace, lists, out_control );
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.bonded += t_elapsed;
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "bonded_forces - ");
#endif

    t_start = Get_Time( );
    Compute_NonBonded_Forces( system, control, data, workspace,
                              lists, out_control );
    t_elapsed = Get_Timing_Info( t_start );
    data->timing.nonb += t_elapsed;
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "nonbondeds - ");
#endif

    Compute_Total_Force( system, control, data, workspace, lists );
    //Print_Total_Force( system, control, data, workspace, lists, out_control );
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "totalforces - ");
    //Print_Total_Force( system, control, data, workspace, lists, out_control );
#endif

#ifdef TEST_FORCES
    Print_Total_Force( system, control, data, workspace, lists, out_control );
    Compare_Total_Forces( system, control, data, workspace, lists, out_control );
#endif
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "forces - ");
#endif
}
