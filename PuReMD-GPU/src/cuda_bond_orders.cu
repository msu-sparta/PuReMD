/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
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

#include "cuda_bond_orders.h"

#include "bond_orders.h"
#include "list.h"
#include "lookup.h"
#include "print_utils.h"
#include "vector.h"
#include "index_utils.h"

#include "cuda_utils.h"
#include "cuda_helpers.h"


HOST_DEVICE void Cuda_Add_dBond_to_Forces_NPT( int i, int pj, reax_atom *atoms, 
        simulation_data *data, static_storage *workspace, 
        list *bonds )
{
    bond_data *nbr_j, *nbr_k;
    bond_order_data *bo_ij, *bo_ji; 
    dbond_coefficients coef;
    rvec temp, ext_press;
    ivec rel_box;
    int pk, k, j;

    /* Initializations */
    nbr_j = &(bonds->select.bond_list[pj]);
    j = nbr_j->nbr;
    bo_ij = &(nbr_j->bo_data);
    bo_ji = &(bonds->select.bond_list[ nbr_j->sym_index ].bo_data);

    coef.C1dbo = bo_ij->C1dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C2dbo = bo_ij->C2dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C3dbo = bo_ij->C3dbo * (bo_ij->Cdbo + bo_ji->Cdbo);

    coef.C1dbopi = bo_ij->C1dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C2dbopi = bo_ij->C2dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C3dbopi = bo_ij->C3dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C4dbopi = bo_ij->C4dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);

    coef.C1dbopi2 = bo_ij->C1dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C2dbopi2 = bo_ij->C2dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C3dbopi2 = bo_ij->C3dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C4dbopi2 = bo_ij->C4dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);

    coef.C1dDelta = bo_ij->C1dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);
    coef.C2dDelta = bo_ij->C2dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);
    coef.C3dDelta = bo_ij->C3dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);


    /************************************
     * forces related to atom i          *
     * first neighbors of atom i         *
     ************************************/
    for( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk ) {
        nbr_k = &(bonds->select.bond_list[pk]);
        k = nbr_k->nbr;

        rvec_Scale( temp, -coef.C2dbo, nbr_k->bo_data.dBOp );       /*2nd,dBO*/
        rvec_ScaledAdd( temp, -coef.C2dDelta, nbr_k->bo_data.dBOp );/*dDelta*/
        rvec_ScaledAdd( temp, -coef.C3dbopi, nbr_k->bo_data.dBOp ); /*3rd,dBOpi*/
        rvec_ScaledAdd( temp, -coef.C3dbopi2, nbr_k->bo_data.dBOp );/*3rd,dBOpi2*/

        /* force */
        rvec_Add( atoms[k].f, temp );
        /* pressure */
        rvec_iMultiply( ext_press, nbr_k->rel_box, temp );
        rvec_Add( data->ext_press, ext_press );
    }

    /* then atom i itself  */
    rvec_Scale( temp, coef.C1dbo, bo_ij->dBOp );                      /*1st, dBO*/
    rvec_ScaledAdd( temp, coef.C2dbo, workspace->dDeltap_self[i] );   /*2nd, dBO*/

    rvec_ScaledAdd( temp, coef.C1dDelta, bo_ij->dBOp );               /*1st, dBO*/
    rvec_ScaledAdd( temp, coef.C2dDelta, workspace->dDeltap_self[i] );/*2nd, dBO*/

    rvec_ScaledAdd( temp, coef.C1dbopi, bo_ij->dln_BOp_pi );         /*1st,dBOpi*/
    rvec_ScaledAdd( temp, coef.C2dbopi, bo_ij->dBOp );               /*2nd,dBOpi*/
    rvec_ScaledAdd( temp, coef.C3dbopi, workspace->dDeltap_self[i] );/*3rd,dBOpi*/

    rvec_ScaledAdd(temp, coef.C1dbopi2, bo_ij->dln_BOp_pi2) ;      /*1st,dBO_pi2*/
    rvec_ScaledAdd(temp, coef.C2dbopi2, bo_ij->dBOp);              /*2nd,dBO_pi2*/
    rvec_ScaledAdd(temp, coef.C3dbopi2, workspace->dDeltap_self[i]);/*3rd,dBO_pi2*/

    /* force */
    rvec_Add( atoms[i].f, temp );
    /* ext pressure due to i dropped, counting force on j only will be enough */


    /****************************************************************************
     * forces and pressure related to atom j                                    *
     * first neighbors of atom j                                                *
     ***************************************************************************/
    for( pk = Start_Index(j, bonds); pk < End_Index(j, bonds); ++pk ) {
        nbr_k = &(bonds->select.bond_list[pk]);
        k = nbr_k->nbr;

        rvec_Scale( temp, -coef.C3dbo, nbr_k->bo_data.dBOp );       /*3rd,dBO*/
        rvec_ScaledAdd( temp, -coef.C3dDelta, nbr_k->bo_data.dBOp );/*dDelta*/ 
        rvec_ScaledAdd( temp, -coef.C4dbopi, nbr_k->bo_data.dBOp ); /*4th,dBOpi*/
        rvec_ScaledAdd( temp, -coef.C4dbopi2, nbr_k->bo_data.dBOp );/*4th,dBOpi2*/

        /* force */
        rvec_Add( atoms[k].f, temp );
        /* pressure */
        if( k != i ) {
            ivec_Sum(rel_box, nbr_k->rel_box, nbr_j->rel_box);//k's rel_box  wrt i
            rvec_iMultiply( ext_press, rel_box, temp );
            rvec_Add( data->ext_press, ext_press );
        }
    }

    /* then atom j itself */
    rvec_Scale( temp, -coef.C1dbo, bo_ij->dBOp );                     /*1st, dBO*/
    rvec_ScaledAdd( temp, coef.C3dbo, workspace->dDeltap_self[j] );   /*2nd, dBO*/

    rvec_ScaledAdd( temp, -coef.C1dDelta, bo_ij->dBOp );              /*1st, dBO*/
    rvec_ScaledAdd( temp, coef.C3dDelta, workspace->dDeltap_self[j] );/*2nd, dBO*/

    rvec_ScaledAdd( temp, -coef.C1dbopi, bo_ij->dln_BOp_pi );        /*1st,dBOpi*/
    rvec_ScaledAdd( temp, -coef.C2dbopi, bo_ij->dBOp );              /*2nd,dBOpi*/
    rvec_ScaledAdd( temp, coef.C4dbopi, workspace->dDeltap_self[j] );/*3rd,dBOpi*/

    rvec_ScaledAdd(temp, -coef.C1dbopi2, bo_ij->dln_BOp_pi2);       /*1st,dBOpi2*/
    rvec_ScaledAdd(temp, -coef.C2dbopi2, bo_ij->dBOp);              /*2nd,dBOpi2*/
    rvec_ScaledAdd(temp, coef.C4dbopi2, workspace->dDeltap_self[j]);/*3rd,dBOpi2*/

    /* force */
    rvec_Add( atoms[j].f, temp );
    /* pressure */
    rvec_iMultiply( ext_press, nbr_j->rel_box, temp );
    rvec_Add( data->ext_press, ext_press );
}


HOST_DEVICE void Cuda_Add_dBond_to_Forces ( int i, int pj, reax_atom *atoms, 
        static_storage *workspace, list *bonds )
{
    bond_data *nbr_j, *nbr_k;
    bond_order_data *bo_ij, *bo_ji; 
    dbond_coefficients coef;
    int pk, k, j;
    rvec t_f;

    /* Initializations */ 
    nbr_j = &(bonds->select.bond_list[pj]);
    j = nbr_j->nbr;

    if (i < j)
    {
        bo_ij = &(nbr_j->bo_data);
        bo_ji = &(bonds->select.bond_list[ nbr_j->sym_index ].bo_data);
    } else {
        bo_ji = &(nbr_j->bo_data);
        bo_ij = &(bonds->select.bond_list[ nbr_j->sym_index ].bo_data);
    }

    coef.C1dbo = bo_ij->C1dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C2dbo = bo_ij->C2dbo * (bo_ij->Cdbo + bo_ji->Cdbo);
    coef.C3dbo = bo_ij->C3dbo * (bo_ij->Cdbo + bo_ji->Cdbo);

    coef.C1dbopi = bo_ij->C1dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C2dbopi = bo_ij->C2dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C3dbopi = bo_ij->C3dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);
    coef.C4dbopi = bo_ij->C4dbopi * (bo_ij->Cdbopi + bo_ji->Cdbopi);

    coef.C1dbopi2 = bo_ij->C1dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C2dbopi2 = bo_ij->C2dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C3dbopi2 = bo_ij->C3dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);
    coef.C4dbopi2 = bo_ij->C4dbopi2 * (bo_ij->Cdbopi2 + bo_ji->Cdbopi2);

    coef.C1dDelta = bo_ij->C1dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);
    coef.C2dDelta = bo_ij->C2dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);
    coef.C3dDelta = bo_ij->C3dbo * (workspace->CdDelta[i]+workspace->CdDelta[j]);

    if ( i < j) {
        for( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk ) {
            nbr_k = &(bonds->select.bond_list[pk]);
            k = nbr_k->nbr;
            rvec_MakeZero (t_f);

            rvec_ScaledAdd( t_f, -coef.C2dbo, nbr_k->bo_data.dBOp ); 
            /*2nd, dBO*/
            rvec_ScaledAdd( t_f, -coef.C2dDelta, nbr_k->bo_data.dBOp );
            /*dDelta*/
            rvec_ScaledAdd( t_f, -coef.C3dbopi, nbr_k->bo_data.dBOp );
            /*3rd, dBOpi*/
            rvec_ScaledAdd( t_f, -coef.C3dbopi2, nbr_k->bo_data.dBOp );
            /*3rd, dBOpi2*/

            //Store in the temp place
            rvec_Add (nbr_k->t_f, t_f);
        }

        rvec_ScaledAdd( atoms[i].f, coef.C1dbo, bo_ij->dBOp );
        /*1st, dBO*/
        rvec_ScaledAdd( atoms[i].f, coef.C2dbo, workspace->dDeltap_self[i] );
        /*2nd, dBO*/

        rvec_ScaledAdd(atoms[i].f, coef.C1dDelta, bo_ij->dBOp);
        /*1st, dBO*/
        rvec_ScaledAdd(atoms[i].f, coef.C2dDelta, workspace->dDeltap_self[i]);
        /*2nd, dBO*/

        rvec_ScaledAdd( atoms[i].f, coef.C1dbopi, bo_ij->dln_BOp_pi );
        /*1st, dBOpi*/
        rvec_ScaledAdd( atoms[i].f, coef.C2dbopi, bo_ij->dBOp );
        /*2nd, dBOpi*/
        rvec_ScaledAdd( atoms[i].f, coef.C3dbopi, workspace->dDeltap_self[i]);
        /*3rd, dBOpi*/

        rvec_ScaledAdd( atoms[i].f, coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
        /*1st, dBO_pi2*/
        rvec_ScaledAdd( atoms[i].f, coef.C2dbopi2, bo_ij->dBOp );
        /*2nd, dBO_pi2*/
        rvec_ScaledAdd( atoms[i].f, coef.C3dbopi2, workspace->dDeltap_self[i]);
        /*3rd, dBO_pi2*/
    }
    else 
    {
        for( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk ) {
            nbr_k = &(bonds->select.bond_list[pk]);
            k = nbr_k->nbr;
            rvec_MakeZero (t_f);

            rvec_ScaledAdd( t_f, -coef.C3dbo, nbr_k->bo_data.dBOp );
            /*3rd, dBO*/
            rvec_ScaledAdd( t_f, -coef.C3dDelta, nbr_k->bo_data.dBOp );
            /*dDelta*/ 
            rvec_ScaledAdd( t_f, -coef.C4dbopi, nbr_k->bo_data.dBOp );
            /*4th, dBOpi*/
            rvec_ScaledAdd( t_f, -coef.C4dbopi2, nbr_k->bo_data.dBOp );
            /*4th, dBOpi2*/

            //Store in the temp place
            rvec_Add (nbr_k->t_f, t_f);
        }

        rvec_ScaledAdd( atoms[i].f, -coef.C1dbo, bo_ij->dBOp );
        /*1st, dBO*/
        rvec_ScaledAdd( atoms[i].f, coef.C3dbo, workspace->dDeltap_self[i] );
        /*2nd, dBO*/

        rvec_ScaledAdd( atoms[i].f, -coef.C1dDelta, bo_ij->dBOp );
        /*1st, dBO*/
        rvec_ScaledAdd(atoms[i].f, coef.C3dDelta, workspace->dDeltap_self[i]);
        /*2nd, dBO*/

        rvec_ScaledAdd( atoms[i].f, -coef.C1dbopi, bo_ij->dln_BOp_pi );
        /*1st, dBOpi*/
        rvec_ScaledAdd( atoms[i].f, -coef.C2dbopi, bo_ij->dBOp );
        /*2nd, dBOpi*/
        rvec_ScaledAdd(atoms[i].f, coef.C4dbopi, workspace->dDeltap_self[i]);
        /*3rd, dBOpi*/

        rvec_ScaledAdd( atoms[i].f, -coef.C1dbopi2, bo_ij->dln_BOp_pi2 );
        /*1st, dBOpi2*/
        rvec_ScaledAdd( atoms[i].f, -coef.C2dbopi2, bo_ij->dBOp );
        /*2nd, dBOpi2*/
        rvec_ScaledAdd(atoms[i].f, coef.C4dbopi2, workspace->dDeltap_self[i]);
        /*3rd, dBOpi2*/
    }
}


HOST_DEVICE void Cuda_dbond_to_Forces_postprocess (int i, reax_atom *atoms, list *bonds)
{
    int pk;
    bond_data *nbr_k, *nbr_k_sym;

    /*
       for( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk ) {
       nbr_k = &(bonds->select.bond_list[pk]);
       rvec_Add (atoms[i].f, nbr_k->t_f);
       }
     */

    for( pk = Start_Index(i, bonds); pk < End_Index(i, bonds); ++pk ) {
        nbr_k = &(bonds->select.bond_list[pk]);
        nbr_k_sym = &( bonds->select.bond_list [nbr_k->sym_index] );

        rvec_Add (atoms[i].f, nbr_k_sym->t_f);
    }
}


GLOBAL void Cuda_Calculate_Bond_Orders_Init (  reax_atom *atoms, global_parameters g_params, single_body_parameters *sbp,
        static_storage workspace, int num_atom_types, int N )
{
    int i, type_i;
    real p_boc1, p_boc2;
    single_body_parameters *sbp_i;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    p_boc1 = g_params.l[0];
    p_boc2 = g_params.l[1];

    /* Calculate Deltaprime, Deltaprime_boc values */
    type_i = atoms[i].type;
    sbp_i = &(sbp[type_i]);
    workspace.Deltap[i] = workspace.total_bond_order[i] - sbp_i->valency;
    workspace.Deltap_boc[i] = 
        workspace.total_bond_order[i] - sbp_i->valency_val;
    workspace.total_bond_order[i] = 0;
}


/* A very important and crucial assumption here is that each segment
   belonging to a different atom in nbrhoods->nbr_list is sorted in its own.
   This can either be done in the general coordinator function or here */
GLOBAL void Cuda_Calculate_Bond_Orders (  reax_atom *atoms, global_parameters g_params, single_body_parameters *sbp,
        two_body_parameters *tbp, static_storage workspace, list bonds,
        list dDeltas, list dBOs, int num_atom_types, int N )
{
    int i, j, pj, type_i, type_j;
    int start_i, end_i;
    int num_bonds, sym_index;
    real p_boc1, p_boc2;
    real val_i, Deltap_i, Deltap_boc_i;
    real val_j, Deltap_j, Deltap_boc_j;
    real temp, f1, f2, f3, f4, f5, f4f5, exp_f4, exp_f5;
    real exp_p1i,    exp_p2i, exp_p1j, exp_p2j;
    real u1_ij, u1_ji, Cf1A_ij, Cf1B_ij, Cf1_ij, Cf1_ji;
    real Cf45_ij, Cf45_ji, p_lp1;
    real A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji;
    real explp1;
    two_body_parameters *twbp;
    bond_order_data *bo_ij, *bo_ji;
    single_body_parameters *sbp_i, *sbp_j;

#if defined(TEST_FORCES)
    int  k, pk, start_j, end_j;
    int  top_dbo=0, top_dDelta=0;
    dbond_data *pdbo;
    dDelta_data *ptop_dDelta;

#endif

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    num_bonds = 0;
    p_boc1 = g_params.l[0];
    p_boc2 = g_params.l[1];

    /* Calculate Deltaprime, Deltaprime_boc values */
    //for( i = 0; i < system->N; ++i ) {
    /*
       if (i < N) {
       type_i = atoms[i].type;
       sbp_i = &(sbp[type_i]);
       workspace.Deltap[i] = workspace.total_bond_order[i] - sbp_i->valency;
       workspace.Deltap_boc[i] = 
       workspace.total_bond_order[i] - sbp_i->valency_val;
       workspace.total_bond_order[i] = 0;

       }

       __syncthreads ();
     */


    // fprintf( stderr, "done with uncorrected bond orders\n" );

    /* Corrected Bond Order calculations */
    //for( i = 0; i < system->N; ++i ) {
    type_i = atoms[i].type;
    sbp_i = &(sbp[type_i]);
    val_i = sbp_i->valency;
    Deltap_i = workspace.Deltap[i];
    Deltap_boc_i = workspace.Deltap_boc[i];
    start_i = Start_Index(i, &bonds);
    end_i = End_Index(i, &bonds);
    //fprintf( stderr, "i:%d Dp:%g Dbocp:%g s:%d e:%d\n",
    //       i+1, Deltap_i, Deltap_boc_i, start_i, end_i );

    for( pj = start_i; pj < end_i; ++pj ) {
        j = bonds.select.bond_list[pj].nbr;
        type_j = atoms[j].type;
        bo_ij = &( bonds.select.bond_list[pj].bo_data );
        //fprintf( stderr, "\tj:%d - ubo: %8.3f\n", j+1, bo_ij->BO );

        if( i < j ) {
            twbp = &( tbp[ index_tbp (type_i,type_j,num_atom_types) ] );          
#ifdef TEST_FORCES
            Set_Start_Index( pj, top_dbo, &dBOs );
            /* fprintf( stderr, "%6d%6d%23.15e%23.15e%23.15e\n", 
               workspace->reverse_map[i], workspace->reverse_map[j], 
               twbp->ovc, twbp->v13cor, bo_ij->BO ); */
#endif
            if( twbp->ovc < 0.001 && twbp->v13cor < 0.001 ) {
                /* There is no correction to bond orders nor to derivatives of 
                   bond order prime! So we leave bond orders unchanged and 
                   set derivative of bond order coefficients s.t. 
                   dBO = dBOp & dBOxx = dBOxxp in Add_dBO_to_Forces */
                bo_ij->C1dbo = 1.0;
                bo_ij->C2dbo = 0.0;
                bo_ij->C3dbo = 0.0; 

                bo_ij->C1dbopi = 1.0;
                bo_ij->C2dbopi = 0.0;
                bo_ij->C3dbopi = 0.0;
                bo_ij->C4dbopi = 0.0;

                bo_ij->C1dbopi2 = 1.0; 
                bo_ij->C2dbopi2 = 0.0;
                bo_ij->C3dbopi2 = 0.0;
                bo_ij->C4dbopi2 = 0.0;

#ifdef TEST_FORCES
                pdbo = &(dBOs.select.dbo_list[ top_dbo ]);

                // compute dBO_ij/dr_i
                pdbo->wrt = i;
                rvec_Copy( pdbo->dBO, bo_ij->dBOp );
                rvec_Scale( pdbo->dBOpi, bo_ij->BO_pi, bo_ij->dln_BOp_pi );
                rvec_Scale( pdbo->dBOpi2, bo_ij->BO_pi2, bo_ij->dln_BOp_pi2 );

                // compute dBO_ij/dr_j
                pdbo++;
                pdbo->wrt = j;
                rvec_Scale( pdbo->dBO,-1.0,bo_ij->dBOp );
                rvec_Scale( pdbo->dBOpi,-bo_ij->BO_pi,bo_ij->dln_BOp_pi );
                rvec_Scale( pdbo->dBOpi2,-bo_ij->BO_pi2,bo_ij->dln_BOp_pi2 );

                top_dbo += 2;
#endif
            }
            else {
                val_j = sbp[type_j].valency;
                Deltap_j = workspace.Deltap[j];
                Deltap_boc_j = workspace.Deltap_boc[j];

                /* on page 1 */
                if( twbp->ovc >= 0.001 ) {
                    /* Correction for overcoordination */        
                    exp_p1i = EXP( -p_boc1 * Deltap_i );
                    exp_p2i = EXP( -p_boc2 * Deltap_i );
                    exp_p1j = EXP( -p_boc1 * Deltap_j );
                    exp_p2j = EXP( -p_boc2 * Deltap_j );

                    f2 = exp_p1i + exp_p1j;            
                    f3 = -1.0 / p_boc2 * log( 0.5 * ( exp_p2i  + exp_p2j ) );
                    f1 = 0.5 * ( ( val_i + f2 )/( val_i + f2 + f3 ) + 
                            ( val_j + f2 )/( val_j + f2 + f3 ) );

                    /*fprintf( stderr,"%6d%6d\t%g %g   j:%g %g  p_boc:%g %g\n",
                      i+1, j+1, val_i, Deltap_i, val_j, Deltap_j, p_boc1, p_boc2 );
                      fprintf( stderr,"\tf:%g  %g  %g, exp:%g %g %g %g\n", 
                      f1, f2, f3, exp_p1i, exp_p2i, exp_p1j, exp_p2j );*/

                    /* Now come the derivates */        
                    /* Bond Order pages 5-7, derivative of f1 */
                    temp = f2 + f3;
                    u1_ij = val_i + temp;
                    u1_ji = val_j + temp;
                    Cf1A_ij = 0.5 * f3 * (1.0 / SQR( u1_ij ) + 1.0 / SQR( u1_ji ));
                    Cf1B_ij = -0.5 * (( u1_ij - f3 ) / SQR( u1_ij ) + 
                            ( u1_ji - f3 ) / SQR( u1_ji ));

                    //Cf1_ij = -Cf1A_ij * p_boc1 * exp_p1i + 
                    //          Cf1B_ij * exp_p2i / ( exp_p2i + exp_p2j );
                    Cf1_ij = 0.50 * ( -p_boc1 * exp_p1i / u1_ij - 
                            ((val_i+f2) / SQR(u1_ij)) * 
                            ( -p_boc1 * exp_p1i + 
                              exp_p2i / ( exp_p2i + exp_p2j ) ) + 
                            -p_boc1 * exp_p1i / u1_ji - 
                            ((val_j+f2)/SQR(u1_ji)) * ( -p_boc1*exp_p1i +  
                            exp_p2i / ( exp_p2i + exp_p2j ) ));

                    Cf1_ji = -Cf1A_ij * p_boc1 * exp_p1j + 
                        Cf1B_ij * exp_p2j / ( exp_p2i + exp_p2j ); 
                    //fprintf( stderr, "\tCf1:%g  %g\n", Cf1_ij, Cf1_ji );
                }
                else {
                    /* No overcoordination correction! */
                    f1 = 1.0;
                    Cf1_ij = Cf1_ji = 0.0;          
                }

                if( twbp->v13cor >= 0.001 ) {
                    /* Correction for 1-3 bond orders */
                    exp_f4 =EXP(-(twbp->p_boc4 * SQR( bo_ij->BO ) - 
                                Deltap_boc_i) * twbp->p_boc3 + twbp->p_boc5);
                    exp_f5 =EXP(-(twbp->p_boc4 * SQR( bo_ij->BO ) - 
                                Deltap_boc_j) * twbp->p_boc3 + twbp->p_boc5);

                    f4 = 1. / (1. + exp_f4);
                    f5 = 1. / (1. + exp_f5);
                    f4f5 = f4 * f5;

                    /* Bond Order pages 8-9, derivative of f4 and f5 */
                    /*temp = twbp->p_boc5 - 
                      twbp->p_boc3 * twbp->p_boc4 * SQR( bo_ij->BO );
                      u_ij = temp + twbp->p_boc3 * Deltap_boc_i;
                      u_ji = temp + twbp->p_boc3 * Deltap_boc_j;
                      Cf45_ij = Cf45( u_ij, u_ji ) / f4f5;
                      Cf45_ji = Cf45( u_ji, u_ij ) / f4f5;*/
                    Cf45_ij = -f4 * exp_f4;
                    Cf45_ji = -f5 * exp_f5;
                }
                else {
                    f4 = f5 = f4f5 = 1.0;
                    Cf45_ij = Cf45_ji = 0.0;
                }

                /* Bond Order page 10, derivative of total bond order */
                A0_ij = f1 * f4f5;
                A1_ij = -2 * twbp->p_boc3 * twbp->p_boc4 * bo_ij->BO * 
                    (Cf45_ij + Cf45_ji);
                A2_ij = Cf1_ij / f1 + twbp->p_boc3 * Cf45_ij;
                A2_ji = Cf1_ji / f1 + twbp->p_boc3 * Cf45_ji;
                A3_ij = A2_ij + Cf1_ij / f1;
                A3_ji = A2_ji + Cf1_ji / f1;

                /*fprintf( stderr, "\tBO: %f, A0: %f, A1: %f, A2_ij: %f 
A2_ji: %f, A3_ij: %f, A3_ji: %f\n",
bo_ij->BO, A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji );*/

                /* find corrected bond order values and their deriv coefs */
                bo_ij->BO    = bo_ij->BO    * A0_ij;
                bo_ij->BO_pi = bo_ij->BO_pi * A0_ij *f1;
                bo_ij->BO_pi2= bo_ij->BO_pi2* A0_ij *f1;
                bo_ij->BO_s  = bo_ij->BO - ( bo_ij->BO_pi + bo_ij->BO_pi2 );

                bo_ij->C1dbo = A0_ij + bo_ij->BO * A1_ij;
                bo_ij->C2dbo = bo_ij->BO * A2_ij;
                bo_ij->C3dbo = bo_ij->BO * A2_ji; 

                bo_ij->C1dbopi = f1*f1*f4*f5;
                bo_ij->C2dbopi = bo_ij->BO_pi * A1_ij;
                bo_ij->C3dbopi = bo_ij->BO_pi * A3_ij;
                bo_ij->C4dbopi = bo_ij->BO_pi * A3_ji;

                bo_ij->C1dbopi2 = f1*f1*f4*f5;
                bo_ij->C2dbopi2 = bo_ij->BO_pi2 * A1_ij;
                bo_ij->C3dbopi2 = bo_ij->BO_pi2 * A3_ij;

#ifdef TEST_FORCES
                /*fprintf( stderr, "%6d%6d%13.6f%13.6f%13.6f%13.6f\n", 
                  i+1, j+1, bo_ij->BO, bo_ij->C1dbo, Cf45_ij, Cf45_ji );*/

                /* fprintf( stderr, "%6d%6d%13.6f%13.6f%13.6f%13.6f\n",
                //"%6d%6d%10.6f%10.6f%10.6f%10.6f\n%10.6f%10.6f%10.6f\n%10.6f%10.6f%10.6f%10.6f\n%10.6f%10.6f%10.6f%10.6f\n\n",
                workspace->orig_id[i], workspace->orig_id[j]
                A0_ij, A1_ij, A2_ij, A2_ji, A3_ij, A3_ji
                bo_ij->BO, bo_ij->BO_pi, bo_ij->BO_pi2, bo_ij->BO_s,
                bo_ij->C1dbo, bo_ij->C2dbo, bo_ij->C3dbo, 
                bo_ij->C1dbopi,bo_ij->C2dbopi,bo_ij->C3dbopi,bo_ij->C4dbopi,
                bo_ij->C1dbopi2,bo_ij->C2dbopi2,bo_ij->C3dbopi2,bo_ij->C4dbopi2
                ); */

                Calculate_dBO( i, pj, workspace, lists, &top_dbo );
#endif
            }

            /* neglect bonds that are < 1e-10 */
            if( bo_ij->BO < 1e-10 )
                bo_ij->BO = 0.0;
            if( bo_ij->BO_s < 1e-10 )
                bo_ij->BO_s = 0.0;
            if( bo_ij->BO_pi < 1e-10 )
                bo_ij->BO_pi = 0.0;
            if( bo_ij->BO_pi2 < 1e-10 )
                bo_ij->BO_pi2 = 0.0;

            workspace.total_bond_order[i] += bo_ij->BO; // now keeps total_BO


            /* fprintf( stderr, "%d %d\t%g %g %g %g\n
Cdbo:\t%g %g %g\n
Cdbopi:\t%g %g %g %g\n
Cdbopi2:%g %g %g %g\n\n", 
i+1, j+1, bonds->select.bond_list[ pj ].d, 
bo_ij->BO,bo_ij->BO_pi, bo_ij->BO_pi2, 
bo_ij->C1dbo, bo_ij->C2dbo, bo_ij->C3dbo,
bo_ij->C1dbopi, bo_ij->C2dbopi, bo_ij->C3dbopi, bo_ij->C4dbopi,
bo_ij->C1dbopi2, bo_ij->C2dbopi2, 
bo_ij->C3dbopi2, bo_ij->C4dbopi2 ); */

            /* fprintf( stderr, "%d %d, BO:%f BO_s:%f BO_pi:%f BO_pi2:%f\n",
               i+1,j+1,bo_ij->BO,bo_ij->BO_s,bo_ij->BO_pi,bo_ij->BO_pi2 ); */

#ifdef TEST_FORCES
            Set_End_Index( pj, top_dbo, &dBOs );
            //Add_dBO( system, lists, i, pj, 1.0, workspace->dDelta );
#endif
        }
        /*
           else {
        // We only need to update bond orders from bo_ji
        //   everything else is set in uncorrected_bo calculations 
        sym_index = bonds.select.bond_list[pj].sym_index;
        bo_ji = &(bonds.select.bond_list[ sym_index ].bo_data);
        bo_ij->BO = bo_ji->BO;
        bo_ij->BO_s = bo_ji->BO_s;
        bo_ij->BO_pi = bo_ji->BO_pi;
        bo_ij->BO_pi2 = bo_ji->BO_pi2;

        workspace.total_bond_order[i] += bo_ij->BO; // now keeps total_BO

#ifdef TEST_FORCES
        //Add_dBO( system, lists, j, sym_index, 1.0, workspace.dDelta );
#endif
}      
         */
    }

#ifdef TEST_FORCES 
    // fprintf( stderr, "dDelta computations\nj:" );
    Set_Start_Index( i, top_dDelta, &dDeltas );
    ptop_dDelta = &( dDeltas.select.dDelta_list[top_dDelta] );
    
    for( pj = start_i; pj < end_i; ++pj ) {
        j = bonds.select.bond_list[pj].nbr;
        // fprintf( stderr, "%d  ", j );
    
        if( !rvec_isZero( workspace.dDelta[j] ) ) {
            ptop_dDelta->wrt = j;
            rvec_Copy( ptop_dDelta->dVal, workspace.dDelta[j] );
            rvec_MakeZero( workspace.dDelta[j] );
            ++top_dDelta, ++ptop_dDelta;
        }
    
        start_j = Start_Index(j, &bonds);
        end_j = End_Index(j, &bonds);     
        for( pk = start_j; pk < end_j; ++pk ) {
            k = bonds.select.bond_list[pk].nbr;    
            if( !rvec_isZero( workspace.dDelta[k] ) ) {
                ptop_dDelta->wrt = k;
                rvec_Copy( ptop_dDelta->dVal, workspace.dDelta[k] );
                rvec_MakeZero( workspace.dDelta[k] );
                ++top_dDelta, ++ptop_dDelta;
            }
        }
    }
    
    Set_End_Index( i, top_dDelta, &dDeltas );
    
    /*for( pj=Start_Index(i,dDeltas); pj<End_Index(i,dDeltas); ++pj )
      fprintf( stdout, "dDel: %d %d [%g %g %g]\n",
      i+1, dDeltas->select.dDelta_list[pj].wrt+1,
      dDeltas->select.dDelta_list[pj].dVal[0], 
      dDeltas->select.dDelta_list[pj].dVal[1], 
      dDeltas->select.dDelta_list[pj].dVal[2] );*/
#endif
    //}
    
    /*fprintf(stderr,"\tCalculated actual bond orders ...\n" );
      fprintf(stderr,"%6s%8s%8s%8s%8s%8s%8s%8s\n", 
      "atom", "Delta", "Delta_e", "Delta_boc", "nlp", 
      "Delta_lp", "Clp", "dDelta_lp" );*/
    
    /*
       p_lp1 = g_params.l[15];
    
    //get the kernel ID for the following computation
    j = i;
    
    // Calculate some helper variables that are  used at many places 
    //  throughout force calculations 
    //for( j = 0; j < system->N; ++j ) {
    type_j = atoms[j].type;
    sbp_j = &(sbp[ type_j ]);
    
    workspace.Delta[j] = workspace.total_bond_order[j] - sbp_j->valency;
    workspace.Delta_e[j] = workspace.total_bond_order[j] - sbp_j->valency_e;
    workspace.Delta_boc[j] = workspace.total_bond_order[j] - 
    sbp_j->valency_boc;
    
    workspace.vlpex[j] =  workspace.Delta_e[j] - 
    2.0 * (int)(workspace.Delta_e[j]/2.0);
    explp1 = EXP(-p_lp1 * SQR(2.0 + workspace.vlpex[j]));
    workspace.nlp[j] = explp1 - (int)(workspace.Delta_e[j] / 2.0);
    workspace.Delta_lp[j] = sbp_j->nlp_opt - workspace.nlp[j];
    workspace.Clp[j] = 2.0 * p_lp1 * explp1 * (2.0 + workspace.vlpex[j]);
    // Adri uses different dDelta_lp values than the ones in notes... //
    workspace.dDelta_lp[j] = workspace.Clp[j];
    //workspace->dDelta_lp[j] = workspace->Clp[j] + (0.5-workspace->Clp[j]) *
    //((fabs(workspace->Delta_e[j]/2.0 - 
    //       (int)(workspace->Delta_e[j]/2.0)) < 0.1) ? 1 : 0 );
    
    if( sbp_j->mass > 21.0 ) {
    workspace.nlp_temp[j] = 0.5 * (sbp_j->valency_e - sbp_j->valency);
    workspace.Delta_lp_temp[j] = sbp_j->nlp_opt - workspace.nlp_temp[j];
    workspace.dDelta_lp_temp[j] = 0.;
    }
    else {
    workspace.nlp_temp[j] = workspace.nlp[j];
    workspace.Delta_lp_temp[j] = sbp_j->nlp_opt - workspace.nlp_temp[j];
    workspace.dDelta_lp_temp[j] = workspace.Clp[j];
    }
    
    //fprintf( stderr, "%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
    //j, workspace->Delta[j], workspace->Delta_e[j], workspace->Delta_boc[j], 
    //workspace->nlp[j], system->reaxprm.sbp[type_j].nlp_opt,
    //workspace->Delta_lp[j], workspace->Clp[j], workspace->dDelta_lp[j] );
    //}
     */
    
    //Print_Bonds( system, bonds, "sbonds.out" );
    
#if defined(DEBUG)
    //fprintf( stderr, "Number of bonds: %d\n", num_bonds );
    //Print_Bond_Orders( system, control, data, workspace, lists, out_control );
#endif
}


GLOBAL void Cuda_Update_Uncorrected_BO (  static_storage workspace, list bonds, int N )
{
    int i, j, pj;
    int start_i, end_i;
    int sym_index;

    bond_order_data *bo_ij, *bo_ji;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    start_i = Start_Index(i, &bonds);
    end_i = End_Index(i, &bonds);

    for( pj = start_i; pj < end_i; ++pj ) {

        j = bonds.select.bond_list[pj].nbr;
        bo_ij = &( bonds.select.bond_list[pj].bo_data );

        if( i >= j ) {
            // We only need to update bond orders from bo_ji
            //   everything else is set in uncorrected_bo calculations 
            sym_index = bonds.select.bond_list[pj].sym_index;
            bo_ji = &(bonds.select.bond_list[ sym_index ].bo_data);
            bo_ij->BO = bo_ji->BO;
            bo_ij->BO_s = bo_ji->BO_s;
            bo_ij->BO_pi = bo_ji->BO_pi;
            bo_ij->BO_pi2 = bo_ji->BO_pi2;

            workspace.total_bond_order[i] += bo_ij->BO; // now keeps total_BO
        }
    }
}


GLOBAL void Cuda_Update_Workspace_After_Bond_Orders(  reax_atom *atoms, global_parameters g_params, single_body_parameters *sbp,
        static_storage workspace, int N )
{
    int j, type_j;
    real explp1;
    real p_lp1;
    single_body_parameters *sbp_i, *sbp_j;

    j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    p_lp1 = g_params.l[15];

    /* Calculate some helper variables that are  used at many places 
       throughout force calculations */
    //for( j = 0; j < system->N; ++j ) {
    type_j = atoms[j].type;
    sbp_j = &(sbp[ type_j ]);

    workspace.Delta[j] = workspace.total_bond_order[j] - sbp_j->valency;
    workspace.Delta_e[j] = workspace.total_bond_order[j] - sbp_j->valency_e;
    workspace.Delta_boc[j] = workspace.total_bond_order[j] - 
        sbp_j->valency_boc;

    workspace.vlpex[j] =  workspace.Delta_e[j] - 
        2.0 * (int)(workspace.Delta_e[j]/2.0);
    explp1 = EXP(-p_lp1 * SQR(2.0 + workspace.vlpex[j]));
    workspace.nlp[j] = explp1 - (int)(workspace.Delta_e[j] / 2.0);
    workspace.Delta_lp[j] = sbp_j->nlp_opt - workspace.nlp[j];
    workspace.Clp[j] = 2.0 * p_lp1 * explp1 * (2.0 + workspace.vlpex[j]);
    /* Adri uses different dDelta_lp values than the ones in notes... */
    workspace.dDelta_lp[j] = workspace.Clp[j];
    //workspace->dDelta_lp[j] = workspace->Clp[j] + (0.5-workspace->Clp[j]) *
    //((fabs(workspace->Delta_e[j]/2.0 - 
    //       (int)(workspace->Delta_e[j]/2.0)) < 0.1) ? 1 : 0 );

    if( sbp_j->mass > 21.0 ) {
        workspace.nlp_temp[j] = 0.5 * (sbp_j->valency_e - sbp_j->valency);
        workspace.Delta_lp_temp[j] = sbp_j->nlp_opt - workspace.nlp_temp[j];
        workspace.dDelta_lp_temp[j] = 0.;
    }
    else {
        workspace.nlp_temp[j] = workspace.nlp[j];
        workspace.Delta_lp_temp[j] = sbp_j->nlp_opt - workspace.nlp_temp[j];
        workspace.dDelta_lp_temp[j] = workspace.Clp[j];
    }

    //fprintf( stderr, "%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
    //j, workspace->Delta[j], workspace->Delta_e[j], workspace->Delta_boc[j], 
    //workspace->nlp[j], system->reaxprm.sbp[type_j].nlp_opt,
    //workspace->Delta_lp[j], workspace->Clp[j], workspace->dDelta_lp[j] );
    //}

}


//Import from the forces file. 
GLOBAL void Cuda_Compute_Total_Force (reax_atom *atoms, simulation_data *data, 
        static_storage workspace, list p_bonds, int ensemble, int N)
{
    int i, pj;
    list *bonds = &p_bonds;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) 
    {
        for (pj = Start_Index (i, bonds); pj < End_Index (i, bonds); ++pj)
        {
            //int j = bonds->select.bond_list[pj].nbr;
            if (ensemble == NVE || ensemble == NVT || ensemble == bNVT) 
                Cuda_Add_dBond_to_Forces (i, pj, atoms, &workspace, bonds );
            else 
                Cuda_Add_dBond_to_Forces_NPT (i, pj, atoms, data, &workspace, bonds );
        }
    }
}


GLOBAL void Cuda_Compute_Total_Force_PostProcess (reax_atom *atoms, simulation_data *data, 
        static_storage workspace, list p_bonds, int ensemble, int N)
{
    int i, pj;
    list *bonds = &p_bonds;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) 
    {
        if (ensemble == NVE || ensemble == NVT || ensemble == bNVT) 
            Cuda_dbond_to_Forces_postprocess (i, atoms, bonds );
    }
}
