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

#include "single_body_interactions.h"
#include "bond_orders.h"
#include "list.h"
#include "lookup.h"
#include "vector.h"

#include "index_utils.h"
#include "cuda_helpers.h"


void LonePair_OverUnder_Coordination_Energy( reax_system *system, 
					     control_params *control, 
					     simulation_data *data,
					     static_storage *workspace, 
					     list **lists, 
					     output_controls *out_control )
{
  int i, j, pj, type_i, type_j;
  real Delta_lpcorr, dfvl;
  real e_lp, expvd2, inv_expvd2, dElp, CElp, DlpVi;
  real e_lph, Di, vov3, deahu2dbo, deahu2dsbo;
  real e_ov, CEover1, CEover2, CEover3, CEover4;
  real exp_ovun1, exp_ovun2, sum_ovun1, sum_ovun2;
  real exp_ovun2n, exp_ovun6, exp_ovun8;
  real inv_exp_ovun1, inv_exp_ovun2, inv_exp_ovun2n, inv_exp_ovun8;
  real e_un, CEunder1, CEunder2, CEunder3, CEunder4;
  real p_lp1, p_lp2, p_lp3;
  real p_ovun2, p_ovun3, p_ovun4, p_ovun5, p_ovun6, p_ovun7, p_ovun8;

  single_body_parameters *sbp_i, *sbp_j;
  two_body_parameters *twbp;
  bond_data *pbond;
  bond_order_data *bo_ij; 
  list *bonds = (*lists) + BONDS;

  /* Initialize parameters */
  p_lp1 = system->reaxprm.gp.l[15];
  p_lp3 = system->reaxprm.gp.l[5];
  p_ovun3 = system->reaxprm.gp.l[32];
  p_ovun4 = system->reaxprm.gp.l[31];
  p_ovun6 = system->reaxprm.gp.l[6];
  p_ovun7 = system->reaxprm.gp.l[8];
  p_ovun8 = system->reaxprm.gp.l[9];

  for( i = 0; i < system->N; ++i ) {
    /* set the parameter pointer */
    type_i = system->atoms[i].type;
    sbp_i = &(system->reaxprm.sbp[ type_i ]);

    /* lone-pair Energy */
    p_lp2 = sbp_i->p_lp2;      
    expvd2 = EXP( -75 * workspace->Delta_lp[i] );
    inv_expvd2 = 1. / (1. + expvd2 );
      
    /* calculate the energy */
    data->E_Lp += e_lp = 
      p_lp2 * workspace->Delta_lp[i] * inv_expvd2;
	
    dElp = p_lp2 * inv_expvd2 + 
      75 * p_lp2 * workspace->Delta_lp[i] * expvd2 * SQR(inv_expvd2);
    CElp = dElp * workspace->dDelta_lp[i];

    workspace->CdDelta[i] += CElp;      // lp - 1st term

#ifdef TEST_ENERGY
    fprintf( out_control->elp, "%23.15e%23.15e%23.15e%23.15e\n", 
	     p_lp2, workspace->Delta_lp_temp[i], expvd2, dElp );
    fprintf( out_control->elp, "%6d%23.15e%23.15e%23.15e\n",
	     workspace->orig_id[i]+1, workspace->nlp[i], e_lp, data->E_Lp );
#endif
#ifdef TEST_FORCES
    Add_dDelta( system, lists, i, CElp, workspace->f_lp );  // lp - 1st term
#endif

    /* correction for C2 */
    if( system->reaxprm.gp.l[5] > 0.001 && 
	!strcmp( system->reaxprm.sbp[type_i].name, "C" ) )
      for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
	if( i < bonds->select.bond_list[pj].nbr ) {
	  j = bonds->select.bond_list[pj].nbr;
	  type_j = system->atoms[j].type;
	      
	  if( !strcmp( system->reaxprm.sbp[type_j].name, "C" ) ) {
	    twbp = &( system->reaxprm.tbp[ index_tbp (type_i,type_j,&system->reaxprm) ]);
	    bo_ij = &( bonds->select.bond_list[pj].bo_data );
	    Di = workspace->Delta[i];
	    vov3 = bo_ij->BO - Di - 0.040*POW(Di, 4.);

	    if( vov3 > 3. ) {
	      data->E_Lp += e_lph = p_lp3 * SQR(vov3-3.0);
	      //estrain(i) += e_lph;

	      deahu2dbo = 2.*p_lp3*(vov3 - 3.);
	      deahu2dsbo = 2.*p_lp3*(vov3 - 3.)*(-1. - 0.16*POW(Di, 3.));

	      bo_ij->Cdbo += deahu2dbo;
	      workspace->CdDelta[i] += deahu2dsbo;
#ifdef TEST_ENERGY
	      fprintf(out_control->elp,"C2cor%6d%6d%23.15e%23.15e%23.15e\n",
		      // workspace->orig_id[i], workspace->orig_id[j],
		      i+1, j+1, e_lph, deahu2dbo, deahu2dsbo );
#endif
#ifdef TEST_FORCES
	      Add_dBO(system, lists, i, pj, deahu2dbo, workspace->f_lp);
	      Add_dDelta(system, lists, i, deahu2dsbo, workspace->f_lp);
#endif
	    }
	  }
	      
	}
  }


  for( i = 0; i < system->N; ++i ) {
    type_i = system->atoms[i].type;
    sbp_i = &(system->reaxprm.sbp[ type_i ]);

    /* over-coordination energy */
    if( sbp_i->mass > 21.0 ) 
      dfvl = 0.0;
    else dfvl = 1.0; // only for 1st-row elements
      
    p_ovun2 = sbp_i->p_ovun2;
    sum_ovun1 = 0;
    sum_ovun2 = 0;

    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
      j = bonds->select.bond_list[pj].nbr;
      type_j = system->atoms[j].type;	  
      bo_ij = &(bonds->select.bond_list[pj].bo_data);
      sbp_j = &(system->reaxprm.sbp[ type_j ]);
      twbp = &(system->reaxprm.tbp[ index_tbp (type_i,type_j,&system->reaxprm) ]);

      sum_ovun1 += twbp->p_ovun1 * twbp->De_s * bo_ij->BO;
      sum_ovun2 += (workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j])*
	( bo_ij->BO_pi + bo_ij->BO_pi2 );

      /*fprintf( stdout, "%4d%4d%23.15e%23.15e%23.15e\n", 
	i+1, j+1, 
	dfvl * workspace->Delta_lp_temp[j],
	sbp_j->nlp_opt,
	workspace->nlp_temp[j] );*/
    }
			
    exp_ovun1 = p_ovun3 * EXP( p_ovun4 * sum_ovun2 );
    inv_exp_ovun1 = 1.0 / (1 + exp_ovun1);
    Delta_lpcorr  = workspace->Delta[i] - 
      (dfvl*workspace->Delta_lp_temp[i]) * inv_exp_ovun1;
	
    exp_ovun2 = EXP( p_ovun2 * Delta_lpcorr );
    inv_exp_ovun2 = 1.0 / (1.0 + exp_ovun2);
	
    DlpVi = 1.0 / (Delta_lpcorr + sbp_i->valency + 1e-8 );
    CEover1 = Delta_lpcorr * DlpVi * inv_exp_ovun2;
			
    data->E_Ov += e_ov = sum_ovun1 * CEover1;

    CEover2 = sum_ovun1 * DlpVi * inv_exp_ovun2 *
      ( 1.0 - Delta_lpcorr*( DlpVi + p_ovun2 * exp_ovun2 * inv_exp_ovun2 ) );

    CEover3 = CEover2 * (1.0 - dfvl*workspace->dDelta_lp[i]*inv_exp_ovun1 );

    CEover4 = CEover2 * (dfvl*workspace->Delta_lp_temp[i]) * 
      p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1);

      
    /* under-coordination potential */
    p_ovun2 = sbp_i->p_ovun2;
    p_ovun5 = sbp_i->p_ovun5;
	
    exp_ovun2n = 1.0 / exp_ovun2;
    exp_ovun6 = EXP( p_ovun6 * Delta_lpcorr );
    exp_ovun8 = p_ovun7 * EXP(p_ovun8 * sum_ovun2);
    inv_exp_ovun2n = 1.0 / (1.0 + exp_ovun2n);
    inv_exp_ovun8 = 1.0 / (1.0 + exp_ovun8);

    data->E_Un += e_un =
      -p_ovun5 * (1.0 - exp_ovun6) * inv_exp_ovun2n * inv_exp_ovun8;
			
    CEunder1 = inv_exp_ovun2n * ( p_ovun5*p_ovun6*exp_ovun6*inv_exp_ovun8 +
				  p_ovun2 * e_un * exp_ovun2n);
    CEunder2 = -e_un * p_ovun8 * exp_ovun8 * inv_exp_ovun8;
    CEunder3 = CEunder1 * (1.0 - dfvl*workspace->dDelta_lp[i]*inv_exp_ovun1);
    CEunder4 = CEunder1 * (dfvl*workspace->Delta_lp_temp[i]) * 
      p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1) + CEunder2;

    //fprintf( stdout, "%6d%23.15e%23.15e%23.15e\n",
    //       i+1, sum_ovun2, e_ov, e_un );

    /* forces */
    workspace->CdDelta[i] += CEover3;   // OvCoor - 2nd term
    workspace->CdDelta[i] += CEunder3;  // UnCoor - 1st term

#ifdef TEST_FORCES
    Add_dDelta( system, lists, i, CEover3, workspace->f_ov );  // OvCoor - 2nd
    Add_dDelta( system, lists, i, CEunder3, workspace->f_un ); // UnCoor - 1st
#endif
      

    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ){
      pbond = &(bonds->select.bond_list[pj]);
      j = pbond->nbr;
      type_j = system->atoms[j].type;
      bo_ij = &(pbond->bo_data);
      twbp  = &(system->reaxprm.tbp[ index_tbp (type_i,type_j,&system->reaxprm) ]);


      bo_ij->Cdbo += CEover1 * twbp->p_ovun1 * twbp->De_s; // OvCoor - 1st  
      workspace->CdDelta[j] += CEover4*(1.0 - dfvl*workspace->dDelta_lp[j])*
	(bo_ij->BO_pi + bo_ij->BO_pi2); // OvCoor - 3a
      bo_ij->Cdbopi += CEover4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//OvCoor-3b
      bo_ij->Cdbopi2 += CEover4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//OvCoor-3b


      workspace->CdDelta[j] += CEunder4*(1.0-dfvl*workspace->dDelta_lp[j]) *
	(bo_ij->BO_pi + bo_ij->BO_pi2);   // UnCoor - 2a
      bo_ij->Cdbopi += CEunder4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//UnCoor-2b
      bo_ij->Cdbopi2 += CEunder4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//UnCoor-2b


#ifdef TEST_ENERGY
      /* fprintf( out_control->eov, "%6d%23.15e%23.15e"
	 workspace->orig_id[j]+1,
	 //twbp->p_ovun1,twbp->De_s,Delta_lpcorr*DlpVi*inv_exp_ovun2,
	 CEover1*twbp->p_ovun1*twbp->De_s, CEover3 ); */

      /*fprintf( out_control->eov, "%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n", 
	workspace->orig_id[j]+1, 
	CEover4,
	CEover4*
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]),
	CEover4 * (bo_ij->BO_pi + bo_ij->BO_pi2), 
	(1.0 - dfvl*workspace->dDelta_lp[j]),
	CEover4 * (1.0 - dfvl*workspace->dDelta_lp[j]) * 
	(bo_ij->BO_pi + bo_ij->BO_pi2) );*/
	  
      /* fprintf( out_control->eun, "%6d%23.15e\n",
	 workspace->orig_id[j]+1, CEunder3 ); */

      /*fprintf( out_control->eun, "%6d%23.15e%23.15e%23.15e%23.15e\n",
	workspace->orig_id[j]+1,
	CEunder4,
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]),
	CEunder4*
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]),
	CEunder4*(1.0 - dfvl*workspace->dDelta_lp[j])*
	(bo_ij->BO_pi + bo_ij->BO_pi2) );*/
#endif

#ifdef TEST_FORCES
      Add_dBO( system, lists, i, pj, CEover1 * twbp->p_ovun1 * twbp->De_s, 
	       workspace->f_ov ); // OvCoor - 1st term

      Add_dDelta( system, lists, j,
		  CEover4 * (1.0 - dfvl*workspace->dDelta_lp[j]) * 
		  (bo_ij->BO_pi+bo_ij->BO_pi2), workspace->f_ov );//OvCoor3a

      Add_dBOpinpi2( system, lists, i, pj, 
		     CEover4 * (workspace->Delta[j] - 
				dfvl * workspace->Delta_lp_temp[j]),
		     CEover4 * (workspace->Delta[j] - 
				dfvl * workspace->Delta_lp_temp[j]),
		     workspace->f_ov, workspace->f_ov ); // OvCoor - 3b

      Add_dDelta( system, lists, j,
		  CEunder4 * (1.0 - dfvl*workspace->dDelta_lp[j]) * 
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

#ifdef TEST_ENERGY      

    fprintf( out_control->eov, "%6d%15.8f%15.8f%15.8f\n", 
	     i+1, DlpVi, Delta_lpcorr, sbp_i->valency ); 

    fprintf( out_control->eov, "%6d%15.8f%15.8f\n", 
	     i+1/*workspace->orig_id[i]+1*/, e_ov, data->E_Ov + data->E_Un );

    fprintf( out_control->eov, "%6d%15.8f%15.8f\n", 
	     i+1/*workspace->orig_id[i]+1*/, e_un, data->E_Ov + data->E_Un );
#endif
  }
}









//CUDA Functions
GLOBAL void Cuda_LonePair_OverUnder_Coordination_Energy ( reax_atom *atoms, global_parameters g_params, 
																	single_body_parameters *sbp, two_body_parameters *tbp, 
																	static_storage p_workspace, simulation_data *data,
																	list p_bonds, int N, int num_atom_types )
{
  int i, j, pj, type_i, type_j;
  real Delta_lpcorr, dfvl;
  real e_lp, expvd2, inv_expvd2, dElp, CElp, DlpVi;
  real e_lph, Di, vov3, deahu2dbo, deahu2dsbo;
  real e_ov, CEover1, CEover2, CEover3, CEover4;
  real exp_ovun1, exp_ovun2, sum_ovun1, sum_ovun2;
  real exp_ovun2n, exp_ovun6, exp_ovun8;
  real inv_exp_ovun1, inv_exp_ovun2, inv_exp_ovun2n, inv_exp_ovun8;
  real e_un, CEunder1, CEunder2, CEunder3, CEunder4;
  real p_lp1, p_lp2, p_lp3;
  real p_ovun2, p_ovun3, p_ovun4, p_ovun5, p_ovun6, p_ovun7, p_ovun8;

  single_body_parameters *sbp_i, *sbp_j;
  two_body_parameters *twbp;
  bond_data *pbond;
  bond_order_data *bo_ij; 
  list *bonds = &p_bonds;
  static_storage *workspace = &p_workspace;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  //if (i >= N) return;

  /* Initialize parameters */
  p_lp1 = g_params.l[15];
  p_lp3 = g_params.l[5];
  p_ovun3 = g_params.l[32];
  p_ovun4 = g_params.l[31];
  p_ovun6 = g_params.l[6];
  p_ovun7 = g_params.l[8];
  p_ovun8 = g_params.l[9];

  //for( i = 0; i < system->N; ++i ) {
  if (i < N) {
    // set the parameter pointer 
    type_i = atoms[i].type;
    sbp_i = &(sbp[ type_i ]);

    // lone-pair Energy 
    p_lp2 = sbp_i->p_lp2;      
    expvd2 = EXP( -75 * workspace->Delta_lp[i] );
    inv_expvd2 = 1. / (1. + expvd2 );
      
    // calculate the energy 
    e_lp = p_lp2 * workspace->Delta_lp[i] * inv_expvd2;
	
	 //PERFORMANCE IMPACT
    atomicAdd (&data->E_Lp, e_lp);
	
    dElp = p_lp2 * inv_expvd2 + 
      75 * p_lp2 * workspace->Delta_lp[i] * expvd2 * SQR(inv_expvd2);
    CElp = dElp * workspace->dDelta_lp[i];

	//PERFORMANCE IMPACT
    //workspace->CdDelta[i] += CElp;      // lp - 1st term
    atomicAdd (&workspace->CdDelta[i], CElp);


#ifdef TEST_ENERGY
		//TODO
    //fprintf( out_control->elp, "%23.15e%23.15e%23.15e%23.15e\n", 
	  //   p_lp2, workspace->Delta_lp_temp[i], expvd2, dElp );
    //fprintf( out_control->elp, "%6d%23.15e%23.15e%23.15e\n",
	  //   workspace->orig_id[i]+1, workspace->nlp[i], e_lp, data->E_Lp );
#endif
#ifdef TEST_FORCES
	 //TODO
    //Add_dDelta( system, lists, i, CElp, workspace->f_lp );  // lp - 1st term
	 //TODO
#endif

    // correction for C2 
    if( g_params.l[5] > 0.001 && 
	!cuda_strcmp( sbp[type_i].name, "C" , 15) )
      for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
	if( i < bonds->select.bond_list[pj].nbr ) {
	  j = bonds->select.bond_list[pj].nbr;
	  type_j = atoms[j].type;
	      
	  if( !cuda_strcmp( sbp[type_j].name, "C", 15 ) ) {
	    twbp = &( tbp[ index_tbp (type_i,type_j,num_atom_types) ]);
	    bo_ij = &( bonds->select.bond_list[pj].bo_data );
	    Di = workspace->Delta[i];
	    vov3 = bo_ij->BO - Di - 0.040*POW(Di, 4.);

	    if( vov3 > 3. ) {

		   //PERFORMANCE IMPACT
	      e_lph = p_lp3 * SQR(vov3-3.0);
	      atomicAdd (&data->E_Lp, e_lph );
	      //estrain(i) += e_lph;

	      deahu2dbo = 2.*p_lp3*(vov3 - 3.);
	      deahu2dsbo = 2.*p_lp3*(vov3 - 3.)*(-1. - 0.16*POW(Di, 3.));

	      bo_ij->Cdbo += deahu2dbo;


			//PERFORMANCE IMPACT
	      atomicAdd (&workspace->CdDelta[i], deahu2dsbo);
#ifdef TEST_ENERGY
			//TODO
	      //fprintf(out_control->elp,"C2cor%6d%6d%23.15e%23.15e%23.15e\n",
		      // workspace->orig_id[i], workspace->orig_id[j],
		    //  i+1, j+1, e_lph, deahu2dbo, deahu2dsbo );
#endif
#ifdef TEST_FORCES
			//TODO
	      //Add_dBO(system, lists, i, pj, deahu2dbo, workspace->f_lp);
	      //Add_dDelta(system, lists, i, deahu2dsbo, workspace->f_lp);
#endif
	    }
	  }
	      
	}
  } // end of if statement for the all the threads

  __syncthreads ();

	 //TODO
	 if (i >= N) return;
	 //TODO


  //for( i = 0; i < system->N; ++i ) {
    type_i = atoms[i].type;
    sbp_i = &(sbp[ type_i ]);

    // over-coordination energy 
    if( sbp_i->mass > 21.0 ) 
      dfvl = 0.0;
    else dfvl = 1.0; // only for 1st-row elements
      
    p_ovun2 = sbp_i->p_ovun2;
    sum_ovun1 = 0;
    sum_ovun2 = 0;

    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
      j = bonds->select.bond_list[pj].nbr;
      type_j = atoms[j].type;	  
      bo_ij = &(bonds->select.bond_list[pj].bo_data);
      sbp_j = &(sbp[ type_j ]);
      twbp = &(tbp[ index_tbp (type_i,type_j,num_atom_types) ]);

      sum_ovun1 += twbp->p_ovun1 * twbp->De_s * bo_ij->BO;
      sum_ovun2 += (workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j])*
	( bo_ij->BO_pi + bo_ij->BO_pi2 );

      //fprintf( stdout, "%4d%4d%23.15e%23.15e%23.15e\n", 
	//i+1, j+1, 
	//dfvl * workspace->Delta_lp_temp[j],
	//sbp_j->nlp_opt,
	//workspace->nlp_temp[j] );
    }

	 //__syncthreads ();


			
    exp_ovun1 = p_ovun3 * EXP( p_ovun4 * sum_ovun2 );
    inv_exp_ovun1 = 1.0 / (1 + exp_ovun1);
    Delta_lpcorr  = workspace->Delta[i] - 
      (dfvl*workspace->Delta_lp_temp[i]) * inv_exp_ovun1;
	
    exp_ovun2 = EXP( p_ovun2 * Delta_lpcorr );
    inv_exp_ovun2 = 1.0 / (1.0 + exp_ovun2);
	
    DlpVi = 1.0 / (Delta_lpcorr + sbp_i->valency + 1e-8 );
    CEover1 = Delta_lpcorr * DlpVi * inv_exp_ovun2;
			
	//PERFORMANCE IMPACT
    //data->E_Ov += e_ov = sum_ovun1 * CEover1;
    e_ov = sum_ovun1 * CEover1;
    atomicAdd (&data->E_Ov, e_ov ); 

    CEover2 = sum_ovun1 * DlpVi * inv_exp_ovun2 *
      ( 1.0 - Delta_lpcorr*( DlpVi + p_ovun2 * exp_ovun2 * inv_exp_ovun2 ) );

    CEover3 = CEover2 * (1.0 - dfvl*workspace->dDelta_lp[i]*inv_exp_ovun1 );

    CEover4 = CEover2 * (dfvl*workspace->Delta_lp_temp[i]) * 
      p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1);

      
    // under-coordination potential 
    p_ovun2 = sbp_i->p_ovun2;
    p_ovun5 = sbp_i->p_ovun5;
	
    exp_ovun2n = 1.0 / exp_ovun2;
    exp_ovun6 = EXP( p_ovun6 * Delta_lpcorr );
    exp_ovun8 = p_ovun7 * EXP(p_ovun8 * sum_ovun2);
    inv_exp_ovun2n = 1.0 / (1.0 + exp_ovun2n);
    inv_exp_ovun8 = 1.0 / (1.0 + exp_ovun8);

	//PERFORMANCE IMPACT
    e_un = -p_ovun5 * (1.0 - exp_ovun6) * inv_exp_ovun2n * inv_exp_ovun8;
    atomicAdd (&data->E_Un, e_un );
			
    CEunder1 = inv_exp_ovun2n * ( p_ovun5*p_ovun6*exp_ovun6*inv_exp_ovun8 +
				  p_ovun2 * e_un * exp_ovun2n);
    CEunder2 = -e_un * p_ovun8 * exp_ovun8 * inv_exp_ovun8;
    CEunder3 = CEunder1 * (1.0 - dfvl*workspace->dDelta_lp[i]*inv_exp_ovun1);
    CEunder4 = CEunder1 * (dfvl*workspace->Delta_lp_temp[i]) * 
      p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1) + CEunder2;

    //fprintf( stdout, "%6d%23.15e%23.15e%23.15e\n",
    //       i+1, sum_ovun2, e_ov, e_un );

    // forces 
	 //PERFORMANCE IMPACT
    atomicAdd (&workspace->CdDelta[i] , CEover3);   // OvCoor - 2nd term
    atomicAdd (&workspace->CdDelta[i], CEunder3);  // UnCoor - 1st term

#ifdef TEST_FORCES
	 //TODO
    //Add_dDelta( system, lists, i, CEover3, workspace->f_ov );  // OvCoor - 2nd
    //Add_dDelta( system, lists, i, CEunder3, workspace->f_un ); // UnCoor - 1st
#endif
      
	//__syncthreads ();

    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ){
      pbond = &(bonds->select.bond_list[pj]);
      j = pbond->nbr;
      type_j = atoms[j].type;
      bo_ij = &(pbond->bo_data);
      twbp  = &(tbp[ index_tbp (type_i,type_j,num_atom_types) ]);


      bo_ij->Cdbo += CEover1 * twbp->p_ovun1 * twbp->De_s; // OvCoor - 1st  

		//PERFORMANCE IMPACT
      atomicAdd (&workspace->CdDelta[j], CEover4*(1.0 - dfvl*workspace->dDelta_lp[j])* (bo_ij->BO_pi + bo_ij->BO_pi2)); // OvCoor - 3a

      bo_ij->Cdbopi += CEover4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//OvCoor-3b
      bo_ij->Cdbopi2 += CEover4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//OvCoor-3b


		//PERFORMANCE IMPACT
      atomicAdd (&workspace->CdDelta[j], CEunder4*(1.0-dfvl*workspace->dDelta_lp[j]) * (bo_ij->BO_pi + bo_ij->BO_pi2) );   // UnCoor - 2a

      bo_ij->Cdbopi += CEunder4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//UnCoor-2b
      bo_ij->Cdbopi2 += CEunder4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//UnCoor-2b


#ifdef TEST_ENERGY
//       fprintf( out_control->eov, "%6d%23.15e%23.15e"
//	 workspace->orig_id[j]+1,
	 //twbp->p_ovun1,twbp->De_s,Delta_lpcorr*DlpVi*inv_exp_ovun2,
//	 CEover1*twbp->p_ovun1*twbp->De_s, CEover3 ); 

//      fprintf( out_control->eov, "%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n", 
//	workspace->orig_id[j]+1, 
//	CEover4,
//	CEover4*
//	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]),
//	CEover4 * (bo_ij->BO_pi + bo_ij->BO_pi2), 
//	(1.0 - dfvl*workspace->dDelta_lp[j]),
//	CEover4 * (1.0 - dfvl*workspace->dDelta_lp[j]) * 
//	(bo_ij->BO_pi + bo_ij->BO_pi2) );
	  
 //      fprintf( out_control->eun, "%6d%23.15e\n",
//	 workspace->orig_id[j]+1, CEunder3 ); 

 //     fprintf( out_control->eun, "%6d%23.15e%23.15e%23.15e%23.15e\n",
//	workspace->orig_id[j]+1,
//	CEunder4,
//	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]),
//	CEunder4*
//	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]),
//	CEunder4*(1.0 - dfvl*workspace->dDelta_lp[j])*
//	(bo_ij->BO_pi + bo_ij->BO_pi2) );
#endif

#ifdef TEST_FORCES
		//TODO
//      Add_dBO( system, lists, i, pj, CEover1 * twbp->p_ovun1 * twbp->De_s, 
//	       workspace->f_ov ); // OvCoor - 1st term

 //     Add_dDelta( system, lists, j,
//		  CEover4 * (1.0 - dfvl*workspace->dDelta_lp[j]) * 
//		  (bo_ij->BO_pi+bo_ij->BO_pi2), workspace->f_ov );//OvCoor3a

 //     Add_dBOpinpi2( system, lists, i, pj, 
//		     CEover4 * (workspace->Delta[j] - 
//				dfvl * workspace->Delta_lp_temp[j]),
//		     CEover4 * (workspace->Delta[j] - 
//				dfvl * workspace->Delta_lp_temp[j]),
//		     workspace->f_ov, workspace->f_ov ); // OvCoor - 3b

 //     Add_dDelta( system, lists, j,
//		  CEunder4 * (1.0 - dfvl*workspace->dDelta_lp[j]) * 
//		  (bo_ij->BO_pi + bo_ij->BO_pi2),
//		  workspace->f_un ); // UnCoor - 2a

 //     Add_dBOpinpi2( system, lists, i, pj, 
//		     CEunder4 * (workspace->Delta[j] - 
//				 dfvl * workspace->Delta_lp_temp[j]),
//		     CEunder4 * (workspace->Delta[j] - 
//				 dfvl * workspace->Delta_lp_temp[j]),
//		     workspace->f_un, workspace->f_un ); // UnCoor - 2b
#endif
    }

#ifdef TEST_ENERGY      

	//TODO
	//replace the code here... you deleted for compiling
	//TODO
#endif
  //} .. end of for loop


}



////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//TEST ONLY CODE -- See if this is working.
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////

//CUDA Functions
GLOBAL void test_LonePair_OverUnder_Coordination_Energy ( reax_atom *atoms, global_parameters g_params, 
																	single_body_parameters *sbp, two_body_parameters *tbp, 
																	static_storage p_workspace, simulation_data *data,
																	list p_bonds, int N, int num_atom_types, 
																	real *E_Lp, real *E_Ov, real *E_Un)
{
  int i, j, pj, type_i, type_j;
  real Delta_lpcorr, dfvl;
  real e_lp, expvd2, inv_expvd2, dElp, CElp, DlpVi;
  real e_lph, Di, vov3, deahu2dbo, deahu2dsbo;
  real e_ov, CEover1, CEover2, CEover3, CEover4;
  real exp_ovun1, exp_ovun2, sum_ovun1, sum_ovun2;
  real exp_ovun2n, exp_ovun6, exp_ovun8;
  real inv_exp_ovun1, inv_exp_ovun2, inv_exp_ovun2n, inv_exp_ovun8;
  real e_un, CEunder1, CEunder2, CEunder3, CEunder4;
  real p_lp1, p_lp2, p_lp3;
  real p_ovun2, p_ovun3, p_ovun4, p_ovun5, p_ovun6, p_ovun7, p_ovun8;

  single_body_parameters *sbp_i, *sbp_j;
  two_body_parameters *twbp;
  bond_data *pbond;
  bond_order_data *bo_ij; 
  list *bonds = &p_bonds;
  static_storage *workspace = &p_workspace;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  /* Initialize parameters */
  p_lp1 = g_params.l[15];
  p_lp3 = g_params.l[5];
  p_ovun3 = g_params.l[32];
  p_ovun4 = g_params.l[31];
  p_ovun6 = g_params.l[6];
  p_ovun7 = g_params.l[8];
  p_ovun8 = g_params.l[9];

/*
  if (i < N) {
    // set the parameter pointer 
    type_i = atoms[i].type;
    sbp_i = &(sbp[ type_i ]);

    // lone-pair Energy 
    p_lp2 = sbp_i->p_lp2;      
    expvd2 = EXP( -75 * workspace->Delta_lp[i] );
    inv_expvd2 = 1. / (1. + expvd2 );
      
    // calculate the energy 
    e_lp = p_lp2 * workspace->Delta_lp[i] * inv_expvd2;
    //atomicAdd (&data->E_Lp, e_lp );
	 E_Lp [ i ] = e_lp;

    dElp = p_lp2 * inv_expvd2 + 
      75 * p_lp2 * workspace->Delta_lp[i] * expvd2 * SQR(inv_expvd2);
    CElp = dElp * workspace->dDelta_lp[i];

    workspace->CdDelta[i] += CElp;      // lp - 1st term

    // correction for C2 
    if( g_params.l[5] > 0.001 && 
	!cuda_strcmp( sbp[type_i].name, "C" , 15) )
      for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
	if( i < bonds->select.bond_list[pj].nbr ) {
	  j = bonds->select.bond_list[pj].nbr;
	  type_j = atoms[j].type;
	      
	  if( !cuda_strcmp( sbp[type_j].name, "C", 15 ) ) {
	    twbp = &( tbp[ index_tbp (type_i,type_j,num_atom_types) ]);
	    bo_ij = &( bonds->select.bond_list[pj].bo_data );
	    Di = workspace->Delta[i];
	    vov3 = bo_ij->BO - Di - 0.040*POW(Di, 4.);

	    if( vov3 > 3. ) {

	     	e_lph = p_lp3 * SQR(vov3-3.0);
			E_Lp [i] += e_lph;
	     	//atomicAdd (&data->E_Lp, e_lph );
	      //estrain(i) += e_lph;

	      deahu2dbo = 2.*p_lp3*(vov3 - 3.);
	      deahu2dsbo = 2.*p_lp3*(vov3 - 3.)*(-1. - 0.16*POW(Di, 3.));

	      bo_ij->Cdbo += deahu2dbo;

	      workspace->CdDelta[i] += deahu2dsbo;
	    }
	  }
	}
  } // end of if statement for the all the threads

  __syncthreads ();

	 if (i >= N) return;

*/

    type_i = atoms[i].type;
    sbp_i = &(sbp[ type_i ]);

    // over-coordination energy 
    if( sbp_i->mass > 21.0 ) 
      dfvl = 0.0;
    else dfvl = 1.0; // only for 1st-row elements
      
    p_ovun2 = sbp_i->p_ovun2;
    sum_ovun1 = 0;
    sum_ovun2 = 0;

    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
      j = bonds->select.bond_list[pj].nbr;
      type_j = atoms[j].type;	  
      bo_ij = &(bonds->select.bond_list[pj].bo_data);
      sbp_j = &(sbp[ type_j ]);
      twbp = &(tbp[ index_tbp (type_i,type_j,num_atom_types) ]);

      sum_ovun1 += twbp->p_ovun1 * twbp->De_s * bo_ij->BO;
      sum_ovun2 += (workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j])*
	( bo_ij->BO_pi + bo_ij->BO_pi2 );
    }

			
    exp_ovun1 = p_ovun3 * EXP( p_ovun4 * sum_ovun2 );
    inv_exp_ovun1 = 1.0 / (1 + exp_ovun1);
    Delta_lpcorr  = workspace->Delta[i] - 
      (dfvl*workspace->Delta_lp_temp[i]) * inv_exp_ovun1;
	
    exp_ovun2 = EXP( p_ovun2 * Delta_lpcorr );
    inv_exp_ovun2 = 1.0 / (1.0 + exp_ovun2);
	
    DlpVi = 1.0 / (Delta_lpcorr + sbp_i->valency + 1e-8 );
    CEover1 = Delta_lpcorr * DlpVi * inv_exp_ovun2;
			
    e_ov = sum_ovun1 * CEover1;
	 E_Ov [ i ] = e_ov;
    //atomicAdd ( &data->E_Ov, e_ov );

    CEover2 = sum_ovun1 * DlpVi * inv_exp_ovun2 *
      ( 1.0 - Delta_lpcorr*( DlpVi + p_ovun2 * exp_ovun2 * inv_exp_ovun2 ) );

    CEover3 = CEover2 * (1.0 - dfvl*workspace->dDelta_lp[i]*inv_exp_ovun1 );

    CEover4 = CEover2 * (dfvl*workspace->Delta_lp_temp[i]) * 
      p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1);

      
    // under-coordination potential 
    p_ovun2 = sbp_i->p_ovun2;
    p_ovun5 = sbp_i->p_ovun5;
	
    exp_ovun2n = 1.0 / exp_ovun2;
    exp_ovun6 = EXP( p_ovun6 * Delta_lpcorr );
    exp_ovun8 = p_ovun7 * EXP(p_ovun8 * sum_ovun2);
    inv_exp_ovun2n = 1.0 / (1.0 + exp_ovun2n);
    inv_exp_ovun8 = 1.0 / (1.0 + exp_ovun8);

    e_un = -p_ovun5 * (1.0 - exp_ovun6) * inv_exp_ovun2n * inv_exp_ovun8;
	 E_Un [i] = e_un;
    //atomicAdd ( &data->E_Un, e_un );
			
    CEunder1 = inv_exp_ovun2n * ( p_ovun5*p_ovun6*exp_ovun6*inv_exp_ovun8 +
				  p_ovun2 * e_un * exp_ovun2n);
    CEunder2 = -e_un * p_ovun8 * exp_ovun8 * inv_exp_ovun8;
    CEunder3 = CEunder1 * (1.0 - dfvl*workspace->dDelta_lp[i]*inv_exp_ovun1);
    CEunder4 = CEunder1 * (dfvl*workspace->Delta_lp_temp[i]) * 
      p_ovun4 * exp_ovun1 * SQR(inv_exp_ovun1) + CEunder2;

    // forces 
    workspace->CdDelta[i] += CEover3;   // OvCoor - 2nd term
    workspace->CdDelta[i] += CEunder3;  // UnCoor - 1st term

    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ){
      pbond = &(bonds->select.bond_list[pj]);
      j = pbond->nbr;
      type_j = atoms[j].type;
      bo_ij = &(pbond->bo_data);
      twbp  = &(tbp[ index_tbp (type_i,type_j,num_atom_types) ]);

      bo_ij->Cdbo += CEover1 * twbp->p_ovun1 * twbp->De_s; // OvCoor - 1st  

      //workspace->CdDelta[j] += CEover4*(1.0 - dfvl*workspace->dDelta_lp[j])* (bo_ij->BO_pi + bo_ij->BO_pi2); // OvCoor - 3a
		pbond->scratch += CEover4*(1.0 - dfvl*workspace->dDelta_lp[j])* (bo_ij->BO_pi + bo_ij->BO_pi2); // OvCoor - 3a

      bo_ij->Cdbopi += CEover4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//OvCoor-3b
      bo_ij->Cdbopi2 += CEover4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//OvCoor-3b

      //workspace->CdDelta[j] += CEunder4*(1.0-dfvl*workspace->dDelta_lp[j]) * (bo_ij->BO_pi + bo_ij->BO_pi2) ;   // UnCoor - 2a
      pbond->scratch += CEunder4*(1.0-dfvl*workspace->dDelta_lp[j]) * (bo_ij->BO_pi + bo_ij->BO_pi2) ;   // UnCoor - 2a

      bo_ij->Cdbopi += CEunder4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//UnCoor-2b
      bo_ij->Cdbopi2 += CEunder4 * 
	(workspace->Delta[j] - dfvl*workspace->Delta_lp_temp[j]);//UnCoor-2b

    }
}

///////////////////////////////////////////////////////////
GLOBAL void test_LonePair_OverUnder_Coordination_Energy_LP ( reax_atom *atoms, global_parameters g_params, 
																	single_body_parameters *sbp, two_body_parameters *tbp, 
																	static_storage p_workspace, simulation_data *data,
																	list p_bonds, int N, int num_atom_types, 
																	real *E_Lp, real *E_Ov, real *E_Un)
{
  int i, j, pj, type_i, type_j;
  real Delta_lpcorr, dfvl;
  real e_lp, expvd2, inv_expvd2, dElp, CElp, DlpVi;
  real e_lph, Di, vov3, deahu2dbo, deahu2dsbo;
  real e_ov, CEover1, CEover2, CEover3, CEover4;
  real exp_ovun1, exp_ovun2, sum_ovun1, sum_ovun2;
  real exp_ovun2n, exp_ovun6, exp_ovun8;
  real inv_exp_ovun1, inv_exp_ovun2, inv_exp_ovun2n, inv_exp_ovun8;
  real e_un, CEunder1, CEunder2, CEunder3, CEunder4;
  real p_lp1, p_lp2, p_lp3;
  real p_ovun2, p_ovun3, p_ovun4, p_ovun5, p_ovun6, p_ovun7, p_ovun8;

  single_body_parameters *sbp_i, *sbp_j;
  two_body_parameters *twbp;
  bond_data *pbond;
  bond_order_data *bo_ij; 
  list *bonds = &p_bonds;
  static_storage *workspace = &p_workspace;

  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  /* Initialize parameters */
  p_lp1 = g_params.l[15];
  p_lp3 = g_params.l[5];
  p_ovun3 = g_params.l[32];
  p_ovun4 = g_params.l[31];
  p_ovun6 = g_params.l[6];
  p_ovun7 = g_params.l[8];
  p_ovun8 = g_params.l[9];

    // set the parameter pointer 
    type_i = atoms[i].type;
    sbp_i = &(sbp[ type_i ]);

    // lone-pair Energy 
    p_lp2 = sbp_i->p_lp2;      
    expvd2 = EXP( -75 * workspace->Delta_lp[i] );
    inv_expvd2 = 1. / (1. + expvd2 );
      
    // calculate the energy 
    e_lp = p_lp2 * workspace->Delta_lp[i] * inv_expvd2;
    //atomicAdd (&data->E_Lp, e_lp );
	 E_Lp [ i ] = e_lp;

    dElp = p_lp2 * inv_expvd2 + 
      75 * p_lp2 * workspace->Delta_lp[i] * expvd2 * SQR(inv_expvd2);
    CElp = dElp * workspace->dDelta_lp[i];

    workspace->CdDelta[i] += CElp;      // lp - 1st term

    // correction for C2 
    if( g_params.l[5] > 0.001 && 
	!cuda_strcmp( sbp[type_i].name, "C" , 15) )
      for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
	if( i < bonds->select.bond_list[pj].nbr ) {
	  j = bonds->select.bond_list[pj].nbr;
	  type_j = atoms[j].type;
	      
	  if( !cuda_strcmp( sbp[type_j].name, "C", 15 ) ) {
	    twbp = &( tbp[ index_tbp (type_i,type_j,num_atom_types) ]);
	    bo_ij = &( bonds->select.bond_list[pj].bo_data );
	    Di = workspace->Delta[i];
	    vov3 = bo_ij->BO - Di - 0.040*POW(Di, 4.);

	    if( vov3 > 3. ) {

	     	e_lph = p_lp3 * SQR(vov3-3.0);
			E_Lp [i] += e_lph;
	     	//atomicAdd (&data->E_Lp, e_lph );
	      //estrain(i) += e_lph;

	      deahu2dbo = 2.*p_lp3*(vov3 - 3.);
	      deahu2dsbo = 2.*p_lp3*(vov3 - 3.)*(-1. - 0.16*POW(Di, 3.));

	      bo_ij->Cdbo += deahu2dbo;

	      workspace->CdDelta[i] += deahu2dsbo;
	    }
	  }
	}
}
///////////////////////////////////////////////////////////

GLOBAL void test_LonePair_Postprocess ( reax_atom *atoms, global_parameters g_params, 
																	single_body_parameters *sbp, two_body_parameters *tbp, 
																	static_storage p_workspace, simulation_data *data,
																	list p_bonds, int N, int num_atom_types )
{
  int i, j, pj, type_i, type_j;

  single_body_parameters *sbp_i, *sbp_j;
  two_body_parameters *twbp;
  bond_data *pbond, *sbond;
  bond_data *dbond_index_bond, *sym_index_bond;
  bond_order_data *bo_ij; 
  list *bonds = &p_bonds;
  static_storage *workspace = &p_workspace;

  i = blockIdx.x * blockDim.x + threadIdx.x;

  if ( i >= N) return;

  for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ){

		/*
  		pbond = &(bonds->select.bond_list[pj]);
		dbond_index_bond = &( bonds->select.bond_list[ pbond->dbond_index ] );
		workspace->CdDelta [i] += dbond_index_bond->scratch;
		*/

		sbond = &(bonds->select.bond_list [pj]);
		sym_index_bond = &( bonds->select.bond_list[ sbond->sym_index ]);
		workspace->CdDelta [i] += sym_index_bond->scratch;
  }

}
