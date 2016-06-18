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

#ifndef __TWO_BODY_INTERACTIONS_H_
#define __TWO_BODY_INTERACTIONS_H_

#include <mytypes.h>
#include "index_utils.h"

void Bond_Energy( reax_system*, control_params*, simulation_data*, 
		  static_storage*, list**, output_controls* );
void vdW_Coulomb_Energy( reax_system*, control_params*, simulation_data*,
			 static_storage*, list**, output_controls* );
void LR_vdW_Coulomb( reax_system*, control_params*, int, int, real, LR_data* );
void Tabulated_vdW_Coulomb_Energy( reax_system*, control_params*, simulation_data*,
				   static_storage*, list**, output_controls* );

//CUDA functions
GLOBAL void Cuda_Bond_Energy ( reax_atom *, global_parameters , single_body_parameters *, two_body_parameters *, 
								simulation_data *, static_storage , list , int , int, real *);

GLOBAL void Cuda_vdW_Coulomb_Energy( reax_atom *,   
                                       two_body_parameters *,
                                       global_parameters , 
                                       control_params *, 
                                       simulation_data *,  
                                       list , 
                                       real *, real *, rvec *, 
                                       int , int );

GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy ( reax_atom *, control_params *, simulation_data *,
																list , real *, real *, rvec *, 
																LR_lookup_table *, int , int , int ) ;
GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy_1 ( reax_atom *, control_params *, simulation_data *,
																list , real *, real *, rvec *, 
																LR_lookup_table *, int , int , int ) ;
GLOBAL void Cuda_Tabulated_vdW_Coulomb_Energy_2 ( reax_atom *, control_params *, simulation_data *,
																list , real *, real *, rvec *, 
																LR_lookup_table *, int , int , int ) ;

HOST_DEVICE void LR_vdW_Coulomb( global_parameters , two_body_parameters *, 
                            control_params *, int , int , real , LR_data * , int);

HOST_DEVICE inline void LR_vdW_Coulomb(    global_parameters g_params, two_body_parameters *tbp, 
                              control_params *control, 
                              int i, int j, real r_ij, LR_data *lr, int num_atom_types )
{
  real p_vdW1 = g_params.l[28];
  real p_vdW1i = 1.0 / p_vdW1;
  real powr_vdW1, powgi_vdW1;
  real tmp, fn13, exp1, exp2;
  real Tap, dTap, dfn13;
  real dr3gamij_1, dr3gamij_3;
  real e_core, de_core;
  two_body_parameters *twbp;

  twbp = &(tbp[ index_tbp (i,j, num_atom_types) ]); 
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
    
  dTap = 7*control->Tap7 * r_ij + 6*control->Tap6;
  dTap = dTap * r_ij + 5*control->Tap5;
  dTap = dTap * r_ij + 4*control->Tap4;
  dTap = dTap * r_ij + 3*control->Tap3;
  dTap = dTap * r_ij + 2*control->Tap2;
  dTap += control->Tap1/r_ij;


  /* vdWaals calculations */
  powr_vdW1 = POW(r_ij, p_vdW1);
  powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1);

  fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
  exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
  exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

  lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);
  /* fprintf(stderr,"vdW: Tap:%f, r: %f, f13:%f, D:%f, Energy:%f,\
     Gamma_w:%f, p_vdw: %f, alpha: %f, r_vdw: %f, %lf %lf\n",
     Tap, r_ij, fn13, twbp->D, Tap * twbp->D * (exp1 - 2.0 * exp2), 
     powgi_vdW1, p_vdW1, twbp->alpha, twbp->r_vdW, exp1, exp2); */

  dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) * POW(r_ij, p_vdW1 - 2.0);

  lr->CEvd = dTap * twbp->D * (exp1 - 2 * exp2) -
    Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;

  /*vdWaals Calculations*/
  if(g_params.vdw_type==1 || g_params.vdw_type==3)
    { // shielding
      powr_vdW1 = POW(r_ij, p_vdW1);
      powgi_vdW1 = POW( 1.0 / twbp->gamma_w, p_vdW1);

      fn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i );
      exp1 = EXP( twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );
      exp2 = EXP( 0.5 * twbp->alpha * (1.0 - fn13 / twbp->r_vdW) );

      lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);

      dfn13 = POW( powr_vdW1 + powgi_vdW1, p_vdW1i - 1.0) *
   POW(r_ij, p_vdW1 - 2.0);

      lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
   Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2) * dfn13;
    }
  else{ // no shielding
    exp1 = EXP( twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );
    exp2 = EXP( 0.5 * twbp->alpha * (1.0 - r_ij / twbp->r_vdW) );

    lr->e_vdW = Tap * twbp->D * (exp1 - 2.0 * exp2);

    lr->CEvd = dTap * twbp->D * (exp1 - 2.0 * exp2) -
      Tap * twbp->D * (twbp->alpha / twbp->r_vdW) * (exp1 - exp2);
  }

  if(g_params.vdw_type==2 || g_params.vdw_type==3)
    { // innner wall
      e_core = twbp->ecore * EXP(twbp->acore * (1.0-(r_ij/twbp->rcore)));
      lr->e_vdW += Tap * e_core;

      de_core = -(twbp->acore/twbp->rcore) * e_core;
      lr->CEvd += dTap * e_core + Tap * de_core;
    }

  /* Coulomb calculations */
  dr3gamij_1 = ( r_ij * r_ij * r_ij + twbp->gamma );
  dr3gamij_3 = POW( dr3gamij_1 , 0.33333333333333 );

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

#endif
