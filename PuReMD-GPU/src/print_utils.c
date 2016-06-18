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

#include "print_utils.h"
#include "list.h"
#include "pdb_tools.h"
#include "system_props.h"
#include "vector.h"


#ifdef TEST_FORCES
void Dummy_Printer( reax_system *system, control_params *control, 
		    simulation_data *data, static_storage *workspace, 
		    list **lists, output_controls *out_control )
{
}


void Print_Bond_Orders( reax_system *system, control_params *control, 
			simulation_data *data, static_storage *workspace, 
			list **lists, output_controls *out_control )
{
  int  i, pj, pk;
  bond_order_data *bo_ij;
  list *bonds = (*lists) + BONDS;
  list *dBOs  = (*lists) + DBO;
  dbond_data *dbo_k;


  /* bond orders */
  fprintf( out_control->fbo, "%6s%6s%12s%12s%12s%12s%12s\n", 
	   "atom1", "atom2", "r_ij", "total_bo", "bo_s", "bo_p", "bo_pp" );
  
  for( i = 0; i < system->N; ++i )
    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
      bo_ij = &(bonds->select.bond_list[pj].bo_data);	
      fprintf( out_control->fbo, "%6d%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n", 
	       //workspace->orig_id[i], 
	       //workspace->orig_id[bonds->select.bond_list[pj].nbr],
	       i+1,
	       bonds->select.bond_list[pj].nbr+1,
	       bonds->select.bond_list[pj].d,
	       bo_ij->BO, bo_ij->BO_s, bo_ij->BO_pi, bo_ij->BO_pi2 );
    }
  
  /* derivatives of bond orders */
  /* fprintf( out_control->fbo, "%6s%6s%10s%10s%10s%10s\n", 
     "atom1", "atom2", "total_bo", "bo_s", "bo_p", "bo_pp"\n ); */
  
  for( i = 0; i < system->N; ++i )
    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
      /*fprintf( out_control->fdbo, "%6d %6d\tstart: %6d\tend: %6d\n", 
	workspace->orig_id[i], 
	workspace->orig_id[bonds->select.bond_list[pj].nbr],
	Start_Index( pj, dBOs ), End_Index( pj, dBOs ) );*/
      
      for( pk = Start_Index(pj,dBOs); pk < End_Index(pj,dBOs); ++pk ) {
	dbo_k = &(dBOs->select.dbo_list[pk]);
	
	//if( !rvec_isZero( dbo_k->dBO ) )
	fprintf( out_control->fdbo, "%6d%6d%6d%23.15e%23.15e%23.15e\n",
		 workspace->orig_id[i], 
		 workspace->orig_id[bonds->select.bond_list[pj].nbr], 
		 workspace->orig_id[dbo_k->wrt],
		 dbo_k->dBO[0], dbo_k->dBO[1], dbo_k->dBO[2] );
	
	fprintf( out_control->fdbo, "%6d%6d%6d%23.15e%23.15e%23.15e\n",
		 workspace->orig_id[i], 
		 workspace->orig_id[bonds->select.bond_list[pj].nbr], 
		 workspace->orig_id[dbo_k->wrt],
		 dbo_k->dBOpi[0], dbo_k->dBOpi[1], dbo_k->dBOpi[2] );
	
	fprintf( out_control->fdbo, "%6d%6d%6d%23.15e%23.15e%23.15e\n",
		 workspace->orig_id[i], 
		 workspace->orig_id[bonds->select.bond_list[pj].nbr], 
		 workspace->orig_id[dbo_k->wrt],
		 dbo_k->dBOpi2[0], dbo_k->dBOpi2[1], dbo_k->dBOpi2[2] ); 
      }
    }
  
  fflush(out_control->fdbo);
}



void Print_Bond_Forces( reax_system *system, control_params *control, 
			simulation_data *data, static_storage *workspace, 
			list **lists, output_controls *out_control )
{
  int i;

  fprintf( out_control->fbond, "%d\n", data->step );
  fprintf( out_control->fbond, "%6s\t%s\n", "atom", "fbond" ); 

  for( i = 0; i < system->N; ++i )
    fprintf(out_control->fbond, "%6d %23.15e%23.15e%23.15e\n",
	    workspace->orig_id[i], 
	    workspace->f_be[i][0],workspace->f_be[i][1],workspace->f_be[i][2]);
}



void Print_LonePair_Forces( reax_system *system, control_params *control, 
			    simulation_data *data, static_storage *workspace, 
			    list **lists, output_controls *out_control )
{
  int i;

  fprintf( out_control->flp, "%d\n", data->step );
  fprintf( out_control->flp, "%6s\t%s\n", "atom","f_lonepair" );

  for( i = 0; i < system->N; ++i )
    fprintf(out_control->flp, "%6d %23.15e%23.15e%23.15e\n",
	    workspace->orig_id[i], 
	    workspace->f_lp[i][0],workspace->f_lp[i][1],workspace->f_lp[i][2]);

  fflush(out_control->flp);
}



void Print_OverUnderCoor_Forces( reax_system *system, control_params *control, 
				 simulation_data *data, 
				 static_storage *workspace, list **lists, 
				 output_controls *out_control )
{
  int i;

  fprintf( out_control->fatom, "%d\n", data->step );
  fprintf( out_control->fatom, "%6s\t%-38s%-38s%-38s\n", 
	   "atom","f_atom", "f_over", "f_under" );

  for( i = 0; i < system->N; ++i ){
    if( rvec_isZero( workspace->f_un[i] ) )
      fprintf( out_control->fatom, 
	       "%6d %23.15e%23.15e%23.15e 0 0 0\n",
	       workspace->orig_id[i], workspace->f_ov[i][0], 
	       workspace->f_ov[i][1], workspace->f_ov[i][2] );
    else
      fprintf( out_control->fatom, 
	       "%6d %23.15e%23.15e%23.15e %23.15e%23.15e%23.15e"\
	       "%23.15e%23.15e%23.15e\n",
	       workspace->orig_id[i],
	       workspace->f_un[i][0] + workspace->f_ov[i][0],
	       workspace->f_un[i][1] + workspace->f_ov[i][1],
	       workspace->f_un[i][2] + workspace->f_ov[i][2],
	       workspace->f_ov[i][0], workspace->f_ov[i][1], 
	       workspace->f_ov[i][2], 
	       workspace->f_un[i][0], workspace->f_un[i][1],
	       workspace->f_un[i][2] );
  }

  fflush(out_control->fatom);
}



void Print_Three_Body_Forces( reax_system *system, control_params *control, 
			      simulation_data *data, static_storage *workspace, 
			      list **lists, output_controls *out_control )
{
  int j;
  
  fprintf( out_control->f3body, "%d\n", data->step );
  fprintf( out_control->f3body, "%6s%-37s%-37s%-37s%-38s\n", 
	   "atom", "3-body total", "f_ang", "f_pen", "f_coa" );
  
  for( j = 0; j < system->N; ++j ){
    if( rvec_isZero(workspace->f_pen[j]) && rvec_isZero(workspace->f_coa[j]) )
      fprintf( out_control->f3body, "%6d %23.15e%23.15e%23.15e  0 0 0  0 0 0\n",
	       workspace->orig_id[j], workspace->f_ang[j][0],
	       workspace->f_ang[j][1], workspace->f_ang[j][2] );
    else if( rvec_isZero(workspace->f_coa[j]) )
      fprintf( out_control->f3body, 
	       "%6d %23.15e%23.15e%23.15e %23.15e%23.15e%23.15e "\
	       "%23.15e%23.15e%23.15e\n",
	       workspace->orig_id[j], 
	       workspace->f_ang[j][0] + workspace->f_pen[j][0],
	       workspace->f_ang[j][1] + workspace->f_pen[j][1],
	       workspace->f_ang[j][2] + workspace->f_pen[j][2],
	       workspace->f_ang[j][0], workspace->f_ang[j][1],
	       workspace->f_ang[j][2],
	       workspace->f_pen[j][0], workspace->f_pen[j][1],
	       workspace->f_pen[j][2] );
    else{
      fprintf( out_control->f3body, "%6d %23.15e%23.15e%23.15e ",
	       workspace->orig_id[j], 
	       workspace->f_ang[j][0] + workspace->f_pen[j][0] + 
	       workspace->f_coa[j][0],
	       workspace->f_ang[j][1] + workspace->f_pen[j][1] + 
	       workspace->f_coa[j][1],
	       workspace->f_ang[j][2] + workspace->f_pen[j][2] + 
	       workspace->f_coa[j][2] );
    
      fprintf( out_control->f3body, 
	       "%23.15e%23.15e%23.15e %23.15e%23.15e%23.15e "\
	       "%23.15e%23.15e%23.15e\n",
	       workspace->f_ang[j][0], workspace->f_ang[j][1],
	       workspace->f_ang[j][2],
	       workspace->f_pen[j][0], workspace->f_pen[j][1],
	       workspace->f_pen[j][2],
	       workspace->f_coa[j][0], workspace->f_coa[j][1],
	       workspace->f_coa[j][2] );
    }
  }

  fflush(out_control->f3body);
}



void Print_Hydrogen_Bond_Forces( reax_system *system, control_params *control, 
				 simulation_data *data, 
				 static_storage *workspace, list **lists,
				 output_controls *out_control )
{
  int j;
  
  fprintf( out_control->fhb, "%d\n", data->step );
  fprintf( out_control->fhb, "%6s\t%-38s\n", "atom", "f_hb" );
  
  for( j = 0; j < system->N; ++j )
    fprintf(out_control->fhb, "%6d\t[%23.15e%23.15e%23.15e]\n",
	    workspace->orig_id[j],
	    workspace->f_hb[j][0],workspace->f_hb[j][1],workspace->f_hb[j][2]);

  fflush(out_control->fhb);
}

	     

void Print_Four_Body_Forces( reax_system *system, control_params *control, 
			     simulation_data *data, static_storage *workspace, 
			     list **lists, output_controls *out_control )
{
  int j;

  fprintf( out_control->f4body, "%6s\t%-38s%-38s%-38s\n", 
	   "atom", "4-body total", "f_tor", "f_con" );

  for( j = 0; j < system->N; ++j ){
    if( !rvec_isZero( workspace->f_con[j] ) )
      fprintf( out_control->f4body, 
	       "%6d %23.15e%23.15e%23.15e %23.15e%23.15e%23.15e "\
	       "%23.15e%23.15e%23.15e\n",
	       workspace->orig_id[j],	     
	       workspace->f_tor[j][0] + workspace->f_con[j][0], 
	       workspace->f_tor[j][1] + workspace->f_con[j][1], 
	       workspace->f_tor[j][2] + workspace->f_con[j][2],
	       workspace->f_tor[j][0], workspace->f_tor[j][1], 
	       workspace->f_tor[j][2],
	       workspace->f_con[j][0], workspace->f_con[j][1], 
	       workspace->f_con[j][2] );
    else
      fprintf( out_control->f4body, 
	       "%6d %23.15e%23.15e%23.15e  0 0 0\n",
	       workspace->orig_id[j], workspace->f_tor[j][0],
	       workspace->f_tor[j][1], workspace->f_tor[j][2] );
  }

  fflush(out_control->f4body);
}



void Print_vdW_Coulomb_Forces( reax_system *system, control_params *control, 
			       simulation_data *data, static_storage *workspace,
			       list **lists, output_controls *out_control )
{
  int  i;

  fprintf( out_control->fnonb, "%d\n", data->step );
  fprintf( out_control->fnonb, "%6s\t%-38s%-38s%-38s\n", 
	   "atom", "nonbonded total", "f_vdw", "f_ele" );

  for( i = 0; i < system->N; ++i )
    if( !rvec_isZero(workspace->f_ele[i]) )
      fprintf(out_control->fnonb, 
	      "%6d %23.15e%23.15e%23.15e %23.15e%23.15e%23.15e "\
	      "%23.15e%23.15e%23.15e\n", 
	      workspace->orig_id[i], 
	      workspace->f_vdw[i][0] + workspace->f_ele[i][0],
	      workspace->f_vdw[i][1] + workspace->f_ele[i][1], 
	      workspace->f_vdw[i][2] + workspace->f_ele[i][2],
	      workspace->f_vdw[i][0], workspace->f_vdw[i][1], 
	      workspace->f_vdw[i][2],
	      workspace->f_ele[i][0], workspace->f_ele[i][1], 
	      workspace->f_ele[i][2] );
    else
      fprintf(out_control->fnonb, 
	      "%6d %23.15e%23.15e%23.15e  0 0 0\n",
	      workspace->orig_id[i], workspace->f_vdw[i][0], 
	      workspace->f_vdw[i][1], workspace->f_vdw[i][2] );

  fflush(out_control->fnonb);
}


void Compare_Total_Forces( reax_system *system, control_params *control, 
			   simulation_data *data, static_storage *workspace, 
			   list **lists, output_controls *out_control )
{
  int i;

  fprintf( out_control->ftot2, "%d\n", data->step );
  fprintf( out_control->ftot2, "%6s\t%-38s%-38s\n", 
	   "atom", "f_total", "test_force total" );

  for( i = 0; i < system->N; ++i )
    fprintf( out_control->ftot2, 
	     "%6d %23.15e%23.15e%23.15e vs %23.15e%23.15e%23.15e\n", 
	     workspace->orig_id[i], 
	     system->atoms[i].f[0],system->atoms[i].f[1],system->atoms[i].f[2],
	     workspace->f_be[i][0]+
	     workspace->f_lp[i][0]+workspace->f_ov[i][0]+workspace->f_un[i][0]+
	     workspace->f_ang[i][0]+workspace->f_pen[i][0]+
	     workspace->f_coa[i][0]+workspace->f_hb[i][0]+
	     workspace->f_tor[i][0]+workspace->f_con[i][0]+
	     workspace->f_vdw[i][0]+workspace->f_ele[i][0],
	     workspace->f_be[i][1]+ 
	     workspace->f_lp[i][1]+workspace->f_ov[i][1]+workspace->f_un[i][1]+ 
             workspace->f_ang[i][1]+workspace->f_pen[i][1]+
	     workspace->f_coa[i][1]+workspace->f_hb[i][1]+ 
	     workspace->f_tor[i][1]+workspace->f_con[i][1]+
	     workspace->f_vdw[i][1]+workspace->f_ele[i][1], 
	     workspace->f_be[i][2]+
	     workspace->f_lp[i][2]+workspace->f_ov[i][2]+workspace->f_un[i][2]+ 
             workspace->f_ang[i][2]+workspace->f_pen[i][2]+
	     workspace->f_coa[i][2]+workspace->f_hb[i][2]+ 
	     workspace->f_tor[i][2]+workspace->f_con[i][2]+
	     workspace->f_vdw[i][2]+workspace->f_ele[i][2] );

  fflush(out_control->ftot2);
}


void Init_Force_Test_Functions( )
{
  Print_Interactions[0] = Print_Bond_Orders;
  Print_Interactions[1] = Print_Bond_Forces;
  Print_Interactions[2] = Print_LonePair_Forces;
  Print_Interactions[3] = Print_OverUnderCoor_Forces;
  Print_Interactions[4] = Print_Three_Body_Forces;
  Print_Interactions[5] = Print_Four_Body_Forces;
  Print_Interactions[6] = Print_Hydrogen_Bond_Forces;
  Print_Interactions[7] = Dummy_Printer;
  Print_Interactions[8] = Dummy_Printer; 
  Print_Interactions[9] = Dummy_Printer;
}
#endif


char *Get_Element( reax_system *system, int i )
{
  return &( system->reaxprm.sbp[system->atoms[i].type].name[0] );
}


char *Get_Atom_Name( reax_system *system, int i )
{
  return &(system->atoms[i].name[0]);
}


/* near nbrs contain both i-j, j-i nbrhood info */
void Print_Near_Neighbors( reax_system *system, control_params *control, 
			   static_storage *workspace, list **lists )
{
  int   i, j, id_i, id_j;
  char  fname[MAX_STR];
  FILE *fout;
  list *near_nbrs = &((*lists)[NEAR_NBRS]);

  sprintf( fname, "%s.near_nbrs", control->sim_name );
  fout = fopen( fname, "w" );

  fprintf( fout, "hello:!\n" );
  
  for( i = 0; i < system->N; ++i ) {
    id_i = workspace->orig_id[i];
    
    for( j = Start_Index(i,near_nbrs); j < End_Index(i,near_nbrs); ++j ) {
      id_j = workspace->orig_id[near_nbrs->select.near_nbr_list[j].nbr];
      
      // if( id_i < id_j )
      fprintf( fout, "%6d%6d%23.15e%23.15e%23.15e%23.15e\n", 
	       id_i, id_j,
	       near_nbrs->select.near_nbr_list[j].d,
	       near_nbrs->select.near_nbr_list[j].dvec[0],
	       near_nbrs->select.near_nbr_list[j].dvec[1],
	       near_nbrs->select.near_nbr_list[j].dvec[2] );
    }
  }
  
  fclose( fout );
}

/* near nbrs contain both i-j, j-i nbrhood info */
void Print_Near_Neighbors2( reax_system *system, control_params *control, 
			   static_storage *workspace, list **lists )
{
  int   i, j, id_i, id_j;
  char  fname[MAX_STR];
  FILE *fout;
  list *near_nbrs = &((*lists)[NEAR_NBRS]);

  sprintf( fname, "%s.near_nbrs_lgj", control->sim_name );
  fout = fopen( fname, "w" );

  for( i = 0; i < system->N; ++i ) {
    id_i = workspace->orig_id[i];
    fprintf( fout, "%6d:",id_i);
    for( j = Start_Index(i,near_nbrs); j < End_Index(i,near_nbrs); ++j ) {
      id_j = workspace->orig_id[near_nbrs->select.near_nbr_list[j].nbr];
      fprintf( fout, "%6d", id_j);
      
      /* fprintf( fout, "%6d%6d%23.15e%23.15e%23.15e%23.15e\n", 
	 id_i, id_j,
	 near_nbrs->select.near_nbr_list[j].d,
	 near_nbrs->select.near_nbr_list[j].dvec[0],
	 near_nbrs->select.near_nbr_list[j].dvec[1],
	 near_nbrs->select.near_nbr_list[j].dvec[2] ); */	
    }
    fprintf( fout, "\n");
  }
  
  fclose( fout );
}

/* far nbrs contain only i-j nbrhood info, no j-i. */
void Print_Far_Neighbors( reax_system *system, control_params *control, 
			  static_storage *workspace, list **lists )
{
  int   i, j, id_i, id_j;
  char  fname[MAX_STR];
  FILE *fout;
  list *far_nbrs = &((*lists)[FAR_NBRS]);

  sprintf( fname, "%s.far_nbrs", control->sim_name );
  fout = fopen( fname, "w" );

  for( i = 0; i < system->N; ++i ) {
    id_i = workspace->orig_id[i];
    
    for( j = Start_Index(i, far_nbrs); j < End_Index(i, far_nbrs); ++j ) {
      id_j = workspace->orig_id[far_nbrs->select.far_nbr_list[j].nbr];
      
      fprintf( fout, "%6d%6d%23.15e%23.15e%23.15e%23.15e\n",
	       id_i, id_j,
	       far_nbrs->select.far_nbr_list[j].d,
	       far_nbrs->select.far_nbr_list[j].dvec[0],
	       far_nbrs->select.far_nbr_list[j].dvec[1],
	       far_nbrs->select.far_nbr_list[j].dvec[2] );
      
      fprintf( fout, "%6d%6d%23.15e%23.15e%23.15e%23.15e\n",
	       id_j, id_i,
	       far_nbrs->select.far_nbr_list[j].d,
	       -far_nbrs->select.far_nbr_list[j].dvec[0],
	       -far_nbrs->select.far_nbr_list[j].dvec[1],
	       -far_nbrs->select.far_nbr_list[j].dvec[2] );
    }
  }
  
  fclose( fout );
}

int fn_qsort_intcmp( const void *a, const void *b ) 
{ 
  return( *(int *)a - *(int *)b); 
} 

void Print_Far_Neighbors2( reax_system *system, control_params *control, 
			  static_storage *workspace, list **lists )
{
  int   i, j, id_i, id_j;
  char  fname[MAX_STR];
  FILE *fout;
  list *far_nbrs = &((*lists)[FAR_NBRS]);

  sprintf( fname, "%s.far_nbrs_lgj", control->sim_name );
  fout = fopen( fname, "w" );
  int num=0;
  int temp[500];

  for( i = 0; i < system->N; ++i ) {
    id_i = workspace->orig_id[i];
    num=0;
    fprintf( fout, "%6d:",id_i);
    
    for( j = Start_Index(i, far_nbrs); j < End_Index(i, far_nbrs); ++j ) {
      id_j = workspace->orig_id[far_nbrs->select.far_nbr_list[j].nbr];
      temp[num++] = id_j;
    }
    qsort(&temp, num, sizeof(int), fn_qsort_intcmp);
    for(j=0; j < num; j++)
      fprintf(fout, "%6d",temp[j]);
    fprintf( fout, "\n");
  }
  fclose( fout );
}


void Print_Total_Force( reax_system *system, control_params *control, 
			simulation_data *data, static_storage *workspace, 
			list **lists, output_controls *out_control )
{
  int i;
#if !defined(TEST_FORCES)
  char temp[1000];
  sprintf( temp, "%s.ftot", control->sim_name );
  out_control->ftot = fopen( temp, "w" );
#endif

  for( i = 0; i < system->N; ++i )
    fprintf(out_control->ftot, "%6d %23.15e %23.15e %23.15e\n", 
    //fprintf(out_control->ftot, "%6d %19.9e %19.9e %19.9e\n", 
    //fprintf(out_control->ftot, "%3d %12.6f %12.6f %12.6f\n", 
	    workspace->orig_id[i], 
	    system->atoms[i].f[0],system->atoms[i].f[1],system->atoms[i].f[2]);

  fflush(out_control->ftot);
#if !defined(TEST_FORCES)
  fclose(out_control->ftot);
#endif
}


void Output_Results( reax_system *system, control_params *control, 
		     simulation_data *data, static_storage *workspace, 
		     list **lists, output_controls *out_control )
{
  int i, type_i, f_update;
  real q;
  real t_elapsed = 0;;


  /* Compute Polarization Energy */
  data->E_Pol = 0.0;
  for( i = 0; i < system->N; i++ ) {
    q = system->atoms[i].q;
    type_i = system->atoms[i].type;
    
    data->E_Pol += ( system->reaxprm.sbp[ type_i ].chi * q + 
		     (system->reaxprm.sbp[ type_i ].eta / 2.0) * SQR(q) ) * 
      KCALpMOL_to_EV;
    /* fprintf( stderr, "%6d%23.15e%23.15e%23.15e%23.15e\n", 
       i, q, system->reaxprm.sbp[ type_i ].chi, 
       system->reaxprm.sbp[ type_i ].eta, data->E_Pol ); */
  }
  
  data->E_Pot = data->E_BE + data->E_Ov + data->E_Un  + data->E_Lp +
    data->E_Ang + data->E_Pen + data->E_Coa + data->E_HB + 
    data->E_Tor + data->E_Con + 
    data->E_vdW + data->E_Ele + data->E_Pol;
  
  data->E_Tot = data->E_Pot + E_CONV*data->E_Kin;
  

  /* output energies if it is the time */
  if( out_control->energy_update_freq > 0 && 
      data->step % out_control->energy_update_freq == 0 )  {
#if defined(TEST_ENERGY) || defined(DEBUG) || defined(DEBUG_FOCUS)
    fprintf( out_control->out, 
	     "%-6d%24.15e%24.15e%24.15e%13.5f%13.5f%16.5f%13.5f%13.5f\n",
	     data->step, data->E_Tot, data->E_Pot, E_CONV*data->E_Kin, 
	     data->therm.T, control->T, system->box.volume, data->iso_bar.P, 
	     (control->P[0] + control->P[1] + control->P[2])/3 );
    
    fprintf( out_control->pot, 
	     "%-6d%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e\n",
	     data->step,
	     data->E_BE,
	     data->E_Ov + data->E_Un,  data->E_Lp,
	     data->E_Ang+data->E_Pen, data->E_Coa, data->E_HB,
	     data->E_Tor, data->E_Con, 
	     data->E_vdW, data->E_Ele, data->E_Pol );
#else
    fprintf( out_control->out, 
	     "%-6d%16.2f%16.2f%16.2f%11.2f%11.2f%13.2f%13.5f%13.5f\n",
	     data->step, data->E_Tot, data->E_Pot, E_CONV*data->E_Kin, 
	     data->therm.T, control->T, system->box.volume, data->iso_bar.P, 
	     (control->P[0] + control->P[1] + control->P[2])/3 );
      
    fprintf( out_control->pot, 
	     "%-6d%13.2f%13.2f%13.2f%13.2f%13.2f%13.2f%13.2f%13.2f%13.2f%13.2f%13.2f\n",
	     data->step,
	     data->E_BE,
	     data->E_Ov + data->E_Un,  data->E_Lp,
	     data->E_Ang + data->E_Pen, data->E_Coa, data->E_HB,
	     data->E_Tor, data->E_Con, 
	     data->E_vdW, data->E_Ele, data->E_Pol );
#endif
    
#ifdef __PRINT_CPU_RESULTS__
    t_elapsed = Get_Timing_Info( data->timing.total );
    if( data->step == data->prev_steps )
      f_update = 1;
    else f_update = out_control->energy_update_freq;

    fprintf( out_control->log,"%6d%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f\n",
	     data->step, t_elapsed/f_update,
	     data->timing.nbrs/f_update,
	     data->timing.init_forces/f_update,
	     data->timing.bonded/f_update,
	     data->timing.nonb/f_update,
	     data->timing.QEq/f_update,
	     (double)data->timing.matvecs/f_update );
#else
    t_elapsed = Get_Timing_Info( d_timing.total );
    if( data->step == data->prev_steps )
      f_update = 1;
    else f_update = out_control->energy_update_freq;

    fprintf( out_control->log,"%6d%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f%10.2f\n",
	     data->step, t_elapsed/f_update,
	     d_timing.nbrs/f_update,
	     d_timing.init_forces/f_update,
	     d_timing.bonded/f_update,
	     d_timing.nonb/f_update,
	     d_timing.QEq/f_update,
	     (double)d_timing.matvecs/f_update );

#endif

	//fprintf (stderr, " total %10.5f \n", t_elapsed);
	/*
	fprintf (stderr, " step %6d \n", data->step );
	fprintf (stderr, " nbrs %10.2f \n", d_timing.nbrs);
	fprintf (stderr, " init forces %10.5f \n", d_timing.init_forces);
	fprintf (stderr, " bonded %10.2f \n", d_timing.bonded);
	fprintf (stderr, " nonb %10.2f \n", d_timing.nonb);
	fprintf (stderr, " QEq %10.2f \n", d_timing.QEq);
	fprintf (stderr, " matvecs %d \n", d_timing.matvecs);
	*/

    
    data->timing.total = Get_Time( );
    data->timing.nbrs = 0;
    data->timing.init_forces = 0;
    data->timing.bonded = 0;
    data->timing.nonb = 0;
    data->timing.QEq = 0;
    data->timing.matvecs = 0;

    d_timing.total = Get_Time( );
    d_timing.nbrs = 0;
    d_timing.init_forces = 0;
    d_timing.bonded = 0;
    d_timing.nonb = 0;
    d_timing.QEq = 0;
    d_timing.matvecs = 0;
    
    fflush( out_control->out );
    fflush( out_control->pot );
    fflush( out_control->log );

    /* output pressure */
    if( control->ensemble == NPT || control->ensemble == iNPT || 
	control->ensemble == sNPT ) {
      fprintf( out_control->prs, "%-8d%13.6f%13.6f%13.6f",
	       data->step, 
	       data->int_press[0], data->int_press[1], data->int_press[2] );
      
      /* external pressure is calculated together with forces */ 
      fprintf( out_control->prs, "%13.6f%13.6f%13.6f",
	       data->ext_press[0], data->ext_press[1], data->ext_press[2] );
      
      fprintf( out_control->prs, "%13.6f\n", data->kin_press );
      
      fprintf( out_control->prs, 
	       "%-8d%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f%13.6f\n",
	       data->step, 
	       system->box.box_norms[0], system->box.box_norms[1], 
	       system->box.box_norms[2],
	       data->tot_press[0], data->tot_press[1], data->tot_press[2], 
	       control->P[0],control->P[1],control->P[2],system->box.volume );
      fflush( out_control->prs);
    }
  }
  
  if( out_control->write_steps > 0 && 
      data->step % out_control->write_steps == 0 ){
    // t_start = Get_Time( );
    out_control->append_traj_frame( system, control, data, 
				    workspace, lists, out_control );

    //Write_PDB( system, control, data, workspace, *lists+BONDS, out_control );
    // t_elapsed = Get_Timing_Info( t_start );
    // fprintf(stdout, "append_frame took %.6f seconds\n", t_elapsed );
  }

  // fprintf( stderr, "output_results... done\n" );
}



void Print_Linear_System( reax_system *system, control_params *control, 
			  static_storage *workspace, int step )
{
  int   i, j;
  char  fname[100];
  sparse_matrix *H;
  FILE *out;

  sprintf( fname, "%s.state%d.out", control->sim_name, step );
  out = fopen( fname, "w" );

  for( i = 0; i < system->N; i++ ) 
    fprintf( out, "%6d%2d%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e%24.15e\n", 
	     workspace->orig_id[i],system->atoms[i].type, 
	     system->atoms[i].x[0], system->atoms[i].x[1],
	     system->atoms[i].x[2], 
	     workspace->s[0*system->N + i], workspace->b_s[i], 
	     workspace->t[0*system->N + i], workspace->b_t[i]  );
  fclose( out );
  
  // sprintf( fname, "x2_%d", step );
  // out = fopen( fname, "w" );
  // for( i = 0; i < system->N; i++ ) 
  // fprintf( out, "%g\n", workspace->s_t[i+system->N] );
  // fclose( out );

 
  sprintf( fname, "%s.H%d.out", control->sim_name, step );
  out = fopen( fname, "w" );
  H = &workspace->H;

  for( i = 0; i < system->N; ++i ) {	  
    for( j = H->start[i]; j < H->start[i+1]-1; ++j ) {
      fprintf( out, "%6d%6d %24.15e\n", 
	       workspace->orig_id[i], workspace->orig_id[H->entries[j].j], 
	       H->entries[j].val );
      
      fprintf( out, "%6d%6d %24.15e\n", 
	       workspace->orig_id[H->entries[j].j], workspace->orig_id[i], 
	       H->entries[j].val );
    }
    // the diagonal entry
    fprintf( out, "%6d%6d %24.15e\n", 
	     workspace->orig_id[i], workspace->orig_id[i], H->entries[j].val );
  }
  
  fclose( out );
  
  /*sprintf( fname, "%s.b_s%d", control->sim_name, step );
    out = fopen( fname, "w" );
    for( i = 0; i < system->N; i++ ) 
    fprintf( out, "%12.7f\n", workspace->b_s[i] );
    fclose( out );
    
    sprintf( fname, "%s.b_t%d", control->sim_name, step );
    out = fopen( fname, "w" );
    for( i = 0; i < system->N; i++ ) 
    fprintf( out, "%12.7f\n", workspace->b_t[i] );
    fclose( out );*/
}



void Print_Charges( reax_system *system, control_params *control, 
		    static_storage *workspace, int step )
{
  int   i;
  char fname[100];
  FILE *fout;

  sprintf( fname, "%s.q%d", control->sim_name, step );
  fout = fopen( fname, "w" );

  for( i = 0; i < system->N; ++i )
    fprintf( fout, "%6d%12.7f%12.7f%12.7f\n", 
	     workspace->orig_id[i], 
	     workspace->s[0*system->N + i], workspace->t[0*system->N + i], system->atoms[i].q );

  fclose( fout );
}



void Print_Soln( static_storage *workspace, 
		 real *x, real *b_prm, real *b, int N )
{
  int i;

  fprintf( stdout, "%6s%10s%10s%10s\n", "id", "x", "b_prm", "b" );

  for( i = 0; i < N; ++i )
    fprintf( stdout, "%6d%10.4f%10.4f%10.4f\n", 
	     workspace->orig_id[i], x[i], b_prm[i], b[i] );

  fflush( stdout );
}


void Print_Sparse_Matrix( sparse_matrix *A )
{
  int i, j;

  for( i = 0; i < 10; ++i ) {
    fprintf( stderr, "i:%d  j(val):", i );
    for( j = A->start[i]; j < A->start[i+1]; ++j )
      fprintf( stderr, "%d(%.4f) ", A->entries[j].j, A->entries[j].val );
    fprintf( stderr, "\n" );
  }
}


void Print_Sparse_Matrix2( sparse_matrix *A, char *fname )
{
  int i, j;
  FILE *f = fopen( fname, "w" );

  for( i = 0; i < A->n; ++i )
    for( j = A->start[i]; j < A->start[i+1]; ++j )
      fprintf( f, "%d%d %.15e\n", A->entries[j].j, i, A->entries[j].val );

  fclose(f);
}


void Print_Bonds( reax_system *system, list *bonds, char *fname )
{
  int i, pj;
  bond_data *pbond;
  bond_order_data *bo_ij;
  FILE *f = fopen( fname, "w" );

  for( i = 0; i < system->N; ++i )
    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
      pbond = &(bonds->select.bond_list[pj]);
      bo_ij = &(pbond->bo_data);	
      //fprintf( f, "%6d%6d%23.15e%23.15e%23.15e%23.15e%23.15e\n", 
      //       i+1, pbond->nbr+1, pbond->d,
      //       bo_ij->BO, bo_ij->BO_s, bo_ij->BO_pi, bo_ij->BO_pi2 );
      fprintf( f, "%6d%6d %9.5f %9.5f\n",
	       i+1, pbond->nbr+1, pbond->d, bo_ij->BO );
    }
  fclose(f);
}


void Print_Bond_List2( reax_system *system, list *bonds, char *fname )
{
  int i,j, id_i, id_j, nbr, pj;
  FILE *f = fopen( fname, "w" );
  int temp[500];
  int num=0;

  for( i = 0; i < system->N; ++i ) {
    num=0;
    id_i = i+1; //system->atoms[i].orig_id;
    fprintf( f, "%6d:", id_i);
    for( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj ) {
      nbr = bonds->select.bond_list[pj].nbr;	
      id_j = nbr+1; //system->my_atoms[nbr].orig_id;
      if( id_i < id_j ) 
	temp[num++] = id_j;
    }
    
    qsort(&temp, num, sizeof(int), fn_qsort_intcmp);
    for(j=0; j < num; j++)
      fprintf(f, "%6d", temp[j] );
    fprintf(f, "\n");
  }
}


#ifdef LGJ
Print_XYZ_Serial(reax_system* system, static_storage *workspace)
{
  rvec p;

 char  fname[100];
  FILE *fout;
  sprintf( fname, "READ_PDB.0" );
  fout      = fopen( fname, "w" );
int i;
for(i=0; i < system->N; i++)
  fprintf( fout, "%6d%24.15e%24.15e%24.15e\n",
	   workspace->orig_id[i], 
	   p[0] = system->atoms[i].x[0],
	   p[1] = system->atoms[i].x[1],
	   p[2] = system->atoms[i].x[2]);
 
 
 fclose(fout);
}
#endif
