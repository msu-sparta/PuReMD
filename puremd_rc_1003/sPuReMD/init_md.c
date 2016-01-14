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

#include "init_md.h"
#include "allocate.h"
#include "box.h"
#include "forces.h"
#include "grid.h"
#include "GMRES.h"
#include "integrate.h"
#include "neighbors.h"
#include "list.h"
#include "lookup.h"
#include "print_utils.h"
#include "reset_utils.h"
#include "system_props.h"
#include "traj.h"
#include "vector.h"


void Generate_Initial_Velocities( reax_system *system, real T )
{
  int i;
  real scale, norm;
  
  
  if( T <= 0.1 ) {
    for (i=0; i < system->N; i++)
      rvec_MakeZero( system->atoms[i].v );
#if defined(DEBUG)
    fprintf( stderr, "no random velocities...\n" );
#endif
  }
  else {
    for( i = 0; i < system->N; i++ ) {
      rvec_Random( system->atoms[i].v );
      
      norm = rvec_Norm_Sqr( system->atoms[i].v );
      scale = SQRT( system->reaxprm.sbp[ system->atoms[i].type ].mass * 
		    norm / (3.0 * K_B * T) );
      
      rvec_Scale( system->atoms[i].v, 1.0/scale, system->atoms[i].v );
      
      /*fprintf( stderr, "v = %f %f %f\n", 
	system->atoms[i].v[0],system->atoms[i].v[1],system->atoms[i].v[2]);
	fprintf( stderr, "scale = %f\n", scale );
	fprintf( stderr, "v = %f %f %f\n",
	system->atoms[i].v[0],system->atoms[i].v[1],system->atoms[i].v[2]);*/
    }
  }
}


void Init_System( reax_system *system, control_params *control, 
		  simulation_data *data )
{
  int i;
  rvec dx;

  if( !control->restart )
    Reset_Atoms( system );
      
  Compute_Total_Mass( system, data );
  Compute_Center_of_Mass( system, data, stderr );

  /* reposition atoms */
  // just fit the atoms to the periodic box
  if( control->reposition_atoms == 0 ) {
    rvec_MakeZero( dx );
  }
  // put the center of mass to the center of the box
  else if( control->reposition_atoms == 1 ) {
    rvec_Scale( dx, 0.5, system->box.box_norms );
    rvec_ScaledAdd( dx, -1., data->xcm );
  }
  // put the center of mass to the origin
  else if( control->reposition_atoms == 2 ) {
    rvec_Scale( dx, -1., data->xcm );
  }
  else {
    fprintf( stderr, "UNKNOWN OPTION: reposition_atoms. Terminating...\n" );
    exit( UNKNOWN_OPTION );
  }
  
  for( i = 0; i < system->N; ++i ) {
    Inc_on_T3( system->atoms[i].x, dx, &(system->box) );
    /*fprintf( stderr, "%6d%2d%8.3f%8.3f%8.3f\n", 
      i, system->atoms[i].type, 
      system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2] );*/
  }
  
  /* Initialize velocities so that desired init T can be attained */
  if( !control->restart || (control->restart && control->random_vel) )  
    Generate_Initial_Velocities( system, control->T_init );
  
  Setup_Grid( system );
}


void Init_Simulation_Data( reax_system *system, control_params *control, 
			   simulation_data *data, output_controls *out_control, 
			   evolve_function *Evolve )
{
 
  Reset_Simulation_Data( data );

  if( !control->restart )  
    data->step = data->prev_steps = 0;

  switch( control->ensemble ) {
  case NVE:
    data->N_f = 3 * system->N;
    *Evolve = Velocity_Verlet_NVE;
    break;
    
    
  case NVT:
    data->N_f = 3 * system->N + 1;
    //control->Tau_T = 100 * data->N_f * K_B * control->T_final;
    if( !control->restart || (control->restart && control->random_vel) ) {
      data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin - 
					   data->N_f * K_B * control->T );
      data->therm.v_xi = data->therm.G_xi * control->dt;
      data->therm.v_xi_old = 0;
      data->therm.xi = 0;
#if defined(DEBUG_FOCUS)
      fprintf( stderr, "init_md: G_xi=%f Tau_T=%f E_kin=%f N_f=%f v_xi=%f\n",
	       data->therm.G_xi, control->Tau_T, data->E_Kin, 
	       data->N_f, data->therm.v_xi );
#endif
    }
    
    *Evolve = Velocity_Verlet_Nose_Hoover_NVT_Klein;
    break;
    
    
  case NPT: // Anisotropic NPT
    fprintf( stderr, "THIS OPTION IS NOT YET IMPLEMENTED! TERMINATING...\n" );
    exit( UNKNOWN_OPTION );
    data->N_f = 3 * system->N + 9;
    if( !control->restart ) {
      data->therm.G_xi = control->Tau_T * (2.0 * data->E_Kin - 
					   data->N_f * K_B * control->T );
      data->therm.v_xi = data->therm.G_xi * control->dt;
      data->iso_bar.eps = 0.33333 * log(system->box.volume);
      //data->inv_W = 1. / (data->N_f*K_B*control->T*SQR(control->Tau_P));
      //Compute_Pressure( system, data, workspace );
    }
    *Evolve = Velocity_Verlet_Berendsen_Isotropic_NPT;
    break;
    
    
  case sNPT: // Semi-Isotropic NPT
    data->N_f = 3 * system->N + 4;
    *Evolve = Velocity_Verlet_Berendsen_SemiIsotropic_NPT;
    break;
    
    
  case iNPT: // Isotropic NPT
    data->N_f = 3 * system->N + 2;
    *Evolve = Velocity_Verlet_Berendsen_Isotropic_NPT;
    break;

  case bNVT:
  	 data->N_f = 3 * system->N + 1;
	 *Evolve = Velocity_Verlet_Berendsen_NVT;
	 fprintf (stderr, " Initializing Velocity_Verlet_Berendsen_NVT .... \n");
  break;
    
  default:
    break;
  }
  
  Compute_Kinetic_Energy( system, data );
  
  /* init timing info */
  data->timing.start = Get_Time( );
  data->timing.total = data->timing.start;
  data->timing.nbrs = 0;
  data->timing.init_forces = 0;
  data->timing.bonded = 0;
  data->timing.nonb = 0;
  data->timing.QEq = 0;
  data->timing.matvecs = 0;
}


void Init_Workspace( reax_system *system, control_params *control, 
		     static_storage *workspace )
{  
  int i;

  /* Allocate space for hydrogen bond list */
  workspace->hbond_index = (int *) malloc( system->N * sizeof( int ) );

  /* bond order related storage  */
  workspace->total_bond_order = (real *) malloc( system->N * sizeof( real ) );
  workspace->Deltap           = (real *) malloc( system->N * sizeof( real ) );
  workspace->Deltap_boc       = (real *) malloc( system->N * sizeof( real ) );
  workspace->dDeltap_self     = (rvec *) malloc( system->N * sizeof( rvec ) );

  workspace->Delta	      = (real *) malloc( system->N * sizeof( real ) );
  workspace->Delta_lp	      = (real *) malloc( system->N * sizeof( real ) );
  workspace->Delta_lp_temp    = (real *) malloc( system->N * sizeof( real ) );
  workspace->dDelta_lp	      = (real *) malloc( system->N * sizeof( real ) );
  workspace->dDelta_lp_temp   = (real *) malloc( system->N * sizeof( real ) );
  workspace->Delta_e          = (real *) malloc( system->N * sizeof( real ) );
  workspace->Delta_boc        = (real *) malloc( system->N * sizeof( real ) );
  workspace->nlp	      = (real *) malloc( system->N * sizeof( real ) );
  workspace->nlp_temp	      = (real *) malloc( system->N * sizeof( real ) );
  workspace->Clp	      = (real *) malloc( system->N * sizeof( real ) );
  workspace->CdDelta	      = (real *) malloc( system->N * sizeof( real ) );
  workspace->vlpex	      = (real *) malloc( system->N * sizeof( real ) );

  /* QEq storage */
  workspace->H        = NULL;
  workspace->L        = NULL;
  workspace->U        = NULL;
  workspace->droptol  = (real *) calloc( system->N, sizeof( real ) );
  workspace->w        = (real *) calloc( system->N, sizeof( real ) );
  workspace->Hdia_inv = (real *) calloc( system->N, sizeof( real ) );
  workspace->b        = (real *) calloc( system->N * 2, sizeof( real ) );
  workspace->b_s      = (real *) calloc( system->N, sizeof( real ) );
  workspace->b_t      = (real *) calloc( system->N, sizeof( real ) );
  workspace->b_prc    = (real *) calloc( system->N * 2, sizeof( real ) );
  workspace->b_prm    = (real *) calloc( system->N * 2, sizeof( real ) );
  workspace->s_t      = (real *) calloc( system->N * 2, sizeof( real ) );
  workspace->s        = (real**) calloc( 5, sizeof( real* ) );
  workspace->t        = (real**) calloc( 5, sizeof( real* ) );
  for( i = 0; i < 5; ++i ) {
    workspace->s[i] = (real *) calloc( system->N, sizeof( real ) );
    workspace->t[i] = (real *) calloc( system->N, sizeof( real ) );
  }
  // workspace->s_old    = (real *) calloc( system->N, sizeof( real ) );
  // workspace->t_old    = (real *) calloc( system->N, sizeof( real ) );
  // workspace->s_oldest = (real *) calloc( system->N, sizeof( real ) );
  // workspace->t_oldest = (real *) calloc( system->N, sizeof( real ) );

  for( i = 0; i < system->N; ++i ) {
    workspace->Hdia_inv[i] = 1./system->reaxprm.sbp[system->atoms[i].type].eta;
    workspace->b_s[i] = -system->reaxprm.sbp[ system->atoms[i].type ].chi;
    workspace->b_t[i] = -1.0;
    
    workspace->b[i] = -system->reaxprm.sbp[ system->atoms[i].type ].chi;
    workspace->b[i+system->N] = -1.0;
  }
  
  /* GMRES storage */
  workspace->y  = (real *)  calloc( RESTART+1, sizeof( real ) );
  workspace->z  = (real *)  calloc( RESTART+1, sizeof( real ) );
  workspace->g  = (real *)  calloc( RESTART+1, sizeof( real ) );
  workspace->h  = (real **) calloc( RESTART+1, sizeof( real*) );
  workspace->hs = (real *)  calloc( RESTART+1, sizeof( real ) );
  workspace->hc = (real *)  calloc( RESTART+1, sizeof( real ) );
  workspace->rn = (real **) calloc( RESTART+1, sizeof( real*) );
  workspace->v  = (real **) calloc( RESTART+1, sizeof( real*) );

  for( i = 0; i < RESTART+1; ++i )
    {
      workspace->h[i]  = (real *) calloc( RESTART+1, sizeof( real ) );
      workspace->rn[i] = (real *) calloc( system->N * 2, sizeof( real ) );
      workspace->v[i]  = (real *) calloc( system->N, sizeof( real ) );
    }

  /* CG storage */
  workspace->r = (real *) calloc( system->N, sizeof( real ) );
  workspace->d = (real *) calloc( system->N, sizeof( real ) );
  workspace->q = (real *) calloc( system->N, sizeof( real ) );
  workspace->p = (real *) calloc( system->N, sizeof( real ) );


  /* integrator storage */
  workspace->a = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_old = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->v_const = (rvec *) malloc( system->N * sizeof( rvec ) );


  /* storage for analysis */
  if( control->molec_anal || control->diffusion_coef )
    {
      workspace->mark = (int *) calloc( system->N, sizeof(int) );
      workspace->old_mark = (int *) calloc( system->N, sizeof(int) );
    }
  else 
    workspace->mark = workspace->old_mark = NULL;

  if( control->diffusion_coef )
      workspace->x_old = (rvec *) calloc( system->N, sizeof( rvec ) );
  else workspace->x_old = NULL;
  
  
#ifdef TEST_FORCES
  workspace->dDelta = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_ele = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_vdw = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_bo = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_be = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_lp = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_ov = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_un = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_ang = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_coa = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_pen = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_hb = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_tor = (rvec *) malloc( system->N * sizeof( rvec ) );
  workspace->f_con = (rvec *) malloc( system->N * sizeof( rvec ) );
#endif

  workspace->realloc.num_far = -1;
  workspace->realloc.Htop = -1;
  workspace->realloc.hbonds = -1;
  workspace->realloc.bonds = -1;
  workspace->realloc.num_3body = -1;
  workspace->realloc.gcell_atoms = -1;

  Reset_Workspace( system, workspace );
}


void Init_Lists( reax_system *system, control_params *control, 
		 simulation_data *data, static_storage *workspace, 
		 list **lists, output_controls *out_control )
{
  int i, num_nbrs, num_hbonds, num_bonds, num_3body, Htop;
  int *hb_top, *bond_top;

  num_nbrs = Estimate_NumNeighbors( system, control, workspace, lists );
  if( !Make_List(system->N, num_nbrs, TYP_FAR_NEIGHBOR, (*lists)+FAR_NBRS) ) {
    fprintf(stderr, "Problem in initializing far nbrs list. Terminating!\n");
    exit( INIT_ERR );
  }
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "memory allocated: far_nbrs = %ldMB\n", 
	   num_nbrs * sizeof(far_neighbor_data) / (1024*1024) );
#endif

  Generate_Neighbor_Lists(system,control,data,workspace,lists,out_control);
  Htop = 0;
  hb_top = (int*) calloc( system->N, sizeof(int) );
  bond_top = (int*) calloc( system->N, sizeof(int) );
  num_3body = 0;
  Estimate_Storage_Sizes( system, control, lists, 
			  &Htop, hb_top, bond_top, &num_3body );
  
  Allocate_Matrix( &(workspace->H), system->N, Htop );
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "estimated storage - Htop: %d\n", Htop );
  fprintf( stderr, "memory allocated: H = %ldMB\n", 
	   Htop * sizeof(sparse_matrix_entry) / (1024*1024) );
#endif

  workspace->num_H = 0;
  if( control->hb_cut > 0 ) {
    /* init H indexes */
    for( i = 0; i < system->N; ++i )
      if( system->reaxprm.sbp[ system->atoms[i].type ].p_hbond == 1 ) // H atom
	workspace->hbond_index[i] = workspace->num_H++;
      else workspace->hbond_index[i] = -1;
    
    Allocate_HBond_List( system->N, workspace->num_H, workspace->hbond_index, 
			 hb_top, (*lists)+HBONDS );
    num_hbonds = hb_top[system->N-1];
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "estimated storage - num_hbonds: %d\n", num_hbonds );
    fprintf( stderr, "memory allocated: hbonds = %ldMB\n", 
	     num_hbonds * sizeof(hbond_data) / (1024*1024) );
#endif
  }
  
  /* bonds list */
  Allocate_Bond_List( system->N, bond_top, (*lists)+BONDS );
  num_bonds = bond_top[system->N-1];
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "estimated storage - num_bonds: %d\n", num_bonds );
  fprintf( stderr, "memory allocated: bonds = %ldMB\n", 
	   num_bonds * sizeof(bond_data) / (1024*1024) );
#endif

//fprintf (stderr, " **** sizeof 3 body : %d \n", sizeof (three_body_interaction_data));
//fprintf (stderr, " **** num_3body : %d \n", num_3body);
//fprintf (stderr, " **** num_bonds : %d \n", num_bonds);

  /* 3bodies list */
  if(!Make_List(num_bonds, num_3body, TYP_THREE_BODY, (*lists)+THREE_BODIES)) {
    fprintf( stderr, "Problem in initializing angles list. Terminating!\n" );
    exit( INIT_ERR );
  }
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "estimated storage - num_3body: %d\n", num_3body );
  fprintf( stderr, "memory allocated: 3-body = %ldMB\n", 
	   num_3body * sizeof(three_body_interaction_data) / (1024*1024) );
#endif
#ifdef TEST_FORCES
  if(!Make_List( system->N, num_bonds * 8, TYP_DDELTA, (*lists) + DDELTA )) {
    fprintf( stderr, "Problem in initializing dDelta list. Terminating!\n" );
    exit( INIT_ERR );
  }

  if( !Make_List( num_bonds, num_bonds*MAX_BONDS*3, TYP_DBO, (*lists)+DBO ) ) {
    fprintf( stderr, "Problem in initializing dBO list. Terminating!\n" );
    exit( INIT_ERR );
  }
#endif

  free( hb_top );
  free( bond_top );
}


void Init_Out_Controls(reax_system *system, control_params *control, 
		       static_storage *workspace, output_controls *out_control)
{
  char temp[1000];

  /* Init trajectory file */
  if( out_control->write_steps > 0 ) { 
    strcpy( temp, control->sim_name );
    strcat( temp, ".trj" );
    out_control->trj = fopen( temp, "w" );
    out_control->write_header( system, control, workspace, out_control );
  }

  if( out_control->energy_update_freq > 0 ) {
    /* Init out file */
    strcpy( temp, control->sim_name );
    strcat( temp, ".out" );
    out_control->out = fopen( temp, "w" );
    fprintf( out_control->out, "%-6s%16s%16s%16s%11s%11s%13s%13s%13s\n",
	     "step", "total energy", "poten. energy", "kin. energy", 
	     "temp.", "target", "volume", "press.", "target" );
    fflush( out_control->out );
    
    /* Init potentials file */
    strcpy( temp, control->sim_name );
    strcat( temp, ".pot" );
    out_control->pot = fopen( temp, "w" );
    fprintf( out_control->pot, 
	     "%-6s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s%13s\n",
	     "step", "ebond", "eatom", "elp", "eang", "ecoa", "ehb", 
	     "etor", "econj", "evdw","ecoul", "epol" );
    fflush( out_control->pot );
    
    /* Init log file */
    strcpy( temp, control->sim_name );
    strcat( temp, ".log" );
    out_control->log = fopen( temp, "w" );
    fprintf( out_control->log, "%-6s%10s%10s%10s%10s%10s%10s%10s\n", 
	     "step", "total", "neighbors", "init", "bonded", 
	     "nonbonded", "QEq", "matvec" );
  }

  /* Init pressure file */
  if( control->ensemble == NPT || 
      control->ensemble == iNPT || 
      control->ensemble == sNPT ) {
    strcpy( temp, control->sim_name );
    strcat( temp, ".prs" );
    out_control->prs = fopen( temp, "w" );
    fprintf( out_control->prs, "%-6s%13s%13s%13s%13s%13s%13s%13s%13s\n",
	     "step", "norm_x", "norm_y", "norm_z", 
	     "press_x", "press_y", "press_z", "target_p", "volume" );
    fflush( out_control->prs );
  }
  
  /* Init molecular analysis file */
  if( control->molec_anal ) {
    sprintf( temp, "%s.mol", control->sim_name );
    out_control->mol = fopen( temp, "w" );
    if( control->num_ignored ) {
      sprintf( temp, "%s.ign", control->sim_name );
      out_control->ign = fopen( temp, "w" );
    } 
  }

  /* Init electric dipole moment analysis file */
  if( control->dipole_anal ) {
    strcpy( temp, control->sim_name );
    strcat( temp, ".dpl" );
    out_control->dpl = fopen( temp, "w" );
    fprintf( out_control->dpl, 
	     "Step      Molecule Count  Avg. Dipole Moment Norm\n" );
    fflush( out_control->dpl );
  }

  /* Init diffusion coef analysis file */
  if( control->diffusion_coef ) {
    strcpy( temp, control->sim_name );
    strcat( temp, ".drft" );
    out_control->drft = fopen( temp, "w" );
    fprintf( out_control->drft, "Step     Type Count   Avg Squared Disp\n" );
    fflush( out_control->drft );
  }


#ifdef TEST_ENERGY
  /* open bond energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".ebond" );
  out_control->ebond = fopen( temp, "w" );

  /* open lone-pair energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".elp" );
  out_control->elp = fopen( temp, "w" );

  /* open overcoordination energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".eov" );
  out_control->eov = fopen( temp, "w" );

  /* open undercoordination energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".eun" );
  out_control->eun = fopen( temp, "w" );

  /* open angle energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".eval" );
  out_control->eval = fopen( temp, "w" );

  /* open penalty energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".epen" );
  out_control->epen = fopen( temp, "w" );

  /* open coalition energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".ecoa" );
  out_control->ecoa = fopen( temp, "w" );

  /* open hydrogen bond energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".ehb" );
  out_control->ehb = fopen( temp, "w" );

  /* open torsion energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".etor" );
  out_control->etor = fopen( temp, "w" );

  /* open conjugation energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".econ" );
  out_control->econ = fopen( temp, "w" );

  /* open vdWaals energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".evdw" );
  out_control->evdw = fopen( temp, "w" );

  /* open coulomb energy file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".ecou" );
  out_control->ecou = fopen( temp, "w" );
#endif


#ifdef TEST_FORCES
  /* open bond orders file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".fbo" );
  out_control->fbo = fopen( temp, "w" );

  /* open bond orders derivatives file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".fdbo" );
  out_control->fdbo = fopen( temp, "w" );

  /* open bond forces file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".fbond" );
  out_control->fbond = fopen( temp, "w" );

  /* open lone-pair forces file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".flp" );
  out_control->flp = fopen( temp, "w" );

  /* open overcoordination forces file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".fatom" );
  out_control->fatom = fopen( temp, "w" );

  /* open angle forces file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".f3body" );
  out_control->f3body = fopen( temp, "w" );

  /* open hydrogen bond forces file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".fhb" );
  out_control->fhb = fopen( temp, "w" );

  /* open torsion forces file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".f4body" );
  out_control->f4body = fopen( temp, "w" );

  /* open nonbonded forces file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".fnonb" );
  out_control->fnonb = fopen( temp, "w" );

  /* open total force file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".ftot" );
  out_control->ftot = fopen( temp, "w" );

  /* open coulomb forces file */
  strcpy( temp, control->sim_name );
  strcat( temp, ".ftot2" );
  out_control->ftot2 = fopen( temp, "w" );
#endif


/* Error handling */
  /* if ( out_control->out == NULL || out_control->pot == NULL || 
     out_control->log == NULL || out_control->mol == NULL || 
     out_control->dpl == NULL || out_control->drft == NULL ||       
     out_control->pdb == NULL )
     {
     fprintf( stderr, "FILE OPEN ERROR. TERMINATING..." );
     exit( CANNOT_OPEN_OUTFILE );
     }*/
}


void Initialize(reax_system *system, control_params *control, 
		simulation_data *data, static_storage *workspace, list **lists, 
		output_controls *out_control, evolve_function *Evolve)
{
  real start, end;
  Randomize();

  Init_System( system, control, data );

  Init_Simulation_Data( system, control, data, out_control, Evolve );

  Init_Workspace( system, control, workspace );
  
  Init_Lists( system, control, data, workspace, lists, out_control );

  Init_Out_Controls( system, control, workspace, out_control );

  /* These are done in forces.c, only forces.c can see all those functions */
  Init_Bonded_Force_Functions( control );
#ifdef TEST_FORCES
  Init_Force_Test_Functions( );
#endif

  if( control->tabulate ) {
    start = Get_Time ();
    Make_LR_Lookup_Table( system, control );
	 end = Get_Timing_Info (start);

	 //fprintf (stderr, "Time for LR Lookup Table calculation is %f \n", end );
  }

#if defined(DEBUG_FOCUS)
  fprintf( stderr, "data structures have been initialized...\n" ); 
#endif
}
