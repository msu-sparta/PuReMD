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

#include "integrate.h"
#include "allocate.h"
#include "box.h"
#include "forces.h"
#include "grid.h"
#include "neighbors.h"
#include "print_utils.h"
#include "QEq.h"
#include "reset_utils.h"
#include "restart.h"
#include "system_props.h"
#include "vector.h"
#include "list.h"

#include "cuda_utils.h"
#include "reduction.h"
#include "validation.h"


void Velocity_Verlet_NVE(reax_system* system, control_params* control, 
			 simulation_data *data, static_storage *workspace, 
			 list **lists, output_controls *out_control )
{
  int i, steps, renbr;
  real inv_m, dt, dt_sqr;
  rvec dx;

  dt = control->dt;
  dt_sqr = SQR(dt);
  steps = data->step - data->prev_steps;
  renbr = (steps % control->reneighbor == 0);
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "step%d: ", data->step );
#endif

  for( i = 0; i < system->N; i++ ) {
    inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;
    
    rvec_ScaledSum( dx, dt, system->atoms[i].v, 
		    0.5 * dt_sqr * -F_CONV * inv_m, system->atoms[i].f );
    Inc_on_T3( system->atoms[i].x, dx, &( system->box ) );
    
    rvec_ScaledAdd( system->atoms[i].v, 
		    0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
  }
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "verlet1 - ");
#endif

  Reallocate( system, workspace, lists, renbr );
  Reset( system, control, data, workspace, lists );
  if( renbr )
    Generate_Neighbor_Lists( system, control, data, workspace, 
			     lists, out_control );  
  Compute_Forces( system, control, data, workspace, lists, out_control );

  for( i = 0; i < system->N; i++ ) {
    inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;
    rvec_ScaledAdd( system->atoms[i].v, 
		    0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
  }
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "verlet2\n");
#endif
}

///////////////////////////////////////////////////////////////////
//Cuda Function -- Velocity Verlet NVE
///////////////////////////////////////////////////////////////////

GLOBAL void Cuda_Velocity_Verlet_NVE_atoms1 (reax_atom *atoms, 
														 single_body_parameters *sbp, 
														 simulation_box *box,
														 int N, real dt)
{
   real inv_m, dt_sqr;
   rvec dx;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

   dt_sqr = SQR(dt);
  //for( i = 0; i < system->N; i++ ) {
    inv_m = 1.0 / sbp[atoms[i].type].mass;
    
    rvec_ScaledSum( dx, dt, atoms[i].v, 
		    0.5 * dt_sqr * -F_CONV * inv_m, atoms[i].f );
    Inc_on_T3( atoms[i].x, dx, box );
    
    rvec_ScaledAdd( atoms[i].v, 
		    0.5 * dt * -F_CONV * inv_m, atoms[i].f );
  //}
}

GLOBAL void Cuda_Velocity_Verlet_NVE_atoms2 (reax_atom *atoms, single_body_parameters *sbp, int N, real dt)
{
   real inv_m;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

  //for( i = 0; i < system->N; i++ ) {
   inv_m = 1.0 / sbp[atoms[i].type].mass;
   rvec_ScaledAdd( atoms[i].v, 
		    0.5 * dt * -F_CONV * inv_m, atoms[i].f );
  //}
}

void Cuda_Velocity_Verlet_NVE(reax_system* system, control_params* control, 
			 simulation_data *data, static_storage *workspace, 
			 list **lists, output_controls *out_control )
{
  int i, steps, renbr;
  real inv_m, dt, dt_sqr;
  rvec dx;
  int blocks, block_size;

  dt = control->dt;
  dt_sqr = SQR(dt);
  steps = data->step - data->prev_steps;
  renbr = (steps % control->reneighbor == 0);

#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "step%d: ", data->step );
#endif

	compute_blocks (&blocks, &block_size, system->N);
	Cuda_Velocity_Verlet_NVE_atoms1 <<<blocks, block_size>>>
											(system->d_atoms, system->reaxprm.d_sbp, 
											(simulation_box *)system->d_box, system->N, dt);
	cudaThreadSynchronize ();

#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "verlet1 - ");
#endif

  Cuda_Reallocate( system, dev_workspace, dev_lists, renbr, data->step );
  Cuda_Reset( system, control, data, workspace, lists );

  if( renbr ) {
		Cuda_Generate_Neighbor_Lists (system, dev_workspace, control, true);
  }

   Cuda_Compute_Forces( system, control, data, workspace, lists, out_control );

	Cuda_Velocity_Verlet_NVE_atoms2<<<blocks, block_size>>>
											(system->d_atoms, system->reaxprm.d_sbp, system->N, dt);
	cudaThreadSynchronize ();

#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "verlet2\n");
#endif
}

void Velocity_Verlet_Nose_Hoover_NVT_Klein(reax_system* system, 
					   control_params* control, 
					   simulation_data *data, 
					   static_storage *workspace, 
					   list **lists, 
					   output_controls *out_control )
{
  int i, itr, steps, renbr;
  real inv_m, coef_v, dt, dt_sqr;
  real E_kin_new, G_xi_new, v_xi_new, v_xi_old;
  rvec dx;
  thermostat *therm;

  dt = control->dt;
  dt_sqr = SQR( dt );
  therm = &( data->therm );
  steps = data->step - data->prev_steps;
  renbr = (steps % control->reneighbor == 0);
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "step%d: ", data->step );
#endif

#ifdef __DEBUG_CUDA__
	 fprintf (stderr, " Entering Velocity_Verlet_Nose_Hoover_NVT_Klein:  coef to update velocity --> %6.10f\n", therm->v_xi_old);
#endif
  
  /* Compute x(t + dt) and copy old forces */
  for (i=0; i < system->N; i++) {
    inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;
    
    rvec_ScaledSum( dx, dt - 0.5 * dt_sqr * therm->v_xi, system->atoms[i].v,
		    0.5 * dt_sqr * inv_m * -F_CONV, system->atoms[i].f );
    
    Inc_on_T3( system->atoms[i].x, dx, &(system->box) );
    
    rvec_Copy( workspace->f_old[i], system->atoms[i].f );
  }
  /* Compute xi(t + dt) */
  therm->xi += ( therm->v_xi * dt + 0.5 * dt_sqr * therm->G_xi );
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "verlet1 - " );
#endif

  Reallocate( system, workspace, lists, renbr );
  Reset( system, control, data, workspace, lists );

  if( renbr )
    Generate_Neighbor_Lists( system, control, data, workspace, 
			     lists, out_control );

  /* Calculate Forces at time (t + dt) */
  Compute_Forces( system,control,data, workspace, lists, out_control );

  /* Compute iteration constants for each atom's velocity */
  for( i = 0; i < system->N; ++i ) {
    inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;
      
    rvec_Scale( workspace->v_const[i], 
		1.0 - 0.5 * dt * therm->v_xi, system->atoms[i].v );
    rvec_ScaledAdd( workspace->v_const[i], 
		    0.5 * dt * inv_m * -F_CONV, workspace->f_old[i] );
    rvec_ScaledAdd( workspace->v_const[i], 
		    0.5 * dt * inv_m * -F_CONV, system->atoms[i].f );
#if defined(DEBUG)
    fprintf( stderr, "atom%d: inv_m=%f, C1=%f, C2=%f, v_const=%f %f %f\n", 
	     i, inv_m, 1.0 - 0.5 * dt * therm->v_xi, 
	     0.5 * dt * inv_m * -F_CONV, workspace->v_const[i][0], 
	     workspace->v_const[i][1], workspace->v_const[i][2] );  
#endif
  }


  v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
  E_kin_new = G_xi_new = v_xi_old = 0;
  itr = 0;
  do {
    itr++;      
    
    /* new values become old in this iteration */
    v_xi_old = v_xi_new;
    coef_v = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
    E_kin_new = 0;

#ifdef __DEBUG_CUDA__
	 fprintf (stderr, " *********** coef to update velocity --> %6.10f, %6.10f, %6.10f\n", coef_v, dt, therm->v_xi_old);
	 //print_sys_atoms (system);
#endif

    for( i = 0; i < system->N; ++i ) {
      rvec_Scale( system->atoms[i].v, coef_v, workspace->v_const[i] );
      
      E_kin_new += ( 0.5*system->reaxprm.sbp[system->atoms[i].type].mass * 
		     rvec_Dot( system->atoms[i].v, system->atoms[i].v ) );
#if defined(DEBUG)
      fprintf( stderr, "itr%d-atom%d: coef_v = %f, v_xi_old = %f\n", 
	       itr, i, coef_v, v_xi_old );
#endif
    }

    G_xi_new = control->Tau_T * ( 2.0 * E_kin_new - 
				  data->N_f * K_B * control->T );
    v_xi_new = therm->v_xi + 0.5 * dt * ( therm->G_xi + G_xi_new );
#if defined(DEBUG)
    fprintf( stderr, "itr%d: G_xi_new = %f, v_xi_new = %f, v_xi_old = %f\n",
	     itr, G_xi_new, v_xi_new, v_xi_old );
#endif
  }
  while( fabs(v_xi_new - v_xi_old ) > 1e-5 );

#ifdef __DEBUG_CUDA__
  fprintf (stderr, " Iteration Count in NVE --> %d \n", itr );
#endif
  

#ifndef __BUILD_DEBUG__
  therm->v_xi_old = therm->v_xi;
  therm->v_xi = v_xi_new;
  therm->G_xi = G_xi_new;  
#endif 

#if defined(DEBUG_FOCUS)  
  fprintf( stderr,"vel scale\n" );
#endif 
}




///////////////////////////////////////////////////////////////////
//Cuda Function -- Velocity_Verlet_Nose_Hoover_NVT_Klein
///////////////////////////////////////////////////////////////////

GLOBAL void Compute_X_t_dt (real dt, real dt_sqr, thermostat p_therm,
									reax_atom *atoms, single_body_parameters *sbp, 
									simulation_box *box,
									static_storage p_workspace, int N)
{

  	real inv_m;
	rvec dx;
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= N) return;

	static_storage *workspace = &p_workspace;
	thermostat *therm = &p_therm;

  /* Compute x(t + dt) and copy old forces */
  //for (i=0; i < system->N; i++) {
    inv_m = 1.0 / sbp[atoms[i].type].mass;
    
    rvec_ScaledSum( dx, dt - 0.5 * dt_sqr * therm->v_xi, atoms[i].v,
		    0.5 * dt_sqr * inv_m * -F_CONV, atoms[i].f );
    
    Inc_on_T3( atoms[i].x, dx, box );
    
    rvec_Copy( workspace->f_old[i], atoms[i].f );
  //}

}

GLOBAL void Update_Velocity (reax_atom *atoms, single_body_parameters *sbp, 
									static_storage p_workspace, real dt, thermostat p_therm, 
									int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

  	real inv_m;
	static_storage *workspace = &p_workspace;
	thermostat *therm = &p_therm;

  //for( i = 0; i < system->N; ++i ) {
    inv_m = 1.0 / sbp[atoms[i].type].mass;
      
    rvec_Scale( workspace->v_const[i], 
		1.0 - 0.5 * dt * therm->v_xi, atoms[i].v );
    rvec_ScaledAdd( workspace->v_const[i], 
		    0.5 * dt * inv_m * -F_CONV, workspace->f_old[i] );
    rvec_ScaledAdd( workspace->v_const[i], 
		    0.5 * dt * inv_m * -F_CONV, atoms[i].f );
  //}
}

GLOBAL void E_Kin_Reduction (reax_atom *atoms, static_storage p_workspace,
										single_body_parameters *sbp, 
										real *per_block_results, real coef_v, const size_t n)
{
  extern __shared__ real sdata[];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  real x = 0;
  static_storage *workspace = &p_workspace;

  if(i < n)
  {
    rvec_Scale( atoms[i].v, coef_v, workspace->v_const[i] );
    x = ( 0.5 * sbp[atoms[i].type].mass * 
		     rvec_Dot( atoms[i].v, atoms[i].v ) );
  }
  sdata[threadIdx.x] = x;
  __syncthreads();

  for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
  {
    if(threadIdx.x < offset)
    {   
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }   

    __syncthreads();
  }

  if(threadIdx.x == 0)
  {
    per_block_results[blockIdx.x] = sdata[0];
  }
}


void Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein(reax_system* system, 
					   control_params* control, 
					   simulation_data *data, 
					   static_storage *workspace, 
					   list **lists, 
					   output_controls *out_control )
{
  int i, itr, steps, renbr;
  real inv_m, coef_v, dt, dt_sqr;
  real E_kin_new, G_xi_new, v_xi_new, v_xi_old;
  rvec dx;
  thermostat *therm;

  real *results = (real *)scratch;

  dt = control->dt;
  dt_sqr = SQR( dt );
  therm = &( data->therm );
  steps = data->step - data->prev_steps;
  renbr = (steps % control->reneighbor == 0);

#ifdef __DEBUG_CUDA__
	 fprintf (stderr, " Device: Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein --> coef to update velocity --> %6.10f\n", therm->v_xi_old);
#endif

#if defined(DEBUG_FOCUS)
  fprintf( stderr, "step%d: ", data->step );
#endif

	Compute_X_t_dt <<< BLOCKS, BLOCK_SIZE >>>
						(dt, dt_sqr, data->therm, system->d_atoms, 
						system->reaxprm.d_sbp, system->d_box, *dev_workspace, system->N);
	cudaThreadSynchronize ();
	cudaCheckError ();

  /* Compute xi(t + dt) */
  therm->xi += ( therm->v_xi * dt + 0.5 * dt_sqr * therm->G_xi );
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "verlet1 - " );
#endif

  Cuda_Reallocate( system, dev_workspace, dev_lists, renbr, data->step );
  Cuda_Reset( system, control, data, workspace, lists );

  if( renbr ) {
  		//generate_neighbor_lists here
		Cuda_Generate_Neighbor_Lists (system, dev_workspace, control, true);
	}

  /* Calculate Forces at time (t + dt) */
  Cuda_Compute_Forces( system,control,data, workspace, lists, out_control );

  /* Compute iteration constants for each atom's velocity */
  Update_Velocity <<< BLOCKS, BLOCK_SIZE >>>
  						(system->d_atoms, system->reaxprm.d_sbp, *dev_workspace,
							dt, *therm, system->N );
	cudaThreadSynchronize ();
	cudaCheckError ();


  v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
  E_kin_new = G_xi_new = v_xi_old = 0;
  itr = 0;
  do {
    itr++;      
    
    /* new values become old in this iteration */
    v_xi_old = v_xi_new;
    coef_v = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
    E_kin_new = 0;

	 /*reduction for the E_Kin_new here*/
#ifdef __DEBUG_CUDA__
	 fprintf (stderr, " Device: coef to update velocity --> %6.10f, %6.10f, %6.10f\n", coef_v, dt, therm->v_xi_old);
#endif
	 cuda_memset (results, 0, 2 * BLOCK_SIZE * REAL_SIZE, RES_SCRATCH );
	 E_Kin_Reduction <<< BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>>
							(system->d_atoms, *dev_workspace, system->reaxprm.d_sbp, 
								results, coef_v, system->N);
	 cudaThreadSynchronize ();
	 cudaCheckError ();

	 Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>>
	 						(results, results + BLOCKS_POW_2, BLOCKS_POW_2);
	 cudaThreadSynchronize ();
	 cudaCheckError ();

	 copy_host_device (&E_kin_new, results + BLOCKS_POW_2, REAL_SIZE, cudaMemcpyDeviceToHost, RES_SCRATCH ); 

    G_xi_new = control->Tau_T * ( 2.0 * E_kin_new - 
				  data->N_f * K_B * control->T );
    v_xi_new = therm->v_xi + 0.5 * dt * ( therm->G_xi + G_xi_new );
#if defined(DEBUG)
    fprintf( stderr, "itr%d: G_xi_new = %f, v_xi_new = %f, v_xi_old = %f\n",
	     itr, G_xi_new, v_xi_new, v_xi_old );
#endif
  }
  while( fabs(v_xi_new - v_xi_old ) > 1e-5 );

#ifdef __DEBUG_CUDA__
  fprintf (stderr, " Iteration Count in NVE --> %d \n", itr );
#endif

  therm->v_xi_old = therm->v_xi;
  therm->v_xi = v_xi_new;
  therm->G_xi = G_xi_new;  
#if defined(DEBUG_FOCUS)  
  fprintf( stderr,"vel scale\n" );
#endif 
}

///////////////////////////////////////////////////////////////////
//Cuda Function -- Velocity_Verlet_Nose_Hoover_NVT_Klein
///////////////////////////////////////////////////////////////////


/* uses Berendsen-type coupling for both T and P. 
   All box dimensions are scaled by the same amount, 
   there is no change in the angles between axes. */
void Velocity_Verlet_Berendsen_Isotropic_NPT( reax_system* system, 
					      control_params* control, 
					      simulation_data *data,
					      static_storage *workspace, 
					      list **lists, 
					      output_controls *out_control )
{
  int i, steps, renbr;
  real inv_m, dt, lambda, mu;
  rvec dx;

  dt = control->dt;
  steps = data->step - data->prev_steps;
  renbr = (steps % control->reneighbor == 0);
#if defined(DEBUG_FOCUS)
  //fprintf( out_control->prs, 
  //         "tau_t: %g  tau_p: %g  dt/tau_t: %g  dt/tau_p: %g\n", 
  //control->Tau_T, control->Tau_P, dt / control->Tau_T, dt / control->Tau_P );
  fprintf( stderr, "step %d: ", data->step );
#endif

  /* velocity verlet, 1st part */
  for( i = 0; i < system->N; i++ ) {
    inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;
    /* Compute x(t + dt) */
    rvec_ScaledSum( dx, dt, system->atoms[i].v, 
		    0.5 * -F_CONV * inv_m * SQR(dt), system->atoms[i].f );
    Inc_on_T3( system->atoms[i].x, dx, &(system->box) );
    /* Compute v(t + dt/2) */
    rvec_ScaledAdd( system->atoms[i].v, 
		    0.5 * -F_CONV * inv_m * dt, system->atoms[i].f );
    /*fprintf( stderr, "%6d   %15.8f %15.8f %15.8f   %15.8f %15.8f %15.8f\n", 
      workspace->orig_id[i], 
      system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2],
      0.5 * SQR(dt) * -F_CONV * inv_m * system->atoms[i].f[0], 
      0.5 * SQR(dt) * -F_CONV * inv_m * system->atoms[i].f[1], 
      0.5 * SQR(dt) * -F_CONV * inv_m * system->atoms[i].f[2] ); */
  }
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "verlet1 - " );
#endif

  Reallocate( system, workspace, lists, renbr );  
  Reset( system, control, data, workspace, lists );
  if( renbr ) {
    Update_Grid( system );
    Generate_Neighbor_Lists( system, control, data, workspace,
			     lists, out_control );
  }
  Compute_Forces( system, control, data, workspace, lists, out_control );

  /* velocity verlet, 2nd part */
  for( i = 0; i < system->N; i++ ) {
    inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;
    /* Compute v(t + dt) */
    rvec_ScaledAdd( system->atoms[i].v, 
		    0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
    /* fprintf( stderr, "%6d   %15f %15f %15f   %15.8f %15.8f %15.8f\n", 
       workspace->orig_id[i], 
       system->atoms[i].v[0], system->atoms[i].v[1], system->atoms[i].v[2],
       0.5 * dt * -F_CONV * inv_m * system->atoms[i].f[0], 
       0.5 * dt * -F_CONV * inv_m * system->atoms[i].f[1], 
       0.5 * dt * -F_CONV * inv_m * system->atoms[i].f[2] );*/
  }
  //Compute_Kinetic_Energy( system, data );   
  Compute_Pressure_Isotropic( system, control, data, out_control );
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "verlet2 - " );
#endif

  /* pressure scaler */
  mu = POW( 1.0 + (dt / control->Tau_P[0]) * (data->iso_bar.P - control->P[0]),
	    1.0 / 3 );
  if( mu < MIN_dV ) 
    mu = MIN_dV;
  else if( mu > MAX_dV )
    mu = MAX_dV;

  /* temperature scaler */
  lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
  if( lambda < MIN_dT )
    lambda = MIN_dT;
  else if (lambda > MAX_dT )
    lambda = MAX_dT;
  lambda = SQRT( lambda );

  /* Scale velocities and positions at t+dt */
  for( i = 0; i < system->N; ++i ) {
    rvec_Scale( system->atoms[i].v, lambda, system->atoms[i].v );
    /* IMPORTANT: What Adri does with scaling positions first to 
       unit coordinates and then back to cartesian coordinates essentially 
       is scaling the coordinates with mu^2. However, this causes unphysical 
       modifications on the system because box dimensions
       are being scaled with mu! We need to discuss this with Adri! */
    rvec_Scale( system->atoms[i].x, mu, system->atoms[i].x );
  }
  //Compute_Kinetic_Energy( system, data );
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "scaling - " );
#endif

  Update_Box_Isotropic( &(system->box), mu );
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "updated box\n" );
#endif
}


/* uses Berendsen-type coupling for both T and P. 
   All box dimensions are scaled by the same amount, 
   there is no change in the angles between axes. */
void Velocity_Verlet_Berendsen_SemiIsotropic_NPT( reax_system* system, 
						  control_params* control, 
						  simulation_data *data,
						  static_storage *workspace, 
						  list **lists, 
						  output_controls *out_control )
{
  int i, d, steps, renbr;
  real dt, inv_m, lambda;
  rvec dx, mu;

  dt = control->dt;
  steps = data->step - data->prev_steps;
  renbr = (steps % control->reneighbor == 0);
#if defined(DEBUG_FOCUS)
  //fprintf( out_control->prs, 
  //         "tau_t: %g  tau_p: %g  dt/tau_t: %g  dt/tau_p: %g\n", 
  //control->Tau_T, control->Tau_P, dt / control->Tau_T, dt / control->Tau_P );
  fprintf( stderr, "step %d: ", data->step );
#endif

  /* velocity verlet, 1st part */
  for( i = 0; i < system->N; i++ ) {
    inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass; 
    /* Compute x(t + dt) */
    rvec_ScaledSum( dx, dt, system->atoms[i].v, 
		    0.5 * -F_CONV * inv_m * SQR(dt), system->atoms[i].f );
    Inc_on_T3( system->atoms[i].x, dx, &(system->box) );
    /* Compute v(t + dt/2) */
    rvec_ScaledAdd( system->atoms[i].v, 
		    0.5 * -F_CONV * inv_m * dt, system->atoms[i].f );
    /*fprintf( stderr, "%6d   %15.8f %15.8f %15.8f   %15.8f %15.8f %15.8f\n", 
      workspace->orig_id[i], 
      system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2],
      0.5 * SQR(dt) * -F_CONV * inv_m * system->atoms[i].f[0], 
      0.5 * SQR(dt) * -F_CONV * inv_m * system->atoms[i].f[1], 
      0.5 * SQR(dt) * -F_CONV * inv_m * system->atoms[i].f[2] ); */
  }
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "verlet1 - " );
#endif

  Reallocate( system, workspace, lists, renbr ); 
  Reset( system, control, data, workspace, lists );
  if( renbr ) {
    Update_Grid( system );
    Generate_Neighbor_Lists( system, control, data, workspace, 
			     lists, out_control );
  }
  Compute_Forces( system, control, data, workspace, lists, out_control );

  /* velocity verlet, 2nd part */
  for( i = 0; i < system->N; i++ ) {
    inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;
    /* Compute v(t + dt) */
    rvec_ScaledAdd( system->atoms[i].v, 
		    0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
    /* fprintf( stderr, "%6d   %15f %15f %15f   %15.8f %15.8f %15.8f\n", 
       workspace->orig_id[i], 
       system->atoms[i].v[0], system->atoms[i].v[1], system->atoms[i].v[2],
       0.5 * dt * -F_CONV * inv_m * system->atoms[i].f[0], 
       0.5 * dt * -F_CONV * inv_m * system->atoms[i].f[1], 
       0.5 * dt * -F_CONV * inv_m * system->atoms[i].f[2] );*/
  }
  //Compute_Kinetic_Energy( system, data );   
  Compute_Pressure_Isotropic( system, control, data, out_control );
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "verlet2 - " );
#endif

  /* pressure scaler */
  for( d = 0; d < 3; ++d ){
    mu[d] = POW( 1.0+(dt/control->Tau_P[d])*(data->tot_press[d]-control->P[d]),
		 1.0 / 3 );
    if( mu[d] < MIN_dV ) 
      mu[d] = MIN_dV;
    else if( mu[d] > MAX_dV )
      mu[d] = MAX_dV;
  }

  /* temperature scaler */
  lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
  if( lambda < MIN_dT )
    lambda = MIN_dT;
  else if (lambda > MAX_dT )
    lambda = MAX_dT;
  lambda = SQRT( lambda );

  /* Scale velocities and positions at t+dt */
  for( i = 0; i < system->N; ++i ) {
    rvec_Scale( system->atoms[i].v, lambda, system->atoms[i].v );
    /* IMPORTANT: What Adri does with scaling positions first to 
       unit coordinates and then back to cartesian coordinates essentially 
       is scaling the coordinates with mu^2. However, this causes unphysical 
       modifications on the system because box dimensions
       are being scaled with mu! We need to discuss this with Adri! */
    for( d = 0; d < 3; ++d )
      system->atoms[i].x[d] = system->atoms[i].x[d] * mu[d];
  }
  //Compute_Kinetic_Energy( system, data );
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "scaling - " );
#endif

  Update_Box_SemiIsotropic( &(system->box), mu );
#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "updated box & grid\n" );
#endif
}



/************************************************/
/* BELOW FUNCTIONS ARE NOT BEING USED ANYMORE!  */
/*                                              */
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
/************************************************/

#ifdef ANISOTROPIC

void Velocity_Verlet_Nose_Hoover_NVT(reax_system* system, 
				     control_params* control, 
				     simulation_data *data,
				     static_storage *workspace, 
				     list **lists, 
				     output_controls *out_control )
{
  int i;
  real inv_m;
  real dt = control->dt;
  real dt_sqr = SQR(dt);
  rvec dx;

  for (i=0; i < system->N; i++)
    {
      inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;

      // Compute x(t + dt)
      rvec_ScaledSum( dx, dt, system->atoms[i].v, 
		      0.5 * dt_sqr * -F_CONV * inv_m, system->atoms[i].f );
      Inc_on_T3_Gen( system->atoms[i].x, dx, &(system->box) );

      // Compute v(t + dt/2)
      rvec_ScaledAdd( system->atoms[i].v, 
		      -0.5 * dt * data->therm.xi, system->atoms[i].v );
      rvec_ScaledAdd( system->atoms[i].v, 
		      0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
    }

  // Compute zeta(t + dt/2), E_Kininetic(t + dt/2)
  // IMPORTANT: What will be the initial value of zeta? and what is g?
  data->therm.xi += 0.5 * dt * control->Tau_T  * 
    ( 2.0 * data->E_Kin - data->N_f * K_B * control->T );

  Reset( system, control, data, workspace );
  fprintf(out_control->log,"reset-"); fflush( out_control->log );

  Generate_Neighbor_Lists( system, control, data, workspace, 
			   lists, out_control );
  fprintf(out_control->log,"nbrs-"); fflush( out_control->log );

  /* QEq( system, control, workspace, lists[FAR_NBRS], out_control );
     fprintf(out_control->log,"qeq-"); fflush( out_control->log ); */

  Compute_Forces( system, control, data, workspace, lists, out_control );
  fprintf(out_control->log,"forces\n"); fflush( out_control->log );

  //Compute_Kinetic_Energy( system, data );

  for( i = 0; i < system->N; i++ )
    {
      inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;

      // compute v(t + dt)
      rvec_ScaledAdd( system->atoms[i].v, 
		      -0.5 * dt * data->therm.xi, system->atoms[i].v );
      rvec_ScaledAdd( system->atoms[i].v, 
		      0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
    }
  
  // Compute zeta(t + dt)
  data->therm.xi += 0.5*dt * control->Tau_T  * ( 2.0 * data->E_Kin - 
						 data->N_f * K_B * control->T );

  fprintf( out_control->log,"Xi: %8.3f %8.3f %8.3f\n", 
	   data->therm.xi, data->E_Kin, data->N_f * K_B * control->T ); 
  fflush( out_control->log );
}



void Velocity_Verlet_Isotropic_NPT( reax_system* system, 
				    control_params* control, 
				    simulation_data *data,
				    static_storage *workspace, 
				    list **lists, 
				    output_controls *out_control )
{
  int i, itr;
  real deps, v_eps_new=0, v_eps_old=0, G_xi_new;
  real dxi, v_xi_new=0, v_xi_old=0, a_eps_new;
  real inv_m, exp_deps, inv_3V;
  real E_kin, P_int, P_int_const;
  real coef_v, coef_v_eps;
  real dt = control->dt;
  real dt_sqr = SQR( dt );
  thermostat *therm = &( data->therm );
  isotropic_barostat *iso_bar = &( data->iso_bar );
  simulation_box *box = &( system->box );
  rvec dx, dv;

  // Here we just calculate how much to increment eps, xi, v_eps, v_xi.
  // Commits are done after positions and velocities of atoms are updated
  // because position, velocity updates uses v_eps, v_xi terms; 
  // yet we need EXP( deps ) to be able to calculate 
  // positions and velocities accurately.  
  iso_bar->a_eps = control->Tau_P * 
    ( 3.0 * box->volume * (iso_bar->P - control->P) + 
      6.0 * data->E_Kin / data->N_f ) - iso_bar->v_eps * therm->v_xi;
  deps = dt * iso_bar->v_eps + 0.5 * dt_sqr * iso_bar->a_eps;
  exp_deps = EXP( deps );

  therm->G_xi = control->Tau_T * ( 2.0 * data->E_Kin + 
				   SQR( iso_bar->v_eps ) / control->Tau_P - 
				   (data->N_f +1) * K_B * control->T );
  dxi = therm->v_xi * dt + 0.5 * therm->G_xi * dt_sqr;

  fprintf(out_control->log, "a: %12.6f   eps: %12.6f   deps: %12.6f\n", 
	  iso_bar->a_eps, iso_bar->v_eps, iso_bar->eps);
  fprintf(out_control->log, "G: %12.6f   xi : %12.6f   dxi : %12.6f\n", 
	  therm->G_xi, therm->v_xi, therm->xi );

  // Update positions and velocities
  // NOTE: v_old, v_xi_old, v_eps_old are meant to be the old values 
  // in the iteration not the old values at time t or before!
  for (i=0; i < system->N; i++)
    {
      inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;

      // Compute x(t + dt)
      rvec_ScaledSum( workspace->a[i], -F_CONV * inv_m, system->atoms[i].f, 
		      -( (2.0 + 3.0/data->N_f) * iso_bar->v_eps + therm->v_xi ),
		      system->atoms[i].v );
      rvec_ScaledSum( dx, dt, system->atoms[i].v, 
		      0.5 * dt_sqr, workspace->a[i] );
      Inc_on_T3( system->atoms[i].x, dx, &(system->box) );
      rvec_Scale( system->atoms[i].x, exp_deps, system->atoms[i].x );
    }

  // Commit updates
  therm->xi += dxi;
  iso_bar->eps += deps;
  //Update_Box_Isotropic( EXP( 3.0 * iso_bar->eps ), &(system->box) );
  Update_Box_Isotropic( &(system->box), EXP( 3.0 * iso_bar->eps ) );


  // Calculate new forces, f(t + dt)
  Reset( system, control, data, workspace );
  fprintf(out_control->log,"reset-"); fflush( out_control->log );

  Generate_Neighbor_Lists( system, control, data, workspace, 
			   lists, out_control );
  fprintf(out_control->log,"nbrs-"); fflush( out_control->log );

  /* QEq( system, control, workspace, lists[FAR_NBRS], out_control );
     fprintf(out_control->log,"qeq-"); fflush( out_control->log ); */

  Compute_Forces( system, control, data, workspace, lists, out_control );
  fprintf(out_control->log,"forces\n"); fflush( out_control->log );


  // Compute iteration constants for each atom's velocity and for P_internal
  // Compute kinetic energy for initial velocities of the iteration
  P_int_const = E_kin = 0;
  for( i = 0; i < system->N; ++i )
    {
      inv_m = 1.0 / system->reaxprm.sbp[system->atoms[i].type].mass;

      rvec_ScaledSum( dv, 0.5 * dt, workspace->a[i], 
		      0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
      rvec_Add( dv, system->atoms[i].v );
      rvec_Scale( workspace->v_const[i], exp_deps, dv );

      P_int_const += ( -F_CONV * 
		       rvec_Dot( system->atoms[i].f, system->atoms[i].x ) );

      E_kin += (0.5 * system->reaxprm.sbp[system->atoms[i].type].mass * 
		rvec_Dot( system->atoms[i].v, system->atoms[i].v ) );
    }

  
  // Compute initial p_int
  inv_3V = 1.0 / (3.0 * system->box.volume);
  P_int = inv_3V * ( 2.0 * E_kin + P_int_const );

  v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
  v_eps_new = iso_bar->v_eps_old + 2.0 * dt * iso_bar->a_eps;

  itr = 0;
  do
    {
      itr++;
      // new values become old in this iteration
      v_xi_old = v_xi_new;
      v_eps_old = v_eps_new;


      for( i = 0; i < system->N; ++i )
	{
	  coef_v = 1.0 / (1.0 + 0.5 * dt * exp_deps * 
			  ( (2.0 + 3.0/data->N_f) * v_eps_old + v_xi_old ) );
	  rvec_Scale( system->atoms[i].v, coef_v, workspace->v_const[i] );
	}
      

      coef_v_eps = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
      a_eps_new = 3.0 * control->Tau_P * 
	( system->box.volume * (P_int - control->P) + 2.0 * E_kin / data->N_f );
      v_eps_new = coef_v_eps * ( iso_bar->v_eps + 
				 0.5 * dt * ( iso_bar->a_eps + a_eps_new ) );


      G_xi_new = control->Tau_T * ( 2.0 * E_kin + 
				    SQR( v_eps_old ) / control->Tau_P - 
				    (data->N_f + 1) * K_B * control->T );
      v_xi_new = therm->v_xi + 0.5 * dt * ( therm->G_xi + G_xi_new );
      

      E_kin = 0;
      for( i = 0; i < system->N; ++i )
	E_kin += (0.5 * system->reaxprm.sbp[system->atoms[i].type].mass * 
		  rvec_Dot( system->atoms[i].v, system->atoms[i].v ) );

      P_int = inv_3V * ( 2.0*E_kin + P_int_const );


      fprintf( out_control->log, 
	       "itr %d E_kin: %8.3f veps_n:%8.3f veps_o:%8.3f vxi_n:%8.3f vxi_o: %8.3f\n", 
	       itr, E_kin, v_eps_new, v_eps_old, v_xi_new, v_xi_old );
    }
  while( fabs(v_eps_new - v_eps_old) + fabs(v_xi_new - v_xi_old) > 2e-3 );


  therm->v_xi_old = therm->v_xi;
  therm->v_xi = v_xi_new;
  therm->G_xi = G_xi_new;

  iso_bar->v_eps_old = iso_bar->v_eps;
  iso_bar->v_eps = v_eps_new;
  iso_bar->a_eps = a_eps_new;

  fprintf( out_control->log, "V: %8.3ff\tsides{%8.3f, %8.3f, %8.3f}\n", 
	   system->box.volume, 
	   system->box.box[0][0],system->box.box[1][1],system->box.box[2][2] );
  fprintf(out_control->log,"eps:\ta- %8.3f  v- %8.3f  eps- %8.3f\n", 
	  iso_bar->a_eps, iso_bar->v_eps, iso_bar->eps);
  fprintf(out_control->log,"xi: \tG- %8.3f  v- %8.3f  xi - %8.3f\n", 
	  therm->G_xi, therm->v_xi, therm->xi);
}

#endif


////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
/* uses Berendsen-type coupling for both T and P. 
   All box dimensions are scaled by the same amount, 
   there is no change in the angles between axes. */
void Velocity_Verlet_Berendsen_NVT( reax_system* system,
                control_params* control,
                simulation_data *data,
                static_storage *workspace,
                list **lists,
                output_controls *out_control
                )
{
  int i, steps, renbr;
  real inv_m, dt, lambda;
  rvec dx;
  reax_atom *atom;

  fprintf (stderr, " Velocity_Verlet_Berendsen_NVT: step :%d \n", data->step);

#if defined(DEBUG_FOCUS)
  fprintf( stderr, "step%d\n", data->step );
#endif
  dt = control->dt;
  steps = data->step - data->prev_steps;
  renbr = (steps % control->reneighbor == 0);

  /* velocity verlet, 1st part */
  for( i = 0; i < system->N; i++ ) {
    atom = &(system->atoms[i]);
    inv_m = 1.0 / system->reaxprm.sbp[atom->type].mass;
    /* Compute x(t + dt) */
    rvec_ScaledSum( dx, dt, atom->v, 0.5 * -F_CONV * inv_m * SQR(dt), atom->f );
    rvec_Add( atom->x, dx );
    /* Compute v(t + dt/2) */
    rvec_ScaledAdd( atom->v, 0.5 * -F_CONV * inv_m * dt, atom->f );
  }
#if defined(DEBUG_FOCUS)
  fprintf(stderr, "step%d: verlet1 done\n", data->step);
#endif

  Reallocate( system, workspace, lists, renbr );
  Reset( system, control, data, workspace, lists );

  if( renbr )
    Generate_Neighbor_Lists( system, control, data, workspace, lists, out_control );

  Compute_Forces( system, control, data, workspace,
        lists, out_control );

  /* velocity verlet, 2nd part */
  for( i = 0; i < system->N; i++ ) {
    atom = &(system->atoms[i]);
    inv_m = 1.0 / system->reaxprm.sbp[atom->type].mass;
    /* Compute v(t + dt) */
    rvec_ScaledAdd( atom->v, 0.5 * dt * -F_CONV * inv_m, atom->f );
  }
#if defined(DEBUG_FOCUS)  
  fprintf(stderr, "step%d: verlet2 done\n", data->step);
#endif

  /* temperature scaler */
  Compute_Kinetic_Energy( system, data );
  lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
  if( lambda < MIN_dT )
    lambda = MIN_dT;
  else if (lambda > MAX_dT )
    lambda = MAX_dT;
  lambda = SQRT( lambda );

  /* Scale velocities and positions at t+dt */
  for( i = 0; i < system->N; ++i ) {
    atom = &(system->atoms[i]);
    rvec_Scale( atom->v, lambda, atom->v );
  }
  Compute_Kinetic_Energy( system, data );

#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "step%d: scaled velocities\n",
      data->step );
#endif
}

GLOBAL void ker_update_velocity_1 (reax_atom *atoms,
                                       single_body_parameters *sbp,
                                       real dt,
													simulation_box *box,
                                       int N)
{
   real inv_m;
   rvec dx;
   reax_atom *atom;
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if ( i >= N ) return;

  /* velocity verlet, 1st part */
  //for( i = 0; i < system->n; i++ ) { 
    atom = &(atoms[i]);
    inv_m = 1.0 / sbp[atom->type].mass;
    /* Compute x(t + dt) */
    rvec_ScaledSum( dx, dt, atom->v, 0.5 * -F_CONV * inv_m * SQR(dt), atom->f );
    rvec_Add( atom->x, dx );

	 /* Metin's suggestion to rebox the atoms */
	 /* bNVT fix */
    Inc_on_T3( atoms[i].x, dx, box );
	 /* bNVT fix */

    /* Compute v(t + dt/2) */
    rvec_ScaledAdd( atom->v, 0.5 * -F_CONV * inv_m * dt, atom->f );
  //}
}

void bNVT_update_velocity_part1 (reax_system *system, simulation_box *box, real dt)
{
   ker_update_velocity_1 <<< BLOCKS, BLOCK_SIZE>>>
      (system->d_atoms, system->reaxprm.d_sbp, dt, box, system->N);
   cudaThreadSynchronize ();
   cudaCheckError ();
}

GLOBAL void ker_update_velocity_2 (reax_atom *atoms,
                                       single_body_parameters *sbp,
                                       real dt,
                                       int N)
{
   reax_atom *atom;
   real inv_m;
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if ( i >= N ) return;

  /* velocity verlet, 2nd part */
  //for( i = 0; i < system->n; i++ ) { 
    atom = &(atoms[i]);
    inv_m = 1.0 / sbp[atom->type].mass;
    /* Compute v(t + dt) */
    rvec_ScaledAdd( atom->v, 0.5 * dt * -F_CONV * inv_m, atom->f );
  //}
}

void bNVT_update_velocity_part2 (reax_system *system, real dt)
{
   ker_update_velocity_2 <<< BLOCKS, BLOCK_SIZE >>>
      (system->d_atoms, system->reaxprm.d_sbp, dt, system->N);
   cudaThreadSynchronize ();
   cudaCheckError ();
}

GLOBAL void ker_scale_velocities (reax_atom *atoms, real lambda, int N)
{
   reax_atom *atom;
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if ( i >= N ) return;

  /* Scale velocities and positions at t+dt */
  //for( i = 0; i < system->n; ++i ) {
    atom = &(atoms[i]);
    rvec_Scale( atom->v, lambda, atom->v );
  //}
}

void bNVT_scale_velocities (reax_system *system, real lambda)
{
   ker_scale_velocities <<< BLOCKS, BLOCK_SIZE >>>
      (system->d_atoms, lambda, system->N);
   cudaThreadSynchronize ();
   cudaCheckError ();
}

void Cuda_Velocity_Verlet_Berendsen_NVT( reax_system* system,
                control_params* control,
                simulation_data *data,
                static_storage *workspace,
                list **lists,
                output_controls *out_control
                )
{
  int i, steps, renbr;
  real inv_m, dt, lambda;
  rvec dx;
  reax_atom *atom;

#if defined(DEBUG_FOCUS)
  fprintf( stderr, "step%d\n", data->step );
#endif
  dt = control->dt;
  steps = data->step - data->prev_steps;
  renbr = (steps % control->reneighbor == 0);

  /* velocity verlet, 1st part 
  for( i = 0; i < system->N; i++ ) { 
    atom = &(system->atoms[i]);
    inv_m = 1.0 / system->reaxprm.sbp[atom->type].mass;
    // Compute x(t + dt) 
    rvec_ScaledSum( dx, dt, atom->v, 0.5 * -F_CONV * inv_m * SQR(dt), atom->f );
    rvec_Add( atom->x, dx );
    // Compute v(t + dt/2) 
    rvec_ScaledAdd( atom->v, 0.5 * -F_CONV * inv_m * dt, atom->f );
  }
  */
  bNVT_update_velocity_part1 (system, (simulation_box *) system->d_box, dt);

#if defined(DEBUG_FOCUS)
  fprintf(stderr, "step%d: verlet1 done\n", data->step);
#endif

  Cuda_Reallocate( system, dev_workspace, dev_lists, renbr, data->step );
  Cuda_Reset( system, control, data, workspace, lists );

  if( renbr ) {
    Cuda_Generate_Neighbor_Lists( system, workspace, control, true);
  }

  Cuda_Compute_Forces( system, control, data, workspace,
        lists, out_control );

  /* velocity verlet, 2nd part 
  for( i = 0; i < system->N; i++ ) {
    atom = &(system->atoms[i]);
    inv_m = 1.0 / system->reaxprm.sbp[atom->type].mass;
    // Compute v(t + dt) 
    rvec_ScaledAdd( atom->v, 0.5 * dt * -F_CONV * inv_m, atom->f );
  }
  */
  bNVT_update_velocity_part2 (system, dt);
#if defined(DEBUG_FOCUS)  
  fprintf(stderr, "step%d: verlet2 done\n", data->step);
#endif

  /* temperature scaler */
  Cuda_Compute_Kinetic_Energy( system, data );
  //get the latest temperature from the device to the host.
  copy_host_device (&data->therm, &((simulation_data *)data->d_simulation_data)->therm,
                     sizeof (thermostat), cudaMemcpyDeviceToHost, RES_SIMULATION_DATA );

  lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
  if( lambda < MIN_dT )
    lambda = MIN_dT;
  else if (lambda > MAX_dT )
    lambda = MAX_dT;
  lambda = SQRT( lambda );

  //fprintf (stderr, "step:%d lambda -> %f \n", data->step, lambda);

  /* Scale velocities and positions at t+dt 
  for( i = 0; i < system->N; ++i ) {
    atom = &(system->atoms[i]);
    rvec_Scale( atom->v, lambda, atom->v );
  }
  */
  bNVT_scale_velocities (system, lambda);
  Cuda_Compute_Kinetic_Energy( system, data );

#if defined(DEBUG_FOCUS)  
  fprintf( stderr, "step%d: scaled velocities\n",
      data->step );
#endif

}


