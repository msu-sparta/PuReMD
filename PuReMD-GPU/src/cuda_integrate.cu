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

#include "cuda_integrate.h"

#include "allocate.h"
#include "box.h"
#include "forces.h"
#include "grid.h"
#include "print_utils.h"
#include "reset_utils.h"
#include "system_props.h"
#include "vector.h"
#include "list.h"

#include "cuda_utils.h"
#include "cuda_reduction.h"
#include "cuda_allocate.h"
#include "cuda_forces.h"
#include "cuda_grid.h"
#include "cuda_neighbors.h"
#include "cuda_QEq.h"
#include "cuda_reset_utils.h"
#include "cuda_system_props.h"
#include "validation.h"


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
        Cuda_Generate_Neighbor_Lists (system, dev_workspace, control, TRUE);
    }

    Cuda_Compute_Forces( system, control, data, workspace, lists, out_control );

    Cuda_Velocity_Verlet_NVE_atoms2<<<blocks, block_size>>>
        (system->d_atoms, system->reaxprm.d_sbp, system->N, dt);
    cudaThreadSynchronize ();

#if defined(DEBUG_FOCUS)  
    fprintf( stderr, "verlet2\n");
#endif
}


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

    if( renbr )
    {
        //generate_neighbor_lists here
        Cuda_Generate_Neighbor_Lists (system, dev_workspace, control, TRUE);
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

#ifdef __DEBUG_CUDA__
        fprintf (stderr, " Device: coef to update velocity --> %6.10f, %6.10f, %6.10f\n", coef_v, dt, therm->v_xi_old);
#endif

        /*reduction for the E_Kin_new here*/
        cuda_memset (results, 0, 2 * BLOCK_SIZE * REAL_SIZE, RES_SCRATCH );
        E_Kin_Reduction <<< BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>>
            (system->d_atoms, *dev_workspace, system->reaxprm.d_sbp, 
             results, coef_v, system->N);
        cudaThreadSynchronize ();
        cudaCheckError ();

        Cuda_reduction<<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>>
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
        Cuda_Generate_Neighbor_Lists( system, workspace, control, TRUE);
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


