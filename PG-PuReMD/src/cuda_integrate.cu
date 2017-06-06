
#include "cuda_integrate.h"
#include "reax_types.h"

#include "vector.h"
#include "cuda_utils.h"


CUDA_GLOBAL void k_update_velocity_1( reax_atom *my_atoms, 
        single_body_parameters *sbp, real dt, int n )
{
    real inv_m;
    rvec dx;
    reax_atom *atom;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* velocity verlet, 1st part */
    atom = &(my_atoms[i]);
    inv_m = 1.0 / sbp[atom->type].mass;
    /* Compute x(t + dt) */
    rvec_ScaledSum( dx, dt, atom->v, 0.5 * -F_CONV * inv_m * SQR(dt), atom->f );
    rvec_Add( atom->x, dx );
    /* Compute v(t + dt/2) */
    rvec_ScaledAdd( atom->v, 0.5 * -F_CONV * inv_m * dt, atom->f );
}


void bNVT_update_velocity_part1( reax_system *system, real dt )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_update_velocity_1 <<< blocks, DEF_BLOCK_SIZE >>>
        (system->d_my_atoms, system->reax_param.d_sbp, dt, system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_update_velocity_2( reax_atom *my_atoms, 
        single_body_parameters *sbp, real dt, int n )
{
    reax_atom *atom;
    real inv_m;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* velocity verlet, 2nd part */
    atom = &(my_atoms[i]);
    inv_m = 1.0 / sbp[atom->type].mass;
    /* Compute v(t + dt) */
    rvec_ScaledAdd( atom->v, 0.5 * dt * -F_CONV * inv_m, atom->f );
}


void bNVT_update_velocity_part2( reax_system *system, real dt )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_update_velocity_2 <<< blocks, DEF_BLOCK_SIZE >>>
        (system->d_my_atoms, system->reax_param.d_sbp, dt, system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_scale_velocities( reax_atom *my_atoms, real lambda, int n )
{
    reax_atom *atom;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* Scale velocities and positions at t+dt */
    atom = &(my_atoms[i]);
    rvec_Scale( atom->v, lambda, atom->v );
}


void bNVT_scale_velocities( reax_system *system, real lambda )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);
    k_scale_velocities <<< blocks, DEF_BLOCK_SIZE >>>
        (system->d_my_atoms, lambda, system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );
}
