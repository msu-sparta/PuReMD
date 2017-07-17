
#include "cuda_integrate.h"

#include "cuda_allocate.h"
#include "cuda_forces.h"
#include "cuda_integrate.h"
#include "cuda_copy.h"
#include "cuda_neighbors.h"
#include "cuda_reset_tools.h"
#include "cuda_system_props.h"
#include "cuda_utils.h"

#include "../comm_tools.h"
#include "../grid.h"
#include "../vector.h"


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


int Cuda_Velocity_Verlet_Berendsen_NVT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, steps, renbr, ret;
    static int verlet_part1_done = FALSE, estimate_nbrs_done = 0;
    real inv_m, dt, lambda;
    rvec dx;
    reax_atom *atom;
    int *bond_top, *hb_top;
    int Htop, num_3body;
    int total_hbonds, count, total_bonds;
    int bond_cap, cap_3body;
    real t_over_start, t_over_elapsed;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;
    ret = SUCCESS;

    Cuda_ReAllocate( system, control, data, workspace, lists, mpi_data );

    if ( verlet_part1_done == FALSE )
    {
        /* velocity verlet, 1st part */
        bNVT_update_velocity_part1( system, dt );
        verlet_part1_done = TRUE;

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: verlet1 done\n", system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        if ( renbr )
        {
            Update_Grid( system, control, mpi_data->world );
        }

        Output_Sync_Atoms( system );
        Comm_Atoms( system, control, data, workspace, lists, mpi_data, renbr );
        Sync_Atoms( system );

        /* synch the Grid to the Device here */
        Sync_Grid( &system->my_grid, &system->d_my_grid );

        init_blocks( system );

#if defined(__CUDA_DEBUG_LOG__)
        fprintf( stderr, "p:%d - Matvec BLocks: %d, blocksize: %d \n",
                system->my_rank, MATVEC_BLOCKS, MATVEC_BLOCK_SIZE );
#endif
    }
    
    Cuda_Reset( system, control, data, workspace, lists );

    if ( renbr )
    {
#if defined(DEBUG)
        t_over_start  = Get_Time ();
#endif

        if ( estimate_nbrs_done == 0 )
        {
            //TODO: move far_nbrs reallocation checks outside of renbr frequency check
            ret = Cuda_Estimate_Neighbors( system, data->step );
            estimate_nbrs_done = 1;
        }

        if ( ret == SUCCESS && estimate_nbrs_done == 1 )
        {
            Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );
            estimate_nbrs_done = 2;
    
#if defined(DEBUG)
            t_over_elapsed  = Get_Timing_Info( t_over_start );
            fprintf( stderr, "p%d --> Overhead (Step-%d) %f \n",
                    system->my_rank, data->step, t_over_elapsed );
#endif
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Cuda_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        /* velocity verlet, 2nd part */
        bNVT_update_velocity_part2( system, dt );

#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: verlet2 done\n", system->my_rank, data->step);
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        /* temperature scaler */
        Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

        lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
        if ( lambda < MIN_dT )
        {
            lambda = MIN_dT;
        }
        else if (lambda > MAX_dT )
        {
            lambda = MAX_dT;
        }
        lambda = SQRT( lambda );

        /* Scale velocities and positions at t+dt */
        bNVT_scale_velocities( system, lambda );

        Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: scaled velocities\n",
                 system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        verlet_part1_done = FALSE;
        estimate_nbrs_done = 0;
    }

    return ret;
}
