
#include "cuda_integrate.h"

#include "cuda_allocate.h"
#include "cuda_box.h"
#include "cuda_forces.h"
#include "cuda_integrate.h"
#include "cuda_copy.h"
#include "cuda_neighbors.h"
#include "cuda_reduction.h"
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
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

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


void update_velocity_part1( reax_system *system, real dt )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_update_velocity_1 <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, dt, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_update_velocity_2( reax_atom *my_atoms, 
        single_body_parameters *sbp, real dt, int n )
{
    reax_atom *atom;
    real inv_m;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

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


void update_velocity_part2( reax_system *system, real dt )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_update_velocity_2 <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, dt, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_nhNVT_update_velocity_1( reax_atom *my_atoms, 
        single_body_parameters *sbp, real dt, int n )
{
    real inv_m;
    rvec dx;
    reax_atom *atom;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* velocity verlet, 1st part */
    atom = &(my_atoms[i]);
    inv_m = 1.0 / sbp[atom->type].mass;
    rvec_ScaledSum( dx, dt, atom->v, 0.5 * -F_CONV * inv_m * SQR(dt), atom->f );
    rvec_Add( atom->x, dx );
    rvec_Copy( atom->f_old, atom->f );
}


void nhNVT_update_velocity_part1( reax_system *system, real dt )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_nhNVT_update_velocity_1 <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, dt, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_nhNVT_update_velocity_2( reax_atom *my_atoms, rvec * v_const,
        single_body_parameters *sbp, real dt, real v_xi, int n )
{
    reax_atom *atom;
    real inv_m;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* velocity verlet, 2nd part */
    atom = &(my_atoms[i]);
    inv_m = 1.0 / sbp[atom->type].mass;
    /* Compute v(t + dt) */
    rvec_Scale( v_const[i], 1.0 - 0.5 * dt * v_xi, atom->v );
    rvec_ScaledAdd( v_const[i], 0.5 * dt * inv_m * -F_CONV, atom->f_old );
    rvec_ScaledAdd( v_const[i], 0.5 * dt * inv_m * -F_CONV, atom->f );
}


void nhNVT_update_velocity_part2( reax_system *system, storage *workspace, real dt, real v_xi )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_nhNVT_update_velocity_2 <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, workspace->v_const, system->reax_param.d_sbp, dt, v_xi, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_nhNVT_update_velocity_3( reax_atom *my_atoms, rvec *v_const,
        single_body_parameters *sbp, real dt, real v_xi_old, real * my_ekin, int n )
{
    reax_atom *atom;
    real coef_v;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    atom = &(my_atoms[i]);
    coef_v = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
    rvec_Scale( atom->v, coef_v, v_const[i] );
    my_ekin[i] = (0.5 * sbp[atom->type].mass * rvec_Dot(atom->v, atom->v));
}


int nhNVT_update_velocity_part3( reax_system *system, storage *workspace,
       real dt, real v_xi_old, real * d_my_ekin, real * d_total_my_ekin )
{
    int blocks, my_ekin;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_nhNVT_update_velocity_3 <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, workspace->v_const, system->reax_param.d_sbp, dt, v_xi_old, d_my_ekin, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );

    Cuda_Reduction_Sum( d_my_ekin, d_total_my_ekin, system->n );

    copy_host_device( &my_ekin, d_total_my_ekin, sizeof(int), 
            cudaMemcpyDeviceToHost, "nhNVT_update_velocity_part3::d_total_my_ekin" );

    return my_ekin;
}


CUDA_GLOBAL void k_bNVT_scale_velocities( reax_atom *my_atoms, real lambda, int n )
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

    k_bNVT_scale_velocities <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, lambda, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


CUDA_GLOBAL void k_bNVP_scale_velocities( reax_atom *my_atoms, real lambda,
        real mu0, real mu1, real mu2, int n )
{
    reax_atom *atom;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* Scale velocities and positions at t+dt */
    atom = &(my_atoms[i]);
    rvec_Scale( atom->v, lambda, atom->v );
//    rvec_Multiply( atom->x, mu, atom->x );
    atom->x[0] = mu0 * atom->x[0];
    atom->x[1] = mu1 * atom->x[1];
    atom->x[2] = mu2 * atom->x[2];
}


void bNVP_scale_velocities( reax_system *system, real lambda, rvec mu )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_bNVP_scale_velocities <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, lambda, mu[0], mu[1], mu[2], system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
}


int Cuda_Velocity_Verlet_NVE( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, far_nbrs_done = FALSE;
    real dt;
#if defined(DEBUG)
    real t_over_start, t_over_elapsed;
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step %d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;
    ret = SUCCESS;

    if ( verlet_part1_done == FALSE )
    {
        update_velocity_part1( system, dt );

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

        /* sync grid to device */
        Sync_Grid( &system->my_grid, &system->d_my_grid );

        init_blocks( system );
    }

    Cuda_ReAllocate( system, control, data, workspace, lists, mpi_data );

    Cuda_Reset( system, control, data, workspace, lists );

    if ( renbr && far_nbrs_done == FALSE )
    {
#if defined(DEBUG)
        t_over_start  = Get_Time( );
#endif

        ret = Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );

        if ( ret != SUCCESS )
        {
            Cuda_Estimate_Neighbors( system );
        }
        if ( ret == SUCCESS )
        {
            far_nbrs_done = TRUE;
        }
    
#if defined(DEBUG)
        t_over_elapsed = Get_Timing_Info( t_over_start );
        fprintf( stderr, "p%d --> Overhead (Step-%d) %f \n",
                system->my_rank, data->step, t_over_elapsed );
#endif
    }

    if ( ret == SUCCESS )
    {
        ret = Cuda_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        update_velocity_part2( system, dt );

        verlet_part1_done = FALSE;
        far_nbrs_done = FALSE;
    }
    
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: verlet2 done\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    return ret;
}


int Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein( reax_system* system,
        control_params* control, simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control, mpi_datatypes *mpi_data )
{
    int itr, steps, renbr, ret;
    real *d_my_ekin, *d_total_my_ekin;
    static int verlet_part1_done = FALSE, far_nbrs_done = FALSE;
    real dt, dt_sqr;
    real my_ekin, new_ekin;
    real G_xi_new, v_xi_new, v_xi_old;
    thermostat *therm;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    dt = control->dt;
    dt_sqr = SQR(dt);
    therm = &( data->therm );
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE )
    {
        nhNVT_update_velocity_part1( system, dt );
    
        /* Compute xi(t + dt) */
        therm->xi += ( therm->v_xi * dt + 0.5 * dt_sqr * therm->G_xi );

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

        /* sync grid to device */
        Sync_Grid( &system->my_grid, &system->d_my_grid );

        init_blocks( system );
    }

    Cuda_ReAllocate( system, control, data, workspace, lists, mpi_data );

    Cuda_Reset( system, control, data, workspace, lists );

    if ( renbr && far_nbrs_done == FALSE )
    {
#if defined(DEBUG)
        t_over_start  = Get_Time( );
#endif

        ret = Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );

        if ( ret != SUCCESS )
        {
            Cuda_Estimate_Neighbors( system );
        }
        if ( ret == SUCCESS )
        {
            far_nbrs_done = TRUE;
        }

#if defined(DEBUG)
        t_over_elapsed = Get_Timing_Info( t_over_start );
        fprintf( stderr, "p%d --> Overhead (Step-%d) %f \n",
                system->my_rank, data->step, t_over_elapsed );
#endif
    }

    if ( ret == SUCCESS )
    {
        ret = Cuda_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        /* Compute iteration constants for each atom's velocity */
        nhNVT_update_velocity_part2( system, dev_workspace, dt, therm->v_xi );
    
        v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
        my_ekin = G_xi_new = v_xi_old = 0;
        itr = 0;

        cuda_malloc( (void **) &d_my_ekin, sizeof(real) * system->n, FALSE,
                "Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein::d_my_ekin" );
        cuda_malloc( (void **) &d_total_my_ekin, sizeof(real), FALSE,
                "Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein::d_total_my_ekin" );

        do
        {
            itr++;
    
            /* new values become old in this iteration */
            v_xi_old = v_xi_new;
    
            my_ekin = nhNVT_update_velocity_part3( system, dev_workspace, dt, v_xi_old,
                    d_my_ekin, d_total_my_ekin );
    
            MPI_Allreduce( &my_ekin, &new_ekin, 1, MPI_DOUBLE, MPI_SUM,
                    mpi_data->comm_mesh3D  );
    
            G_xi_new = control->Tau_T * ( 2.0 * new_ekin - data->N_f * K_B * control->T );
            v_xi_new = therm->v_xi + 0.5 * dt * ( therm->G_xi + G_xi_new );
        }
        while ( FABS(v_xi_new - v_xi_old) > 1e-5 );
        therm->v_xi_old = therm->v_xi;
        therm->v_xi = v_xi_new;
        therm->G_xi = G_xi_new;

        cuda_free( d_total_my_ekin,
                "Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein::d_total_my_ekin" );
        cuda_free( d_my_ekin,
                "Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein::d_my_ekin" );

        verlet_part1_done = FALSE;
        far_nbrs_done = FALSE;
    }
    
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: verlet2 done\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    return ret;
}


/* uses Berendsen-type coupling for both T and P.
   All box dimensions are scaled by the same amount,
   there is no change in the angles between axes. */
int Cuda_Velocity_Verlet_Berendsen_NVT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, far_nbrs_done = FALSE;
    real dt, lambda;
#if defined(DEBUG)
    real t_over_start, t_over_elapsed;
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;
    ret = SUCCESS;

    if ( verlet_part1_done == FALSE )
    {
        /* velocity verlet, 1st part */
        update_velocity_part1( system, dt );

        verlet_part1_done = TRUE;

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: verlet1 done\n", system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        Cuda_ReAllocate( system, control, data, workspace, lists, mpi_data );

        if ( renbr )
        {
            Update_Grid( system, control, mpi_data->world );
        }

        Output_Sync_Atoms( system );
        Comm_Atoms( system, control, data, workspace, lists, mpi_data, renbr );
        Sync_Atoms( system );

        /* sync grid to device */
        Sync_Grid( &system->my_grid, &system->d_my_grid );

        init_blocks( system );
    
        Cuda_Reset( system, control, data, workspace, lists );
    }
    else
    {
        Cuda_ReAllocate( system, control, data, workspace, lists, mpi_data );
    
        Cuda_Reset( system, control, data, workspace, lists );
    }

    if ( renbr && far_nbrs_done == FALSE )
    {
#if defined(DEBUG)
        t_over_start  = Get_Time( );
#endif

        ret = Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );

        if ( ret != SUCCESS )
        {
            Cuda_Estimate_Neighbors( system );
        }
        if ( ret == SUCCESS )
        {
            far_nbrs_done = TRUE;
        }
        
#if defined(DEBUG)
        t_over_elapsed  = Get_Timing_Info( t_over_start );
        fprintf( stderr, "p%d --> Overhead (Step-%d) %f \n",
                system->my_rank, data->step, t_over_elapsed );
#endif
    }

    if ( ret == SUCCESS )
    {
        ret = Cuda_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        /* velocity verlet, 2nd part */
        update_velocity_part2( system, dt );

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
        far_nbrs_done = FALSE;
    }

    return ret;
}


/* uses Berendsen-type coupling for both T and P.
 * All box dimensions are scaled by the same amount,
 * there is no change in the angles between axes. */
int Cuda_Velocity_Verlet_Berendsen_NPT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, far_nbrs_done = FALSE;
    real dt;
#if defined(DEBUG)
    real t_over_start, t_over_elapsed;
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step %d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;
    ret = SUCCESS;

    if ( verlet_part1_done == FALSE )
    {
        update_velocity_part1( system, dt );

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

        /* sync grid to device */
        Sync_Grid( &system->my_grid, &system->d_my_grid );

        init_blocks( system );
    }

    Cuda_ReAllocate( system, control, data, workspace, lists, mpi_data );

    Cuda_Reset( system, control, data, workspace, lists );

    if ( renbr && far_nbrs_done == FALSE )
    {
#if defined(DEBUG)
        t_over_start  = Get_Time( );
#endif

        ret = Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );

        if ( ret != SUCCESS )
        {
            Cuda_Estimate_Neighbors( system );
        }
        if ( ret == SUCCESS )
        {
            far_nbrs_done = TRUE;
        }
    
#if defined(DEBUG)
        t_over_elapsed = Get_Timing_Info( t_over_start );
        fprintf( stderr, "p%d --> Overhead (Step-%d) %f \n",
                system->my_rank, data->step, t_over_elapsed );
#endif
    }

    if ( ret == SUCCESS )
    {
        ret = Cuda_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        update_velocity_part2( system, dt );

        Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
        Cuda_Compute_Pressure( system, control, data, mpi_data );
        Cuda_Scale_Box( system, control, data, mpi_data );

        verlet_part1_done = FALSE;
        far_nbrs_done = FALSE;
    }
    
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d: verlet2 done\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    return ret;
}
