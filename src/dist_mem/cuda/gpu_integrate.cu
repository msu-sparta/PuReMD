
#include "gpu_integrate.h"

#include "gpu_allocate.h"
#include "gpu_box.h"
#include "gpu_environment.h"
#include "gpu_forces.h"
#include "gpu_integrate.h"
#include "gpu_copy.h"
#include "gpu_neighbors.h"
#include "gpu_reduction.h"
#include "gpu_reset_tools.h"
#include "gpu_system_props.h"
#include "gpu_utils.h"

#include "../comm_tools.h"
#include "../grid.h"
#if defined(DEBUG_FOCUS)
  #include "../tool_box.h"
#endif
#include "../vector.h"


GPU_GLOBAL void k_velocity_verlet_part1( reax_atom * const my_atoms, 
        single_body_parameters const * const sbp, real scalar1, real scalar2, real dt, int n )
{
    int i;
    real inv_m;
    rvec dx;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n ) {
        return;
    }

    /* velocity verlet, 1st part */
    atom = &my_atoms[i];
    inv_m = 1.0 / sbp[atom->type].mass;

    /* Compute x(t + dt) */
    rvec_ScaledSum( dx, dt, atom->v, scalar2 * inv_m, atom->f );
    rvec_Add( atom->x, dx );

    /* Compute v(t + dt/2) */
    rvec_ScaledAdd( atom->v, scalar1 * inv_m, atom->f );
}


GPU_GLOBAL void k_velocity_verlet_part2( reax_atom * const my_atoms, 
        single_body_parameters const * const sbp, real scalar1, int n )
{
    int i;
    reax_atom *atom;
    real inv_m;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n ) {
        return;
    }

    /* velocity verlet, 2nd part */
    atom = &my_atoms[i];
    inv_m = 1.0 / sbp[atom->type].mass;

    /* Compute v(t + dt) */
    rvec_ScaledAdd( atom->v, scalar1 * inv_m, atom->f );
}


GPU_GLOBAL void k_velocity_verlet_nose_hoover_nvt( reax_atom * const my_atoms, 
        single_body_parameters const * const sbp, real scalar2, real dt, int n )
{
    int i;
    real inv_m;
    rvec dx;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n ) {
        return;
    }

    /* velocity verlet, 1st part */
    atom = &my_atoms[i];
    inv_m = 1.0 / sbp[atom->type].mass;

    rvec_ScaledSum( dx, dt, atom->v, scalar2 * inv_m, atom->f );
    rvec_Add( atom->x, dx );
    rvec_Copy( atom->f_old, atom->f );
}

GPU_GLOBAL void k_velocity_verlet_nose_hoover_nvt_part2( reax_atom * const my_atoms,
        rvec * const v_const, single_body_parameters const * const sbp,
        real scalar1, real dt, real v_xi, int n )
{
    reax_atom *atom;
    real inv_m;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n ) {
        return;
    }

    /* velocity verlet, 2nd part */
    atom = &my_atoms[i];
    inv_m = 1.0 / sbp[atom->type].mass;

    /* Compute v(t + dt) */
    rvec_Scale( v_const[i], 1.0 - 0.5 * dt * v_xi, atom->v );
    rvec_ScaledAdd( v_const[i], scalar1 * inv_m, atom->f_old );
    rvec_ScaledAdd( v_const[i], scalar1 * inv_m, atom->f );
}


GPU_GLOBAL void k_velocity_verlet_nose_hoover_nvt_part3( reax_atom * const my_atoms,
        rvec const * const v_const, single_body_parameters const * const sbp,
        real dt, real v_xi_old, real * const my_ekin, int n )
{
    int i;
    real coef_v;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n ) {
        return;
    }

    atom = &my_atoms[i];

    coef_v = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
    rvec_Scale( atom->v, coef_v, v_const[i] );
    my_ekin[i] = 0.5 * sbp[atom->type].mass * rvec_Dot( atom->v, atom->v );
}


GPU_GLOBAL void k_scale_velocites_berendsen_nvt( reax_atom * const my_atoms,
        real lambda, int n )
{
    int i;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n ) {
        return;
    }

    /* Scale velocities and positions at t+dt */
    atom = &my_atoms[i];
    rvec_Scale( atom->v, lambda, atom->v );
}


GPU_GLOBAL void k_scale_velocities_npt( reax_atom * const my_atoms, real lambda,
        real mu0, real mu1, real mu2, int n )
{
    int i;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n ) {
        return;
    }

    /* Scale velocities and positions at t+dt */
    atom = &my_atoms[i];

    rvec_Scale( atom->v, lambda, atom->v );
//    rvec_Multiply( atom->x, mu, atom->x );
    atom->x[0] = mu0 * atom->x[0];
    atom->x[1] = mu1 * atom->x[1];
    atom->x[2] = mu2 * atom->x[2];
}


static void Velocity_Verlet_Part1( reax_system * const system,
        control_params const * const control, real dt )
{
    k_velocity_verlet_part1 <<< control->blocks_n, control->gpu_block_size,
                            0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp,
          -0.5 * dt * F_CONV, -0.5 * SQR(dt) * F_CONV, dt, system->n );
    cudaCheckError( );
}


static void Velocity_Verlet_Part2( reax_system * const system,
        control_params const * const control, real dt )
{
    k_velocity_verlet_part2 <<< control->blocks_n, control->gpu_block_size,
                            0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp,
          -0.5 * dt * F_CONV, system->n );
    cudaCheckError( );
}


static void Velocity_Verlet_Nose_Hoover_NVT_Part1( reax_system *system,
        control_params *control, real dt )
{
    k_velocity_verlet_nose_hoover_nvt <<< control->blocks_n, control->gpu_block_size,
                                      0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp,
          -0.5 * SQR(dt) * F_CONV, dt, system->n );
    cudaCheckError( );
}


static void Velocity_Verlet_Nose_Hoover_NVT_Part2( reax_system * const system,
        control_params const * const control, storage * const workspace,
        real dt, real v_xi )
{
    k_velocity_verlet_nose_hoover_nvt_part2 <<< control->blocks_n, control->gpu_block_size,
                                      0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, workspace->v_const, system->reax_param.d_sbp,
          -0.5 * dt * F_CONV, dt, v_xi, system->n );
    cudaCheckError( );
}


static real Velocity_Verlet_Nose_Hoover_NVT_Part3( reax_system * const system,
        control_params const * const control, storage * const workspace,
        real dt, real v_xi_old, real * const d_my_ekin, real * const d_total_my_ekin )
{
    real my_ekin;

    k_velocity_verlet_nose_hoover_nvt_part3 <<< control->blocks_n, control->gpu_block_size,
                                            0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, workspace->v_const, system->reax_param.d_sbp,
          dt, v_xi_old, d_my_ekin, system->n );
    cudaCheckError( );

    GPU_Reduction_Sum( d_my_ekin, d_total_my_ekin, system->n, 0, control->gpu_streams[0] );

    sCudaMemcpyAsync( &my_ekin, d_total_my_ekin, sizeof(real), 
            cudaMemcpyDeviceToHost, control->gpu_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->gpu_streams[0] );

    return my_ekin;
}


static void GPU_Scale_Velocities_Berendsen_NVT( reax_system * const system,
        control_params const * const control, real lambda )
{
    k_scale_velocites_berendsen_nvt <<< control->blocks_n, control->gpu_block_size,
                                    0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, lambda, system->n );
    cudaCheckError( );
}


void GPU_Scale_Velocities_NPT( reax_system * const system,
        control_params const * const control, real lambda, rvec mu )
{
    k_scale_velocities_npt <<< control->blocks_n, control->gpu_block_size,
                           0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, lambda, mu[0], mu[1], mu[2], system->n );
    cudaCheckError( );
}


int GPU_Velocity_Verlet_NVE( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE, gpu_copy = FALSE;
    real dt;

    ret = SUCCESS;
    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE ) {
        Velocity_Verlet_Part1( system, control, dt );

        GPU_Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        GPU_Copy_Atoms_Device_to_Host( system, control );
        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

#if defined(GPU_DEVICE_PACK)
        if ( renbr == TRUE ) {
            //TODO: remove once Comm_Atoms ported
            GPU_Copy_MPI_Data_Host_to_Device( control, mpi_data );
        }
#endif

        GPU_Init_Block_Sizes( system, control );

        verlet_part1_done = TRUE;
    }

    GPU_Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    if ( gpu_copy == FALSE ) {
        GPU_Copy_Atoms_Host_to_Device( system, control );

        if ( renbr == TRUE ) {
            GPU_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );
        }

        GPU_Reset( system, control, data, workspace, lists, renbr );

        gpu_copy = TRUE;
    }

    if ( renbr == TRUE && gen_nbr_list == FALSE ) {
        ret = GPU_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        if ( ret == SUCCESS ) {
            gen_nbr_list = TRUE;
        } else {
            GPU_Estimate_Num_Neighbors( system, control, data );
        }
    }

    if ( ret == SUCCESS ) {
        ret = GPU_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS ) {
        Velocity_Verlet_Part2( system, control, dt );

        verlet_part1_done = FALSE;
        gpu_copy = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


int GPU_Velocity_Verlet_Nose_Hoover_NVT_Klein( reax_system* system,
        control_params* control, simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control, mpi_datatypes *mpi_data )
{
    int itr, steps, renbr, ret, ret_mpi;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE, gpu_copy = FALSE;
    real *d_my_ekin, *d_total_my_ekin;
    real dt, dt_sqr;
    real my_ekin, new_ekin;
    real G_xi_new, v_xi_new, v_xi_old;
    thermostat *therm;

    ret = SUCCESS;
    dt = control->dt;
    dt_sqr = SQR(dt);
    therm = &data->therm;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE ) {
        Velocity_Verlet_Nose_Hoover_NVT_Part1( system, control, dt );
    
        /* Compute xi(t + dt) */
        therm->xi += therm->v_xi * dt + 0.5 * dt_sqr * therm->G_xi;

        GPU_Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        if ( renbr == TRUE ) {
            Update_Grid( system, control, MPI_COMM_WORLD );
        }

        GPU_Copy_Atoms_Device_to_Host( system, control );
        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

#if defined(GPU_DEVICE_PACK)
        if ( renbr == TRUE ) {
            //TODO: remove once Comm_Atoms ported
            GPU_Copy_MPI_Data_Host_to_Device( control, mpi_data );
        }
#endif

        GPU_Init_Block_Sizes( system, control );

        verlet_part1_done = TRUE;
    }

    GPU_Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    if ( gpu_copy == FALSE ) {
        GPU_Copy_Atoms_Host_to_Device( system, control );

        if ( renbr == TRUE ) {
            GPU_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );
        }

        GPU_Reset( system, control, data, workspace, lists, renbr );

        gpu_copy = TRUE;
    }

    if ( renbr == TRUE && gen_nbr_list == FALSE ) {
        ret = GPU_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        if ( ret == SUCCESS ) {
            gen_nbr_list = TRUE;
        } else {
            GPU_Estimate_Num_Neighbors( system, control, data );
        }
    }

    if ( ret == SUCCESS ) {
        ret = GPU_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS ) {
        /* Compute iteration constants for each atom's velocity */
        Velocity_Verlet_Nose_Hoover_NVT_Part2( system, control,
                workspace->d_workspace, dt, therm->v_xi );
    
        v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
        my_ekin = G_xi_new = v_xi_old = 0;
        itr = 0;

        sCudaMalloc( (void **) &d_my_ekin, sizeof(real) * system->n,
                __FILE__, __LINE__ );
        sCudaMalloc( (void **) &d_total_my_ekin, sizeof(real),
                __FILE__, __LINE__ );

        do {
            itr++;
    
            /* new values become old in this iteration */
            v_xi_old = v_xi_new;
    
            my_ekin = Velocity_Verlet_Nose_Hoover_NVT_Part3( system, control,
                    workspace->d_workspace, dt, v_xi_old,
                    d_my_ekin, d_total_my_ekin );
    
            ret_mpi = MPI_Allreduce( &my_ekin, &new_ekin, 1, MPI_DOUBLE, MPI_SUM,
                    mpi_data->comm_mesh3D  );
            Check_MPI_Error( ret_mpi, __FILE__, __LINE__ );
    
            G_xi_new = control->Tau_T * ( 2.0 * new_ekin - data->N_f * K_B * control->T );
            v_xi_new = therm->v_xi + 0.5 * dt * ( therm->G_xi + G_xi_new );
        } while ( FABS(v_xi_new - v_xi_old) > 1e-5 );
        therm->v_xi_old = therm->v_xi;
        therm->v_xi = v_xi_new;
        therm->G_xi = G_xi_new;

        sCudaFree( d_total_my_ekin, __FILE__, __LINE__ );
        sCudaFree( d_my_ekin, __FILE__, __LINE__ );

        verlet_part1_done = FALSE;
        gpu_copy = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


/* uses Berendsen-type coupling for both T and P.
   All box dimensions are scaled by the same amount,
   there is no change in the angles between axes. */
int GPU_Velocity_Verlet_Berendsen_NVT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE, gpu_copy = FALSE;
    real dt, lambda;

    ret = SUCCESS;
    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE ) {
        Velocity_Verlet_Part1( system, control, dt );

        GPU_Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        if ( renbr == TRUE ) {
            Update_Grid( system, control, MPI_COMM_WORLD );
        }

        GPU_Copy_Atoms_Device_to_Host( system, control );
        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

#if defined(GPU_DEVICE_PACK)
        if ( renbr == TRUE ) {
            //TODO: remove once Comm_Atoms ported
            GPU_Copy_MPI_Data_Host_to_Device( control, mpi_data );
        }
#endif

        GPU_Init_Block_Sizes( system, control );

        verlet_part1_done = TRUE;
    }

    GPU_Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    if ( gpu_copy == FALSE ) {
        GPU_Copy_Atoms_Host_to_Device( system, control );

        if ( renbr == TRUE ) {
            GPU_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );
        }

        GPU_Reset( system, control, data, workspace, lists, renbr );

        gpu_copy = TRUE;
    }

    if ( renbr == TRUE && gen_nbr_list == FALSE ) {
        ret = GPU_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        if ( ret == SUCCESS ) {
            gen_nbr_list = TRUE;
        } else {
            GPU_Estimate_Num_Neighbors( system, control, data );
        }
    }

    if ( ret == SUCCESS ) {
        ret = GPU_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS ) {
        Velocity_Verlet_Part2( system, control, dt );

        /* temperature scaler */
        GPU_Compute_Kinetic_Energy( system, control, workspace,
                data, mpi_data->comm_mesh3D );

        lambda = 1.0 + ((dt * 1.0e-12) / control->Tau_T)
            * (control->T / data->therm.T - 1.0);
        
        if ( lambda < MIN_dT ) {
            lambda = MIN_dT;
        }

        lambda = SQRT( lambda );

        if ( lambda > MAX_dT ) {
            lambda = MAX_dT;
        }

        /* Scale velocities and positions at t+dt */
        GPU_Scale_Velocities_Berendsen_NVT( system, control, lambda );

        GPU_Compute_Kinetic_Energy( system, control, workspace,
                data, mpi_data->comm_mesh3D );

        verlet_part1_done = FALSE;
        gpu_copy = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


/* uses Berendsen-type coupling for both T and P.
 * All box dimensions are scaled by the same amount,
 * there is no change in the angles between axes. */
int GPU_Velocity_Verlet_Berendsen_NPT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE, gpu_copy = FALSE;
    real dt;

    ret = SUCCESS;
    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE ) {
        Velocity_Verlet_Part1( system, control, dt );

        GPU_Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        if ( renbr == TRUE ) {
            Update_Grid( system, control, MPI_COMM_WORLD );
        }

        GPU_Copy_Atoms_Device_to_Host( system, control );
        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

#if defined(GPU_DEVICE_PACK)
        if ( renbr == TRUE ) {
            //TODO: remove once Comm_Atoms ported
            GPU_Copy_MPI_Data_Host_to_Device( control, mpi_data );
        }
#endif

        GPU_Init_Block_Sizes( system, control );

        verlet_part1_done = TRUE;
    }

    GPU_Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    if ( gpu_copy == FALSE ) {
        GPU_Copy_Atoms_Host_to_Device( system, control );

        if ( renbr == TRUE ) {
            GPU_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );
        }

        GPU_Reset( system, control, data, workspace, lists, renbr );

        gpu_copy = TRUE;
    }

    if ( renbr == TRUE && gen_nbr_list == FALSE ) {
        ret = GPU_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        if ( ret == SUCCESS ) {
            gen_nbr_list = TRUE;
        } else {
            GPU_Estimate_Num_Neighbors( system, control, data );
        }
    }

    if ( ret == SUCCESS ) {
        ret = GPU_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS ) {
        Velocity_Verlet_Part2( system, control, dt );

        GPU_Compute_Kinetic_Energy( system, control,
                workspace, data, mpi_data->comm_mesh3D );
        GPU_Compute_Pressure( system, control,
                workspace, data, mpi_data );
        GPU_Scale_Box( system, control,
                workspace, data, mpi_data );

        verlet_part1_done = FALSE;
        gpu_copy = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}
