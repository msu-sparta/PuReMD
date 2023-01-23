
#include "hip_integrate.h"

#include "hip_allocate.h"
#include "hip_box.h"
#include "hip_environment.h"
#include "hip_forces.h"
#include "hip_integrate.h"
#include "hip_copy.h"
#include "hip_neighbors.h"
#include "hip_reduction.h"
#include "hip_reset_tools.h"
#include "hip_system_props.h"
#include "hip_utils.h"

#include "../comm_tools.h"
#include "../grid.h"
#if defined(DEBUG_FOCUS)
  #include "../tool_box.h"
#endif
#include "../vector.h"


GPU_GLOBAL void k_velocity_verlet_part1( reax_atom *my_atoms, 
        single_body_parameters *sbp, real dt, int n )
{
    int i;
    real inv_m;
    rvec dx;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* velocity verlet, 1st part */
    atom = &my_atoms[i];
    inv_m = 1.0 / sbp[atom->type].mass;

    /* Compute x(t + dt) */
    rvec_ScaledSum( dx, dt, atom->v, -0.5 * F_CONV * inv_m * SQR(dt), atom->f );
    rvec_Add( atom->x, dx );

    /* Compute v(t + dt/2) */
    rvec_ScaledAdd( atom->v, -0.5 * F_CONV * inv_m * dt, atom->f );
}


GPU_GLOBAL void k_velocity_verlet_part2( reax_atom *my_atoms, 
        single_body_parameters *sbp, real dt, int n )
{
    int i;
    reax_atom *atom;
    real inv_m;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* velocity verlet, 2nd part */
    atom = &my_atoms[i];
    inv_m = 1.0 / sbp[atom->type].mass;

    /* Compute v(t + dt) */
    rvec_ScaledAdd( atom->v, -0.5 * dt * F_CONV * inv_m, atom->f );
}


GPU_GLOBAL void k_velocity_verlet_nose_hoover_nvt( reax_atom *my_atoms, 
        single_body_parameters *sbp, real dt, int n )
{
    int i;
    real inv_m;
    rvec dx;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* velocity verlet, 1st part */
    atom = &my_atoms[i];
    inv_m = 1.0 / sbp[atom->type].mass;

    rvec_ScaledSum( dx, dt, atom->v, -0.5 * F_CONV * inv_m * SQR(dt), atom->f );
    rvec_Add( atom->x, dx );
    rvec_Copy( atom->f_old, atom->f );
}

GPU_GLOBAL void k_velocity_verlet_nose_hoover_nvt( reax_atom *my_atoms, rvec * v_const,
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
    atom = &my_atoms[i];
    inv_m = 1.0 / sbp[atom->type].mass;

    /* Compute v(t + dt) */
    rvec_Scale( v_const[i], 1.0 - 0.5 * dt * v_xi, atom->v );
    rvec_ScaledAdd( v_const[i], 0.5 * dt * inv_m * -F_CONV, atom->f_old );
    rvec_ScaledAdd( v_const[i], 0.5 * dt * inv_m * -F_CONV, atom->f );
}


GPU_GLOBAL void k_velocity_verlet_nose_hoover_nvt_part3( reax_atom *my_atoms, rvec *v_const,
        single_body_parameters *sbp, real dt, real v_xi_old, real * my_ekin, int n )
{
    int i;
    real coef_v;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    atom = &my_atoms[i];

    coef_v = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
    rvec_Scale( atom->v, coef_v, v_const[i] );
    my_ekin[i] = 0.5 * sbp[atom->type].mass * rvec_Dot( atom->v, atom->v );
}


GPU_GLOBAL void k_scale_velocites_berendsen_nvt( reax_atom *my_atoms, real lambda, int n )
{
    reax_atom *atom;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* Scale velocities and positions at t+dt */
    atom = &my_atoms[i];
    rvec_Scale( atom->v, lambda, atom->v );
}


GPU_GLOBAL void k_scale_velocities_npt( reax_atom *my_atoms, real lambda,
        real mu0, real mu1, real mu2, int n )
{
    int i;
    reax_atom *atom;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
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


static void Velocity_Verlet_Part1( reax_system *system, control_params *control, real dt )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_velocity_verlet_part1 <<< blocks, DEF_BLOCK_SIZE, 0, control->hip_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, dt, system->n );
    hipCheckError( );
}


static void Velocity_Verlet_Part2( reax_system *system, control_params *control, real dt )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_velocity_verlet_part2 <<< blocks, DEF_BLOCK_SIZE, 0, control->hip_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, dt, system->n );
    hipCheckError( );
}


static void Velocity_Verlet_Nose_Hoover_NVT_Part1( reax_system *system,
        control_params *control, real dt )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_velocity_verlet_nose_hoover_nvt <<< blocks, DEF_BLOCK_SIZE, 0,
                                      control->hip_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, dt, system->n );
    hipCheckError( );
}


static void Velocity_Verlet_Nose_Hoover_NVT_Part2( reax_system *system, control_params *control,
        storage *workspace, real dt, real v_xi )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_velocity_verlet_nose_hoover_nvt <<< blocks, DEF_BLOCK_SIZE, 0, control->hip_streams[0] >>>
        ( system->d_my_atoms, workspace->v_const,
          system->reax_param.d_sbp, dt, v_xi, system->n );
    hipCheckError( );
}


static real Velocity_Verlet_Nose_Hoover_NVT_Part3( reax_system *system,
        control_params *control, storage *workspace, real dt, real v_xi_old,
        real *d_my_ekin, real *d_total_my_ekin )
{
    int blocks;
    real my_ekin;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_velocity_verlet_nose_hoover_nvt_part3 <<< blocks, DEF_BLOCK_SIZE, 0, control->hip_streams[0] >>>
        ( system->d_my_atoms, workspace->v_const, system->reax_param.d_sbp,
          dt, v_xi_old, d_my_ekin, system->n );
    hipCheckError( );

    Hip_Reduction_Sum( d_my_ekin, d_total_my_ekin, system->n, 0, control->hip_streams[0] );

    sHipMemcpyAsync( &my_ekin, d_total_my_ekin, sizeof(real), 
            hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    return my_ekin;
}


static void Hip_Scale_Velocities_Berendsen_NVT( reax_system *system,
        control_params * control, real lambda )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_scale_velocites_berendsen_nvt <<< blocks, DEF_BLOCK_SIZE,
                                    0, control->hip_streams[0] >>>
        ( system->d_my_atoms, lambda, system->n );
    hipCheckError( );
}


void Hip_Scale_Velocities_NPT( reax_system *system, control_params * control,
        real lambda, rvec mu )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE
        + ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_scale_velocities_npt <<< blocks, DEF_BLOCK_SIZE, 0, control->hip_streams[0] >>>
        ( system->d_my_atoms, lambda, mu[0], mu[1], mu[2], system->n );
    hipCheckError( );
}


int Hip_Velocity_Verlet_NVE( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE, hip_copy = FALSE;
    real dt;

    ret = SUCCESS;
    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE )
    {
        Velocity_Verlet_Part1( system, control, dt );

        Hip_Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        Hip_Copy_Atoms_Device_to_Host( system, control );
        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

#if defined(GPU_DEVICE_PACK)
        if ( renbr == TRUE )
        {
            //TODO: remove once Comm_Atoms ported
            Hip_Copy_MPI_Data_Host_to_Device( control, mpi_data );
        }
#endif

        Hip_Init_Block_Sizes( system, control );

        verlet_part1_done = TRUE;
    }

    Hip_Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    if ( hip_copy == FALSE )
    {
        Hip_Copy_Atoms_Host_to_Device( system, control );

        if ( renbr == TRUE )
        {
            Hip_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );
        }

        Hip_Reset( system, control, data, workspace, lists );

        hip_copy = TRUE;
    }

    if ( renbr == TRUE && gen_nbr_list == FALSE )
    {
        ret = Hip_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        if ( ret == SUCCESS )
        {
            gen_nbr_list = TRUE;
        }
        else
        {
            Hip_Estimate_Num_Neighbors( system, control, data );
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Hip_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        Velocity_Verlet_Part2( system, control, dt );

        verlet_part1_done = FALSE;
        hip_copy = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


int Hip_Velocity_Verlet_Nose_Hoover_NVT_Klein( reax_system* system,
        control_params* control, simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control, mpi_datatypes *mpi_data )
{
    int itr, steps, renbr, ret, ret_mpi;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE, hip_copy = FALSE;
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

    if ( verlet_part1_done == FALSE )
    {
        Velocity_Verlet_Nose_Hoover_NVT_Part1( system, control, dt );
    
        /* Compute xi(t + dt) */
        therm->xi += therm->v_xi * dt + 0.5 * dt_sqr * therm->G_xi;

        Hip_Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        if ( renbr == TRUE )
        {
            Update_Grid( system, control, MPI_COMM_WORLD );
        }

        Hip_Copy_Atoms_Device_to_Host( system, control );
        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

#if defined(GPU_DEVICE_PACK)
        if ( renbr == TRUE )
        {
            //TODO: remove once Comm_Atoms ported
            Hip_Copy_MPI_Data_Host_to_Device( control, mpi_data );
        }
#endif

        Hip_Init_Block_Sizes( system, control );

        verlet_part1_done = TRUE;
    }

    Hip_Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    if ( hip_copy == FALSE )
    {
        Hip_Copy_Atoms_Host_to_Device( system, control );

        if ( renbr == TRUE )
        {
            Hip_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );
        }

        Hip_Reset( system, control, data, workspace, lists );

        hip_copy = TRUE;
    }

    if ( renbr == TRUE && gen_nbr_list == FALSE )
    {
        ret = Hip_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        if ( ret == SUCCESS )
        {
            gen_nbr_list = TRUE;
        }
        else
        {
            Hip_Estimate_Num_Neighbors( system, control, data );
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Hip_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        /* Compute iteration constants for each atom's velocity */
        Velocity_Verlet_Nose_Hoover_NVT_Part2( system, control,
                workspace->d_workspace, dt, therm->v_xi );
    
        v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
        my_ekin = G_xi_new = v_xi_old = 0;
        itr = 0;

        sHipMalloc( (void **) &d_my_ekin, sizeof(real) * system->n,
                __FILE__, __LINE__ );
        sHipMalloc( (void **) &d_total_my_ekin, sizeof(real),
                __FILE__, __LINE__ );

        do
        {
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
        }
        while ( FABS(v_xi_new - v_xi_old) > 1e-5 );
        therm->v_xi_old = therm->v_xi;
        therm->v_xi = v_xi_new;
        therm->G_xi = G_xi_new;

        sHipFree( d_total_my_ekin, __FILE__, __LINE__ );
        sHipFree( d_my_ekin, __FILE__, __LINE__ );

        verlet_part1_done = FALSE;
        hip_copy = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


/* uses Berendsen-type coupling for both T and P.
   All box dimensions are scaled by the same amount,
   there is no change in the angles between axes. */
int Hip_Velocity_Verlet_Berendsen_NVT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE, hip_copy = FALSE;
    real dt, lambda;

    ret = SUCCESS;
    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE )
    {
        Velocity_Verlet_Part1( system, control, dt );

        Hip_Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        if ( renbr == TRUE )
        {
            Update_Grid( system, control, MPI_COMM_WORLD );
        }

        Hip_Copy_Atoms_Device_to_Host( system, control );
        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

#if defined(GPU_DEVICE_PACK)
        if ( renbr == TRUE )
        {
            //TODO: remove once Comm_Atoms ported
            Hip_Copy_MPI_Data_Host_to_Device( control, mpi_data );
        }
#endif

        Hip_Init_Block_Sizes( system, control );

        verlet_part1_done = TRUE;
    }

    Hip_Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    if ( hip_copy == FALSE )
    {
        Hip_Copy_Atoms_Host_to_Device( system, control );

        if ( renbr == TRUE )
        {
            Hip_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );
        }

        Hip_Reset( system, control, data, workspace, lists );

        hip_copy = TRUE;
    }

    if ( renbr == TRUE && gen_nbr_list == FALSE )
    {
        ret = Hip_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        if ( ret == SUCCESS )
        {
            gen_nbr_list = TRUE;
        }
        else
        {
            Hip_Estimate_Num_Neighbors( system, control, data );
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Hip_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        Velocity_Verlet_Part2( system, control, dt );

        /* temperature scaler */
        Hip_Compute_Kinetic_Energy( system, control, workspace,
                data, mpi_data->comm_mesh3D );

        lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
        if ( lambda < MIN_dT )
        {
            lambda = MIN_dT;
        }
        else if ( lambda > MAX_dT )
        {
            lambda = MAX_dT;
        }
        lambda = SQRT( lambda );

        /* Scale velocities and positions at t+dt */
        Hip_Scale_Velocities_Berendsen_NVT( system, control, lambda );

        Hip_Compute_Kinetic_Energy( system, control, workspace,
                data, mpi_data->comm_mesh3D );

        verlet_part1_done = FALSE;
        hip_copy = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


/* uses Berendsen-type coupling for both T and P.
 * All box dimensions are scaled by the same amount,
 * there is no change in the angles between axes. */
int Hip_Velocity_Verlet_Berendsen_NPT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE, hip_copy = FALSE;
    real dt;

    ret = SUCCESS;
    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE )
    {
        Velocity_Verlet_Part1( system, control, dt );

        Hip_Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        if ( renbr == TRUE )
        {
            Update_Grid( system, control, MPI_COMM_WORLD );
        }

        Hip_Copy_Atoms_Device_to_Host( system, control );
        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

#if defined(GPU_DEVICE_PACK)
        if ( renbr == TRUE )
        {
            //TODO: remove once Comm_Atoms ported
            Hip_Copy_MPI_Data_Host_to_Device( control, mpi_data );
        }
#endif

        Hip_Init_Block_Sizes( system, control );

        verlet_part1_done = TRUE;
    }

    Hip_Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    if ( hip_copy == FALSE )
    {
        Hip_Copy_Atoms_Host_to_Device( system, control );

        if ( renbr == TRUE )
        {
            Hip_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );
        }

        Hip_Reset( system, control, data, workspace, lists );

        hip_copy = TRUE;
    }

    if ( renbr == TRUE && gen_nbr_list == FALSE )
    {
        ret = Hip_Generate_Neighbor_Lists( system, control, data, workspace, lists );

        if ( ret == SUCCESS )
        {
            gen_nbr_list = TRUE;
        }
        else
        {
            Hip_Estimate_Num_Neighbors( system, control, data );
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Hip_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        Velocity_Verlet_Part2( system, control, dt );

        Hip_Compute_Kinetic_Energy( system, control,
                workspace, data, mpi_data->comm_mesh3D );
        Hip_Compute_Pressure( system, control,
                workspace, data, mpi_data );
        Hip_Scale_Box( system, control,
                workspace, data, mpi_data );

        verlet_part1_done = FALSE;
        hip_copy = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}
