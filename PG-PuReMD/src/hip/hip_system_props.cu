#include "hip/hip_runtime.h"

#include "hip_system_props.h"

#include "hip_copy.h"
#include "hip_helpers.h"
#include "hip_random.h"
#include "hip_reduction.h"
#include "hip_utils.h"
#include "hip_vector.h"

#include "../comm_tools.h"
#include "../tool_box.h"
#include "../vector.h"

#include <hipcub/config.hpp>
#include <hipcub/block/block_reduce.hpp>


/* mask used to determine which threads within a warp participate in operations */
#define FULL_WARP_MASK (0xFFFFFFFF)


GPU_GLOBAL void k_center_of_mass_xcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *xcm_g, size_t n )
{
    extern __shared__ hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage temp_block[];
    unsigned int i;
    rvec xcm;
    real m;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_Scale( xcm, m, atoms[i].x );
    }
    else
    {
        rvec_MakeZero( xcm );
    }

    xcm[0] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xcm[0]);
    __syncthreads( );
    xcm[1] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xcm[1]);
    __syncthreads( );
    xcm[2] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xcm[2]);

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
#if !defined(GPU_ACCUM_ATOMIC)
        rvec_Copy( xcm_g[blockIdx.x], xcm );
#else
        atomicAdd( (double *) &xcm_g[0][0], (double) xcm[0] );
        atomicAdd( (double *) &xcm_g[0][1], (double) xcm[1] );
        atomicAdd( (double *) &xcm_g[0][2], (double) xcm[2] );
#endif
    }
}


GPU_GLOBAL void k_center_of_mass_vcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *vcm_g, size_t n )
{
    extern __shared__ hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage temp_block[];
    unsigned int i;
    real m;
    rvec vcm;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_Scale( vcm, m, atoms[i].v );
    }
    else
    {
        rvec_MakeZero( vcm );
    }

    vcm[0] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(vcm[0]);
    __syncthreads( );
    vcm[1] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(vcm[1]);
    __syncthreads( );
    vcm[2] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(vcm[2]);

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
#if !defined(GPU_ACCUM_ATOMIC)
        rvec_Copy( vcm_g[blockIdx.x], vcm );
#else
        atomicAdd( (double *) &vcm_g[0][0], (double) vcm[0] );
        atomicAdd( (double *) &vcm_g[0][1], (double) vcm[1] );
        atomicAdd( (double *) &vcm_g[0][2], (double) vcm[2] );
#endif
    }
}


GPU_GLOBAL void k_center_of_mass_amcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *amcm_g, size_t n )
{
    extern __shared__ hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage temp_block[];
    unsigned int i;
    real m;
    rvec amcm;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_Cross( amcm, atoms[i].x, atoms [i].v );
        rvec_Scale( amcm, m, amcm );
    }
    else
    {
        rvec_MakeZero( amcm );
    }

    amcm[0] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(amcm[0]);
    __syncthreads( );
    amcm[1] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(amcm[1]);
    __syncthreads( );
    amcm[2] = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(amcm[2]);

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
#if !defined(GPU_ACCUM_ATOMIC)
        rvec_Copy( amcm_g[blockIdx.x], amcm );
#else
        atomicAdd( (double *) &amcm_g[0][0], (double) amcm[0] );
        atomicAdd( (double *) &amcm_g[0][1], (double) amcm[1] );
        atomicAdd( (double *) &amcm_g[0][2], (double) amcm[2] );
#endif
    }
}


GPU_GLOBAL void k_compute_inertial_tensor_part1( single_body_parameters *sbp,
        reax_atom *atoms, real *t_g, real xcm0, real xcm1, real xcm2, size_t n )
{
    extern __shared__ hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage temp_block[];
    unsigned int i;
    real xx, xy, xz, yy, yz, zz, m;
    rvec diff, xcm;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    xcm[0] = xcm0;
    xcm[1] = xcm1;
    xcm[2] = xcm2;

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1.0, atoms[i].x, -1.0, xcm );
        xx = diff[0] * diff[0] * m;
        xy = diff[0] * diff[1] * m;
        xz = diff[0] * diff[2] * m;
        yy = diff[1] * diff[1] * m;
        yz = diff[1] * diff[2] * m;
        zz = diff[2] * diff[2] * m;
    }
    else
    {
        xx = 0.0;
        xy = 0.0;
        xz = 0.0;
        yy = 0.0;
        yz = 0.0;
        zz = 0.0;
    }

    xx = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xx);
    __syncthreads( );
    xy = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xy);
    __syncthreads( );
    xz = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xz);
    __syncthreads( );
    yy = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(yy);
    __syncthreads( );
    yz = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(yz);
    __syncthreads( );
    zz = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(zz);

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        t_g[ blockIdx.x * 6 ] = xx;
        t_g[ blockIdx.x * 6 + 1 ] = xy;
        t_g[ blockIdx.x * 6 + 2 ] = xz;
        t_g[ blockIdx.x * 6 + 3 ] = yy;
        t_g[ blockIdx.x * 6 + 4 ] = yz;
        t_g[ blockIdx.x * 6 + 5 ] = zz;
    }
}


GPU_GLOBAL void k_compute_inertial_tensor_part2( real *input, real *output, size_t n )
{
    extern __shared__ hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage temp_block[];
    unsigned int i;
    real xx, xy, xz, yy, yz, zz;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        xx = input[ i * 6 ];
        xy = input[ i * 6 + 1 ];
        xz = input[ i * 6 + 2 ];
        yy = input[ i * 6 + 3 ];
        yz = input[ i * 6 + 4 ];
        zz = input[ i * 6 + 5 ];
    }
    else
    {
        xx = 0.0;
        xy = 0.0;
        xz = 0.0;
        yy = 0.0;
        yz = 0.0;
        zz = 0.0;
    }

    xx = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xx);
    __syncthreads( );
    xy = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xy);
    __syncthreads( );
    xz = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(xz);
    __syncthreads( );
    yy = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(yy);
    __syncthreads( );
    yz = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(yz);
    __syncthreads( );
    zz = hipcub::BlockReduce<double, DEF_BLOCK_SIZE>(*temp_block).Sum(zz);

    if ( threadIdx.x == 0 )
    {
        output[0] = xx;
        output[1] = xy;
        output[2] = xz;
        output[3] = yy;
        output[4] = yz;
        output[5] = zz;
    }
}


/* Copy the atom masses to a contigous array in global memory
 * for later reduction (sum) */
GPU_GLOBAL void k_compute_total_mass( single_body_parameters *sbp, reax_atom *my_atoms, 
        real *M_g, int n )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    M_g[i] = sbp[ my_atoms[i].type ].mass;
}


GPU_GLOBAL void k_compute_kinetic_energy( single_body_parameters *sbp, reax_atom *my_atoms, 
        real *e_kin_g, int n )
{
    unsigned int i;
    rvec p;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    rvec_Scale( p, sbp[ my_atoms[i].type ].mass, my_atoms[i].v );
    e_kin_g[i] = 0.5 * rvec_Dot( p, my_atoms[i].v );
}


/* Generate zero atom velocities */
GPU_GLOBAL void k_atom_velocities_zero( reax_atom *my_atoms, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    rvec_MakeZero( my_atoms[i].v );
}


/* Generate random atom velocities according
 * to the prescribed initial temperature */
GPU_GLOBAL void k_atom_velocities_random( single_body_parameters *sbp,
        reax_atom *my_atoms, real T, int n )
{
    int i;
    real m, scale, norm;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    hip_rvec_Random( my_atoms[i].v );

    norm = rvec_Norm_Sqr( my_atoms[i].v );
    m = sbp[ my_atoms[i].type ].mass;
    scale = SQRT( m * norm / (3.0 * K_B * T) );

    rvec_Scale( my_atoms[i].v, 1.0 / scale, my_atoms[i].v );
}


GPU_GLOBAL void k_compute_pressure( reax_atom *my_atoms, simulation_box *big_box,
        rvec *int_press, int n )
{
    reax_atom *p_atom;
    rvec tx;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    p_atom = &my_atoms[i];
    rvec_MakeZero( int_press[i] );

    /* transform x into unit box coordinates, store in tx */
    Transform_to_UnitBox( p_atom->x, big_box, 1, tx );

    /* this atom's contribution to internal pressure */
    rvec_Multiply( int_press[i], p_atom->f, tx );
}


static void Hip_Compute_Momentum( reax_system *system, control_params *control,
        storage *workspace, rvec xcm, rvec vcm, rvec amcm )
{
    rvec *spad;

    sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
#if !defined(GPU_ACCUM_ATOMIC)
            sizeof(rvec) * (control->blocks + 1),
#else
            sizeof(rvec),
#endif
            __FILE__, __LINE__ );
    spad = (rvec *) workspace->scratch[0];

    // xcm
    k_center_of_mass_xcm <<< control->blocks, control->block_size,
                                sizeof(hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                                control->hip_streams[0] >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad, system->n );
    hipCheckError( );
    
#if !defined(GPU_ACCUM_ATOMIC)
    Hip_Reduction_Sum( spad, &spad[control->blocks], control->blocks,
            0, control->hip_streams[0] );
#endif

    sHipMemcpyAsync( xcm,
#if !defined(GPU_ACCUM_ATOMIC)
            &spad[control->blocks],
#else
            spad,
#endif
            sizeof(rvec), hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    
    // vcm
    k_center_of_mass_vcm <<< control->blocks, control->block_size,
                         sizeof(hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                         control->hip_streams[0] >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad, system->n );
    hipCheckError( );
    
#if !defined(GPU_ACCUM_ATOMIC)
    Hip_Reduction_Sum( spad, &spad[control->blocks], control->blocks,
            0 control->hip_streams[0] );
#endif

    sHipMemcpyAsync( vcm,
#if !defined(GPU_ACCUM_ATOMIC)
            &spad[control->blocks],
#else
            spad,
#endif
            sizeof(rvec), hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    
    // amcm
    k_center_of_mass_amcm <<< control->blocks, control->block_size,
                          sizeof(hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                          control->hip_streams[0] >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad, system->n );
    hipCheckError( );
    
#if !defined(GPU_ACCUM_ATOMIC)
    Hip_Reduction_Sum( spad, &spad[control->blocks], control->blocks,
            0, control->hip_streams[0] );
#endif

    sHipMemcpyAsync( amcm, 
#if !defined(GPU_ACCUM_ATOMIC)
            &spad[control->blocks],
#else
            spad,
#endif
            sizeof(rvec), hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
}


static void Hip_Compute_Inertial_Tensor( reax_system *system, control_params *control,
        storage *workspace, real *t, rvec my_xcm )
{
    real *spad;

    sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            sizeof(real) * 6 * (control->blocks + 1), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[0];
    sHipMemsetAsync( spad, 0, sizeof(real) * 6 * (control->blocks + 1),
            control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    k_compute_inertial_tensor_part1 <<< control->blocks, control->block_size,
                                sizeof(hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                                control->hip_streams[0] >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad,
          my_xcm[0], my_xcm[1], my_xcm[2], system->n );
    hipCheckError( );

    k_compute_inertial_tensor_part2 <<< 1, control->blocks_pow_2,
                                     sizeof(hipcub::BlockReduce<double, DEF_BLOCK_SIZE>::TempStorage),
                                     control->hip_streams[0] >>>
        ( spad, &spad[6 * control->blocks], control->blocks );
    hipCheckError( );

    sHipMemcpyAsync( t, &spad[6 * control->blocks],
        sizeof(real) * 6, hipMemcpyDeviceToHost,
        control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
}


/* Initialize atom velocities according to the prescribed parameters */
void Hip_Generate_Initial_Velocities( reax_system *system,
        control_params *control, real T )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    if ( T <= 0.1 || control->random_vel == FALSE )
    {
        /* warnings if conflicts between initial temperature and control file parameter */
        if ( control->random_vel == TRUE )
        {
            fprintf( stderr, "[ERROR] conflicting control file parameters\n" );
            fprintf( stderr, "[INFO] random_vel = 1 and small initial temperature (t_init = %f)\n", T );
            fprintf( stderr, "[INFO] set random_vel = 0 to resolve this (atom initial velocites set to zero)\n" );
            MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
        }
        else if ( T > 0.1 )
        {
            fprintf( stderr, "[ERROR] conflicting control file paramters\n" );
            fprintf( stderr, "[INFO] random_vel = 0 and large initial temperature (t_init = %f)\n", T );
            fprintf( stderr, "[INFO] set random_vel = 1 to resolve this (random atom initial velocites according to t_init)\n" );
            MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
        }

        k_atom_velocities_zero <<< blocks, DEF_BLOCK_SIZE, 0, control->hip_streams[0] >>>
            ( system->d_my_atoms, system->n );
    }
    else
    {
        if ( T <= 0.0 )
        {
            fprintf( stderr, "[ERROR] random atom initial velocities specified with invalid temperature (%f). Terminating...\n",
                  T );
            MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
        }

        Hip_Randomize( );

        k_atom_velocities_random <<< blocks, DEF_BLOCK_SIZE, 0, control->hip_streams[0] >>>
            ( system->reax_param.d_sbp, system->d_my_atoms, T, system->n );
    }
}


extern "C" void Hip_Compute_Kinetic_Energy( reax_system *system,
        control_params *control, storage *workspace, simulation_data *data,
        MPI_Comm comm )
{
    int ret;
    real *kinetic_energy;

    sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            sizeof(real) * (system->n + 1), __FILE__, __LINE__ );
    kinetic_energy = (real *) workspace->scratch[0];

    k_compute_kinetic_energy <<< control->blocks, control->block_size, 0, control->hip_streams[0] >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, kinetic_energy, system->n );
    hipCheckError( );

    /* note: above kernel sums the kinetic energy contribution within blocks,
     * and this call finishes the global reduction across all blocks */
    Hip_Reduction_Sum( kinetic_energy, &kinetic_energy[system->n], system->n,
            0, control->hip_streams[0] );

    sHipMemcpyAsync( &data->my_en.e_kin, &kinetic_energy[system->n],
            sizeof(real), hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    ret = MPI_Allreduce( &data->my_en.e_kin, &data->sys_en.e_kin,
            1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    data->therm.T = (2.0 * data->sys_en.e_kin) / (data->N_f * K_B);

    /* avoid T being an absolute zero, might cause F.P.E! */
    if ( FABS(data->therm.T) < ALMOST_ZERO )
    {
        data->therm.T = ALMOST_ZERO;
    }
}


void Hip_Compute_Total_Mass( reax_system *system, control_params *control,
        storage *workspace, simulation_data *data, MPI_Comm comm  )
{
    int ret;
    real my_M, *spad;

    sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            sizeof(real) * (system->n + 1), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[0];

    k_compute_total_mass <<< control->blocks, control->block_size, 0, control->hip_streams[0]  >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad, system->n );
    hipCheckError( );

    Hip_Reduction_Sum( spad, &spad[system->n], system->n, 0, control->hip_streams[0] );

    sHipMemcpyAsync( &my_M, &spad[system->n], sizeof(real), 
            hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    ret = MPI_Allreduce( &my_M, &data->M, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    data->inv_M = 1.0 / data->M;
}


extern "C" void Hip_Compute_Center_of_Mass( reax_system *system,
        control_params *control, storage *workspace, simulation_data *data,
        mpi_datatypes *mpi_data, MPI_Comm comm )
{
    int ret;
    real det; //xx, xy, xz, yy, yz, zz;
    real tmp_mat[6], tot_mat[6];
    rvec my_xcm, my_vcm, my_amcm, my_avcm;
    rvec tvec;
    rtensor mat, inv;

    rvec_MakeZero( my_xcm );  // position of CoM
    rvec_MakeZero( my_vcm );  // velocity of CoM
    rvec_MakeZero( my_amcm ); // angular momentum of CoM
    rvec_MakeZero( my_avcm ); // angular velocity of CoM

    /* Compute the position, vel. and ang. momentum about the center of mass */
    Hip_Compute_Momentum( system, control, workspace, my_xcm, my_vcm, my_amcm );

    ret = MPI_Allreduce( my_xcm, data->xcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Allreduce( my_vcm, data->vcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Allreduce( my_amcm, data->amcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    rvec_Scale( data->xcm, data->inv_M, data->xcm );
    rvec_Scale( data->vcm, data->inv_M, data->vcm );
    rvec_Cross( tvec, data->xcm, data->vcm );
    rvec_ScaledAdd( data->amcm, -data->M, tvec );
    data->etran_cm = 0.5 * data->M * rvec_Norm_Sqr( data->vcm );

    /* Calculate and then invert the inertial tensor */
    Hip_Compute_Inertial_Tensor( system, control, workspace, tmp_mat, data->xcm );

    ret = MPI_Reduce( tmp_mat, tot_mat, 6, MPI_DOUBLE, MPI_SUM, MASTER_NODE, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    if ( system->my_rank == MASTER_NODE )
    {
        mat[0][0] = tot_mat[3] + tot_mat[5];  // yy + zz;
        mat[0][1] = mat[1][0] = -tot_mat[1];  // -xy;
        mat[0][2] = mat[2][0] = -tot_mat[2];  // -xz;
        mat[1][1] = tot_mat[0] + tot_mat[5];  // xx + zz;
        mat[2][1] = mat[1][2] = -tot_mat[4];  // -yz;
        mat[2][2] = tot_mat[0] + tot_mat[3];  // xx + yy;

        /* invert the inertial tensor */
        det = ( mat[0][0] * mat[1][1] * mat[2][2] +
                mat[0][1] * mat[1][2] * mat[2][0] +
                mat[0][2] * mat[1][0] * mat[2][1] ) -
              ( mat[0][0] * mat[1][2] * mat[2][1] +
                mat[0][1] * mat[1][0] * mat[2][2] +
                mat[0][2] * mat[1][1] * mat[2][0] );

        inv[0][0] = mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1];
        inv[0][1] = mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2];
        inv[0][2] = mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1];
        inv[1][0] = mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2];
        inv[1][1] = mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0];
        inv[1][2] = mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2];
        inv[2][0] = mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1];
        inv[2][1] = mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1];
        inv[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

        if ( det > ALMOST_ZERO )
        {
            rtensor_Scale( inv, 1.0 / det, inv );
        }
        else
        {
            rtensor_MakeZero( inv );
        }

        /* Compute the angular velocity about the centre of mass */
        rtensor_MatVec( data->avcm, inv, data->amcm );
    }

    ret = MPI_Bcast( data->avcm, 3, MPI_DOUBLE, MASTER_NODE, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* Compute the rotational energy */
    data->erot_cm = 0.5 * E_CONV * rvec_Dot( data->avcm, data->amcm );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "xcm:  %24.15e %24.15e %24.15e\n",
             data->xcm[0], data->xcm[1], data->xcm[2] );
    fprintf( stderr, "vcm:  %24.15e %24.15e %24.15e\n",
             data->vcm[0], data->vcm[1], data->vcm[2] );
    fprintf( stderr, "amcm: %24.15e %24.15e %24.15e\n",
             data->amcm[0], data->amcm[1], data->amcm[2] );
    fprintf( stderr, "mat:  %f %f %f\n     %f %f %f\n     %f %f %f\n",
       mat[0][0], mat[0][1], mat[0][2],
       mat[1][0], mat[1][1], mat[1][2],
       mat[2][0], mat[2][1], mat[2][2] );
    fprintf( stderr, "inv:  %g %g %g\n     %g %g %g\n     %g %g %g\n",
       inv[0][0], inv[0][1], inv[0][2],
       inv[1][0], inv[1][1], inv[1][2],
       inv[2][0], inv[2][1], inv[2][2] );
    fprintf( stderr, "avcm: %24.15e %24.15e %24.15e\n",
             data->avcm[0], data->avcm[1], data->avcm[2] );
#endif
}


/* IMPORTANT: This function assumes that current kinetic energy
 * the system is already computed
 *
 * IMPORTANT: In Klein's paper, it is stated that a dU/dV term needs
 *  to be added when there are long-range interactions or long-range
 *  corrections to short-range interactions present.
 *  We may want to add that for more accuracy.
 */
void Hip_Compute_Pressure( reax_system* system, control_params *control,
        storage *workspace, simulation_data* data, mpi_datatypes *mpi_data )
{
    int ret;
    rvec *spad_rvec, int_press;
    simulation_box *big_box;
    
    big_box = &system->big_box;

    /* 0: both int and ext, 1: ext only, 2: int only */
    if ( control->press_mode == 0 || control->press_mode == 2 )
    {
        sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
                sizeof(rvec) * (system->n + 1), __FILE__, __LINE__ );
        spad_rvec = (rvec *) workspace->scratch[0];

        k_compute_pressure <<< control->blocks, control->block_size, 0,
                           control->hip_streams[0] >>>
            ( system->d_my_atoms, system->d_big_box, spad_rvec,
              system->n );

        Hip_Reduction_Sum( spad_rvec, &spad_rvec[system->n], system->n,
                0, control->hip_streams[0] );

        sHipMemcpyAsync( &int_press, &spad_rvec[system->n],
                sizeof(rvec), hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
        hipStreamSynchronize( control->hip_streams[0] );
    }

    /* sum up internal and external pressure */
    ret = MPI_Allreduce( int_press, data->int_press,
            3, MPI_DOUBLE, MPI_SUM, mpi_data->comm_mesh3D );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Allreduce( data->my_ext_press, data->ext_press,
            3, MPI_DOUBLE, MPI_SUM, mpi_data->comm_mesh3D );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* kinetic contribution */
    data->kin_press = 2.0 * (E_CONV * data->sys_en.e_kin)
        / (3.0 * big_box->V * P_CONV);

    /* Calculate total pressure in each direction */
    data->tot_press[0] = data->kin_press -
        (( data->int_press[0] + data->ext_press[0] ) /
         ( big_box->box_norms[1] * big_box->box_norms[2] * P_CONV ));

    data->tot_press[1] = data->kin_press -
        (( data->int_press[1] + data->ext_press[1] ) /
         ( big_box->box_norms[0] * big_box->box_norms[2] * P_CONV ));

    data->tot_press[2] = data->kin_press -
        (( data->int_press[2] + data->ext_press[2] ) /
         ( big_box->box_norms[0] * big_box->box_norms[1] * P_CONV ));

    /* Average pressure for the whole box */
    data->iso_bar.P =
        ( data->tot_press[0] + data->tot_press[1] + data->tot_press[2] ) / 3.0;
}
