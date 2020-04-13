
#include "cuda_system_props.h"

#include "cuda_copy.h"
#include "cuda_utils.h"
#include "cuda_random.h"
#include "cuda_reduction.h"
#include "cuda_vector.h"

#include "../tool_box.h"
#include "../vector.h"


/* mask used to determine which threads within a warp participate in operations */
#define FULL_MASK (0xFFFFFFFF)


CUDA_GLOBAL void k_center_of_mass_blocks_xcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *res_xcm, size_t n )
{
    extern __shared__ rvec my_xcm[];
    unsigned int i, index, mask;
    int offset;
    rvec xcm;
    real m;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_Scale( xcm, m, atoms[i].x );

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            xcm[0] += __shfl_down_sync( mask, xcm[0], offset );
            xcm[1] += __shfl_down_sync( mask, xcm[1], offset );
            xcm[2] += __shfl_down_sync( mask, xcm[2], offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            rvec_Copy( my_xcm[ threadIdx.x >> 5 ], xcm );
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            rvec_Add( my_xcm[threadIdx.x], my_xcm[index] );
        }

        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        rvec_Copy( res_xcm[blockIdx.x], my_xcm[0] );
    }
}


CUDA_GLOBAL void k_center_of_mass_blocks_vcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *res_vcm, size_t n )
{
    extern __shared__ rvec my_vcm[];
    unsigned int i, index, mask;
    int offset;
    real m;
    rvec vcm;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_Scale( vcm, m, atoms[i].v );

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            vcm[0] += __shfl_down_sync( mask, vcm[0], offset );
            vcm[1] += __shfl_down_sync( mask, vcm[1], offset );
            vcm[2] += __shfl_down_sync( mask, vcm[2], offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            rvec_Copy( my_vcm[ threadIdx.x >> 5 ], vcm );
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            rvec_Add( my_vcm[threadIdx.x], my_vcm[index] );
        }
        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        rvec_Copy( res_vcm[blockIdx.x], my_vcm[0] );
    }
}


CUDA_GLOBAL void k_center_of_mass_blocks_amcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *res_amcm, size_t n )
{
    extern __shared__ rvec my_amcm[];
    unsigned int i, index, mask;
    int offset;
    real m;
    rvec amcm, tmp;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_Cross( tmp, atoms[i].x, atoms [i].v );
        rvec_Scale( amcm, m, tmp );

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            amcm[0] += __shfl_down_sync( mask, amcm[0], offset );
            amcm[1] += __shfl_down_sync( mask, amcm[1], offset );
            amcm[2] += __shfl_down_sync( mask, amcm[2], offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            rvec_Copy( my_amcm[ threadIdx.x >> 5 ], amcm );
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            rvec_Add( my_amcm[threadIdx.x], my_amcm[index] );
        }
        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        rvec_Copy( res_amcm[blockIdx.x], my_amcm[0] );
    }
}


CUDA_GLOBAL void k_compute_center_mass_tensor( real *input, real *output, size_t n )
{
    extern __shared__ real xx[];
    extern __shared__ real xy[];
    extern __shared__ real xz[];
    extern __shared__ real yy[];
    extern __shared__ real yz[];
    extern __shared__ real zz[];
    unsigned int i, index, xx_i, xy_i, xz_i, yy_i, yz_i, zz_i;
    int offset;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    xx_i = threadIdx.x;
    xy_i = blockDim.x;
    xz_i = 2 * blockDim.x;
    yy_i = 3 * blockDim.x;
    yz_i = 4 * blockDim.x;
    zz_i = 5 * blockDim.x;

    xx[xx_i] = 0.0;
    xy[xy_i + threadIdx.x] = 0.0;
    xz[xz_i + threadIdx.x] = 0.0;
    yy[yy_i + threadIdx.x] = 0.0;
    yz[yz_i + threadIdx.x] = 0.0;
    zz[zz_i + threadIdx.x] = 0.0;

    if ( i < n )
    {
        xx[ xx_i ] = input[ threadIdx.x * 6 ];
        xy[ xy_i + threadIdx.x ] = input[ threadIdx.x * 6 + 1 ];
        xz[ xz_i + threadIdx.x ] = input[ threadIdx.x * 6 + 2 ];
        yy[ yy_i + threadIdx.x ] = input[ threadIdx.x * 6 + 3 ];
        yz[ yz_i + threadIdx.x ] = input[ threadIdx.x * 6 + 4 ];
        zz[ zz_i + threadIdx.x ] = input[ threadIdx.x * 6 + 5 ];
    }
    __syncthreads( );

    for ( offset = blockDim.x / 2; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            xx[ threadIdx.x ] += xx[ index ];
            xy[ xy_i + threadIdx.x ] += xy[ xy_i + index ];
            xz[ xz_i + threadIdx.x ] += xz[ xz_i + index ];
            yy[ yy_i + threadIdx.x ] += yy[ yy_i + index ];
            yz[ yz_i + threadIdx.x ] += yz[ yz_i + index ];
            zz[ zz_i + threadIdx.x ] += zz[ zz_i + index ];
        }
        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        output[0] = xx[0];
        output[1] = xy[xy_i];
        output[2] = xz[xz_i];
        output[3] = xz[yy_i];
        output[4] = xz[yz_i];
        output[5] = xz[zz_i];
    }
}


CUDA_GLOBAL void k_compute_center_mass_xx_xy( single_body_parameters *sbp,
        reax_atom *atoms, real *results, real xcm0, real xcm1, real xcm2,
        size_t n )
{
    extern __shared__ real my_results_xx[];
    extern __shared__ real my_results_xy[];
    unsigned int xy_i, i, index, mask;
    int offset;
    real xx, xy, m;
    rvec diff, xcm;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    xy_i = blockDim.x;
    xcm[0] = xcm0;
    xcm[1] = xcm1;
    xcm[2] = xcm2;

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1.0, atoms[i].x, -1.0, xcm );
        xx = diff[0] * diff[0] * m;
        xy = diff[0] * diff[1] * m;

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            xx += __shfl_down_sync( mask, xx, offset );
            xy += __shfl_down_sync( mask, xy, offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            my_results_xx[threadIdx.x >> 5] = xx;    
            my_results_xy[threadIdx.x >> 5] = xy;    
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            my_results_xx[ threadIdx.x ] += my_results_xx[ index ];
            my_results_xy[ xy_i + threadIdx.x ] += my_results_xy[ xy_i + index ];
        }
        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        results[ blockIdx.x * 6 ] = my_results_xx[ 0 ];
        results[ blockIdx.x * 6 + 1 ] = my_results_xy[ xy_i + 0 ];
    }
}


CUDA_GLOBAL void k_compute_center_mass_xz_yy( single_body_parameters *sbp,
        reax_atom *atoms, real *results, real xcm0, real xcm1, real xcm2,
        size_t n )
{
    extern __shared__ real my_results_xz[];
    extern __shared__ real my_results_yy[];
    unsigned int yy_i, i, index, mask;
    int offset;
    real xz, yy, m;
    rvec diff, xcm;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    yy_i = blockDim.x;
    xcm[0] = xcm0;
    xcm[1] = xcm1;
    xcm[2] = xcm2;

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1.0, atoms[i].x, -1.0, xcm );
        xz = diff[0] * diff[2] * m;
        yy = diff[1] * diff[1] * m;

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            xz += __shfl_down_sync( mask, xz, offset );
            yy += __shfl_down_sync( mask, yy, offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            my_results_xz[threadIdx.x >> 5] = xz;    
            my_results_yy[threadIdx.x >> 5] = yy;    
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            my_results_xz[ threadIdx.x ] += my_results_xz[ index ];
            my_results_yy[ yy_i + threadIdx.x ] += my_results_yy[ yy_i + index ];
        }
        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        results[ blockIdx.x * 6 + 2 ] = my_results_xz[ 0 ];
        results[ blockIdx.x * 6 + 3 ] = my_results_yy[ yy_i + 0 ];
    }
}


CUDA_GLOBAL void k_compute_center_mass_yz_zz( single_body_parameters *sbp,
        reax_atom *atoms, real *results, real xcm0, real xcm1, real xcm2,
        size_t n )
{
    extern __shared__ real my_results_yz[];
    extern __shared__ real my_results_zz[];
    unsigned int i, zz_i, index, mask;
    int offset;
    real yz, zz, m;
    rvec diff, xcm;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    zz_i = blockDim.x;
    xcm[0] = xcm0;
    xcm[1] = xcm1;
    xcm[2] = xcm2;

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1.0, atoms[i].x, -1.0, xcm );
        yz = diff[1] * diff[2] * m;
        zz = diff[2] * diff[2] * m;

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            yz += __shfl_down_sync( mask, yz, offset );
            zz += __shfl_down_sync( mask, zz, offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            my_results_yz[threadIdx.x >> 5] = yz;    
            my_results_zz[threadIdx.x >> 5] = zz;    
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            my_results_yz[ threadIdx.x ] += my_results_yz[ index ];
            my_results_zz[ zz_i + threadIdx.x ] += my_results_zz[ zz_i + index ];
        }
        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        results[ blockIdx.x * 6 + 4 ] = my_results_yz[ 0 ];
        results[ blockIdx.x * 6 + 5 ] = my_results_zz[ zz_i + 0 ];
    }
}


CUDA_GLOBAL void k_compute_total_mass( single_body_parameters *sbp, reax_atom *my_atoms, 
        real *block_results, int n )
{
    extern __shared__ real M_s[];
    unsigned int i, mask;
    int offset;
    real M;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    if ( i < n )
    {
        M = sbp[ my_atoms[i].type ].mass;

        for ( offset = 16; offset > 0; offset /= 2 )
        {
            M += __shfl_down_sync( mask, M, offset );
        }

        if ( threadIdx.x % 32 == 0 )
        {
            M_s[threadIdx.x >> 5] = M;
        }
    }
    __syncthreads( );

    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            M_s[threadIdx.x] += M_s[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        block_results[blockIdx.x] = M_s[0];
    }
}


CUDA_GLOBAL void k_compute_kinetic_energy( single_body_parameters *sbp, reax_atom *my_atoms, 
        real *block_results, int n )
{
    extern __shared__ real e_kin_s[];
    unsigned int i, mask;
    int offset;
    real e_kin;
    rvec p;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    if ( i < n )
    {
        rvec_Scale( p, sbp[ my_atoms[i].type ].mass, my_atoms[ i ].v );
        e_kin = 0.5 * rvec_Dot( p, my_atoms[ i ].v );

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            e_kin += __shfl_down_sync( mask, e_kin, offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            e_kin_s[threadIdx.x >> 5] = e_kin;
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            e_kin_s[threadIdx.x] += e_kin_s[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        block_results[blockIdx.x] = e_kin_s[0];
    }
}


CUDA_GLOBAL void k_generate_initial_velocities( single_body_parameters *sbp, reax_atom *my_atoms, 
        real T, int n )
{
    int i;
    real m, scale, norm;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    if ( T <= 0.1 )
    {
        rvec_MakeZero( my_atoms[i].v );
    }
    else
    {
        cuda_rvec_Random( my_atoms[i].v );

        norm = rvec_Norm_Sqr( my_atoms[i].v );
        m = sbp[ my_atoms[i].type ].mass;
        scale = SQRT( m * norm / (3.0 * K_B * T) );

        rvec_Scale( my_atoms[i].v, 1. / scale, my_atoms[i].v );
    }
}


CUDA_GLOBAL void k_compute_pressure( reax_atom *my_atoms, simulation_box *big_box,
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


static void Cuda_Compute_Momentum( reax_system *system, control_params *control,
        storage *workspace, rvec xcm, rvec vcm, rvec amcm )
{
    rvec *spad;

    spad = (rvec *) workspace->scratch;

    // xcm
    cuda_memset( spad, 0, sizeof(rvec) * (control->blocks_pow_2 + 1),
            "Cuda_Compute_Momentum::tmp" );
    
    k_center_of_mass_blocks_xcm <<< control->blocks_pow_2, control->block_size,
                                sizeof(rvec) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<< 1, control->blocks_pow_2, sizeof(rvec) * control->blocks_pow_2 >>>
            ( spad, &spad[control->blocks_pow_2], control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    copy_host_device( xcm, &spad[control->blocks_pow_2],
            sizeof(rvec), cudaMemcpyDeviceToHost, "Cuda_Compute_Momentum::xcm" );
    
    // vcm
    cuda_memset( spad, 0, sizeof(rvec) * (control->blocks_pow_2 + 1),
            "Cuda_Compute_Momentum::tmp" );
    
    k_center_of_mass_blocks_vcm <<< control->blocks_pow_2, control->block_size,
                                sizeof(rvec) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<< 1, control->blocks_pow_2, sizeof(rvec) * control->blocks_pow_2 >>>
        ( spad, &spad[control->blocks_pow_2], control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    copy_host_device( vcm, &spad[control->blocks_pow_2], sizeof(rvec),
        cudaMemcpyDeviceToHost, "Cuda_Compute_Momentum::vcm" );
    
    // amcm
    cuda_memset( spad, 0,  sizeof (rvec) * (control->blocks_pow_2 + 1),
            "Cuda_Compute_Momentum::tmp");
    
    k_center_of_mass_blocks_amcm <<< control->blocks_pow_2, control->block_size,
                                 sizeof(rvec) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<< 1, control->blocks_pow_2, sizeof(rvec) * control->blocks_pow_2 >>>
        ( spad, &spad[control->blocks_pow_2], control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    copy_host_device( amcm, &spad[control->blocks_pow_2], sizeof(rvec),
        cudaMemcpyDeviceToHost, "Cuda_Compute_Momentum::amcm" );
}


static void Cuda_Compute_Inertial_Tensor( reax_system *system, control_params *control,
        storage *workspace, real *t, rvec my_xcm )
{
    real *spad;

    spad = (real *) workspace->scratch;
    cuda_memset( spad, 0, sizeof(real) * 6 * (control->blocks_pow_2 + 1),
            "Cuda_Compute_Intertial_Tensor::tmp" );

    k_compute_center_mass_xx_xy <<< control->blocks_pow_2, control->block_size,
                                sizeof(real) * 2 * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_compute_center_mass_xz_yy <<< control->blocks_pow_2, control->block_size,
                                sizeof(real) * 2 * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_compute_center_mass_yz_zz <<< control->blocks_pow_2, control->block_size,
                                sizeof(real) * 2 * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, spad,
          my_xcm[0], my_xcm[1], my_xcm[2], system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_compute_center_mass_tensor <<< 1, control->blocks_pow_2,
                              sizeof(real) * 6 * control->blocks_pow_2 >>>
        ( spad, &spad[control->blocks_pow_2 * 6], control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( t, &spad[6 * control->blocks_pow_2],
        sizeof(real) * 6, cudaMemcpyDeviceToHost,
        "Cuda_Compute_Intertial_Tensor::t" );
}


extern "C" void Cuda_Sync_Simulation_Data( simulation_data *data )
{
    Output_Sync_Simulation_Data( data, (simulation_data *)data->d_simulation_data );
}


void Cuda_Generate_Initial_Velocities( reax_system *system, real T )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    if ( T > 0.1 )
    {
        Cuda_Randomize( );
    }

    k_generate_initial_velocities <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, T, system->n );
}


void Cuda_Compute_Kinetic_Energy( reax_system* system, control_params *control,
        storage *workspace, simulation_data* data, MPI_Comm comm )
{
    real *block_energy;

    block_energy = (real *) workspace->scratch;
    data->my_en.e_kin = 0.0;

    cuda_memset( block_energy, 0, sizeof(real) * (control->blocks_pow_2 + 1), "kinetic_energy:tmp" );

    k_compute_kinetic_energy <<< control->blocks, control->block_size, sizeof(real) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, block_energy, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* note: above kernel sums the kinetic energy contribution within blocks,
     * and this call finishes the global reduction across all blocks */
    Cuda_Reduction_Sum( block_energy, &block_energy[control->blocks_pow_2], control->blocks_pow_2 );

    copy_host_device( &data->my_en.e_kin, &block_energy[control->blocks_pow_2],
            sizeof(real), cudaMemcpyDeviceToHost, "kinetic_energy:tmp" );

    MPI_Allreduce( &data->my_en.e_kin, &data->sys_en.e_kin,
            1, MPI_DOUBLE, MPI_SUM, comm );

    data->therm.T = (2.0 * data->sys_en.e_kin) / (data->N_f * K_B);

    /* avoid T being an absolute zero, might cause F.P.E! */
    if ( FABS(data->therm.T) < ALMOST_ZERO )
    {
        data->therm.T = ALMOST_ZERO;
    }
}


void Check_Energy( simulation_data* data )
{
    if ( IS_NAN_REAL(data->my_en.e_kin) )
    {
        fprintf( stderr, "[ERROR] NaN detected for kinetic energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }

    if ( IS_NAN_REAL(data->my_en.e_pot) )
    {
        fprintf( stderr, "[ERROR] NaN detected for potential energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }

    if ( IS_NAN_REAL(data->my_en.e_tot) )
    {
        fprintf( stderr, "[ERROR] NaN detected for total energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }
}


void Cuda_Compute_Total_Mass( reax_system *system, control_params *control,
        storage *workspace, simulation_data *data, MPI_Comm comm  )
{
    real tmp;
    real *block_mass;

    block_mass = (real * ) workspace->scratch;

    cuda_memset( block_mass, 0, sizeof(real) * (1 + control->blocks_pow_2),
            "Cuda_Compute_Total_Mass::block_mass" );

    k_compute_total_mass <<< control->blocks, control->block_size, sizeof(real) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, block_mass, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* note: above kernel sums the mass contribution within blocks,
     * and this call finishes the global reduction across all blocks */
    Cuda_Reduction_Sum( block_mass, &block_mass[control->blocks_pow_2], control->blocks_pow_2 );

    copy_host_device( &tmp, &block_mass[control->blocks_pow_2], sizeof(real), 
            cudaMemcpyDeviceToHost, "total_mass:tmp" );

    MPI_Allreduce( &tmp, &data->M, 1, MPI_DOUBLE, MPI_SUM, comm );

    data->inv_M = 1.0 / data->M;
}


void Cuda_Compute_Center_of_Mass( reax_system *system, control_params *control,
        storage *workspace, simulation_data *data, mpi_datatypes *mpi_data, MPI_Comm comm )
{
    int i;
    real det; //xx, xy, xz, yy, yz, zz;
    real tmp_mat[6], tot_mat[6];
    rvec my_xcm, my_vcm, my_amcm, my_avcm;
    rvec tvec;
    rtensor mat, inv;

    rvec_MakeZero( my_xcm );  // position of CoM
    rvec_MakeZero( my_vcm );  // velocity of CoM
    rvec_MakeZero( my_amcm ); // angular momentum of CoM
    rvec_MakeZero( my_avcm ); // angular velocity of CoM

    /* Compute the position, vel. and ang. momentum about the centre of mass */
    Cuda_Compute_Momentum( system, control, workspace, my_xcm, my_vcm, my_amcm );

    MPI_Allreduce( my_xcm, data->xcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( my_vcm, data->vcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( my_amcm, data->amcm, 3, MPI_DOUBLE, MPI_SUM, comm );

    rvec_Scale( data->xcm, data->inv_M, data->xcm );
    rvec_Scale( data->vcm, data->inv_M, data->vcm );
    rvec_Cross( tvec, data->xcm, data->vcm );
    rvec_ScaledAdd( data->amcm, -data->M, tvec );
    data->etran_cm = 0.5 * data->M * rvec_Norm_Sqr( data->vcm );

    /* Calculate and then invert the inertial tensor */
    for ( i = 0; i < 6; ++i )
    {
        tmp_mat[i] = 0.0;
    }

    Cuda_Compute_Inertial_Tensor( system, control, workspace, tmp_mat, my_xcm );

    MPI_Reduce( tmp_mat, tot_mat, 6, MPI_DOUBLE, MPI_SUM, MASTER_NODE, comm );

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

    MPI_Bcast( data->avcm, 3, MPI_DOUBLE, MASTER_NODE, comm );

    /* Compute the rotational energy */
    data->erot_cm = 0.5 * E_CONV * rvec_Dot( data->avcm, data->amcm );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "xcm:  %24.15e %24.15e %24.15e\n",
             data->xcm[0], data->xcm[1], data->xcm[2] );
    fprintf( stderr, "vcm:  %24.15e %24.15e %24.15e\n",
             data->vcm[0], data->vcm[1], data->vcm[2] );
    fprintf( stderr, "amcm: %24.15e %24.15e %24.15e\n",
             data->amcm[0], data->amcm[1], data->amcm[2] );
    /* fprintf( stderr, "mat:  %f %f %f\n     %f %f %f\n     %f %f %f\n",
       mat[0][0], mat[0][1], mat[0][2],
       mat[1][0], mat[1][1], mat[1][2],
       mat[2][0], mat[2][1], mat[2][2] );
       fprintf( stderr, "inv:  %g %g %g\n     %g %g %g\n     %g %g %g\n",
       inv[0][0], inv[0][1], inv[0][2],
       inv[1][0], inv[1][1], inv[1][2],
       inv[2][0], inv[2][1], inv[2][2] ); */
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
void Cuda_Compute_Pressure( reax_system* system, control_params *control,
        storage *workspace, simulation_data* data, mpi_datatypes *mpi_data )
{
    int blocks, block_size, blocks_n, blocks_pow_2_n;
    rvec *rvec_spad;
    rvec int_press;
    simulation_box *big_box;
    
    rvec_spad = (rvec *) workspace->scratch;
    big_box = &system->big_box;

    /* 0: both int and ext, 1: ext only, 2: int only */
    if ( control->press_mode == 0 || control->press_mode == 2 )
    {
        blocks = system->n / DEF_BLOCK_SIZE + 
            ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

        compute_blocks( &blocks_n, &block_size, system->n );
        compute_nearest_pow_2( blocks_n, &blocks_pow_2_n );

        k_compute_pressure <<< blocks, DEF_BLOCK_SIZE >>>
            ( system->d_my_atoms, system->d_big_box, rvec_spad,
              system->n );

        k_reduction_rvec <<< blocks_n, block_size, sizeof(rvec) * block_size >>>
            ( rvec_spad, rvec_spad + system->n,  system->n );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        k_reduction_rvec <<< 1, blocks_pow_2_n, sizeof(rvec) * blocks_pow_2_n >>>
            ( rvec_spad + system->n, rvec_spad + system->n + blocks_n, blocks_n );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        copy_host_device( &int_press, rvec_spad + system->n + blocks_n, sizeof(rvec), 
                cudaMemcpyDeviceToHost, "Cuda_Compute_Pressure::d_int_press" );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d:p_int(%10.5f %10.5f %10.5f)p_ext(%10.5f %10.5f %10.5f)\n",
            system->my_rank, int_press[0], int_press[1], int_press[2],
            data->my_ext_press[0], data->my_ext_press[1], data->my_ext_press[2] );
#endif

    /* sum up internal and external pressure */
    MPI_Allreduce( int_press, data->int_press,
            3, MPI_DOUBLE, MPI_SUM, mpi_data->comm_mesh3D );
    MPI_Allreduce( data->my_ext_press, data->ext_press,
            3, MPI_DOUBLE, MPI_SUM, mpi_data->comm_mesh3D );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: %10.5f %10.5f %10.5f\n",
             system->my_rank,
             data->int_press[0], data->int_press[1], data->int_press[2] );
    fprintf( stderr, "p%d: %10.5f %10.5f %10.5f\n",
             system->my_rank,
             data->ext_press[0], data->ext_press[1], data->ext_press[2] );
#endif

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
