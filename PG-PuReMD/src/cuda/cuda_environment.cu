#include "cuda_environment.h"

#include "cuda_utils.h"


static void compute_blocks( int *blocks, int *block_size, int threads )
{
    *block_size = DEF_BLOCK_SIZE; // threads per block
    *blocks = (threads + (DEF_BLOCK_SIZE - 1)) / DEF_BLOCK_SIZE; // blocks per grid
}


static void compute_nearest_multiple_32( int blocks, int *result )
{
    *result = ((blocks + 31) / 32) * 32;
}


extern "C" void Cuda_Setup_Environment( int rank, int nprocs, int gpus_per_node )
{

    int deviceCount;
    cudaError_t flag;
    
    flag = cudaGetDeviceCount( &deviceCount );

    if ( flag != cudaSuccess || deviceCount < 1 )
    {
        fprintf( stderr, "[ERROR] no CUDA capable device(s) found. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
    else if ( deviceCount < gpus_per_node || gpus_per_node < 1 )
    {
        fprintf( stderr, "[ERROR] invalid number of CUDA capable devices requested (gpus_per_node = %d). Terminating...\n",
                gpus_per_node );
        exit( INVALID_INPUT );
    }

    /* assign the GPU for each process */
    //TODO: handle condition where # CPU procs > # GPUs
    flag = cudaSetDevice( rank % gpus_per_node );

    if ( flag == cudaErrorInvalidDevice )
    {
        fprintf( stderr, "[ERROR] invalid CUDA device ID set (%d). Terminating...\n",
              rank % gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }
    else if ( flag == cudaErrorDeviceAlreadyInUse )
    {
        fprintf( stderr, "[ERROR] CUDA device with specified ID already in use (%d). Terminating...\n",
                rank % gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }

    //TODO: revisit additional device configurations
//    cudaDeviceSetLimit( cudaLimitStackSize, 8192 );
//    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
}


extern "C" void Cuda_Init_Block_Sizes( reax_system *system,
        control_params *control )
{
    compute_blocks( &control->blocks, &control->block_size, system->n );
    compute_nearest_multiple_32( control->blocks, &control->blocks_pow_2 );

    compute_blocks( &control->blocks_n, &control->block_size_n, system->N );
    compute_nearest_multiple_32( control->blocks_n, &control->blocks_pow_2_n );
}


extern "C" void Cuda_Cleanup_Environment( )
{
    cudaDeviceReset( );
    cudaDeviceSynchronize( );
}
