#include "cuda_environment.h"

#include "cuda_utils.h"


static void compute_blocks( int *blocks, int *block_size, int count )
{
    *block_size = DEF_BLOCK_SIZE; // threads per block
    *blocks = (int) CEIL( (double) count / DEF_BLOCK_SIZE ); // blocks per grid
}


static void compute_nearest_pow_2( int blocks, int *result )
{
  *result = (int) EXP2( CEIL( LOG2((double) blocks) ) );
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
    cudaSetDevice( rank % gpus_per_node );

//    cudaDeviceSetLimit( cudaLimitStackSize, 8192 );
//    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
//    cudaCheckError( );
}


extern "C" void Cuda_Init_Block_Sizes( reax_system *system,
        control_params *control )
{
    compute_blocks( &control->blocks, &control->block_size, system->n );
    compute_nearest_pow_2( control->blocks, &control->blocks_pow_2 );

    compute_blocks( &control->blocks_n, &control->block_size_n, system->N );
    compute_nearest_pow_2( control->blocks_n, &control->blocks_pow_2_n );
}


extern "C" void Cuda_Cleanup_Environment( )
{
    cudaDeviceReset( );
    cudaDeviceSynchronize( );
}
