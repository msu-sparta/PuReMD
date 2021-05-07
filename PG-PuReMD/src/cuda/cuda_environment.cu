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


extern "C" void Cuda_Setup_Environment( reax_system const * const system,
        control_params * const control )
{

    int i, least_priority, greatest_priority, deviceCount;
    cudaError_t ret;
    
    ret = cudaGetDeviceCount( &deviceCount );

    if ( ret != cudaSuccess || deviceCount < 1 )
    {
        fprintf( stderr, "[ERROR] no CUDA capable device(s) found. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
    else if ( deviceCount < control->gpus_per_node || control->gpus_per_node < 1 )
    {
        fprintf( stderr, "[ERROR] invalid number of CUDA capable devices requested (gpus_per_node = %d). Terminating...\n",
                control->gpus_per_node );
        exit( INVALID_INPUT );
    }

    /* assign the GPU for each process */
    //TODO: handle condition where # CPU procs > # GPUs
    ret = cudaSetDevice( system->my_rank % control->gpus_per_node );

    if ( ret == cudaErrorInvalidDevice )
    {
        fprintf( stderr, "[ERROR] invalid CUDA device ID set (%d). Terminating...\n",
              system->my_rank % control->gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }
    else if ( ret == cudaErrorDeviceAlreadyInUse )
    {
        fprintf( stderr, "[ERROR] CUDA device with specified ID already in use (%d). Terminating...\n",
                system->my_rank % control->gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }

    ret = cudaDeviceGetStreamPriorityRange( &least_priority, &greatest_priority );

    if ( ret != cudaSuccess )
    {
        fprintf( stderr, "[ERROR] CUDA strema priority query failed. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }

    /* stream assignment (default to 0 for any kernel not listed):
     * 0: init dist, init CM, bond order (uncorrected/corrected), lone pair/over coord/under coord
     * 1: (after init dist) init bonds, (after bond order) bonds, valence angles, torsions
     * 2: (after init dist) init hbonds, (after bonds) hbonds
     * 3: (after init dist) van der Waals
     * 4: (after init CM) CM, Coulomb
     */
    for ( i = 0; i < MAX_CUDA_STREAMS; ++i )
    {
        /* all non-CM streams of equal priority */
        if ( i < MAX_CUDA_STREAMS - 1 )
        {
            ret = cudaStreamCreateWithPriority( &control->streams[i], cudaStreamNonBlocking, least_priority );
        }
        /* CM gets highest priority due to MPI comms and cudaMemcpy's */
        else
        {
            ret = cudaStreamCreateWithPriority( &control->streams[i], cudaStreamNonBlocking, greatest_priority );
        }

        if ( ret != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] CUDA stream creation failed (%d). Terminating...\n",
                    i );
            exit( CANNOT_INITIALIZE );
        }
    }

    /* stream event assignment:
     * 0: init dist done (stream 0)
     * 1: init CM done (stream 4)
     * 2: init bonds done (stream 1)
     * 3: init hbonds done (stream 2)
     * 4: bond orders done (stream 0)
     * 5: bonds done (stream 1)
     */
    for ( i = 0; i < MAX_CUDA_STREAM_EVENTS; ++i )
    {
        ret = cudaEventCreateWithFlags( &control->stream_events[i], cudaEventDisableTiming );

        if ( ret != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] CUDA event creation failed (%d). Terminating...\n",
                    i );
            exit( CANNOT_INITIALIZE );
        }
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


extern "C" void Cuda_Cleanup_Environment( control_params const * const control )
{
    int i;
    cudaError_t ret;

    for ( i = 0; i < MAX_CUDA_STREAMS; ++i )
    {
        ret = cudaStreamDestroy( control->streams[i] );

        if ( ret != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] CUDA strema destruction failed (%d). Terminating...\n",
                    i );
            exit( CANNOT_INITIALIZE );
        }
    }

    cudaDeviceReset( );
    cudaDeviceSynchronize( );
}
