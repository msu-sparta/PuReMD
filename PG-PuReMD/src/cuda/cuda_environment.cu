#include "cuda_environment.h"

#include "cuda_utils.h"


/* compute num. of blocks required given block size and total threads */
static void compute_blocks( int *blocks, int block_size, int total_threads )
{
    *blocks = (total_threads + (block_size - 1)) / block_size;
}


/* round num. of blocks up to the nearest multiple of warp size */
static void compute_nearest_multiple_warp( int blocks, int *new_blocks )
{
    *new_blocks = ((blocks + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
}


extern "C" void Cuda_Setup_Environment( reax_system const * const system,
        control_params * const control )
{
    int i, least_priority, greatest_priority, is_stream_priority_supported;
    int deviceCount;
    cudaError_t ret;
#if defined(USE_CUBLAS)
    cublasStatus_t ret_cublas;
#endif
    
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

    ret = cudaDeviceGetAttribute( &is_stream_priority_supported,
            cudaDevAttrStreamPrioritiesSupported,
            system->my_rank % control->gpus_per_node );

    if ( ret != cudaSuccess )
    {
        fprintf( stderr, "[ERROR] cudaDeviceGetAttribute failure. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }

    if ( is_stream_priority_supported == 1 )
    {
        ret = cudaDeviceGetStreamPriorityRange( &least_priority, &greatest_priority );
    
        if ( ret != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] CUDA stream priority query failed. Terminating...\n" );
            exit( CANNOT_INITIALIZE );
        }
    
        /* stream assignment (default to 0 for any kernel not listed):
         * 0: init dist, (after init bonds) bond order (uncorrected/corrected), lone pair/over coord/under coord
         * 1: (after init dist) init bonds, (after bond order) bonds, valence angles, torsions
         * 2: (after init dist) init hbonds, (after bonds) hbonds
         * 3: (after init dist) van der Waals
         * 4: init CM, CM, Coulomb
         */
        for ( i = MAX_GPU_STREAMS - 1; i >= 0; --i )
        {
            if ( MAX_GPU_STREAMS - 1 - i < control->gpu_streams )
            {
                /* all non-CM cuda_streams of equal priority */
                if ( i != MAX_GPU_STREAMS - 1 )
                {
                    ret = cudaStreamCreateWithPriority( &control->cuda_streams[i], cudaStreamNonBlocking, least_priority );
                }
                /* CM gets highest priority due to MPI comms and cudaMemcpy's */
                else
                {
                    ret = cudaStreamCreateWithPriority( &control->cuda_streams[i], cudaStreamNonBlocking, greatest_priority );
                }
        
                if ( ret != cudaSuccess )
                {
                    fprintf( stderr, "[ERROR] cudaStreamCreateWithPriority failure (%d). Terminating...\n",
                            i );
                    exit( CANNOT_INITIALIZE );
                }
            }
            else
            {
                control->cuda_streams[i] = control->cuda_streams[MAX_GPU_STREAMS - 1 - ((MAX_GPU_STREAMS - 1 - i) % control->gpu_streams)];
            }
        }
    }
    else
    {
        /* stream assignment (default to 0 for any kernel not listed):
         * 0: init dist, bond order (uncorrected/corrected), lone pair/over coord/under coord
         * 1: (after init dist) init bonds, (after bond order) bonds
         * 2: (after init dist) init hbonds, (after bond order) hbonds
         * 3: (after bond order) valence angles, torsions
         * 4: (after init dist) van der Waals
         * 5: (after init dist) init CM, CM, Coulomb
         */
        for ( i = MAX_GPU_STREAMS - 1; i >= 0; --i )
        {
            if ( MAX_GPU_STREAMS - 1 - i < control->gpu_streams )
            {
                ret = cudaStreamCreateWithFlags( &control->cuda_streams[i], cudaStreamNonBlocking );
        
                if ( ret != cudaSuccess )
                {
                    fprintf( stderr, "[ERROR] cudaStreamCreateWithFlags failure (%d). Terminating...\n",
                            i );
                    exit( CANNOT_INITIALIZE );
                }
            }
            else
            {
                control->cuda_streams[i] = control->cuda_streams[MAX_GPU_STREAMS - 1 - ((MAX_GPU_STREAMS - 1 - i) % control->gpu_streams)];
            }
       }
    }

    for ( i = 0; i < GPU_STREAM_SYNC_EVENT_N; ++i )
    {
        ret = cudaEventCreateWithFlags( &control->cuda_stream_events[i], cudaEventDisableTiming );

        if ( ret != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] cudaEventCreateWithFlags failure (%d). Terminating...\n",
                    i );
            exit( CANNOT_INITIALIZE );
        }
    }

#if defined(LOG_PERFORMANCE)
    for ( i = 0; i < GPU_TIMING_EVENT_N; ++i )
    {
        ret = cudaEventCreate( &control->cuda_time_events[i] );

        if ( ret != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] cudaEventCreate failure (%d). Terminating...\n", i );
            exit( CANNOT_INITIALIZE );
        }
    }
#endif

#if defined(USE_CUBLAS)
    ret_cublas = cublasCreate( &control->cublas_handle );

    if ( ret_cublas != CUBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] cuBLAS initialization failure. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }

    ret_cublas = cublasSetStream( control->cublas_handle, control->cuda_streams[5] );

    if ( ret_cublas != CUBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] cublasSetStream failure. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
#endif

    //TODO: revisit additional device configurations
//    cudaDeviceSetLimit( cudaLimitStackSize, 8192 );
//    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
}


extern "C" void Cuda_Init_Block_Sizes( reax_system *system,
        control_params *control )
{
    compute_blocks( &control->blocks_n, control->gpu_block_size, system->n );
    compute_blocks( &control->blocks_warp_n, control->gpu_block_size, system->n * WARP_SIZE );
    compute_nearest_multiple_warp( control->blocks_n, &control->blocks_pow_2_n );

    compute_blocks( &control->blocks_N, control->gpu_block_size, system->N );
    compute_blocks( &control->blocks_warp_N, control->gpu_block_size, system->N * WARP_SIZE );
}


extern "C" void Cuda_Cleanup_Environment( control_params const * const control )
{
    int i;
    cudaError_t ret;
#if defined(USE_CUBLAS)
    cublasStatus_t ret_cublas;
#endif

    for ( i = MAX_GPU_STREAMS - 1; i >= 0; --i )
    {
        if ( MAX_GPU_STREAMS - 1 - i < control->gpu_streams )
        {
            ret = cudaStreamDestroy( control->cuda_streams[i] );
    
            if ( ret != cudaSuccess )
            {
                fprintf( stderr, "[ERROR] CUDA stream destruction failed (%d). Terminating...\n",
                        i );
                exit( CANNOT_INITIALIZE );
            }
        }
    }

    for ( i = 0; i < GPU_STREAM_SYNC_EVENT_N; ++i )
    {
        ret = cudaEventDestroy( control->cuda_stream_events[i] );

        if ( ret != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] CUDA event destruction failure (%d). Terminating...\n",
                    i );
            exit( RUNTIME_ERROR );
        }
    }

#if defined(LOG_PERFORMANCE)
    for ( i = 0; i < GPU_TIMING_EVENT_N; ++i )
    {
        ret = cudaEventDestroy( control->cuda_time_events[i] );

        if ( ret != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] CUDA event destruction failure (%d). Terminating...\n", i );
            exit( RUNTIME_ERROR );
        }
    }
#endif

#if defined(USE_CUBLAS)
    ret_cublas = cublasDestroy( control->cublas_handle );

    if ( ret_cublas != CUBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] cuBLAS cleanup failure. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
#endif
}


extern "C" void Cuda_Print_Mem_Usage( simulation_data const * const data )
{
    int rank;
    size_t total, free;
    cudaError_t ret;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    ret = cudaMemGetInfo( &free, &total );

    if ( ret != cudaSuccess )
    {
        fprintf( stderr,
                "[WARNING] could not get message usage info from device\n"
                "    [INFO] CUDA API error code: %d\n",
                ret );
        return;
    }

    fprintf( stderr, "[INFO] step %d on MPI processor %d, Total: %zu bytes (%7.2f MB) Free %zu bytes (%7.2f MB)\n", 
            data->step, rank,
            total, (long long int) total / (1024.0 * 1024.0),
            free, (long long int) free / (1024.0 * 1024.0) );
    fflush( stderr );
}
