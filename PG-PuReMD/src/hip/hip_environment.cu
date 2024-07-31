#include "hip_environment.h"

#include "hip_utils.h"


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


extern "C" void Hip_Setup_Environment( reax_system const * const system,
        control_params * const control )
{
    int i, least_priority, greatest_priority, is_stream_priority_supported;
    int deviceCount;
    hipError_t ret;
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret_hipblas;
#endif
    
    ret = hipGetDeviceCount( &deviceCount );

    if ( ret != hipSuccess || deviceCount < 1 )
    {
        fprintf( stderr, "[ERROR] no HIP capable device(s) found. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
    else if ( deviceCount < control->gpus_per_node || control->gpus_per_node < 1 )
    {
        fprintf( stderr, "[ERROR] invalid number of HIP capable devices requested (gpus_per_node = %d). Terminating...\n",
                control->gpus_per_node );
        exit( INVALID_INPUT );
    }

    /* assign the GPU for each process */
    //TODO: handle condition where # CPU procs > # GPUs
    ret = hipSetDevice( system->my_rank % control->gpus_per_node );

    if ( ret == hipErrorInvalidDevice )
    {
        fprintf( stderr, "[ERROR] invalid HIP device ID set (%d). Terminating...\n",
              system->my_rank % control->gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }
    else if ( ret == hipErrorContextAlreadyInUse )
    {
        fprintf( stderr, "[ERROR] HIP device with specified ID already in use (%d). Terminating...\n",
                system->my_rank % control->gpus_per_node );
        exit( CANNOT_INITIALIZE );
    }

//    ret = hipDeviceGetAttribute( &is_stream_priority_supported,
//            hipDeviceAttributeStreamPrioritiesSupported,
//            system->my_rank % control->gpus_per_node );
//
//    if ( ret != hipSuccess )
//    {
//        fprintf( stderr, "[ERROR] hipDeviceGetAttribute failure. Terminating...\n" );
//        exit( CANNOT_INITIALIZE );
//    }
//
//    if ( is_stream_priority_supported == 1 )
//    {
//        ret = hipDeviceGetStreamPriorityRange( &least_priority, &greatest_priority );
//    
//        if ( ret != hipSuccess )
//        {
//            fprintf( stderr, "[ERROR] HIP stream priority query failed. Terminating...\n" );
//            exit( CANNOT_INITIALIZE );
//        }
//    
//        /* stream assignment (default to 0 for any kernel not listed):
//         * 0: init dist, (after init bonds) bond order (uncorrected/corrected), lone pair/over coord/under coord
//         * 1: (after init dist) init bonds, (after bond order) bonds, valence angles, torsions
//         * 2: (after init dist) init hbonds, (after bonds) hbonds
//         * 3: (after init dist) van der Waals
//         * 4: init CM, CM, Coulomb
//         */
//        for ( i = MAX_GPU_STREAMS - 1; i >= 0; --i )
//        {
//            if ( MAX_GPU_STREAMS - 1 - i < control->gpu_streams )
//            {
//                /* all non-CM hip_streams of equal priority */
//                if ( i != MAX_GPU_STREAMS - 1 )
//                {
//                    ret = hipStreamCreateWithPriority( &control->hip_streams[i], hipStreamNonBlocking, least_priority );
//                }
//                /* CM gets highest priority due to MPI comms and hipMemcpy's */
//                else
//                {
//                    ret = hipStreamCreateWithPriority( &control->hip_streams[i], hipStreamNonBlocking, greatest_priority );
//                }
//        
//                if ( ret != hipSuccess )
//                {
//                    fprintf( stderr, "[ERROR] hipStreamCreateWithPriority failure (%d). Terminating...\n",
//                            i );
//                    exit( CANNOT_INITIALIZE );
//                }
//            }
//            else
//            {
//                control->hip_streams[i] = control->hip_streams[MAX_GPU_STREAMS - 1 - ((MAX_GPU_STREAMS - 1 - i) % control->gpu_streams)];
//            }
//        }
//    }
//    else
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
                ret = hipStreamCreateWithFlags( &control->hip_streams[i], hipStreamNonBlocking );
        
                if ( ret != hipSuccess )
                {
                    fprintf( stderr, "[ERROR] hipStreamCreateWithFlags failure (%d). Terminating...\n",
                            i );
                    exit( CANNOT_INITIALIZE );
                }
            }
            else
            {
                control->hip_streams[i] = control->hip_streams[MAX_GPU_STREAMS - 1 - ((MAX_GPU_STREAMS - 1 - i) % control->gpu_streams)];
            }
       }
    }

    for ( i = 0; i < GPU_STREAM_SYNC_EVENT_N; ++i )
    {
        ret = hipEventCreateWithFlags( &control->hip_stream_events[i], hipEventDisableTiming );

        if ( ret != hipSuccess )
        {
            fprintf( stderr, "[ERROR] hipEventCreateWithFlags failure (%d). Terminating...\n",
                    i );
            exit( CANNOT_INITIALIZE );
        }
    }

#if defined(LOG_PERFORMANCE)
    for ( i = 0; i < GPU_TIMING_EVENT_N; ++i )
    {
        ret = hipEventCreate( &control->hip_time_events[i] );

        if ( ret != hipSuccess )
        {
            fprintf( stderr, "[ERROR] hipEventCreate failure (%d). Terminating...\n", i );
            exit( CANNOT_INITIALIZE );
        }
    }
#endif

#if defined(USE_HIPBLAS)
    ret_hipblas = hipblasCreate( &control->hipblas_handle );

    if ( ret_hipblas != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] cuBLAS initialization failure. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }

    ret_hipblas = hipblasSetStream( control->hipblas_handle, control->hip_streams[5] );

    if ( ret_hipblas != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasSetStream failure. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
#endif

    //TODO: revisit additional device configurations
//    hipDeviceSetLimit( hipLimitStackSize, 8192 );
//    hipDeviceSetCacheConfig( hipFuncCachePreferL1 );
}


extern "C" void Hip_Init_Block_Sizes( reax_system *system,
        control_params *control )
{
    compute_blocks( &control->blocks_n, control->gpu_block_size, system->n );
    compute_blocks( &control->blocks_warp_n, control->gpu_block_size, system->n * WARP_SIZE );
    compute_nearest_multiple_warp( control->blocks_n, &control->blocks_pow_2_n );

    compute_blocks( &control->blocks_N, control->gpu_block_size, system->N );
    compute_blocks( &control->blocks_warp_N, control->gpu_block_size, system->N * WARP_SIZE );
}


extern "C" void Hip_Cleanup_Environment( control_params const * const control )
{
    int i;
    hipError_t ret;
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret_hipblas;
#endif

    for ( i = MAX_GPU_STREAMS - 1; i >= 0; --i )
    {
        if ( MAX_GPU_STREAMS - 1 - i < control->gpu_streams )
        {
            ret = hipStreamDestroy( control->hip_streams[i] );
    
            if ( ret != hipSuccess )
            {
                fprintf( stderr, "[ERROR] HIP stream destruction failed (%d). Terminating...\n",
                        i );
                exit( CANNOT_INITIALIZE );
            }
        }
    }

    for ( i = 0; i < GPU_STREAM_SYNC_EVENT_N; ++i )
    {
        ret = hipEventDestroy( control->hip_stream_events[i] );

        if ( ret != hipSuccess )
        {
            fprintf( stderr, "[ERROR] HIP event destruction failure (%d). Terminating...\n",
                    i );
            exit( RUNTIME_ERROR );
        }
    }

#if defined(LOG_PERFORMANCE)
    for ( i = 0; i < GPU_TIMING_EVENT_N; ++i )
    {
        ret = hipEventDestroy( control->hip_time_events[i] );

        if ( ret != hipSuccess )
        {
            fprintf( stderr, "[ERROR] HIP event destruction failure (%d). Terminating...\n", i );
            exit( RUNTIME_ERROR );
        }
    }
#endif

#if defined(USE_HIPBLAS)
    ret_hipblas = hipblasDestroy( control->hipblas_handle );

    if ( ret_hipblas != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] cuBLAS cleanup failure. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }
#endif
}


extern "C" void Hip_Print_Mem_Usage( simulation_data const * const data )
{
    int rank;
    size_t total, free;
    hipError_t ret;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    ret = hipMemGetInfo( &free, &total );

    if ( ret != hipSuccess )
    {
        fprintf( stderr,
                "[WARNING] could not get message usage info from device\n"
                "    [INFO] HIP API error code: %d\n",
                ret );
        return;
    }

    fprintf( stderr, "[INFO] step %d on MPI processor %d, Total: %zu bytes (%7.2f MB) Free %zu bytes (%7.2f MB)\n", 
            data->step, rank,
            total, (long long int) total / (1024.0 * 1024.0),
            free, (long long int) free / (1024.0 * 1024.0) );
    fflush( stderr );
}
