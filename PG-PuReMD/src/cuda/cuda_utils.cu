#include "cuda_utils.h"


extern "C" void cuda_malloc( void **ptr, size_t size, int mem_set, const char *msg )
{

    cudaError_t retVal = cudaSuccess;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting %zu bytes for %s\n",
            size, msg );
    fflush( stderr );
#endif

    retVal = cudaMalloc( ptr, size );

    if ( retVal != cudaSuccess )
    {
        fprintf( stderr, "[ERROR] failed to allocate memory on device for resouce %s\n", msg );
        fprintf( stderr, "    [INFO] CUDA API error code: %d, requested memory size (in bytes): %lu\n", 
                retVal, size );
        exit( INSUFFICIENT_MEMORY );
    }  

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] granted memory at address: %p\n", *ptr );
    fflush( stderr );
#endif

    if ( mem_set == TRUE )
    {
        retVal = cudaMemset( *ptr, 0, size );

        if( retVal != cudaSuccess )
        {
            fprintf( stderr, "[ERROR] failed to memset memory on device for resource %s\n", msg );
            fprintf( stderr, "    [INFO] CUDA API error code: %d, requested memory size (in bytes): %lu\n", 
                    retVal, size );
            exit( INSUFFICIENT_MEMORY );
        }
    }  
}


extern "C" void cuda_free( void *ptr, const char *msg )
{

    cudaError_t retVal = cudaSuccess;

    if ( !ptr )
    {
        return;
    }  

    retVal = cudaFree( ptr );

    if( retVal != cudaSuccess )
    {
        fprintf( stderr, "[WARNING] failed to release memory on device for resource %s\n",
                msg );
        fprintf( stderr, "    [INFO] CUDA API error code: %d, memory address: %ld\n", 
                retVal, (long int) ptr );
        return;
    }  
}


extern "C" void cuda_memset( void *ptr, int data, size_t count, const char *msg )
{
    cudaError_t retVal = cudaSuccess;

    retVal = cudaMemset( ptr, data, count );

    if( retVal != cudaSuccess )
    {
        fprintf( stderr, "[ERROR] failed to memset memory on device for resource %s\n", msg );
        fprintf( stderr, "    [INFO] CUDA API error code: %d\n", retVal );
        exit( RUNTIME_ERROR );
    }
}


/* Checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space */
void cuda_check_malloc( void **ptr, size_t *cur_size, size_t new_size, const char *msg )
{
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] requesting %zu bytes for %s (%zu currently allocated)\n",
            new_size, msg, *cur_size );
    fflush( stderr );
#endif

    assert( new_size > 0 );

    if ( new_size > *cur_size )
    {
        if ( *cur_size > 0 || *ptr != NULL )
        {
            cuda_free( *ptr, msg );
        }

        //TODO: look into using aligned alloc's
        /* intentionally over-allocate to reduce the number of allocation operations,
         * and record the new allocation size */
        *cur_size = (size_t) CEIL( new_size * SAFE_ZONE );
        cuda_malloc( ptr, *cur_size, 0, msg );
    }
}


extern "C" void copy_host_device( void *host, void *dev, size_t size,
        cudaMemcpyKind dir, const char *msg )
{
    cudaError_t retVal = cudaErrorNotReady;

    if( dir == cudaMemcpyHostToDevice )
    {
        retVal = cudaMemcpy( dev, host, size, cudaMemcpyHostToDevice );
    }
    else
    {
        retVal = cudaMemcpy( host, dev, size, cudaMemcpyDeviceToHost );
    }

    if( retVal != cudaSuccess )
    {
        fprintf( stderr,
                "[ERROR] could not copy resource %s from host to device\n    [INFO] CUDA API error code: %d\n",
                msg, retVal );
        exit( INSUFFICIENT_MEMORY );
    }
}


extern "C" void copy_device( void *dest, void *src, size_t size, const char *msg )
{
    cudaError_t retVal;

    retVal = cudaMemcpy( dest, src, size, cudaMemcpyDeviceToDevice );

    if( retVal != cudaSuccess )
    {
        fprintf( stderr,
                "[ERROR] could not copy resource %s from device to device\n    [INFO] CUDA API error code: %d\n",
                msg, retVal );
        exit( INSUFFICIENT_MEMORY );
    }
}


extern "C" void compute_blocks( int *blocks, int *block_size, int count )
{
    *block_size = CUDA_BLOCK_SIZE;
    *blocks = (int) CEIL((double) count / CUDA_BLOCK_SIZE);
}


extern "C" void compute_matvec_blocks( int *blocks, int count )
{

    *blocks = (int) CEIL((double) count * MATVEC_KER_THREADS_PER_ROW / MATVEC_BLOCK_SIZE);
}


extern "C" void compute_nearest_pow_2( int blocks, int *result )
{

  *result = (int) EXP2( CEIL( LOG2((double) blocks) ) );
}


extern "C" void Cuda_Print_Mem_Usage( )
{
    size_t total, free;
    cudaError_t retVal;

    retVal = cudaMemGetInfo( &free, &total );

    if ( retVal != cudaSuccess )
    {
        fprintf( stderr,
                "[WARNING] could not get message usage info from device\n    [INFO] CUDA API error code: %d\n",
                retVal );
        return;
    }

    fprintf( stderr, "Total: %zu bytes (%7.2f MB)\nFree %zu bytes (%7.2f MB)\n", 
            total, (long long int)total/(1024.0*1024.0),
            free, (long long int)free/(1024.0*1024.0) );
}


extern "C" void Cuda_Init_Block_Sizes( reax_system *system, control_params *control )
{
    compute_blocks( &control->blocks, &control->block_size, system->n );
    compute_nearest_pow_2( control->blocks, &control->blocks_pow_2 );

    compute_blocks( &control->blocks_n, &control->block_size, system->N );
    compute_nearest_pow_2( control->blocks_n, &control->blocks_pow_2_n );

    compute_matvec_blocks( &control->matvec_blocks, system->N );

#if defined(__CUDA_DEBUG_LOG__)
    fprintf( stderr, "[INFO] control->matvec_blocks = %d, control->matvec_blocksize = %d, system->N = %d\n",
            control->matvec_blocks, MATVEC_BLOCK_SIZE, system->N );
#endif
}
