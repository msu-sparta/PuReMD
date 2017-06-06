#include "cuda_utils.h"


extern "C" void cuda_malloc( void **ptr, size_t size, int mem_set, const char *msg )
{

    cudaError_t retVal = cudaSuccess;

    retVal = cudaMalloc( ptr, size );

    if ( retVal != cudaSuccess )
    {
        fprintf( stderr, "ERROR: failed to allocate memory on device for resouce %s\n", msg );
        fprintf( stderr, "CUDA API error code: %d, requested memory size (in bytes): %lu\n", 
                retVal, size );
        exit( INSUFFICIENT_MEMORY );
    }  

    if ( mem_set == TRUE )
    {
        retVal = cudaMemset( *ptr, 0, size );

        if( retVal != cudaSuccess )
        {
            fprintf( stderr, "ERROR: failed to memset memory on device for resource %s\n", msg );
            fprintf( stderr, "CUDA API error code: %d, requested memory size (in bytes): %lu\n", 
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
        fprintf( stderr,
                "WARNING: failed to release memory on device for resource %s\nCUDA API error code: %d, memory address: %ld\n", 
                msg, retVal, (long int) ptr );
        return;
    }  
}


extern "C" void cuda_memset( void *ptr, int data, size_t count, const char *msg )
{
    cudaError_t retVal = cudaSuccess;

    retVal = cudaMemset( ptr, data, count );

    if( retVal != cudaSuccess )
    {
        fprintf( stderr,
                "ERROR: failed to memset memory on device for resource %s\nCUDA API error code: %d\n", 
                msg, retVal );
        exit( INSUFFICIENT_MEMORY );
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
                "ERROR: could not copy resource %s from host to device\nCUDA API error code: %d n",
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
                "ERROR: could not copy resource %s from device to device\nCUDA API error code: %d\n",
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


extern "C" void print_device_mem_usage( )
{
    size_t total, free;
    cudaError_t retVal;

    retVal = cudaMemGetInfo( &free, &total );

    if ( retVal != cudaSuccess )
    {
        fprintf( stderr,
                "WARNING: could not get message usage info from device\nCUDA API error code: %d\n",
                retVal );
        return;
    }

    fprintf( stderr, "Total: %zu bytes (%7.2f MB)\nFree %zu bytes (%7.2f MB)\n", 
            total, (long long int)total/(1024.0*1024.0),
            free, (long long int)free/(1024.0*1024.0) );
}
