#ifndef __CUDA_UTILS_H_
#define __CUDA_UTILS_H_

#include "../reax_types.h"


void cuda_malloc( void **, size_t, int, const char * );

void cuda_free( void *, const char * );

void cuda_memset( void *, int , size_t , const char * );

void cuda_check_malloc( void **, size_t *, size_t, const char * );

void copy_host_device( void *, void *, size_t, enum cudaMemcpyKind, const char * );

void copy_device( void *, void *, size_t, const char * );

void Cuda_Print_Mem_Usage( );


#define cudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
static inline void __cudaCheckError( const char *file, const int line )
{
    cudaError err;
#if defined(DEBUG_FOCUS)
    int rank;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    fprintf( stderr, "[INFO] cudaCheckError: p%d, file %.*s, line %d\n", rank, (int) strlen(file), file, line );
    fflush( stderr );
#endif

#if defined(DEBUG)
    /* Block until tasks in stream are complete in order to enable
     * more pinpointed error checking. However, this will affect performance. */
    err = cudaDeviceSynchronize( );
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "[ERROR] runtime error encountered with cudaDeviceSynchronize( ) at: %s:%d\n", file, line );
        fprintf( stderr, "    [INFO] CUDA API error code: %d\n", err );
        fprintf( stderr, "    [INFO] CUDA API error name: %s\n", cudaGetErrorName( err ) );
        fprintf( stderr, "    [INFO] CUDA API error text: %s\n", cudaGetErrorString( err ) );
        exit( RUNTIME_ERROR );
    }
#endif

    err = cudaPeekAtLastError( );
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "[ERROR] runtime error encountered: %s:%d\n", file, line );
        fprintf( stderr, "    [INFO] CUDA API error code: %d\n", err );
        fprintf( stderr, "    [INFO] CUDA API error name: %s\n", cudaGetErrorName( err ) );
        fprintf( stderr, "    [INFO] CUDA API error text: %s\n", cudaGetErrorString( err ) );
#if !defined(DEBUG)
        fprintf( stderr, "    [WARNING] CUDA error info may not be precise due to async nature of CUDA kernels!"
               " Rebuild in debug mode to get more accurate accounts of errors (--enable-debug=yes with configure script).\n" );
#endif
        exit( RUNTIME_ERROR );
    }
}


#endif
