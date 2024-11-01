#ifndef __GPU_UTILS_H_
#define __GPU_UTILS_H_

#include "../reax_types.h"


void sCudaMalloc( void **, size_t, const char * const, int );

void sCudaHostAlloc( void **, size_t, unsigned int, const char * const, int );

void sCudaFree( void *, const char * const, int );

void sCudaFreeHost( void *, const char * const, int );

void sCudaMemset( void *, int, size_t, const char * const, int );

void sCudaMemsetAsync( void *, int, size_t, cudaStream_t, const char * const, int );

void sCudaCheckMalloc( void **, size_t *, size_t, const char * const, int );

void sCudaMemcpy( void * const, void const * const, size_t,
        enum cudaMemcpyKind, const char * const, int );

void sCudaMemcpyAsync( void * const, void const * const, size_t,
        enum cudaMemcpyKind, cudaStream_t, const char * const, int );

void sCudaHostAllocCheck( void **, size_t *, size_t, unsigned int, int, real,
        const char * const, int );

void sCudaHostReallocCheck( void **, size_t *, size_t, unsigned int, int, real,
        const char * const, int );


#define cudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
static inline void __cudaCheckError( const char *file, const int line )
{
    cudaError_t err;
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
