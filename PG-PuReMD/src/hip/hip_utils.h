#ifndef __HIP_UTILS_H_
#define __HIP_UTILS_H_

#include "../reax_types.h"


void sHipMalloc( void **, size_t, const char * const, int );

void sHipHostMalloc( void **, size_t, unsigned int, const char * const, int );

void sHipFree( void *, const char * const, int );

void sHipHostFree( void *, const char * const, int );

void sHipMemset( void *, int, size_t, const char * const, int );

void sHipMemsetAsync( void *, int, size_t, hipStream_t, const char * const, int );

void sHipCheckMalloc( void **, size_t *, size_t, const char * const, int );

void sHipMemcpy( void * const, void const * const, size_t,
        enum hipMemcpyKind, const char * const, int );

void sHipMemcpyAsync( void * const, void const * const, size_t,
        enum hipMemcpyKind, hipStream_t, const char * const, int );

void sHipHostMallocCheck( void **, size_t *, size_t, unsigned int, int, real,
        const char * const, int );

void sHipHostReallocCheck( void **, size_t *, size_t, unsigned int, int, real,
        const char * const, int );


#define hipCheckError() __hipCheckError( __FILE__, __LINE__ )
static inline void __hipCheckError( const char *file, const int line )
{
    hipError_t err;
#if defined(DEBUG_FOCUS)
    int rank;

    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    fprintf( stderr, "[INFO] hipCheckError: p%d, file %.*s, line %d\n", rank, (int) strlen(file), file, line );
    fflush( stderr );
#endif

#if defined(DEBUG)
    /* Block until tasks in stream are complete in order to enable
     * more pinpointed error checking. However, this will affect performance. */
    err = hipDeviceSynchronize( );
    if ( hipSuccess != err )
    {
        fprintf( stderr, "[ERROR] runtime error encountered with hipDeviceSynchronize( ) at: %s:%d\n", file, line );
        fprintf( stderr, "    [INFO] HIP API error code: %d\n", err );
        fprintf( stderr, "    [INFO] HIP API error name: %s\n", hipGetErrorName( err ) );
        fprintf( stderr, "    [INFO] HIP API error text: %s\n", hipGetErrorString( err ) );
        exit( RUNTIME_ERROR );
    }
#endif

    err = hipPeekAtLastError( );
    if ( hipSuccess != err )
    {
        fprintf( stderr, "[ERROR] runtime error encountered: %s:%d\n", file, line );
        fprintf( stderr, "    [INFO] HIP API error code: %d\n", err );
        fprintf( stderr, "    [INFO] HIP API error name: %s\n", hipGetErrorName( err ) );
        fprintf( stderr, "    [INFO] HIP API error text: %s\n", hipGetErrorString( err ) );
#if !defined(DEBUG)
        fprintf( stderr, "    [WARNING] HIP error info may not be precise due to async nature of HIP kernels!"
               " Rebuild in debug mode to get more accurate accounts of errors (--enable-debug=yes with configure script).\n" );
#endif
        exit( RUNTIME_ERROR );
    }
}


#endif
