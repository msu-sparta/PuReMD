#include "hip_utils.h"


/* Safe wrapper around hipMalloc and hipMemsetAsync (optionally)
 *
 * ptr: pointer to allocated device memory
 * count: reqested allocation size in bytes
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sHipMalloc( void **ptr, size_t count, const char * const filename,
        int line )
{
    int rank;
    hipError_t ret;

#if defined(DEBUG)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    fprintf( stderr, "[INFO] sHipMalloc: requesting %zu bytes at line %d in file %.*s on MPI processor %d\n",
            count, line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif

    ret = hipMalloc( ptr, count );

    if ( ret != hipSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = hipGetErrorString( ret );

        fprintf( stderr, "[ERROR] HIP error: hipMalloc failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }  

#if defined(DEBUG)
    fprintf( stderr, "[INFO] sHipMalloc: granted memory at address %p at line %d in file %.*s on MPI processor %d\n",
            *ptr, line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif
}


/* Safe wrapper around hipFree
 *
 * ptr: device pointer to memory to free
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sHipFree( void *ptr, const char * const filename, int line )
{
    int rank;
    hipError_t ret;

    if ( !ptr )
    {
        return;
    }  

#if defined(DEBUG)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    fprintf( stderr, "[INFO] sHipFree: freeing ptr at line %d in file %.*s on MPI processor %d\n",
            line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif

    ret = hipFree( ptr );

    if( ret != hipSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = hipGetErrorString( ret );

        fprintf( stderr, "[WARNING] HIP error: hipFree failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );
        fprintf( stderr, "  [INFO] Memory address: %ld\n", 
                (long int) ptr );

        return;
    }  
}


/* Safe wrapper around hipMemset
 *
 * ptr: address to device memory for which to set memory
 * data: value to set each byte of memory
 * count: num. bytes of memory to set beginning at specified address
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sHipMemset( void *ptr, int data, size_t count,
        const char * const filename, int line )
{
    int rank;
    hipError_t ret;

    ret = hipMemset( ptr, data, count );

    if ( ret != hipSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = hipGetErrorString( ret );

        fprintf( stderr, "[ERROR] HIP error: hipMemset failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}


/* Safe wrapper around hipMemsetAsync
 *
 * ptr: address to device memory for which to set memory
 * data: value to set each byte of memory
 * count: num. bytes of memory to set beginning at specified address
 * s: HIP stream to perform memset in
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sHipMemsetAsync( void *ptr, int data, size_t count,
        hipStream_t s, const char * const filename, int line )
{
    int rank;
    hipError_t ret;

    ret = hipMemsetAsync( ptr, data, count, s );

    if ( ret != hipSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = hipGetErrorString( ret );

        fprintf( stderr, "[ERROR] HIP error: hipMemsetAsync failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}


/* Checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space
 *
 * ptr: pointer to allocated device memory (if required)
 * cur_size: current allocation size in bytes
 * new_size: reqested new allocation size in bytes
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sHipCheckMalloc( void **ptr, size_t *cur_size, size_t new_size,
        const char * const filename, int line )
{
    assert( new_size > 0 || *cur_size > 0 );

    if ( new_size > *cur_size )
    {
#if defined(DEBUG)
        int rank;
    
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
        fprintf( stderr, "[INFO] sHipCheckMalloc: requesting %zu bytes (%zu currently allocated) at line %d in file %.*s on MPI processor %d\n",
                new_size, *cur_size, line, (int) strlen(filename), filename, rank );
        fflush( stderr );
#endif

        if ( *cur_size != 0 )
        {
            sHipFree( *ptr, filename, line );
        }

        //TODO: look into using aligned alloc's
        /* intentionally over-allocate to reduce the number of allocation operations,
         * and record the new allocation size */
        *cur_size = (size_t) CEIL( new_size * SAFE_ZONE );
        sHipMalloc( ptr, *cur_size, filename, line );
    }
}


/* Safe wrapper around hipMemcpy
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: HIP enum specifying address types for dest and src
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sHipMemcpy( void * const dest, void const * const src, size_t count,
        hipMemcpyKind dir, const char * const filename, int line )
{
    int rank;
    hipError_t ret;

    ret = hipMemcpy( dest, src, count, dir );

    if ( ret != hipSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = hipGetErrorString( ret );

        fprintf( stderr, "[ERROR] HIP error: hipMemcpy failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}


/* Safe wrapper around hipMemcpyAsync
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: HIP enum specifying address types for dest and src
 * s: HIP stream to perform the copy in
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sHipMemcpyAsync( void * const dest, void const * const src, size_t count,
        hipMemcpyKind dir, hipStream_t s, const char * const filename, int line )
{
    int rank;
    hipError_t ret;

    ret = hipMemcpyAsync( dest, src, count, dir, s );

    if ( ret != hipSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = hipGetErrorString( ret );

        fprintf( stderr, "[ERROR] HIP error: hipMemcpyAsync failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}
