#include "cuda_utils.h"


/* Safe wrapper around cudaMalloc and cudaMemsetAsync (optionally)
 *
 * ptr: pointer to allocated device memory
 * count: reqested allocation size in bytes
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaMalloc( void **ptr, size_t count, const char * const filename,
        int line )
{
    int rank;
    cudaError_t ret;

#if defined(DEBUG_FOCUS)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    fprintf( stderr, "[INFO] sCudaMalloc: requesting %zu bytes at line %d in file %.*s on MPI processor %d\n",
            count, line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif

    ret = cudaMalloc( ptr, count );

    if ( ret != cudaSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = cudaGetErrorString( ret );

        fprintf( stderr, "[ERROR] CUDA error: cudaMalloc failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }  

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] sCudaMalloc: granted memory at address %p at line %d in file %.*s on MPI processor %d\n",
            *ptr, line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif
}


/* Safe wrapper around cudaFree
 *
 * ptr: device pointer to memory to free
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaFree( void *ptr, const char * const filename, int line )
{
    int rank;
    cudaError_t ret;

    if ( !ptr )
    {
        return;
    }  

#if defined(DEBUG_FOCUS)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    fprintf( stderr, "[INFO] sCudaFree: freeing ptr at line %d in file %.*s on MPI processor %d\n",
            line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif

    ret = cudaFree( ptr );

    if( ret != cudaSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = cudaGetErrorString( ret );

        fprintf( stderr, "[WARNING] CUDA error: cudaFree failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );
        fprintf( stderr, "  [INFO] Memory address: %ld\n", 
                (long int) ptr );

        return;
    }  
}


/* Safe wrapper around cudaMemset
 *
 * ptr: address to device memory for which to set memory
 * data: value to set each byte of memory
 * count: num. bytes of memory to set beginning at specified address
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaMemset( void *ptr, int data, size_t count,
        const char * const filename, int line )
{
    int rank;
    cudaError_t ret;

    ret = cudaMemset( ptr, data, count );

    if ( ret != cudaSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = cudaGetErrorString( ret );

        fprintf( stderr, "[ERROR] CUDA error: cudaMemset failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}


/* Safe wrapper around cudaMemsetAsync
 *
 * ptr: address to device memory for which to set memory
 * data: value to set each byte of memory
 * count: num. bytes of memory to set beginning at specified address
 * s: CUDA stream to perform memset in
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaMemsetAsync( void *ptr, int data, size_t count,
        cudaStream_t s, const char * const filename, int line )
{
    int rank;
    cudaError_t ret;

    ret = cudaMemsetAsync( ptr, data, count, s );

    if ( ret != cudaSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = cudaGetErrorString( ret );

        fprintf( stderr, "[ERROR] CUDA error: cudaMemsetAsync failure\n" );
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
void sCudaCheckMalloc( void **ptr, size_t *cur_size, size_t new_size,
        const char * const filename, int line )
{
    assert( new_size > 0 );

    if ( new_size > *cur_size )
    {
#if defined(DEBUG_FOCUS)
        int rank;
    
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
        fprintf( stderr, "[INFO] sCudaCheckMalloc: requesting %zu bytes (%zu currently allocated) at line %d in file %.*s on MPI processor %d\n",
                new_size, *cur_size, line, (int) strlen(filename), filename, rank );
        fflush( stderr );
#endif

        if ( *cur_size != 0 )
        {
            sCudaFree( *ptr, filename, line );
        }

        //TODO: look into using aligned alloc's
        /* intentionally over-allocate to reduce the number of allocation operations,
         * and record the new allocation size */
        *cur_size = (size_t) CEIL( new_size * SAFE_ZONE );
        sCudaMalloc( ptr, *cur_size, filename, line );
    }
}


/* Safe wrapper around cudaMemcpy
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: CUDA enum specifying address types for dest and src
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaMemcpy( void * const dest, void const * const src, size_t count,
        cudaMemcpyKind dir, const char * const filename, int line )
{
    int rank;
    cudaError_t ret;

    ret = cudaMemcpy( dest, src, count, dir );

    if ( ret != cudaSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = cudaGetErrorString( ret );

        fprintf( stderr, "[ERROR] CUDA error: cudaMemcpy failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}


/* Safe wrapper around cudaMemcpyAsync
 *
 * dest: address to be copied to
 * src: address to be copied from
 * count: num. bytes to copy
 * dir: CUDA enum specifying address types for dest and src
 * s: CUDA stream to perform the copy in
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaMemcpyAsync( void * const dest, void const * const src, size_t count,
        cudaMemcpyKind dir, cudaStream_t s, const char * const filename, int line )
{
    int rank;
    cudaError_t ret;

    ret = cudaMemcpyAsync( dest, src, count, dir, s );

    if ( ret != cudaSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = cudaGetErrorString( ret );

        fprintf( stderr, "[ERROR] CUDA error: cudaMemcpyAsync failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }
}


void Cuda_Print_Mem_Usage( )
{
    size_t total, free;
    cudaError_t ret;

    ret = cudaMemGetInfo( &free, &total );

    if ( ret != cudaSuccess )
    {
        fprintf( stderr,
                "[WARNING] could not get message usage info from device\n"
                "    [INFO] CUDA API error code: %d\n",
                ret );
        return;
    }

    fprintf( stderr, "Total: %zu bytes (%7.2f MB)\nFree %zu bytes (%7.2f MB)\n", 
            total, (long long int) total / (1024.0 * 1024.0),
            free, (long long int) free / (1024.0 * 1024.0) );
}
