#include "cuda_utils.h"


/* Safe wrapper around cudaMalloc
 *
 * ptr: pointer to allocated device memory
 * size: reqested allocation size in bytes
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaMalloc( void **ptr, size_t size, const char * const filename,
        int line )
{
    int rank;
    cudaError_t ret;

#if defined(DEBUG_FOCUS)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    fprintf( stderr, "[INFO] sCudaMalloc: requesting %zu bytes at line %d in file %.*s on MPI processor %d\n",
            size, line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif

    ret = cudaMalloc( ptr, size );

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


/* Safe wrapper around cudaHostAlloc
 *
 * ptr: pointer to allocated device memory
 * size: reqested allocation size in bytes
 * flags: requested properties of allocated memory
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaHostAlloc( void **ptr, size_t size, unsigned int flags, const char * const filename,
        int line )
{
    int rank;
    cudaError_t ret;

#if defined(DEBUG_FOCUS)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    fprintf( stderr, "[INFO] sCudaHostAlloc: requesting %zu bytes at line %d in file %.*s on MPI processor %d\n",
            size, line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif

    ret = cudaHostAlloc( ptr, size, flags );

    if ( ret != cudaSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = cudaGetErrorString( ret );

        fprintf( stderr, "[ERROR] CUDA error: cudaHostAlloc failure\n" );
        fprintf( stderr, "  [INFO] At line %d in file %.*s on MPI processor %d\n",
                line, (int) strlen(filename), filename, rank );
        fprintf( stderr, "  [INFO] Error code: %d\n", ret );
        fprintf( stderr, "  [INFO] Error message: %.*s\n", (int) strlen(str), str );

        MPI_Abort( MPI_COMM_WORLD, RUNTIME_ERROR );
    }  

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "[INFO] sCudaHostAlloc: granted memory at address %p with flags %u at line %d in file %.*s on MPI processor %d\n",
            *ptr, flags, line, (int) strlen(filename), filename, rank );
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

    if ( ptr == NULL )
    {
        fprintf( stderr, "[WARNING] trying to free the already NULL pointer\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        return;
    }  

#if defined(DEBUG_FOCUS)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    fprintf( stderr, "[INFO] sCudaFree: freeing ptr at line %d in file %.*s on MPI processor %d\n",
            line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif

    ret = cudaFree( ptr );

    if ( ret != cudaSuccess )
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


/* Safe wrapper around cudaFreeHost
 *
 * ptr: device pointer to memory to free
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 */
void sCudaFreeHost( void * ptr, const char * const filename, int line )
{
    int rank;
    cudaError_t ret;

    if ( ptr == NULL )
    {
        fprintf( stderr, "[WARNING] trying to free the already NULL pointer\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        return;
    }  

#if defined(DEBUG_FOCUS)
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    fprintf( stderr, "[INFO] sCudaFreeHost: freeing ptr at line %d in file %.*s on MPI processor %d\n",
            line, (int) strlen(filename), filename, rank );
    fflush( stderr );
#endif

    ret = cudaFreeHost( ptr );

    if ( ret != cudaSuccess )
    {
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
        const char *str = cudaGetErrorString( ret );

        fprintf( stderr, "[WARNING] CUDA error: cudaFreeHost failure\n" );
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
    assert( new_size > 0 || *cur_size > 0 );

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


/* Safe wrapper around check first and reallocate if needed routine for pinned memory:
 * checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space
 *
 * ptr: pointer to memory allocation
 * cur_size: num. of bytes currently allocated
 * new_size: num. of bytes to be newly allocated, if needed
 * flags: requested properties of allocated memory
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 * */
void sCudaHostAllocCheck( void **ptr, size_t *cur_size, size_t new_size,
        unsigned int flags, int over_alloc, real over_alloc_factor,
        const char * const filename, int line )
{
    assert( new_size > 0 || *cur_size > 0 );

    if ( new_size > *cur_size )
    {
#if defined(DEBUG_FOCUS)
        int rank;
    
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
        fprintf( stderr, "[INFO] sCudaHostAllocCheck: requesting %zu bytes (%zu currently allocated) with flags %u at line %d in file %.*s on MPI processor %d\n",
                new_size, *cur_size, flags, line, (int) strlen(filename), filename, rank );
        fflush( stderr );
#endif

        if ( *cur_size != 0 )
        {
            sCudaFreeHost( *ptr, filename, line );
        }

        if ( over_alloc == TRUE )
        {
            *cur_size = (int) CEIL( new_size * over_alloc_factor );
        }
        else
        {
            *cur_size = new_size;
        }

        sCudaHostAlloc( ptr, *cur_size, flags, filename, line );
    }
}


/* Safe wrapper around check first and reallocate if needed
 * while preserving current memory contents routine for pinned memory:
 * checks if the amount of space currently allocated to ptr is sufficient,
 * and, if not, frees any space allocated to ptr before allocating the
 * requested amount of space
 *
 * ptr: pointer to memory allocation
 * cur_size: num. of bytes currently allocated
 * new_size: num. of bytes to be newly allocated, if needed
 * flags: requested properties of allocated memory
 * filename: NULL-terminated source filename where function call originated
 * line: line of source file where function call originated
 * */
void sCudaHostReallocCheck( void **ptr, size_t *cur_size, size_t new_size,
        unsigned int flags, int over_alloc, real over_alloc_factor,
        const char * const filename, int line )
{
    void *old_ptr;
    size_t old_ptr_size;

    assert( new_size > 0 || *cur_size > 0 );

    if ( new_size > *cur_size )
    {
#if defined(DEBUG_FOCUS)
        int rank;
    
        MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
        fprintf( stderr, "[INFO] sCudaHostReallocCheck: requesting %zu bytes (%zu currently allocated) with flags %u at line %d in file %.*s on MPI processor %d\n",
                new_size, *cur_size, flags, line, (int) strlen(filename), filename, rank );
        fflush( stderr );
#endif

        old_ptr = *ptr;
        old_ptr_size = *cur_size;
        *ptr = NULL;

        if ( over_alloc == TRUE )
        {
            *cur_size = (int) CEIL( new_size * over_alloc_factor );
        }
        else
        {
            *cur_size = new_size;
        }

        sCudaHostAlloc( ptr, *cur_size, flags, filename, line );

        if ( old_ptr_size != 0 )
        {
            sCudaMemcpy( *ptr, old_ptr, old_ptr_size, cudaMemcpyHostToHost,
                    __FILE__, __LINE__ );

            sCudaFreeHost( old_ptr, filename, line );
        }
    }
}
