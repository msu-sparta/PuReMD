#include "hip_dense_lin_alg.h"

#include "hip_reduction.h"
#include "hip_utils.h"

#include "../comm_tools.h"


#if !defined(USE_HIPBLAS)
/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
GPU_GLOBAL void k_vector_copy( real * const dest, real const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] = v[i];
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
GPU_GLOBAL void k_vector_copy_rvec2( rvec2 * const dest, rvec2 const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i][0] = v[i][0];
    dest[i][1] = v[i][1];
}


GPU_GLOBAL void k_vector_copy_from_rvec2( real * const dst, rvec2 const * const src,
        int index, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    dst[i] = src[i][index];
}


GPU_GLOBAL void k_vector_copy_to_rvec2( rvec2 * const dst, real const * const src,
        int index, int n )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    dst[i][index] = src[i];
}


/* computed the scaled sum of two dense vector and store
 * the result in a third vector (SAXPY operation in BLAS)
 *
 * inputs:
 *  c, d: scaling constants
 *  v, y: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector containing the scaled sum
 */
GPU_GLOBAL void k_vector_sum( real * const dest, real c, real const * const v,
        real d, real const * const y, unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] = c * v[i] + d * y[i];
}


GPU_GLOBAL void k_vector_sum_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v, 
        real d0, real d1, rvec2 const * const y, unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i][0] = c0 * v[i][0] + d0 * y[i][0];
    dest[i][1] = c1 * v[i][1] + d1 * y[i][1];
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
GPU_GLOBAL void k_vector_add( real * const dest, real c, real const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] += c * v[i];
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
GPU_GLOBAL void k_vector_add_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i][0] += c0 * v[i][0];
    dest[i][1] += c1 * v[i][1];
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 * output:
 *  dest: vector with the result of the multiplication
 */
GPU_GLOBAL void k_vector_mult( real * const dest, real const * const v1,
        real const * const v2, unsigned k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] = v1[i] * v2[i];
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 * output:
 *  dest: vector with the result of the multiplication
 */
GPU_GLOBAL void k_vector_mult_rvec2( rvec2 * const dest, rvec2 const * const v1,
        rvec2 const * const v2, unsigned k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i][0] = v1[i][0] * v2[i][0];
    dest[i][1] = v1[i][1] * v2[i][1];
}
#endif


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 *  s: HIP stream
 * output:
 *  dest: vector copied into
 */
void Vector_Copy( real * const dest, real const * const v,
        unsigned int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    ret = hipblasDcopy( handle, k, v, 1, dest, 1 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDcopy failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }
#else
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_copy <<< blocks, block_size, 0, s >>>
        ( dest, v, k );
    hipCheckError( );
#endif
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 *  block_size: HIP threads per block
 *  s: HIP stream
 * output:
 *  dest: vector copied into
 */
void Vector_Copy_rvec2( rvec2 * const dest, rvec2 const * const v,
        unsigned int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    ret = hipblasDcopy( handle, 2 * k, (double *) v, 1, (double *) dest, 1 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDcopy failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }
#else
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_copy_rvec2 <<< blocks, block_size, 0, s >>>
        ( dest, v, k );
    hipCheckError( );
#endif
}


void Vector_Copy_From_rvec2( real * const dst, rvec2 const * const src,
        int index, int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    ret = hipblasDcopy( handle, k, &((double *) src)[index], 2, dst, 1 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDcopy failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }
#else
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_copy_from_rvec2 <<< blocks, block_size, 0, s >>>
        ( dst, src, index, k );
    hipCheckError( );
#endif
}


void Vector_Copy_To_rvec2( rvec2 * const dst, real const * const src,
        int index, int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    ret = hipblasDcopy( handle, k, src, 1, &((double *) dst)[index], 2 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDcopy failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }
#else
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_copy_to_rvec2 <<< blocks, block_size, 0, s >>>
        ( dst, src, index, k );
    hipCheckError( );
#endif
}


/* computed the scaled sum of two dense vector and store
 * the result in a third vector (SAXPY operation in BLAS)
 *
 * inputs:
 *  c, d: scaling constants
 *  v, y: dense vector whose entries to scale
 *  k: number of entries in the vectors
 *  block_size: HIP threads per block
 *  s: HIP stream
 * output:
 *  dest: vector containing the scaled sum
 */
void Vector_Sum( real * const dest, real c, real const * const v,
        real d, real const * const y, unsigned int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    if ( dest == v && dest == y )
    {
        real temp = c + d;

        ret = hipblasDscal( handle, k, &temp, dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }
    }
    else if ( dest == v )
    {
        ret = hipblasDscal( handle, k, &c, dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &d, y, 1, dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }
    }
    else if ( dest == y )
    {
        ret = hipblasDscal( handle, k, &d, dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &c, v, 1, dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }
    }
    else
    {
        ret = hipblasDcopy( handle, k, v, 1, dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDcopy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDscal( handle, k, &c, dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &d, y, 1, dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }
    }
#else
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_sum <<< blocks, block_size, 0, s >>>
        ( dest, c, v, d, y, k );
    hipCheckError( );
#endif
}


void Vector_Sum_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        real d0, real d1, rvec2 const * const y, unsigned int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    if ( dest == v && dest == y )
    {
        real temp0 = c0 + d0;
        real temp1 = c1 + d1;

        ret = hipblasDscal( handle, k, &temp0, (double *) dest, 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDscal( handle, k, &temp1, &((double *) dest)[1], 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }
    }
    else if ( dest == v )
    {
        ret = hipblasDscal( handle, k, &c0, (double *) dest, 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDscal( handle, k, &c1, &((double *) dest)[1], 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &d0, (double *) y, 2, (double *) dest, 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &d1, &((double *) y)[1], 2, &((double *) dest)[1], 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }
    }
    else if ( dest == y )
    {
        ret = hipblasDscal( handle, k, &d0, (double *) dest, 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDscal( handle, k, &d1, &((double *) dest)[1], 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &c0, (double *) v, 2, (double *) dest, 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &c1, &((double *) v)[1], 2, &((double *) dest)[1], 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }
    }
    else
    {
        ret = hipblasDcopy( handle, 2 * k, (double *) v, 1, (double *) dest, 1 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDcopy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDscal( handle, k, &c0, (double *) dest, 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDscal( handle, k, &c1, &((double *) dest)[1], 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDscal failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &d0, (double *) y, 2, (double *) dest, 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }

        ret = hipblasDaxpy( handle, k, &d1, &((double *) y)[1], 2, &((double *) dest)[1], 2 );

        if ( ret != HIPBLAS_STATUS_SUCCESS )
        {
            fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
            exit( RUNTIME_ERROR );
        }
    }
#else
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_sum_rvec2 <<< blocks, block_size, 0, s >>> 
        ( dest, c0, c1, v, d0, d1, y, k );
    hipCheckError( );
#endif
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 *  block_size: HIP threads per block
 *  s: HIP stream
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
void Vector_Add( real * const dest, real c, real const * const v,
        unsigned int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    ret = hipblasDaxpy( handle, k, &c, v, 1, dest, 1 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }
#else
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_add <<< blocks, block_size, 0, s >>>
        ( dest, c, v, k );
    hipCheckError( );
#endif
}


/* add the scaled sum of a dense vector to another vector
 * and store in-place
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in the vectors
 *  block_size: HIP threads per block
 *  s: HIP stream
 * output:
 *  dest: vector to accumulate with the scaled sum
 */
void Vector_Add_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        unsigned int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    ret = hipblasDaxpy( handle, k, &c0, (double *) v, 2, (double *) dest, 2 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }

    ret = hipblasDaxpy( handle, k, &c1, &((double *) v)[1], 2, &((double *) dest)[1], 2 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDaxpy failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }
#else
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_add_rvec2 <<< blocks, block_size, 0, s >>>
        ( dest, c0, c1, v, k );
    hipCheckError( );
#endif
}


#if !defined(USE_HIPBLAS)
/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 *  block_size: HIP threads per block
 *  s: HIP stream
 * output:
 *  dest: vector with the result of the multiplication
 */
static void Vector_Mult( real * const dest, real const * const v1,
        real const * const v2, unsigned int k, int block_size, hipStream_t s )
{
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_mult <<< blocks, block_size, 0, s >>>
        ( dest, v1, v2, k );
    hipCheckError( );
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 *  block_size: HIP threads per block
 *  s: HIP stream
 * output:
 *  dest: vector with the result of the multiplication
 */
static void Vector_Mult_rvec2( rvec2 * const dest, rvec2 const * const v1,
        rvec2 const * const v2, unsigned int k, int block_size, hipStream_t s )
{
    int blocks;

    blocks = (k / block_size)
        + ((k % block_size == 0) ? 0 : 1);

    k_vector_mult_rvec2 <<< blocks, block_size, 0, s >>>
        ( dest, v1, v2, k );
    hipCheckError( );
}
#endif


/* compute the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 *  comm: MPI communicator
 * output:
 *  dot: inner product of the two vector
 */
real Dot( storage * const workspace,
        real const * const v1, real const * const v2,
        unsigned int k, MPI_Comm comm,
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    int ret;
    real sum;
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
    real *spad;
#else
    real temp;
#endif
    hipblasStatus_t ret_hipblas;

    /* global reduction (sum) of local device sums and store on host */
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(real), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[5];

    ret_hipblas = hipblasDdot( handle, k, v1, 1, v2, 1, &spad );

    if ( ret_hipblas != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDdot failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }

    hipStreamSynchronize( s );
    ret = MPI_Allreduce( &spad[k], &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
#else
    ret_hipblas = hipblasDdot( handle, k, v1, 1, v2, 1, &temp );

    if ( ret_hipblas != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDdot failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }

    ret = MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
#endif
#else
    int ret;
    real sum, *spad;
#if !defined(OMPI_HAVE_MPI_EXT_ROCM) || !OMPI_HAVE_MPI_EXT_ROCM
    real temp;
#endif

    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(real) * (k + 1), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[5];

    Vector_Mult( spad, v1, v2, k, block_size, s );

    /* local reduction (sum) on device */
    Hip_Reduction_Sum( spad, &spad[k], k, 5, s );

    /* global reduction (sum) of local device sums and store on host */
#if defined(OMPI_HAVE_MPI_EXT_ROCM) && OMPI_HAVE_MPI_EXT_ROCM
    hipStreamSynchronize( s );
    ret = MPI_Allreduce( &spad[k], &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
#else
    sHipMemcpyAsync( &temp, &spad[k], sizeof(real),
            hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );

    ret = MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
#endif
#endif

    return sum;
}


/* compute the local portions of the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 *  s: HIP stream
 * output:
 *  dot: inner product of the two vector
 */
real Dot_local( storage * const workspace,
        real const * const v1, real const * const v2,
        unsigned int k, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    real sum;
    hipblasStatus_t ret;

    ret = hipblasDdot( handle, k, v1, 1, v2, 1, &sum );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDdot failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }
#else
    real sum, *spad;

    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(real) * (k + 1), __FILE__, __LINE__ );
    spad = (real *) workspace->scratch[5];

    Vector_Mult( spad, v1, v2, k, block_size, s );

    /* local reduction (sum) on device */
    Hip_Reduction_Sum( spad, &spad[k], k, 5, s );

    //TODO: keep result of reduction on devie and pass directly to ROCm-aware MPI
    sHipMemcpyAsync( &sum, &spad[k], sizeof(real),
            hipMemcpyDeviceToHost, s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );
#endif

    return sum;
}


/* compute the local portions of the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 *  s: HIP stream
 * output:
 *  dot: inner product of the two vector
 */
void Dot_local_rvec2( storage * const workspace,
        rvec2 const * const v1, rvec2 const * const v2,
        unsigned int k, real * sum1, real * sum2, 
#if defined(USE_HIPBLAS)
        hipblasHandle_t handle
#else
        int block_size, hipStream_t s
#endif
        )
{
#if defined(USE_HIPBLAS)
    hipblasStatus_t ret;

    ret = hipblasDdot( handle, k, (double *) v1, 2, (double *) v2, 2, sum1 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDdot failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }

    ret = hipblasDdot( handle, k, &((double *) v1)[1], 2, &((double *) v2)[1], 2, sum2 );

    if ( ret != HIPBLAS_STATUS_SUCCESS )
    {
        fprintf( stderr, "[ERROR] hipblasDdot failure. Terminating...\n" );
        exit( RUNTIME_ERROR );
    }
#else
    rvec2 sum, *spad;

    sHipCheckMalloc( &workspace->scratch[5], &workspace->scratch_size[5],
            sizeof(rvec2) * (k + 1), __FILE__, __LINE__ );
    spad = (rvec2 *) workspace->scratch[5];

    Vector_Mult_rvec2( spad, v1, v2, k, block_size, s );

    /* local reduction (sum) on device */
    Hip_Reduction_Sum( spad, &spad[k], k, 5, s );

    //TODO: keep result of reduction on devie and pass directly to ROCm-aware MPI
    sHipMemcpyAsync( &sum, &spad[k], sizeof(rvec2), hipMemcpyDeviceToHost,
            s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );

    *sum1 = sum[0];
    *sum2 = sum[1];
#endif
}
