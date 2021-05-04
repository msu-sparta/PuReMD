#include "cuda_dense_lin_alg.h"

#include "cuda_reduction.h"
#include "cuda_utils.h"

#include "../comm_tools.h"


/* sets all entries of a dense vector to zero
 *
 * inputs:
 *  v: dense vector
 *  k: number of entries in v
 * output: v with entries set to zero
 */
CUDA_GLOBAL void k_vector_makezero( real * const v, unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    v[i] = ZERO;
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
CUDA_GLOBAL void k_vector_copy( real * const dest, real const * const v,
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
CUDA_GLOBAL void k_vector_copy_rvec2( rvec2 * const dest, rvec2 const * const v,
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


CUDA_GLOBAL void k_vector_copy_from_rvec2( real * const dst, rvec2 const * const src,
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


CUDA_GLOBAL void k_vector_copy_to_rvec2( rvec2 * const dst, real const * const src,
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


/* scales the entries of a dense vector by a constant
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in v
 * output:
 *  dest: with entries scaled
 */
CUDA_GLOBAL void k_vector_scale( real * const dest, real c, real const * const v,
        unsigned int k )
{
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= k )
    {
        return;
    }

    dest[i] = c * v[i];
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
CUDA_GLOBAL void k_vector_sum( real * const dest, real c, real const * const v,
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


CUDA_GLOBAL void k_vector_sum_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v, 
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
CUDA_GLOBAL void k_vector_add( real * const dest, real c, real const * const v,
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
CUDA_GLOBAL void k_vector_add_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
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
CUDA_GLOBAL void k_vector_mult( real * const dest, real const * const v1,
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
CUDA_GLOBAL void k_vector_mult_rvec2( rvec2 * const dest, rvec2 const * const v1,
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


/* sets all entries of a dense vector to zero
 *
 * inputs:
 *  v: dense vector
 *  k: number of entries in v
 * output: v with entries set to zero
 */
void Vector_MakeZero( real * const v, unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_makezero <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( v, k );
    cudaCheckError( );
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
void Vector_Copy( real * const dest, real const * const v,
        unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_copy <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, v, k );
    cudaCheckError( );
}


/* copy the entries from one vector to another
 *
 * inputs:
 *  v: dense vector to copy
 *  k: number of entries in v
 * output:
 *  dest: vector copied into
 */
void Vector_Copy_rvec2( rvec2 * const dest, rvec2 const * const v,
        unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_copy_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, v, k );
    cudaCheckError( );
}


void Vector_Copy_From_rvec2( real * const dst, rvec2 const * const src,
        int index, int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_copy_from_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dst, src, index, k );
    cudaCheckError( );
}


void Vector_Copy_To_rvec2( rvec2 * const dst, real const * const src,
        int index, int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_copy_to_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dst, src, index, k );
    cudaCheckError( );
}


/* scales the entries of a dense vector by a constant
 *
 * inputs:
 *  c: scaling constant
 *  v: dense vector whose entries to scale
 *  k: number of entries in v
 * output:
 *  dest: with entries scaled
 */
void Vector_Scale( real * const dest, real c, real const * const v,
        unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_scale <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, c, v, k );
    cudaCheckError( );
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
void Vector_Sum( real * const dest, real c, real const * const v,
        real d, real const * const y, unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_sum <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, c, v, d, y, k );
    cudaCheckError( );
}


void Vector_Sum_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        real d0, real d1, rvec2 const * const y, unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_sum_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>> 
        ( dest, c0, c1, v, d0, d1, y, k );
    cudaCheckError( );
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
void Vector_Add( real * const dest, real c, real const * const v,
        unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_add <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, c, v, k );
    cudaCheckError( );
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
void Vector_Add_rvec2( rvec2 * const dest, real c0, real c1, rvec2 const * const v,
        unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_add_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, c0, c1, v, k );
    cudaCheckError( );
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 * output:
 *  dest: vector with the result of the multiplication
 */
void Vector_Mult( real * const dest, real const * const v1,
        real const * const v2, unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_mult <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, v1, v2, k );
    cudaCheckError( );
}


/* element-wise multiplication of a dense vector to another vector
 *
 * inputs:
 *  v1, v2: dense vectors whose entries to multiply
 *  k: number of entries in the vectors
 * output:
 *  dest: vector with the result of the multiplication
 */
void Vector_Mult_rvec2( rvec2 * const dest, rvec2 const * const v1,
        rvec2 const * const v2, unsigned int k, cudaStream_t s )
{
    int blocks;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    k_vector_mult_rvec2 <<< blocks, DEF_BLOCK_SIZE, 0, s >>>
        ( dest, v1, v2, k );
    cudaCheckError( );
}


/* compute the 2-norm (Euclidean) of a dense vector
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1: dense vector
 *  k: number of entries in the vector
 *  comm: MPI communicator
 *  s: CUDA stream
 * output:
 *  norm: 2-norm
 */
real Norm( storage * const workspace,
        real const * const v1, unsigned int k, MPI_Comm comm, cudaStream_t s )
{
    return SQRT( Dot( workspace, v1, v1, k, comm, s ) );
}


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
        unsigned int k, MPI_Comm comm, cudaStream_t s )
{
    int ret;
    real sum, *spad;
//#if !defined(MPIX_CUDA_AWARE_SUPPORT) || !MPIX_CUDA_AWARE_SUPPORT
    real temp;
//#endif

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(real) * (k + 1), "Dot::workspace->scratch" );
    spad = (real *) workspace->scratch;

    Vector_Mult( spad, v1, v2, k );

    /* local reduction (sum) on device */
    Cuda_Reduction_Sum( spad, &spad[k], k, s );

    /* global reduction (sum) of local device sums and store on host */
//#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
//    ret = MPI_Allreduce( &spad[k], &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
//    Check_MPI_Error( ret, __FILE__, __LINE__ );
//#else
    sCudaMemcpyAsync( &temp, &spad[k], sizeof(real),
            cudaMemcpyDeviceToHost, s, __FILE__, __LINE__ );
    cudaStreamSynchronize( s );

    ret = MPI_Allreduce( &temp, &sum, 1, MPI_DOUBLE, MPI_SUM, comm );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
//#endif

    return sum;
}


/* compute the local portions of the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 * output:
 *  dot: inner product of the two vector
 */
real Dot_local( storage * const workspace,
        real const * const v1, real const * const v2,
        unsigned int k, cudaStream_t s )
{
    real sum, *spad;

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(real) * (k + 1), "Dot_local::workspace->scratch" );
    spad = (real *) workspace->scratch;

    Vector_Mult( spad, v1, v2, k );

    /* local reduction (sum) on device */
    Cuda_Reduction_Sum( spad, &spad[k], k, s );

    //TODO: keep result of reduction on devie and pass directly to CUDA-aware MPI
    sCudaMemcpyAsync( &sum, &spad[k], sizeof(real),
            cudaMemcpyDeviceToHost, s, __FILE__, __LINE__ );
    cudaStreamSynchronize( s );

    return sum;
}


/* compute the local portions of the inner product of two dense vectors
 *
 * inputs:
 *  workspace: storage container for workspace structures
 *  v1, v2: dense vectors
 *  k: number of entries in the vectors
 * output:
 *  dot: inner product of the two vector
 */
void Dot_local_rvec2( storage * const workspace,
        rvec2 const * const v1, rvec2 const * const v2,
        unsigned int k, real * sum1, real * sum2, cudaStream_t s )
{
    int blocks;
    rvec2 sum, *spad;

    blocks = (k / DEF_BLOCK_SIZE)
        + ((k % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(rvec2) * (k + blocks + 1), "Dot_local_rvec2::workspace->scratch" );
    spad = (rvec2 *) workspace->scratch;

    Vector_Mult_rvec2( spad, v1, v2, k );

    /* local reduction (sum) on device */
//    Cuda_Reduction_Sum( spad, &spad[k], k, s );

    k_reduction_rvec2 <<< blocks, DEF_BLOCK_SIZE,
                      sizeof(rvec2) * (DEF_BLOCK_SIZE / 32), s >>>
        ( spad, &spad[k], k );
    cudaCheckError( );

    k_reduction_rvec2 <<< 1, ((blocks + 31) / 32) * 32,
                      sizeof(rvec2) * ((blocks + 31) / 32), s >>>
        ( &spad[k], &spad[k + blocks], blocks );
    cudaCheckError( );

    //TODO: keep result of reduction on devie and pass directly to CUDA-aware MPI
    sCudaMemcpyAsync( &sum, &spad[k + blocks], sizeof(rvec2),
            cudaMemcpyDeviceToHost, s, __FILE__, __LINE__ );
    cudaStreamSynchronize( s );

    *sum1 = sum[0];
    *sum2 = sum[1];
}
