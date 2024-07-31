#include "hip/hip_runtime.h"

#include "hip_reduction.h"

#include "hip_helpers.h"
#include "hip_utils.h"

#include "../vector.h"

#include <hipcub/device/device_reduce.hpp>
#include <hipcub/device/device_scan.hpp>
#include <hipcub/block/block_reduce.hpp>


//struct RvecSum
//{
//    template <typename T>
//    GPU_HOST_DEVICE __forceinline__
//    T operator()(const T &a, const T &b) const
//    {
//        T c;
//        return c {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
//    }
//};


/* Performs a device-wide partial reduction (sum) on input in 2 stages:
 *  1) Perform a warp-level sum of parts of input assigned to warps
 *  2) Perform an block-level sum of the warp-local partial sums
 * The block-level sums are written to global memory pointed to by results
 *  in accordance to their block IDs.
 */
GPU_GLOBAL void k_reduction_rvec( rvec const * const input, rvec * const results, size_t n )
{
    extern __shared__ hipcub::BlockReduce<double, GPU_BLOCK_SIZE>::TempStorage temp_block[];
    rvec data;
    unsigned int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        rvec_Copy( data, input[i] );
    }
    else
    {
        rvec_MakeZero( data );
    }

    data[0] = hipcub::BlockReduce<double, GPU_BLOCK_SIZE>(*temp_block).Sum(data[0]);
    __syncthreads( );
    data[1] = hipcub::BlockReduce<double, GPU_BLOCK_SIZE>(*temp_block).Sum(data[1]);
    __syncthreads( );
    data[2] = hipcub::BlockReduce<double, GPU_BLOCK_SIZE>(*temp_block).Sum(data[2]);

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        atomicAdd( (double *) &results[0][0], (double) data[0] );
        atomicAdd( (double *) &results[0][1], (double) data[1] );
        atomicAdd( (double *) &results[0][2], (double) data[2] );
    }
}


GPU_GLOBAL void k_reduction_rvec2( rvec2 const * const input, rvec2 * const results, size_t n )
{
    extern __shared__ hipcub::BlockReduce<double, GPU_BLOCK_SIZE>::TempStorage temp_block[];
    unsigned int i;
    rvec2 data;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        data[0] = input[i][0];
        data[1] = input[i][1];
    }
    else
    {
        data[0] = 0.0;
        data[1] = 0.0;
    }

    data[0] = hipcub::BlockReduce<double, GPU_BLOCK_SIZE>(*temp_block).Sum(data[0]);
    __syncthreads( );
    data[1] = hipcub::BlockReduce<double, GPU_BLOCK_SIZE>(*temp_block).Sum(data[1]);

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        atomicAdd( (double *) &results[0][0], (double) data[0] );
        atomicAdd( (double *) &results[0][1], (double) data[1] );
    }
}


/* Perform a device-wide reduction (sum operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction
 * n: num. elements to reduce
 * s: stream to perform the reduction in
 */
void Hip_Reduction_Sum( int *d_array, int *d_dest, size_t n, int s_index, hipStream_t s )
{
    static void *d_temp_storage[MAX_GPU_STREAMS] = { NULL };
    static size_t temp_storage_bytes[MAX_GPU_STREAMS] = { 0 };
    void *temp = NULL;
    size_t temp_bytes = 0;

    /* determine temporary device storage requirements */
    hipcub::DeviceReduce::Sum( temp, temp_bytes, d_array, d_dest, n, s );
    hipCheckError( );

    /* allocate temporary storage */
    sHipCheckMalloc( &d_temp_storage[s_index], &temp_storage_bytes[s_index],
            temp_bytes, __FILE__, __LINE__ );

    /* run sum-reduction */
    hipcub::DeviceReduce::Sum( d_temp_storage[s_index], temp_storage_bytes[s_index],
            d_array, d_dest, n, s );
    hipCheckError( );
}


/* Perform a device-wide reduction (sum operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction
 * n: num. elements to reduce
 * s: stream to perform the reduction in
 */
void Hip_Reduction_Sum( real *d_array, real *d_dest, size_t n, int s_index, hipStream_t s )
{
    static void *d_temp_storage[MAX_GPU_STREAMS] = { NULL };
    static size_t temp_storage_bytes[MAX_GPU_STREAMS] = { 0 };
    void *temp = NULL;
    size_t temp_bytes = 0;

    /* determine temporary device storage requirements */
    hipcub::DeviceReduce::Sum( temp, temp_bytes, d_array, d_dest, n, s );
    hipCheckError( );

    /* allocate temporary storage */
    sHipCheckMalloc( &d_temp_storage[s_index], &temp_storage_bytes[s_index],
            temp_bytes, __FILE__, __LINE__ );

    /* run sum-reduction */
    hipcub::DeviceReduce::Sum( d_temp_storage[s_index], temp_storage_bytes[s_index],
            d_array, d_dest, n, s );
    hipCheckError( );
}


/* Perform a device-wide reduction (sum operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction
 * n: num. elements to reduce
 * s: stream to perform the reduction in
 */
void Hip_Reduction_Sum( rvec *d_array, rvec *d_dest, size_t n,
        int s_index, hipStream_t s )
{
//    void *d_temp_storage = NULL;
//    size_t temp_storage_bytes = 0;
//    RvecSum sum_op;
//    rvec init = {0.0, 0.0, 0.0};
//
//    /* determine temporary device storage requirements */
//    hipcub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes,
//            d_array, d_dest, n, sum_op, init, s );
//    hipCheckError( );
//
//    /* allocate temporary storage */
//    sHipMalloc( &d_temp_storage, temp_storage_bytes, __FILE__, __LINE__ );
//
//    /* run sum-reduction */
//    hipcub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes,
//            d_array, d_dest, n, sum_op, init, s );
//    hipCheckError( );
//
//    /* deallocate temporary storage */
//    sHipFree( d_temp_storage, __FILE__, __LINE__ );

    int blocks;

    blocks = n / GPU_BLOCK_SIZE
        + ((n % GPU_BLOCK_SIZE == 0) ? 0 : 1);

    sHipMemsetAsync( d_dest, 0, sizeof(rvec), s, __FILE__, __LINE__ );

    k_reduction_rvec <<< blocks, GPU_BLOCK_SIZE,
                     sizeof(hipcub::BlockReduce<double, GPU_BLOCK_SIZE>::TempStorage), s >>> 
        ( d_array, d_dest, n );
    hipCheckError( ); 
}


/* Perform a device-wide reduction (sum operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction
 * n: num. elements to reduce
 * s: stream to perform the reduction in
 */
void Hip_Reduction_Sum( rvec2 *d_array, rvec2 *d_dest, size_t n,
        int s_index, hipStream_t s )
{
    int blocks;

    blocks = n / GPU_BLOCK_SIZE
        + ((n % GPU_BLOCK_SIZE == 0) ? 0 : 1);

    sHipMemsetAsync( d_dest, 0, sizeof(rvec2), s, __FILE__, __LINE__ );

    k_reduction_rvec2 <<< blocks, GPU_BLOCK_SIZE,
                     sizeof(hipcub::BlockReduce<double, GPU_BLOCK_SIZE>::TempStorage), s >>> 
        ( d_array, d_dest, n );
    hipCheckError( ); 
}


/* Perform a device-wide scan (partial sum operation)
 *
 * d_src: device array to scan
 * d_dest: device array to hold result of scan */
void Hip_Scan_Excl_Sum( int *d_src, int *d_dest, size_t n, int s_index, hipStream_t s )
{
    static void *d_temp_storage[MAX_GPU_STREAMS] = { NULL };
    static size_t temp_storage_bytes[MAX_GPU_STREAMS] = { 0 };
    void *temp = NULL;
    size_t temp_bytes = 0;

    /* determine temporary device storage requirements */
    hipcub::DeviceScan::ExclusiveSum( temp, temp_bytes, d_src, d_dest, n, s );
    hipCheckError( );

    /* allocate temporary storage */
    sHipCheckMalloc( &d_temp_storage[s_index], &temp_storage_bytes[s_index],
            temp_bytes, __FILE__, __LINE__ );

    /* run exclusive prefix sum */
    hipcub::DeviceScan::ExclusiveSum( d_temp_storage[s_index], temp_storage_bytes[s_index],
            d_src, d_dest, n, s );
    hipCheckError( );
}
