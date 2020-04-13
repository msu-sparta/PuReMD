
#include "cuda_reduction.h"

#include "cuda_utils.h"

#include "../vector.h"

#include "../cub/cub/device/device_reduce.cuh"
#include "../cub/cub/device/device_scan.cuh"


/* mask used to determine which threads within a warp participate in operations */
#define FULL_MASK (0xFFFFFFFF)


//struct RvecSum
//{
//    template <typename T>
//    __device__ __forceinline__
//    T operator()(const T &a, const T &b) const
//    {
//        T c;
//        return c {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
//    }
//};


/* Perform a device-wide reduction (sum operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction */
void Cuda_Reduction_Sum( int *d_array, int *d_dest, size_t n )
{
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    /* determine temporary device storage requirements */
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes,
            d_array, d_dest, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* allocate temporary storage */
    cuda_malloc( &d_temp_storage, temp_storage_bytes, FALSE,
            "Cuda_Reduction_Sum::d_temp_storage" );

    /* run sum-reduction */
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes,
            d_array, d_dest, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* deallocate temporary storage */
    cuda_free( d_temp_storage, "Cuda_Reduction_Sum::d_temp_storage" );
}


/* Perform a device-wide reduction (sum operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction */
void Cuda_Reduction_Sum( real *d_array, real *d_dest, size_t n )
{
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    /* determine temporary device storage requirements */
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes,
            d_array, d_dest, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* allocate temporary storage */
    cuda_malloc( &d_temp_storage, temp_storage_bytes, FALSE,
            "Cuda_Reduction_Sum::temp_storage" );

    /* run sum-reduction */
    cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes,
            d_array, d_dest, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* deallocate temporary storage */
    cuda_free( d_temp_storage, "Cuda_Reduction_Sum::temp_storage" );
}


///* Perform a device-wide reduction (sum operation)
// *
// * d_array: device array to reduce
// * d_dest: device pointer to hold result of reduction */
//void Cuda_Reduction_Sum( rvec *d_array, rvec *d_dest, size_t n )
//{
//    void *d_temp_storage = NULL;
//    size_t temp_storage_bytes = 0;
//    RvecSum sum_op;
//    rvec init = {0.0, 0.0, 0.0};
//
//    /* determine temporary device storage requirements */
//    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes,
//            d_array, d_dest, n, sum_op, init );
//    cudaDeviceSynchronize( );
//    cudaCheckError( );
//
//    /* allocate temporary storage */
//    cuda_malloc( &d_temp_storage, temp_storage_bytes, FALSE,
//            "cub::reduce::temp_storage" );
//
//    /* run sum-reduction */
//    cub::DeviceReduce::Reduce( d_temp_storage, temp_storage_bytes,
//            d_array, d_dest, n, sum_op, init );
//    cudaDeviceSynchronize( );
//    cudaCheckError( );
//
//    /* deallocate temporary storage */
//    cuda_free( d_temp_storage, "cub::reduce::temp_storage" );
//}


/* Perform a device-wide reduction (max operation)
 *
 * d_array: device array to reduce
 * d_dest: device pointer to hold result of reduction */
void Cuda_Reduction_Max( int *d_array, int *d_dest, size_t n )
{
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    /* determine temporary device storage requirements */
    cub::DeviceReduce::Max( d_temp_storage, temp_storage_bytes,
            d_array, d_dest, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* allocate temporary storage */
    cuda_malloc( &d_temp_storage, temp_storage_bytes, FALSE,
            "Cuda_Reduction_Max::temp_storage" );

    /* run exclusive prefix sum */
    cub::DeviceReduce::Max( d_temp_storage, temp_storage_bytes,
            d_array, d_dest, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* deallocate temporary storage */
    cuda_free( d_temp_storage, "Cuda_Reduction_Max::temp_storage" );
}


/* Perform a device-wide scan (partial sum operation)
 *
 * d_src: device array to scan
 * d_dest: device array to hold result of scan */
void Cuda_Scan_Excl_Sum( int *d_src, int *d_dest, size_t n )
{
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    /* determine temporary device storage requirements */
    cub::DeviceScan::ExclusiveSum( d_temp_storage, temp_storage_bytes,
            d_src, d_dest, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* allocate temporary storage */
    cuda_malloc( &d_temp_storage, temp_storage_bytes, FALSE,
            "Cuda_Scan_Excl_Sum::temp_storage" );

    /* run exclusive prefix sum */
    cub::DeviceScan::ExclusiveSum( d_temp_storage, temp_storage_bytes,
            d_src, d_dest, n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    /* deallocate temporary storage */
    cuda_free( d_temp_storage, "Cuda_Scan_Excl_Sum::temp_storage" );
}


/* Performs a device-wide partial reduction (sum) on input in 2 stages:
 *  1) Perform a warp-level sum of parts of input assigned to warps
 *  2) Perform an block-level sum of the warp-local partial sums
 * The block-level sums are written to global memory pointed to by results
 *  in accordance to their block IDs.
 */
CUDA_GLOBAL void k_reduction_rvec( rvec *input, rvec *results, size_t n )
{
    extern __shared__ rvec data_s[];
    rvec data;
    unsigned int i, mask;
    int offset;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    if ( i < n )
    {
        rvec_Copy( data, input[i] );

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            data[0] += __shfl_down_sync( mask, data[0], offset );
            data[1] += __shfl_down_sync( mask, data[1], offset );
            data[2] += __shfl_down_sync( mask, data[2], offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            rvec_Copy( data_s[threadIdx.x >> 5], data );
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            rvec_Add( data_s[threadIdx.x], data_s[threadIdx.x + offset] );
        }

        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        rvec_Copy( results[blockIdx.x], data_s[0] );
    }
}


CUDA_GLOBAL void k_reduction_rvec2( rvec2 *input, rvec2 *results, size_t n )
{
    extern __shared__ rvec2 data_rvec2_s[];
    rvec2 data;
    unsigned int i, mask;
    int offset;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    mask = __ballot_sync( FULL_MASK, i < n );

    if ( i < n )
    {
        data[0] = input[i][0];
        data[1] = input[i][1];

        /* warp-level sum using registers within a warp */
        for ( offset = 16; offset > 0; offset /= 2 )
        {
            data[0] += __shfl_down_sync( mask, data[0], offset );
            data[1] += __shfl_down_sync( mask, data[1], offset );
        }

        /* first thread within a warp writes warp-level sum to shared memory */
        if ( threadIdx.x % 32 == 0 )
        {
            data_rvec2_s[threadIdx.x >> 5][0] = data[0];
            data_rvec2_s[threadIdx.x >> 5][1] = data[1];
        }
    }
    __syncthreads( );

    /* block-level sum using shared memory */
    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            data_rvec2_s[threadIdx.x][0] += data_rvec2_s[threadIdx.x + offset][0];
            data_rvec2_s[threadIdx.x][1] += data_rvec2_s[threadIdx.x + offset][1];
        }

        __syncthreads( );
    }

    /* one thread writes the block-level partial sum
     * of the reduction back to global memory */
    if ( threadIdx.x == 0 )
    {
        results[blockIdx.x][0] = data_rvec2_s[0][0];
        results[blockIdx.x][1] = data_rvec2_s[0][1];
    }
}
