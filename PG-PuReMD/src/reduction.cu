
#include "reduction.h"
#include "vector.h"

#include "cuda_shuffle.h"

CUDA_GLOBAL void k_reduction(const real *input, real *per_block_results, const size_t n)
{
#if defined(__SM_35__)
    extern __shared__ real my_results[];
    real sdata;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
        x = input[i];

    sdata = x;
    __syncthreads();

    for(int z = 16; z >=1; z/=2)
        sdata+= shfl ( sdata, z);

    if (threadIdx.x % 32 == 0)
        my_results[threadIdx.x >> 5] = sdata;

    __syncthreads ();

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset)
            my_results[threadIdx.x] += my_results[threadIdx.x + offset];

        __syncthreads();
    }

    if(threadIdx.x == 0)
        per_block_results[blockIdx.x] = my_results[0];

#else

    extern __shared__ real sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdata[0];
    }
#endif
}

CUDA_GLOBAL void k_reduction_rvec (rvec *input, rvec *results, size_t n)
{
#if defined(__SM_35__)


    extern __shared__ rvec my_rvec[];
    rvec sdata;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    rvec_MakeZero( sdata );

    if(i < n)
        rvec_Copy (sdata, input[i]);

    __syncthreads();

    for(int z = 16; z >=1; z/=2){
        sdata[0] += shfl ( sdata[0], z);
        sdata[1] += shfl ( sdata[1], z);
        sdata[2] += shfl ( sdata[2], z);
    }

    if (threadIdx.x % 32 == 0)
        rvec_Copy( my_rvec[threadIdx.x >> 5] , sdata );

    __syncthreads ();

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset)
            rvec_Add( my_rvec[threadIdx.x], my_rvec[threadIdx.x + offset] );

        __syncthreads();
    }

    if(threadIdx.x == 0)
        rvec_Add (results[blockIdx.x], my_rvec[0]);


#else


    extern __shared__ rvec svec_data[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    rvec x;

    rvec_MakeZero (x);

    if(i < n)
    {
        rvec_Copy (x, input[i]);
    }

    rvec_Copy (svec_data[threadIdx.x], x);
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            rvec_Add (svec_data[threadIdx.x], svec_data[threadIdx.x + offset]);
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        //rvec_Copy (results[blockIdx.x], svec_data[0]);
        rvec_Add (results[blockIdx.x], svec_data[0]);
    }
#endif


}

CUDA_GLOBAL void k_reduction_rvec2 (rvec2 *input, rvec2 *results, size_t n)
{
#if defined(__SM_35__)

    extern __shared__ rvec2 my_rvec2[];
    rvec2 sdata;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[0] = 0.0;
    sdata[1] = 0.0;

    if(i < n){
        sdata[0] = input[i][0];
        sdata[1] = input[i][1];
    }

    __syncthreads();

    for(int z = 16; z >=1; z/=2){
        sdata[0] += shfl ( sdata[0], z);
        sdata[1] += shfl ( sdata[1], z);
    }

    if (threadIdx.x % 32 == 0){
        my_rvec2[threadIdx.x >> 5][0] = sdata[0];
        my_rvec2[threadIdx.x >> 5][1] = sdata[1];
    }

    __syncthreads ();

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset){
            my_rvec2[threadIdx.x][0] += my_rvec2[threadIdx.x + offset][0];
            my_rvec2[threadIdx.x][1] += my_rvec2[threadIdx.x + offset][1];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0){
        results[blockIdx.x][0] = my_rvec2[0][0];
        results[blockIdx.x][1] = my_rvec2[0][1];
    }

#else
    extern __shared__ rvec2 svec2_data[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    rvec2 x;

    x[0] = 0.0;
    x[1] = 0.0;

    if(i < n)
    {
        x[0] += input[i][0];
        x[1] += input[i][1];
    }

    svec2_data [threadIdx.x][0] = x[0];
    svec2_data [threadIdx.x][1] = x[1];
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            svec2_data [threadIdx.x][0] += svec2_data [threadIdx.x + offset][0];
            svec2_data [threadIdx.x][1] += svec2_data [threadIdx.x + offset][1];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        //rvec_Copy (results[blockIdx.x], svec_data[0]);
        results [blockIdx.x][0] += svec2_data [0][0];
        results [blockIdx.x][1] += svec2_data [0][1];
    }
#endif
}

CUDA_GLOBAL void k_dot (const real *a, const real *b, real *per_block_results, const size_t n )
{
#if defined(__SM_35__)

    extern __shared__ real my_dot[];
    real sdot;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdot = 0.0;
    if(i < n)
        sdot = a[i] * b[i];

    __syncthreads();

    for(int z = 16; z >=1; z/=2)
        sdot += shfl ( sdot, z);

    if (threadIdx.x % 32 == 0)
        my_dot[threadIdx.x >> 5] = sdot;

    __syncthreads ();

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset)
            my_dot[threadIdx.x] += my_dot[threadIdx.x + offset];

        __syncthreads();
    }

    if(threadIdx.x == 0)
        per_block_results[blockIdx.x] = my_dot[0];

#else

    extern __shared__ real sdot[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        x = a[i] * b[i];
    }
    sdot[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            sdot[threadIdx.x] += sdot[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        per_block_results[blockIdx.x] = sdot[0];
    }

#endif

}

CUDA_GLOBAL void k_norm (const real *input, real *per_block_results, const size_t n, int pass)
{
#if defined(__SM_35__)

    extern __shared__ real my_norm[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real snorm = 0.0;

    if(i < n)
        snorm = SQR (input[i]);

    __syncthreads();

    for(int z = 16; z >=1; z/=2)
        snorm += shfl ( snorm, z);

    if (threadIdx.x % 32 == 0)
        my_norm[threadIdx.x >> 5] = snorm;

    __syncthreads ();

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset)
            my_norm[threadIdx.x] += my_norm[threadIdx.x + offset];

        __syncthreads();
    }

    if(threadIdx.x == 0)
        per_block_results[blockIdx.x] = my_norm[0];

#else
    extern __shared__ real snorm[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
        x = SQR (input[i]);

    snorm[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            snorm[threadIdx.x] += snorm[threadIdx.x + offset];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0)
        per_block_results[blockIdx.x] = snorm[0];


#endif


}

CUDA_GLOBAL void k_norm_rvec2 (const rvec2 *input, rvec2 *per_block_results, const size_t n, int pass)
{
#if defined(__SM_35__)

    extern __shared__ rvec2 my_norm2[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    rvec2 snorm2;
    snorm2[0] = snorm2[1] = 0;

    if(i < n) {
        if (pass == INITIAL) {    
            snorm2[0] = SQR (input[i][0]);
            snorm2[1] = SQR (input[i][1]);
        } else {
            snorm2[0] = input[i][0];
            snorm2[1] = input[i][1];
        }
    }
    __syncthreads();

    for(int z = 16; z >=1; z/=2){
        snorm2[0] += shfl ( snorm2[0], z);
        snorm2[1] += shfl ( snorm2[1], z);
    }

    if (threadIdx.x % 32 == 0){
        my_norm2[threadIdx.x >> 5][0] = snorm2[0];
        my_norm2[threadIdx.x >> 5][1] = snorm2[1];
    }

    __syncthreads ();

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset){
            my_norm2[threadIdx.x][0] += my_norm2[threadIdx.x + offset][0];
            my_norm2[threadIdx.x][1] += my_norm2[threadIdx.x + offset][1];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0) {
        per_block_results[blockIdx.x][0] = my_norm2[0][0];
        per_block_results[blockIdx.x][1] = my_norm2[0][1];
    }

#else

    extern __shared__ rvec2 snorm2[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    rvec2 x;
    x[0] = x[1] = 0;

    if(i < n) {
        if (pass == INITIAL) {    
            x[0] = SQR (input[i][0]);
            x[1] = SQR (input[i][1]);
        } else {
            x[0] = input[i][0];
            x[1] = input[i][1];
        }
    }

    snorm2[threadIdx.x][0] = x[0];
    snorm2[threadIdx.x][1] = x[1];
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            snorm2[threadIdx.x][0] += snorm2[threadIdx.x + offset][0];
            snorm2[threadIdx.x][1] += snorm2[threadIdx.x + offset][1];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0) {
        per_block_results[blockIdx.x][0] = snorm2[0][0];
        per_block_results[blockIdx.x][1] = snorm2[0][1];
    }
#endif
}

CUDA_GLOBAL void k_dot_rvec2 (const rvec2 *a, rvec2 *b, rvec2 *res, const size_t n)
{
#if defined(__SM_35__)

    extern __shared__ rvec2 my_dot2[];
    rvec2 sdot2;

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdot2[0] = sdot2[1] = 0;

    if(i < n) {
        sdot2[0] = a[i][0] * b[i][0];
        sdot2[1] = a[i][1] * b[i][1];
    }

    __syncthreads();

    for(int z = 16; z >=1; z/=2){
        sdot2[0] += shfl ( sdot2[0], z);
        sdot2[1] += shfl ( sdot2[1], z);
    }

    if (threadIdx.x % 32 == 0){
        my_dot2[threadIdx.x >> 5][0] = sdot2[0];
        my_dot2[threadIdx.x >> 5][1] = sdot2[1];
    }

    __syncthreads ();

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset){
            my_dot2[threadIdx.x][0] += my_dot2[threadIdx.x + offset][0];
            my_dot2[threadIdx.x][1] += my_dot2[threadIdx.x + offset][1];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0) {
        res[blockIdx.x][0] = my_dot2[0][0];
        res[blockIdx.x][1] = my_dot2[0][1];
    }


#else
    extern __shared__ rvec2 sdot2[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    rvec2 x;
    x[0] = x[1] = 0;

    if(i < n) {
        x[0] = a[i][0] * b[i][0];
        x[1] = a[i][1] * b[i][1];
    }

    sdot2[threadIdx.x][0] = x[0];
    sdot2[threadIdx.x][1] = x[1];
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            sdot2[threadIdx.x][0] += sdot2[threadIdx.x + offset][0];
            sdot2[threadIdx.x][1] += sdot2[threadIdx.x + offset][1];
        }

        __syncthreads();
    }

    if(threadIdx.x == 0) {
        res[blockIdx.x][0] = sdot2[0][0];
        res[blockIdx.x][1] = sdot2[0][1];
    }
#endif
}

//////////////////////////////////////////////////
//vector functions
//////////////////////////////////////////////////

CUDA_GLOBAL void k_vector_sum( real* dest, real c, real* v, real d, real* y, int k )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= k) return;

    dest[i] = c * v[i] + d * y[i];
}


CUDA_GLOBAL void k_vector_mul( real* dest, real* v, real* y, int k )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= k) return;

    dest[i] = v[i] * y[i];
}

CUDA_GLOBAL void k_rvec2_mul( rvec2* dest, rvec2* v, rvec2* y, int k )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= k) return;

    dest[i][0] = v[i][0] * y[i][0];
    dest[i][1] = v[i][1] * y[i][1];
}

CUDA_GLOBAL void k_rvec2_pbetad (rvec2 *dest, rvec2 *a, 
        real beta0, real beta1, 
        rvec2 *b, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n) return;

    dest[i][0] = a[i][0] + beta0 * b[i][0];
    dest[i][1] = a[i][1] + beta1 * b[i][1];
}
