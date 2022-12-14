
#ifndef __CUDA_REDUCTION_H__
#define __CUDA_REDUCTION_H__

#include "../reax_types.h"


void Cuda_Reduction_Sum( int *, int *, size_t, int, cudaStream_t );

void Cuda_Reduction_Sum( real *, real *, size_t, int, cudaStream_t );

void Cuda_Reduction_Sum( rvec *, rvec *, size_t, int, cudaStream_t );

void Cuda_Reduction_Sum( rvec2 *, rvec2 *, size_t, int, cudaStream_t );

void Cuda_Scan_Excl_Sum( int *, int *, size_t, int, cudaStream_t );


#endif
