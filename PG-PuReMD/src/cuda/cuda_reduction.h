
#ifndef __CUDA_REDUCTION_H__
#define __CUDA_REDUCTION_H__

#include "../reax_types.h"


void Cuda_Reduction_Sum( int *, int *, size_t );

void Cuda_Reduction_Sum( real *, real *, size_t );

//void Cuda_Reduction_Sum( rvec *, rvec *, size_t );

void Cuda_Reduction_Max( int *, int *, size_t );

void Cuda_Scan_Excl_Sum( int *, int *, size_t );

CUDA_GLOBAL void k_reduction( const real *, real *, const size_t );

CUDA_GLOBAL void k_reduction_rvec( rvec *, rvec *, size_t );

CUDA_GLOBAL void k_reduction_rvec2( rvec2 *, rvec2 *, size_t );


#endif
