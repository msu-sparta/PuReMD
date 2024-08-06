
#ifndef __GPU_REDUCTION_H_
#define __GPU_REDUCTION_H_

#include "../reax_types.h"


void GPU_Reduction_Sum( int *, int *, size_t, int, cudaStream_t );

void GPU_Reduction_Sum( real *, real *, size_t, int, cudaStream_t );

void GPU_Reduction_Sum( rvec *, rvec *, size_t, int, cudaStream_t );

void GPU_Reduction_Sum( rvec2 *, rvec2 *, size_t, int, cudaStream_t );

void GPU_Scan_Excl_Sum( int *, int *, size_t, int, cudaStream_t );


#endif
