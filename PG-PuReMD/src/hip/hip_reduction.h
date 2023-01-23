
#ifndef __HIP_REDUCTION_H__
#define __HIP_REDUCTION_H__

#include "../reax_types.h"


void Hip_Reduction_Sum( int *, int *, size_t, int, hipStream_t );

void Hip_Reduction_Sum( real *, real *, size_t, int, hipStream_t );

void Hip_Reduction_Sum( rvec *, rvec *, size_t, int, hipStream_t );

void Hip_Reduction_Sum( rvec2 *, rvec2 *, size_t, int, hipStream_t );

void Hip_Scan_Excl_Sum( int *, int *, size_t, int, hipStream_t );


#endif
