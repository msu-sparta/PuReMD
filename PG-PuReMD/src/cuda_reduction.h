
#ifndef __CUDA_REDUCTION_H__
#define __CUDA_REDUCTION_H__

#include "reax_types.h"

#define  INITIAL  0
#define  FINAL    1


CUDA_GLOBAL void k_reduction( const real *, real *, const size_t );
CUDA_GLOBAL void k_reduction_rvec( rvec *, rvec *, size_t );
CUDA_GLOBAL void k_reduction_rvec2( rvec2 *, rvec2 *, size_t );
CUDA_GLOBAL void k_norm( const real *, real *, const size_t, int );
CUDA_GLOBAL void k_dot( const real *, const real *, real *,
        const size_t );

CUDA_GLOBAL void k_vector_sum( real*, real, real*, real,
        real*, int );
CUDA_GLOBAL void k_rvec2_pbetad( rvec2 *, rvec2 *, real, real,
        rvec2 *, int );
CUDA_GLOBAL void k_rvec2_mul( rvec2*, rvec2*, rvec2*, int );
CUDA_GLOBAL void k_vector_mul( real*, real*, real*, int );
CUDA_GLOBAL void k_norm_rvec2( const rvec2 *, rvec2 *, const size_t, int );
CUDA_GLOBAL void k_dot_rvec2( const rvec2 *, rvec2 *, rvec2 *, const size_t );


#endif
