
#ifndef __REDUCTION_H__
#define __REDUCTION_H__

#include "cuda_reax_constants.h"
#include "reax_types.h"

#define  INITIAL  0
#define  FINAL    1


CUDA_GLOBAL void k_reduction (const real *, real *, const size_t );
CUDA_GLOBAL void k_reduction_rvec (rvec *, rvec *, size_t );
CUDA_GLOBAL void k_reduction_rvec2 (rvec2 *, rvec2 *, size_t );
CUDA_GLOBAL void k_norm (const real *input, real *per_block_results, const size_t n, int pass);
CUDA_GLOBAL void k_dot (const real *a, const real *b, real *per_block_results, const size_t n);

CUDA_GLOBAL void k_vector_sum( real* , real , real* , real , real* , int ) ;
CUDA_GLOBAL void k_rvec2_pbetad (rvec2 *dest, rvec2 *a, 
                                 real beta0, real beta1, 
											rvec2 *b, int n);
CUDA_GLOBAL void k_rvec2_mul( rvec2* dest, rvec2* v, rvec2* y, int k ) ;
CUDA_GLOBAL void k_vector_mul( real* dest, real* v, real* y, int k ) ;
CUDA_GLOBAL void k_norm_rvec2 (const rvec2 *input, rvec2 *per_block_results, const size_t n, int pass);
CUDA_GLOBAL void k_dot_rvec2 (const rvec2 *a, rvec2 *b, rvec2 *res, const size_t n);




#endif
