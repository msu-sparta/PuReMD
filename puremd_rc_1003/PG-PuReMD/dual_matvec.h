

#ifndef __DUAL_MATVEC__H_
#define __DUAL_MATVEC__H_

#include "reax_types.h"
#include "cuda_reax_constants.h"

CUDA_GLOBAL void k_dual_matvec (sparse_matrix , rvec2 *, rvec2 *, int );
CUDA_GLOBAL void k_dual_matvec_csr(sparse_matrix , rvec2 *, rvec2 *, int );

#endif
