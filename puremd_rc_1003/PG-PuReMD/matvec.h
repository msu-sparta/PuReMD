

#ifndef __MATVEC__H_
#define __MATVEC__H_

#include "reax_types.h"
#include "cuda_reax_constants.h"

CUDA_GLOBAL void k_matvec (sparse_matrix , real *, real *, int );
CUDA_GLOBAL void k_matvec_csr(sparse_matrix , real *, real *, int );

#endif
