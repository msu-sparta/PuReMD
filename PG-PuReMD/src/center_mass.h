
#ifndef __CENTER_MASS_H__
#define __CENTER_MASS_H__

#include "reax_types.h"
#include "reax_types.h"

CUDA_GLOBAL void center_of_mass_blocks (single_body_parameters *, reax_atom *,
                            rvec *res_xcm, 
                            rvec *res_vcm, 
                            rvec *res_amcm, 
                            size_t n);

#if defined(__SM_35__)
CUDA_GLOBAL void center_of_mass_blocks_xcm (single_body_parameters *, reax_atom *,
                            rvec *res_xcm,
                            size_t n);
CUDA_GLOBAL void center_of_mass_blocks_vcm (single_body_parameters *, reax_atom *,
                            rvec *res_vcm,
                            size_t n);
CUDA_GLOBAL void center_of_mass_blocks_amcm (single_body_parameters *, reax_atom *,
                            rvec *res_amcm,
                            size_t n);
#endif


CUDA_GLOBAL void center_of_mass (rvec *xcm, 
                            rvec *vcm, 
                            rvec *amcm, 
                            rvec *res_xcm,
                            rvec *res_vcm,
                            rvec *res_amcm,
                            size_t n);

CUDA_GLOBAL void compute_center_mass (single_body_parameters *sbp, 
                                reax_atom *atoms,
                                real *results, 
								real xcm0, real xcm1, real xcm2,
                                size_t n);

CUDA_GLOBAL void compute_center_mass (real *input, real *output, size_t n);

#if defined(__SM_35__)
CUDA_GLOBAL void compute_center_mass_xx_xy (single_body_parameters *, reax_atom *, real *, real , real , real , size_t );
CUDA_GLOBAL void compute_center_mass_xz_yy (single_body_parameters *, reax_atom *, real *, real , real , real , size_t );
CUDA_GLOBAL void compute_center_mass_yz_zz (single_body_parameters *, reax_atom *, real *, real , real , real , size_t );
#endif

#endif
