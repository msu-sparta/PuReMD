
#ifndef __CUDA_ENVIRONMENT_H__
#define __CUDA_ENVIRONMENT_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Setup_Environment( reax_system const * const,
        control_params * const );

void Cuda_Init_Block_Sizes( reax_system *, control_params * );

void Cuda_Cleanup_Environment( control_params const * const );

void Cuda_Print_Mem_Usage( simulation_data const * const );

#ifdef __cplusplus
}
#endif


#endif
