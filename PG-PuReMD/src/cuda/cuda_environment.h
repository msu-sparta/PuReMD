
#ifndef __CUDA_ENVIRONMENT_H__
#define __CUDA_ENVIRONMENT_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Setup_Environment( int, int, int );

void Cuda_Init_Block_Sizes( reax_system *, control_params * );

void Cuda_Cleanup_Environment( );

#ifdef __cplusplus
}
#endif


#endif
