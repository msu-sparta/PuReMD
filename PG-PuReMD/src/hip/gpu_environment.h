
#ifndef __GPU_ENVIRONMENT_H__
#define __GPU_ENVIRONMENT_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void GPU_Setup_Environment( reax_system const * const,
        control_params * const );

void GPU_Init_Block_Sizes( reax_system *, control_params * );

void GPU_Cleanup_Environment( control_params const * const );

void GPU_Print_Mem_Usage( simulation_data const * const );

#ifdef __cplusplus
}
#endif


#endif
