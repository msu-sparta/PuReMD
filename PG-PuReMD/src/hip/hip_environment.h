
#ifndef __HIP_ENVIRONMENT_H__
#define __HIP_ENVIRONMENT_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Hip_Setup_Environment( reax_system const * const,
        control_params * const );

void Hip_Init_Block_Sizes( reax_system *, control_params * );

void Hip_Cleanup_Environment( control_params const * const );

void Hip_Print_Mem_Usage( simulation_data const * const );

#ifdef __cplusplus
}
#endif


#endif
