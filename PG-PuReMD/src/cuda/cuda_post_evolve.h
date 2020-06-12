
#ifndef __CUDA_POST_EVOLVE_H__
#define __CUDA_POST_EVOLVE_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

void Cuda_Remove_CoM_Velocities( reax_system *, control_params *,
        simulation_data * );

#ifdef __cplusplus
}
#endif


#endif
