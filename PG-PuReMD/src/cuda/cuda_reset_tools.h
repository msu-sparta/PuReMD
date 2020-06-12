
#ifndef __CUDA_RESET_TOOLS_H__
#define __CUDA_RESET_TOOLS_H__

#include "../reax_types.h"


void Cuda_Reset_Workspace( reax_system *, storage * );

void Cuda_Reset_Atoms_HBond_Indices( reax_system *, control_params *, storage * );

int Cuda_Reset_Neighbor_Lists( reax_system *, control_params *,
        storage *, reax_list ** );

#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Reset( reax_system*, control_params*, simulation_data*,
        storage*, reax_list** );

#ifdef __cplusplus
}
#endif


#endif
