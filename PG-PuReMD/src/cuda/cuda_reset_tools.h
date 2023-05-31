
#ifndef __CUDA_RESET_TOOLS_H__
#define __CUDA_RESET_TOOLS_H__

#include "../reax_types.h"


void Cuda_Reset_Workspace( reax_system * const, control_params const * const,
        storage * const );

void Cuda_Reset_Atoms_HBond_Indices( reax_system * const, control_params const * const,
        storage * const );

#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Reset( reax_system * const, control_params const * const, simulation_data * const,
        storage * const, reax_list ** const );

#ifdef __cplusplus
}
#endif


#endif
