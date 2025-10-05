
#ifndef __GPU_RESET_TOOLS_H_
#define __GPU_RESET_TOOLS_H_

#include "../reax_types.h"


void GPU_Reset_Workspace( reax_system const * const, control_params const * const,
        storage * const, cudaStream_t );

void GPU_Reset_Atoms_HBond_Indices( reax_system * const, control_params const * const,
        storage * const );

#ifdef __cplusplus
extern "C"  {
#endif

void GPU_Reset( reax_system * const, control_params const * const, simulation_data * const,
        storage * const, reax_list ** const, int );

#ifdef __cplusplus
}
#endif


#endif
