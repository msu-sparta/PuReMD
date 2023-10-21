
#ifndef __CUDA_NEIGHBORS_H__
#define __CUDA_NEIGHBORS_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

int Cuda_Generate_Neighbor_Lists( reax_system * const, control_params const * const,
        simulation_data * const, storage * const, reax_list ** const );

#ifdef __cplusplus
}
#endif

void Cuda_Estimate_Num_Neighbors( reax_system * const, control_params const * const,
        simulation_data * const );


#endif
