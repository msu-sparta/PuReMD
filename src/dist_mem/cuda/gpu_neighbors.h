
#ifndef __GPU_NEIGHBORS_H_
#define __GPU_NEIGHBORS_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

int GPU_Generate_Neighbor_Lists( reax_system * const, control_params const * const,
        simulation_data * const, storage * const, reax_list ** const );

#ifdef __cplusplus
}
#endif

void GPU_Estimate_Num_Neighbors( reax_system * const, control_params const * const,
        simulation_data * const );


#endif
