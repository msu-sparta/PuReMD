
#ifndef __GPU_POST_EVOLVE_H__
#define __GPU_POST_EVOLVE_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

void GPU_Remove_CoM_Velocities( reax_system * const, control_params const * const,
        simulation_data const * const );

#ifdef __cplusplus
}
#endif


#endif
