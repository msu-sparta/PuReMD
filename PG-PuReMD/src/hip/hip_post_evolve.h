
#ifndef __HIP_POST_EVOLVE_H__
#define __HIP_POST_EVOLVE_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

void Hip_Remove_CoM_Velocities( reax_system * const, control_params const * const,
        simulation_data const * const );

#ifdef __cplusplus
}
#endif


#endif
