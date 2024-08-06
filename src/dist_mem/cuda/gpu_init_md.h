
#ifndef __GPU_INIT_MD_H_
#define __GPU_INIT_MD_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

void GPU_Initialize( reax_system * const, control_params * const, simulation_data * const,
        storage * const, reax_list ** const, output_controls * const, mpi_datatypes * const );

#ifdef __cplusplus
}
#endif


#endif
