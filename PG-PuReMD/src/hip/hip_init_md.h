
#ifndef __HIP_INIT_MD_H__
#define __HIP_INIT_MD_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

void Hip_Initialize( reax_system*, control_params*, simulation_data*,
        storage*, reax_list**, output_controls*, mpi_datatypes* );

#ifdef __cplusplus
}
#endif


#endif
