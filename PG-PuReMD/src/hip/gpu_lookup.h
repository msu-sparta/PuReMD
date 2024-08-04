
#ifndef __GPU_LOOKUP_H__
#define __GPU_LOOKUP_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif

void GPU_Copy_LR_Lookup_Table_Host_to_Device( reax_system *, control_params *,
        storage *, int * );

#ifdef __cplusplus
}
#endif


#endif
