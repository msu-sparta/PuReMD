#ifndef __CUDA_COPY_H_
#define __CUDA_COPY_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Copy_Atoms_Host_to_Device( reax_system * );

void Cuda_Copy_Grid_Host_to_Device( grid *, grid * );

void Cuda_Copy_System_Host_to_Device( reax_system * );

void Prep_Device_For_Output( reax_system *, simulation_data * );

void Output_Sync_Lists( reax_list *host, reax_list *device, int type );

void Cuda_Copy_Atoms_Device_to_Host( reax_system * );

void Output_Sync_Simulation_Data( simulation_data *, simulation_data * );

#ifdef __cplusplus
}
#endif


#endif
