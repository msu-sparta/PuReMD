#ifndef __CUDA_COPY_H_
#define __CUDA_COPY_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Copy_Atoms_Host_to_Device( reax_system * );

void Cuda_Copy_Grid_Host_to_Device( grid *, grid * );

void Cuda_Copy_System_Host_to_Device( reax_system * );

void Cuda_Copy_List_Device_to_Host( reax_list *, reax_list *, int );

void Cuda_Copy_Atoms_Device_to_Host( reax_system * );

void Cuda_Copy_Simulation_Data_Device_to_Host( simulation_data *, simulation_data * );

void Cuda_Copy_MPI_Data_Host_to_Device( mpi_datatypes * );

#ifdef __cplusplus
}
#endif


#endif
