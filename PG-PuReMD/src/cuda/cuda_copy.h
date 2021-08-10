#ifndef __CUDA_COPY_H_
#define __CUDA_COPY_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Copy_Atoms_Host_to_Device( reax_system *, control_params * );

void Cuda_Copy_Grid_Host_to_Device( control_params *, grid *, grid * );

void Cuda_Copy_System_Host_to_Device( reax_system *, control_params * );

void Cuda_Copy_List_Device_to_Host( control_params *, reax_list *, reax_list *, int );

void Cuda_Copy_Atoms_Device_to_Host( reax_system *, control_params * );

void Cuda_Copy_Simulation_Data_Device_to_Host( control_params const * const,
        simulation_data * const, simulation_data * const );

void Cuda_Copy_MPI_Data_Host_to_Device( control_params *, mpi_datatypes * );

#ifdef __cplusplus
}
#endif


#endif
