
#ifndef __CUDA_SYSTEM_PROPS_H__
#define __CUDA_SYSTEM_PROPS_H__

#include "../reax_types.h"


void Cuda_Generate_Initial_Velocities( reax_system *, control_params *, real );

void Cuda_Compute_Total_Mass( reax_system *, control_params *,
        storage *, simulation_data *, MPI_Comm );

void Cuda_Compute_Pressure( reax_system *, control_params *,
        storage *, simulation_data *, mpi_datatypes * );

#ifdef __cplusplus
extern "C"  {
#endif

void Cuda_Compute_Kinetic_Energy( reax_system *, control_params *,
        storage *, simulation_data *, MPI_Comm );

void Cuda_Compute_Center_of_Mass( reax_system *, control_params *,
        storage *, simulation_data *, mpi_datatypes *, MPI_Comm );

#ifdef __cplusplus
}
#endif


#endif
