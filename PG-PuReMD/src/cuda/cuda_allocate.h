#ifndef __CUDA_ALLOCATE_H_
#define __CUDA_ALLOCATE_H_

#include "../reax_types.h"

#ifdef __cplusplus
extern "C"  {
#endif


void Cuda_Allocate_System( reax_system * );

void Cuda_Allocate_Grid( reax_system * );

void Cuda_Allocate_Simulation_Data( simulation_data * );

void Cuda_Allocate_Workspace( reax_system *, control_params *, storage *, int, int );

void Cuda_Allocate_Matrix( sparse_matrix *, int, int );

void Cuda_Allocate_Control( control_params * );

void Cuda_Deallocate_Grid_Cell_Atoms( reax_system * );

void Cuda_Allocate_Grid_Cell_Atoms( reax_system *, int );

void Cuda_Reallocate_System( reax_system *, int , int , char * );

void Cuda_Deallocate_Workspace( control_params *, storage * );

void Cuda_Deallocate_Matrix( sparse_matrix * );

void Cuda_ReAllocate( reax_system*, control_params*, simulation_data*, storage*,
        reax_list**, mpi_datatypes* );


#ifdef __cplusplus
}
#endif

#endif
