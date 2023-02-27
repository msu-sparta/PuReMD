#ifndef __CUDA_ALLOCATE_H_
#define __CUDA_ALLOCATE_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void * sCudaHostAllocWrapper( size_t, const char * const, int );

void * sCudaHostReallocWrapper( void *, size_t, size_t, const char * const, int );

void * sCudaHostCallocWrapper( size_t, size_t, const char * const, int );

void sCudaFreeHostWrapper( void *, const char * const, int );

#ifdef __cplusplus
}
#endif

void Cuda_Allocate_System( reax_system *, control_params * );

void Cuda_Allocate_Grid( reax_system *, control_params * );

void Cuda_Allocate_Simulation_Data( simulation_data *, cudaStream_t );

void Cuda_Allocate_Control( control_params * );

void Cuda_Allocate_Workspace_Part1( reax_system *, control_params *, storage *, int );

void Cuda_Allocate_Workspace_Part2( reax_system *, control_params *, storage *, int );

void Cuda_Allocate_Matrix( sparse_matrix * const, int, int, int, int, cudaStream_t );

void Cuda_Deallocate_Grid_Cell_Atoms( reax_system * );

void Cuda_Allocate_Grid_Cell_Atoms( reax_system *, int );

void Cuda_Deallocate_Workspace_Part1( control_params *, storage * );

void Cuda_Deallocate_Workspace_Part2( control_params *, storage * );

void Cuda_Deallocate_Matrix( sparse_matrix * );

void Cuda_Reallocate_Part1( reax_system *, control_params *, simulation_data *, storage *,
        reax_list **, mpi_datatypes * );

void Cuda_Reallocate_Part2( reax_system *, control_params *, simulation_data *, storage *,
        reax_list **, mpi_datatypes * );


#endif
