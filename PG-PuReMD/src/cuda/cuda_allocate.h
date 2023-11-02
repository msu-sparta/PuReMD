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

void Cuda_Allocate_System( reax_system * const, control_params const * const );

void Cuda_Allocate_Grid( reax_system * const, control_params const * const );

void Cuda_Allocate_Simulation_Data( simulation_data * const, cudaStream_t );

void Cuda_Allocate_Workspace_Part1( control_params const * const, storage * const, int );

void Cuda_Allocate_Workspace_Part2( control_params const * const, storage * const, int );

void Cuda_Allocate_Matrix( sparse_matrix * const, int, int, int, int, cudaStream_t );

void Cuda_Deallocate_Grid_Cell_Atoms( reax_system * const );

void Cuda_Allocate_Grid_Cell_Atoms( reax_system * const, int );

void Cuda_Deallocate_Workspace_Part1( control_params const * const, storage * const );

void Cuda_Deallocate_Workspace_Part2( control_params const * const, storage * const );

void Cuda_Deallocate_Matrix( sparse_matrix * const );

void Cuda_Reallocate_Part1( reax_system * const, control_params const * const,
        simulation_data * const, storage * const,
        reax_list ** const, mpi_datatypes * const );

void Cuda_Reallocate_Part2( reax_system * const, control_params const * const,
        simulation_data * const, storage * const,
        reax_list ** const, mpi_datatypes * const );


#endif
