#ifndef __GPU_ALLOCATE_H_
#define __GPU_ALLOCATE_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void * sHipHostMallocWrapper( size_t, const char * const, int );

void * sHipHostReallocWrapper( void *, size_t, size_t, const char * const, int );

void * sHipHostCallocWrapper( size_t, size_t, const char * const, int );

void sHipHostFreeWrapper( void *, const char * const, int );

#ifdef __cplusplus
}
#endif

void GPU_Allocate_System( reax_system * const, control_params const * const );

void GPU_Allocate_Grid( reax_system * const, control_params const * const );

void GPU_Allocate_Simulation_Data( simulation_data * const, hipStream_t );

void GPU_Allocate_Workspace_Part1( control_params const * const, storage * const, int );

void GPU_Allocate_Workspace_Part2( control_params const * const, storage * const, int );

void GPU_Allocate_Matrix( sparse_matrix * const, int, int, int, int, hipStream_t );

void GPU_Deallocate_Grid_Cell_Atoms( reax_system * const );

void GPU_Allocate_Grid_Cell_Atoms( reax_system * const, int );

void GPU_Deallocate_Workspace_Part1( control_params const * const, storage * const );

void GPU_Deallocate_Workspace_Part2( control_params const * const, storage * const );

void GPU_Deallocate_Matrix( sparse_matrix * const );

void GPU_Reallocate_Part1( reax_system * const, control_params const * const,
        simulation_data * const, storage * const,
        reax_list ** const, mpi_datatypes * const );

void GPU_Reallocate_Part2( reax_system * const, control_params const * const,
        simulation_data * const, storage * const,
        reax_list ** const, mpi_datatypes * const );


#endif
