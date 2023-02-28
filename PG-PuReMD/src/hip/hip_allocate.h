#ifndef __HIP_ALLOCATE_H_
#define __HIP_ALLOCATE_H_

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

void Hip_Allocate_System( reax_system *, control_params * );

void Hip_Allocate_Grid( reax_system *, control_params * );

void Hip_Allocate_Simulation_Data( simulation_data *, hipStream_t );

void Hip_Allocate_Control( control_params * );

void Hip_Allocate_Workspace_Part1( reax_system *, control_params *, storage *, int );

void Hip_Allocate_Workspace_Part2( reax_system *, control_params *, storage *, int );

void Hip_Allocate_Matrix( sparse_matrix * const, int, int, int, int, hipStream_t );

void Hip_Deallocate_Grid_Cell_Atoms( reax_system * );

void Hip_Allocate_Grid_Cell_Atoms( reax_system *, int );

void Hip_Deallocate_Workspace_Part1( control_params *, storage * );

void Hip_Deallocate_Workspace_Part2( control_params *, storage * );

void Hip_Deallocate_Matrix( sparse_matrix * );

void Hip_Reallocate_Part1( reax_system *, control_params *, simulation_data *, storage *,
        reax_list **, mpi_datatypes * );

void Hip_Reallocate_Part2( reax_system *, control_params *, simulation_data *, storage *,
        reax_list **, mpi_datatypes * );


#endif
