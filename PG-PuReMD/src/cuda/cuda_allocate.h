#ifndef __CUDA_ALLOCATE_H_
#define __CUDA_ALLOCATE_H_

#include "../reax_types.h"


void Cuda_Allocate_System( reax_system * );

void Cuda_Allocate_Grid( reax_system * );

void Cuda_Allocate_Simulation_Data( simulation_data * );

void Cuda_Allocate_Control( control_params * );

void Cuda_Allocate_Workspace_Part1( reax_system *, control_params *, storage *, int );

void Cuda_Allocate_Workspace_Part2( reax_system *, control_params *, storage *, int );

void Cuda_Allocate_Matrix( sparse_matrix * const, int, int, int, int );

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
