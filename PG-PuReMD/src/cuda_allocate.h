#ifndef __CUDA_ALLOCATE_H_
#define __CUDA_ALLOCATE_H_

#include "reax_types.h"

#ifdef __cplusplus
extern "C"  {
#endif


void dev_alloc_system( reax_system * );

void dev_alloc_grid( reax_system * );

void dev_alloc_simulation_data( simulation_data * );

void dev_alloc_workspace( reax_system *, control_params *, storage *, int, int, char * );

void dev_alloc_matrix( sparse_matrix *, int, int );

void dev_alloc_control( control_params * );

void dev_dealloc_grid_cell_atoms( reax_system * );

void dev_alloc_grid_cell_atoms( reax_system *, int );

void dev_realloc_system( reax_system *, int , int , char * );

void dev_dealloc_workspace( control_params *, storage * );

void dev_dealloc_matrix( sparse_matrix * );

void Cuda_ReAllocate( reax_system*, control_params*, simulation_data*, storage*,
        reax_list**, mpi_datatypes* );


#ifdef __cplusplus
}
#endif

#endif
