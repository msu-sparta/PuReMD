#ifndef __DEV_ALLOC_H_
#define __DEV_ALLOC_H_

#include "reax_types.h"

#ifdef __cplusplus
extern "C"  {
#endif


int dev_alloc_system (reax_system *);
int dev_alloc_grid (reax_system *);
int dev_alloc_simulation_data (simulation_data *);
int dev_alloc_workspace (reax_system *, control_params *, storage *, int, int, char *);
int dev_alloc_matrix (sparse_matrix *, int, int);
int dev_alloc_control (control_params *);

int dev_dealloc_grid_cell_atoms (reax_system *);
int dev_alloc_grid_cell_atoms (reax_system *, int );
int dev_realloc_system (reax_system *, int , int , char *);
int dev_dealloc_workspace( control_params *, storage * );
int dev_dealloc_matrix (sparse_matrix *);

#ifdef __cplusplus
}
#endif

#endif
