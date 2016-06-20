

#ifndef __VALIDATION_H__
#define __VALIDATION_H__

#include "reax_types.h"

#ifdef __cplusplus
extern "C" {
#endif

int validate_neighbors (reax_system *, reax_list **lists);
int validate_sym_dbond_indices (reax_system *system, storage *workspace, reax_list **lists);

int validate_bonds (reax_system *, storage *, reax_list **);
int validate_hbonds (reax_system *, storage *, reax_list **);
int validate_sparse_matrix (reax_system *, storage *);

int validate_grid (reax_system *);
int validate_workspace (reax_system *, storage *);

int validate_data (reax_system *, simulation_data *);
int validate_three_bodies (reax_system *, storage *, reax_list **);
int validate_atoms (reax_system *, reax_list **);

int print_sparse_matrix (sparse_matrix *H);
int print_sparse_matrix_host (sparse_matrix *H);

int print_host_rvec2 (rvec2 *, int);
int print_device_rvec2 (rvec2 *, int);

int print_host_array (real *, int);
int print_device_array (real *, int);

void compare_rvec2( rvec2 *host, rvec2 *device, int N, char *msg);
void compare_array (real *host, real *device, int N, char *msg);

int 	check_zeros_host (rvec2 *host, int n, char *);
int 	check_zeros_device (rvec2 *device, int n, char *);



#ifdef __cplusplus
}
#endif

#endif
