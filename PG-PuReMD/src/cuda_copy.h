#ifndef __CUDA_COPY_H_
#define __CUDA_COPY_H_

#include "reax_types.h"

#ifdef __cplusplus
extern "C"  {
#endif

void Sync_Atoms (reax_system *);
void Sync_Grid (grid *, grid *);
void Sync_System (reax_system *);
void Sync_Control (control_params *, control_params *, enum cudaMemcpyKind);
void Sync_Matrix (sparse_matrix *, sparse_matrix *, enum cudaMemcpyKind);
void Sync_Output_Control (output_controls *, enum cudaMemcpyKind);
void Sync_Workspace (storage *workspace, enum cudaMemcpyKind);

void Prep_Device_For_Output (reax_system *, simulation_data *);
void Output_Sync_Lists (reax_list *host, reax_list *device, int type );
void Output_Sync_Atoms (reax_system *);
void Output_Sync_Simulation_Data (simulation_data *, simulation_data *);


#ifdef __cplusplus
}
#endif

#endif
