
#ifndef __CUDA_FORCES_H__
#define __CUDA_FORCES_H__

#include "../reax_types.h"


void Cuda_Init_Neighbor_Indices( reax_system *, control_params *, reax_list * );

void Cuda_Init_HBond_Indices( reax_system *, storage *, reax_list *,
        cudaStream_t );

void Cuda_Init_Bond_Indices( reax_system *, reax_list *, cudaStream_t );

void Cuda_Init_Sparse_Matrix_Indices( reax_system *, sparse_matrix *,
       cudaStream_t );

void Cuda_Init_Three_Body_Indices( int *, int, reax_list ** );

void Cuda_Estimate_Storages( reax_system *, control_params *, storage *,
        reax_list **, int, int, int, int );

int Cuda_Init_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Cuda_Init_Forces_No_Charges( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Cuda_Compute_Bonded_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

void Cuda_Compute_NonBonded_Forces( reax_system *, control_params *,
        simulation_data *, storage *, reax_list **, output_controls *,
        mpi_datatypes * );

#ifdef __cplusplus
extern "C" {
#endif

int Cuda_Compute_Forces( reax_system*, control_params*, simulation_data*,
        storage*, reax_list**, output_controls*, mpi_datatypes* );

#ifdef __cplusplus
}
#endif


#endif
