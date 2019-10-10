
#ifndef __CUDA_FORCES_H__
#define __CUDA_FORCES_H__

#include "../reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif


void Cuda_Init_Neighbor_Indices( reax_system *, reax_list ** );

void Cuda_Init_HBond_Indices( reax_system *, reax_list ** );

void Cuda_Init_Bond_Indices( reax_system *, reax_list ** );

void Cuda_Init_Sparse_Matrix_Indices( reax_system *, sparse_matrix * );

void Cuda_Init_Three_Body_Indices( int *, int, reax_list ** );

void Cuda_Estimate_Storages( reax_system *, control_params *, reax_list **,
        int, int, int, int );

int Cuda_Estimate_Storage_Three_Body( reax_system *, control_params *,
        int, reax_list **, int *, int * );

int Cuda_Init_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Cuda_Init_Forces_No_Charges( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Cuda_Compute_Bonded_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

void Cuda_Compute_NonBonded_Forces( reax_system *, control_params *,
        simulation_data *, storage *, reax_list **, output_controls *,
        mpi_datatypes * );

int Cuda_Compute_Forces( reax_system*, control_params*, simulation_data*,
        storage*, reax_list**, output_controls*, mpi_datatypes* );


#ifdef __cplusplus
}
#endif


#endif
