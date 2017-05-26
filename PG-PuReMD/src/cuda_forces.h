
#ifndef __CUDA_FORCES_H__
#define __CUDA_FORCES_H__

#include "reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif


void Cuda_Estimate_Storages( reax_system *, control_params *, reax_list **, int, int,
        int *, int *, int *, int * );

int Cuda_Estimate_Sparse_Matrix( reax_system *, control_params *,
        simulation_data *, reax_list ** );

int Cuda_Init_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Cuda_Init_Forces_No_Charges( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Cuda_Validate_Lists( reax_system *, storage *, reax_list **, control_params *,
        int, int, int, int );

void Cuda_Compute_Bonded_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );


void Cuda_Compute_NonBonded_Forces( reax_system *, control_params *,
        simulation_data *, storage *, reax_list **, output_controls *,
        mpi_datatypes * );


#ifdef __cplusplus
}
#endif


#endif
