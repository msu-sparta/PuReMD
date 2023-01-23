
#ifndef __HIP_FORCES_H__
#define __HIP_FORCES_H__

#include "../reax_types.h"


void Hip_Init_Neighbor_Indices( reax_system *, control_params *, reax_list * );

void Hip_Init_HBond_Indices( reax_system *, storage *, reax_list *,
        hipStream_t );

void Hip_Init_Bond_Indices( reax_system *, reax_list *, hipStream_t );

void Hip_Init_Sparse_Matrix_Indices( reax_system *, sparse_matrix *,
       hipStream_t );

void Hip_Init_Three_Body_Indices( int *, int, reax_list ** );

void Hip_Estimate_Storages( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, int, int, int, int );

int Hip_Init_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Hip_Init_Forces_No_Charges( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

int Hip_Compute_Bonded_Forces( reax_system *, control_params *, simulation_data *,
        storage *, reax_list **, output_controls * );

void Hip_Compute_NonBonded_Forces( reax_system *, control_params *,
        simulation_data *, storage *, reax_list **, output_controls *,
        mpi_datatypes * );

#ifdef __cplusplus
extern "C" {
#endif

int Hip_Compute_Forces( reax_system*, control_params*, simulation_data*,
        storage*, reax_list**, output_controls*, mpi_datatypes* );

#ifdef __cplusplus
}
#endif


#endif
