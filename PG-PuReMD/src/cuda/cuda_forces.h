
#ifndef __CUDA_FORCES_H__
#define __CUDA_FORCES_H__

#include "../reax_types.h"


void Cuda_Init_Neighbor_Indices( reax_system * const, control_params const * const,
        reax_list * const );

void Cuda_Init_HBond_Indices( reax_system * const, storage * const, reax_list * const,
        int, cudaStream_t );

void Cuda_Init_Bond_Indices( reax_system * const, reax_list * const, int, cudaStream_t );

void Cuda_Init_Sparse_Matrix_Indices( reax_system * const, sparse_matrix * const, int,
       cudaStream_t );

void Cuda_Init_Three_Body_Indices( int *, int, reax_list ** );

void Cuda_Estimate_Storages( reax_system * const, control_params * const,
        simulation_data * const, storage * const, reax_list ** const,
        int, int, int, int );

int Cuda_Init_Forces( reax_system * const, control_params * const, simulation_data * const,
        storage * const, reax_list ** const, output_controls * const );

int Cuda_Init_Forces_No_Charges( reax_system * const, control_params * const, simulation_data * const,
        storage * const, reax_list ** const, output_controls * const );

int Cuda_Compute_Bonded_Forces( reax_system * const, control_params * const, simulation_data * const,
        storage * const, reax_list ** const, output_controls * const );

#ifdef __cplusplus
extern "C" {
#endif

int Cuda_Compute_Forces( reax_system * const, control_params * const,
        simulation_data * const, storage * const, reax_list ** const,
        output_controls * const, mpi_datatypes * const );

#ifdef __cplusplus
}
#endif


#endif
