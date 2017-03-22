
#ifndef __CUDA_NEIGHBORS_H__
#define __CUDA_NEIGHBORS_H__

#include "reax_types.h"


#ifdef __cplusplus
extern "C" {
#endif


void Cuda_Generate_Neighbor_Lists( reax_system *, simulation_data *, storage *, reax_list ** );
void Cuda_Estimate_Neighbors( reax_system *, int *);
void Cuda_Init_Neighbors_Indices (int *, int);

void Cuda_Init_HBond_Indices (int *, int);
void Cuda_Init_Bond_Indices (int *, int, int);

//void validate_neighbors (reax_system *system);


#ifdef __cplusplus
}
#endif

#endif
