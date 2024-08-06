#ifndef __GPU_COPY_H_
#define __GPU_COPY_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void GPU_Copy_Atoms_Host_to_Device( reax_system * const, control_params const * const );

void GPU_Copy_Matrix_Host_to_Device( sparse_matrix const * const,
        sparse_matrix * const, hipStream_t );

void GPU_Copy_Grid_Host_to_Device( control_params const * const, grid const * const,
        grid * const );

void GPU_Copy_System_Host_to_Device( reax_system * const, control_params const * const );

void GPU_Copy_List_Device_to_Host( control_params const * const,
        reax_list * const, reax_list * const, int );

void GPU_Copy_Atoms_Device_to_Host( reax_system * const,
        control_params const * const );

void GPU_Copy_Matrix_Device_to_Host( sparse_matrix * const,
        sparse_matrix const * const, hipStream_t );

void GPU_Copy_Simulation_Data_Device_to_Host( control_params const * const,
        simulation_data * const );

void GPU_Copy_MPI_Data_Host_to_Device( control_params const * const, mpi_datatypes * const );

#ifdef __cplusplus
}
#endif


#endif
