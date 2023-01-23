#ifndef __HIP_COPY_H_
#define __HIP_COPY_H_

#include "../reax_types.h"


#ifdef __cplusplus
extern "C"  {
#endif

void Hip_Copy_Atoms_Host_to_Device( reax_system *, control_params * );

void Hip_Copy_Matrix_Host_to_Device( sparse_matrix const * const,
        sparse_matrix * const, hipStream_t );

void Hip_Copy_Grid_Host_to_Device( control_params *, grid *, grid * );

void Hip_Copy_System_Host_to_Device( reax_system *, control_params * );

void Hip_Copy_List_Device_to_Host( control_params *, reax_list *, reax_list *, int );

void Hip_Copy_Atoms_Device_to_Host( reax_system * const,
        control_params const * const );

void Hip_Copy_Matrix_Device_to_Host( sparse_matrix * const,
        sparse_matrix const * const, hipStream_t );

void Hip_Copy_Simulation_Data_Device_to_Host( control_params const * const,
        simulation_data * const, simulation_data * const );

void Hip_Copy_MPI_Data_Host_to_Device( control_params *, mpi_datatypes * );

#ifdef __cplusplus
}
#endif


#endif
