
#ifndef __GPU_SYSTEM_PROPS_H__
#define __GPU_SYSTEM_PROPS_H__

#include "../reax_types.h"


void GPU_Generate_Initial_Velocities( reax_system * const,
        control_params const * const, real );

void GPU_Compute_Total_Mass( reax_system * const, control_params const * const,
        storage * const, simulation_data * const, MPI_Comm );

void GPU_Compute_Pressure( reax_system * const, control_params const * const,
        storage * const, simulation_data * const, mpi_datatypes * const );

#ifdef __cplusplus
extern "C"  {
#endif

void GPU_Compute_Kinetic_Energy( reax_system * const, control_params const * const,
        storage * const, simulation_data * const, MPI_Comm );

void GPU_Compute_Center_of_Mass( reax_system * const, control_params const * const,
        storage * const, simulation_data * const, mpi_datatypes * const, MPI_Comm );

#ifdef __cplusplus
}
#endif


#endif
