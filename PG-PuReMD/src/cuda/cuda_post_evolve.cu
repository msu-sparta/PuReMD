
#include "cuda_post_evolve.h"

#include "cuda_utils.h"

#include "../vector.h"


/* remove translation and rotational terms from center of mass velocities */
GPU_GLOBAL void k_remove_center_of_mass_velocities( reax_atom * const my_atoms, 
        simulation_data const * const data, int n )
{
    int i;
    rvec diff, cross;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* remove translational term */
    rvec_ScaledAdd( my_atoms[i].v, -1.0, data->vcm );

    /* remove rotational term */
    rvec_ScaledSum( diff, 1.0, my_atoms[i].x, -1.0, data->xcm );
    rvec_Cross( cross, data->avcm, diff );
    rvec_ScaledAdd( my_atoms[i].v, -1.0, cross );
}


extern "C" void Cuda_Remove_CoM_Velocities( reax_system * const system,
        control_params const * const control, simulation_data const * const data )
{
    k_remove_center_of_mass_velocities <<< control->blocks_n, control->gpu_block_size,
                                       0, control->cuda_streams[0] >>>
        ( system->d_my_atoms, data->d_simulation_data, system->n );
    cudaCheckError( );
}
