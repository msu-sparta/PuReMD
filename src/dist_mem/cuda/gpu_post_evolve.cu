
#include "gpu_post_evolve.h"

#include "gpu_utils.h"

#include "../vector.h"


/* remove translation and rotational terms from center of mass velocities */
GPU_GLOBAL void k_remove_center_of_mass_velocities( reax_atom * const my_atoms, 
        const rvec xcm, const rvec vcm, const rvec avcm, int n )
{
    int i;
    rvec diff, cross;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    /* remove translational term */
    rvec_ScaledAdd( my_atoms[i].v, -1.0, vcm );

    /* remove rotational term */
    rvec_ScaledSum( diff, 1.0, my_atoms[i].x, -1.0, xcm );
    rvec_Cross( cross, avcm, diff );
    rvec_ScaledAdd( my_atoms[i].v, -1.0, cross );
}


extern "C" void GPU_Remove_CoM_Velocities( reax_system * const system,
        control_params const * const control, simulation_data const * const data )
{
    k_remove_center_of_mass_velocities <<< control->blocks_n, control->gpu_block_size,
                                       0, control->gpu_streams[0] >>>
        ( system->d_my_atoms, data->xcm, data->vcm, data->avcm, system->n );
    cudaCheckError( );
}
