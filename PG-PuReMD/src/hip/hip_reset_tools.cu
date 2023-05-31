
#include "hip_reset_tools.h"

#include "hip_list.h"
#include "hip_utils.h"
#include "hip_reduction.h"

#include "../reset_tools.h"
#include "../vector.h"


GPU_GLOBAL void k_reset_workspace( storage workspace, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    workspace.CdDelta[i] = 0.0;
    rvec_MakeZero( workspace.f[i] );
}


GPU_GLOBAL void k_reset_hindex( reax_atom *my_atoms, single_body_parameters *sbp,
        int * hindex, int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    my_atoms[i].Hindex = i;

    if ( sbp[ my_atoms[i].type ].p_hbond == H_ATOM
            || sbp[ my_atoms[i].type ].p_hbond == H_BONDING_ATOM )
    {
#if !defined(GPU_ACCUM_ATOMIC)
        hindex[i] = 1;
    }
    else
    {
        hindex[i] = 0;
    }
#else
        atomicAdd( hindex, 1 );
    }
#endif
}

void Hip_Reset_Workspace( reax_system * const system, control_params const * const control,
        storage * const workspace )
{
    int blocks;

    blocks = system->total_cap / control->gpu_block_size
        + ((system->total_cap % control->gpu_block_size == 0 ) ? 0 : 1);

    k_reset_workspace <<< blocks, control->gpu_block_size, 0, control->hip_streams[0] >>>
        ( *(workspace->d_workspace), system->total_cap );
    hipCheckError( );
}


void Hip_Reset_Atoms_HBond_Indices( reax_system * const system, control_params const * const control,
        storage * const workspace )
{
#if !defined(GPU_ACCUM_ATOMIC)
    int *hindex;

    sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    hindex = (int *) workspace->scratch[0];
#else
    sHipMemsetAsync( system->d_num_H_atoms, 0, sizeof(int), 
            control->hip_streams[0], __FILE__, __LINE__ );
#endif

    k_reset_hindex <<< control->blocks_N, control->gpu_block_size,
                   0, control->hip_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
#if !defined(GPU_ACCUM_ATOMIC)
          hindex, 
#else
          system->d_num_H_atoms,
#endif
          system->total_cap );
    hipCheckError( );

#if !defined(GPU_ACCUM_ATOMIC)
    Hip_Reduction_Sum( hindex, system->d_num_H_atoms, system->N, 0, control->hip_streams[0] );
#endif

    sHipMemcpyAsync( &system->num_H_atoms, system->d_num_H_atoms, sizeof(int), 
            hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );

    hipStreamSynchronize( control->hip_streams[0] );
}


extern "C" void Hip_Reset( reax_system * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists )
{
    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
    {
        Hip_Reset_Atoms_HBond_Indices( system, control, workspace );
    }

    Reset_Simulation_Data( data );

    if ( control->virial )
    {
        Reset_Pressures( data );
    }

    Hip_Reset_Workspace( system, control, workspace );
}
