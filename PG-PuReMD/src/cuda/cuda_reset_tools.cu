
#include "cuda_reset_tools.h"

#include "cuda_list.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"

#include "../reset_tools.h"
#include "../vector.h"


CUDA_GLOBAL void k_reset_workspace( storage workspace, int N )
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


CUDA_GLOBAL void k_reset_hindex( reax_atom *my_atoms, single_body_parameters *sbp,
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
#if !defined(CUDA_ACCUM_ATOMIC)
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

void Cuda_Reset_Workspace( reax_system *system, control_params *control,
        storage *workspace )
{
    int blocks;

    blocks = system->total_cap / DEF_BLOCK_SIZE
        + ((system->total_cap % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    k_reset_workspace <<< blocks, DEF_BLOCK_SIZE, 0, control->streams[0] >>>
        ( *(workspace->d_workspace), system->total_cap );
    cudaCheckError( );
}


void Cuda_Reset_Atoms_HBond_Indices( reax_system* system, control_params *control,
        storage *workspace )
{
#if !defined(CUDA_ACCUM_ATOMIC)
    int *hindex;

    sCudaCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    hindex = (int *) workspace->scratch[0];
#endif

    k_reset_hindex <<< control->blocks_n, control->block_size_n, 0,
                   control->streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
#if !defined(CUDA_ACCUM_ATOMIC)
          hindex, 
#else
          system->d_numH,
#endif
          system->total_cap );
    cudaCheckError( );

#if !defined(CUDA_ACCUM_ATOMIC)
    Cuda_Reduction_Sum( hindex, system->d_numH, system->N, 0, control->streams[0] );
#endif

    sCudaMemcpyAsync( &system->numH, system->d_numH, sizeof(int), 
            cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[0] );
}


extern "C" void Cuda_Reset( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists )
{
    Cuda_Reset_Atoms_HBond_Indices( system, control, workspace );

    Reset_Simulation_Data( data );

    if ( control->virial )
    {
        Reset_Pressures( data );
    }

    Cuda_Reset_Workspace( system, control, workspace );
}
