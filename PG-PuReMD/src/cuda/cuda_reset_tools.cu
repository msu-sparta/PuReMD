
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

    workspace.total_bond_order[i] = 0.0;
    workspace.CdDelta[i] = 0.0;
    rvec_MakeZero( workspace.dDeltap_self[i] );
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

    if ( sbp[ my_atoms[i].type ].p_hbond == H_ATOM
            || sbp[ my_atoms[i].type ].p_hbond == H_BONDING_ATOM )
    {
        hindex[i] = 1;
    }
    else
    {
        hindex[i] = 0;
    }

//    my_atoms[i].Hindex = hindex[i];
    my_atoms[i].Hindex = i;
}

void Cuda_Reset_Workspace( reax_system *system, storage *workspace )
{
    int blocks;

    blocks = system->total_cap / DEF_BLOCK_SIZE
        + ((system->total_cap % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    k_reset_workspace <<< blocks, DEF_BLOCK_SIZE >>>
        ( *(workspace->d_workspace), system->total_cap );
    cudaDeviceSynchronize( );
    cudaCheckError( );
}


void Cuda_Reset_Atoms_HBond_Indices( reax_system* system, control_params *control,
        storage *workspace )
{
    int *hindex;

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(int) * system->N,
            "Cuda_Reset_Atoms_HBond_Indices::workspace->scratch" );
    hindex = (int *) workspace->scratch;

    k_reset_hindex <<< control->blocks_n, control->block_size_n >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, hindex, system->N );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    Cuda_Reduction_Sum( hindex, system->d_numH, system->N );

    copy_host_device( &system->numH, system->d_numH, sizeof(int), 
            cudaMemcpyDeviceToHost, "Cuda_Reset_Atoms_HBond_Indices::d_numH" );
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

    Cuda_Reset_Workspace( system, workspace );
}
