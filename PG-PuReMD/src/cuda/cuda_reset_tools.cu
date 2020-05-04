
#include "cuda_reset_tools.h"

#include "cuda_list.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"

#include "../reset_tools.h"


extern "C"
{

void Cuda_Reset_Workspace( reax_system *system, storage *workspace )
{
    cuda_memset( workspace->d_workspace->total_bond_order, 0,
            system->total_cap * sizeof(real), "total_bond_order" );
    cuda_memset( workspace->d_workspace->dDeltap_self, 0,
            system->total_cap * sizeof(rvec), "dDeltap_self" );
    cuda_memset( workspace->d_workspace->CdDelta, 0,
            system->total_cap * sizeof(real), "CdDelta" );
    cuda_memset( workspace->d_workspace->f, 0,
            system->total_cap * sizeof(rvec), "f" );
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


void Cuda_Reset_Atoms( reax_system* system, control_params *control,
        storage *workspace )
{
    int blocks, *hindex;

    blocks = system->N / DEF_BLOCK_SIZE
        + ((system->N % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

    cuda_check_malloc( &workspace->scratch, &workspace->scratch_size,
            sizeof(int) * system->N,
            "Cuda_Reset_Atoms::workspace->scratch" );
    hindex = (int *) workspace->scratch;

    k_reset_hindex <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, hindex, system->N );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    Cuda_Reduction_Sum( hindex, system->d_numH, system->N );

    copy_host_device( &system->numH, system->d_numH, sizeof(int), 
            cudaMemcpyDeviceToHost, "Cuda_Reset_Atoms::d_numH" );
}


void Cuda_Reset( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists )
{
    Cuda_Reset_Atoms( system, control, workspace );

    Reset_Simulation_Data( data );

    if ( control->virial )
    {
        Reset_Pressures( data );
    }

    Cuda_Reset_Workspace( system, workspace );
}


}
