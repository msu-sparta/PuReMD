
#include "cuda_reset_tools.h"

#include "cuda_list.h"
#include "cuda_utils.h"
#if !defined(GPU_ACCUM_ATOMIC)
  #include "cuda_reduction.h"
#endif

#include "../reset_tools.h"
#include "../vector.h"


GPU_GLOBAL void k_reset_workspace( real * const CdDelta,
#if !defined(GPU_STREAM_SINGLE_ACCUM)
        real * const CdDelta_bonds, real * const CdDelta_multi,
        real * const CdDelta_tor, real * const CdDelta_val,
#endif
        rvec * const f,
#if !defined(GPU_STREAM_SINGLE_ACCUM)
        rvec * const f_hb,
#if defined(FUSED_VDW_COULOMB)
        rvec * const f_vdw_clmb,
#else
        rvec * const f_vdw, rvec * const f_clmb,
#endif
        rvec * const f_tor, rvec * const f_val,
#endif
        int N )
{
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    CdDelta[i] = 0.0;
#if !defined(GPU_STREAM_SINGLE_ACCUM)
    CdDelta_bonds[i] = 0.0;
    CdDelta_multi[i] = 0.0;
    CdDelta_tor[i] = 0.0;
    CdDelta_val[i] = 0.0;
#endif
    rvec_MakeZero( f[i] );
#if !defined(GPU_STREAM_SINGLE_ACCUM)
    rvec_MakeZero( f_hb[i] );
#if defined(FUSED_VDW_COULOMB)
    rvec_MakeZero( f_vdw_clmb[i] );
#else
    rvec_MakeZero( f_vdw[i] );
    rvec_MakeZero( f_clmb[i] );
#endif
    rvec_MakeZero( f_tor[i] );
    rvec_MakeZero( f_val[i] );
#endif
}


GPU_GLOBAL void k_reset_hindex( reax_atom * const my_atoms,
        single_body_parameters const * const sbp, int * const hindex, int N )
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

void Cuda_Reset_Workspace( reax_system const * const system,
        control_params const * const control, storage * const workspace )
{
    int blocks;

    blocks = system->total_cap / control->gpu_block_size
        + ((system->total_cap % control->gpu_block_size == 0 ) ? 0 : 1);

    k_reset_workspace <<< blocks, control->gpu_block_size, 0, control->cuda_streams[0] >>>
        ( workspace->d_workspace->CdDelta,
#if !defined(GPU_STREAM_SINGLE_ACCUM)
          workspace->d_workspace->CdDelta_bonds, workspace->d_workspace->CdDelta_multi,
          workspace->d_workspace->CdDelta_tor, workspace->d_workspace->CdDelta_val,
#endif
          workspace->d_workspace->f,
#if !defined(GPU_STREAM_SINGLE_ACCUM)
          workspace->d_workspace->f_hb,
#if defined(FUSED_VDW_COULOMB)
          workspace->d_workspace->f_vdw_clmb,
#else
          workspace->d_workspace->f_vdw, workspace->d_workspace->f_clmb,
#endif
          workspace->d_workspace->f_tor, workspace->d_workspace->f_val,
#endif
          system->total_cap );
    cudaCheckError( );
}


void Cuda_Reset_Atoms_HBond_Indices( reax_system * const system, control_params const * const control,
        storage * const workspace )
{
#if defined(GPU_ACCUM_ATOMIC)
    sCudaMemsetAsync( system->d_num_H_atoms, 0, sizeof(int), 
            control->cuda_streams[0], __FILE__, __LINE__ );
#else
    int *hindex;

    sCudaCheckMalloc( &workspace->d_workspace->scratch[0],
            &workspace->d_workspace->scratch_size[0],
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    hindex = (int *) workspace->d_workspace->scratch[0];
#endif

    k_reset_hindex <<< control->blocks_N, control->gpu_block_size,
                   0, control->cuda_streams[0] >>>
        ( system->d_my_atoms, system->reax_param.d_sbp, 
#if defined(GPU_ACCUM_ATOMIC)
          system->d_num_H_atoms,
#else
          hindex, 
#endif
          system->total_cap );
    cudaCheckError( );

#if !defined(GPU_ACCUM_ATOMIC)
    Cuda_Reduction_Sum( hindex, system->d_num_H_atoms, system->N, 0, control->cuda_streams[0] );
#endif

    sCudaMemcpyAsync( &system->num_H_atoms, system->d_num_H_atoms, sizeof(int), 
            cudaMemcpyDeviceToHost, control->cuda_streams[0], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->cuda_streams[0] );
}


extern "C" void Cuda_Reset( reax_system * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists )
{
    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
    {
        Cuda_Reset_Atoms_HBond_Indices( system, control, workspace );
    }

    Reset_Simulation_Data( data );

    if ( control->virial )
    {
        Reset_Pressures( data );
    }

    Cuda_Reset_Workspace( system, control, workspace );
}
