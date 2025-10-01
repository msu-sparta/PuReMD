
#include "gpu_lookup.h"

#include "gpu_utils.h"

#include "../index_utils.h"


extern "C" void GPU_Copy_LR_Lookup_Table_Host_to_Device( reax_system *system,
        control_params *control, storage *workspace, int *aggregated )
{
    int i, j;
    int num_atom_types;
    LR_data *d_y;
    cubic_spline_coef *temp;

    num_atom_types = system->reax_param.num_atom_types;

    sCudaMalloc( (void **) &workspace->d_workspace->LR,
            sizeof(LR_lookup_table) * SQR(num_atom_types), __FILE__, __LINE__ );

    sCudaMemcpyAsync( workspace->d_workspace->LR, workspace->LR,
            sizeof(LR_lookup_table) * SQR(num_atom_types), 
            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );

    for ( i = 0; i < num_atom_types; ++i ) {
        if ( aggregated[i] > 0 ) {
            for ( j = i; j < num_atom_types; ++j ) {
                if ( aggregated[j] > 0 ) {
                    sCudaMalloc( (void **) &d_y,
                            sizeof(LR_data) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( d_y, workspace->LR[index_lr(i, j, num_atom_types)].y,
                            sizeof(LR_data) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_workspace->LR[index_lr(i, j, num_atom_types)].y, &d_y,
                            sizeof(LR_data *),
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[index_lr(i, j, num_atom_types)].H,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_workspace->LR[index_lr(i, j, num_atom_types)].H, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[index_lr(i, j, num_atom_types)].vdW,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_workspace->LR[index_lr(i, j, num_atom_types)].vdW, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[index_lr(i, j, num_atom_types)].CEvd,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_workspace->LR[index_lr(i, j, num_atom_types)].CEvd, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[index_lr(i, j, num_atom_types)].ele,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_workspace->LR[index_lr(i, j, num_atom_types)].ele, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[index_lr(i, j, num_atom_types)].CEclmb,
                            sizeof(cubic_spline_coef) * workspace->LR[index_lr(i, j, num_atom_types)].n,
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_workspace->LR[index_lr(i, j, num_atom_types)].CEclmb, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->gpu_streams[0], __FILE__, __LINE__ );
                }
            }
        }
    }

    cudaStreamSynchronize( control->gpu_streams[0] );
}
