
#include "cuda_lookup.h"

#include "cuda_utils.h"

#include "../index_utils.h"


extern "C" void Cuda_Copy_LR_Lookup_Table_Host_to_Device( reax_system *system,
        control_params *control, storage *workspace, int *aggregated )
{
    int i, j;
    int num_atom_types;
    LR_data *d_y;
    cubic_spline_coef *temp;

    num_atom_types = system->reax_param.num_atom_types;

    fprintf( stderr, "Copying the LR Lookyp Table to the device ... \n" );

    sCudaMalloc( (void **) &workspace->d_LR,
            sizeof(LR_lookup_table) * num_atom_types * num_atom_types,
            __FILE__, __LINE__ );

    /*
       for( i = 0; i < MAX_ATOM_TYPES; ++i )
       existing_types[i] = 0;

       for( i = 0; i < system->N; ++i )
       existing_types[ system->atoms[i].type ] = 1;
     */

    sCudaMemcpyAsync( workspace->d_LR, workspace->LR,
            sizeof(LR_lookup_table) * (num_atom_types * num_atom_types), 
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

    for( i = 0; i < num_atom_types; ++i )
    {
        if ( aggregated[i] )
        {
            for( j = i; j < num_atom_types; ++j )
            {
                if ( aggregated[j] )
                {
                    sCudaMalloc( (void **) &d_y,
                            sizeof(LR_data) * (control->tabulate + 1),
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( d_y, workspace->LR[ index_lr(i, j, num_atom_types) ].y,
                            sizeof(LR_data) * (control->tabulate + 1),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].y, &d_y,
                            sizeof(LR_data *),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[ index_lr(i, j, num_atom_types) ].H,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].H, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[ index_lr(i, j, num_atom_types) ].vdW,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].vdW, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[ index_lr(i, j, num_atom_types) ].CEvd,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].CEvd, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp,workspace->LR[ index_lr(i, j, num_atom_types) ].ele,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].ele, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

                    sCudaMalloc( (void **) &temp,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            __FILE__, __LINE__ );
                    sCudaMemcpyAsync( temp, workspace->LR[ index_lr(i, j, num_atom_types) ].CEclmb,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
                    sCudaMemcpyAsync( &workspace->d_LR[ index_lr(i, j, num_atom_types) ].CEclmb, &temp,
                            sizeof(cubic_spline_coef *),
                            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
                }
            }
        }
    }

    cudaStreamSynchronize( control->streams[0] );
}
