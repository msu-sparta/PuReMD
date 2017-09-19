
#include "cuda_lookup.h"

#include "cuda_utils.h"

#include "../index_utils.h"


void copy_LR_table_to_device( reax_system *system, control_params *control,
        int *aggregated )
{
    int i, j;
    int num_atom_types;
    LR_data *d_y;
    cubic_spline_coef *temp;

    num_atom_types = system->reax_param.num_atom_types;

    fprintf( stderr, "Copying the LR Lookyp Table to the device ... \n" );

    cuda_malloc( (void **) &d_LR,
            sizeof(LR_lookup_table) * ( num_atom_types * num_atom_types ),
            FALSE, "LR_lookup:table" );

    /*
       for( i = 0; i < MAX_ATOM_TYPES; ++i )
       existing_types[i] = 0;

       for( i = 0; i < system->N; ++i )
       existing_types[ system->atoms[i].type ] = 1;
     */

    copy_host_device( LR, d_LR,
            sizeof(LR_lookup_table) * (num_atom_types * num_atom_types), 
            cudaMemcpyHostToDevice, "LR_lookup:table" );

    for( i = 0; i < num_atom_types; ++i )
    {
        if ( aggregated[i] )
        {
            for( j = i; j < num_atom_types; ++j )
            {
                if ( aggregated[j] )
                {
                    cuda_malloc( (void **) &d_y,
                            sizeof(LR_data) * (control->tabulate + 1), FALSE, "LR_lookup:d_y" );
                    copy_host_device( LR[ index_lr(i, j, num_atom_types) ].y, d_y, 
                            sizeof(LR_data) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:y" );
                    copy_host_device ( &d_y, &d_LR[ index_lr(i, j, num_atom_types) ].y, 
                            sizeof(LR_data *), cudaMemcpyHostToDevice, "LR_lookup:y" );

                    cuda_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:h" );
                    copy_host_device( LR [ index_lr (i, j, num_atom_types) ].H, temp, 
                            sizeof(cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:h" );
                    copy_host_device( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].H, 
                            sizeof(cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:h" );

                    cuda_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:vdW" );
                    copy_host_device( LR [ index_lr (i, j, num_atom_types) ].vdW, temp, 
                            sizeof(cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:vdW" );
                    copy_host_device( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].vdW,
                            sizeof(cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:vdW" );

                    cuda_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:CEvd" );
                    copy_host_device( LR [ index_lr (i, j, num_atom_types) ].CEvd, temp, 
                            sizeof(cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:CEvd" );
                    copy_host_device( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].CEvd, 
                            sizeof(cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:CDvd");

                    cuda_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:ele" );
                    copy_host_device( LR [ index_lr (i, j, num_atom_types) ].ele, temp,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:ele" );
                    copy_host_device( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].ele,
                            sizeof(cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:ele" );

                    cuda_malloc( (void **) &temp, sizeof(cubic_spline_coef) * (control->tabulate + 1), FALSE, "LR_lookup:ceclmb" );
                    copy_host_device( LR [ index_lr (i, j, num_atom_types) ].CEclmb, temp,
                            sizeof(cubic_spline_coef) * (control->tabulate + 1), cudaMemcpyHostToDevice, "LR_lookup:ceclmb" );
                    copy_host_device( &temp, &d_LR [ index_lr (i, j, num_atom_types) ].CEclmb,
                            sizeof(cubic_spline_coef *), cudaMemcpyHostToDevice, "LR_lookup:ceclmb" );
                }
            }
        }
    }

    fprintf( stderr, "Copy of the LR Lookup Table to the device complete ... \n" );
}

