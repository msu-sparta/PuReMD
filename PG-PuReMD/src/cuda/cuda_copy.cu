
#include "cuda_copy.h"

#include "cuda_utils.h"

#include "../list.h"
#include "../vector.h"


/* Copy grid info from host to device */
extern "C" void Cuda_Copy_Grid_Host_to_Device( control_params *control,
        grid *host, grid *device )
{
    int total;

    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    sCudaMemcpyAsync( device->str, host->str, sizeof(int) * total,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( device->end, host->end, sizeof(int) * total,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( device->cutoff, host->cutoff, sizeof(real) * total,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( device->nbrs_x, host->nbrs_x, sizeof(ivec) * total * host->max_nbrs,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( device->nbrs_cp, host->nbrs_cp, sizeof(rvec) * total * host->max_nbrs,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( device->rel_box, host->rel_box, sizeof(ivec) * total,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

    device->total = host->total;
    device->max_atoms = host->max_atoms;
    device->max_nbrs = host->max_nbrs;
    ivec_Copy( device->ncells, host->ncells );
    rvec_Copy( device->cell_len, host->cell_len );
    rvec_Copy( device->inv_len, host->inv_len );
    ivec_Copy( device->bond_span, host->bond_span );
    ivec_Copy( device->nonb_span, host->nonb_span );
    ivec_Copy( device->vlist_span, host->vlist_span );
    ivec_Copy( device->native_cells, host->native_cells );
    ivec_Copy( device->native_str, host->native_str );
    ivec_Copy( device->native_end, host->native_end );
    device->ghost_cut = host->ghost_cut;
    ivec_Copy( device->ghost_span, host->ghost_span );
    ivec_Copy( device->ghost_nonb_span, host->ghost_nonb_span );
    ivec_Copy( device->ghost_hbond_span, host->ghost_hbond_span );
    ivec_Copy( device->ghost_bond_span, host->ghost_bond_span );

    cudaStreamSynchronize( control->streams[0] );
}


/* Copy atom info from host to device */
extern "C" void Cuda_Copy_Atoms_Host_to_Device( reax_system *system, control_params *control )
{
    sCudaMemcpyAsync( system->d_my_atoms, system->my_atoms, sizeof(reax_atom) * system->N,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[0] );
}


/* Copy sparse matrix from host to device */
extern "C" void Cuda_Copy_Matrix_Host_to_Device( sparse_matrix const * const A,
        sparse_matrix * const d_A, cudaStream_t s )
{
    assert( d_A->n_max >= A->n_max );
    assert( d_A->m >= A->m );

    sCudaMemcpyAsync( d_A->start, A->start, sizeof(int) * A->n_max,
            cudaMemcpyHostToDevice, s, __FILE__, __LINE__ );
    sCudaMemcpyAsync( d_A->end, A->end, sizeof(int) * A->n_max,
            cudaMemcpyHostToDevice, s, __FILE__, __LINE__ );
    sCudaMemcpyAsync( d_A->j, A->j, sizeof(int) * A->m,
            cudaMemcpyHostToDevice, s, __FILE__, __LINE__ );
    sCudaMemcpyAsync( d_A->val, A->val, sizeof(real) * A->m,
            cudaMemcpyHostToDevice, s, __FILE__, __LINE__ );

    d_A->n = A->n;

    cudaStreamSynchronize( s );
}


/* Copy atomic system info from host to device */
extern "C" void Cuda_Copy_System_Host_to_Device( reax_system *system,
        control_params *control )
{
    Cuda_Copy_Atoms_Host_to_Device( system, control );

    sCudaMemcpyAsync( system->d_my_box, &system->my_box,
            sizeof(simulation_box),
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

    sCudaMemcpyAsync( system->d_my_ext_box, &system->my_ext_box,
            sizeof(simulation_box),
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

    sCudaMemcpyAsync( system->reax_param.d_sbp, system->reax_param.sbp,
            sizeof(single_body_parameters) * system->reax_param.num_atom_types,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->reax_param.d_tbp, system->reax_param.tbp,
            sizeof(two_body_parameters) * POW(system->reax_param.num_atom_types, 2),
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->reax_param.d_thbp, system->reax_param.thbp,
            sizeof(three_body_header) * POW(system->reax_param.num_atom_types, 3),
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->reax_param.d_hbp, system->reax_param.hbp,
            sizeof(hbond_parameters) * POW(system->reax_param.num_atom_types, 3),
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->reax_param.d_fbp, system->reax_param.fbp,
            sizeof(four_body_header) * POW(system->reax_param.num_atom_types, 4),
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

    sCudaMemcpyAsync( system->reax_param.d_gp.l, system->reax_param.gp.l,
            sizeof(real) * system->reax_param.gp.n_global,
            cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[0] );

    system->reax_param.d_gp.n_global = system->reax_param.gp.n_global; 
    system->reax_param.d_gp.vdw_type = system->reax_param.gp.vdw_type; 
}


/* Copy atom info from device to host */
extern "C" void Cuda_Copy_Atoms_Device_to_Host( reax_system * const system,
        control_params const * const control )
{
    sCudaMemcpyAsync( system->my_atoms, system->d_my_atoms,
            sizeof(reax_atom) * system->N,
            cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );

    cudaStreamSynchronize( control->streams[0] );
}


/* Copy sparse matrix from device to host */
extern "C" void Cuda_Copy_Matrix_Device_to_Host( sparse_matrix * const A,
        sparse_matrix const * const d_A, cudaStream_t s )
{
    assert( A->n_max >= d_A->n_max );
    assert( A->m >= d_A->m );

    sCudaMemcpyAsync( A->start, d_A->start, sizeof(int) * d_A->n_max,
            cudaMemcpyDeviceToHost, s, __FILE__, __LINE__ );
    sCudaMemcpyAsync( A->end, d_A->end, sizeof(int) * d_A->n_max,
            cudaMemcpyDeviceToHost, s, __FILE__, __LINE__ );
    sCudaMemcpyAsync( A->j, d_A->j, sizeof(int) * d_A->m,
            cudaMemcpyDeviceToHost, s, __FILE__, __LINE__ );
    sCudaMemcpyAsync( A->val, d_A->val, sizeof(real) * d_A->m,
            cudaMemcpyDeviceToHost, s, __FILE__, __LINE__ );

    A->n = d_A->n;

    cudaStreamSynchronize( s );
}


/* Copy simulation data from device to host */
extern "C" void Cuda_Copy_Simulation_Data_Device_to_Host( control_params const * const control,
        simulation_data * const data, simulation_data * const d_data )
{
    sCudaMemcpyAsync( &data->my_en, &d_data->my_en, sizeof(energy_data), 
            cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
    if ( control->virial == 1 )
    {
        sCudaMemcpyAsync( &data->kin_press, &d_data->kin_press, sizeof(real), 
                cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
        sCudaMemcpyAsync( data->int_press, d_data->int_press, sizeof(rvec), 
                cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
        sCudaMemcpyAsync( data->ext_press, d_data->ext_press, sizeof(rvec), 
                cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
    }

    cudaStreamSynchronize( control->streams[0] );
}


/* Copy interaction lists from device to host,
 * with allocation for the host list */
extern "C" void Cuda_Copy_List_Device_to_Host( control_params *control,
        reax_list *host_list, reax_list *device_list, int type )
{
    int format;

//    assert( device_list != NULL );
//    assert( device_list->allocated == TRUE );

    format = host_list->format;

    if ( host_list != NULL && host_list->allocated == TRUE )
    {
        Delete_List( host_list );
    }
    Make_List( device_list->n, device_list->max_intrs, type, format, host_list );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, " [INFO] trying to copy %d list from device to host\n", type );
#endif

    sCudaMemcpyAsync( host_list->index, device_list->index,
            sizeof(int) * device_list->n,
            cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( host_list->end_index, device_list->end_index,
            sizeof(int) * device_list->n,
            cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );

    switch ( type )
    {   
        case TYP_FAR_NEIGHBOR:
            sCudaMemcpyAsync( host_list->far_nbr_list.nbr, device_list->far_nbr_list.nbr,
                    sizeof(int) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
            sCudaMemcpyAsync( host_list->far_nbr_list.rel_box, device_list->far_nbr_list.rel_box,
                    sizeof(ivec) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
            sCudaMemcpyAsync( host_list->far_nbr_list.d, device_list->far_nbr_list.d,
                    sizeof(real) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
            sCudaMemcpyAsync( host_list->far_nbr_list.dvec, device_list->far_nbr_list.dvec,
                    sizeof(rvec) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
            break;

        case TYP_BOND:
            sCudaMemcpyAsync( host_list->bond_list, device_list->bond_list,
                    sizeof(bond_data) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
            break;

        case TYP_HBOND:
            sCudaMemcpyAsync( host_list->hbond_list, device_list->hbond_list,
                    sizeof(hbond_data) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
            break;

        case TYP_THREE_BODY:
            sCudaMemcpyAsync( host_list->three_body_list,
                    device_list->three_body_list,
                    sizeof(three_body_interaction_data ) * device_list->max_intrs,
                    cudaMemcpyDeviceToHost, control->streams[0], __FILE__, __LINE__ );
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown list synching from device to host (%d)\n",
                    type );
            exit( INVALID_INPUT );
            break;
    }  

    cudaStreamSynchronize( control->streams[0] );
}

/* Copy atom info from device to host */
extern "C" void Cuda_Copy_MPI_Data_Host_to_Device( control_params *control,
        mpi_datatypes *mpi_data )
{
    sCudaCheckMalloc( &mpi_data->d_in1_buffer, &mpi_data->d_in1_buffer_size,
            mpi_data->in1_buffer_size, __FILE__, __LINE__ );

    sCudaCheckMalloc( &mpi_data->d_in2_buffer, &mpi_data->d_in2_buffer_size,
            mpi_data->in2_buffer_size, __FILE__, __LINE__ );

    for ( int i = 0; i < MAX_NBRS; ++i )
    {
        mpi_data->d_out_buffers[i].cnt = mpi_data->out_buffers[i].cnt;

        sCudaCheckMalloc( (void **) &mpi_data->d_out_buffers[i].index,
                &mpi_data->d_out_buffers[i].index_size,
                mpi_data->out_buffers[i].index_size, __FILE__, __LINE__ );

        sCudaCheckMalloc( &mpi_data->d_out_buffers[i].out_atoms,
                &mpi_data->d_out_buffers[i].out_atoms_size,
                mpi_data->out_buffers[i].out_atoms_size, __FILE__, __LINE__ );

        /* index is set during SendRecv and reused during MPI comms afterward,
         * so copy to device while SendRecv is still done on the host */
        sCudaMemcpyAsync( mpi_data->d_out_buffers[i].index, mpi_data->out_buffers[i].index,
                mpi_data->out_buffers[i].index_size,
                cudaMemcpyHostToDevice, control->streams[0], __FILE__, __LINE__ );
    }

    cudaStreamSynchronize( control->streams[0] );
}
