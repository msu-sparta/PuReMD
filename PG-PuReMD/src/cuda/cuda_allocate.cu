
#include "cuda_allocate.h"

#include "cuda_allocate.h"
#include "cuda_forces.h"
#include "cuda_list.h"
#include "cuda_neighbors.h"
#include "cuda_utils.h"

#include "../allocate.h"
#include "../index_utils.h"
#include "../tool_box.h"
#include "../vector.h"


GPU_GLOBAL void k_init_nbrs( ivec * const nbrs, int N )
{
    int i;
   
    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= N )
    {
        return;
    }

    nbrs[i][0] = -1; 
    nbrs[i][1] = -1; 
    nbrs[i][2] = -1; 
}


extern "C" void * sCudaHostAllocWrapper( size_t n, const char * const filename, int line )
{
    void *ptr;

    if ( n == 0 )
    {
        fprintf( stderr, "[ERROR] failed to allocate %zu bytes for array\n",
                n );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

    sCudaHostAlloc( &ptr, n, cudaHostAllocPortable, filename, line );

    return ptr;
}


extern "C" void * sCudaHostReallocWrapper( void *ptr, size_t cur_size, size_t new_size,
        const char * const filename, int line )
{
    void *new_ptr;

    if ( new_size == 0 )
    {
        fprintf( stderr, "[ERROR] sCudaHostReallocWrapper: failed to reallocate %zu bytes for array\n",
                new_size );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

    sCudaHostAlloc( &new_ptr, new_size, cudaHostAllocPortable, filename, line );

    if ( cur_size != 0 )
    {
        sCudaMemcpy( new_ptr, ptr, cur_size, cudaMemcpyHostToHost, filename, line );

        sCudaFreeHost( ptr, filename, line );
    }

    return new_ptr;
}


extern "C" void * sCudaHostCallocWrapper( size_t n, size_t size,
        const char * const filename, int line )
{
    void *ptr;

    sCudaHostAlloc( &ptr, n * size, cudaHostAllocPortable, filename, line );

    memset( ptr, 0, n * size );

    return ptr;
}


extern "C" void sCudaFreeHostWrapper( void * ptr, const char * const filename,
        int line )
{
    if ( ptr == NULL )
    {
        fprintf( stderr, "[WARNING] trying to free the already NULL pointer\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        return;
    }

    sCudaFreeHost( ptr, filename, line );
}


static void Cuda_Reallocate_List( reax_list * const list, size_t n, size_t max_intrs,
        int type )
{
    Cuda_Delete_List( list );
    Cuda_Make_List( n, max_intrs, type, list );
}


static void Cuda_Reallocate_System_Part1( reax_system * const system,
        control_params const * const control, storage * const workspace,
        int local_cap_old )
{
    int *temp;

    sCudaCheckMalloc( &workspace->d_workspace->scratch[0],
            &workspace->d_workspace->scratch_size[0], sizeof(int) * local_cap_old,
            __FILE__, __LINE__ );
    temp = (int *) workspace->d_workspace->scratch[0];

    sCudaMemcpyAsync( temp, system->d_cm_entries, sizeof(int) * local_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_cm_entries, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_cm_entries,
            sizeof(int) * system->local_cap, __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_cm_entries, temp, sizeof(int) * local_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

    sCudaMemcpyAsync( temp, system->d_max_cm_entries, sizeof(int) * local_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_max_cm_entries, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_max_cm_entries,
            sizeof(int) * system->local_cap, __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_max_cm_entries, temp, sizeof(int) * local_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
}


static void Cuda_Reallocate_System_Part2( reax_system * const system,
        control_params const * const control, storage * const workspace,
        int total_cap_old )
{
    int *temp;
    reax_atom *temp_atom;

    sCudaCheckMalloc( &workspace->d_workspace->scratch[0],
            &workspace->d_workspace->scratch_size[0],
            MAX( sizeof(reax_atom), sizeof(int) ) * total_cap_old,
            __FILE__, __LINE__ );

    temp_atom = (reax_atom *) workspace->d_workspace->scratch[0];

    /* free the existing storage for atoms, leave other info allocated */
    sCudaMemcpyAsync( temp_atom, system->d_my_atoms, sizeof(reax_atom) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_my_atoms, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_my_atoms,
            sizeof(reax_atom) * system->total_cap, __FILE__, __LINE__ );
    sCudaMemsetAsync( system->d_my_atoms, FALSE,
            sizeof(reax_atom) * system->total_cap,
            control->cuda_streams[0], __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_my_atoms, temp_atom, sizeof(reax_atom) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

    temp = (int *) workspace->d_workspace->scratch[0];

    /* list management */
    sCudaMemcpyAsync( temp, system->d_far_nbrs, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_far_nbrs, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_far_nbrs,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_far_nbrs, temp, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

    sCudaMemcpyAsync( temp, system->d_max_far_nbrs, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_max_far_nbrs, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_max_far_nbrs,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_max_far_nbrs, temp, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

    sCudaMemcpyAsync( temp, system->d_bonds, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_bonds, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_bonds,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_bonds, temp, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

    sCudaMemcpyAsync( temp, system->d_max_bonds, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_max_bonds, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_max_bonds,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_max_bonds, temp, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

    sCudaMemcpyAsync( temp, system->d_hbonds, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_hbonds, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_hbonds,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_hbonds, temp, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );

    sCudaMemcpyAsync( temp, system->d_max_hbonds, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaFree( system->d_max_hbonds, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_max_hbonds,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sCudaMemcpyAsync( system->d_max_hbonds, temp, sizeof(int) * total_cap_old,
            cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
}


void Cuda_Allocate_Grid( reax_system * const system,
        control_params const * const control )
{
    int total;
//    grid_cell local_cell;
    grid *host = &system->my_grid;
    grid *device = &system->d_my_grid;
//    ivec *nbrs_x = (ivec *) workspace->d_workspace->scratch[0];

    total = host->ncells[0] * host->ncells[1] * host->ncells[2];
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

    sCudaMalloc( (void **) &device->str, sizeof(int) * total,
            __FILE__, __LINE__ );
    sCudaMalloc( (void **) &device->end, sizeof(int) * total,
            __FILE__, __LINE__ );
    sCudaMalloc( (void **) &device->cutoff, sizeof(real) * total,
            __FILE__, __LINE__ );

    sCudaMalloc( (void **) &device->nbrs_x, sizeof(ivec) * total * host->max_nbrs,
            __FILE__, __LINE__ );
    sCudaMalloc( (void **) &device->nbrs_cp, sizeof(rvec) * total * host->max_nbrs,
            __FILE__, __LINE__ );
    sCudaMalloc( (void **) &device->rel_box, sizeof(ivec) * total,
            __FILE__, __LINE__ );

//    int blocks = (host->max_nbrs) / control->gpu_block_size + ((host->max_nbrs) % control->gpu_block_size == 0 ? 0 : 1); 
//
//    k_init_nbrs <<< blocks, control->gpu_block_size >>>
//        ( nbrs_x, host->max_nbrs );
//    cudaCheckError( );
//
//    sCudaMalloc( (void **)& device->cells, sizeof(grid_cell) * total,
//            __FILE__, __LINE__ );
//    fprintf( stderr, " Device cells address --> %ld \n", device->cells );
//    sCudaMalloc( (void **) &device->order,
//            sizeof(ivec) * (host->total + 1), __FILE__, __LINE__ );
//
//    local_cell.top = local_cell.mark = local_cell.str = local_cell.end = 0;
//    fprintf( stderr, "Total cells to be allocated -- > %d \n", total );
//    for (int i = 0; i < total; i++)
//    {
//        //fprintf( stderr, "Address of the local atom -> %ld  \n", &local_cell );
//
//        sCudaMalloc( (void **) &local_cell.atoms, sizeof(int) * host->max_atoms,
//                __FILE__, __LINE__ );
//        //fprintf( stderr, "Allocated address of the atoms --> %ld  (%d)\n", local_cell.atoms, host->max_atoms );
//
//        sCudaMalloc( (void **) &local_cell.nbrs_x, sizeof(ivec) * host->max_nbrs,
//                __FILE__, __LINE__ );
//        sCudaMemcpyAsync( local_cell.nbrs_x, nbrs_x, host->max_nbrs * sizeof(ivec),
//                cudaMemcpyDeviceToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
//        cudaStreamSynchronize( control->cuda_streams[0] );
//        //fprintf( stderr, "Allocated address of the nbrs_x--> %ld \n", local_cell.nbrs_x );
//
//        sCudaMalloc( (void **) &local_cell.nbrs_cp, sizeof(rvec) * host->max_nbrs,
//                __FILE__, __LINE__ );
//        //fprintf( stderr, "Allocated address of the nbrs_cp--> %ld \n", local_cell.nbrs_cp );
//
//        //sCudaMalloc( (void **) &local_cell.nbrs, sizeof(grid_cell *) * host->max_nbrs,
//        //      __FILE__, __LINE__ );
//        //fprintf( stderr, "Allocated address of the nbrs--> %ld \n", local_cell.nbrs );
//
//        sCudaMemcpyAsync( &device->cells[i], &local_cell, sizeof(grid_cell),
//                cudaMemcpyHostToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
//        cudaStreamSynchronize( control->cuda_streams[0] );
//    }
}


void Cuda_Deallocate_Grid_Cell_Atoms( reax_system * const system,
        control_params const * const control )
{
    int i, total;
    grid_cell local_cell;
    grid *host, *device;

    host = &system->my_grid;
    device = &system->d_my_grid;
    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    for ( i = 0; i < total; ++i )
    {
        sCudaMemcpyAsync( &local_cell, &device->cells[i], sizeof(grid_cell),
                cudaMemcpyDeviceToHost, control->cuda_streams[0], __FILE__, __LINE__ );
        cudaStreamSynchronize( control->cuda_streams[0] );

        sCudaFree( local_cell.atoms, __FILE__, __LINE__ );
    }
}


void Cuda_Allocate_Grid_Cell_Atoms( reax_system * const system,
        control_params const *const control, int cap )
{
    int i, total;
    grid_cell local_cell;
    grid *host, *device;

    host = &system->my_grid;
    device = &system->d_my_grid;
    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    for ( i = 0; i < total; i++ )
    {
        sCudaMemcpyAsync( &local_cell, &device->cells[i], sizeof(grid_cell),
                cudaMemcpyDeviceToHost, control->cuda_streams[0], __FILE__, __LINE__ );
        cudaStreamSynchronize( control->cuda_streams[0] );
        sCudaMalloc( (void **) &local_cell.atoms, sizeof(int) * cap, 
                __FILE__, __LINE__ );
        sCudaMemcpyAsync( &local_cell, &device->cells[i], sizeof(grid_cell),
                cudaMemcpyHostToDevice, control->cuda_streams[0], __FILE__, __LINE__ );
        cudaStreamSynchronize( control->cuda_streams[0] );
    }
}


void Cuda_Allocate_System( reax_system * const system,
        control_params const * const control )
{
    /* atoms */
    sCudaMalloc( (void **) &system->d_my_atoms,
            system->total_cap * sizeof(reax_atom), __FILE__, __LINE__ );
    sCudaMemsetAsync( system->d_my_atoms, FALSE,
            system->total_cap * sizeof(reax_atom),
            control->cuda_streams[0], __FILE__, __LINE__ );
    cudaStreamSynchronize( control->cuda_streams[0] );
    sCudaMalloc( (void **) &system->d_num_H_atoms, sizeof(int), __FILE__, __LINE__ );

    /* list management */
    sCudaMalloc( (void **) &system->d_far_nbrs,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_max_far_nbrs,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_total_far_nbrs,
            sizeof(int), __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->d_bonds,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_max_bonds,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_total_bonds,
            sizeof(int), __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->d_hbonds,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_max_hbonds,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_total_hbonds,
            sizeof(int), __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->d_cm_entries,
            system->local_cap * sizeof(int), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_max_cm_entries,
            system->local_cap * sizeof(int), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_total_cm_entries,
            sizeof(int), __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->d_total_thbodies,
            sizeof(int), __FILE__, __LINE__ );

    /* simulation boxes */
    sCudaMalloc( (void **) &system->d_big_box,
            sizeof(simulation_box), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_my_box,
            sizeof(simulation_box), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &system->d_my_ext_box,
            sizeof(simulation_box), __FILE__, __LINE__ );

    /* interaction parameters */
    sCudaMalloc( (void **) &system->reax_param.d_sbp,
            system->reax_param.num_atom_types * sizeof(single_body_parameters),
            __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->reax_param.d_tbp,
            SQR( system->reax_param.num_atom_types ) * sizeof(two_body_parameters), 
            __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->reax_param.d_thbp,
            CUBE( system->reax_param.num_atom_types ) * sizeof(three_body_header),
            __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->reax_param.d_hbp,
            CUBE( system->reax_param.num_atom_types ) * sizeof(hbond_parameters),
            __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->reax_param.d_fbp,
            FOURTH( system->reax_param.num_atom_types ) * sizeof(four_body_header),
            __FILE__, __LINE__ );

    sCudaMalloc( (void **) &system->reax_param.gp.d_l,
            system->reax_param.gp.n_global * sizeof(real),
            __FILE__, __LINE__ );
}


void Cuda_Allocate_Simulation_Data( simulation_data * const data, cudaStream_t s )
{
    sCudaMalloc( (void **) &data->d_my_ext_press,
            sizeof(rvec), __FILE__, __LINE__ );
    sCudaMalloc( (void **) &data->d_my_en,
            sizeof(real) * E_N, __FILE__, __LINE__ );
    cudaStreamSynchronize( s );
}


void Cuda_Allocate_Workspace_Part1( control_params const * const control, 
        storage * const workspace, int local_cap )
{
    int local_rvec;

    local_rvec = sizeof(rvec) * local_cap;

    /* integrator storage */
    if ( control->ensemble == nhNVT )
    {
        sCudaMalloc( (void **) &workspace->v_const, local_rvec,
                __FILE__, __LINE__ );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        sCudaMalloc( (void **) &workspace->mark, local_cap * sizeof(int),
                __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->old_mark, local_cap * sizeof(int),
                __FILE__, __LINE__ );
    }
    else
    {
        workspace->mark = NULL;
        workspace->old_mark = NULL;
    }

    if ( control->diffusion_coef )
    {
        sCudaMalloc( (void **) &workspace->x_old, local_cap * sizeof(rvec),
                __FILE__, __LINE__ );
    }
    else
    {
        workspace->x_old = NULL;
    }
}


void Cuda_Allocate_Workspace_Part2( control_params const * const control, 
        storage * const workspace, int total_cap )
{
    int total_real, total_rvec;
#if defined(DUAL_SOLVER)
    int total_rvec2;
#endif

    total_real = sizeof(real) * total_cap;
    total_rvec = sizeof(rvec) * total_cap;
#if defined(DUAL_SOLVER)
    total_rvec2 = sizeof(rvec2) * total_cap;
#endif

    /* bond order related storage  */
    sCudaMalloc( (void **) &workspace->total_bond_order, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->Deltap, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->Deltap_boc, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->dDeltap_self, total_rvec, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->Delta, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->Delta_lp, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->Delta_lp_temp, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->dDelta_lp, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->dDelta_lp_temp, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->Delta_e, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->Delta_boc, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->nlp, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->nlp_temp, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->Clp, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->vlpex, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->bond_mark, total_real, __FILE__, __LINE__ );

    /* charge matrix storage */
    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        sCudaMalloc( (void **) &workspace->Hdia_inv, total_real,
                __FILE__, __LINE__ );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        sCudaMalloc( (void **) &workspace->droptol, total_real,
                __FILE__, __LINE__ );
    }
    sCudaMalloc( (void **) &workspace->b_s, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->b_t, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->s, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->t, total_real, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
    sCudaMalloc( (void **) &workspace->b, total_rvec2, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->x, total_rvec2, __FILE__, __LINE__ );
#endif

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
        sCudaMalloc( (void **) &workspace->b_prc, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->b_prm, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->y,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->z,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->g,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->h,
                SQR(control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->hs,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->hc,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->v,
                SQR(control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        break;

    case SDM_S:
        sCudaMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sCudaMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
#endif
        break;

    case CG_S:
        sCudaMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sCudaMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
#endif
        break;

    case BiCGStab_S:
        sCudaMalloc( (void **) &workspace->y, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->g, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->z, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->r_hat, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q_hat, total_real, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sCudaMalloc( (void **) &workspace->y2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->g2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->z2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->r_hat2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q_hat2, total_rvec2, __FILE__, __LINE__ );
#endif
        break;

    case PIPECG_S:
        sCudaMalloc( (void **) &workspace->z, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->m, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->n, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->u, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->w, total_real, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sCudaMalloc( (void **) &workspace->z2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->m2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->n2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->u2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->w2, total_rvec2, __FILE__, __LINE__ );
#endif
        break;

    case PIPECR_S:
        sCudaMalloc( (void **) &workspace->z, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->m, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->n, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->u, total_real, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->w, total_real, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sCudaMalloc( (void **) &workspace->z2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->m2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->n2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->u2, total_rvec2, __FILE__, __LINE__ );
        sCudaMalloc( (void **) &workspace->w2, total_rvec2, __FILE__, __LINE__ );
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    cudaStreamSynchronize( control->cuda_streams[0] );

    /* force related storage */
    sCudaMalloc( (void **) &workspace->CdDelta, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->f, total_rvec, __FILE__, __LINE__ );
#if !defined(GPU_STREAM_SINGLE_ACCUM)
    sCudaMalloc( (void **) &workspace->CdDelta_bonds, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->CdDelta_multi, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->CdDelta_tor, total_real, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->f_hb, total_rvec, __FILE__, __LINE__ );
#if defined(FUSED_VDW_COULOMB)
    sCudaMalloc( (void **) &workspace->f_vdw_clmb, total_rvec, __FILE__, __LINE__ );
#else
    sCudaMalloc( (void **) &workspace->f_vdw, total_rvec, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &workspace->f_clmb, total_rvec, __FILE__, __LINE__ );
#endif
    sCudaMalloc( (void **) &workspace->f_tor, total_rvec, __FILE__, __LINE__ );
#endif
}


void Cuda_Deallocate_Workspace_Part1( control_params const * const control,
        storage * const workspace )
{
    /* Nose-Hoover integrator */
    if ( control->ensemble == nhNVT )
    {
        sCudaFree( workspace->v_const, __FILE__, __LINE__ );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        sCudaFree( workspace->mark, __FILE__, __LINE__ );
        sCudaFree( workspace->old_mark, __FILE__, __LINE__ );
    }
    else
    {
        workspace->mark = NULL;
        workspace->old_mark = NULL;
    }

    if ( control->diffusion_coef )
    {
        sCudaFree( workspace->x_old, __FILE__, __LINE__ );
    }
    else
    {
        workspace->x_old = NULL;
    }
}


void Cuda_Deallocate_Workspace_Part2( control_params const * const control,
        storage * const workspace )
{
    /* bond order related storage  */
    sCudaFree( workspace->total_bond_order, __FILE__, __LINE__ );
    sCudaFree( workspace->Deltap, __FILE__, __LINE__ );
    sCudaFree( workspace->Deltap_boc, __FILE__, __LINE__ );
    sCudaFree( workspace->dDeltap_self, __FILE__, __LINE__ );
    sCudaFree( workspace->Delta, __FILE__, __LINE__ );
    sCudaFree( workspace->Delta_lp, __FILE__, __LINE__ );
    sCudaFree( workspace->Delta_lp_temp, __FILE__, __LINE__ );
    sCudaFree( workspace->dDelta_lp, __FILE__, __LINE__ );
    sCudaFree( workspace->dDelta_lp_temp, __FILE__, __LINE__ );
    sCudaFree( workspace->Delta_e, __FILE__, __LINE__ );
    sCudaFree( workspace->Delta_boc, __FILE__, __LINE__ );
    sCudaFree( workspace->nlp, __FILE__, __LINE__ );
    sCudaFree( workspace->nlp_temp, __FILE__, __LINE__ );
    sCudaFree( workspace->Clp, __FILE__, __LINE__ );
    sCudaFree( workspace->vlpex, __FILE__, __LINE__ );
    sCudaFree( workspace->bond_mark, __FILE__, __LINE__ );

    /* charge matrix storage */
    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        sCudaFree( workspace->Hdia_inv, __FILE__, __LINE__ );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        sCudaFree( workspace->droptol, __FILE__, __LINE__ );
    }
    sCudaFree( workspace->b_s, __FILE__, __LINE__ );
    sCudaFree( workspace->b_t, __FILE__, __LINE__ );
    sCudaFree( workspace->s, __FILE__, __LINE__ );
    sCudaFree( workspace->t, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
    sCudaFree( workspace->b, __FILE__, __LINE__ );
    sCudaFree( workspace->x, __FILE__, __LINE__ );
#endif

    switch ( control->cm_solver_type )
    {
        case GMRES_S:
        case GMRES_H_S:
            sCudaFree( workspace->b_prc, __FILE__, __LINE__ );
            sCudaFree( workspace->b_prm, __FILE__, __LINE__ );
            sCudaFree( workspace->y, __FILE__, __LINE__ );
            sCudaFree( workspace->z, __FILE__, __LINE__ );
            sCudaFree( workspace->g, __FILE__, __LINE__ );
            sCudaFree( workspace->h, __FILE__, __LINE__ );
            sCudaFree( workspace->hs, __FILE__, __LINE__ );
            sCudaFree( workspace->hc, __FILE__, __LINE__ );
            sCudaFree( workspace->v, __FILE__, __LINE__ );
            break;

        case CG_S:
            sCudaFree( workspace->r, __FILE__, __LINE__ );
            sCudaFree( workspace->d, __FILE__, __LINE__ );
            sCudaFree( workspace->q, __FILE__, __LINE__ );
            sCudaFree( workspace->p, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sCudaFree( workspace->r2, __FILE__, __LINE__ );
            sCudaFree( workspace->d2, __FILE__, __LINE__ );
            sCudaFree( workspace->q2, __FILE__, __LINE__ );
            sCudaFree( workspace->p2, __FILE__, __LINE__ );
#endif
            break;

        case SDM_S:
            sCudaFree( workspace->r, __FILE__, __LINE__ );
            sCudaFree( workspace->d, __FILE__, __LINE__ );
            sCudaFree( workspace->q, __FILE__, __LINE__ );
            sCudaFree( workspace->p, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sCudaFree( workspace->r2, __FILE__, __LINE__ );
            sCudaFree( workspace->d2, __FILE__, __LINE__ );
            sCudaFree( workspace->q2, __FILE__, __LINE__ );
            sCudaFree( workspace->p2, __FILE__, __LINE__ );
#endif
            break;

        case BiCGStab_S:
            sCudaFree( workspace->y, __FILE__, __LINE__ );
            sCudaFree( workspace->g, __FILE__, __LINE__ );
            sCudaFree( workspace->z, __FILE__, __LINE__ );
            sCudaFree( workspace->r, __FILE__, __LINE__ );
            sCudaFree( workspace->d, __FILE__, __LINE__ );
            sCudaFree( workspace->q, __FILE__, __LINE__ );
            sCudaFree( workspace->p, __FILE__, __LINE__ );
            sCudaFree( workspace->r_hat, __FILE__, __LINE__ );
            sCudaFree( workspace->q_hat, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sCudaFree( workspace->y2, __FILE__, __LINE__ );
            sCudaFree( workspace->g2, __FILE__, __LINE__ );
            sCudaFree( workspace->z2, __FILE__, __LINE__ );
            sCudaFree( workspace->r2, __FILE__, __LINE__ );
            sCudaFree( workspace->d2, __FILE__, __LINE__ );
            sCudaFree( workspace->q2, __FILE__, __LINE__ );
            sCudaFree( workspace->p2, __FILE__, __LINE__ );
            sCudaFree( workspace->r_hat2, __FILE__, __LINE__ );
            sCudaFree( workspace->q_hat2, __FILE__, __LINE__ );
#endif
            break;

        case PIPECG_S:
            sCudaFree( workspace->z, __FILE__, __LINE__ );
            sCudaFree( workspace->r, __FILE__, __LINE__ );
            sCudaFree( workspace->d, __FILE__, __LINE__ );
            sCudaFree( workspace->q, __FILE__, __LINE__ );
            sCudaFree( workspace->p, __FILE__, __LINE__ );
            sCudaFree( workspace->m, __FILE__, __LINE__ );
            sCudaFree( workspace->n, __FILE__, __LINE__ );
            sCudaFree( workspace->u, __FILE__, __LINE__ );
            sCudaFree( workspace->w, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sCudaFree( workspace->z2, __FILE__, __LINE__ );
            sCudaFree( workspace->r2, __FILE__, __LINE__ );
            sCudaFree( workspace->d2, __FILE__, __LINE__ );
            sCudaFree( workspace->q2, __FILE__, __LINE__ );
            sCudaFree( workspace->p2, __FILE__, __LINE__ );
            sCudaFree( workspace->m2, __FILE__, __LINE__ );
            sCudaFree( workspace->n2, __FILE__, __LINE__ );
            sCudaFree( workspace->u2, __FILE__, __LINE__ );
            sCudaFree( workspace->w2, __FILE__, __LINE__ );
#endif
            break;

        case PIPECR_S:
            sCudaFree( workspace->z, __FILE__, __LINE__ );
            sCudaFree( workspace->r, __FILE__, __LINE__ );
            sCudaFree( workspace->d, __FILE__, __LINE__ );
            sCudaFree( workspace->q, __FILE__, __LINE__ );
            sCudaFree( workspace->p, __FILE__, __LINE__ );
            sCudaFree( workspace->m, __FILE__, __LINE__ );
            sCudaFree( workspace->n, __FILE__, __LINE__ );
            sCudaFree( workspace->u, __FILE__, __LINE__ );
            sCudaFree( workspace->w, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sCudaFree( workspace->z2, __FILE__, __LINE__ );
            sCudaFree( workspace->r2, __FILE__, __LINE__ );
            sCudaFree( workspace->d2, __FILE__, __LINE__ );
            sCudaFree( workspace->q2, __FILE__, __LINE__ );
            sCudaFree( workspace->p2, __FILE__, __LINE__ );
            sCudaFree( workspace->m2, __FILE__, __LINE__ );
            sCudaFree( workspace->n2, __FILE__, __LINE__ );
            sCudaFree( workspace->u2, __FILE__, __LINE__ );
            sCudaFree( workspace->w2, __FILE__, __LINE__ );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    /* force related storage */
    sCudaFree( workspace->CdDelta, __FILE__, __LINE__ );
    sCudaFree( workspace->f, __FILE__, __LINE__ );
#if !defined(GPU_STREAM_SINGLE_ACCUM)
    sCudaFree( workspace->CdDelta_bonds, __FILE__, __LINE__ );
    sCudaFree( workspace->CdDelta_multi, __FILE__, __LINE__ );
    sCudaFree( workspace->CdDelta_tor, __FILE__, __LINE__ );
    sCudaFree( workspace->f_hb, __FILE__, __LINE__ );
#if defined(FUSED_VDW_COULOMB)
    sCudaFree( workspace->f_vdw_clmb, __FILE__, __LINE__ );
#else
    sCudaFree( workspace->f_vdw, __FILE__, __LINE__ );
    sCudaFree( workspace->f_clmb, __FILE__, __LINE__ );
#endif
    sCudaFree( workspace->f_tor, __FILE__, __LINE__ );
#endif
}


/* Allocate sparse matrix struc
 *
 * H: pointer to struct
 * n: currently utilized number of rows
 * n_max: max number of rows allocated
 * m: max number of entries allocated
 * format: sparse matrix format
 */
void Cuda_Allocate_Matrix( sparse_matrix * const H, int n, int n_max, int m,
       int format, cudaStream_t s )
{
    H->allocated = TRUE;
    H->n = n;
    H->n_max = n_max;
    H->m = m;
    H->format = format;

    sCudaMalloc( (void **) &H->start, sizeof(int) * n_max, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &H->end, sizeof(int) * n_max, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &H->j, sizeof(int) * m, __FILE__, __LINE__ );
    sCudaMalloc( (void **) &H->val, sizeof(real) * m, __FILE__, __LINE__ );
}


void Cuda_Deallocate_Matrix( sparse_matrix * const H )
{
    H->allocated = FALSE;
    H->n = 0;
    H->n_max = 0;
    H->m = 0;

    sCudaFree( H->start, __FILE__, __LINE__ );
    sCudaFree( H->end, __FILE__, __LINE__ );
    sCudaFree( H->j, __FILE__, __LINE__ );
    sCudaFree( H->val, __FILE__, __LINE__ );
}


void Cuda_Reallocate_Part1( reax_system * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, mpi_datatypes * const mpi_data )
{
    int i, j, k, renbr;
    int *realloc;
    grid *g;

    realloc = workspace->realloc;
    g = &system->my_grid;
    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* grid */
    if ( renbr == TRUE && realloc[RE_GCELL_ATOMS] > -1 )
    {
        for ( i = g->native_str[0]; i < g->native_end[0]; i++ )
        {
            for ( j = g->native_str[1]; j < g->native_end[1]; j++ )
            {
                for ( k = g->native_str[2]; k < g->native_end[2]; k++ )
                {
                    sfree( g->cells[ index_grid_3d(i,j,k,g) ].atoms, __FILE__, __LINE__ );
                    g->cells[ index_grid_3d(i,j,k,g) ].atoms = (int *)
                            scalloc( realloc[RE_GCELL_ATOMS], sizeof(int), __FILE__, __LINE__ );
                }
            }
        }

        fprintf( stderr, "p:%d - *** Reallocating Grid Cell Atoms *** Step:%d\n", system->my_rank, data->step );

//        Cuda_Deallocate_Grid_Cell_Atoms( system );
//        Cuda_Allocate_Grid_Cell_Atoms( system, realloc[RE_GCELL_ATOMS] );
        realloc[RE_GCELL_ATOMS] = -1;
    }
}


void Cuda_Reallocate_Part2( reax_system * const system,
        control_params const * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, mpi_datatypes * const mpi_data )
{
    int nflag, Nflag, local_cap_old, total_cap_old, renbr, format;
    int *realloc;
    sparse_matrix *H;

    realloc = workspace->realloc;
    H = &workspace->d_workspace->H;
    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* IMPORTANT: LOOSE ZONES CHECKS ARE DISABLED FOR NOW BY &&'ing with FALSE!!! */
    nflag = FALSE;
    if ( system->n >= (int) CEIL( DANGER_ZONE * system->local_cap )
            || (FALSE && system->n <= (int) CEIL( LOOSE_ZONE * system->local_cap )) )
    {
        nflag = TRUE;
        local_cap_old = system->local_cap;
        system->local_cap = (int) CEIL( system->n * SAFE_ZONE );
    }

    Nflag = FALSE;
    if ( system->N >= (int) CEIL( DANGER_ZONE * system->total_cap )
            || (FALSE && system->N <= (int) CEIL( LOOSE_ZONE * system->total_cap )) )
    {
        Nflag = TRUE;
        total_cap_old = system->total_cap;
        system->total_cap = (int) CEIL( system->N * SAFE_ZONE );
    }

    if ( nflag == TRUE )
    {
        Cuda_Reallocate_System_Part1( system, control, workspace, local_cap_old );

        Cuda_Deallocate_Workspace_Part1( control, workspace->d_workspace );
        Cuda_Allocate_Workspace_Part1( control, workspace->d_workspace,
                system->local_cap );
    }

    if ( Nflag == TRUE )
    {
        Cuda_Reallocate_System_Part2( system, control, workspace, total_cap_old );

        Cuda_Deallocate_Workspace_Part2( control, workspace->d_workspace );
        Cuda_Allocate_Workspace_Part2( control, workspace->d_workspace,
                system->total_cap );
    }

    /* far neighbors */
    if ( renbr == TRUE && (Nflag == TRUE || realloc[RE_FAR_NBRS] == TRUE) )
    {
        Cuda_Reallocate_List( lists[FAR_NBRS], system->total_cap,
                system->total_far_nbrs, TYP_FAR_NEIGHBOR );
        Cuda_Init_Neighbor_Indices( system, control, lists[FAR_NBRS] );
        realloc[RE_FAR_NBRS] = FALSE;
    }

    /* charge matrix */
    if ( nflag == TRUE || realloc[RE_CM] == TRUE )
    {
        format = H->format;

        Cuda_Deallocate_Matrix( H );
        Cuda_Allocate_Matrix( H, system->n, system->local_cap,
                system->total_cm_entries, format, control->cuda_streams[0] );

        realloc[RE_CM] = FALSE;
    }

    /* bonds list */
    if ( Nflag == TRUE || realloc[RE_BONDS] == TRUE )
    {
        Cuda_Reallocate_List( lists[BONDS], system->total_cap,
                system->total_bonds, TYP_BOND );

        realloc[RE_BONDS] = FALSE;
    }

    /* hydrogen bonds list */
    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0
            && (Nflag == TRUE || realloc[RE_HBONDS] == TRUE) )
    {
        Cuda_Reallocate_List( lists[HBONDS], system->total_cap,
                system->total_hbonds, TYP_HBOND );

        realloc[RE_HBONDS] = FALSE;
    }

    /* 3-body list */
    if ( Nflag == TRUE || realloc[RE_THBODY] == TRUE )
    {
        Cuda_Reallocate_List( lists[THREE_BODIES], system->total_thbodies_indices,
                system->total_thbodies, TYP_THREE_BODY );

        realloc[RE_THBODY] = FALSE;
    }
}
