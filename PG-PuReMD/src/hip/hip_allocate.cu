
#include "hip_allocate.h"

#include "hip_allocate.h"
#include "hip_forces.h"
#include "hip_list.h"
#include "hip_neighbors.h"
#include "hip_utils.h"

#include "../allocate.h"
#include "../index_utils.h"
#include "../tool_box.h"
#include "../vector.h"


GPU_GLOBAL void k_init_nbrs( ivec *nbrs, int N )
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


extern "C" void * sHipHostMallocWrapper( size_t n, const char * const filename, int line )
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

    sHipHostMalloc( &ptr, n, hipHostMallocNumaUser | hipHostMallocPortable, filename, line );

    return ptr;
}


extern "C" void * sHipHostReallocWrapper( void *ptr, size_t cur_size, size_t new_size,
        const char * const filename, int line )
{
    void *new_ptr;

    if ( new_size == 0 )
    {
        fprintf( stderr, "[ERROR] sHipHostReallocWrapper: failed to reallocate %zu bytes for array\n",
                new_size );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        exit( INSUFFICIENT_MEMORY );
    }

    sHipHostMalloc( &new_ptr, new_size, hipHostMallocNumaUser | hipHostMallocPortable, filename, line );

    if ( cur_size != 0 )
    {
        sHipMemcpy( new_ptr, ptr, cur_size, hipMemcpyHostToHost, filename, line );

        sHipHostFree( ptr, filename, line );
    }

    return new_ptr;
}


extern "C" void * sHipHostCallocWrapper( size_t n, size_t size,
        const char * const filename, int line )
{
    void *ptr;

    sHipHostMalloc( &ptr, n * size, hipHostMallocNumaUser | hipHostMallocPortable, filename, line );

    memset( ptr, 0, n * size );

    return ptr;
}


extern "C" void sHipHostFreeWrapper( void *ptr, const char * const filename,
        int line )
{
    if ( ptr == NULL )
    {
        fprintf( stderr, "[WARNING] trying to free the already NULL pointer\n" );
        fprintf( stderr, "    [INFO] At line %d in file %.*s\n",
                line, (int) strlen(filename), filename );
        return;
    }

    sHipHostFree( ptr, filename, line );
}


static void Hip_Reallocate_List( reax_list *list, size_t n, size_t max_intrs,
        int type )
{
    Hip_Delete_List( list );
    Hip_Make_List( n, max_intrs, type, list );
}


static void Hip_Reallocate_System_Part1( reax_system *system,
        control_params *control, storage *workspace, int local_cap_old )
{
    int *temp;

    sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            sizeof(int) * local_cap_old,
            __FILE__, __LINE__ );
    temp = (int *) workspace->scratch[0];

    sHipMemcpyAsync( temp, system->d_cm_entries, sizeof(int) * local_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_cm_entries, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_cm_entries,
            sizeof(int) * system->local_cap, __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_cm_entries, temp, sizeof(int) * local_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    sHipMemcpyAsync( temp, system->d_max_cm_entries, sizeof(int) * local_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_max_cm_entries, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_max_cm_entries,
            sizeof(int) * system->local_cap, __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_max_cm_entries, temp, sizeof(int) * local_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
}


static void Hip_Reallocate_System_Part2( reax_system *system, control_params *control,
        storage *workspace, int total_cap_old )
{
    int *temp;
    reax_atom *temp_atom;

    sHipCheckMalloc( &workspace->scratch[0], &workspace->scratch_size[0],
            MAX( sizeof(reax_atom), sizeof(int) ) * total_cap_old,
            __FILE__, __LINE__ );
    temp = (int *) workspace->scratch[0];
    temp_atom = (reax_atom *) workspace->scratch[0];

    /* free the existing storage for atoms, leave other info allocated */
    sHipMemcpyAsync( temp_atom, system->d_my_atoms, sizeof(reax_atom) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_my_atoms, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_my_atoms,
            sizeof(reax_atom) * system->total_cap, __FILE__, __LINE__ );
    sHipMemsetAsync( system->d_my_atoms, FALSE,
            sizeof(reax_atom) * system->total_cap, 
            control->hip_streams[0], __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_my_atoms, temp_atom, sizeof(reax_atom) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    /* list management */
    sHipMemcpyAsync( temp, system->d_far_nbrs, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_far_nbrs, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_far_nbrs,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_far_nbrs, temp, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    sHipMemcpyAsync( temp, system->d_max_far_nbrs, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_max_far_nbrs, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_max_far_nbrs,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_max_far_nbrs, temp, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    sHipMemcpyAsync( temp, system->d_bonds, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_bonds, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_bonds,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_bonds, temp, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    sHipMemcpyAsync( temp, system->d_max_bonds, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_max_bonds, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_max_bonds,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_max_bonds, temp, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    sHipMemcpyAsync( temp, system->d_hbonds, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_hbonds, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_hbonds,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_hbonds, temp, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );

    sHipMemcpyAsync( temp, system->d_max_hbonds, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipFree( system->d_max_hbonds, __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_max_hbonds,
            sizeof(int) * system->total_cap, __FILE__, __LINE__ );
    sHipMemcpyAsync( system->d_max_hbonds, temp, sizeof(int) * total_cap_old,
            hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
}


void Hip_Allocate_Control( control_params *control )
{
    sHipMalloc( (void **)&control->d_control_params,
            sizeof(control_params), __FILE__, __LINE__ );
    sHipMemcpyAsync( control->d_control_params, control,
            sizeof(control_params), hipMemcpyHostToDevice,
            control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
}


void Hip_Allocate_Grid( reax_system *system, control_params *control )
{
    int total;
//    grid_cell local_cell;
    grid *host = &system->my_grid;
    grid *device = &system->d_my_grid;
//    ivec *nbrs_x = (ivec *) workspace->scratch[0];

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

    sHipMalloc( (void **) &device->str, sizeof(int) * total,
            __FILE__, __LINE__ );
    sHipMalloc( (void **) &device->end, sizeof(int) * total,
            __FILE__, __LINE__ );
    sHipMalloc( (void **) &device->cutoff, sizeof(real) * total,
            __FILE__, __LINE__ );

    sHipMalloc( (void **) &device->nbrs_x, sizeof(ivec) * total * host->max_nbrs,
            __FILE__, __LINE__ );
    sHipMalloc( (void **) &device->nbrs_cp, sizeof(rvec) * total * host->max_nbrs,
            __FILE__, __LINE__ );
    sHipMalloc( (void **) &device->rel_box, sizeof(ivec) * total,
            __FILE__, __LINE__ );

//    int blocks = (host->max_nbrs) / control->gpu_block_size + ((host->max_nbrs) % control->gpu_block_size == 0 ? 0 : 1); 
//
//    k_init_nbrs <<< blocks, control->gpu_block_size >>>
//        ( nbrs_x, host->max_nbrs );
//    hipCheckError( );
//
//    sHipMalloc( (void **)& device->cells, sizeof(grid_cell) * total,
//            __FILE__, __LINE__ );
//    fprintf( stderr, " Device cells address --> %ld \n", device->cells );
//    sHipMalloc( (void **) &device->order,
//            sizeof(ivec) * (host->total + 1), __FILE__, __LINE__ );
//
//    local_cell.top = local_cell.mark = local_cell.str = local_cell.end = 0;
//    fprintf( stderr, "Total cells to be allocated -- > %d \n", total );
//    for (int i = 0; i < total; i++)
//    {
//        //fprintf( stderr, "Address of the local atom -> %ld  \n", &local_cell );
//
//        sHipMalloc( (void **) &local_cell.atoms, sizeof(int) * host->max_atoms,
//                __FILE__, __LINE__ );
//        //fprintf( stderr, "Allocated address of the atoms --> %ld  (%d)\n", local_cell.atoms, host->max_atoms );
//
//        sHipMalloc( (void **) &local_cell.nbrs_x, sizeof(ivec) * host->max_nbrs,
//                __FILE__, __LINE__ );
//        sHipMemcpyAsync( local_cell.nbrs_x, nbrs_x, host->max_nbrs * sizeof(ivec),
//                hipMemcpyDeviceToDevice, control->hip_streams[0], __FILE__, __LINE__ );
//        hipStreamSynchronize( control->hip_streams[0] );
//        //fprintf( stderr, "Allocated address of the nbrs_x--> %ld \n", local_cell.nbrs_x );
//
//        sHipMalloc( (void **) &local_cell.nbrs_cp, sizeof(rvec) * host->max_nbrs,
//                __FILE__, __LINE__ );
//        //fprintf( stderr, "Allocated address of the nbrs_cp--> %ld \n", local_cell.nbrs_cp );
//
//        //sHipMalloc( (void **) &local_cell.nbrs, sizeof(grid_cell *) * host->max_nbrs,
//        //      __FILE__, __LINE__ );
//        //fprintf( stderr, "Allocated address of the nbrs--> %ld \n", local_cell.nbrs );
//
//        sHipMemcpyAsync( &device->cells[i], &local_cell, sizeof(grid_cell),
//                hipMemcpyHostToDevice, control->hip_streams[0], __FILE__, __LINE__ );
//        hipStreamSynchronize( control->hip_streams[0] );
//    }
}


void Hip_Deallocate_Grid_Cell_Atoms( reax_system *system, control_params *control )
{
    int i, total;
    grid_cell local_cell;
    grid *host, *device;

    host = &system->my_grid;
    device = &system->d_my_grid;
    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    for ( i = 0; i < total; ++i )
    {
        sHipMemcpyAsync( &local_cell, &device->cells[i], sizeof(grid_cell),
                hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
        hipStreamSynchronize( control->hip_streams[0] );

        sHipFree( local_cell.atoms, __FILE__, __LINE__ );
    }
}


void Hip_Allocate_Grid_Cell_Atoms( reax_system *system, control_params *control,
        int cap )
{
    int i, total;
    grid_cell local_cell;
    grid *host, *device;

    host = &system->my_grid;
    device = &system->d_my_grid;
    total = host->ncells[0] * host->ncells[1] * host->ncells[2];

    for ( i = 0; i < total; i++ )
    {
        sHipMemcpyAsync( &local_cell, &device->cells[i], sizeof(grid_cell),
                hipMemcpyDeviceToHost, control->hip_streams[0], __FILE__, __LINE__ );
        hipStreamSynchronize( control->hip_streams[0] );
        sHipMalloc( (void **) &local_cell.atoms, sizeof(int) * cap, 
                __FILE__, __LINE__ );
        sHipMemcpyAsync( &local_cell, &device->cells[i], sizeof(grid_cell),
                hipMemcpyHostToDevice, control->hip_streams[0], __FILE__, __LINE__ );
        hipStreamSynchronize( control->hip_streams[0] );
    }
}


void Hip_Allocate_System( reax_system *system, control_params *control )
{
    /* atoms */
    sHipMalloc( (void **) &system->d_my_atoms,
            system->total_cap * sizeof(reax_atom), __FILE__, __LINE__ );
    sHipMemsetAsync( system->d_my_atoms, FALSE,
            system->total_cap * sizeof(reax_atom), 
            control->hip_streams[0], __FILE__, __LINE__ );
    hipStreamSynchronize( control->hip_streams[0] );
    sHipMalloc( (void **) &system->d_num_H_atoms, sizeof(int), __FILE__, __LINE__ );

    /* list management */
    sHipMalloc( (void **) &system->d_far_nbrs,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_max_far_nbrs,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_total_far_nbrs,
            sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_realloc_far_nbrs,
            sizeof(int), __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->d_bonds,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_max_bonds,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_total_bonds,
            sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_realloc_bonds,
            sizeof(int), __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->d_hbonds,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_max_hbonds,
            system->total_cap * sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_total_hbonds,
            sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_realloc_hbonds,
            sizeof(int), __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->d_cm_entries,
            system->local_cap * sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_max_cm_entries,
            system->local_cap * sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_total_cm_entries,
            sizeof(int), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_realloc_cm_entries,
            sizeof(int), __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->d_total_thbodies,
            sizeof(int), __FILE__, __LINE__ );

    /* simulation boxes */
    sHipMalloc( (void **) &system->d_big_box,
            sizeof(simulation_box), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_my_box,
            sizeof(simulation_box), __FILE__, __LINE__ );
    sHipMalloc( (void **) &system->d_my_ext_box,
            sizeof(simulation_box), __FILE__, __LINE__ );

    /* interaction parameters */
    sHipMalloc( (void **) &system->reax_param.d_sbp,
            system->reax_param.num_atom_types * sizeof(single_body_parameters),
            __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->reax_param.d_tbp,
            POW( system->reax_param.num_atom_types, 2.0 ) * sizeof(two_body_parameters), 
            __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->reax_param.d_thbp,
            POW( system->reax_param.num_atom_types, 3.0 ) * sizeof(three_body_header),
            __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->reax_param.d_hbp,
            POW( system->reax_param.num_atom_types, 3.0 ) * sizeof(hbond_parameters),
            __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->reax_param.d_fbp,
            POW( system->reax_param.num_atom_types, 4.0 ) * sizeof(four_body_header),
            __FILE__, __LINE__ );

    sHipMalloc( (void **) &system->reax_param.d_gp.l,
            system->reax_param.gp.n_global * sizeof(real),
            __FILE__, __LINE__ );

    system->reax_param.d_gp.n_global = 0;
    system->reax_param.d_gp.vdw_type = 0;
}


void Hip_Allocate_Simulation_Data( simulation_data *data, hipStream_t s )
{
    sHipMalloc( (void **) &data->d_simulation_data,
            sizeof(simulation_data), __FILE__, __LINE__ );
    sHipMalloc( (void **) &data->d_my_en,
            sizeof(energy_data), __FILE__, __LINE__ );

    sHipMemsetAsync( data->d_simulation_data, FALSE, sizeof(simulation_data),
            s, __FILE__, __LINE__ );

    hipStreamSynchronize( s );
}


void Hip_Allocate_Workspace_Part1( reax_system *system, control_params *control, 
        storage *workspace, int local_cap )
{
    int local_rvec;

    local_rvec = sizeof(rvec) * local_cap;

    /* integrator storage */
    if ( control->ensemble == nhNVT )
    {
        sHipMalloc( (void **) &workspace->v_const, local_rvec,
                __FILE__, __LINE__ );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        sHipMalloc( (void **) &workspace->mark, local_cap * sizeof(int),
                __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->old_mark, local_cap * sizeof(int),
                __FILE__, __LINE__ );
    }
    else
    {
        workspace->mark = NULL;
        workspace->old_mark = NULL;
    }

    if ( control->diffusion_coef )
    {
        sHipMalloc( (void **) &workspace->x_old, local_cap * sizeof(rvec),
                __FILE__, __LINE__ );
    }
    else
    {
        workspace->x_old = NULL;
    }
}


void Hip_Allocate_Workspace_Part2( reax_system *system, control_params *control, 
        storage *workspace, int total_cap )
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
    sHipMalloc( (void **) &workspace->total_bond_order, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->Deltap, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->Deltap_boc, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->dDeltap_self, total_rvec, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->Delta, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->Delta_lp, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->Delta_lp_temp, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->dDelta_lp, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->dDelta_lp_temp, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->Delta_e, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->Delta_boc, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->nlp, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->nlp_temp, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->Clp, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->vlpex, total_real, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->bond_mark, total_real, __FILE__, __LINE__ );

    /* charge matrix storage */
    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        sHipMalloc( (void **) &workspace->Hdia_inv, total_real,
                __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->Hdia_inv, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        sHipMalloc( (void **) &workspace->droptol, total_real,
                __FILE__, __LINE__ );
    }
    sHipMalloc( (void **) &workspace->b_s, total_real, __FILE__, __LINE__ );
    sHipMemsetAsync( workspace->b_s, FALSE, total_real, 
            control->hip_streams[0], __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->b_t, total_real, __FILE__, __LINE__ );
    sHipMemsetAsync( workspace->b_t, FALSE, total_real, 
            control->hip_streams[0], __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->s, total_real, __FILE__, __LINE__ );
    sHipMemsetAsync( workspace->s, FALSE, total_real, 
            control->hip_streams[0], __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->t, total_real, __FILE__, __LINE__ );
    sHipMemsetAsync( workspace->t, FALSE, total_real, 
            control->hip_streams[0], __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
    sHipMalloc( (void **) &workspace->b, total_rvec2, __FILE__, __LINE__ );
    sHipMemsetAsync( workspace->b, FALSE, total_rvec2, 
            control->hip_streams[0], __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->x, total_rvec2, __FILE__, __LINE__ );
    sHipMemsetAsync( workspace->x, FALSE, total_rvec2, 
            control->hip_streams[0], __FILE__, __LINE__ );
#endif

    switch ( control->cm_solver_type )
    {
    case GMRES_S:
    case GMRES_H_S:
        sHipMalloc( (void **) &workspace->b_prc,
                total_real, __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->b_prm,
                total_real, __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->y,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->z,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->g,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->h,
                SQR(control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->hs,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->hc,
                (control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->v,
                SQR(control->cm_solver_restart + 1) * sizeof(real), __FILE__, __LINE__ );
        break;

    case SDM_S:
        sHipMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sHipMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
#endif
        break;

    case CG_S:
        sHipMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sHipMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
#endif
        break;

    case BiCGStab_S:
        sHipMalloc( (void **) &workspace->y, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->y, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->g, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->g, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->z, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->z, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->r_hat, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r_hat, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q_hat, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q_hat, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sHipMalloc( (void **) &workspace->y2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->y2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->g2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->g2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->z2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->z2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->r_hat2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r_hat2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q_hat2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q_hat2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
#endif
        break;

    case PIPECG_S:
        sHipMalloc( (void **) &workspace->z, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->z, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->m, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->m, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->n, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->n, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->u, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->u, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->w, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->w, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sHipMalloc( (void **) &workspace->z2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->z2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->m2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->m2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->n2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->n2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->u2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->u2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->w2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->w2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
#endif
        break;

    case PIPECR_S:
        sHipMalloc( (void **) &workspace->z, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->z, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->r, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->m, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->m, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->n, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->n, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->u, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->u, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->w, total_real, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->w, FALSE, total_real, 
                control->hip_streams[0], __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
        sHipMalloc( (void **) &workspace->z2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->z2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->r2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->r2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->d2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->d2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->q2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->q2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->p2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->p2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->m2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->m2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->n2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->n2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->u2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->u2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
        sHipMalloc( (void **) &workspace->w2, total_rvec2, __FILE__, __LINE__ );
        sHipMemsetAsync( workspace->w2, FALSE, total_rvec2, 
                control->hip_streams[0], __FILE__, __LINE__ );
#endif
        break;

    default:
        fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
        exit( INVALID_INPUT );
        break;
    }

    hipStreamSynchronize( control->hip_streams[0] );

    /* force related storage */
    sHipMalloc( (void **) &workspace->f, total_rvec, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->CdDelta, total_real, __FILE__, __LINE__ );
}


void Hip_Deallocate_Workspace_Part1( control_params *control, storage *workspace )
{
    /* Nose-Hoover integrator */
    if ( control->ensemble == nhNVT )
    {
        sHipFree( workspace->v_const, __FILE__, __LINE__ );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        sHipFree( workspace->mark, __FILE__, __LINE__ );
        sHipFree( workspace->old_mark, __FILE__, __LINE__ );
    }
    else
    {
        workspace->mark = NULL;
        workspace->old_mark = NULL;
    }

    if ( control->diffusion_coef )
    {
        sHipFree( workspace->x_old, __FILE__, __LINE__ );
    }
    else
    {
        workspace->x_old = NULL;
    }
}


void Hip_Deallocate_Workspace_Part2( control_params *control, storage *workspace )
{
    /* bond order related storage  */
    sHipFree( workspace->total_bond_order, __FILE__, __LINE__ );
    sHipFree( workspace->Deltap, __FILE__, __LINE__ );
    sHipFree( workspace->Deltap_boc, __FILE__, __LINE__ );
    sHipFree( workspace->dDeltap_self, __FILE__, __LINE__ );
    sHipFree( workspace->Delta, __FILE__, __LINE__ );
    sHipFree( workspace->Delta_lp, __FILE__, __LINE__ );
    sHipFree( workspace->Delta_lp_temp, __FILE__, __LINE__ );
    sHipFree( workspace->dDelta_lp, __FILE__, __LINE__ );
    sHipFree( workspace->dDelta_lp_temp, __FILE__, __LINE__ );
    sHipFree( workspace->Delta_e, __FILE__, __LINE__ );
    sHipFree( workspace->Delta_boc, __FILE__, __LINE__ );
    sHipFree( workspace->nlp, __FILE__, __LINE__ );
    sHipFree( workspace->nlp_temp, __FILE__, __LINE__ );
    sHipFree( workspace->Clp, __FILE__, __LINE__ );
    sHipFree( workspace->vlpex, __FILE__, __LINE__ );
    sHipFree( workspace->bond_mark, __FILE__, __LINE__ );

    /* charge matrix storage */
    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        sHipFree( workspace->Hdia_inv, __FILE__, __LINE__ );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        sHipFree( workspace->droptol, __FILE__, __LINE__ );
    }
    sHipFree( workspace->b_s, __FILE__, __LINE__ );
    sHipFree( workspace->b_t, __FILE__, __LINE__ );
    sHipFree( workspace->s, __FILE__, __LINE__ );
    sHipFree( workspace->t, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
    sHipFree( workspace->b, __FILE__, __LINE__ );
    sHipFree( workspace->x, __FILE__, __LINE__ );
#endif

    switch ( control->cm_solver_type )
    {
        case GMRES_S:
        case GMRES_H_S:
            sHipFree( workspace->b_prc, __FILE__, __LINE__ );
            sHipFree( workspace->b_prm, __FILE__, __LINE__ );
            sHipFree( workspace->y, __FILE__, __LINE__ );
            sHipFree( workspace->z, __FILE__, __LINE__ );
            sHipFree( workspace->g, __FILE__, __LINE__ );
            sHipFree( workspace->h, __FILE__, __LINE__ );
            sHipFree( workspace->hs, __FILE__, __LINE__ );
            sHipFree( workspace->hc, __FILE__, __LINE__ );
            sHipFree( workspace->v, __FILE__, __LINE__ );
            break;

        case CG_S:
            sHipFree( workspace->r, __FILE__, __LINE__ );
            sHipFree( workspace->d, __FILE__, __LINE__ );
            sHipFree( workspace->q, __FILE__, __LINE__ );
            sHipFree( workspace->p, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sHipFree( workspace->r2, __FILE__, __LINE__ );
            sHipFree( workspace->d2, __FILE__, __LINE__ );
            sHipFree( workspace->q2, __FILE__, __LINE__ );
            sHipFree( workspace->p2, __FILE__, __LINE__ );
#endif
            break;

        case SDM_S:
            sHipFree( workspace->r, __FILE__, __LINE__ );
            sHipFree( workspace->d, __FILE__, __LINE__ );
            sHipFree( workspace->q, __FILE__, __LINE__ );
            sHipFree( workspace->p, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sHipFree( workspace->r2, __FILE__, __LINE__ );
            sHipFree( workspace->d2, __FILE__, __LINE__ );
            sHipFree( workspace->q2, __FILE__, __LINE__ );
            sHipFree( workspace->p2, __FILE__, __LINE__ );
#endif
            break;

        case BiCGStab_S:
            sHipFree( workspace->y, __FILE__, __LINE__ );
            sHipFree( workspace->g, __FILE__, __LINE__ );
            sHipFree( workspace->z, __FILE__, __LINE__ );
            sHipFree( workspace->r, __FILE__, __LINE__ );
            sHipFree( workspace->d, __FILE__, __LINE__ );
            sHipFree( workspace->q, __FILE__, __LINE__ );
            sHipFree( workspace->p, __FILE__, __LINE__ );
            sHipFree( workspace->r_hat, __FILE__, __LINE__ );
            sHipFree( workspace->q_hat, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sHipFree( workspace->y2, __FILE__, __LINE__ );
            sHipFree( workspace->g2, __FILE__, __LINE__ );
            sHipFree( workspace->z2, __FILE__, __LINE__ );
            sHipFree( workspace->r2, __FILE__, __LINE__ );
            sHipFree( workspace->d2, __FILE__, __LINE__ );
            sHipFree( workspace->q2, __FILE__, __LINE__ );
            sHipFree( workspace->p2, __FILE__, __LINE__ );
            sHipFree( workspace->r_hat2, __FILE__, __LINE__ );
            sHipFree( workspace->q_hat2, __FILE__, __LINE__ );
#endif
            break;

        case PIPECG_S:
            sHipFree( workspace->z, __FILE__, __LINE__ );
            sHipFree( workspace->r, __FILE__, __LINE__ );
            sHipFree( workspace->d, __FILE__, __LINE__ );
            sHipFree( workspace->q, __FILE__, __LINE__ );
            sHipFree( workspace->p, __FILE__, __LINE__ );
            sHipFree( workspace->m, __FILE__, __LINE__ );
            sHipFree( workspace->n, __FILE__, __LINE__ );
            sHipFree( workspace->u, __FILE__, __LINE__ );
            sHipFree( workspace->w, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sHipFree( workspace->z2, __FILE__, __LINE__ );
            sHipFree( workspace->r2, __FILE__, __LINE__ );
            sHipFree( workspace->d2, __FILE__, __LINE__ );
            sHipFree( workspace->q2, __FILE__, __LINE__ );
            sHipFree( workspace->p2, __FILE__, __LINE__ );
            sHipFree( workspace->m2, __FILE__, __LINE__ );
            sHipFree( workspace->n2, __FILE__, __LINE__ );
            sHipFree( workspace->u2, __FILE__, __LINE__ );
            sHipFree( workspace->w2, __FILE__, __LINE__ );
#endif
            break;

        case PIPECR_S:
            sHipFree( workspace->z, __FILE__, __LINE__ );
            sHipFree( workspace->r, __FILE__, __LINE__ );
            sHipFree( workspace->d, __FILE__, __LINE__ );
            sHipFree( workspace->q, __FILE__, __LINE__ );
            sHipFree( workspace->p, __FILE__, __LINE__ );
            sHipFree( workspace->m, __FILE__, __LINE__ );
            sHipFree( workspace->n, __FILE__, __LINE__ );
            sHipFree( workspace->u, __FILE__, __LINE__ );
            sHipFree( workspace->w, __FILE__, __LINE__ );
#if defined(DUAL_SOLVER)
            sHipFree( workspace->z2, __FILE__, __LINE__ );
            sHipFree( workspace->r2, __FILE__, __LINE__ );
            sHipFree( workspace->d2, __FILE__, __LINE__ );
            sHipFree( workspace->q2, __FILE__, __LINE__ );
            sHipFree( workspace->p2, __FILE__, __LINE__ );
            sHipFree( workspace->m2, __FILE__, __LINE__ );
            sHipFree( workspace->n2, __FILE__, __LINE__ );
            sHipFree( workspace->u2, __FILE__, __LINE__ );
            sHipFree( workspace->w2, __FILE__, __LINE__ );
#endif
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( INVALID_INPUT );
            break;
    }

    /* force related storage */
    sHipFree( workspace->f, __FILE__, __LINE__ );
    sHipFree( workspace->CdDelta, __FILE__, __LINE__ );
}


/* Allocate sparse matrix struc
 *
 * H: pointer to struct
 * n: currently utilized number of rows
 * n_max: max number of rows allocated
 * m: max number of entries allocated
 * format: sparse matrix format
 */
void Hip_Allocate_Matrix( sparse_matrix * const H, int n, int n_max, int m,
       int format, hipStream_t s )
{
    H->allocated = TRUE;
    H->n = n;
    H->n_max = n_max;
    H->m = m;
    H->format = format;

    sHipMalloc( (void **) &H->start, sizeof(int) * n_max, __FILE__, __LINE__ );
    sHipMalloc( (void **) &H->end, sizeof(int) * n_max, __FILE__, __LINE__ );
    sHipMalloc( (void **) &H->j, sizeof(int) * m, __FILE__, __LINE__ );
    sHipMalloc( (void **) &H->val, sizeof(real) * m, __FILE__, __LINE__ );
}


void Hip_Deallocate_Matrix( sparse_matrix *H )
{
    H->allocated = FALSE;
    H->n = 0;
    H->n_max = 0;
    H->m = 0;

    sHipFree( H->start, __FILE__, __LINE__ );
    sHipFree( H->end, __FILE__, __LINE__ );
    sHipFree( H->j, __FILE__, __LINE__ );
    sHipFree( H->val, __FILE__, __LINE__ );
}


void Hip_Reallocate_Part1( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    int i, j, k, renbr;
    reallocate_data *realloc;
    grid *g;

    realloc = workspace->d_workspace->realloc;
    g = &system->my_grid;
    renbr = (data->step - data->prev_steps) % control->reneighbor == 0 ? TRUE : FALSE;

    /* grid */
    if ( renbr == TRUE && realloc->gcell_atoms > -1 )
    {
        for ( i = g->native_str[0]; i < g->native_end[0]; i++ )
        {
            for ( j = g->native_str[1]; j < g->native_end[1]; j++ )
            {
                for ( k = g->native_str[2]; k < g->native_end[2]; k++ )
                {
                    sfree( g->cells[ index_grid_3d(i,j,k,g) ].atoms, __FILE__, __LINE__ );
                    g->cells[ index_grid_3d(i,j,k,g) ].atoms = (int *)
                            scalloc( realloc->gcell_atoms, sizeof(int), __FILE__, __LINE__ );
                }
            }
        }

        fprintf( stderr, "p:%d - *** Reallocating Grid Cell Atoms *** Step:%d\n", system->my_rank, data->step );

//        Hip_Deallocate_Grid_Cell_Atoms( system );
//        Hip_Allocate_Grid_Cell_Atoms( system, realloc->gcell_atoms );
        realloc->gcell_atoms = -1;
    }
}


void Hip_Reallocate_Part2( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    int nflag, Nflag, local_cap_old, total_cap_old, renbr, format;
    reallocate_data *realloc;
    sparse_matrix *H;

    realloc = workspace->d_workspace->realloc;
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
        Hip_Reallocate_System_Part1( system, control, workspace, local_cap_old );

        Hip_Deallocate_Workspace_Part1( control, workspace );
        Hip_Allocate_Workspace_Part1( system, control, workspace,
                system->local_cap );
    }

    if ( Nflag == TRUE )
    {
        Hip_Reallocate_System_Part2( system, control, workspace, total_cap_old );

        Hip_Deallocate_Workspace_Part2( control, workspace );
        Hip_Allocate_Workspace_Part2( system, control, workspace,
                system->total_cap );
    }

    /* far neighbors */
    if ( renbr == TRUE && (Nflag == TRUE || realloc->far_nbrs == TRUE) )
    {
        Hip_Reallocate_List( lists[FAR_NBRS], system->total_cap,
                system->total_far_nbrs, TYP_FAR_NEIGHBOR );
        Hip_Init_Neighbor_Indices( system, control, lists[FAR_NBRS] );
        realloc->far_nbrs = FALSE;
    }

    /* charge matrix */
    if ( nflag == TRUE || realloc->cm == TRUE )
    {
        format = H->format;

        Hip_Deallocate_Matrix( H );
        Hip_Allocate_Matrix( H, system->n, system->local_cap,
                system->total_cm_entries, format, control->hip_streams[0] );

        realloc->cm = FALSE;
    }

    /* bonds list */
    if ( Nflag == TRUE || realloc->bonds == TRUE )
    {
        Hip_Reallocate_List( lists[BONDS], system->total_cap,
                system->total_bonds, TYP_BOND );

        realloc->bonds = FALSE;
    }

    /* hydrogen bonds list */
    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0
            && (Nflag == TRUE || realloc->hbonds == TRUE) )
    {
        Hip_Reallocate_List( lists[HBONDS], system->total_cap,
                system->total_hbonds, TYP_HBOND );

        realloc->hbonds = FALSE;
    }

    /* 3-body list */
    if ( Nflag == TRUE || realloc->thbody == TRUE )
    {
        Hip_Reallocate_List( lists[THREE_BODIES], system->total_thbodies_indices,
                system->total_thbodies, TYP_THREE_BODY );

        realloc->thbody = FALSE;
    }
}
