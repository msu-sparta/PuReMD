
#include "gpu_init_md.h"

#include "gpu_allocate.h"
#include "gpu_list.h"
#include "gpu_copy.h"
#include "gpu_environment.h"
#include "gpu_forces.h"
#include "gpu_integrate.h"
#include "gpu_neighbors.h"
#include "gpu_reset_tools.h"
#include "gpu_system_props.h"
#include "gpu_utils.h"

#include "../box.h"
#include "../comm_tools.h"
#include "../grid.h"
#include "../init_md.h"
#include "../io_tools.h"
#include "../lookup.h"
#include "../random.h"
#include "../reset_tools.h"
#include "../tool_box.h"
#include "../vector.h"


static void GPU_Init_System( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, mpi_datatypes * const mpi_data )
{
    Setup_New_Grid( system, control, MPI_COMM_WORLD );

    /* since all processors read in all atoms and select their local atoms
     * intially, no local atoms comm needed and just bin local atoms */
    Bin_My_Atoms( system, workspace );
    Reorder_My_Atoms( system );

    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type,
            &Count_Boundary_Atoms, &Sort_Boundary_Atoms,
            &Unpack_Exchange_Message, TRUE );

    system->local_cap = MAX( (int) CEIL( system->n * SAFE_ZONE ), MIN_CAP );
    system->total_cap = MAX( (int) CEIL( system->N * SAFE_ZONE ), MIN_CAP );

    system->total_far_nbrs = 0;
    system->total_bonds = 0;
    system->total_hbonds = 0;
    system->total_cm_entries = 0;
    system->total_thbodies = 0;

    Bin_Boundary_Atoms( system );

    GPU_Init_Block_Sizes( system, control );

    GPU_Allocate_System( system, control );
    GPU_Copy_System_Host_to_Device( system, control );

    GPU_Reset_Atoms_HBond_Indices( system, control, workspace );

    GPU_Compute_Total_Mass( system, control, workspace,
            data, mpi_data->comm_mesh3D );

    GPU_Compute_Center_of_Mass( system, control, workspace,
            data, mpi_data, mpi_data->comm_mesh3D );

//    Reposition_Atoms( system, control, data, mpi_data );

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
    {
        GPU_Generate_Initial_Velocities( system, control, control->T_init );
    }

    GPU_Compute_Kinetic_Energy( system, control, workspace,
            data, mpi_data->comm_mesh3D );
}


void GPU_Init_Simulation_Data( reax_system * const system, control_params * const control,
        simulation_data * const data )
{
    data->my_en = (real *) smalloc_pinned( sizeof(real) * E_N, __FILE__, __LINE__ );
    data->sys_en = (real *) smalloc( sizeof(real) * E_N, __FILE__, __LINE__ );

    GPU_Allocate_Simulation_Data( data, control->gpu_streams[0] );

    Reset_Simulation_Data( data );
    Reset_Timing( &data->timing );

    if ( !control->restart )
    {
        data->step = 0;
        data->prev_steps = 0;
    }

    switch ( control->ensemble )
    {
    case NVE:
        data->N_f = 3 * system->bigN;
        control->GPU_Evolve = &GPU_Velocity_Verlet_NVE;
        control->virial = 0;
        break;

    case bNVT:
        data->N_f = 3 * system->bigN + 1;
        control->GPU_Evolve = &GPU_Velocity_Verlet_Berendsen_NVT;
        control->virial = 0;
        break;

    case nhNVT:
        fprintf( stderr, "[WARNING] Nose-Hoover NVT is still under testing.\n" );
        data->N_f = 3 * system->bigN + 1;
        control->GPU_Evolve = &GPU_Velocity_Verlet_Nose_Hoover_NVT_Klein;
        control->virial = 0;
        if ( !control->restart || (control->restart && control->random_vel) )
        {
            data->therm.G_xi = control->Tau_T
                * (2.0 * data->sys_en[E_KIN] - data->N_f * K_B * control->T );
            data->therm.v_xi = data->therm.G_xi * control->dt;
            data->therm.v_xi_old = 0;
            data->therm.xi = 0;
        }
        break;

    /* Semi-Isotropic NPT */
    case sNPT:
        data->N_f = 3 * system->bigN + 4;
        control->GPU_Evolve = &GPU_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Isotropic NPT */
    case iNPT:
        data->N_f = 3 * system->bigN + 2;
        control->GPU_Evolve = &GPU_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Anisotropic NPT */
    case NPT:
        data->N_f = 3 * system->bigN + 9;
        control->GPU_Evolve = &GPU_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;

        fprintf( stderr, "[ERROR] Anisotropic NPT ensemble not yet implemented\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        break;

    default:
        fprintf( stderr, "[ERROR] p%d: Init_Simulation_Data: ensemble not recognized\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        break;
    }
}


void GPU_Init_Workspace( reax_system const * const system, control_params * const control,
        storage * const workspace, mpi_datatypes * const mpi_data )
{
    GPU_Allocate_Workspace_Part1( control, workspace->d_workspace, system->local_cap );
    GPU_Allocate_Workspace_Part2( control, workspace->d_workspace, system->total_cap );

    /* one-off allocations (not to be rerun after reneighboring)
     * and not needed earlier during input file parsing (in PreAllocate_Space) */
    workspace->tap_coef = (real *) smalloc( sizeof(real) * TAPER_COEF_SIZE,
            __FILE__, __LINE__ );
    workspace->dtap_coef = (real *) smalloc( sizeof(real) * DTAPER_COEF_SIZE,
            __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->d_workspace->tap_coef, sizeof(real) * TAPER_COEF_SIZE,
            __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->d_workspace->dtap_coef, sizeof(real) * DTAPER_COEF_SIZE,
            __FILE__, __LINE__ );

    workspace->realloc[RE_FAR_NBRS] = FALSE;
    workspace->realloc[RE_CM] = FALSE;
    workspace->realloc[RE_BONDS] = FALSE;
    workspace->realloc[RE_HBONDS] = FALSE;
    workspace->realloc[RE_THBODY] = FALSE;
    workspace->realloc[RE_GCELL_ATOMS] = 0;

    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        workspace->H.allocated = FALSE;
        workspace->H_spar_patt.allocated = FALSE;
        workspace->H_spar_patt_full.allocated = FALSE;
        workspace->H_app_inv.allocated = FALSE;
        workspace->d_workspace->H_spar_patt.allocated = FALSE;
        workspace->d_workspace->H_spar_patt_full.allocated = FALSE;
        workspace->d_workspace->H_app_inv.allocated = FALSE;
    }

    GPU_Reset_Workspace( system, control, workspace );

    Init_Taper( control, workspace, mpi_data );

    sHipMemcpyAsync( workspace->d_workspace->tap_coef, workspace->tap_coef,
            sizeof(real) * TAPER_COEF_SIZE, hipMemcpyHostToDevice,
            control->gpu_streams[0], __FILE__, __LINE__ );
    sHipMemcpyAsync( workspace->d_workspace->dtap_coef, workspace->dtap_coef,
            sizeof(real) * DTAPER_COEF_SIZE, hipMemcpyHostToDevice,
            control->gpu_streams[0], __FILE__, __LINE__ );
}


void GPU_Init_Lists( reax_system * const system, control_params const * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        mpi_datatypes * const mpi_data )
{
    GPU_Estimate_Num_Neighbors( system, control, data );

    GPU_Make_List( system->total_cap, system->total_far_nbrs,
            TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );
    GPU_Init_Neighbor_Indices( system, control, lists[FAR_NBRS] );

    GPU_Generate_Neighbor_Lists( system, control, data, workspace, lists );

    /* first call to GPU_Estimate_Storages requires
     * setting these manually before allocation */
    workspace->d_workspace->H.n = system->n;
    workspace->d_workspace->H.n_max = system->local_cap;
    workspace->d_workspace->H.format = SYM_FULL_MATRIX;

    /* estimate storage for bonds, hbonds, and sparse matrix */
    GPU_Estimate_Storages( system, control, data, workspace, lists,
            TRUE, TRUE, TRUE, data->step - data->prev_steps );

    GPU_Allocate_Matrix( &workspace->d_workspace->H, system->n,
            system->local_cap, system->total_cm_entries, SYM_FULL_MATRIX,
            control->gpu_streams[0] );
    GPU_Init_Sparse_Matrix_Indices( system, &workspace->d_workspace->H,
            control->gpu_block_size, control->gpu_streams[0] );

    GPU_Make_List( system->total_cap, system->total_bonds,
            TYP_BOND, lists[BONDS] );
    GPU_Init_Bond_Indices( system, lists[BONDS], control->gpu_block_size,
            control->gpu_streams[0] );

    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
    {
        GPU_Make_List( system->total_cap, system->total_hbonds,
                TYP_HBOND, lists[HBONDS] );
        GPU_Init_HBond_Indices( system, workspace, lists[HBONDS], control->gpu_block_size,
                control->gpu_streams[0] );
    }

    /* 3bodies list: since a more accurate estimate of the num.
     * three body interactions requires that bond orders have
     * been computed, delay estimation until computation */
}


extern "C" void GPU_Initialize( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
    int i;

#if defined(_OPENMP)
    #pragma omp parallel default(none) shared(control)
    {
        #pragma omp single
        {
            if ( control->num_threads_set == FALSE )
            {
                /* set using OMP_NUM_THREADS environment variable */
                control->num_threads = omp_get_num_threads( );
                control->num_threads_set = TRUE;
            }
        }
    }

    omp_set_num_threads( control->num_threads );
#else
    control->num_threads = 1;
#endif

    Init_MPI_Datatypes( system, workspace, mpi_data );

#if defined(GPU_DEVICE_PACK)
    if ( MPIX_Query_rocm_support( ) != 1 )
    {
        fprintf( stderr, "[ERROR] ROCm device-side MPI buffer packing/unpacking enabled "
                "but no ROCm-aware support detected. Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }

    mpi_data->d_in1_buffer = NULL;
    mpi_data->d_in1_buffer_size = 0;
    mpi_data->d_in2_buffer = NULL;
    mpi_data->d_in2_buffer_size = 0;

    for ( i = 0; i < MAX_NBRS; ++i )
    {
        mpi_data->d_out_buffers[i].cnt = 0;
        mpi_data->d_out_buffers[i].index = NULL;
        mpi_data->d_out_buffers[i].index_size = 0;
        mpi_data->d_out_buffers[i].out_atoms = NULL;
        mpi_data->d_out_buffers[i].out_atoms_size = 0;
    }
#endif

    GPU_Init_Simulation_Data( system, control, data );

    /* scratch space - set before GPU_Init_Workspace
     * as GPU_Init_System utilizes these variables */
    for ( i = 0; i < MAX_GPU_STREAMS; ++i )
    {
        workspace->scratch[i] = NULL;
        workspace->scratch_size[i] = 0;
    }
    for ( i = 0; i < MAX_GPU_STREAMS; ++i )
    {
        workspace->d_workspace->scratch[i] = NULL;
        workspace->d_workspace->scratch_size[i] = 0;
    }

    /* early allocation before GPU_Init_Workspace for Bin_My_Atoms inside GPU_Init_System */
    workspace->realloc = (int *) smalloc( sizeof(int) * RE_N, __FILE__, __LINE__ );
    sHipMalloc( (void **) &workspace->d_workspace->realloc, sizeof(int) * RE_N,
            __FILE__, __LINE__ );

    GPU_Init_System( system, control, data, workspace, mpi_data );
    /* reset for step 0 */
    Reset_Simulation_Data( data );

    GPU_Allocate_Grid( system, control );
    GPU_Copy_Grid_Host_to_Device( control, &system->my_grid, &system->d_my_grid );

    GPU_Init_Workspace( system, control, workspace, mpi_data );

    GPU_Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Output_Files( system, control, out_control, mpi_data );

    if ( control->tabulate > 0 )
    {
        Make_LR_Lookup_Table( system, control, workspace, mpi_data );
    }

#if defined(DEBUG_FOCUS)
    GPU_Print_Mem_Usage( data );
#endif
}
