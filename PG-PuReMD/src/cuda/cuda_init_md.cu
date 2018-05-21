
#include "cuda_init_md.h"

#include "cuda_allocate.h"
#include "cuda_list.h"
#include "cuda_copy.h"
#include "cuda_forces.h"
#include "cuda_integrate.h"
#include "cuda_neighbors.h"
#include "cuda_reset_tools.h"
#include "cuda_system_props.h"
#include "cuda_utils.h"
#if defined(DEBUG)
  #include "cuda_validation.h"
#endif

#if defined(PURE_REAX)
  #include "../box.h"
  #include "../comm_tools.h"
  #include "../grid.h"
  #include "../init_md.h"
  #include "../io_tools.h"
#ifdef __cplusplus
extern "C" {
#endif
  #include "../lookup.h"
#ifdef __cplusplus
}
#endif
  #include "../random.h"
  #include "../reset_tools.h"
  #include "../tool_box.h"
  #include "../vector.h"
#elif defined(LAMMPS_REAX)
  #include "../reax_box.h"
  #include "../reax_comm_tools.h"
  #include "../reax_grid.h"
  #include "../reax_init_md.h"
  #include "../reax_io_tools.h"
  #include "../reax_list.h"
  #include "../reax_lookup.h"
  #include "../reax_random.h"
  #include "../reax_reset_tools.h"
  #include "../reax_tool_box.h"
  #include "../reax_vector.h"
#endif


void Cuda_Init_ScratchArea( )
{
    cuda_malloc( (void **)&scratch, DEVICE_SCRATCH_SIZE, TRUE, "device:scratch" );

    host_scratch = (void *) smalloc( HOST_SCRATCH_SIZE, "host:scratch" );
}


int Cuda_Init_System( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        mpi_datatypes *mpi_data, char *msg )
{
    int i;
    int nrecv[MAX_NBRS];

    Setup_New_Grid( system, control, MPI_COMM_WORLD );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d GRID:\n", system->my_rank );
    Print_Grid( &(system->my_grid), stderr );
#endif

    Bin_My_Atoms( system, &(workspace->realloc) );
    Reorder_My_Atoms( system, workspace );

    /* estimate N and total capacity */
    for ( i = 0; i < MAX_NBRS; ++i )
    {
        nrecv[i] = 0;
    }

    MPI_Barrier( MPI_COMM_WORLD );
    system->max_recved = 0;
    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
            Estimate_Boundary_Atoms, Unpack_Estimate_Message, TRUE );
    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );
    Bin_Boundary_Atoms( system );

    /* Sync atoms here to continue the computation */
    dev_alloc_system( system );
    Sync_System( system );

    /* estimate numH and Hcap */
    Cuda_Reset_Atoms( system, control );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d local_cap=%d\n",
             system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d total_cap=%d\n",
             system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d H_cap=%d\n",
             system->my_rank, system->numH, system->Hcap );
#endif

    Cuda_Compute_Total_Mass( system, data, mpi_data->comm_mesh3D );

    Cuda_Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );

//    if( Reposition_Atoms( system, control, data, mpi_data, msg ) == FAILURE )
//    {
//        return FAILURE;
//    }

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
    {
        Cuda_Generate_Initial_Velocities( system, control->T_init );
    }

    Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

    return SUCCESS;
}


void Cuda_Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data )
{
    dev_alloc_simulation_data( data );

    Reset_Simulation_Data( data );

    if ( !control->restart )
    {
        data->step = data->prev_steps = 0;
    }

    switch ( control->ensemble )
    {
    case NVE:
        data->N_f = 3 * system->bigN;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_NVE;
        control->virial = 0;
        break;

    case bNVT:
        data->N_f = 3 * system->bigN + 1;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NVT;
        control->virial = 0;
        break;

    case nhNVT:
        fprintf( stderr, "[WARNING] Nose-Hoover NVT is still under testing.\n" );
        data->N_f = 3 * system->bigN + 1;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Nose_Hoover_NVT_Klein;
        control->virial = 0;
        if ( !control->restart || (control->restart && control->random_vel) )
        {
            data->therm.G_xi = control->Tau_T *
                               (2.0 * data->sys_en.e_kin - data->N_f * K_B * control->T );
            data->therm.v_xi = data->therm.G_xi * control->dt;
            data->therm.v_xi_old = 0;
            data->therm.xi = 0;
        }
        break;

    case sNPT: /* Semi-Isotropic NPT */
        data->N_f = 3 * system->bigN + 4;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    case iNPT: /* Isotropic NPT */
        data->N_f = 3 * system->bigN + 2;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    case NPT: /* Anisotropic NPT */
        data->N_f = 3 * system->bigN + 9;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;

        fprintf( stderr, "p%d: init_simulation_data: option not yet implemented\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
        break;

    default:
        fprintf( stderr, "p%d: init_simulation_data: ensemble not recognized\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
    }

    /* initialize the timer(s) */
    MPI_Barrier( MPI_COMM_WORLD );
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.start = Get_Time( );

#if defined(LOG_PERFORMANCE)
        Reset_Timing( &data->timing );
#endif
    }

#if defined(DEBUG)
    fprintf( stderr, "data->N_f: %8.3f\n", data->N_f );
#endif
}


void Cuda_Init_Workspace( reax_system *system, control_params *control,
        storage *workspace )
{
    dev_alloc_workspace( system, control, dev_workspace,
            system->local_cap, system->total_cap );

    memset( &(workspace->realloc), 0, sizeof(reallocate_data) );
    Cuda_Reset_Workspace( system, workspace );

    /* Initialize the Taper function */
    Init_Taper( control, dev_workspace );
}


void Cuda_Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    Cuda_Estimate_Neighbors( system );

    Dev_Make_List( system->total_cap, system->total_far_nbrs,
            TYP_FAR_NEIGHBOR, dev_lists[FAR_NBRS] );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated far_nbrs: num_far=%d, space=%dMB\n",
            system->my_rank, system->total_far_nbrs,
            (int)(system->total_far_nbrs * sizeof(far_neighbor_data) / (1024 * 1024)) );
    fprintf( stderr, "N: %d and total_cap: %d \n", system->N, system->total_cap );
#endif

    Cuda_Init_Neighbor_Indices( system );

    Cuda_Generate_Neighbor_Lists( system, data, workspace, dev_lists );

    /* estimate storage for bonds, hbonds, and sparse matrix */
    Cuda_Estimate_Storages( system, control, dev_lists,
            TRUE, TRUE, TRUE, data->step );

    dev_alloc_matrix( &(dev_workspace->H), system->total_cap, system->total_cm_entries );
    Cuda_Init_Sparse_Matrix_Indices( system, &(dev_workspace->H) );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p:%d - allocated H matrix: max_entries: %d, space=%dMB\n",
            system->my_rank, system->total_cm_entries,
            (int)(system->total_cm_entries * sizeof(sparse_matrix_entry) / (1024 * 1024)) );
#endif

    if ( control->hbond_cut > 0.0 &&  system->numH > 0 )
    {
        Dev_Make_List( system->total_cap, system->total_hbonds, TYP_HBOND, dev_lists[HBONDS] );
        Cuda_Init_HBond_Indices( system );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: allocated hbonds: total_hbonds=%d, space=%dMB\n",
                system->my_rank, system->total_hbonds,
                (int)(system->total_hbonds * sizeof(hbond_data) / (1024 * 1024)) );
#endif
    }

    /* bonds list */
    Dev_Make_List( system->total_cap, system->total_bonds, TYP_BOND, dev_lists[BONDS] );
    Cuda_Init_Bond_Indices( system );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated bonds: total_bonds=%d, space=%dMB\n",
            system->my_rank, total_bonds,
            (int)(total_bonds * sizeof(bond_data) / (1024 * 1024)) );
#endif

    /* 3bodies list: since a more accurate estimate of the num.
     * three body interactions requires that bond orders have
     * been computed, delay estimation until computation */
}


void Cuda_Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    char msg[MAX_STR];

    Cuda_Init_ScratchArea( );

    Init_MPI_Datatypes( system, workspace, mpi_data );

    if ( Cuda_Init_System( system, control, data, workspace, mpi_data, msg ) == FAILURE )
    {
        fprintf( stderr, "[ERROR] p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "[ERROR] p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

    dev_alloc_grid( system );
    Sync_Grid( &system->my_grid, &system->d_my_grid );

    //validate_grid( system );

    Cuda_Init_Simulation_Data( system, control, data );

    Cuda_Init_Workspace( system, control, workspace );

    dev_alloc_control( control );

    Cuda_Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Output_Files( system, control, out_control, mpi_data );

    /* Lookup Tables */
    if ( control->tabulate )
    {
        Init_Lookup_Tables( system, control, dev_workspace->Tap, mpi_data );
    }
}
