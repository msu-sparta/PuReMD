
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


static void Cuda_Init_System( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        mpi_datatypes *mpi_data )
{
    Setup_New_Grid( system, control, MPI_COMM_WORLD );

    Bin_My_Atoms( system, workspace );
    Reorder_My_Atoms( system, workspace );

    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type,
            &Count_Boundary_Atoms, &Sort_Boundary_Atoms,
            &Unpack_Exchange_Message, TRUE );
    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );
    Bin_Boundary_Atoms( system );

    /* Sync atoms here to continue the computation */
    Cuda_Allocate_System( system );
    Sync_System( system );

    /* estimate numH */
    Cuda_Reset_Atoms( system, control, workspace );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d local_cap=%d\n",
             system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d total_cap=%d\n",
             system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d\n",
             system->my_rank, system->numH );
#endif

    Cuda_Compute_Total_Mass( system, control, workspace,
            data, mpi_data->comm_mesh3D );

    Cuda_Compute_Center_of_Mass( system, control, workspace,
            data, mpi_data, mpi_data->comm_mesh3D );

//    Reposition_Atoms( system, control, data, mpi_data );

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
    {
        Cuda_Generate_Initial_Velocities( system, control->T_init );
    }

    Cuda_Compute_Kinetic_Energy( system, control, workspace,
            data, mpi_data->comm_mesh3D );
}


void Cuda_Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data )
{
    Cuda_Allocate_Simulation_Data( data );

    Reset_Simulation_Data( data );
    Reset_Timing( &data->timing );

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

    /* Semi-Isotropic NPT */
    case sNPT:
        data->N_f = 3 * system->bigN + 4;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Isotropic NPT */
    case iNPT:
        data->N_f = 3 * system->bigN + 2;
        control->Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Anisotropic NPT */
    case NPT:
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

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "data->N_f: %8.3f\n", data->N_f );
#endif
}


void Cuda_Init_Workspace( reax_system *system, control_params *control,
        storage *workspace, mpi_datatypes *mpi_data )
{
    Cuda_Allocate_Workspace( system, control, workspace->d_workspace,
            system->local_cap, system->total_cap );

    memset( &workspace->realloc, 0, sizeof(reallocate_data) );
    Cuda_Reset_Workspace( system, workspace );

    Init_Taper( control, workspace->d_workspace, mpi_data );
}


void Cuda_Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    Cuda_Estimate_Neighbors( system );

    Cuda_Make_List( system->total_cap, system->total_far_nbrs,
            TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );

    Cuda_Init_Neighbor_Indices( system, lists );

    Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );

    /* estimate storage for bonds, hbonds, and sparse matrix */
    Cuda_Estimate_Storages( system, control, lists,
            TRUE, TRUE, TRUE, data->step );

    Cuda_Allocate_Matrix( &workspace->d_workspace->H, system->n,
            system->local_cap, system->total_cm_entries, SYM_FULL_MATRIX );
    Cuda_Init_Sparse_Matrix_Indices( system, &workspace->d_workspace->H );

    if ( control->hbond_cut > 0.0 && system->numH > 0 )
    {
        Cuda_Make_List( system->total_cap, system->total_hbonds, TYP_HBOND, lists[HBONDS] );
        Cuda_Init_HBond_Indices( system, workspace, lists );
    }

    /* bonds list */
    Cuda_Make_List( system->total_cap, system->total_bonds, TYP_BOND, lists[BONDS] );
    Cuda_Init_Bond_Indices( system, lists );

    /* 3bodies list: since a more accurate estimate of the num.
     * three body interactions requires that bond orders have
     * been computed, delay estimation until computation */
}


void Cuda_Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    Init_MPI_Datatypes( system, workspace, mpi_data );

    Cuda_Init_System( system, control, data, workspace, mpi_data );

    Cuda_Allocate_Grid( system );
    Sync_Grid( &system->my_grid, &system->d_my_grid );

    Cuda_Init_Simulation_Data( system, control, data );

    Cuda_Init_Workspace( system, control, workspace, mpi_data );

    Cuda_Allocate_Control( control );

    Cuda_Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Output_Files( system, control, out_control, mpi_data );

    /* Lookup Tables */
    if ( control->tabulate )
    {
        Init_Lookup_Tables( system, control, workspace->d_workspace, mpi_data );
    }

    Cuda_Init_Block_Sizes( system, control );

#if defined(DEBUG_FOCUS)
    Cuda_Print_Mem_Usage( );
#endif
}
