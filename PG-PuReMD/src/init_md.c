/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#include "reax_types.h"

#include <stddef.h>

#if defined(PURE_REAX)
  #include "init_md.h"
  #include "allocate.h"
  #include "box.h"
  #include "comm_tools.h"
  #include "forces.h"
  #include "grid.h"
  #include "integrate.h"
  #include "io_tools.h"
  #include "list.h"
  #include "lookup.h"
  #include "neighbors.h"
  #include "random.h"
  #include "reset_tools.h"
  #include "system_props.h"
  #include "tool_box.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_init_md.h"
  #include "reax_allocate.h"
  #include "reax_forces.h"
  #include "reax_io_tools.h"
  #include "reax_list.h"
  #include "reax_lookup.h"
  #include "reax_reset_tools.h"
  #include "reax_system_props.h"
  #include "reax_tool_box.h"
  #include "reax_vector.h"
#endif


#if defined(PURE_REAX)
/************************ initialize system ************************/
int Reposition_Atoms( reax_system *system, control_params *control,
        simulation_data *data, mpi_datatypes *mpi_data, char *msg )
{
    int i;
    rvec dx;

    /* reposition atoms */
    /* fit atoms to periodic box */
    if ( control->reposition_atoms == 0 )
    {
        rvec_MakeZero( dx );
    }
    /* put center of mass to center */
    else if ( control->reposition_atoms == 1 )
    {
        rvec_Scale( dx, 0.5, system->big_box.box_norms );
        rvec_ScaledAdd( dx, -1., data->xcm );
    }
    /* put center of mass to origin */
    else if ( control->reposition_atoms == 2 )
    {
        rvec_Scale( dx, -1., data->xcm );
    }
    else
    {
        strcpy( msg, "[ERROR] reposition_atoms: invalid option" );
        return FAILURE;
    }

    for ( i = 0; i < system->n; ++i )
    {
        // Inc_on_T3_Gen( system->my_atoms[i].x, dx, &(system->big_box) );
        rvec_Add( system->my_atoms[i].x, dx );
    }

    return SUCCESS;
}



void Generate_Initial_Velocities( reax_system *system, real T )
{
    int i;
    real m, scale, norm;

    if ( T <= 0.1 )
    {
        for ( i = 0; i < system->n; i++ )
        {
            rvec_MakeZero( system->my_atoms[i].v );
        }
    }
    else
    {
        Randomize( );

        for ( i = 0; i < system->n; i++ )
        {
            rvec_Random( system->my_atoms[i].v );

            norm = rvec_Norm_Sqr( system->my_atoms[i].v );
            m = system->reax_param.sbp[ system->my_atoms[i].type ].mass;
            scale = SQRT( m * norm / (3.0 * K_B * T) );

            rvec_Scale( system->my_atoms[i].v, 1.0 / scale, system->my_atoms[i].v );
        }
    }
}


void Init_System( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        mpi_datatypes *mpi_data )
{
    int i;
    reax_atom *atom;
    int nrecv[MAX_NBRS];

    Setup_New_Grid( system, control, MPI_COMM_WORLD );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d GRID:\n", system->my_rank );
    Print_Grid( &system->my_grid, stderr );
#endif

    Bin_My_Atoms( system, &workspace->realloc );
    Reorder_My_Atoms( system, workspace );

    /* estimate N and total capacity */
    for ( i = 0; i < MAX_NBRS; ++i )
    {
        nrecv[i] = 0;
    }
    MPI_Barrier( MPI_COMM_WORLD );
    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
            Estimate_Boundary_Atoms, Unpack_Estimate_Message, TRUE );

    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );
    Bin_Boundary_Atoms( system );

    /* estimate numH and Hcap */
    system->numH = 0;
    if ( control->hbond_cut > 0.0 )
    {
        for ( i = 0; i < system->N; ++i )
        {
            atom = &system->my_atoms[i];

            if ( system->reax_param.sbp[ atom->type ].p_hbond == H_ATOM )
            {
                atom->Hindex = system->numH++;
            }
            else
            {
                atom->Hindex = -1;
            }
        }
    }
    //Tried fix
    //system->Hcap = MAX( system->numH * SAFER_ZONE, MIN_CAP );
    system->Hcap = MAX( system->n * SAFER_ZONE, MIN_CAP );

    /* list management */
    system->far_nbrs = (int *) smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->far_nbrs" );
    system->max_far_nbrs = (int *) smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->max_far_nbrs" );

    system->bonds = (int *) smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->bonds" );
    system->max_bonds = (int *) smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->max_bonds" );

    system->hbonds = (int *) smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->hbonds" );
    system->max_hbonds = (int *) smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->max_hbonds" );

    system->cm_entries = (int *) smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::system->cm_entries" );
    system->max_cm_entries = (int *) smalloc( sizeof(int) * system->total_cap,
            "ReAllocate_System::max_cm_entries->max_hbonds" );
    
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d local_cap=%d\n",
             system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d total_cap=%d\n",
             system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d H_cap=%d\n",
             system->my_rank, system->numH, system->Hcap );
#endif

    Compute_Total_Mass( system, data, mpi_data->comm_mesh3D );

    Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );

//    if( Reposition_Atoms( system, control, data, mpi_data ) == FAILURE )
//    {
//        return FAILURE;
//    }

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
    {
        Generate_Initial_Velocities( system, control->T_init );
    }

    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
}


/************************ initialize simulation data ************************/
void Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data )
{
    Reset_Simulation_Data( data );

    if ( !control->restart )
    {
        data->step = data->prev_steps = 0;
    }

    switch ( control->ensemble )
    {
    case NVE:
        data->N_f = 3 * system->bigN;
        Evolve = Velocity_Verlet_NVE;
        control->virial = 0;
        break;

    case bNVT:
        data->N_f = 3 * system->bigN + 1;
        Evolve = Velocity_Verlet_Berendsen_NVT;
        control->virial = 0;
        break;

    case nhNVT:
        fprintf( stderr, "[WARNING] Nose-Hoover NVT is still under testing.\n" );
        data->N_f = 3 * system->bigN + 1;
        Evolve = Velocity_Verlet_Nose_Hoover_NVT_Klein;
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
        Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Isotropic NPT */
    case iNPT:
        data->N_f = 3 * system->bigN + 2;
        Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Anisotropic NPT */
    case NPT:
        fprintf( stderr, "[ERROR] p%d: init_simulation_data: option not yet implemented\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );

        data->N_f = 3 * system->bigN + 9;
        Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        /*if( !control->restart ) {
          data->therm.G_xi = control->Tau_T *
          (2.0 * data->my_en.e_Kin - data->N_f * K_B * control->T );
          data->therm.v_xi = data->therm.G_xi * control->dt;
          data->iso_bar.eps = (1.0 / 3.0) * LOG(system->box.volume);
          data->inv_W = 1.0 /
          ( data->N_f * K_B * control->T * SQR(control->Tau_P) );
          Compute_Pressure( system, control, data, out_control );
          }*/
        break;

    default:
        fprintf( stderr, "[ERROR] p%d: init_simulation_data: ensemble not recognized\n",
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


#elif defined(LAMMPS_REAX)
void Init_System( reax_system *system )
{
    system->big_box.V = 0;
    system->big_box.box_norms[0] = 0;
    system->big_box.box_norms[1] = 0;
    system->big_box.box_norms[2] = 0;

    system->local_cap = (int)(system->n * SAFE_ZONE);
    system->total_cap = (int)(system->N * SAFE_ZONE);

    system->far_nbrs = NULL;
    system->max_far_nbrs = NULL;
    system->bonds = NULL;
    system->max_bonds = NULL;
    system->hbonds = NULL;
    system->max_hbonds = NULL;
    system->cm_entries = NULL;
    system->max_cm_entries = NULL;

#if defined(DEBUG)
    fprintf( stderr, "p%d: local_cap=%d total_cap=%d\n",
             system->my_rank, system->local_cap, system->total_cap );
#endif

    ReAllocate_System( system, system->local_cap, system->total_cap );
}


void Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data )
{
    Reset_Simulation_Data( data );

#if defined(LOG_PERFORMANCE)
    Reset_Timing( &data->timing );
#endif

    //if( !control->restart )
    data->step = data->prev_steps = 0;
}
#endif


/************************ initialize workspace ************************/
/* Initialize Taper params */
void Init_Taper( control_params *control,  storage *workspace )
{
    real d1, d7;
    real swa, swa2, swa3;
    real swb, swb2, swb3;

    swa = control->nonb_low;
    swb = control->nonb_cut;

    if ( FABS( swa ) > 0.01 )
    {
        fprintf( stderr, "[WARNING] non-zero lower Taper-radius cutoff in force field parameters\n" );
    }

    if ( swb < 0 )
    {
        fprintf( stderr, "[ERROR] negative upper Taper-radius cutoff in force field parameters\n" );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
    }
    else if ( swb < 5 )
    {
        fprintf( stderr, "[WARNING] very low Taper-radius cutoff in force field parameters (%f)\n", swb );
    }

    d1 = swb - swa;
    d7 = POW( d1, 7.0 );
    swa2 = SQR( swa );
    swa3 = CUBE( swa );
    swb2 = SQR( swb );
    swb3 = CUBE( swb );

    workspace->Tap[7] =  20.0 / d7;
    workspace->Tap[6] = -70.0 * (swa + swb) / d7;
    workspace->Tap[5] =  84.0 * (swa2 + 3.0 * swa * swb + swb2) / d7;
    workspace->Tap[4] = -35.0 * (swa3 + 9.0 * swa2 * swb + 9.0 * swa * swb2 + swb3 ) / d7;
    workspace->Tap[3] = 140.0 * (swa3 * swb + 3.0 * swa2 * swb2 + swa * swb3 ) / d7;
    workspace->Tap[2] = -210.0 * (swa3 * swb2 + swa2 * swb3) / d7;
    workspace->Tap[1] = 140.0 * swa3 * swb3 / d7;
    workspace->Tap[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2 +
            7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb ) / d7;
}


void Init_Workspace( reax_system *system, control_params *control,
        storage *workspace )
{
    Allocate_Workspace( system, control, workspace, system->local_cap,
            system->total_cap );

    workspace->realloc.far_nbrs = FALSE;
    workspace->realloc.cm = FALSE;
    workspace->realloc.hbonds = FALSE;
    workspace->realloc.bonds = FALSE;
    workspace->realloc.thbody = FALSE;
    workspace->realloc.gcell_atoms = 0;

    Reset_Workspace( system, workspace );

    Init_Taper( control, workspace );
}


/************** setup communication data structures  **************/
void Init_MPI_Datatypes( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data )
{
    int block[11];
//    int i;
    MPI_Aint disp[11];
//    MPI_Aint base;
    MPI_Datatype type[11];
//    mpi_atom sample;
//    boundary_atom b_sample;
//    restart_atom r_sample;
//    rvec rvec_sample;
//    rvec2 rvec2_sample;

    /* setup the world */
    mpi_data->world = MPI_COMM_WORLD;

    /* mpi_atom: orig_id, imprt_id, type, num_bonds, num_hbonds, name,
     * x, v, f_old, s, t */
    block[0] = 1;
    block[1] = 1;
    block[2] = 1;
    block[3] = 1;
    block[4] = 1;
    block[5] = MAX_ATOM_NAME_LEN;
    block[6] = 3;
    block[7] = 3;
    block[8] = 3;
    block[9] = 4;
    block[10] = 4;

//    MPI_Get_address( &sample, &base );
//    MPI_Get_address( &(sample.orig_id), disp + 0 );
//    MPI_Get_address( &(sample.imprt_id), disp + 1 );
//    MPI_Get_address( &(sample.type), disp + 2 );
//    MPI_Get_address( &(sample.num_bonds), disp + 3 );
//    MPI_Get_address( &(sample.num_hbonds), disp + 4 );
//    MPI_Get_address( &(sample.name), disp + 5 );
//    MPI_Get_address( &(sample.x[0]), disp + 6 );
//    MPI_Get_address( &(sample.v[0]), disp + 7 );
//    MPI_Get_address( &(sample.f_old[0]), disp + 8 );
//    MPI_Get_address( &(sample.s[0]), disp + 9 );
//    MPI_Get_address( &(sample.t[0]), disp + 10 );
//    for ( i = 0; i < 11; ++i )
//    {
//        disp[i] -= base;
//    }
    disp[0] = offsetof( mpi_atom, orig_id );
    disp[1] = offsetof( mpi_atom, imprt_id );
    disp[2] = offsetof( mpi_atom, type );
    disp[3] = offsetof( mpi_atom, num_bonds );
    disp[4] = offsetof( mpi_atom, num_hbonds );
    disp[5] = offsetof( mpi_atom, name );
    disp[6] = offsetof( mpi_atom, x );
    disp[7] = offsetof( mpi_atom, v );
    disp[8] = offsetof( mpi_atom, f_old );
    disp[9] = offsetof( mpi_atom, s );
    disp[10] = offsetof( mpi_atom, t );

    type[0] = MPI_INT;
    type[1] = MPI_INT;
    type[2] = MPI_INT;
    type[3] = MPI_INT;
    type[4] = MPI_INT;
    type[5] = MPI_CHAR;
    type[6] = MPI_DOUBLE;
    type[7] = MPI_DOUBLE;
    type[8] = MPI_DOUBLE;
    type[9] = MPI_DOUBLE;
    type[10] = MPI_DOUBLE;

    MPI_Type_create_struct( 11, block, disp, type, &mpi_data->mpi_atom_type );
    MPI_Type_commit( &mpi_data->mpi_atom_type );

    /* boundary_atom - [orig_id, imprt_id, type, num_bonds, num_hbonds, x] */
    block[0] = 1;
    block[1] = 1;
    block[2] = 1;
    block[3] = 1;
    block[4] = 1;
    block[5] = 3;

//    MPI_Get_address( &b_sample, &base );
//    MPI_Get_address( &(b_sample.orig_id), disp + 0 );
//    MPI_Get_address( &(b_sample.imprt_id), disp + 1 );
//    MPI_Get_address( &(b_sample.type), disp + 2 );
//    MPI_Get_address( &(b_sample.num_bonds), disp + 3 );
//    MPI_Get_address( &(b_sample.num_hbonds), disp + 4 );
//    MPI_Get_address( &(b_sample.x[0]), disp + 5 );
//    for ( i = 0; i < 6; ++i )
//    {
//        disp[i] -= base;
//    }
    disp[0] = offsetof( boundary_atom, orig_id );
    disp[1] = offsetof( boundary_atom, imprt_id );
    disp[2] = offsetof( boundary_atom, type );
    disp[3] = offsetof( boundary_atom, num_bonds );
    disp[4] = offsetof( boundary_atom, num_hbonds );
    disp[5] = offsetof( boundary_atom, x );

    type[0] = MPI_INT;
    type[1] = MPI_INT;
    type[2] = MPI_INT;
    type[3] = MPI_INT;
    type[4] = MPI_INT;
    type[5] = MPI_DOUBLE;

    MPI_Type_create_struct( 6, block, disp, type, &mpi_data->boundary_atom_type );
    MPI_Type_commit( &mpi_data->boundary_atom_type );

    /* mpi_rvec */
    block[0] = 3;

//    MPI_Get_address( &rvec_sample, &base );
//    MPI_Get_address( &(rvec_sample[0]), disp + 0 );
//    for ( i = 0; i < 1; ++i )
//    {
//        disp[i] -= base;
//    }
    disp[0] = 0;

    type[0] = MPI_DOUBLE;

    MPI_Type_create_struct( 1, block, disp, type, &mpi_data->mpi_rvec );
    MPI_Type_commit( &mpi_data->mpi_rvec );

    /* mpi_rvec2 */
    block[0] = 2;

//    MPI_Get_address( &rvec2_sample, &base );
//    MPI_Get_address( &(rvec2_sample[0]), disp + 0 );
//    for ( i = 0; i < 1; ++i )
//    {
//        disp[i] -= base;
//    }
    disp[0] = 0;

    type[0] = MPI_DOUBLE;

    MPI_Type_create_struct( 1, block, disp, type, &mpi_data->mpi_rvec2 );
    MPI_Type_commit( &mpi_data->mpi_rvec2 );

    /* restart_atom - [orig_id, type, name, x, v] */
    block[0] = 1;
    block[1] = 1 ;
    block[2] = MAX_ATOM_NAME_LEN;
    block[3] = 3;
    block[4] = 3;

//    MPI_Get_address( &r_sample, &base );
//    MPI_Get_address( &(r_sample.orig_id), disp + 0 );
//    MPI_Get_address( &(r_sample.type), disp + 1 );
//    MPI_Get_address( &(r_sample.name), disp + 2 );
//    MPI_Get_address( &(r_sample.x[0]), disp + 3 );
//    MPI_Get_address( &(r_sample.v[0]), disp + 4 );
//    for ( i = 0; i < 5; ++i )
//    {
//        disp[i] -= base;
//    }
    disp[0] = offsetof( restart_atom, orig_id );
    disp[1] = offsetof( restart_atom, type );
    disp[2] = offsetof( restart_atom, name );
    disp[3] = offsetof( restart_atom, x );
    disp[4] = offsetof( restart_atom, v );

    type[0] = MPI_INT;
    type[1] = MPI_INT;
    type[2] = MPI_CHAR;
    type[3] = MPI_DOUBLE;
    type[4] = MPI_DOUBLE;

    MPI_Type_create_struct( 5, block, disp, type, &mpi_data->restart_atom_type );
    MPI_Type_commit( &mpi_data->restart_atom_type );

    mpi_data->in1_buffer = NULL;
    mpi_data->in2_buffer = NULL;
}


/********************** allocate lists *************************/
void Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data )
{
    int ret;

    Estimate_Num_Neighbors( system );

    Make_List( system->total_cap, system->total_far_nbrs, TYP_FAR_NEIGHBOR, lists[FAR_NBRS] );
    Init_List_Indices( lists[FAR_NBRS], system->max_far_nbrs );

    ret = Generate_Neighbor_Lists( system, data, workspace, lists );
    if ( ret != SUCCESS )
    {
        fprintf( stderr, "[ERROR] p%d: failed to generate neighbor lists. Terminating...\n", system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }
    
    Estimate_Storages( system, control, lists );
    
    Allocate_Matrix( &workspace->H, system->n, system->total_cm_entries );
    Init_Matrix_Row_Indices( &workspace->H, system->max_cm_entries );

    if ( control->hbond_cut > 0.0 )
    {
        Make_List( system->total_cap, system->total_hbonds, TYP_HBOND, lists[HBONDS] );
        Init_List_Indices( lists[HBONDS], system->max_hbonds );
    }

    Make_List( system->total_cap, system->total_bonds, TYP_BOND, lists[BONDS] );
    Init_List_Indices( lists[BONDS], system->max_bonds );

    Make_List( system->total_bonds, system->total_thbodies, TYP_THREE_BODY, lists[THREE_BODIES] );

#if defined(TEST_FORCES)
    Make_List( system->total_cap, system->total_bonds * 8, TYP_DDELTA, lists[DDELTAS] );
    Make_List( system->total_bonds, system->total_bonds * 50, TYP_DBO, lists[DBOS] );
#endif
}


#if defined(PURE_REAX)
void Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    char msg[MAX_STR];

    Init_MPI_Datatypes( system, workspace, mpi_data );

    Init_System( system, control, data, workspace, mpi_data );

    Init_Simulation_Data( system, control, data );

    Init_Workspace( system, control, workspace );

    Init_Lists( system, control, data, workspace, lists, mpi_data );

    if ( Init_Output_Files( system, control, out_control, mpi_data, msg ) == FAILURE )
    {
        fprintf( stderr, "[ERROR] p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "[ERROR] p%d: could not open output files! terminating...\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

    if ( control->tabulate )
    {
        Init_Lookup_Tables( system, control, workspace->Tap, mpi_data );
    }

    Init_Force_Functions( control );

#ifdef TEST_FORCES
//    Init_Force_Test_Functions( );
//    fprintf( stderr, "p%d: initialized force test functions\n", system->my_rank );
#endif
}


void Pure_Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    Init_Simulation_Data( system, control, data );

    Init_Workspace( system, control, workspace );

    Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Force_Functions( control );
}


#elif defined(LAMMPS_REAX)
void Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    char msg[MAX_STR];

    Init_System( system );

    Init_Simulation_Data( system, control, data );

    Init_Workspace( system, control, workspace );

    Init_MPI_Datatypes( system, workspace, mpi_data );

    Init_Lists( system, control, workspace, lists );

    if ( Init_Output_Files( system, control, out_control, mpi_data, msg ) == FAILURE)
    {
        fprintf( stderr, "[ERROR] p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "[ERROR] p%d: could not open output files! terminating...\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

    if ( control->tabulate )
    {
        Init_Lookup_Tables( system, control, workspace->Tap, mpi_data );
    }

    Init_Force_Functions( );
    }
#endif
