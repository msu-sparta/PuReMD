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

#if (defined(HAVE_CONFIG_H) && !defined(__CONFIG_H_))
  #define __CONFIG_H_
  #include "../../common/include/config.h"
#endif

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

#include <stddef.h>


#if defined(PURE_REAX)
/************************ initialize system ************************/
static void Reposition_Atoms( reax_system * const system, control_params * const control,
        simulation_data * const data, mpi_datatypes * const mpi_data )
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
        rvec_ScaledAdd( dx, -1.0, data->xcm );
    }
    /* put center of mass to origin */
    else if ( control->reposition_atoms == 2 )
    {
        rvec_Scale( dx, -1.0, data->xcm );
    }
    else
    {
        fprintf( stderr, "[ERROR] p%d: Reposition_Atoms: invalid option (%d)\n",
              system->my_rank, control->reposition_atoms );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
    }

    for ( i = 0; i < system->n; ++i )
    {
//        Inc_on_T3_Gen( system->my_atoms[i].x, dx, &system->big_box );
        rvec_Add( system->my_atoms[i].x, dx );
    }
}



void Generate_Initial_Velocities( reax_system * const system,
        control_params * const control, real T )
{
    int i;
    real m, scale, norm;

    if ( T <= 0.1 || control->random_vel == FALSE )
    {
        /* warnings if conflicts between initial temperature and control file parameter */
        if ( control->random_vel == TRUE )
        {
            fprintf( stderr, "[ERROR] conflicting control file parameters\n" );
            fprintf( stderr, "[INFO] random_vel = 1 and small initial temperature (t_init = %f)\n", T );
            fprintf( stderr, "[INFO] set random_vel = 0 to resolve this (atom initial velocites set to zero)\n" );
            MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
        }
        else if ( T > 0.1 )
        {
            fprintf( stderr, "[ERROR] conflicting control file paramters\n" );
            fprintf( stderr, "[INFO] random_vel = 0 and large initial temperature (t_init = %f)\n", T );
            fprintf( stderr, "[INFO] set random_vel = 1 to resolve this (random atom initial velocites according to t_init)\n" );
            MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
        }

        for ( i = 0; i < system->n; i++ )
        {
            rvec_MakeZero( system->my_atoms[i].v );
        }
    }
    else
    {
        if ( T <= 0.0 )
        {
            fprintf( stderr, "[ERROR] random atom initial velocities specified with invalid temperature (%f). Terminating...\n",
                  T );
            MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
        }

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


void Init_System( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        mpi_datatypes * const mpi_data )
{
    int i;
    reax_atom *atom;

    Setup_New_Grid( system, control, MPI_COMM_WORLD );

    /* since all processors read in all atoms and select their local atoms
     * intially, no local atoms comm needed and just bin local atoms */
    Bin_My_Atoms( system, workspace );
    Reorder_My_Atoms( system, workspace );

    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type,
            &Count_Boundary_Atoms, &Sort_Boundary_Atoms,
            &Unpack_Exchange_Message, TRUE );

    system->total_cap = MAX( (int) CEIL( system->N * SAFE_ZONE ), MIN_CAP );

    Bin_Boundary_Atoms( system );
#if defined(NEUTRAL_TERRITORY)
    Estimate_NT_Atoms( system, mpi_data );
#endif

    /* estimate numH */
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

    /* list management */
    system->far_nbrs = smalloc( sizeof(int) * system->total_cap,
            "Init_System::system->far_nbrs" );
    system->max_far_nbrs = smalloc( sizeof(int) * system->total_cap,
            "Init_System::system->max_far_nbrs" );

    system->bonds = smalloc( sizeof(int) * system->total_cap,
            "Init_System::system->bonds" );
    system->max_bonds = smalloc( sizeof(int) * system->total_cap,
            "Init_System::system->max_bonds" );

    system->hbonds = smalloc( sizeof(int) * system->total_cap,
            "Init_System::system->hbonds" );
    system->max_hbonds = smalloc( sizeof(int) * system->total_cap,
            "Init_System::system->max_hbonds" );

    system->cm_entries = smalloc( sizeof(int) * system->local_cap,
            "Init_System::system->cm_entries" );
    system->max_cm_entries = smalloc( sizeof(int) * system->local_cap,
            "Init_System::max_cm_entries->max_hbonds" );
    
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d, local_cap=%d\n",
             system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d, total_cap=%d\n",
             system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d\n",
             system->my_rank, system->numH );
#endif

    Compute_Total_Mass( system, data, mpi_data->comm_mesh3D );

    Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );

//    Reposition_Atoms( system, control, data, mpi_data );

    /* initialize velocities so that desired init T can be attained */
    if ( control->restart == FALSE
            || (control->restart == TRUE && control->random_vel == TRUE) )
    {
        Generate_Initial_Velocities( system, control, control->T_init );
    }

    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
}


/************************ initialize simulation data ************************/
void Init_Simulation_Data( reax_system * const system, control_params * const control,
        simulation_data * const data, mpi_datatypes * const mpi_data )
{
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
        control->Evolve = &Velocity_Verlet_NVE;
        control->virial = 0;
        break;

    case bNVT:
        data->N_f = 3 * system->bigN;
        control->Evolve = &Velocity_Verlet_Berendsen_NVT;
        control->virial = 0;
        break;

    case nhNVT:
        fprintf( stderr, "[WARNING] Nose-Hoover NVT is still under testing.\n" );
        data->N_f = 3 * system->bigN + 1;
        control->Evolve = &Velocity_Verlet_Nose_Hoover_NVT_Klein;
        control->virial = 0;
        if ( !control->restart || (control->restart && control->random_vel) )
        {
            data->therm.G_xi = control->Tau_T
                * (2.0 * data->sys_en.e_kin - data->N_f * K_B * control->T );
            data->therm.v_xi = data->therm.G_xi * control->dt;
            data->therm.v_xi_old = 0;
            data->therm.xi = 0;
        }
        break;

    /* Semi-Isotropic NPT */
    case sNPT:
        data->N_f = 3 * system->bigN + 4;
        control->Evolve = &Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Isotropic NPT */
    case iNPT:
        data->N_f = 3 * system->bigN + 2;
        control->Evolve = &Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    /* Anisotropic NPT */
    case NPT:
        fprintf( stderr, "[ERROR] p%d: Init_Simulation_Data: option not yet implemented (anisotropic NPT ensemble)\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );

        data->N_f = 3 * system->bigN + 9;
        control->Evolve = &Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
//        if( !control->restart )
//        {
//            data->therm.G_xi = control->Tau_T
//                * (2.0 * data->my_en.e_Kin - data->N_f * K_B * control->T );
//            data->therm.v_xi = data->therm.G_xi * control->dt;
//            data->iso_bar.eps = (1.0 / 3.0) * LOG(system->box.volume);
//            data->inv_W = 1.0
//                / ( data->N_f * K_B * control->T * SQR(control->Tau_P) );
//            Compute_Pressure( system, control, data, out_control );
//        }
        break;

    default:
        fprintf( stderr, "[ERROR] p%d: Init_Simulation_Data: ensemble not recognized\n",
              system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }
}


#elif defined(LAMMPS_REAX)
void Init_System( reax_system * const system )
{
    system->big_box.V = 0.0;
    system->big_box.box_norms[0] = 0.0;
    system->big_box.box_norms[1] = 0.0;
    system->big_box.box_norms[2] = 0.0;

    system->local_cap = (int) CEIL( system->n * SAFE_ZONE );
    system->total_cap = (int) CEIL( system->N * SAFE_ZONE );

    system->far_nbrs = NULL;
    system->max_far_nbrs = NULL;
    system->bonds = NULL;
    system->max_bonds = NULL;
    system->hbonds = NULL;
    system->max_hbonds = NULL;
    system->cm_entries = NULL;
    system->max_cm_entries = NULL;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: local_cap=%d total_cap=%d\n",
             system->my_rank, system->local_cap, system->total_cap );
#endif

    Reallocate_System_Part1( system, system->local_cap );
    Reallocate_System_Part2( system, system->total_cap );
}


void Init_Simulation_Data( reax_system * const system, control_params * const control,
        simulation_data * const data )
{
    Reset_Simulation_Data( data );
    Reset_Timing( &data->timing );

    //if( !control->restart )
    data->step = 0;
    data->prev_steps = 0;
}
#endif


/************************ initialize workspace ************************/
/* Initialize Taper params */
void Init_Taper( control_params * const control,  storage * const workspace,
        mpi_datatypes * const mpi_data )
{
    real d1, d7;
    const real swa = control->nonb_low;
    const real swb = control->nonb_cut;
    real swa2, swa3;
    real swb2, swb3;

    if ( FABS( swa ) > 0.01 )
    {
        fprintf( stderr, "[WARNING] non-zero lower Taper-radius cutoff in force field parameters\n" );
    }

    if ( swb < 0.0 )
    {
        fprintf( stderr, "[ERROR] negative upper Taper-radius cutoff in force field parameters\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }
    else if ( swb < 5.0 )
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
    workspace->Tap[0] = (-35.0 * swa3 * swb2 * swb2 + 21.0 * swa2 * swb3 * swb2
            + 7.0 * swa * swb3 * swb3 + swb3 * swb3 * swb ) / d7;
}


void Init_Workspace( reax_system * const system, control_params * const control,
        storage * const workspace, mpi_datatypes * const mpi_data )
{
    Allocate_Workspace_Part1( system, control, workspace, system->local_cap );
    Allocate_Workspace_Part2( system, control, workspace, system->total_cap );

    workspace->realloc.far_nbrs = FALSE;
    workspace->realloc.cm = FALSE;
    workspace->realloc.hbonds = FALSE;
    workspace->realloc.bonds = FALSE;
    workspace->realloc.thbody = FALSE;
    workspace->realloc.gcell_atoms = 0;

    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        workspace->H_full.allocated = FALSE;
        workspace->H_spar_patt.allocated = FALSE;
        workspace->H_spar_patt_full.allocated = FALSE;
        workspace->H_app_inv.allocated = FALSE;
    }

    Reset_Workspace( system, workspace );

    Init_Taper( control, workspace, mpi_data );
}


/* Setup communication data structures
 * */
void Init_MPI_Datatypes( reax_system * const system, storage * const workspace,
        mpi_datatypes * const mpi_data )
{
    int i, ret, block[11];
    MPI_Aint disp[11], base;
    MPI_Datatype type[11], temp_type;
    mpi_atom m_sample[1];
    boundary_atom b_sample[1];
    restart_atom r_sample[1];

    /* mpi_atom */
    block[0] = 1;
    block[1] = 1;
    block[2] = 1;
    block[3] = 1;
    block[4] = 1;
    block[5] = sizeof(m_sample[0].name) / sizeof(char);
    block[6] = 3;
    block[7] = 3;
    block[8] = 3;
    block[9] = 4;
    block[10] = 4;

    MPI_Get_address( &m_sample[0], &base );
    MPI_Get_address( &m_sample[0].orig_id, &disp[0] );
    MPI_Get_address( &m_sample[0].imprt_id, &disp[1] );
    MPI_Get_address( &m_sample[0].type, &disp[2] );
    MPI_Get_address( &m_sample[0].num_bonds, &disp[3] );
    MPI_Get_address( &m_sample[0].num_hbonds, &disp[4] );
    MPI_Get_address( &m_sample[0].name, &disp[5] );
    MPI_Get_address( &m_sample[0].x, &disp[6] );
    MPI_Get_address( &m_sample[0].v, &disp[7] );
    MPI_Get_address( &m_sample[0].f_old, &disp[8] );
    MPI_Get_address( &m_sample[0].s, &disp[9] );
    MPI_Get_address( &m_sample[0].t, &disp[10] );
    for ( i = 0; i < 11; ++i )
    {
        disp[i] = MPI_Aint_diff( disp[i], base );
    }

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

    ret = MPI_Type_create_struct( 11, block, disp, type, &temp_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    /* account for struct padding after members */
    ret = MPI_Type_create_resized( temp_type, 0, sizeof(mpi_atom),
            &mpi_data->mpi_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_commit( &mpi_data->mpi_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_free( &temp_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* boundary_atom */
    block[0] = 1;
    block[1] = 1;
    block[2] = 1;
    block[3] = 1;
    block[4] = 1;
    block[5] = 3;

    MPI_Get_address( &b_sample[0], &base );
    MPI_Get_address( &b_sample[0].orig_id, &disp[0] );
    MPI_Get_address( &b_sample[0].imprt_id, &disp[1] );
    MPI_Get_address( &b_sample[0].type, &disp[2] );
    MPI_Get_address( &b_sample[0].num_bonds, &disp[3] );
    MPI_Get_address( &b_sample[0].num_hbonds, &disp[4] );
    MPI_Get_address( &b_sample[0].x, &disp[5] );
    for ( i = 0; i < 6; ++i )
    {
        disp[i] = MPI_Aint_diff( disp[i], base );
    }

    type[0] = MPI_INT;
    type[1] = MPI_INT;
    type[2] = MPI_INT;
    type[3] = MPI_INT;
    type[4] = MPI_INT;
    type[5] = MPI_DOUBLE;

    ret = MPI_Type_create_struct( 6, block, disp, type, &temp_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    /* account for struct padding after members */
    ret = MPI_Type_create_resized( temp_type, 0, sizeof(boundary_atom),
            &mpi_data->boundary_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_commit( &mpi_data->boundary_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_free( &temp_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* mpi_rvec */
    ret = MPI_Type_contiguous( 3, MPI_DOUBLE, &mpi_data->mpi_rvec );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_commit( &mpi_data->mpi_rvec );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* mpi_rvec2 */
    ret = MPI_Type_contiguous( 2, MPI_DOUBLE, &mpi_data->mpi_rvec2 );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_commit( &mpi_data->mpi_rvec2 );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* restart_atom */
    block[0] = 1;
    block[1] = 1 ;
    block[2] = sizeof(r_sample[0].name) / sizeof(char);
    block[3] = 3;
    block[4] = 3;

    MPI_Get_address( &r_sample[0], &base );
    MPI_Get_address( &r_sample[0].orig_id, &disp[0] );
    MPI_Get_address( &r_sample[0].type, &disp[1] );
    MPI_Get_address( &r_sample[0].name, &disp[2] );
    MPI_Get_address( &r_sample[0].x, &disp[3] );
    MPI_Get_address( &r_sample[0].v, &disp[4] );
    for ( i = 0; i < 5; ++i )
    {
        disp[i] = MPI_Aint_diff( disp[i], base );
    }

    type[0] = MPI_INT;
    type[1] = MPI_INT;
    type[2] = MPI_CHAR;
    type[3] = MPI_DOUBLE;
    type[4] = MPI_DOUBLE;

    ret = MPI_Type_create_struct( 5, block, disp, type, &temp_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    /* account for struct padding after members */
    ret = MPI_Type_create_resized( temp_type, 0, sizeof(restart_atom),
            &mpi_data->restart_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_commit( &mpi_data->restart_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_free( &temp_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    mpi_data->in1_buffer = NULL;
    mpi_data->in1_buffer_size = 0;
    mpi_data->in2_buffer = NULL;
    mpi_data->in2_buffer_size = 0;

    for ( i = 0; i < MAX_NBRS; ++i )
    {
        mpi_data->out_buffers[i].cnt = 0;
        mpi_data->out_buffers[i].index = NULL;
        mpi_data->out_buffers[i].index_size = 0;
        mpi_data->out_buffers[i].out_atoms = NULL;
        mpi_data->out_buffers[i].out_atoms_size = 0;
    }

#if defined(NEUTRAL_TERRITORY)
    for ( i = 0; i < MAX_NT_NBRS; ++i )
    {
        mpi_data->in_nt_buffer[i] = NULL;
    }

    for ( i = 0; i < MAX_NT_NBRS; ++i )
    {
        mpi_data->out_nt_buffers[i].cnt = 0;
        mpi_data->out_nt_buffers[i].index = NULL;
        mpi_data->out_nt_buffers[i].index_size = 0;
        mpi_data->out_nt_buffers[i].out_atoms = NULL;
        mpi_data->out_nt_buffers[i].out_atoms_size = 0;
    }
#endif
}


/* Allocate and initialize lists
 * */
void Init_Lists( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        mpi_datatypes * const mpi_data )
{
    int ret, far_nbr_list_format, cm_format, matrix_dim;

    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
        far_nbr_list_format = FULL_LIST;
        cm_format = SYM_FULL_MATRIX;
    }
    else
    {
#if defined(NEUTRAL_TERRITORY)
        far_nbr_list_format = FULL_LIST;
        cm_format = SYM_HALF_MATRIX;
#else
        far_nbr_list_format = HALF_LIST;
        cm_format = SYM_HALF_MATRIX;
#endif
    }

    Estimate_Num_Neighbors( system, data, far_nbr_list_format );

    Make_List( system->total_cap, system->total_far_nbrs, TYP_FAR_NEIGHBOR,
            far_nbr_list_format, lists[FAR_NBRS] );
    Init_List_Indices( lists[FAR_NBRS], system->max_far_nbrs );

    ret = Generate_Neighbor_Lists( system, data, workspace, lists );
    if ( ret != SUCCESS )
    {
        fprintf( stderr, "[ERROR] p%d: failed to generate neighbor lists. Terminating...\n", system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }
    
    Estimate_Storages( system, control, lists, &matrix_dim, cm_format );
    
#if defined(NEUTRAL_TERRITORY)
    Allocate_Matrix( &workspace->H, matrix_dim, system->local_cap, system->total_cm_entries, cm_format );
#else
    Allocate_Matrix( &workspace->H, system->n, system->local_cap, system->total_cm_entries, cm_format );
#endif
    Init_Matrix_Row_Indices( &workspace->H, system->max_cm_entries );

    if ( control->hbond_cut > 0.0 )
    {
        Make_List( system->total_cap, system->total_hbonds, TYP_HBOND,
                HALF_LIST, lists[HBONDS] );
        Init_List_Indices( lists[HBONDS], system->max_hbonds );
    }

    Make_List( system->total_cap, system->total_bonds, TYP_BOND,
            HALF_LIST, lists[BONDS] );
    Init_List_Indices( lists[BONDS], system->max_bonds );

    Make_List( system->total_bonds, system->total_thbodies, TYP_THREE_BODY,
            HALF_LIST, lists[THREE_BODIES] );

#if defined(TEST_FORCES)
    Make_List( system->total_cap, system->total_bonds * 8, TYP_DDELTA,
            HALF_LIST, lists[DDELTAS] );
    Make_List( system->total_bonds, system->total_bonds * 50, TYP_DBO,
            HALF_LIST, lists[DBOS] );
#endif
}


#if defined(PURE_REAX)
void Initialize( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
    Init_MPI_Datatypes( system, workspace, mpi_data );

    Init_Simulation_Data( system, control, data, mpi_data );

    Init_System( system, control, data, workspace, mpi_data );
    /* reset for step 0 */
    Reset_Simulation_Data( data );

    Init_Workspace( system, control, workspace, mpi_data );

    Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Output_Files( system, control, out_control, mpi_data );

    if ( control->tabulate > 0 )
    {
        Make_LR_Lookup_Table( system, control, workspace, mpi_data );
    }

    Init_Force_Functions( control );

#if defined(TEST_FORCES)
//    Init_Force_Test_Functions( );
//    fprintf( stderr, "p%d: initialized force test functions\n", system->my_rank );
#endif
}


void Pure_Initialize( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
    Init_Simulation_Data( system, control, data, mpi_data );

    Init_Workspace( system, control, workspace, mpi_data );

    Init_Lists( system, control, data, workspace, lists, mpi_data );

    Init_Force_Functions( control );
}


#elif defined(LAMMPS_REAX)
void Initialize( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control,
        mpi_datatypes * const mpi_data )
{
    Init_Simulation_Data( system, control, data );

    Init_System( system );
    /* reset for step 0 */
    Reset_Simulation_Data( data );

    Init_Workspace( system, control, workspace );

    Init_MPI_Datatypes( system, workspace, mpi_data );

    Init_Lists( system, control, workspace, lists );

    Init_Output_Files( system, control, out_control, mpi_data );

    if ( control->tabulate > 0 )
    {
        Make_LR_Lookup_Table( system, control, workspace, mpi_data );
    }

    Init_Force_Functions( );
    }
#endif


static void Finalize_System( reax_system * const system, control_params * const control,
        simulation_data * const data )
{
    reax_interaction * const reax = &system->reax_param;

    Deallocate_Grid( &system->my_grid );

    sfree( reax->gp.l, "Finalize_System::reax->gp.l" );

    sfree( reax->sbp, "Finalize_System::reax->sbp" );
    sfree( reax->tbp, "Finalize_System::reax->tbp" );
    sfree( reax->thbp, "Finalize_System::reax->thbp" );
    sfree( reax->hbp, "Finalize_System::reax->hbp" );
    sfree( reax->fbp, "Finalize_System::reax->fbp" );

    sfree( system->far_nbrs, "Finalize_System::system->far_nbrs" );
    sfree( system->max_far_nbrs, "Finalize_System::system->max_far_nbrs" );
    sfree( system->bonds, "Finalize_System::system->bonds" );
    sfree( system->max_bonds, "Finalize_System::system->max_bonds" );
    sfree( system->hbonds, "Finalize_System::system->hbonds" );
    sfree( system->max_hbonds, "Finalize_System::system->max_hbonds" );
    sfree( system->cm_entries, "Finalize_System::system->cm_entries" );
    sfree( system->max_cm_entries, "Finalize_System::system->max_cm_entries" );

    sfree( system->my_atoms, "Finalize_System::system->atoms" );
}


static void Finalize_Simulation_Data( reax_system * const system, control_params * const control,
        simulation_data * const data, output_controls * const out_control )
{
}


static void Finalize_Workspace( reax_system * const system, control_params * const control,
        storage * const workspace )
{
    sfree( workspace->total_bond_order, "Finalize_Workspace::workspace->total_bond_order" );
    sfree( workspace->Deltap, "Finalize_Workspace::workspace->Deltap" );
    sfree( workspace->Deltap_boc, "Finalize_Workspace::workspace->Deltap_boc" );
    sfree( workspace->dDeltap_self, "Finalize_Workspace::workspace->dDeltap_self" );
    sfree( workspace->Delta, "Finalize_Workspace::workspace->Delta" );
    sfree( workspace->Delta_lp, "Finalize_Workspace::workspace->Delta_lp" );
    sfree( workspace->Delta_lp_temp, "Finalize_Workspace::workspace->Delta_lp_temp" );
    sfree( workspace->dDelta_lp, "Finalize_Workspace::workspace->dDelta_lp" );
    sfree( workspace->dDelta_lp_temp, "Finalize_Workspace::workspace->dDelta_lp_temp" );
    sfree( workspace->Delta_e, "Finalize_Workspace::workspace->Delta_e" );
    sfree( workspace->Delta_boc, "Finalize_Workspace::workspace->Delta_boc" );
    sfree( workspace->nlp, "Finalize_Workspace::workspace->nlp" );
    sfree( workspace->nlp_temp, "Finalize_Workspace::workspace->nlp_temp" );
    sfree( workspace->Clp, "Finalize_Workspace::workspace->Clp" );
    sfree( workspace->CdDelta, "Finalize_Workspace::workspace->CdDelta" );
    sfree( workspace->vlpex, "Finalize_Workspace::workspace->vlpex" );
    sfree( workspace->bond_mark, "Finalize_Workspace::bond_mark" );

    Deallocate_Matrix( &workspace->H );
    if ( control->cm_solver_pre_comp_type == SAI_PC )
    {
//        Deallocate_Matrix( &workspace->H_full );
//        Deallocate_Matrix( &workspace->H_spar_patt );
//        Deallocate_Matrix( &workspace->H_spar_patt_full );
//        Deallocate_Matrix( &workspace->H_app_inv );
    }

    if ( control->cm_solver_pre_comp_type == JACOBI_PC )
    {
        sfree( workspace->Hdia_inv, "Finalize_Workspace::workspace->Hdia_inv" );
    }
    if ( control->cm_solver_pre_comp_type == ICHOLT_PC
            || control->cm_solver_pre_comp_type == ILUT_PC
            || control->cm_solver_pre_comp_type == ILUTP_PC
            || control->cm_solver_pre_comp_type == FG_ILUT_PC )
    {
        sfree( workspace->droptol, "Finalize_Workspace::workspace->droptol" );
    }
    sfree( workspace->b_s, "Finalize_Workspace::workspace->b_s" );
    sfree( workspace->b_t, "Finalize_Workspace::workspace->b_t" );
    sfree( workspace->b_prc, "Finalize_Workspace::workspace->b_prc" );
    sfree( workspace->b_prm, "Finalize_Workspace::workspace->b_prm" );
    sfree( workspace->s, "Finalize_Workspace::workspace->s" );
    sfree( workspace->t, "Finalize_Workspace::workspace->t" );
    sfree( workspace->b, "Finalize_Workspace::workspace->b" );
    sfree( workspace->x, "Finalize_Workspace::workspace->x" );

    switch ( control->cm_solver_type )
    {
        case GMRES_S:
        case GMRES_H_S:
            sfree( workspace->y, "Finalize_Workspace::workspace->y" );
            sfree( workspace->z, "Finalize_Workspace::workspace->z" );
            sfree( workspace->g, "Finalize_Workspace::workspace->g" );
            sfree( workspace->h, "Finalize_Workspace::workspace->h" );
            sfree( workspace->hs, "Finalize_Workspace::workspace->hs" );
            sfree( workspace->hc, "Finalize_Workspace::workspace->hc" );
            sfree( workspace->v, "Finalize_Workspace::workspace->v" );
            break;

        case CG_S:
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            sfree( workspace->r2, "Finalize_Workspace::workspace->r2" );
            sfree( workspace->d2, "Finalize_Workspace::workspace->d2" );
            sfree( workspace->q2, "Finalize_Workspace::workspace->q2" );
            sfree( workspace->p2, "Finalize_Workspace::workspace->p2" );
            break;

        case SDM_S:
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            sfree( workspace->r2, "Finalize_Workspace::workspace->r2" );
            sfree( workspace->d2, "Finalize_Workspace::workspace->d2" );
            sfree( workspace->q2, "Finalize_Workspace::workspace->q2" );
            sfree( workspace->p2, "Finalize_Workspace::workspace->p2" );
            break;

        case BiCGStab_S:
            sfree( workspace->y, "Finalize_Workspace::workspace->y" );
            sfree( workspace->g, "Finalize_Workspace::workspace->g" );
            sfree( workspace->z, "Finalize_Workspace::workspace->z" );
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            sfree( workspace->r_hat, "Finalize_Workspace::workspace->r_hat" );
            sfree( workspace->q_hat, "Finalize_Workspace::workspace->q_hat" );
            break;

        case PIPECG_S:
            sfree( workspace->z, "Finalize_Workspace::workspace->z" );
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            sfree( workspace->m, "Finalize_Workspace::workspace->m" );
            sfree( workspace->n, "Finalize_Workspace::workspace->n" );
            sfree( workspace->u, "Finalize_Workspace::workspace->u" );
            sfree( workspace->w, "Finalize_Workspace::workspace->w" );
            sfree( workspace->z2, "Finalize_Workspace::workspace->z2" );
            sfree( workspace->r2, "Finalize_Workspace::workspace->r2" );
            sfree( workspace->d2, "Finalize_Workspace::workspace->d2" );
            sfree( workspace->q2, "Finalize_Workspace::workspace->q2" );
            sfree( workspace->p2, "Finalize_Workspace::workspace->p2" );
            sfree( workspace->m2, "Finalize_Workspace::workspace->m2" );
            sfree( workspace->n2, "Finalize_Workspace::workspace->n2" );
            sfree( workspace->u2, "Finalize_Workspace::workspace->u2" );
            sfree( workspace->w2, "Finalize_Workspace::workspace->w2" );
            break;

        case PIPECR_S:
            sfree( workspace->z, "Finalize_Workspace::workspace->z" );
            sfree( workspace->r, "Finalize_Workspace::workspace->r" );
            sfree( workspace->d, "Finalize_Workspace::workspace->d" );
            sfree( workspace->q, "Finalize_Workspace::workspace->q" );
            sfree( workspace->p, "Finalize_Workspace::workspace->p" );
            sfree( workspace->m, "Finalize_Workspace::workspace->m" );
            sfree( workspace->n, "Finalize_Workspace::workspace->n" );
            sfree( workspace->u, "Finalize_Workspace::workspace->u" );
            sfree( workspace->w, "Finalize_Workspace::workspace->w" );
            break;

        default:
            fprintf( stderr, "[ERROR] Unknown charge method linear solver type. Terminating...\n" );
            exit( UNKNOWN_OPTION );
            break;
    }

    /* integrator storage */
    if ( control->ensemble == nhNVT )
    {
        sfree( workspace->v_const, "Finalize_Workspace::workspace->v_const" );
    }

    /* storage for analysis */
    if ( control->molecular_analysis || control->diffusion_coef )
    {
        sfree( workspace->mark, "Finalize_Workspace::workspace->mark" );
        sfree( workspace->old_mark, "Finalize_Workspace::workspace->old_mark" );
    }

    if ( control->diffusion_coef )
    {
        sfree( workspace->x_old, "Finalize_Workspace::workspace->x_old" );
    }

    /* force-related storage */
    sfree( workspace->f, "Finalize_Workspace::workspace->f" );

    /* space for keeping restriction info, if any */
    if ( control->restrict_bonds )
    {
        sfree( workspace->restricted, "Finalize_Workspace::workspace->restricted" );
        sfree( workspace->restricted_list, "Finalize_Workspace::workspace->restricted_list" );
    }

#if defined(TEST_FORCES)
    sfree( workspace->dDelta, "Finalize_Workspace::workspace->dDelta" );
    sfree( workspace->f_ele, "Finalize_Workspace::workspace->f_ele" );
    sfree( workspace->f_vdw, "Finalize_Workspace::workspace->f_vdw" );
    sfree( workspace->f_bo, "Finalize_Workspace::workspace->f_bo" );
    sfree( workspace->f_be, "Finalize_Workspace::workspace->f_be" );
    sfree( workspace->f_lp, "Finalize_Workspace::workspace->f_lp" );
    sfree( workspace->f_ov, "Finalize_Workspace::workspace->f_ov" );
    sfree( workspace->f_un, "Finalize_Workspace::workspace->f_un" );
    sfree( workspace->f_ang, "Finalize_Workspace::workspace->f_ang" );
    sfree( workspace->f_coa, "Finalize_Workspace::workspace->f_coa" );
    sfree( workspace->f_pen, "Finalize_Workspace::workspace->f_pen" );
    sfree( workspace->f_hb, "Finalize_Workspace::workspace->f_hb" );
    sfree( workspace->f_tor, "Finalize_Workspace::workspace->f_tor" );
    sfree( workspace->f_con, "Finalize_Workspace::workspace->f_con" );
    sfree( workspace->f_tot, "Finalize_Workspace::workspace->f_tot" );

    sfree( workspace->rcounts, "Finalize_Workspace::workspace->rcounts" );
    sfree( workspace->displs, "Finalize_Workspace::workspace->displs" );
    sfree( workspace->id_all, "Finalize_Workspace::workspace->id_all" );
    sfree( workspace->f_all, "Finalize_Workspace::workspace->f_all" );
#endif
}


static void Finalize_Lists( control_params * const control, reax_list ** const lists )
{
    int i;

    for ( i = 0; i < LIST_N; ++i )
    {
        Delete_List( lists[i] );
        sfree( lists[i], "Finalize_Lists::lists[i]" );
    }
}


static void Finalize_MPI_Datatypes( mpi_datatypes * const mpi_data )
{
    int ret;

    Deallocate_MPI_Buffers( mpi_data );

    ret = MPI_Type_free( &mpi_data->mpi_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_free( &mpi_data->boundary_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_free( &mpi_data->mpi_rvec );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_free( &mpi_data->mpi_rvec2 );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_free( &mpi_data->restart_atom_type );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    MPI_Comm_free( &mpi_data->comm_mesh3D );
}


/* Deallocate top-level data structures, close file handles, etc.
 *
 */
void Finalize( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data,
        const int output_enabled )
{
    if ( control->tabulate )
    {
        Finalize_LR_Lookup_Table( system, control, workspace, mpi_data );
    }

    if ( output_enabled == TRUE )
    {
        Finalize_Output_Files( system, control, out_control );
    }

    Finalize_Lists( control, lists );

    Finalize_Workspace( system, control, workspace );

    Finalize_Simulation_Data( system, control, data, out_control );

    Finalize_System( system, control, data );

    Finalize_MPI_Datatypes( mpi_data );
}
