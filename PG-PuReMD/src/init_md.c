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

#ifdef HAVE_CUDA
  #include "cuda_allocate.h"
  #include "cuda_list.h"
  #include "cuda_copy.h"
  #include "cuda_forces.h"
  #include "cuda_init_md.h"
  #include "cuda_neighbors.h"
  #include "cuda_reset_tools.h"
  #include "cuda_validation.h"
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


#if defined(PURE_REAX)
/************************ initialize system ************************/
int Reposition_Atoms( reax_system *system, control_params *control,
        simulation_data *data, mpi_datatypes *mpi_data, char *msg )
{
    int i;
    rvec dx;

    /* reposition atoms */
    if ( control->reposition_atoms == 0 )  //fit atoms to periodic box
    {
        rvec_MakeZero( dx );
    }
    else if ( control->reposition_atoms == 1 )  //put center of mass to center
    {
        rvec_Scale( dx, 0.5, system->big_box.box_norms );
        rvec_ScaledAdd( dx, -1., data->xcm );
    }
    else if ( control->reposition_atoms == 2 )  //put center of mass to origin
    {
        rvec_Scale( dx, -1., data->xcm );
    }
    else
    {
        strcpy( msg, "reposition_atoms: invalid option" );
        return FAILURE;
    }

    for ( i = 0; i < system->n; ++i )
        // Inc_on_T3_Gen( system->my_atoms[i].x, dx, &(system->big_box) );
        rvec_Add( system->my_atoms[i].x, dx );

    return SUCCESS;
}



void Generate_Initial_Velocities( reax_system *system, real T )
{
    int i;
    real m, scale, norm;


    if ( T <= 0.1 )
    {
        for ( i = 0; i < system->n; i++ )
            rvec_MakeZero( system->my_atoms[i].v );
    }
    else
    {
        Randomize();

        for ( i = 0; i < system->n; i++ )
        {
            rvec_Random( system->my_atoms[i].v );

            norm = rvec_Norm_Sqr( system->my_atoms[i].v );
            m = system->reax_param.sbp[ system->my_atoms[i].type ].mass;
            scale = SQRT( m * norm / (3.0 * K_B * T) );

            rvec_Scale( system->my_atoms[i].v, 1. / scale, system->my_atoms[i].v );

            // fprintf( stderr, "v = %f %f %f\n",
            // system->my_atoms[i].v[0],
            // system->my_atoms[i].v[1],
            // system->my_atoms[i].v[2] );

            // fprintf( stderr, "scale = %f\n", scale );
            // fprintf( stderr, "v = %f %f %f\n",
            // system->my_atoms[i].v[0],
            // system->my_atoms[i].v[1],
            // system->my_atoms[i].v[2] );
        }
    }
}


int Init_System( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        mpi_datatypes *mpi_data, char *msg )
{
    int i;
    reax_atom *atom;
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
    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
            Estimate_Boundary_Atoms, Unpack_Estimate_Message, TRUE );
    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );
    Bin_Boundary_Atoms( system );

    /* estimate numH and Hcap */
    system->numH = 0;
    if ( control->hbond_cut > 0.0 )
    {
        for ( i = 0; i < system->n; ++i )
        {
            atom = &(system->my_atoms[i]);

            if ( system->reax_param.sbp[ atom->type ].p_hbond == 1 )
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
    
// Sudhir-style below
/*
    system->numH = 0;
    if ( control->hbond_cut > 0.0 )
        for ( i = 0; i < system->n; ++i )
        {
            atom = &(system->my_atoms[i]);
            if ( system->reax_param.sbp[ atom->type ].p_hbond == 1 )
                atom->Hindex = system->numH++;
            else atom->Hindex = -1;
        }
    system->Hcap = MAX( system->numH * SAFER_ZONE, MIN_CAP );
*/

    //Sync_System( system );

    //Allocate_System( system, system->local_cap, system->total_cap, msg );

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
//    if( Reposition_Atoms( system, control, data, mpi_data, msg ) == FAILURE )
//    {
//        return FAILURE;
//    }

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
    {
        Generate_Initial_Velocities( system, control->T_init );
    }
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

    return SUCCESS;
}


#ifdef HAVE_CUDA
int Cuda_Init_System( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        mpi_datatypes *mpi_data, char *msg )
{
    int i, ret;
    reax_atom *atom;
    int nrecv[MAX_NBRS];

    Setup_New_Grid( system, control, MPI_COMM_WORLD );
    fprintf( stderr, "    [SETUP NEW GRID]\n" );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d GRID:\n", system->my_rank );
    Print_Grid( &(system->my_grid), stderr );
#endif

    Bin_My_Atoms( system, &(workspace->realloc) );
    fprintf( stderr, "    [BIN MY ATOMS]\n" );
    Reorder_My_Atoms( system, workspace );
    fprintf( stderr, "    [REORDER MY ATOMS]\n" );

    /* estimate N and total capacity */
    for ( i = 0; i < MAX_NBRS; ++i )
    {
        nrecv[i] = 0;
    }

    MPI_Barrier( MPI_COMM_WORLD );
    system->max_recved = 0;
    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
            Estimate_Boundary_Atoms, Unpack_Estimate_Message, TRUE );
    fprintf( stderr, "    [SEND_RECV:ESTIMATE_BOUNDARY_ATOMS]\n" );
    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );
    Bin_Boundary_Atoms( system );
    fprintf( stderr, "    [BIN_BOUNDARY_ATOMS]\n" );

    system->max_far_nbrs = (int*)
        scalloc( system->total_cap, sizeof(int), "system:max_far_nbrs" );
    system->max_bonds = (int*)
        scalloc( system->total_cap, sizeof(int), "system:max_bonds" );

    /* estimate numH and Hcap */
    system->numH = 0;
    if ( control->hbond_cut > 0.0 )
    {
        //TODO
        //for( i = 0; i < system->n; ++i ) {
        for ( i = 0; i < system->N; ++i )
        {
            atom = &(system->my_atoms[i]);
            atom->Hindex = i;
            //FIX - 4 - Added fix for HBond Issue
            if ( system->reax_param.sbp[ atom->type ].p_hbond == 1 )
            {
                system->numH++;
            }
            //else atom->Hindex = -1;
        }
    }
    system->Hcap = MAX( system->numH * SAFER_ZONE, MIN_CAP );

    /* Sync atoms here to continue the computation */
    dev_alloc_system( system );
    fprintf( stderr, "    [DEV ALLOC SYSTEM]\n" );
    Sync_System( system );
    fprintf( stderr, "    [SYNC SYSTEM]\n" );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d local_cap=%d\n",
             system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d total_cap=%d\n",
             system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d H_cap=%d\n",
             system->my_rank, system->numH, system->Hcap );
#endif

    Cuda_Compute_Total_Mass( system, data, mpi_data->comm_mesh3D );
    fprintf( stderr, "    [CUDA COMPUTE TOTAL MASS]\n" );
    Cuda_Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );
    fprintf( stderr, "    [CUDA COMPUTE CENTER OF MASS]\n" );
//    if( Reposition_Atoms( system, control, data, mpi_data, msg ) == FAILURE )
//    {
//        return FAILURE;
//    }

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
    {
        Generate_Initial_Velocities( system, control->T_init );
        fprintf( stderr, "    [GENERATE INITIAL VELOCITIES]\n" );
    }

    Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
    fprintf( stderr, "    [CUDA COMPUTE K.E.]\n" );

    return SUCCESS;
}
#endif


/************************ initialize simulation data ************************/
void Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data, char *msg )
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
        fprintf( stderr, "WARNING: Nose-Hoover NVT is still under testing.\n" );
        //return FAILURE;
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

    case sNPT: /* Semi-Isotropic NPT */
        data->N_f = 3 * system->bigN + 4;
        Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
            Reset_Pressures( data );
        break;

    case iNPT: /* Isotropic NPT */
        data->N_f = 3 * system->bigN + 2;
        Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
            Reset_Pressures( data );
        break;

    case NPT: /* Anisotropic NPT */
        strcpy( msg, "init_simulation_data: option not yet implemented" );
        return FAILURE;

        data->N_f = 3 * system->bigN + 9;
        Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        /*if( !control->restart ) {
          data->therm.G_xi = control->Tau_T *
          (2.0 * data->my_en.e_Kin - data->N_f * K_B * control->T );
          data->therm.v_xi = data->therm.G_xi * control->dt;
          data->iso_bar.eps = (1.0 / 3.0) * log(system->box.volume);
          data->inv_W = 1.0 /
          ( data->N_f * K_B * control->T * SQR(control->Tau_P) );
          Compute_Pressure( system, control, data, out_control );
          }*/
        break;

    default:
        strcpy( msg, "init_simulation_data: ensemble not recognized" );
        return FAILURE;
    }

    /* initialize the timer(s) */
    MPI_Barrier( MPI_COMM_WORLD );  // wait for everyone to come here
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


#ifdef HAVE_CUDA
void Cuda_Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data, char *msg )
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
        Cuda_Evolve = Velocity_Verlet_NVE;
        control->virial = 0;
        break;

    case bNVT:
        data->N_f = 3 * system->bigN + 1;
        Cuda_Evolve = Cuda_Velocity_Verlet_Berendsen_NVT;
        control->virial = 0;
        break;

    case nhNVT:
        fprintf( stderr, "WARNING: Nose-Hoover NVT is still under testing.\n" );
        //return FAILURE;
        data->N_f = 3 * system->bigN + 1;
        Cuda_Evolve = Velocity_Verlet_Nose_Hoover_NVT_Klein;
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
        Cuda_Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    case iNPT: /* Isotropic NPT */
        data->N_f = 3 * system->bigN + 2;
        Cuda_Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        if ( !control->restart )
        {
            Reset_Pressures( data );
        }
        break;

    case NPT: /* Anisotropic NPT */
        strcpy( msg, "init_simulation_data: option not yet implemented" );
        return FAILURE;

        data->N_f = 3 * system->bigN + 9;
        Cuda_Evolve = Velocity_Verlet_Berendsen_NPT;
        control->virial = 1;
        break;

    default:
        strcpy( msg, "init_simulation_data: ensemble not recognized" );
        return FAILURE;
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
#endif


#elif defined(LAMMPS_REAX)
int Init_System( reax_system *system, char *msg )
{
    system->big_box.V = 0;
    system->big_box.box_norms[0] = 0;
    system->big_box.box_norms[1] = 0;
    system->big_box.box_norms[2] = 0;

    system->local_cap = (int)(system->n * SAFE_ZONE);
    system->total_cap = (int)(system->N * SAFE_ZONE);

#if defined(DEBUG)
    fprintf( stderr, "p%d: local_cap=%d total_cap=%d\n",
             system->my_rank, system->local_cap, system->total_cap );
#endif

    Allocate_System( system, system->local_cap, system->total_cap, msg );

    return SUCCESS;
}


void Init_Simulation_Data( reax_system *system, control_params *control,
                          simulation_data *data, char *msg )
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
        fprintf( stderr, "Warning: non-zero lower Taper-radius cutoff\n" );
    }

    if ( swb < 0 )
    {
        fprintf( stderr, "Negative upper Taper-radius cutoff\n" );
        MPI_Abort( MPI_COMM_WORLD,  INVALID_INPUT );
    }
    else if ( swb < 5 )
    {
        fprintf( stderr, "Warning: very low Taper-radius cutoff: %f\n", swb );
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
        storage *workspace, char *msg )
{
    Allocate_Workspace( system, control, workspace, system->local_cap,
            system->total_cap, msg );

    memset( &(workspace->realloc), 0, sizeof(reallocate_data) );
    Reset_Workspace( system, workspace );

    /* Initialize the Taper function */
    Init_Taper( control, workspace );
}


#ifdef HAVE_CUDA
void Cuda_Init_Workspace( reax_system *system, control_params *control,
        storage *workspace, char *msg )
{
    dev_alloc_workspace( system, control, dev_workspace,
            system->local_cap, system->total_cap, msg );

    memset( &(workspace->realloc), 0, sizeof(reallocate_data) );
    Cuda_Reset_Workspace( system, workspace );

    /* Initialize the Taper function */
    Init_Taper( control, dev_workspace );
}
#endif


/************** setup communication data structures  **************/
int Init_MPI_Datatypes( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data, char *msg )
{
    int i, block[11];
    MPI_Aint base, disp[11];
    MPI_Datatype type[11];
    mpi_atom sample;
    boundary_atom b_sample;
    restart_atom r_sample;
    rvec rvec_sample;
    rvec2 rvec2_sample;

    /* setup the world */
    mpi_data->world = MPI_COMM_WORLD;

    /* allocate mpi buffers  */
    //Allocate_MPI_Buffers( mpi_data, system->est_recv,
    //              system->gcell_cap, system->my_nbrs, msg );
    //tmp = 0;
    //#if defined(DEBUG_FOCUS)
    //for( i = 0; i < MAX_NBRS; ++i )
    //if( i != MYSELF )
    //  tmp += system->my_nbrs[i].est_send;

    //fprintf( stderr, "p%d: allocated mpi_buffers: recv=%d send=%d total=%dMB\n",
    //   system->my_rank, system->est_recv, tmp,
    //   (int)((system->est_recv+tmp)*sizeof(boundary_atom)/(1024*1024)) );
    //#endif
    //if( ret != SUCCESS )
    //  return ret;

    /* mpi_atom - [orig_id, imprt_id, type, num_bonds, num_hbonds, name,
                   x, v, f_old, s, t] */
    block[0] = block[1] = block[2] = block[3] = block[4] = 1;
    block[5] = 8;
    block[6] = block[7] = block[8] = 3;
    block[9] = block[10] = 4;

    MPI_Address( &(sample.orig_id), disp + 0 );
    MPI_Address( &(sample.imprt_id), disp + 1 );
    MPI_Address( &(sample.type), disp + 2 );
    MPI_Address( &(sample.num_bonds), disp + 3 );
    MPI_Address( &(sample.num_hbonds), disp + 4 );
    MPI_Address( &(sample.name), disp + 5 );
    MPI_Address( &(sample.x[0]), disp + 6 );
    MPI_Address( &(sample.v[0]), disp + 7 );
    MPI_Address( &(sample.f_old[0]), disp + 8 );
    MPI_Address( &(sample.s[0]), disp + 9 );
    MPI_Address( &(sample.t[0]), disp + 10 );

    base = (MPI_Aint)(&(sample));
    for ( i = 0; i < 11; ++i )
    {
        disp[i] -= base;
    }

    type[0] = type[1] = type[2] = type[3] = type[4] = MPI_INT;
    type[5] = MPI_CHAR;
    type[6] = type[7] = type[8] = type[9] = type[10] = MPI_DOUBLE;

    MPI_Type_struct( 11, block, disp, type, &(mpi_data->mpi_atom_type) );
    MPI_Type_commit( &(mpi_data->mpi_atom_type) );

    /* boundary_atom - [orig_id, imprt_id, type, num_bonds, num_hbonds, x] */
    block[0] = block[1] = block[2] = block[3] = block[4] = 1;
    block[5] = 3;

    MPI_Address( &(b_sample.orig_id), disp + 0 );
    MPI_Address( &(b_sample.imprt_id), disp + 1 );
    MPI_Address( &(b_sample.type), disp + 2 );
    MPI_Address( &(b_sample.num_bonds), disp + 3 );
    MPI_Address( &(b_sample.num_hbonds), disp + 4 );
    MPI_Address( &(b_sample.x[0]), disp + 5 );

    base = (MPI_Aint)(&(b_sample));
    for ( i = 0; i < 6; ++i )
    {
        disp[i] -= base;
    }

    type[0] = type[1] = type[2] = type[3] = type[4] = MPI_INT;
    type[5] = MPI_DOUBLE;

    MPI_Type_struct( 6, block, disp, type, &(mpi_data->boundary_atom_type) );
    MPI_Type_commit( &(mpi_data->boundary_atom_type) );

    /* mpi_rvec */
    block[0] = 3;
    MPI_Address( &(rvec_sample[0]), disp + 0 );
    base = disp[0];
    for ( i = 0; i < 1; ++i )
    {
        disp[i] -= base;
    }
    type[0] = MPI_DOUBLE;
    MPI_Type_struct( 1, block, disp, type, &(mpi_data->mpi_rvec) );
    MPI_Type_commit( &(mpi_data->mpi_rvec) );

    /* mpi_rvec2 */
    block[0] = 2;
    MPI_Address( &(rvec2_sample[0]), disp + 0 );
    base = disp[0];
    for ( i = 0; i < 1; ++i )
    {
        disp[i] -= base;
    }
    type[0] = MPI_DOUBLE;
    MPI_Type_struct( 1, block, disp, type, &(mpi_data->mpi_rvec2) );
    MPI_Type_commit( &(mpi_data->mpi_rvec2) );

    /* restart_atom - [orig_id, type, name[8], x, v] */
    block[0] = block[1] = 1 ;
    block[2] = 8;
    block[3] = block[4] = 3;

    MPI_Address( &(r_sample.orig_id), disp + 0 );
    MPI_Address( &(r_sample.type), disp + 1 );
    MPI_Address( &(r_sample.name), disp + 2 );
    MPI_Address( &(r_sample.x[0]), disp + 3 );
    MPI_Address( &(r_sample.v[0]), disp + 4 );

    base = (MPI_Aint)(&(r_sample));
    for ( i = 0; i < 5; ++i )
    {
        disp[i] -= base;
    }

    type[0] = type[1] = MPI_INT;
    type[2] = MPI_CHAR;
    type[3] = type[4] = MPI_DOUBLE;

    MPI_Type_struct( 5, block, disp, type, &(mpi_data->restart_atom_type) );
    MPI_Type_commit( &(mpi_data->restart_atom_type) );

    return SUCCESS;
}


/********************** allocate lists *************************/
int Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data, char *msg )
{
    int i, num_nbrs;
    int total_hbonds, total_bonds, bond_cap, num_3body, cap_3body, Htop;
    int *hb_top, *bond_top;

    //for( i = 0; i < MAX_NBRS; ++i ) nrecv[i] = system->my_nbrs[i].est_recv;
    //system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
    //        Sort_Boundary_Atoms, Unpack_Exchange_Message, TRUE );

    num_nbrs = Estimate_NumNeighbors( system, lists );
    Make_List( system->total_cap, num_nbrs, TYP_FAR_NEIGHBOR, *lists + FAR_NBRS );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated far_nbrs: num_far=%d, space=%dMB\n",
            system->my_rank, num_nbrs,
            (int)(num_nbrs * sizeof(far_neighbor_data) / (1024 * 1024)) );
#endif

    Generate_Neighbor_Lists( system, data, workspace, lists );
    bond_top = (int*) calloc( system->total_cap, sizeof(int) );
    hb_top = (int*) calloc( system->local_cap, sizeof(int) );
//    hb_top = (int*) calloc( system->Hcap, sizeof(int) );
    
    Estimate_Storages( system, control, lists,
            &Htop, hb_top, bond_top, &num_3body );
//    Host_Estimate_Sparse_Matrix( system, control, lists, system->local_cap, system->total_cap,
//            &Htop, hb_top, bond_top, &num_3body );
    
    Allocate_Matrix( &(workspace->H), system->local_cap, Htop );
    
    //MATRIX CHANGES
    //workspace->L = NULL;
    //workspace->U = NULL;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated H matrix: Htop=%d, space=%dMB\n",
             system->my_rank, Htop,
             (int)(Htop * sizeof(sparse_matrix_entry) / (1024 * 1024)) );
#endif

    if ( control->hbond_cut > 0.0 )
    {
        // init H indexes
        total_hbonds = 0;
        for ( i = 0; i < system->n; ++i )
        {
            system->my_atoms[i].num_hbonds = hb_top[i];
            total_hbonds += hb_top[i];
        }
        total_hbonds = MAX( total_hbonds * SAFER_ZONE, MIN_CAP * MIN_HBONDS );
        // DANIEL, to make Mpi_Not_Gpu_Validate_Lists() not complain that max_hbonds is 0
        system->max_hbonds = total_hbonds * SAFER_ZONE;

        Make_List( system->Hcap, total_hbonds, TYP_HBOND, *lists + HBONDS );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: allocated hbonds: total_hbonds=%d, space=%dMB\n",
                 system->my_rank, total_hbonds,
                 (int)(total_hbonds * sizeof(hbond_data) / (1024 * 1024)) );
#endif
    }

    /* bonds list */
    //Allocate_Bond_List( system->N, bond_top, (*lists)+BONDS );
    //num_bonds = bond_top[system->N-1];
    total_bonds = 0;
    for ( i = 0; i < system->N; ++i )
    {
        system->my_atoms[i].num_bonds = bond_top[i];
        total_bonds += bond_top[i];
        // DANIEL, to make Mpi_Not_Gpu_Validate_Lists() not complain that max_bonds is 0
        system->max_bonds[i] = MAX( bond_top[i], MIN_BONDS );
    }
    bond_cap = MAX( total_bonds * SAFE_ZONE, MIN_CAP * MIN_BONDS );

    Make_List( system->total_cap, bond_cap, TYP_BOND, *lists + BONDS);

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated bonds: total_bonds=%d, space=%dMB\n",
             system->my_rank, bond_cap,
             (int)(bond_cap * sizeof(bond_data) / (1024 * 1024)) );
#endif

    /* 3bodies list */
    cap_3body = MAX( num_3body * SAFE_ZONE, MIN_3BODIES );
    Make_List(bond_cap, cap_3body, TYP_THREE_BODY, *lists + THREE_BODIES);

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated 3-body list: num_3body=%d, space=%dMB\n",
             system->my_rank, cap_3body,
             (int)(cap_3body * sizeof(three_body_interaction_data) / (1024 * 1024)) );
#endif

#if defined(TEST_FORCES)
    Make_List(system->total_cap, bond_cap * 8, TYP_DDELTA, (*lists) + DDELTAS);

    fprintf( stderr, "p%d: allocated dDelta list: num_ddelta=%d space=%ldMB\n",
             system->my_rank, bond_cap * 30,
             bond_cap * 8 * sizeof(dDelta_data) / (1024 * 1024) );

    Make_List( bond_cap, bond_cap * 50, TYP_DBO, (*lists) + DBOS);

    fprintf( stderr, "p%d: allocated dbond list: num_dbonds=%d space=%ldMB\n",
             system->my_rank, bond_cap * MAX_BONDS * 3,
             bond_cap * MAX_BONDS * 3 * sizeof(dbond_data) / (1024 * 1024) );
#endif

    free( hb_top );
    free( bond_top );

    return SUCCESS;
}


#ifdef HAVE_CUDA
int Cuda_Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data, char *msg )
{
    int i, count, ret;
    int num_nbrs, total_hbonds, total_bonds, total_3body, Htop;
    int *nbr_indices, *hb_top, *bond_top, *thbody;
   
    nbr_indices = (int *) host_scratch;
    bond_top = (int*) calloc( system->total_cap, sizeof(int) );
    hb_top = (int*) calloc( system->total_cap, sizeof(int) );

    for ( i = 0; i < system->total_cap; i++ )
    {
        system->max_far_nbrs[i] = MIN_NBRS;
    }

    /* ignore returned error, as system->max_far_nbrs is not yet set */
    ret = Cuda_Estimate_Neighbors( system, nbr_indices );

    /* count neighbors for list creation */
    num_nbrs = 0;
    for (i = 0; i < system->total_cap; i++)
    {
        num_nbrs += system->max_far_nbrs[i];
        nbr_indices[i] = system->max_far_nbrs[i];
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "DEVICE total neighbors entries: %d \n", num_nbrs );
#endif

    Dev_Make_List( system->total_cap, num_nbrs, TYP_FAR_NEIGHBOR, *dev_lists + FAR_NBRS );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated far_nbrs: num_far=%d, space=%dMB\n",
            system->my_rank, num_nbrs,
            (int)(num_nbrs * sizeof(far_neighbor_data) / (1024 * 1024)) );
    fprintf( stderr, "N: %d and total_cap: %d \n", system->N, system->total_cap );
#endif

    Cuda_Init_Neighbor_Indices( nbr_indices, system->total_cap );

    Cuda_Generate_Neighbor_Lists( system, data, workspace, dev_lists );

    Cuda_Estimate_Storages( system, control, dev_lists, &Htop,
            hb_top, bond_top );

    //TODO - CARVER FIX

    Cuda_Estimate_Sparse_Matrix( system, control, data, dev_lists );

    dev_alloc_matrix( &(dev_workspace->H), system->total_cap,
            system->total_cap * system->max_sparse_entries );
    dev_workspace->H.n = system->n;

    //THIS IS INITIALIZED in the init_forces function to system->n
    //but this is never used in the code.
    //GPU maintains the H matrix to be (NXN) symmetric matrix.

    //TODO - CARVER FIX

    //MATRIX CHANGES
    //workspace->L = NULL;
    //workspace->U = NULL;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p:%d - allocated H matrix: max_entries: %d, cap: %d \n",
            system->my_rank, system->max_sparse_entries, dev_workspace->H.m );
    fprintf( stderr, "p%d: allocated H matrix: Htop=%d, space=%dMB\n",
            system->my_rank, Htop,
            (int)(Htop * sizeof(sparse_matrix_entry) / (1024 * 1024)) );
#endif

    // FIX - 4 - Added addition check here for hydrogen Bonds
    if ( control->hbond_cut > 0.0  &&  system->numH > 0 )
    {
        /* init H indexes */
        total_hbonds = 0;
        count = 0;

        for ( i = 0; i < system->N; ++i )
        {
            //system->my_atoms[i].num_hbonds = hb_top[i];
            //TODO
            hb_top[i] = MAX( hb_top[i] * 4, MIN_HBONDS * 4);
            total_hbonds += hb_top[i];
            if ( hb_top[i] > 0 )
            {
                ++count;
            }
        }
        total_hbonds = MAX( total_hbonds, MIN_CAP * MIN_HBONDS );

        Dev_Make_List( system->total_cap, system->total_cap *
                system->max_hbonds, TYP_HBOND, *dev_lists + HBONDS );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "**** Total HBonds allocated --> %d total_cap: %d per atom: %d, max_hbonds: %d \n",
                total_hbonds, system->total_cap, (total_hbonds /
                    system->total_cap), system->max_hbonds );
#endif

        //TODO
        //Cuda_Init_HBond_Indices (hb_top, system->n);
        /****/
        //THIS IS COMMENTED OUT - CHANGE ORIGINAL
        //Cuda_Init_HBond_Indices (hb_top, system->N);
        //THIS IS COMMENTED OUT - CHANGE ORIGINAL
        /****/

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: allocated hbonds: total_hbonds=%d, space=%dMB\n",
                system->my_rank, total_hbonds,
                (int)(total_hbonds * sizeof(hbond_data) / (1024 * 1024)) );
#endif
    }

    /* bonds list */
    total_bonds = 0;
    for ( i = 0; i < system->total_cap; ++i )
    {
        total_bonds += system->max_bonds[i];
        bond_top[i] = system->max_bonds[i];
    }

    Dev_Make_List( system->total_cap, total_bonds, TYP_BOND, *dev_lists + BONDS );
    Make_List( system->total_cap, total_bonds, TYP_BOND, *lists + BONDS );

    Cuda_Init_Bond_Indices( bond_top, system->total_cap );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated bonds: total_bonds=%d, space=%dMB\n",
            system->my_rank, total_bonds,
            (int)(total_bonds * sizeof(bond_data) / (1024 * 1024)) );
#endif

    /* 3bodies list */
    thbody = (int *) host_scratch;
    memset( thbody, 0, sizeof(int) * (*dev_lists + BONDS)->num_intrs );

    Cuda_Estimate_Storages_Three_Body( system, control, dev_lists,
            &total_3body, thbody );

    Dev_Make_List( (*dev_lists + BONDS)->num_intrs, total_3body,
            TYP_THREE_BODY, (*dev_lists + THREE_BODIES) );
//    Make_List( (*lists + BONDS)->num_intrs, total_3body,
//            TYP_THREE_BODY, (*lists + THREE_BODIES) );

    Cuda_Init_Three_Body_Indices( thbody, (*dev_lists + BONDS)->num_intrs );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated 3-body list: total_3body=%d, space=%dMB\n",
            system->my_rank, total_3body,
            (int)(total_3body * sizeof(three_body_interaction_data) / (1024 * 1024)) );
#endif

    free( hb_top );
    free( bond_top );

    return SUCCESS;
}
#endif


#if defined(PURE_REAX)
void Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{

    host_scratch = (void *)malloc( HOST_SCRATCH_SIZE );

    char msg[MAX_STR];

    if ( Init_MPI_Datatypes( system, workspace, mpi_data, msg ) == FAILURE )
    {
        fprintf( stderr, "p%d: init_mpi_datatypes: could not create datatypes\n",
                 system->my_rank );
        fprintf( stderr, "p%d: mpi_data couldn't be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized mpi datatypes\n", system->my_rank );
#endif

    if ( Init_System(system, control, data, workspace, mpi_data, msg) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: system initialized\n", system->my_rank );
#endif

    Init_Simulation_Data( system, control, data, msg );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized simulation data\n", system->my_rank );
#endif

    Init_Workspace( system, control, workspace, msg );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized workspace\n", system->my_rank );
#endif

    if ( Init_Lists( system, control, data, workspace, lists, mpi_data, msg ) ==
            FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized lists\n", system->my_rank );
#endif

    if (Init_Output_Files(system, control, out_control, mpi_data, msg) == FAILURE)
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: could not open output files! terminating...\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: output files opened\n", system->my_rank );
#endif

    if ( control->tabulate )
    {
        if ( Init_Lookup_Tables(system, control, workspace->Tap, mpi_data, msg) == FAILURE )
        {
            fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
            fprintf( stderr, "p%d: couldn't create lookup table! terminating.\n",
                     system->my_rank );
            MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
        }

#if defined(DEBUG)
        fprintf( stderr, "p%d: initialized lookup tables\n", system->my_rank );
#endif
    }

    Init_Force_Functions( control );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized force functions\n", system->my_rank );
#endif

    /*#ifdef TEST_FORCES
      Init_Force_Test_Functions();
      fprintf(stderr,"p%d: initialized force test functions\n",system->my_rank);
      #endif */
}


void Pure_Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    char msg[MAX_STR];

    Init_Simulation_Data( system, control, data, msg );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized simulation data\n", system->my_rank );
#endif
    fprintf( stderr, "p%d: pure initialized simulation data\n", system->my_rank );

    Init_Workspace( system, control, workspace, msg );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized workspace\n", system->my_rank );
#endif
    fprintf( stderr, "p%d: pure initialized workspace\n", system->my_rank );

    if ( Init_Lists( system, control, data, workspace, lists, mpi_data, msg ) ==
            FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }
#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized lists\n", system->my_rank );
#endif
    fprintf( stderr, "p%d: pure initialized lists done \n", system->my_rank );

    Init_Force_Functions( control );
}


#ifdef HAVE_CUDA
void Cuda_Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    char msg[MAX_STR];
    real t_start, t_end;

    /* HOST/DEVICE SCRATCH */
    Cuda_Init_ScratchArea( );

    /* MPI_DATATYPES */
    if ( Init_MPI_Datatypes( system, workspace, mpi_data, msg ) == FAILURE )
    {
        fprintf( stderr, "p%d: init_mpi_datatypes: could not create datatypes\n",
                 system->my_rank );
        fprintf( stderr, "p%d: mpi_data couldn't be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }
    fprintf( stderr, "  [INIT MPI DATATYPES]\n" );

    /* SYSTEM */
    if ( Cuda_Init_System( system, control, data, workspace, mpi_data, msg ) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }
    fprintf( stderr, "  [CUDA INIT SYSTEM]\n" );

    /* GRID */
    dev_alloc_grid( system );
    Sync_Grid( &system->my_grid, &system->d_my_grid );
    fprintf( stderr, "  [DEV ALLOC GRID]\n" );

    //validate_grid( system );

    /* SIMULATION_DATA */
    Cuda_Init_Simulation_Data( system, control, data, msg );
    fprintf( stderr, "  [CUDA INIT SIMULATION DATA]\n" );

    /* WORKSPACE */
    Cuda_Init_Workspace( system, control, workspace, msg );
    fprintf( stderr, "  [CUDA INIT WORKSPACE]\n" );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized workspace\n", system->my_rank );
#endif

    //Sync the taper here from host to device.

    /* CONTROL */
    dev_alloc_control( control );
    fprintf( stderr, "  [DEV ALLOC CONTROL]\n" );

    /* LISTS */
    if ( Cuda_Init_Lists( system, control, data, workspace, lists, mpi_data, msg ) ==
            FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }
    fprintf( stderr, "  [CUDA INIT LISTS]\n" );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized lists\n", system->my_rank );
#endif

    /* OUTPUT Files */
    if ( Init_Output_Files( system, control, out_control, mpi_data, msg ) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: could not open output files! terminating...\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }
    fprintf( stderr, "  [INIT OUTPUT FILES]\n" );

#if defined(DEBUG)
    fprintf( stderr, "p%d: output files opened\n", system->my_rank );
#endif

    /* Lookup Tables */
    if ( control->tabulate )
    {
        if ( Init_Lookup_Tables( system, control, dev_workspace->Tap, mpi_data, msg ) == FAILURE )
        {
            fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
            fprintf( stderr, "p%d: couldn't create lookup table! terminating.\n",
                     system->my_rank );
            MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
        }
        fprintf( stderr, "  [INIT LOOKUP TABLES]\n" );

#if defined(DEBUG)
        fprintf( stderr, "p%d: initialized lookup tables\n", system->my_rank );
#endif
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: Device Initialization Done \n", system->my_rank );
#endif
}
#endif


#elif defined(LAMMPS_REAX)
void Initialize( reax_system *system, control_params *control,
                 simulation_data *data, storage *workspace,
                 reax_list **lists, output_controls *out_control,
                 mpi_datatypes *mpi_data )
{
    char msg[MAX_STR];

    host_scratch = (void *)malloc( HOST_SCRATCH_SIZE );

    if ( Init_System(system, msg) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: system initialized\n", system->my_rank );
#endif

    Init_Simulation_Data( system, control, data, msg );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized simulation data\n", system->my_rank );
#endif

    Init_Workspace( system, control, workspace, msg );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized workspace\n", system->my_rank );
#endif

    if ( Init_MPI_Datatypes( system, workspace, mpi_data, msg ) == FAILURE )
    {
        fprintf( stderr, "p%d: init_mpi_datatypes: could not create datatypes\n",
                 system->my_rank );
        fprintf( stderr, "p%d: mpi_data couldn't be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized mpi datatypes\n", system->my_rank );
#endif

    if ( Init_Lists( system, control, workspace, lists, msg ) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized lists\n", system->my_rank );
#endif

    if ( Init_Output_Files(system, control, out_control, mpi_data, msg) == FAILURE)
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: could not open output files! terminating...\n",
                 system->my_rank );
        MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
    }

#if defined(DEBUG)
    fprintf( stderr, "p%d: output files opened\n", system->my_rank );
#endif

    if ( control->tabulate )
    {
        if ( Init_Lookup_Tables( system, control, workspace->Tap, mpi_data, msg ) == FAILURE )
        {
            fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
            fprintf( stderr, "p%d: couldn't create lookup table! terminating.\n",
                     system->my_rank );
            MPI_Abort( MPI_COMM_WORLD, CANNOT_INITIALIZE );
        }

#if defined(DEBUG)
        fprintf( stderr, "p%d: initialized lookup tables\n", system->my_rank );
#endif
    }

    Init_Force_Functions( );

#if defined(DEBUG)
    fprintf( stderr, "p%d: initialized force functions\n", system->my_rank );
#endif

    /*#if defined(TEST_FORCES)
      Init_Force_Test_Functions();
      fprintf(stderr,"p%d: initialized force test functions\n",system->my_rank);
#endif*/
    }
#endif
