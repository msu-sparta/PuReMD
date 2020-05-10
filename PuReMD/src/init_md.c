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
        simulation_data *data, mpi_datatypes *mpi_data,
        char *msg )
{
    int   i;
    rvec  dx;

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
            scale = sqrt( m * norm / (3.0 * K_B * T) );

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

    Setup_New_Grid( system, control, mpi_data->world );
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d GRID:\n", system->my_rank );
    Print_Grid( &(system->my_grid), stderr );
#endif
    Bin_My_Atoms( system, &(workspace->realloc) );
    Reorder_My_Atoms( system, workspace );

    //fprintf( stderr, "p%d LOC LOC LOC!\n", system->my_rank );
    //MPI_Barrier( mpi_data->world );

    /* estimate N and total capacity */
    for ( i = 0; i < MAX_NBRS; ++i ) nrecv[i] = 0;
    system->max_recved = 0;
    system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
            Estimate_Boundary_Atoms, Unpack_Estimate_Message, 1 );
    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );
    Bin_Boundary_Atoms( system );
#if defined(NEUTRAL_TERRITORY)
    Estimate_NT_Atoms( system, mpi_data );
#endif

    //fprintf( stderr, "p%d SEND RECV SEND!\n", system->my_rank );
    //MPI_Barrier( mpi_data->world );

    /* estimate numH and Hcap */
    system->numH = 0;
    if ( control->hbond_cut > 0 )
        for ( i = 0; i < system->n; ++i )
        {
            atom = &(system->my_atoms[i]);
            if ( system->reax_param.sbp[ atom->type ].p_hbond == 1 )
                atom->Hindex = system->numH++;
            else atom->Hindex = -1;
        }
    system->Hcap = MAX( system->numH * SAFER_ZONE, MIN_CAP );

    //fprintf( stderr, "p%d HCAP HCAP HCAP!\n", system->my_rank );
    //MPI_Barrier( mpi_data->world );

    //Allocate_System( system, system->local_cap, system->total_cap, msg );
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d local_cap=%d\n",
            system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d total_cap=%d\n",
            system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d H_cap=%d\n",
            system->my_rank, system->numH, system->Hcap );
    MPI_Barrier( mpi_data->world );
#endif

    // if( Reposition_Atoms( system, control, data, mpi_data, msg ) == FAILURE )
    //   return FAILURE;

    /* initialize velocities so that desired init T can be attained */
    if ( !control->restart || (control->restart && control->random_vel) )
        Generate_Initial_Velocities( system, control->T_init );

    return SUCCESS;
}


/************************ initialize simulation data ************************/
int Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data, mpi_datatypes *mpi_data,
        evolve_function *Evolve, char *msg )
{
    Reset_Simulation_Data( data, control->virial );

    if ( !control->restart )
    {
        data->step = 0;
        data->prev_steps = 0;
        data->last_pc_step = 0;
        data->refactor = TRUE;
    }

    Compute_Total_Mass( system, data, mpi_data->comm_mesh3D );
    Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

    switch ( control->ensemble )
    {
        case NVE:
            data->N_f = 3 * system->bigN;
            *Evolve = &Velocity_Verlet_NVE;
            break;

        case bNVT:
            data->N_f = 3 * system->bigN + 1;
            *Evolve = &Velocity_Verlet_Berendsen_NVT;
            break;

        case nhNVT:
            fprintf( stderr, "WARNING: Nose-Hoover NVT is still under testing.\n" );
            //return FAILURE;
            data->N_f = 3 * system->bigN + 1;
            *Evolve = &Velocity_Verlet_Nose_Hoover_NVT_Klein;
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
            *Evolve = &Velocity_Verlet_Berendsen_NPT;
            if ( !control->restart )
                Reset_Pressures( data );
            break;

        case iNPT: /* Isotropic NPT */
            data->N_f = 3 * system->bigN + 2;
            *Evolve = &Velocity_Verlet_Berendsen_NPT;
            if ( !control->restart )
                Reset_Pressures( data );
            break;

        case NPT: /* Anisotropic NPT */
            strcpy( msg, "init_simulation_data: option not yet implemented" );
            return FAILURE;

            data->N_f = 3 * system->bigN + 9;
            *Evolve = &Velocity_Verlet_Berendsen_NPT;
            /*if( !control->restart ) {
              data->therm.G_xi = control->Tau_T *
              (2.0 * data->my_en.e_Kin - data->N_f * K_B * control->T );
              data->therm.v_xi = data->therm.G_xi * control->dt;
              data->iso_bar.eps = 0.33333 * log(system->box.volume);
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
    MPI_Barrier( mpi_data->world );  // wait for everyone to come here
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.start = MPI_Wtime();
#if defined(LOG_PERFORMANCE)
        //Reset_Timing( &data->timing );
        /* init timing info */
        data->timing.total = data->timing.start;
        data->timing.comm = ZERO;
        data->timing.nbrs = 0;
        data->timing.init_forces = 0;
        data->timing.bonded = 0;
        data->timing.nonb = 0;
        data->timing.init_dist = ZERO;
        data->timing.init_cm = ZERO;
        data->timing.init_bond = ZERO;
        data->timing.cm = ZERO;
        data->timing.cm_sort = ZERO;
        data->timing.cm_solver_comm = ZERO;
        data->timing.cm_solver_allreduce = ZERO;
        data->timing.cm_solver_pre_comp = ZERO;
        data->timing.cm_solver_pre_app = ZERO;
        data->timing.cm_solver_iters = 0;
        data->timing.cm_solver_spmv = ZERO;
        data->timing.cm_solver_vector_ops = ZERO;
        data->timing.cm_solver_orthog = ZERO;
        data->timing.cm_solver_tri_solve = ZERO;
        data->timing.cm_last_pre_comp = ZERO;
        data->timing.cm_total_loss = ZERO;
        data->timing.cm_optimum = ZERO;
#endif
    }


#if defined(DEBUG_FOCUS)
    fprintf( stderr, "data->N_f: %8.3f\n", data->N_f );
#endif
    return SUCCESS;
}

#elif defined(LAMMPS_REAX)
int Init_System( reax_system *system, control_params *control, char *msg )
{
    int i;
    reax_atom *atom;

    /* determine the local and total capacity */
    system->local_cap = MAX( (int)(system->n * SAFE_ZONE), MIN_CAP );
    system->total_cap = MAX( (int)(system->N * SAFE_ZONE), MIN_CAP );

    /* estimate numH and Hcap */
    system->numH = 0;
    if ( control->hbond_cut > 0 )
        for ( i = 0; i < system->n; ++i )
        {
            atom = &(system->my_atoms[i]);
            if ( system->reax_param.sbp[ atom->type ].p_hbond == 1 )
                atom->Hindex = system->numH++;
            else atom->Hindex = -1;
        }
    system->Hcap = (int)(MAX( system->numH * SAFER_ZONE, MIN_CAP ));

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: n=%d local_cap=%d\n",
            system->my_rank, system->n, system->local_cap );
    fprintf( stderr, "p%d: N=%d total_cap=%d\n",
            system->my_rank, system->N, system->total_cap );
    fprintf( stderr, "p%d: numH=%d H_cap=%d\n",
            system->my_rank, system->numH, system->Hcap );
#endif

    return SUCCESS;
}


int Init_Simulation_Data( reax_system *system, control_params *control,
        simulation_data *data, evolve_function *Evolve, char *msg )
{
    Reset_Simulation_Data( data, control->virial );

    /* initialize the timer(s) */
    if ( system->my_rank == MASTER_NODE )
    {
        data->timing.start = MPI_Wtime();
#if defined(LOG_PERFORMANCE)
        //Reset_Timing( &data->timing );
        /* init timing info */
        data->timing.total = data->timing.start;
        data->timing.comm = ZERO;
        data->timing.nbrs = 0;
        data->timing.init_forces = 0;
        data->timing.bonded = 0;
        data->timing.nonb = 0;
        data->timing.init_dist = ZERO;
        data->timing.init_cm = ZERO;
        data->timing.init_bond = ZERO;
        data->timing.cm = ZERO;
        data->timing.cm_sort = ZERO;
        data->timing.cm_solver_comm = ZERO;
        data->timing.cm_solver_allreduce = ZERO;
        data->timing.cm_solver_pre_comp = ZERO;
        data->timing.cm_solver_pre_app = ZERO;
        data->timing.cm_solver_iters = 0;
        data->timing.cm_solver_spmv = ZERO;
        data->timing.cm_solver_vector_ops = ZERO;
        data->timing.cm_solver_orthog = ZERO;
        data->timing.cm_solver_tri_solve = ZERO;
#endif
    }

    //if( !control->restart )
    data->step = 0;
    data->prev_steps = 0;
    data->last_pc_step = 0;
    data->refactor = TRUE;

    return SUCCESS;
}
#endif



/************************ initialize workspace ************************/
/* Initialize Taper params */
void Init_Taper( control_params *control,  storage *workspace, MPI_Comm comm )
{
    real d1, d7;
    real swa, swa2, swa3;
    real swb, swb2, swb3;

    swa = control->nonb_low;
    swb = control->nonb_cut;

    if ( fabs( swa ) > 0.01 )
    {
        fprintf( stderr, "[WARNING] non-zero value for lower Taper-radius cutoff (%f)\n", swa );
    }

    if ( swb < 0.0 )
    {
        fprintf( stderr, "[ERROR] Negative value for upper Taper-radius cutoff\n" );
        MPI_Abort( comm,  INVALID_INPUT );
    }
    else if ( swb < 5.0 )
    {
        fprintf( stderr, "[WARNING] very low Taper-radius cutoff (%f)\n", swb );
    }

    d1 = swb - swa;
    d7 = POW( d1, 7.0 );
    swa2 = SQR( swa );
    swa3 = swa2 * swa;
    swb2 = SQR( swb );
    swb3 = swb2 * swb;

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


int Init_Workspace( reax_system *system, control_params *control,
        storage *workspace, MPI_Comm comm, char *msg )
{
    int ret;

    ret = Allocate_Workspace( system, control, workspace,
            system->local_cap, system->total_cap, comm, msg );
    if ( ret != SUCCESS )
        return ret;

    workspace->H = NULL;
    workspace->H_full = NULL;
    workspace->H_sp = NULL;
    workspace->H_p = NULL;
    workspace->H_spar_patt = NULL;
    workspace->H_spar_patt_full = NULL;
    workspace->H_app_inv = NULL;
    workspace->L = NULL;
    workspace->U = NULL;

    memset( &workspace->realloc, 0, sizeof(reallocate_data) );
    Reset_Workspace( system, workspace );

    /* Initialize the Taper function */
    Init_Taper( control, workspace, comm );

    return SUCCESS;
}


/************** setup communication data structures  **************/
int Init_MPI_Datatypes( reax_system *system, storage *workspace,
        mpi_datatypes *mpi_data, MPI_Comm comm, char *msg )
{
    int i, block[11];
    MPI_Aint disp[11], base;
    MPI_Datatype type[11], temp_type;
    mpi_atom sample[1];
    boundary_atom b_sample[1];
    restart_atom r_sample[1];

    /* mpi_atom */
    block[0] = 1;
    block[1] = 1;
    block[2] = 1;
    block[3] = 1;
    block[4] = 1;
    block[5] = sizeof(sample[0].name) / sizeof(char);
    block[6] = 3;
    block[7] = 3;
    block[8] = 3;
    block[9] = 4;
    block[10] = 4;

    MPI_Get_address( &sample[0], &base );
    MPI_Get_address( &sample[0].orig_id, &disp[0] );
    MPI_Get_address( &sample[0].imprt_id, &disp[1] );
    MPI_Get_address( &sample[0].type, &disp[2] );
    MPI_Get_address( &sample[0].num_bonds, &disp[3] );
    MPI_Get_address( &sample[0].num_hbonds, &disp[4] );
    MPI_Get_address( &sample[0].name, &disp[5] );
    MPI_Get_address( &sample[0].x, &disp[6] );
    MPI_Get_address( &sample[0].v, &disp[7] );
    MPI_Get_address( &sample[0].f_old, &disp[8] );
    MPI_Get_address( &sample[0].s, &disp[9] );
    MPI_Get_address( &sample[0].t, &disp[10] );
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

    MPI_Type_create_struct( 11, block, disp, type, &temp_type );
    MPI_Type_create_resized( temp_type, 0, sizeof(mpi_atom),
            &mpi_data->mpi_atom_type );
    MPI_Type_commit( &mpi_data->mpi_atom_type );

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

    MPI_Type_create_struct( 6, block, disp, type, &temp_type );
    MPI_Type_create_resized( temp_type, 0, sizeof(boundary_atom),
            &mpi_data->boundary_atom_type );
    MPI_Type_commit( &mpi_data->boundary_atom_type );

    /* mpi_rvec */
    MPI_Type_contiguous( 3, MPI_DOUBLE, &mpi_data->mpi_rvec );
    MPI_Type_commit( &mpi_data->mpi_rvec );

    /* mpi_rvec2 */
    MPI_Type_contiguous( 2, MPI_DOUBLE, &mpi_data->mpi_rvec2 );
    MPI_Type_commit( &mpi_data->mpi_rvec2 );

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

    MPI_Type_create_struct( 5, block, disp, type, &temp_type );
    MPI_Type_create_resized( temp_type, 0, sizeof(restart_atom),
            &mpi_data->restart_atom_type );
    MPI_Type_commit( &mpi_data->restart_atom_type );

    mpi_data->in1_buffer = NULL;
    mpi_data->in2_buffer = NULL;

    for ( i = 0; i < MAX_NBRS; ++i )
    {
        mpi_data->out_buffers[i].cnt = 0;
        mpi_data->out_buffers[i].index = NULL;
        mpi_data->out_buffers[i].out_atoms = NULL;
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
        mpi_data->out_nt_buffers[i].out_atoms = NULL;
    }
#endif

    /* setup the world */
    mpi_data->world = comm;
    MPI_Comm_size( comm, &system->wsize );

    return SUCCESS;
}


/********************** allocate lists *************************/
#if defined(PURE_REAX)
int  Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data, char *msg )
{
    int i, num_nbrs, far_nbr_list_format, cm_format, matrix_dim;
    int total_hbonds, total_bonds, bond_cap, num_3body, cap_3body, Htop;
    int *hb_top, *bond_top;
    MPI_Comm comm;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: before est_nbrs - local_cap=%d, total_cap=%d\n",
            system->my_rank, system->local_cap, system->total_cap );
#endif

    comm = mpi_data->world;

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

    //for( i = 0; i < MAX_NBRS; ++i ) nrecv[i] = system->my_nbrs[i].est_recv;
    //system->N = SendRecv( system, mpi_data, mpi_data->boundary_atom_type, nrecv,
    //        Sort_Boundary_Atoms, Unpack_Exchange_Message, 1 );

    num_nbrs = Estimate_NumNeighbors( system, lists, far_nbr_list_format );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: after est_nbrs - local_cap=%d, total_cap=%d\n",
            system->my_rank, system->local_cap, system->total_cap );
#endif

    if ( !Make_List( system->total_cap, num_nbrs, TYP_FAR_NEIGHBOR,
                far_nbr_list_format, lists[FAR_NBRS], comm ) )
    {
        fprintf(stderr, "Problem in initializing far nbrs list. Terminating!\n");
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated far_nbrs: num_far=%d, space=%dMB\n",
            system->my_rank, num_nbrs,
            (int)(num_nbrs * sizeof(far_neighbor_data) / (1024 * 1024)) );
#endif

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: before gen_nbrs - local_cap=%d, total_cap=%d\n",
            system->my_rank, system->local_cap, system->total_cap );
#endif

    Generate_Neighbor_Lists( system, data, workspace, lists );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: after gen_nbrs - local_cap=%d, total_cap=%d\n",
            system->my_rank, system->local_cap, system->total_cap );
#endif

    bond_top = scalloc( system->total_cap, sizeof(int),
            "Init_Lists::bond_top", comm );
    hb_top = scalloc( system->local_cap, sizeof(int),
            "Init_Lists::hb_top", comm );
//    bond_top = smalloc( system->total_cap * sizeof(int),
//            "Init_Lists::bond_top", comm );
//    hb_top = smalloc( system->local_cap * sizeof(int),
//            "Init_Lists::hb_top", comm );
    
    Estimate_Storages( system, control, lists, &Htop, hb_top, 
            bond_top, &num_3body, comm, &matrix_dim, cm_format );

#if defined(NEUTRAL_TERRITORY)
    Allocate_Matrix( &workspace->H, matrix_dim, Htop, cm_format, comm );
#else
    Allocate_Matrix( &workspace->H, system->local_cap, Htop, cm_format, comm );
#endif
    workspace->L = NULL;
    workspace->U = NULL;
    workspace->H_spar_patt = NULL;
    workspace->H_app_inv = NULL;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated H matrix: Htop=%d, space=%dMB\n",
            system->my_rank, Htop,
            (int)(Htop * sizeof(sparse_matrix_entry) / (1024 * 1024)) );
#endif

    if ( control->hbond_cut > 0 )
    {
        /* init H indexes */
        total_hbonds = 0;
        for ( i = 0; i < system->n; ++i )
        {
            system->my_atoms[i].num_hbonds = hb_top[i];
            total_hbonds += hb_top[i];
        }
        total_hbonds = MAX( total_hbonds * SAFER_ZONE, MIN_CAP * MIN_HBONDS );

        if ( !Make_List( system->Hcap, total_hbonds, TYP_HBOND,
                    HALF_LIST, lists[HBONDS], comm ) )
        {
            fprintf( stderr, "not enough space for hbonds list. terminating!\n" );
            MPI_Abort( comm, INSUFFICIENT_MEMORY );
        }
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: allocated hbonds: total_hbonds=%d, space=%dMB\n",
                system->my_rank, total_hbonds,
                (int)(total_hbonds * sizeof(hbond_data) / (1024 * 1024)) );
#endif
    }

    /* bonds list */
    //Allocate_Bond_List( system->N, bond_top, lists[BONDS] );
    //num_bonds = bond_top[system->N-1];
    total_bonds = 0;
    for ( i = 0; i < system->N; ++i )
    {
        system->my_atoms[i].num_bonds = bond_top[i];
        total_bonds += bond_top[i];
    }
    bond_cap = MAX( total_bonds * SAFE_ZONE, MIN_CAP * MIN_BONDS );

    if ( !Make_List( system->total_cap, bond_cap, TYP_BOND,
                HALF_LIST, lists[BONDS], comm ) )
    {
        fprintf( stderr, "not enough space for bonds list. terminating!\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated bonds: total_bonds=%d, space=%dMB\n",
            system->my_rank, bond_cap,
            (int)(bond_cap * sizeof(bond_data) / (1024 * 1024)) );
#endif

    /* 3bodies list */
    cap_3body = MAX( num_3body * SAFE_ZONE, MIN_3BODIES );
    if ( !Make_List( bond_cap, cap_3body, TYP_THREE_BODY,
                HALF_LIST, lists[THREE_BODIES], comm ) )
    {
        fprintf( stderr, "Problem in initializing angles list. Terminating!\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated 3-body list: num_3body=%d, space=%dMB\n",
            system->my_rank, cap_3body,
            (int)(cap_3body * sizeof(three_body_interaction_data) / (1024 * 1024)) );
#endif

#if defined(TEST_FORCES)
    if ( !Make_List( system->total_cap, bond_cap * 8, TYP_DDELTA,
                HALF_LIST, lists[DDELTAS], comm ) )
    {
        fprintf( stderr, "Problem in initializing dDelta list. Terminating!\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }
    fprintf( stderr, "p%d: allocated dDelta list: num_ddelta=%d space=%ldMB\n",
            system->my_rank, bond_cap * 30,
            bond_cap * 8 * sizeof(dDelta_data) / (1024 * 1024) );

    if ( !Make_List( bond_cap, bond_cap * 50, TYP_DBO, HALF_LIST, lists[DBOS], comm ) )
    {
        fprintf( stderr, "Problem in initializing dBO list. Terminating!\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }
    fprintf( stderr, "p%d: allocated dbond list: num_dbonds=%d space=%ldMB\n",
            system->my_rank, bond_cap * MAX_BONDS * 3,
            bond_cap * MAX_BONDS * 3 * sizeof(dbond_data) / (1024 * 1024) );
#endif

    sfree( hb_top, "Init_Lists::hb_top" );
    sfree( bond_top, "Init_Lists::bond_top" );

    return SUCCESS;
}


#elif defined(LAMMPS_REAX)
int  Init_Lists( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace, reax_list **lists,
        mpi_datatypes *mpi_data, char *msg )
{
    int i, num_nbrs, matrix_dim;
    int total_hbonds, total_bonds, bond_cap, num_3body, cap_3body, Htop;
    int *hb_top, *bond_top;
    int nrecv[MAX_NBRS];
    MPI_Comm comm;

    comm = mpi_data->world;

    bond_top = scalloc( system->total_cap, sizeof(int),
            "Init_Lists::bond_top", comm );
    hb_top = scalloc( system->local_cap, sizeof(int),
            "Init_Lists::hb_top", comm );

    //TODO: add one paramater at the end for charge matrix format - half or full
    Estimate_Storages( system, control, lists, &Htop, hb_top, 
            bond_top, &num_3body, comm, &matrix_dim );

    if ( control->hbond_cut > 0 )
    {
        /* init H indexes */
        total_hbonds = 0;
        for ( i = 0; i < system->n; ++i )
        {
            system->my_atoms[i].num_hbonds = hb_top[i];
            total_hbonds += hb_top[i];
        }
        total_hbonds = (int)(MAX( total_hbonds * SAFER_ZONE, MIN_CAP * MIN_HBONDS ));

        if ( !Make_List( system->Hcap, total_hbonds, TYP_HBOND,
                    HALF_LIST, lists[HBONDS], comm ) )
        {
            fprintf( stderr, "not enough space for hbonds list. terminating!\n" );
            MPI_Abort( comm, INSUFFICIENT_MEMORY );
        }
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: allocated hbonds: total_hbonds=%d, space=%dMB\n",
                system->my_rank, total_hbonds,
                (int)(total_hbonds * sizeof(hbond_data) / (1024 * 1024)) );
#endif
    }

    /* bonds list */
    //Allocate_Bond_List( system->N, bond_top, lists[BONDS] );
    //num_bonds = bond_top[system->N-1];
    total_bonds = 0;
    for ( i = 0; i < system->N; ++i )
    {
        system->my_atoms[i].num_bonds = bond_top[i];
        total_bonds += bond_top[i];
    }
    bond_cap = (int)(MAX( total_bonds * SAFE_ZONE, MIN_CAP * MIN_BONDS ));

    if ( !Make_List( system->total_cap, bond_cap, TYP_BOND,
                HALF_LIST, lists[BONDS], comm ) )
    {
        fprintf( stderr, "not enough space for bonds list. terminating!\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated bonds: total_bonds=%d, space=%dMB\n",
            system->my_rank, bond_cap,
            (int)(bond_cap * sizeof(bond_data) / (1024 * 1024)) );
#endif

    /* 3bodies list */
    cap_3body = (int)(MAX( num_3body * SAFE_ZONE, MIN_3BODIES ));
    if ( !Make_List( bond_cap, cap_3body, TYP_THREE_BODY,
                HALF_LIST, lists[THREE_BODIES], comm ) )
    {
        fprintf( stderr, "Problem in initializing angles list. Terminating!\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: allocated 3-body list: num_3body=%d, space=%dMB\n",
            system->my_rank, cap_3body,
            (int)(cap_3body * sizeof(three_body_interaction_data) / (1024 * 1024)) );
#endif

#if defined(TEST_FORCES)
    if ( !Make_List( system->total_cap, bond_cap * 8, TYP_DDELTA,
                HALF_LIST, lists[DDELTAS], comm ) )
    {
        fprintf( stderr, "Problem in initializing dDelta list. Terminating!\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }
    fprintf( stderr, "p%d: allocated dDelta list: num_ddelta=%d space=%ldMB\n",
            system->my_rank, bond_cap * 30,
            bond_cap * 8 * sizeof(dDelta_data) / (1024 * 1024) );

    if ( !Make_List( bond_cap, bond_cap * 50, TYP_DBO, HALF_LIST, lists[DBOS], comm ) )
    {
        fprintf( stderr, "Problem in initializing dBO list. Terminating!\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }
    fprintf( stderr, "p%d: allocated dbond list: num_dbonds=%d space=%ldMB\n",
            system->my_rank, bond_cap * MAX_BONDS * 3,
            bond_cap * MAX_BONDS * 3 * sizeof(dbond_data) / (1024 * 1024) );
#endif

    sfree( hb_top, "Init_Lists::hb_top" );
    sfree( bond_top, "Init_Lists::bond_top" );

    return SUCCESS;
}
#endif



#if defined(PURE_REAX)
void Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data, evolve_function *Evolve )
{
    char msg[MAX_STR];

    if ( Init_MPI_Datatypes( system, workspace, mpi_data, MPI_COMM_WORLD, msg ) ==
            FAILURE )
    {
        fprintf( stderr, "p%d: init_mpi_datatypes: could not create datatypes\n",
                system->my_rank );
        fprintf( stderr, "p%d: mpi_data couldn't be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized mpi datatypes\n", system->my_rank );
#endif

    if ( Init_System(system, control, data, workspace, mpi_data, msg) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: system initialized\n", system->my_rank );
#endif

    if ( Init_Simulation_Data(system, control, data, mpi_data, Evolve, msg) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: sim_data couldn't be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized simulation data\n", system->my_rank );
#endif

    if ( Init_Workspace( system, control, workspace, mpi_data->world, msg ) ==
            FAILURE )
    {
        fprintf( stderr, "p%d:init_workspace: not enough memory\n",
                system->my_rank );
        fprintf( stderr, "p%d:workspace couldn't be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized workspace\n", system->my_rank );
#endif

    if ( Init_Lists( system, control, data, workspace, lists, mpi_data, msg ) ==
            FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized lists\n", system->my_rank );
#endif

    if ( Init_Output_Files(system, control, out_control, mpi_data, msg) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: could not open output files! terminating...\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: output files opened\n", system->my_rank );
#endif

    if ( control->tabulate )
    {
        if ( Init_Lookup_Tables(system, control, workspace, mpi_data, msg) == FAILURE )
        {
            fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
            fprintf( stderr, "p%d: couldn't create lookup table! terminating.\n",
                    system->my_rank );
            MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
        }

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: initialized lookup tables\n", system->my_rank );
#endif
    }

    Init_Bonded_Force_Functions( control );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized force functions\n", system->my_rank );
#endif

#if defined(TEST_FORCES)
    Init_Force_Test_Functions( control );
    fprintf( stderr, "p%d: initialized force test functions\n",
            system->my_rank );
#endif
}

#elif defined(LAMMPS_REAX)
void Initialize( reax_system *system, control_params *control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data, evolve_function *Evolve, MPI_Comm comm )
{
    char msg[MAX_STR];


    if ( Init_MPI_Datatypes(system, workspace, mpi_data, comm, msg) == FAILURE )
    {
        fprintf( stderr, "p%d: init_mpi_datatypes: could not create datatypes\n",
                system->my_rank );
        fprintf( stderr, "p%d: mpi_data couldn't be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized mpi datatypes\n", system->my_rank );
#endif

    if ( Init_System(system, control, msg) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: system initialized\n", system->my_rank );
#endif

    if ( Init_Simulation_Data( system, control, data, Evolve, msg ) == FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: sim_data couldn't be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized simulation data\n", system->my_rank );
#endif

    if ( Init_Workspace( system, control, workspace, mpi_data->world, msg ) ==
            FAILURE )
    {
        fprintf( stderr, "p%d:init_workspace: not enough memory\n",
                system->my_rank );
        fprintf( stderr, "p%d:workspace couldn't be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized workspace\n", system->my_rank );
#endif

    if ( Init_Lists( system, control, data, workspace, lists, mpi_data, msg ) ==
            FAILURE )
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: system could not be initialized! terminating.\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized lists\n", system->my_rank );
#endif

    if ( Init_Output_Files(system, control, out_control, mpi_data, msg) == FAILURE)
    {
        fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
        fprintf( stderr, "p%d: could not open output files! terminating...\n",
                system->my_rank );
        MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
    }
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: output files opened\n", system->my_rank );
#endif

    if ( control->tabulate )
    {
        if ( Init_Lookup_Tables( system, control, workspace, mpi_data, msg ) == FAILURE )
        {
            fprintf( stderr, "p%d: %s\n", system->my_rank, msg );
            fprintf( stderr, "p%d: couldn't create lookup table! terminating.\n",
                    system->my_rank );
            MPI_Abort( mpi_data->world, CANNOT_INITIALIZE );
        }
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d: initialized lookup tables\n", system->my_rank );
#endif
    }

    Init_Bonded_Force_Functions( control );
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initialized force functions\n", system->my_rank );
#endif

#if defined(TEST_FORCES)
    Init_Force_Test_Functions( control );
    fprintf( stderr, "p%d: initialized force test functions\n",
            system->my_rank );
#endif
}
#endif
