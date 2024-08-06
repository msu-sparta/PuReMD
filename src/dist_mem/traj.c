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
  #include "../common/include/config.h"
#endif

#if defined(PURE_REAX)
  #include "traj.h"

  #include "comm_tools.h"
  #include "list.h"
  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_traj.h"

  #include "reax_comm_tools.h"
  #include "reax_list.h"
  #include "reax_tool_box.h"
#endif

#if defined(HAVE_CUDA)
  #include "cuda/gpu_copy.h"
#elif defined(HAVE_HIP)
  #include "hip/gpu_copy.h"
#endif


int Set_My_Trajectory_View( MPI_File trj, int offset, MPI_Datatype etype,
        MPI_Comm comm, int my_rank, int my_n, int big_n )
{
    int my_disp, type_size, ret;
    MPI_Aint lower_bound, extent;
    MPI_Datatype view;

    /* get old type info */
    ret = MPI_Type_get_extent( etype, &lower_bound, &extent );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    type_size = MPI_Aint_add( lower_bound, extent );

    /* determine where to start writing into the mpi file */
    my_disp = SumScan( my_n, my_rank, MASTER_NODE, comm );
    my_disp -= my_n;
    type_size /= sizeof(char);

    ret = MPI_Type_create_resized( etype, -1 * type_size * my_disp, type_size * (big_n - my_disp), &view );
    Check_MPI_Error( ret, __FILE__, __LINE__ );
    ret = MPI_Type_commit( &view );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    /* create atom_info_view */
    MPI_File_set_view( trj, offset, etype, view, "native", MPI_INFO_NULL );

    return my_disp;
}


int Reallocate_Output_Buffer( output_controls *out_control, int req_space,
        MPI_Comm comm )
{
    if ( out_control->buffer_len > 0 )
    {
        sfree( out_control->buffer, __FILE__, __LINE__ );
    }

    out_control->buffer_len = req_space * SAFE_ZONE;
    out_control->buffer = smalloc( out_control->buffer_len * sizeof(char),
            __FILE__, __LINE__ );
    if ( out_control->buffer == NULL )
    {
        fprintf( stderr,
                 "insufficient memory for required buffer size %d. terminating!\n",
                 (int) (req_space * SAFE_ZONE) );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }

    return SUCCESS;
}


void Write_Skip_Line( output_controls *out_control, mpi_datatypes *mpi_data,
        int my_rank, int skip, int num_section )
{
    MPI_Status status;

    if ( out_control->traj_method == MPI_TRAJ )
    {
        MPI_File_set_view( out_control->trj, out_control->trj_offset,
                           mpi_data->header_line, mpi_data->header_line,
                           "native", MPI_INFO_NULL );

        if ( my_rank == MASTER_NODE )
        {
            sprintf( out_control->line, INT2_LINE, "chars_to_skip_section:",
                     skip, num_section );
            MPI_File_write( out_control->trj, out_control->line, 1,
                            mpi_data->header_line, &status );
        }
        out_control->trj_offset += HEADER_LINE_LEN;
    }
    else
    {
        if ( my_rank == MASTER_NODE )
        {
            fprintf( out_control->strj, INT2_LINE,
                     "chars_to_skip_section:", skip, num_section );
        }
    }
}


int Write_Header( reax_system *system, control_params *control,
                  output_controls *out_control, mpi_datatypes *mpi_data )
{
    int  num_hdr_lines, my_hdr_lines, buffer_req;
    MPI_Status status;
    char ensembles[ens_N][25] = {
        "NVE", "NVT", "fully flexible NPT",
        "semi isotropic NPT", "isotropic NPT",
    };
    char reposition[3][25] = {
        "fit to periodic box", "CoM to center of box",
        "CoM to origin",
    };
    char t_regime[3][25] = {
        "T-coupling only", "step-wise", "constant slope",
    };

    char traj_methods[TF_N][10] = {
        "custom", "xyz",
    };
    char atom_formats[8][40] =  {
        "none", "invalid", "invalid", "invalid",
        "xyz_q", "xyz_q_fxfyfz", "xyz_q_vxvyvz",
        "detailed_atom_info",
    };
    char bond_formats[3][30] = {
        "none", "basic_bond_info", "detailed_bond_info",
    };
    char angle_formats[2][30] = {
        "none", "basic_angle_info",
    };

    /* set header lengths */
    num_hdr_lines = NUM_HEADER_LINES;
    my_hdr_lines = num_hdr_lines * ( system->my_rank == MASTER_NODE );
    buffer_req = my_hdr_lines * HEADER_LINE_LEN;
    if ( buffer_req > out_control->buffer_len * DANGER_ZONE )
    {
        Reallocate_Output_Buffer( out_control, buffer_req, MPI_COMM_WORLD );
    }

    /* only the master node writes into trajectory header */
    if ( system->my_rank == MASTER_NODE )
    {
        /* clear the contents of line & buffer */
        out_control->line[0] = 0;
        out_control->buffer[0] = 0;

        /* to skip the header */
        sprintf( out_control->line, INT_LINE, "chars_to_skip_header:",
                 (num_hdr_lines - 1) * HEADER_LINE_LEN );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        /* general simulation info */
        sprintf( out_control->line, STR_LINE, "simulation_name:",
                 out_control->traj_title );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, INT_LINE, "number_of_atoms:", system->bigN );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "ensemble_type:",
                 ensembles[ control->ensemble ] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, INT_LINE, "number_of_steps:",
                 control->nsteps );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "timestep_length_(in_fs):",
                 control->dt * 1000 );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        /* restart info */
        sprintf( out_control->line, STR_LINE, "is_this_a_restart?:",
                 (control->restart ? "yes" : "no") );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        //sprintf( out_control->line, STR_LINE, "restarted_from_file:",
        //     (control->restart ? control->restart_from : "NA") );
        //strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        //sprintf( out_control->line, STR_LINE, "kept_restart_velocities?:",
        //     (control->restart ? (control->random_vel ? "no":"yes"):"NA") );
        //strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "write_restart_files?:",
                 ((out_control->restart_freq > 0) ? "yes" : "no") );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, INT_LINE, "frequency_to_write_restarts:",
                 out_control->restart_freq );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        /* preferences */
        sprintf( out_control->line, STR_LINE, "tabulate_long_range_intrs?:",
                 (control->tabulate ? "yes" : "no") );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, INT_LINE, "table_size:", control->tabulate );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "restrict_bonds?:",
                 (control->restrict_bonds ? "yes" : "no") );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, INT_LINE, "bond_restriction_length:",
                 control->restrict_bonds );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "reposition_atoms?:",
                 reposition[control->reposition_atoms] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, INT_LINE, "remove_CoM_velocity?:",
                 (control->ensemble == NVE) ? 0 : control->remove_CoM_vel);
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );


        /* cut-off values */
        sprintf( out_control->line, REAL_LINE, "bonded_intr_dist_cutoff:",
                 control->bond_cut );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "nonbonded_intr_dist_cutoff:",
                 control->nonb_cut );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "hbond_dist_cutoff:",
                 control->hbond_cut );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "reax_bond_threshold:",
                 control->bo_cut );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "physical_bond_threshold:",
                 control->bg_cut );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "valence_angle_threshold:",
                 control->thb_cut );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, SCI_LINE, "QEq_tolerance:", control->cm_solver_q_err );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        /* temperature controls */
        sprintf( out_control->line, REAL_LINE, "initial_temperature:",
                 control->T_init );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "target_temperature:",
                 control->T_final );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "thermal_inertia:",
                 control->Tau_T );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "temperature_regime:",
                 t_regime[ control->T_mode ] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "temperature_change_rate_(K/ps):",
                 control->T_rate / control->T_freq );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        /* pressure controls */
        sprintf( out_control->line, REAL3_LINE, "target_pressure_(GPa):",
                 control->P[0], control->P[1], control->P[2] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL3_LINE, "virial_inertia:",
                 control->Tau_P[0], control->Tau_P[1], control->Tau_P[2] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        /* trajectory */
        sprintf( out_control->line, INT_LINE, "energy_dumping_freq:",
                 out_control->energy_update_freq );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, INT_LINE, "trajectory_dumping_freq:",
                 out_control->write_steps );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "compress_trajectory_output?:",
                 (out_control->traj_compress ? "yes" : "no") );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "trajectory_format:",
                 traj_methods[ out_control->traj_method ] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "atom_info:",
                 atom_formats[ out_control->atom_info ] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "bond_info:",
                 bond_formats[ out_control->bond_info ] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, STR_LINE, "angle_info:",
                 angle_formats[ out_control->angle_info ] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        /* analysis */
        //sprintf( out_control->line, STR_LINE, "molecular_analysis:",
        //     (control->molec_anal ? "yes" : "no") );
        //strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, INT_LINE, "molecular_analysis_frequency:",
                 control->molecular_analysis );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );
    }

    /* dump out the buffer */
    if ( out_control->traj_method == MPI_TRAJ )
    {
        out_control->trj_offset = 0;
        Set_My_Trajectory_View( out_control->trj,
                out_control->trj_offset, mpi_data->header_line,
                MPI_COMM_WORLD, system->my_rank,
                my_hdr_lines, num_hdr_lines );
        MPI_File_write_all( out_control->trj, out_control->buffer,
                num_hdr_lines, mpi_data->header_line, &status );
        out_control->trj_offset = (num_hdr_lines) * HEADER_LINE_LEN;
    }
    else
    {
        if ( system->my_rank == MASTER_NODE )
        {
            fprintf( out_control->strj, "%s", out_control->buffer );
        }
    }

    return SUCCESS;
}


int Write_Init_Desc( reax_system *system, control_params *control,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, me, np, cnt, buffer_len, buffer_req, ret;
    reax_atom *p_atom;
    //MPI_Request request;
    MPI_Status status;

    me = system->my_rank;
    np = control->nprocs;

    /* skip info */
    Write_Skip_Line( out_control, mpi_data, me,
                     system->bigN * INIT_DESC_LEN, system->bigN );

    if ( out_control->traj_method == REG_TRAJ && me == MASTER_NODE )
    {
        buffer_req = system->bigN * INIT_DESC_LEN + 1;
    }
    else
    {
        buffer_req = system->n * INIT_DESC_LEN + 1;
    }

    if ( buffer_req > out_control->buffer_len * DANGER_ZONE )
    {
        Reallocate_Output_Buffer( out_control, buffer_req, MPI_COMM_WORLD );
    }

    out_control->line[0] = '\0';
    out_control->buffer[0] = '\0';
    for ( i = 0; i < system->n; ++i )
    {
        p_atom = &system->my_atoms[i];

        sprintf( out_control->line, INIT_DESC,
                 p_atom->orig_id, p_atom->type, p_atom->name,
                 system->reax_param.sbp[ p_atom->type ].mass );
        strncpy( &out_control->buffer[i * INIT_DESC_LEN],
                 out_control->line, INIT_DESC_LEN + 1 );
    }

    if ( out_control->traj_method == MPI_TRAJ )
    {
        Set_My_Trajectory_View( out_control->trj, out_control->trj_offset,
                mpi_data->init_desc_line, MPI_COMM_WORLD,
                me, system->n, system->bigN );
        MPI_File_write( out_control->trj, out_control->buffer, system->n,
                mpi_data->init_desc_line, &status );
        out_control->trj_offset += system->bigN * INIT_DESC_LEN;
    }
    else
    {
        if ( me != MASTER_NODE )
        {
            ret = MPI_Send( out_control->buffer, buffer_req - 1, MPI_CHAR, MASTER_NODE,
                      np * INIT_DESCS + me, MPI_COMM_WORLD );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
        else
        {
            buffer_len = system->n * INIT_DESC_LEN;
            for ( i = 0; i < np; ++i )
            {
                if ( i != MASTER_NODE )
                {
                    ret = MPI_Recv( &out_control->buffer[buffer_len], buffer_req - buffer_len,
                              MPI_CHAR, i, np * INIT_DESCS + i, MPI_COMM_WORLD, &status );
                    Check_MPI_Error( ret, __FILE__, __LINE__ );
                    MPI_Get_count( &status, MPI_CHAR, &cnt );
                    buffer_len += cnt;
                }
            }
            out_control->buffer[buffer_len] = 0;
            fprintf( out_control->strj, "%s", out_control->buffer );
        }
    }

    return SUCCESS;
}


void Init_Traj( reax_system *system, control_params *control,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    char fname[MAX_STR];
    int atom_line_len[ NR_OPT_ATOM ] = { 0, 0, 0, 0,
        ATOM_BASIC_LEN, ATOM_wV_LEN,
        ATOM_wF_LEN, ATOM_FULL_LEN
    };
    int bond_line_len[ NR_OPT_BOND ] = { 0, BOND_BASIC_LEN, BOND_FULL_LEN };
    int angle_line_len[ NR_OPT_ANGLE ] = { 0, ANGLE_BASIC_LEN };
    int ret;

    /* generate trajectory name */
    sprintf( fname, "%s.trj", control->sim_name );

    /* how should I write atoms? */
    out_control->atom_line_len = atom_line_len[ out_control->atom_info ];
    out_control->write_atoms = ( out_control->atom_line_len ? 1 : 0 );
    /* bonds? */
    out_control->bond_line_len = bond_line_len[ out_control->bond_info ];
    out_control->write_bonds = ( out_control->bond_line_len ? 1 : 0 );
    /* angles? */
    out_control->angle_line_len = angle_line_len[ out_control->angle_info ];
    out_control->write_angles = ( out_control->angle_line_len ? 1 : 0 );

    /* allocate line & buffer space */
    out_control->line = scalloc( MAX_TRJ_LINE_LEN + 1, sizeof(char),
           __FILE__, __LINE__ );
    out_control->buffer_len = 0;
    out_control->buffer = NULL;

//    fprintf( stderr, "p%d: init_traj: atom_line_len = %d "
//            "bond_line_len = %d, angle_line_len = %d\n"
//            "max_line = %d, max_buffer_size = %d\n",
//            system->my_rank, out_control->atom_line_len,
//            out_control->bond_line_len, out_control->angle_line_len,
//            MAX_TRJ_LINE_LEN, MAX_TRJ_BUFFER_SIZE );

    /* write trajectory header and atom info, if applicable */
    if ( out_control->traj_method == MPI_TRAJ )
    {
        /* attemp to delete the file to get rid of remnants of previous runs */
        if ( system->my_rank == MASTER_NODE )
        {
            MPI_File_delete( fname, MPI_INFO_NULL );
        }

        /* open a fresh trajectory file */
        ret = MPI_File_open( MPI_COMM_WORLD, fname,
                MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL,
                &out_control->trj );
        Check_MPI_Error( ret, __FILE__, __LINE__ );

        /* build the mpi structs for trajectory */
        /* header_line */
        MPI_Type_contiguous( HEADER_LINE_LEN, MPI_CHAR, &mpi_data->header_line );
        MPI_Type_commit( &mpi_data->header_line );
        /* init_desc_line */
        MPI_Type_contiguous( INIT_DESC_LEN, MPI_CHAR, &mpi_data->init_desc_line );
        MPI_Type_commit( &mpi_data->init_desc_line );
        /* atom */
        MPI_Type_contiguous( out_control->atom_line_len, MPI_CHAR,
                &mpi_data->atom_line );
        MPI_Type_commit( &mpi_data->atom_line );
        /* bonds */
        MPI_Type_contiguous( out_control->bond_line_len, MPI_CHAR,
                &mpi_data->bond_line );
        MPI_Type_commit( &mpi_data->bond_line );
        /* angles */
        MPI_Type_contiguous( out_control->angle_line_len, MPI_CHAR,
                &mpi_data->angle_line );
        MPI_Type_commit( &mpi_data->angle_line );
    }
    else if ( out_control->traj_method == REG_TRAJ )
    {
        if ( system->my_rank == MASTER_NODE )
        {
            out_control->strj = sfopen( fname, "w", __FILE__, __LINE__ );
        }
    }
    else
    {
        fprintf( stderr, "[ERROR] unknown trajectory option. Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, UNKNOWN_OPTION );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: initiated trajectory\n", system->my_rank );
#endif

    Write_Header( system, control, out_control, mpi_data );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: header written\n", system->my_rank );
#endif

    Write_Init_Desc( system, control, out_control, mpi_data );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: atom descriptions written\n", system->my_rank );
#endif
}


int Write_Frame_Header( reax_system *system, control_params *control,
        simulation_data *data, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    int me, num_frm_hdr_lines, my_frm_hdr_lines, buffer_req;
    MPI_Status status;

    me = system->my_rank;
    /* frame header lengths */
    num_frm_hdr_lines = 22;
    my_frm_hdr_lines = num_frm_hdr_lines * ( me == MASTER_NODE );
    buffer_req = my_frm_hdr_lines * HEADER_LINE_LEN;
    if ( buffer_req > out_control->buffer_len * DANGER_ZONE )
    {
        Reallocate_Output_Buffer( out_control, buffer_req, MPI_COMM_WORLD );
    }

    /* only the master node writes into trajectory header */
    if ( me == MASTER_NODE )
    {
        /* clear the contents of line & buffer */
        out_control->line[0] = 0;
        out_control->buffer[0] = 0;

        /* skip info */
        sprintf( out_control->line, INT_LINE, "chars_to_skip_frame_header:",
                 (num_frm_hdr_lines - 1) * HEADER_LINE_LEN );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        /* step & time */
        sprintf( out_control->line, INT_LINE, "step:", data->step );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "time_in_ps:",
                 data->step * control->dt );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );


        /* box info */
        sprintf( out_control->line, REAL_LINE, "volume:", system->big_box.V );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL3_LINE, "box_dimensions:",
                 system->big_box.box_norms[0],
                 system->big_box.box_norms[1],
                 system->big_box.box_norms[2] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL3_LINE,
                 "coordinate_angles:", 90.0, 90.0, 90.0 );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );


        /* system T and P */
        sprintf( out_control->line, REAL_LINE, "temperature:", data->therm.T );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "pressure:",
                 (control->ensemble == iNPT) ?
                 data->iso_bar.P : data->flex_bar.P_scalar );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );


        /* energies */
        sprintf( out_control->line, REAL_LINE, "total_energy:",
                 data->sys_en[E_TOT] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "total_kinetic:",
                 data->sys_en[E_KIN] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "total_potential:",
                 data->sys_en[E_POT] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "bond_energy:",
                 data->sys_en[E_BOND] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "atom_energy:",
                 data->sys_en[E_OV] + data->sys_en[E_UN] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "lone_pair_energy:",
                 data->sys_en[E_LP] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "valence_angle_energy:",
                 data->sys_en[E_ANG] + data->sys_en[E_PEN] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "3-body_conjugation:",
                 data->sys_en[E_COA] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "hydrogen_bond_energy:",
                 data->sys_en[E_HB] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "torsion_angle_energy:",
                 data->sys_en[E_TOR] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "4-body_conjugation:",
                 data->sys_en[E_CON] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "vdWaals_energy:",
                 data->sys_en[E_VDW] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "electrostatics_energy:",
                 data->sys_en[E_ELE] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );

        sprintf( out_control->line, REAL_LINE, "polarization_energy:",
                 data->sys_en[E_POL] );
        strncat( out_control->buffer, out_control->line, HEADER_LINE_LEN + 1 );
    }

    /* dump out the buffer */
    if ( out_control->traj_method == MPI_TRAJ )
    {
        Set_My_Trajectory_View( out_control->trj, out_control->trj_offset,
                mpi_data->header_line, MPI_COMM_WORLD,
                me, my_frm_hdr_lines, num_frm_hdr_lines );

        MPI_File_write_all(out_control->trj, out_control->buffer, my_frm_hdr_lines,
                mpi_data->header_line, &status);
        out_control->trj_offset += (num_frm_hdr_lines) * HEADER_LINE_LEN;
    }
    else
    {
        if ( system->my_rank == MASTER_NODE )
        {
            fprintf( out_control->strj, "%s", out_control->buffer );
        }
    }

    return SUCCESS;
}


int Write_Atoms( reax_system *system, control_params *control,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, me, np, line_len, buffer_len, buffer_req, cnt, ret;
    MPI_Status status;
    reax_atom *p_atom;

    me = system->my_rank;
    np = control->nprocs;
    line_len = out_control->atom_line_len;

    Write_Skip_Line( out_control, mpi_data, me,
            system->bigN * line_len, system->bigN );

    if ( out_control->traj_method == REG_TRAJ && me == MASTER_NODE )
    {
        buffer_req = system->bigN * line_len + 1;
    }
    else
    {
        buffer_req = system->n * line_len + 1;
    }

    if ( buffer_req > out_control->buffer_len * DANGER_ZONE )
    {
        Reallocate_Output_Buffer( out_control, buffer_req, MPI_COMM_WORLD );
    }

    /* fill in buffer */
    out_control->line[0] = 0;
    out_control->buffer[0] = 0;
    for ( i = 0; i < system->n; ++i )
    {
        p_atom = &system->my_atoms[i];

        switch ( out_control->atom_info )
        {
        case OPT_ATOM_BASIC:
            sprintf( out_control->line, ATOM_BASIC,
                     p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2],
                     p_atom->q );
            break;
        case OPT_ATOM_wF:
            sprintf( out_control->line, ATOM_wF,
                     p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2],
                     p_atom->f[0], p_atom->f[1], p_atom->f[2], p_atom->q );
            break;
        case OPT_ATOM_wV:
            sprintf( out_control->line, ATOM_wV,
                     p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2],
                     p_atom->v[0], p_atom->v[1], p_atom->v[2], p_atom->q );
            break;
        case OPT_ATOM_FULL:
            sprintf( out_control->line, ATOM_FULL,
                     p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2],
                     p_atom->v[0], p_atom->v[1], p_atom->v[2],
                     p_atom->f[0], p_atom->f[1], p_atom->f[2], p_atom->q );
            break;
        default:
            fprintf( stderr,
                     "write_traj_atoms: unknown atom trajectroy format!\n");
            MPI_Abort( MPI_COMM_WORLD, UNKNOWN_OPTION );
            break;
        }

        strncpy( &out_control->buffer[i * line_len], out_control->line, line_len + 1 );
    }

    if ( out_control->traj_method == MPI_TRAJ )
    {
        Set_My_Trajectory_View( out_control->trj, out_control->trj_offset,
                mpi_data->atom_line, MPI_COMM_WORLD,
                me, system->n, system->bigN );
        MPI_File_write( out_control->trj, out_control->buffer, system->n,
                mpi_data->atom_line, &status );
        out_control->trj_offset += (system->bigN) * out_control->atom_line_len;
    }
    else
    {
        if ( me != MASTER_NODE )
        {
            ret = MPI_Send( out_control->buffer, buffer_req - 1, MPI_CHAR, MASTER_NODE,
                      np * ATOM_LINES + me, MPI_COMM_WORLD );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
        else
        {
            buffer_len = system->n * line_len;
            for ( i = 0; i < np; ++i )
            {
                if ( i != MASTER_NODE )
                {
                    ret = MPI_Recv( out_control->buffer + buffer_len, buffer_req - buffer_len,
                              MPI_CHAR, i, np * ATOM_LINES + i, MPI_COMM_WORLD, &status );
                    Check_MPI_Error( ret, __FILE__, __LINE__ );
                    MPI_Get_count( &status, MPI_CHAR, &cnt );
                    buffer_len += cnt;
                }
            }
            out_control->buffer[buffer_len] = '\0';
            fprintf( out_control->strj, "%s", out_control->buffer );
        }
    }

    return SUCCESS;
}


int Write_Bonds( reax_system *system, control_params *control, reax_list *bonds,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, j, pj, me, np, my_bonds, num_bonds, ret;
    int line_len, buffer_len, buffer_req, cnt;
    MPI_Status  status;
    bond_data  *bo_ij;

    me = system->my_rank;
    np = control->nprocs;
    line_len = out_control->bond_line_len;

    /* count the number of bonds I will write */
    my_bonds = 0;
    for ( i = 0; i < system->n; ++i )
    {
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            j = bonds->bond_list[pj].nbr;

            if ( system->my_atoms[i].orig_id <= system->my_atoms[j].orig_id &&
                    bonds->bond_list[pj].bo_data.BO >= control->bg_cut )
            {
                ++my_bonds;
            }
        }
    }
    /* allreduce - total number of bonds */
    ret = MPI_Allreduce( &my_bonds, &num_bonds, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    Write_Skip_Line( out_control, mpi_data, me, num_bonds * line_len, num_bonds );

    if ( out_control->traj_method == REG_TRAJ && me == MASTER_NODE )
    {
        buffer_req = num_bonds * line_len + 1;
    }
    else
    {
        buffer_req = my_bonds * line_len + 1;
    }

    if ( buffer_req > out_control->buffer_len * DANGER_ZONE )
    {
        Reallocate_Output_Buffer( out_control, buffer_req, MPI_COMM_WORLD );
    }

    /* fill in the buffer */
    my_bonds = 0;
    out_control->line[0] = 0;
    out_control->buffer[0] = 0;
    for ( i = 0; i < system->n; ++i )
    {
        for ( pj = Start_Index(i, bonds); pj < End_Index(i, bonds); ++pj )
        {
            bo_ij = &bonds->bond_list[pj];
            j = bo_ij->nbr;

            if ( system->my_atoms[i].orig_id <= system->my_atoms[j].orig_id &&
                    bo_ij->bo_data.BO >= control->bg_cut )
            {
                switch ( out_control->bond_info )
                {
                case OPT_BOND_BASIC:
                    sprintf( out_control->line, BOND_BASIC,
                             system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                             bo_ij->d, bo_ij->bo_data.BO );
                    break;
                case OPT_BOND_FULL:
                    sprintf( out_control->line, BOND_FULL,
                             system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                             bo_ij->d, bo_ij->bo_data.BO, bo_ij->bo_data.BO_s,
                             bo_ij->bo_data.BO_pi, bo_ij->bo_data.BO_pi2 );
                    break;
                default:
                    fprintf(stderr, "write_traj_bonds: FATAL! invalid bond_info option");
                    MPI_Abort( MPI_COMM_WORLD, UNKNOWN_OPTION );
                    break;
                }

                strncpy( &out_control->buffer[my_bonds * line_len],
                         out_control->line, line_len + 1 );
                ++my_bonds;
            }
        }
    }

    if ( out_control->traj_method == MPI_TRAJ )
    {
        Set_My_Trajectory_View( out_control->trj, out_control->trj_offset,
                mpi_data->bond_line, MPI_COMM_WORLD,
                me, my_bonds, num_bonds );
        MPI_File_write( out_control->trj, out_control->buffer, my_bonds,
                mpi_data->bond_line, &status );
        out_control->trj_offset += num_bonds * line_len;
    }
    else
    {
        if ( me != MASTER_NODE )
        {
            ret = MPI_Send( out_control->buffer, buffer_req - 1, MPI_CHAR, MASTER_NODE,
                      np * BOND_LINES + me, MPI_COMM_WORLD );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
        else
        {
            buffer_len = my_bonds * line_len;
            for ( i = 0; i < np; ++i )
            {
                if ( i != MASTER_NODE )
                {
                    ret = MPI_Recv( &out_control->buffer[buffer_len], buffer_req - buffer_len,
                              MPI_CHAR, i, np * BOND_LINES + i, MPI_COMM_WORLD, &status );
                    Check_MPI_Error( ret, __FILE__, __LINE__ );
                    MPI_Get_count( &status, MPI_CHAR, &cnt );
                    buffer_len += cnt;
                }
            }
            out_control->buffer[buffer_len] = 0;
            fprintf( out_control->strj, "%s", out_control->buffer );
        }
    }

    return SUCCESS;
}


int Write_Angles( reax_system *system, control_params *control,
        reax_list *bonds, reax_list *thb_intrs,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, j, k, pi, pk, me, np, my_angles, num_angles, ret;
    int line_len, buffer_len, buffer_req, cnt;
    bond_data  *bo_ij, *bo_jk;
    three_body_interaction_data *angle_ijk;
    MPI_Status  status;

    me = system->my_rank;
    np = control->nprocs;
    line_len = out_control->angle_line_len;

    /* count the number of valence angles I will output */
    my_angles = 0;
    for ( j = 0; j < system->n; ++j )
    {
        for ( pi = Start_Index(j, bonds); pi < End_Index(j, bonds); ++pi )
        {
            bo_ij = &bonds->bond_list[pi];
            i = bo_ij->nbr;

            if ( bo_ij->bo_data.BO >= control->bg_cut ) // physical j&i bond
            {
                for ( pk = Start_Index( pi, thb_intrs );
                        pk < End_Index( pi, thb_intrs ); ++pk )
                {
                    angle_ijk = &thb_intrs->three_body_list[pk];
                    k = angle_ijk->thb;
                    bo_jk = &bonds->bond_list[ angle_ijk->pthb ];

                    if ( system->my_atoms[i].orig_id < system->my_atoms[k].orig_id &&
                            bo_jk->bo_data.BO >= control->bg_cut ) // physical j&k bond
                    {
                        ++my_angles;
                    }
                }
            }
        }
    }
    /* total number of valences */
    ret = MPI_Allreduce( &my_angles, &num_angles, 1, MPI_INT,
            MPI_SUM, MPI_COMM_WORLD );
    Check_MPI_Error( ret, __FILE__, __LINE__ );

    Write_Skip_Line( out_control, mpi_data, me, num_angles * line_len, num_angles );

    if ( out_control->traj_method == REG_TRAJ && me == MASTER_NODE )
    {
        buffer_req = num_angles * line_len + 1;
    }
    else
    {
        buffer_req = my_angles * line_len + 1;
    }

    if ( buffer_req > out_control->buffer_len * DANGER_ZONE )
    {
        Reallocate_Output_Buffer( out_control, buffer_req, MPI_COMM_WORLD );
    }

    /* fill in the buffer */
    my_angles = 0;
    out_control->line[0] = '\0';
    out_control->buffer[0] = '\0';
    for ( j = 0; j < system->n; ++j )
    {
        for ( pi = Start_Index(j, bonds); pi < End_Index(j, bonds); ++pi )
        {
            bo_ij = &bonds->bond_list[pi];
            i = bo_ij->nbr;

            if ( bo_ij->bo_data.BO >= control->bg_cut ) // physical j&i bond
            {
                for ( pk = Start_Index( pi, thb_intrs );
                        pk < End_Index( pi, thb_intrs ); ++pk )
                {
                    angle_ijk = &thb_intrs->three_body_list[pk];
                    k = angle_ijk->thb;
                    bo_jk = &bonds->bond_list[ angle_ijk->pthb ];

                    if ( system->my_atoms[i].orig_id < system->my_atoms[k].orig_id &&
                            bo_jk->bo_data.BO >= control->bg_cut )   // physical j&k bond
                    {
                        sprintf( out_control->line, ANGLE_BASIC,
                                 system->my_atoms[i].orig_id, system->my_atoms[j].orig_id,
                                 system->my_atoms[k].orig_id, RAD2DEG( angle_ijk->theta ) );

                        strncpy( &out_control->buffer[my_angles * line_len],
                                 out_control->line, line_len + 1 );
                        ++my_angles;
                    }
                }
            }
        }
    }

    if ( out_control->traj_method == MPI_TRAJ )
    {
        Set_My_Trajectory_View( out_control->trj, out_control->trj_offset,
                mpi_data->angle_line, MPI_COMM_WORLD,
                me, my_angles, num_angles );
        MPI_File_write( out_control->trj, out_control->buffer, my_angles,
                mpi_data->angle_line, &status );
        out_control->trj_offset += num_angles * line_len;
    }
    else
    {
        if ( me != MASTER_NODE )
        {
            ret = MPI_Send( out_control->buffer, buffer_req - 1, MPI_CHAR, MASTER_NODE,
                      np * ANGLE_LINES + me, MPI_COMM_WORLD );
            Check_MPI_Error( ret, __FILE__, __LINE__ );
        }
        else
        {
            buffer_len = my_angles * line_len;
            for ( i = 0; i < np; ++i )
            {
                if ( i != MASTER_NODE )
                {
                    ret = MPI_Recv( &out_control->buffer[buffer_len], buffer_req - buffer_len,
                              MPI_CHAR, i, np * ANGLE_LINES + i, MPI_COMM_WORLD, &status );
                    Check_MPI_Error( ret, __FILE__, __LINE__ );
                    MPI_Get_count( &status, MPI_CHAR, &cnt );
                    buffer_len += cnt;
                }
            }
            out_control->buffer[buffer_len] = '\0';
            fprintf( out_control->strj, "%s", out_control->buffer );
        }
    }

    return SUCCESS;
}


void Append_Frame( reax_system *system, control_params *control,
        simulation_data *data, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
    reax_list *bond_list = NULL, *thb_list = NULL;
#endif

    Write_Frame_Header( system, control, data, out_control, mpi_data );

    if ( out_control->write_atoms )
    {
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
        GPU_Copy_Atoms_Device_to_Host( system, control );
#endif
        Write_Atoms( system, control, out_control, mpi_data );
    }

    if ( out_control->write_bonds )
    {
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
        GPU_Copy_List_Device_to_Host( control, bond_list, lists[BONDS], TYP_BOND );
        Write_Bonds( system, control, bond_list, out_control, mpi_data );
#else
        Write_Bonds( system, control, lists[BONDS], out_control, mpi_data );
#endif
    }

    if ( out_control->write_angles )
    {
#if defined(HAVE_CUDA) || defined(HAVE_HIP)
        GPU_Copy_List_Device_to_Host( control, thb_list, lists[THREE_BODIES],
                TYP_THREE_BODY );
        Write_Angles( system, control, bond_list, thb_list,
                      out_control, mpi_data );
#else
        Write_Angles( system, control, lists[BONDS], lists[THREE_BODIES],
                      out_control, mpi_data );
#endif
    }

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
    if ( bond_list != NULL )
    {
        Delete_List( bond_list );
    }
    if ( thb_list != NULL )
    {
        Delete_List( thb_list );
    }
#endif
}


void End_Traj( int my_rank, output_controls *out_control )
{
    int ret;

    if ( out_control->traj_method == MPI_TRAJ )
    {
        ret = MPI_File_close( &out_control->trj );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }
    else if ( my_rank == MASTER_NODE )
    {
        sfclose( out_control->strj, __FILE__, __LINE__ );
    }

    sfree( out_control->buffer, __FILE__, __LINE__ );
    sfree( out_control->line, __FILE__, __LINE__ );
}
