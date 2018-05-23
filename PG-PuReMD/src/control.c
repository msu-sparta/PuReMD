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
  #include "control.h"
  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_control.h"
  #include "reax_tool_box.h"
#endif


void Read_Control_File( const char * const control_file, control_params* control,
        output_controls *out_control )
{
    FILE *fp;
    char *s, **tmp;
    int c, i, ival;
    real val;

    fp = sfopen( control_file, "r", "Read_Control_File::fp" );

    strcpy( control->sim_name, "default.sim" );
    control->ensemble = NVE;
    control->nsteps = 0;
    control->dt = 0.25;
    control->nprocs = 1;
    control->procs_by_dim[0] = 1;
    control->procs_by_dim[1] = 1;
    control->procs_by_dim[2] = 1;
    control->geo_format = 1;
    control->gpus_per_node = 1;

    control->random_vel = 0;
    control->restart = 0;
    out_control->restart_format = WRITE_BINARY;
    out_control->restart_freq = 0;
    control->reposition_atoms = 0;
    control->restrict_bonds = 0;
    control->remove_CoM_vel = 25;
    out_control->debug_level = 0;
    out_control->energy_update_freq = 0;

    control->reneighbor = 1;
    control->bond_cut = 5.0;
    control->vlist_cut = control->nonb_cut;
    control->bg_cut = 0.3;
    control->thb_cut = 0.001;
    control->hbond_cut = 0.0;

    control->tabulate = 0;

    control->charge_method = QEQ_CM;
    control->charge_freq = 1;
    control->cm_q_net = 0.0;
    control->cm_solver_type = GMRES_S;
    control->cm_solver_max_iters = 100;
    control->cm_solver_restart = 50;
    control->cm_solver_q_err = 0.000001;
    control->cm_domain_sparsify_enabled = FALSE;
    control->cm_init_guess_extrap1 = 3;
    control->cm_init_guess_extrap2 = 2;
    control->cm_domain_sparsity = 1.0;
    control->cm_solver_pre_comp_type = DIAG_PC;
    control->cm_solver_pre_comp_sweeps = 3;
    control->cm_solver_pre_comp_refactor = 100;
    control->cm_solver_pre_comp_droptol = 0.01;
    control->cm_solver_pre_app_type = TRI_SOLVE_PA;
    control->cm_solver_pre_app_jacobi_iters = 50;

    control->T_init = 0.;
    control->T_final = 300.;
    control->Tau_T = 500.0;
    control->T_mode = 0;
    control->T_rate = 1.;
    control->T_freq = 1.;

    control->P[0] = control->P[1] = control->P[2] = 0.000101325;
    control->Tau_P[0] = control->Tau_P[1] = control->Tau_P[2] = 500.0;
    control->Tau_PT[0] = control->Tau_PT[1] = control->Tau_PT[2] = 500.0;
    control->compressibility = 1.0;
    control->press_mode = 0;

    out_control->write_steps = 0;
    out_control->traj_compress = 0;
    out_control->traj_method = REG_TRAJ;
    strcpy( out_control->traj_title, "default_title" );
    out_control->atom_info = 0;
    out_control->bond_info = 0;
    out_control->angle_info = 0;

    control->molecular_analysis = 0;
    control->dipole_anal = 0;
    control->freq_dipole_anal = 0;
    control->diffusion_coef = 0;
    control->freq_diffusion_coef = 0;
    control->restrict_type = 0;

    /* memory allocations */
    s = (char*) smalloc( sizeof(char) * MAX_LINE, "Read_Control_File::s" );
    tmp = (char**) smalloc( sizeof(char*) * MAX_TOKENS, "Read_Control_File::tmp" );
    for (i = 0; i < MAX_TOKENS; i++)
    {
        tmp[i] = (char*) smalloc( sizeof(char) * MAX_LINE, "Read_Control_File::tmp[i]" );
    }

    /* read control parameters file */
    while( fgets( s, MAX_LINE, fp ) )
    {
        c = Tokenize( s, &tmp );

        if ( c > 0 )
        {
            if ( strcmp(tmp[0], "simulation_name") == 0 )
            {
                strcpy( control->sim_name, tmp[1] );
            }
            else if ( strcmp(tmp[0], "ensemble_type") == 0 )
            {
                ival = atoi(tmp[1]);
                control->ensemble = ival;
            }
            else if ( strcmp(tmp[0], "nsteps") == 0 )
            {
                ival = atoi(tmp[1]);
                control->nsteps = ival;
            }
            else if ( strcmp(tmp[0], "dt") == 0)
            {
                val = atof(tmp[1]);
                control->dt = val * 1.e-3;  // convert dt from fs to ps!
            }
            else if ( strcmp(tmp[0], "gpus_per_node") == 0 )
            {
                ival = atoi(tmp[1]);
                control->gpus_per_node = ival;
            }
            else if ( strcmp(tmp[0], "proc_by_dim") == 0 )
            {
                if ( c < 4 )
                {
                    fprintf( stderr, "[ERROR] invalid number of control file parameters (procs_by_dim). terminating!\n" );
                    MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
                }

                ival = atoi(tmp[1]);
                control->procs_by_dim[0] = ival;
                ival = atoi(tmp[2]);
                control->procs_by_dim[1] = ival;
                ival = atoi(tmp[3]);
                control->procs_by_dim[2] = ival;

                control->nprocs = control->procs_by_dim[0] * control->procs_by_dim[1] *
                        control->procs_by_dim[2];
            }
            //else if( strcmp(tmp[0], "restart") == 0 ) {
            //  ival = atoi(tmp[1]);
            //  control->restart = ival;
            //}
            //else if( strcmp(tmp[0], "restart_from") == 0 ) {
            //  strcpy( control->restart_from, tmp[1] );
            //}
            else if ( strcmp(tmp[0], "random_vel") == 0 )
            {
                ival = atoi(tmp[1]);
                control->random_vel = ival;
            }
            else if ( strcmp(tmp[0], "restart_format") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->restart_format = ival;
            }
            else if ( strcmp(tmp[0], "restart_freq") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->restart_freq = ival;
            }
            else if ( strcmp(tmp[0], "reposition_atoms") == 0 )
            {
                ival = atoi(tmp[1]);
                control->reposition_atoms = ival;
            }
            else if ( strcmp(tmp[0], "restrict_bonds") == 0 )
            {
                ival = atoi( tmp[1] );
                control->restrict_bonds = ival;
            }
            else if ( strcmp(tmp[0], "remove_CoM_vel") == 0 )
            {
                ival = atoi(tmp[1]);
                control->remove_CoM_vel = ival;
            }
            else if ( strcmp(tmp[0], "debug_level") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->debug_level = ival;
            }
            else if ( strcmp(tmp[0], "energy_update_freq") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->energy_update_freq = ival;
            }
            else if ( strcmp(tmp[0], "reneighbor") == 0 )
            {
                ival = atoi( tmp[1] );
                control->reneighbor = ival;
            }
            else if ( strcmp(tmp[0], "vlist_buffer") == 0 )
            {
                val = atof(tmp[1]);
                control->vlist_cut = val + control->nonb_cut;
            }
            else if ( strcmp(tmp[0], "nbrhood_cutoff") == 0 )
            {
                val = atof(tmp[1]);
                control->bond_cut = val;
            }
            else if ( strcmp(tmp[0], "bond_graph_cutoff") == 0 )
            {
                val = atof(tmp[1]);
                control->bg_cut = val;
            }
            else if ( strcmp(tmp[0], "thb_cutoff") == 0 )
            {
                val = atof(tmp[1]);
                control->thb_cut = val;
            }
            else if ( strcmp(tmp[0], "hbond_cutoff") == 0 )
            {
                val = atof( tmp[1] );
                control->hbond_cut = val;
            }
            else if ( strcmp(tmp[0], "ghost_cutoff") == 0 )
            {
                val = atof(tmp[1]);
                control->user_ghost_cut = val;
            }
            else if ( strcmp(tmp[0], "tabulate_long_range") == 0 )
            {
                ival = atoi( tmp[1] );
                control->tabulate = ival;
            }
            else if ( strcmp(tmp[0], "charge_method") == 0 )
            {
                ival = atoi( tmp[1] );
                control->charge_method = ival;
            }
            else if ( strcmp(tmp[0], "charge_freq") == 0 )
            {
                ival = atoi( tmp[1] );
                control->charge_freq = ival;
            }
            else if ( strcmp(tmp[0], "cm_q_net") == 0 )
            {
                val = atof( tmp[1] );
                control->cm_q_net = val;
            }
            else if ( strcmp(tmp[0], "cm_solver_type") == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_type = ival;
            }
            else if ( strcmp(tmp[0], "cm_solver_max_iters") == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_max_iters = ival;
            }
            else if ( strcmp(tmp[0], "cm_solver_restart") == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_restart = ival;
            }
            else if ( strcmp(tmp[0], "cm_solver_q_err") == 0 )
            {
                val = atof( tmp[1] );
                control->cm_solver_q_err = val;
            }
            else if ( strcmp(tmp[0], "cm_domain_sparsity") == 0 )
            {
                val = atof( tmp[1] );
                control->cm_domain_sparsity = val;
                if ( val < 1.0 )
                {
                    control->cm_domain_sparsify_enabled = TRUE;
                }
            }
            else if ( strncmp(tmp[0], "cm_init_guess_extrap1", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_init_guess_extrap1 = ival;
            }
            else if ( strncmp(tmp[0], "cm_init_guess_extrap2", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_init_guess_extrap2 = ival;
            }
            else if ( strcmp(tmp[0], "cm_solver_pre_comp_type") == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_comp_type = ival;
            }
            else if ( strcmp(tmp[0], "cm_solver_pre_comp_refactor") == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_comp_refactor = ival;
            }
            else if ( strcmp(tmp[0], "cm_solver_pre_comp_droptol") == 0 )
            {
                val = atof( tmp[1] );
                control->cm_solver_pre_comp_droptol = val;
            }
            else if ( strcmp(tmp[0], "cm_solver_pre_comp_sweeps") == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_comp_sweeps = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_sai_thres", MAX_LINE) == 0 )
            {
                val = atof( tmp[1] );
                control->cm_solver_pre_comp_sai_thres = val;
            }
            else if ( strcmp(tmp[0], "cm_solver_pre_app_type") == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_app_type = ival;
            }
            else if ( strcmp(tmp[0], "cm_solver_pre_app_jacobi_iters") == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_app_jacobi_iters = ival;
            }
            else if ( strcmp(tmp[0], "temp_init") == 0 )
            {
                val = atof(tmp[1]);
                control->T_init = val;

                if ( control->T_init < 0.1 )
                {
                    control->T_init = 0.1;
                }
            }
            else if ( strcmp(tmp[0], "temp_final") == 0 )
            {
                val = atof(tmp[1]);
                control->T_final = val;

                if ( control->T_final < 0.1 )
                {
                    control->T_final = 0.1;
                }
            }
            else if ( strcmp(tmp[0], "t_mass") == 0 )
            {
                val = atof(tmp[1]);
                control->Tau_T = val * 1.e-3;    // convert t_mass from fs to ps
            }
            else if ( strcmp(tmp[0], "t_mode") == 0 )
            {
                ival = atoi(tmp[1]);
                control->T_mode = ival;
            }
            else if ( strcmp(tmp[0], "t_rate") == 0 )
            {
                val = atof(tmp[1]);
                control->T_rate = val;
            }
            else if ( strcmp(tmp[0], "t_freq") == 0 )
            {
                val = atof(tmp[1]);
                control->T_freq = val;
            }
            else if ( strcmp(tmp[0], "pressure") == 0 )
            {
                if ( control->ensemble == iNPT )
                {
                    control->P[0] = control->P[1] = control->P[2] = atof(tmp[1]);
                }
                else if ( control->ensemble == sNPT )
                {
                    control->P[0] = atof(tmp[1]);
                    control->P[1] = atof(tmp[2]);
                    control->P[2] = atof(tmp[3]);
                }
            }
            else if ( strcmp(tmp[0], "p_mass") == 0 )
            {
                // convert p_mass from fs to ps
                if ( control->ensemble == iNPT )
                {
                    control->Tau_P[0] = control->Tau_P[1] = control->Tau_P[2] =
                            atof(tmp[1]) * 1.e-3;
                }
                else if ( control->ensemble == sNPT )
                {
                    control->Tau_P[0] = atof(tmp[1]) * 1.e-3;
                    control->Tau_P[1] = atof(tmp[2]) * 1.e-3;
                    control->Tau_P[2] = atof(tmp[3]) * 1.e-3;
                }
            }
            else if ( strcmp(tmp[0], "pt_mass") == 0 )
            {
                val = atof(tmp[1]);
                control->Tau_PT[0] = control->Tau_PT[1] = control->Tau_PT[2] =
                                         val * 1.e-3;  // convert pt_mass from fs to ps
            }
            else if ( strcmp(tmp[0], "compress") == 0 )
            {
                val = atof(tmp[1]);
                control->compressibility = val;
            }
            else if ( strcmp(tmp[0], "press_mode") == 0 )
            {
                ival = atoi(tmp[1]);
                control->press_mode = ival;
            }
            else if ( strcmp(tmp[0], "geo_format") == 0 )
            {
                ival = atoi( tmp[1] );
                control->geo_format = ival;
            }
            else if ( strcmp(tmp[0], "write_freq") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->write_steps = ival;
            }
            else if ( strcmp(tmp[0], "traj_compress") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->traj_compress = ival;
            }
            else if ( strcmp(tmp[0], "traj_method") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->traj_method = ival;
            }
            else if ( strcmp(tmp[0], "traj_title") == 0 )
            {
                strcpy( out_control->traj_title, tmp[1] );
            }
            else if ( strcmp(tmp[0], "atom_info") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->atom_info += ival * 4;
            }
            else if ( strcmp(tmp[0], "atom_velocities") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->atom_info += ival * 2;
            }
            else if ( strcmp(tmp[0], "atom_forces") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->atom_info += ival * 1;
            }
            else if ( strcmp(tmp[0], "bond_info") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->bond_info = ival;
            }
            else if ( strcmp(tmp[0], "angle_info") == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->angle_info = ival;
            }
            else if ( strcmp(tmp[0], "molecular_analysis") == 0 )
            {
                ival = atoi(tmp[1]);
                control->molecular_analysis = ival;
            }
            else if ( strcmp(tmp[0], "ignore") == 0 )
            {
                control->num_ignored = atoi(tmp[1]);
                for ( i = 0; i < control->num_ignored; ++i )
                {
                    control->ignore[atoi(tmp[i + 2])] = 1;
                }
            }
            else if ( strcmp(tmp[0], "dipole_anal") == 0 )
            {
                ival = atoi(tmp[1]);
                control->dipole_anal = ival;
            }
            else if ( strcmp(tmp[0], "freq_dipole_anal") == 0 )
            {
                ival = atoi(tmp[1]);
                control->freq_dipole_anal = ival;
            }
            else if ( strcmp(tmp[0], "diffusion_coef") == 0 )
            {
                ival = atoi(tmp[1]);
                control->diffusion_coef = ival;
            }
            else if ( strcmp(tmp[0], "freq_diffusion_coef") == 0 )
            {
                ival = atoi(tmp[1]);
                control->freq_diffusion_coef = ival;
            }
            else if ( strcmp(tmp[0], "restrict_type") == 0 )
            {
                ival = atoi(tmp[1]);
                control->restrict_type = ival;
            }
            else
            {
                fprintf( stderr, "[ERROR] unknown control file parameter (%s)\n", tmp[0] );
                MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
            }
        }
    }

    if ( ferror( fp ) )
    {
        fprintf( stderr, "[ERROR] parsing control file failed (I/O error). TERMINATING...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }

    /* determine target T */
    if ( control->T_mode == 0 )
    {
        control->T = control->T_final;
    }
    else
    {
        control->T = control->T_init;
    }

    /* free memory allocations at the top */
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        sfree( tmp[i], "Read_Control_File::tmp[i]" );
    }
    sfree( tmp, "Read_Control_File::tmp" );
    sfree( s, "Read_Control_File::s" );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "control file read\n" );
#endif
}
