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
  #include "control.h"

  #include "tool_box.h"
#elif defined(LAMMPS_REAX)
  #include "reax_control.h"

  #include "reax_tool_box.h"
#endif


void Read_Control_File( const char * const control_file, control_params * const control,
        output_controls * const out_control )
{
    FILE *fp;
    char *s, **tmp;
    int c, i, ival;
    real val;

    fp = sfopen( control_file, "r", __FILE__, __LINE__ );

    /* assign default values */
    strncpy( control->sim_name, "default.sim", sizeof(control->sim_name) - 1 );
    control->sim_name[sizeof(control->sim_name) - 1] = '\0';

    control->ensemble = NVE;
    control->nsteps = 0;
    control->dt = 0.25;
    control->num_threads_set = FALSE;
    control->nprocs = 1;
    control->procs_by_dim[0] = 1;
    control->procs_by_dim[1] = 1;
    control->procs_by_dim[2] = 1;
    control->geo_format = 1;
    control->gpus_per_node = 1;
    control->gpu_streams = MAX_CUDA_STREAMS;

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
    control->cm_solver_type = CG_S;
    control->cm_solver_max_iters = 1000;
    control->cm_solver_restart = 100;
    control->cm_solver_q_err = 0.000001;
    control->cm_domain_sparsify_enabled = FALSE;
    control->cm_init_guess_extrap1 = 3;
    control->cm_init_guess_extrap2 = 2;
    control->cm_domain_sparsity = 1.0;
    control->cm_solver_pre_comp_type = JACOBI_PC;
    control->cm_solver_pre_comp_sweeps = 3;
    control->cm_solver_pre_comp_refactor = 1;
    control->cm_solver_pre_comp_droptol = 0.01;
    control->cm_solver_pre_app_type = TRI_SOLVE_PA;
    control->cm_solver_pre_app_jacobi_iters = 50;
    control->polarization_energy_enabled = TRUE;

    control->T_init = 0.0;
    control->T_final = 300.;
    control->Tau_T = 500.0;
    control->T_mode = 0;
    control->T_rate = 1.0;
    control->T_freq = 1.0;

    control->P[0] = 0.000101325;
    control->P[1] = 0.000101325;
    control->P[2] = 0.000101325;
    control->Tau_P[0] = 500.0;
    control->Tau_P[1] = 500.0;
    control->Tau_P[2] = 500.0;
    control->Tau_PT[0] = 500.0;
    control->Tau_PT[1] = 500.0;
    control->Tau_PT[2] = 500.0;
    control->compressibility = 1.0;
    control->press_mode = 0;

    out_control->write_steps = 0;
    out_control->traj_compress = 0;
    out_control->traj_method = REG_TRAJ;
    strncpy( out_control->traj_title, "default_title", sizeof(out_control->traj_title) - 1 );
    out_control->traj_title[sizeof(out_control->traj_title) - 1] = '\0';
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
    s = smalloc( sizeof(char) * MAX_LINE, __FILE__, __LINE__ );
    tmp = smalloc( sizeof(char*) * MAX_TOKENS, __FILE__, __LINE__ );
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        tmp[i] = smalloc( sizeof(char) * MAX_LINE, __FILE__, __LINE__ );
    }

    /* read control parameters file */
    while ( fgets( s, MAX_LINE, fp ) )
    {
        c = Tokenize( s, &tmp, MAX_LINE );

        if ( c > 0 )
        {
            if ( strncmp(tmp[0], "simulation_name", MAX_LINE) == 0 )
            {
                strncpy( control->sim_name, tmp[1], sizeof(control->sim_name) - 1 );
                control->sim_name[sizeof(control->sim_name) - 1] = '\0';
            }
            else if ( strncmp(tmp[0], "ensemble_type", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->ensemble = ival;
            }
            else if ( strncmp(tmp[0], "nsteps", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->nsteps = ival;
            }
            else if ( strncmp(tmp[0], "dt", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->dt = val * 1.e-3;  // convert dt from fs to ps!
            }
            else if ( strncmp(tmp[0], "num_threads", MAX_LINE) == 0 )
            {
                control->num_threads = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->num_threads_set = TRUE;
            }
            else if ( strncmp(tmp[0], "gpus_per_node", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->gpus_per_node = ival;
            }
            else if ( strncmp(tmp[0], "gpu_streams", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->gpu_streams = ival;

                if ( control->gpu_streams <= 0 || control->gpu_streams > MAX_CUDA_STREAMS )
                {
                    fprintf( stderr, "[ERROR] invalid control file value for gpu_streams (0 < gpu_streams <= %d). Terminating...\n",
                            MAX_CUDA_STREAMS );
                    exit( INVALID_INPUT );
                }
            }
            else if ( strncmp(tmp[0], "proc_by_dim", MAX_LINE) == 0 )
            {
                if ( c < 4 )
                {
                    fprintf( stderr, "[ERROR] invalid number of control file parameters (procs_by_dim). terminating!\n" );
                    MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
                }

                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->procs_by_dim[0] = ival;
                ival = sstrtol( tmp[2], __FILE__, __LINE__ );
                control->procs_by_dim[1] = ival;
                ival = sstrtol( tmp[3], __FILE__, __LINE__ );
                control->procs_by_dim[2] = ival;

                control->nprocs = control->procs_by_dim[0] * control->procs_by_dim[1] *
                        control->procs_by_dim[2];
            }
            else if ( strncmp(tmp[0], "periodic_boundaries", MAX_LINE) == 0 )
            {
                // skip since not supported in distributed memory code
                ;
            }
//            else if( strncmp(tmp[0], "restart", MAX_LINE) == 0 )
//            {
//                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
//                control->restart = ival;
//            }
//            else if( strncmp(tmp[0], "restart_from", MAX_LINE) == 0 )
//            {
//                strncpy( control->restart_from, tmp[1], sizeof(control->restart_from) - 1 );
//                control->restart_from[sizeof(control->restart_from) - 1] = '\0';
//            }
            else if ( strncmp(tmp[0], "random_vel", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->random_vel = ival;
            }
            else if ( strncmp(tmp[0], "restart_format", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->restart_format = ival;
            }
            else if ( strncmp(tmp[0], "restart_freq", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->restart_freq = ival;
            }
            else if ( strncmp(tmp[0], "reposition_atoms", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->reposition_atoms = ival;
            }
            else if ( strncmp(tmp[0], "restrict_bonds", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->restrict_bonds = ival;
            }
            else if ( strncmp(tmp[0], "remove_CoM_vel", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->remove_CoM_vel = ival;
            }
            else if ( strncmp(tmp[0], "debug_level", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->debug_level = ival;
            }
            else if ( strncmp(tmp[0], "energy_update_freq", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->energy_update_freq = ival;
            }
            else if ( strncmp(tmp[0], "reneighbor", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->reneighbor = ival;
            }
            else if ( strncmp(tmp[0], "vlist_buffer", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->vlist_cut = val + control->nonb_cut;
            }
            else if ( strncmp(tmp[0], "nbrhood_cutoff", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->bond_cut = val;
            }
            else if ( strncmp(tmp[0], "bond_graph_cutoff", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->bg_cut = val;
            }
            else if ( strncmp(tmp[0], "thb_cutoff", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->thb_cut = val;
            }
            else if ( strncmp(tmp[0], "hbond_cutoff", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->hbond_cut = val;
            }
            else if ( strncmp(tmp[0], "ghost_cutoff", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->user_ghost_cut = val;
            }
            else if ( strncmp(tmp[0], "tabulate_long_range", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->tabulate = ival;
            }
            else if ( strncmp(tmp[0], "charge_method", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->charge_method = ival;
            }
            else if ( strncmp(tmp[0], "charge_freq", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->charge_freq = ival;
            }
            else if ( strncmp(tmp[0], "cm_q_net", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->cm_q_net = val;
            }
            else if ( strncmp(tmp[0], "cm_solver_type", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_type = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_max_iters", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_max_iters = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_restart", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_restart = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_q_err", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_q_err = val;
            }
            else if ( strncmp(tmp[0], "cm_domain_sparsity", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->cm_domain_sparsity = val;
                if ( val < 1.0 )
                {
                    control->cm_domain_sparsify_enabled = TRUE;
                }
            }
            else if ( strncmp(tmp[0], "cm_init_guess_extrap1", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_init_guess_extrap1 = ival;
            }
            else if ( strncmp(tmp[0], "cm_init_guess_extrap2", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_init_guess_extrap2 = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_type", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_pre_comp_type = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_refactor", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_pre_comp_refactor = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_droptol", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_pre_comp_droptol = val;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_sweeps", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_pre_comp_sweeps = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_sai_thres", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_pre_comp_sai_thres = val;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_app_type", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_pre_app_type = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_app_jacobi_iters", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->cm_solver_pre_app_jacobi_iters = ival;
            }
            else if ( strncmp(tmp[0], "include_polarization_energy", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->polarization_energy_enabled = ival;
            }
            else if ( strncmp(tmp[0], "temp_init", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->T_init = val;

                if ( control->T_init < 0.001 )
                {
                    control->T_init = 0.001;
                }
            }
            else if ( strncmp(tmp[0], "temp_final", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->T_final = val;

                if ( control->T_final < 0.1 )
                {
                    control->T_final = 0.1;
                }
            }
            else if ( strncmp(tmp[0], "t_mass", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                /* convert from fs to s */
                control->Tau_T = val * 1.0e-15;
            }
            else if ( strncmp(tmp[0], "t_mode", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->T_mode = ival;
            }
            else if ( strncmp(tmp[0], "t_rate", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->T_rate = val;
            }
            else if ( strncmp(tmp[0], "t_freq", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->T_freq = val;
            }
            else if ( strncmp(tmp[0], "pressure", MAX_LINE) == 0 )
            {
                if ( control->ensemble == iNPT )
                {
                    val = sstrtod( tmp[1], __FILE__, __LINE__ );
                    control->P[0] = val;
                    control->P[1] = val;
                    control->P[2] = val;
                }
                else if ( control->ensemble == sNPT )
                {
                    control->P[0] = sstrtod( tmp[1], __FILE__, __LINE__ );
                    control->P[1] = sstrtod( tmp[2], __FILE__, __LINE__ );
                    control->P[2] = sstrtod( tmp[3], __FILE__, __LINE__ );
                }
            }
            else if ( strncmp(tmp[0], "p_mass", MAX_LINE) == 0 )
            {
                // convert p_mass from fs to ps
                if ( control->ensemble == iNPT )
                {
                    val = sstrtod( tmp[1], __FILE__, __LINE__ ) * 1.e-3;
                    control->Tau_P[0] = val;
                    control->Tau_P[1] = val;
                    control->Tau_P[2] = val;
                }
                else if ( control->ensemble == sNPT )
                {
                    control->Tau_P[0] = sstrtod( tmp[1], __FILE__, __LINE__ ) * 1.e-3;
                    control->Tau_P[1] = sstrtod( tmp[2], __FILE__, __LINE__ ) * 1.e-3;
                    control->Tau_P[2] = sstrtod( tmp[3], __FILE__, __LINE__ ) * 1.e-3;
                }
            }
            else if ( strncmp(tmp[0], "pt_mass", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->Tau_PT[0] = control->Tau_PT[1] = control->Tau_PT[2] =
                                         val * 1.e-3;  // convert pt_mass from fs to ps
            }
            else if ( strncmp(tmp[0], "compress", MAX_LINE) == 0 )
            {
                val = sstrtod( tmp[1], __FILE__, __LINE__ );
                control->compressibility = val;
            }
            else if ( strncmp(tmp[0], "press_mode", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->press_mode = ival;
            }
            else if ( strncmp(tmp[0], "geo_format", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->geo_format = ival;
            }
            else if ( strncmp(tmp[0], "write_freq", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->write_steps = ival;
            }
            else if ( strncmp(tmp[0], "traj_compress", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->traj_compress = ival;
            }
            else if ( strncmp(tmp[0], "traj_format", MAX_LINE) == 0 )
            {
                // skip since not applicable to distributed memory code
                ;
            }
            else if ( strncmp(tmp[0], "traj_method", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->traj_method = ival;
            }
            else if ( strncmp(tmp[0], "traj_title", MAX_LINE) == 0 )
            {
                strncpy( out_control->traj_title, tmp[1], sizeof(out_control->traj_title) - 1 );
                out_control->traj_title[sizeof(out_control->traj_title) - 1] = '\0';
            }
            else if ( strncmp(tmp[0], "atom_info", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->atom_info += ival * 4;
            }
            else if ( strncmp(tmp[0], "atom_velocities", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->atom_info += ival * 2;
            }
            else if ( strncmp(tmp[0], "atom_forces", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->atom_info += ival * 1;
            }
            else if ( strncmp(tmp[0], "bond_info", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->bond_info = ival;
            }
            else if ( strncmp(tmp[0], "angle_info", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                out_control->angle_info = ival;
            }
            else if ( strncmp(tmp[0], "test_forces", MAX_LINE) == 0 )
            {
                // skip since not supported in distributed memory code
                ;
            }
            else if ( strncmp(tmp[0], "molecular_analysis", MAX_LINE) == 0 
                    || strncmp(tmp[0], "molec_anal", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->molecular_analysis = ival;
            }
            else if ( strncmp(tmp[0], "ignore", MAX_LINE) == 0 )
            {
                control->num_ignored = sstrtol( tmp[1], __FILE__, __LINE__ );
                for ( i = 0; i < control->num_ignored; ++i )
                {
                    control->ignore[sstrtol( tmp[i + 2], __FILE__, __LINE__ )] = 1;
                }
            }
            else if ( strncmp(tmp[0], "freq_molec_anal", MAX_LINE) == 0 )
            {
                // skip since not supported in distributed memory code
                ;
            }
            else if ( strncmp(tmp[0], "dipole_anal", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->dipole_anal = ival;
            }
            else if ( strncmp(tmp[0], "freq_dipole_anal", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->freq_dipole_anal = ival;
            }
            else if ( strncmp(tmp[0], "diffusion_coef", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->diffusion_coef = ival;
            }
            else if ( strncmp(tmp[0], "freq_diffusion_coef", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
                control->freq_diffusion_coef = ival;
            }
            else if ( strncmp(tmp[0], "restrict_type", MAX_LINE) == 0 )
            {
                ival = sstrtol( tmp[1], __FILE__, __LINE__ );
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
        sfree( tmp[i], __FILE__, __LINE__ );
    }
    sfree( tmp, __FILE__, __LINE__ );
    sfree( s, __FILE__, __LINE__ );

    sfclose( fp, __FILE__, __LINE__ );
}
