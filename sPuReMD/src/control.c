/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator

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

#include "control.h"

#include <ctype.h>
#include <zlib.h>

#include "traj.h"
#include "tool_box.h"


void Read_Control_File( FILE* fp, reax_system *system, control_params* control,
        output_controls *out_control )
{
    char *s, **tmp;
    int c, i, ival;
    real val;

    /* assign default values */
    strncpy( control->sim_name, "default.sim", MAX_STR );

    control->restart = 0;
    out_control->restart_format = WRITE_BINARY;
    out_control->restart_freq = 0;
    strncpy( control->restart_from, "default.res", MAX_STR );
    out_control->restart_freq = 0;
    control->random_vel = 0;

    control->reposition_atoms = 0;

    control->ensemble = NVE;
    control->nsteps = 0;
    control->dt = 0.25;

    control->geo_format = PDB;
    control->restrict_bonds = 0;

    control->periodic_boundaries = 1;

    control->reneighbor = 1;

    /* interaction cutoffs from force field global paramters */
    control->bo_cut = 0.01 * system->reaxprm.gp.l[29];
    control->nonb_low = system->reaxprm.gp.l[11];
    control->nonb_cut = system->reaxprm.gp.l[12];

    /* defaults values for other cutoffs */
    control->vlist_cut = control->nonb_cut;
    control->bond_cut = 5.0;
    control->bg_cut = 0.3;
    control->thb_cut = 0.001;
    control->hbond_cut = 0.0;

    control->tabulate = 0;

    control->charge_method = QEQ_CM;
    control->cm_q_net = 0.0;
    control->cm_solver_type = GMRES_S;
    control->cm_solver_max_iters = 100;
    control->cm_solver_restart = 50;
    control->cm_solver_q_err = 0.000001;
    control->cm_domain_sparsify_enabled = FALSE;
    control->cm_domain_sparsity = 1.0;
    control->cm_init_guess_extrap1 = 3;
    control->cm_init_guess_extrap2 = 2;
    control->cm_solver_pre_comp_type = ICHOLT_PC;
    control->cm_solver_pre_comp_sweeps = 3;
    control->cm_solver_pre_comp_sai_thres = 0.1;
    control->cm_solver_pre_comp_refactor = 100;
    control->cm_solver_pre_comp_droptol = 0.01;
    control->cm_solver_pre_app_type = TRI_SOLVE_PA;
    control->cm_solver_pre_app_jacobi_iters = 50;

    control->T_init = 0.;
    control->T_final = 300.;
    control->Tau_T = 1.0;
    control->T_mode = 0.;
    control->T_rate = 1.;
    control->T_freq = 1.;

    control->P[0] = 0.000101325;
    control->P[1] = 0.000101325;
    control->P[2] = 0.000101325;
    control->Tau_P[0] = 500.0;
    control->Tau_P[1] = 500.0;
    control->Tau_P[2] = 500.0;
    control->Tau_PT = 500.0;
    control->compressibility = 1.0;
    control->press_mode = 0;

    control->remove_CoM_vel = 25;

    out_control->debug_level = 0;
    out_control->energy_update_freq = 0;

    out_control->write_steps = 0;
    out_control->traj_compress = 0;
    out_control->write = fprintf;
    out_control->traj_format = 0;
    out_control->write_header =
        (int (*)( reax_system*, control_params*,
                  static_storage*, void* )) Write_Custom_Header;
    out_control->append_traj_frame =
        (int (*)( reax_system*, control_params*, simulation_data*,
                  static_storage*, reax_list **, void* )) Append_Custom_Frame;

    strncpy( out_control->traj_title, "default_title", 81 );
    out_control->atom_format = 0;
    out_control->bond_info = 0;
    out_control->angle_info = 0;

    control->molec_anal = NO_ANALYSIS;
    control->freq_molec_anal = 0;
    control->num_ignored = 0;
    memset( control->ignore, 0, sizeof(int) * MAX_ATOM_TYPES );

    control->dipole_anal = 0;
    control->freq_dipole_anal = 0;

    control->diffusion_coef = 0;
    control->freq_diffusion_coef = 0;
    control->restrict_type = 0;

    /* memory allocations */
    s = (char*) smalloc( sizeof(char) * MAX_LINE, "Read_Control_File::s" );
    tmp = (char**) smalloc( sizeof(char*) * MAX_TOKENS, "Read_Control_File::tmp" );
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        tmp[i] = (char*) smalloc( sizeof(char) * MAX_LINE,
                "Read_Control_File::tmp[i]" );
    }

    /* read control parameters file */
    while ( fgets( s, MAX_LINE, fp ) )
    {
        c = Tokenize( s, &tmp );

        if ( c > 0 )
        {
            if ( strncmp(tmp[0], "simulation_name", MAX_LINE) == 0 )
            {
                strncpy( control->sim_name, tmp[1], MAX_STR );
            }
            //else if( strncmp(tmp[0], "restart", MAX_LINE) == 0 ) {
            //  ival = atoi(tmp[1]);
            //  control->restart = ival;
            //}
            else if ( strncmp(tmp[0], "restart_format", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->restart_format = ival;
            }
            else if ( strncmp(tmp[0], "restart_freq", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->restart_freq = ival;
            }
            else if ( strncmp(tmp[0], "random_vel", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->random_vel = ival;
            }
            else if ( strncmp(tmp[0], "reposition_atoms", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->reposition_atoms = ival;
            }
            else if ( strncmp(tmp[0], "ensemble_type", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->ensemble = ival;
            }
            else if ( strncmp(tmp[0], "nsteps", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->nsteps = ival;
            }
            else if ( strncmp(tmp[0], "dt", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->dt = val * 1.e-3;  // convert dt from fs to ps!
            }
            else if ( strncmp(tmp[0], "periodic_boundaries", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->periodic_boundaries = ival;
            }
            else if ( strncmp(tmp[0], "geo_format", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->geo_format = ival;
            }
            else if ( strncmp(tmp[0], "restrict_bonds", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->restrict_bonds = ival;
            }
            else if ( strncmp(tmp[0], "tabulate_long_range", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->tabulate = ival;
            }
            else if ( strncmp(tmp[0], "reneighbor", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->reneighbor = ival;
            }
            else if ( strncmp(tmp[0], "vlist_buffer", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->vlist_cut = val + control->nonb_cut;
            }
            else if ( strncmp(tmp[0], "nbrhood_cutoff", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->bond_cut = val;
            }
            else if ( strncmp(tmp[0], "thb_cutoff", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->thb_cut = val;
            }
            else if ( strncmp(tmp[0], "hbond_cutoff", MAX_LINE) == 0 )
            {
                val = atof( tmp[1] );
                control->hbond_cut = val;
            }
            else if ( strncmp(tmp[0], "charge_method", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->charge_method = ival;
            }
            else if ( strncmp(tmp[0], "cm_q_net", MAX_LINE) == 0 )
            {
                val = atof( tmp[1] );
                control->cm_q_net = val;
            }
            else if ( strncmp(tmp[0], "cm_solver_type", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_type = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_max_iters", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_max_iters = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_restart", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_restart = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_q_err", MAX_LINE) == 0 )
            {
                val = atof( tmp[1] );
                control->cm_solver_q_err = val;
            }
            else if ( strncmp(tmp[0], "cm_domain_sparsity", MAX_LINE) == 0 )
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
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_type", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_comp_type = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_refactor", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_comp_refactor = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_droptol", MAX_LINE) == 0 )
            {
                val = atof( tmp[1] );
                control->cm_solver_pre_comp_droptol = val;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_sweeps", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_comp_sweeps = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_comp_sai_thres", MAX_LINE) == 0 )
            {
                val = atof( tmp[1] );
                control->cm_solver_pre_comp_sai_thres = val;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_app_type", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_app_type = ival;
            }
            else if ( strncmp(tmp[0], "cm_solver_pre_app_jacobi_iters", MAX_LINE) == 0 )
            {
                ival = atoi( tmp[1] );
                control->cm_solver_pre_app_jacobi_iters = ival;
            }
            else if ( strncmp(tmp[0], "temp_init", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->T_init = val;

                if ( control->T_init < 0.001 )
                {
                    control->T_init = 0.001;
                }
            }
            else if ( strncmp(tmp[0], "temp_final", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->T_final = val;

                if ( control->T_final < 0.1 )
                {
                    control->T_final = 0.1;
                }
            }
            else if ( strncmp(tmp[0], "t_mass", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                /* convert t_mass from fs to ps */
                control->Tau_T = val * 1.e-3;
            }
            else if ( strncmp(tmp[0], "t_mode", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->T_mode = ival;
            }
            else if ( strncmp(tmp[0], "t_rate", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->T_rate = val;
            }
            else if ( strncmp(tmp[0], "t_freq", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->T_freq = val;
            }
            else if ( strncmp(tmp[0], "pressure", MAX_LINE) == 0 )
            {
                if ( control->ensemble == iNPT )
                {
                    val = atof(tmp[1]);
                    control->P[0] = control->P[1] = control->P[2] = val;
                }
                else if ( control->ensemble == sNPT )
                {
                    val = atof(tmp[1]);
                    control->P[0] = val;

                    val = atof(tmp[2]);
                    control->P[1] = val;

                    val = atof(tmp[3]);
                    control->P[2] = val;
                }
            }
            else if ( strncmp(tmp[0], "p_mass", MAX_LINE) == 0 )
            {
                if ( control->ensemble == iNPT )
                {
                    val = atof(tmp[1]);
                    control->Tau_P[0] = val * 1.e-3;   // convert p_mass from fs to ps
                }
                else if ( control->ensemble == sNPT )
                {
                    val = atof(tmp[1]);
                    control->Tau_P[0] = val * 1.e-3;   // convert p_mass from fs to ps

                    val = atof(tmp[2]);
                    control->Tau_P[1] = val * 1.e-3;   // convert p_mass from fs to ps

                    val = atof(tmp[3]);
                    control->Tau_P[2] = val * 1.e-3;   // convert p_mass from fs to ps
                }
            }
            else if ( strncmp(tmp[0], "pt_mass", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->Tau_PT = val * 1.e-3;  // convert pt_mass from fs to ps
            }
            else if ( strncmp(tmp[0], "compress", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->compressibility = val;
            }
            else if ( strncmp(tmp[0], "press_mode", MAX_LINE) == 0 )
            {
                val = atoi(tmp[1]);
                control->press_mode = val;
            }
            else if ( strncmp(tmp[0], "remove_CoM_vel", MAX_LINE) == 0 )
            {
                val = atoi(tmp[1]);
                control->remove_CoM_vel = val;
            }
            else if ( strncmp(tmp[0], "debug_level", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->debug_level = ival;
            }
            else if ( strncmp(tmp[0], "energy_update_freq", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->energy_update_freq = ival;
            }
            else if ( strncmp(tmp[0], "write_freq", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->write_steps = ival;
            }
            else if ( strncmp(tmp[0], "traj_compress", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->traj_compress = ival;

                if ( out_control->traj_compress )
                    out_control->write = (int (*)(FILE *, const char *, ...)) gzprintf;
                else out_control->write = fprintf;
            }
            else if ( strncmp(tmp[0], "traj_format", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->traj_format = ival;

                if ( out_control->traj_format == 0 )
                {
                    out_control->write_header =
                        (int (*)( reax_system*, control_params*,
                                  static_storage*, void* )) Write_Custom_Header;
                    out_control->append_traj_frame =
                        (int (*)(reax_system*, control_params*, simulation_data*,
                                 static_storage*, reax_list **, void*)) Append_Custom_Frame;
                }
                else if ( out_control->traj_format == 1 )
                {
                    out_control->write_header =
                        (int (*)( reax_system*, control_params*,
                                  static_storage*, void* )) Write_xyz_Header;
                    out_control->append_traj_frame =
                        (int (*)( reax_system*,  control_params*, simulation_data*,
                                  static_storage*, reax_list **, void* )) Append_xyz_Frame;
                }
            }
            else if ( strncmp(tmp[0], "traj_title", MAX_LINE) == 0 )
            {
                strncpy( out_control->traj_title, tmp[1], 81 );
            }
            else if ( strncmp(tmp[0], "atom_info", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->atom_format += ival * 4;
            }
            else if ( strncmp(tmp[0], "atom_velocities", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->atom_format += ival * 2;
            }
            else if ( strncmp(tmp[0], "atom_forces", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->atom_format += ival * 1;
            }
            else if ( strncmp(tmp[0], "bond_info", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->bond_info = ival;
            }
            else if ( strncmp(tmp[0], "angle_info", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                out_control->angle_info = ival;
            }
            else if ( strncmp(tmp[0], "test_forces", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
            }
            else if ( strncmp(tmp[0], "molec_anal", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->molec_anal = ival;
            }
            else if ( strncmp(tmp[0], "freq_molec_anal", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->freq_molec_anal = ival;
            }
            else if ( strncmp(tmp[0], "bond_graph_cutoff", MAX_LINE) == 0 )
            {
                val = atof(tmp[1]);
                control->bg_cut = val;
            }
            else if ( strncmp(tmp[0], "ignore", MAX_LINE) == 0 )
            {
                control->num_ignored = atoi(tmp[1]);
                for ( i = 0; i < control->num_ignored; ++i )
                    control->ignore[atoi(tmp[i + 2])] = 1;
            }
            else if ( strncmp(tmp[0], "dipole_anal", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->dipole_anal = ival;
            }
            else if ( strncmp(tmp[0], "freq_dipole_anal", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->freq_dipole_anal = ival;
            }
            else if ( strncmp(tmp[0], "diffusion_coef", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->diffusion_coef = ival;
            }
            else if ( strncmp(tmp[0], "freq_diffusion_coef", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->freq_diffusion_coef = ival;
            }
            else if ( strncmp(tmp[0], "restrict_type", MAX_LINE) == 0 )
            {
                ival = atoi(tmp[1]);
                control->restrict_type = ival;
            }
            else
            {
                fprintf( stderr, "WARNING: unknown parameter %s\n", tmp[0] );
                exit( UNKNOWN_OPTION );
            }
        }
    }

    if (ferror(fp))
    {
        fprintf( stderr, "Error reading control file. Terminating.\n" );
        exit( INVALID_INPUT );
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

    /* derived cutoffs */
    control->nonb_sp_cut = control->nonb_cut * control->cm_domain_sparsity;

    system->g.cell_size = control->vlist_cut / 2.0;
    for ( i = 0; i < 3; ++i )
    {
        system->g.spread[i] = 2;
    }

    /* free memory allocations at the top */
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        sfree( tmp[i], "Read_Control_File::tmp[i]" );
    }
    sfree( tmp, "Read_Control_File::tmp" );
    sfree( s, "Read_Control_File::s" );

#if defined(DEBUG_FOCUS)
    fprintf( stderr,
             "en=%d steps=%d dt=%.5f opt=%d T=%.5f P=%.5f %.5f %.5f\n",
             control->ensemble, control->nsteps, control->dt, control->tabulate,
             control->T, control->P[0], control->P[1], control->P[2] );

    fprintf(stderr, "control file read\n" );
#endif
}
