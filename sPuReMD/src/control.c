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

#include <ctype.h>

#include "control.h"
#include "traj.h"
#include "tool_box.h"


char Read_Control_File( FILE* fp, reax_system *system, control_params* control,
        output_controls *out_control )
{
    char *s, **tmp;
    int c, i;
    real val;
    int ival;

    /* assign default values */
    strcpy( control->sim_name, "default.sim" );

    control->restart = 0;
    out_control->restart_format = WRITE_BINARY;
    out_control->restart_freq = 0;
    strcpy( control->restart_from, "default.res" );
    out_control->restart_freq = 0;
    control->random_vel = 0;

    control->reposition_atoms = 0;

    control->ensemble = NVE;
    control->nsteps = 0;
    control->dt = 0.25;

    control->geo_format = PDB;
    control->restrict_bonds = 0;

    control->periodic_boundaries = 1;
    control->periodic_images[0] = 0;
    control->periodic_images[1] = 0;
    control->periodic_images[2] = 0;

    control->reneighbor = 1;
    control->vlist_cut = 0;
    control->nbr_cut = 4.;
    control->r_cut = 10.;
    control->r_sp_cut = 10.;
    control->max_far_nbrs = 1000;
    control->bo_cut = 0.01;
    control->thb_cut = 0.001;
    control->hb_cut = 7.50;

    control->tabulate = 0;

    control->qeq_solver_type = GMRES_S;
    control->qeq_solver_q_err = 0.000001;
    control->qeq_domain_sparsify_enabled = FALSE;
    control->qeq_domain_sparsity = 1.0;
    control->pre_comp_type = ICHOLT_PC;
    control->pre_comp_sweeps = 3;
    control->pre_comp_refactor = 100;
    control->pre_comp_droptol = 0.01;
    control->pre_app_type = TRI_SOLVE_PA;
    control->pre_app_jacobi_iters = 50;

    control->T_init = 0.;
    control->T_final = 300.;
    control->Tau_T = 1.0;
    control->T_mode = 0.;
    control->T_rate = 1.;
    control->T_freq = 1.;

    control->P[0] = 0.000101325;
    control->P[1] = 0.000101325;
    control->P[2] = 0.000101325;
    control->Tau_P[0]  = 500.0;
    control->Tau_P[1]  = 500.0;
    control->Tau_P[2]  = 500.0;
    control->Tau_PT = 500.0;
    control->compressibility = 1.0;
    control->press_mode = 0;

    control->remove_CoM_vel = 25;

    out_control->debug_level = 0;
    out_control->energy_update_freq = 10;

    out_control->write_steps = 100;
    out_control->traj_compress = 0;
    out_control->write = fprintf;
    out_control->traj_format = 0;
    out_control->write_header =
        (int (*)( reax_system*, control_params*,
                  static_storage*, void* )) Write_Custom_Header;
    out_control->append_traj_frame =
        (int (*)( reax_system*, control_params*, simulation_data*,
                  static_storage*, list **, void* )) Append_Custom_Frame;

    strcpy( out_control->traj_title, "default_title" );
    out_control->atom_format = 0;
    out_control->bond_info = 0;
    out_control->angle_info = 0;

    control->molec_anal = NO_ANALYSIS;
    control->freq_molec_anal = 0;
    control->bg_cut = 0.3;
    control->num_ignored = 0;
    memset( control->ignore, 0, sizeof(int)*MAX_ATOM_TYPES );

    control->dipole_anal = 0;
    control->freq_dipole_anal = 0;

    control->diffusion_coef = 0;
    control->freq_diffusion_coef = 0;
    control->restrict_type = 0;

    /* memory allocations */
    s = (char*) malloc(sizeof(char) * MAX_LINE);
    tmp = (char**) malloc(sizeof(char*)*MAX_TOKENS);
    for (i = 0; i < MAX_TOKENS; i++)
        tmp[i] = (char*) malloc(sizeof(char) * MAX_LINE);

    /* read control parameters file */
    while (fgets(s, MAX_LINE, fp))
    {
        c = Tokenize(s, &tmp);

        if ( strcmp(tmp[0], "simulation_name") == 0 )
        {
            strcpy( control->sim_name, tmp[1] );
        }
        //else if( strcmp(tmp[0], "restart") == 0 ) {
        //  ival = atoi(tmp[1]);
        //  control->restart = ival;
        //}
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
        else if ( strcmp(tmp[0], "random_vel") == 0 )
        {
            ival = atoi(tmp[1]);
            control->random_vel = ival;
        }
        else if ( strcmp(tmp[0], "reposition_atoms") == 0 )
        {
            ival = atoi(tmp[1]);
            control->reposition_atoms = ival;
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
        else if ( strcmp(tmp[0], "dt") == 0 )
        {
            val = atof(tmp[1]);
            control->dt = val * 1.e-3;  // convert dt from fs to ps!
        }
        else if ( strcmp(tmp[0], "periodic_boundaries") == 0 )
        {
            ival = atoi( tmp[1] );
            control->periodic_boundaries = ival;
        }
        else if ( strcmp(tmp[0], "periodic_images") == 0 )
        {
            ival = atoi(tmp[1]);
            control->periodic_images[0] = ival;
            ival = atoi(tmp[2]);
            control->periodic_images[1] = ival;
            ival = atoi(tmp[3]);
            control->periodic_images[2] = ival;
        }
        else if ( strcmp(tmp[0], "geo_format") == 0 )
        {
            ival = atoi( tmp[1] );
            control->geo_format = ival;
        }
        else if ( strcmp(tmp[0], "restrict_bonds") == 0 )
        {
            ival = atoi( tmp[1] );
            control->restrict_bonds = ival;
        }
        else if ( strcmp(tmp[0], "tabulate_long_range") == 0 )
        {
            ival = atoi( tmp[1] );
            control->tabulate = ival;
        }
        else if ( strcmp(tmp[0], "reneighbor") == 0 )
        {
            ival = atoi( tmp[1] );
            control->reneighbor = ival;
        }
        else if ( strcmp(tmp[0], "vlist_buffer") == 0 )
        {
            val = atof(tmp[1]);
            control->vlist_cut = val;
        }
        else if ( strcmp(tmp[0], "nbrhood_cutoff") == 0 )
        {
            val = atof(tmp[1]);
            control->nbr_cut = val;
        }
        else if ( strcmp(tmp[0], "thb_cutoff") == 0 )
        {
            val = atof(tmp[1]);
            control->thb_cut = val;
        }
        else if ( strcmp(tmp[0], "hbond_cutoff") == 0 )
        {
            val = atof( tmp[1] );
            control->hb_cut = val;
        }
        else if ( strcmp(tmp[0], "qeq_solver_type") == 0 )
        {
            ival = atoi( tmp[1] );
            control->qeq_solver_type = ival;
        }
        else if ( strcmp(tmp[0], "qeq_solver_q_err") == 0 )
        {
            val = atof( tmp[1] );
            control->qeq_solver_q_err = val;
        }
        else if ( strcmp(tmp[0], "qeq_domain_sparsity") == 0 )
        {
            val = atof( tmp[1] );
            control->qeq_domain_sparsity = val;
            control->qeq_domain_sparsify_enabled = TRUE;
        }
        else if ( strcmp(tmp[0], "pre_comp_type") == 0 )
        {
            ival = atoi( tmp[1] );
            control->pre_comp_type = ival;
        }
        else if ( strcmp(tmp[0], "pre_comp_refactor") == 0 )
        {
            ival = atoi( tmp[1] );
            control->pre_comp_refactor = ival;
        }
        else if ( strcmp(tmp[0], "pre_comp_droptol") == 0 )
        {
            val = atof( tmp[1] );
            control->pre_comp_droptol = val;
        }
        else if ( strcmp(tmp[0], "pre_comp_sweeps") == 0 )
        {
            ival = atoi( tmp[1] );
            control->pre_comp_sweeps = ival;
        }
        else if ( strcmp(tmp[0], "pre_app_type") == 0 )
        {
            ival = atoi( tmp[1] );
            control->pre_app_type = ival;
        }
        else if ( strcmp(tmp[0], "pre_app_jacobi_iters") == 0 )
        {
            ival = atoi( tmp[1] );
            control->pre_app_jacobi_iters = ival;
        }
        else if ( strcmp(tmp[0], "temp_init") == 0 )
        {
            val = atof(tmp[1]);
            control->T_init = val;

            if ( control->T_init < 0.001 )
                control->T_init = 0.001;
        }
        else if ( strcmp(tmp[0], "temp_final") == 0 )
        {
            val = atof(tmp[1]);
            control->T_final = val;

            if ( control->T_final < 0.1 )
                control->T_final = 0.1;
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
        else if ( strcmp(tmp[0], "p_mass") == 0 )
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
        else if ( strcmp(tmp[0], "pt_mass") == 0 )
        {
            val = atof(tmp[1]);
            control->Tau_PT = val * 1.e-3;  // convert pt_mass from fs to ps
        }
        else if ( strcmp(tmp[0], "compress") == 0 )
        {
            val = atof(tmp[1]);
            control->compressibility = val;
        }
        else if ( strcmp(tmp[0], "press_mode") == 0 )
        {
            val = atoi(tmp[1]);
            control->press_mode = val;
        }
        else if ( strcmp(tmp[0], "remove_CoM_vel") == 0 )
        {
            val = atoi(tmp[1]);
            control->remove_CoM_vel = val;
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
        else if ( strcmp(tmp[0], "write_freq") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->write_steps = ival;
        }
        else if ( strcmp(tmp[0], "traj_compress") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->traj_compress = ival;

            if ( out_control->traj_compress )
                out_control->write = (int (*)(FILE *, const char *, ...)) gzprintf;
            else out_control->write = fprintf;
        }
        else if ( strcmp(tmp[0], "traj_format") == 0 )
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
                             static_storage*, list **, void*)) Append_Custom_Frame;
            }
            else if ( out_control->traj_format == 1 )
            {
                out_control->write_header =
                    (int (*)( reax_system*, control_params*,
                              static_storage*, void* )) Write_xyz_Header;
                out_control->append_traj_frame =
                    (int (*)( reax_system*,  control_params*, simulation_data*,
                              static_storage*, list **, void* )) Append_xyz_Frame;
            }
        }
        else if ( strcmp(tmp[0], "traj_title") == 0 )
        {
            strcpy( out_control->traj_title, tmp[1] );
        }
        else if ( strcmp(tmp[0], "atom_info") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->atom_format += ival * 4;
        }
        else if ( strcmp(tmp[0], "atom_velocities") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->atom_format += ival * 2;
        }
        else if ( strcmp(tmp[0], "atom_forces") == 0 )
        {
            ival = atoi(tmp[1]);
            out_control->atom_format += ival * 1;
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
        else if ( strcmp(tmp[0], "test_forces") == 0 )
        {
            ival = atoi(tmp[1]);
        }
        else if ( strcmp(tmp[0], "molec_anal") == 0 )
        {
            ival = atoi(tmp[1]);
            control->molec_anal = ival;
        }
        else if ( strcmp(tmp[0], "freq_molec_anal") == 0 )
        {
            ival = atoi(tmp[1]);
            control->freq_molec_anal = ival;
        }
        else if ( strcmp(tmp[0], "bond_graph_cutoff") == 0 )
        {
            val = atof(tmp[1]);
            control->bg_cut = val;
        }
        else if ( strcmp(tmp[0], "ignore") == 0 )
        {
            control->num_ignored = atoi(tmp[1]);
            for ( i = 0; i < control->num_ignored; ++i )
                control->ignore[atoi(tmp[i + 2])] = 1;
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
            fprintf( stderr, "WARNING: unknown parameter %s\n", tmp[0] );
            exit( UNKNOWN_OPTION );
        }
    }

    if (ferror(fp))
    {
        fprintf(stderr, "Error reading control file. Terminating.\n");
        exit( INVALID_INPUT );
    }

    /* determine target T */
    if ( control->T_mode == 0 )
        control->T = control->T_final;
    else control->T = control->T_init;


    /* near neighbor and far neighbor cutoffs */
    control->bo_cut = 0.01 * system->reaxprm.gp.l[29];
    control->r_low  = system->reaxprm.gp.l[11];
    control->r_cut  = system->reaxprm.gp.l[12];
    control->r_sp_cut  = control->r_cut * control->qeq_domain_sparsity;
    control->vlist_cut += control->r_cut;

    system->g.cell_size = control->vlist_cut / 2.;
    for ( i = 0; i < 3; ++i )
    {
        system->g.spread[i] = 2;
    }

    /* free memory allocations at the top */
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        free( tmp[i] );
    }
    free( tmp );
    free( s );

#if defined(DEBUG_FOCUS)
    fprintf( stderr,
             "en=%d steps=%d dt=%.5f opt=%d T=%.5f P=%.5f %.5f %.5f\n",
             control->ensemble, control->nsteps, control->dt, control->tabulate,
             control->T, control->P[0], control->P[1], control->P[2] );

    fprintf(stderr, "control file read\n" );
#endif

    return SUCCESS;
}
