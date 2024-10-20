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
  #include "reset_tools.h"

  #include "index_utils.h"
  #include "list.h"
  #include "tool_box.h"
  #include "vector.h"
#elif defined(LAMMPS_REAX)
  #include "reax_reset_tools.h"

  #include "reax_index_utils.h"
  #include "reax_list.h"
  #include "reax_tool_box.h"
  #include "reax_vector.h"
#endif


/* Calculate atom indices of local and ghost atoms
 * for hydrogen bonding interactions */
static void Reset_Atoms_HBond_Indices( reax_system * const system, control_params * const control )
{
    int i;
    reax_atom *atom;

    system->num_H_atoms = 0;

    for ( i = 0; i < system->N; ++i )
    {
        atom = &system->my_atoms[i];

        if ( system->reax_param.sbp[atom->type].p_hbond == H_ATOM )
        {
            atom->Hindex = system->num_H_atoms;
            ++(system->num_H_atoms);
        }
        else
        {
            atom->Hindex = -1;
        }
    }
}


static void Reset_Energies( real * const en )
{
    en[E_BOND] = 0.0;
    en[E_OV] = 0.0;
    en[E_UN] = 0.0;
    en[E_LP] = 0.0;
    en[E_ANG] = 0.0;
    en[E_PEN] = 0.0;
    en[E_COA] = 0.0;
    en[E_HB] = 0.0;
    en[E_TOR] = 0.0;
    en[E_CON] = 0.0;
    en[E_VDW] = 0.0;
    en[E_ELE] = 0.0;
    en[E_POL] = 0.0;
    en[E_POT] = 0.0;
    en[E_KIN] = 0.0;
    en[E_TOT] = 0.0;
}


static void Reset_Temperatures( simulation_data * const data )
{
    data->therm.T = 0.0;
    data->therm.xi = 0.0;
    data->therm.v_xi = 0.0;
    data->therm.v_xi_old = 0.0;
    data->therm.G_xi = 0.0;
}


void Reset_Pressures( simulation_data * const data )
{
    data->flex_bar.P_scalar = 0.0;
    rtensor_MakeZero( data->flex_bar.P );

    data->iso_bar.P = 0.0;
    rvec_MakeZero( data->int_press );
    rvec_MakeZero( data->my_ext_press );
    rvec_MakeZero( data->ext_press );
}


void Reset_Simulation_Data( simulation_data * const data )
{
    Reset_Energies( data->my_en );
    Reset_Energies( data->sys_en );
    Reset_Temperatures( data );
    Reset_Pressures( data );
}


void Reset_Timing( reax_timing * const timing )
{
    timing->total = Get_Time( );
    timing->comm = ZERO;
    timing->nbrs = ZERO;
    timing->init_forces = ZERO;
    timing->init_dist = ZERO;
    timing->init_cm = ZERO;
    timing->init_bond = ZERO;
    timing->init_hbond = ZERO;
    timing->bonded = ZERO;
    timing->bond_order = ZERO;
    timing->bonds = ZERO;
    timing->lpovun = ZERO;
    timing->valence = ZERO;
    timing->torsion = ZERO;
    timing->hbonds = ZERO;
    timing->nonb = ZERO;
    timing->cm = ZERO;
    timing->cm_sort = ZERO;
    timing->cm_solver_pre_comp = ZERO;
    timing->cm_solver_pre_app = ZERO;
    timing->cm_solver_comm = ZERO;
    timing->cm_solver_allreduce = ZERO;
    timing->cm_solver_iters = 0;
    timing->cm_solver_spmv = ZERO;
    timing->cm_solver_vector_ops = ZERO;
    timing->cm_solver_orthog = ZERO;
    timing->cm_solver_tri_solve = ZERO;
    timing->num_retries = 0;
}


#ifdef TEST_FORCES
void Reset_Test_Forces( reax_system * const system, storage * const workspace )
{
    memset( workspace->f_ele, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_vdw, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_bo, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_be, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_lp, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_ov, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_un, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_ang, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_coa, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_pen, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_hb, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_tor, 0, sizeof(rvec) * system->total_cap );
    memset( workspace->f_con, 0, sizeof(rvec) * system->total_cap );
}
#endif


void Reset_Workspace( reax_system * const system, storage * const workspace )
{
    int i;

    for ( i = 0; i < system->total_cap; ++i )
    {
        workspace->CdDelta[i] = 0.0;
    }
    for ( i = 0; i < system->total_cap; ++i )
    {
        rvec_MakeZero( workspace->f[i] );
    }

#ifdef TEST_FORCES
    for ( i = 0; i < system->total_cap; ++i )
    {
        rvec_MakeZero( workspace->dDelta[i] );
    }
    Reset_Test_Forces( system, workspace );
#endif
}


void Reset_Grid( grid * const g )
{
    int i, j, k;

    for ( i = 0; i < g->ncells[0]; i++ )
    {
        for ( j = 0; j < g->ncells[1]; j++ )
        {
            for ( k = 0; k < g->ncells[2]; k++ )
            {
                g->cells[ index_grid_3d(i, j, k, g) ].top = 0;
                g->str[ index_grid_3d(i, j, k, g) ] = 0;
                g->end[ index_grid_3d(i, j, k, g) ] = 0;
            }
        }
    }
}


void Reset_Out_Buffers( mpi_out_data * const out_buf, int n )
{
    int i;

    for ( i = 0; i < n; ++i )
    {
        out_buf[i].cnt = 0;
    }
}


void Reset( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace,
        reax_list ** const lists )
{
    if ( system->total_H_atoms > 0 && control->hbond_cut > 0.0 )
    {
        Reset_Atoms_HBond_Indices( system, control );
    }

    Reset_Simulation_Data( data );

    if ( control->virial )
    {
        Reset_Pressures( data );
    }

    Reset_Workspace( system, workspace );
}
