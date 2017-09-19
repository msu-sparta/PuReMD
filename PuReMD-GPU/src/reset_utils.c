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

#include "reset_utils.h"

#include "list.h"
#include "vector.h"


void Reset_Atoms( reax_system* system )
{
    int i;

    for ( i = 0; i < system->N; ++i )
    {
        memset( system->atoms[i].f, 0.0, sizeof(rvec) );
    }
}


void Reset_Pressures( simulation_data *data )
{
    rtensor_MakeZero( data->flex_bar.P );
    data->iso_bar.P = 0;
    rvec_MakeZero( data->int_press );
    rvec_MakeZero( data->ext_press );
    /* fprintf( stderr, "reset: ext_press (%12.6f %12.6f %12.6f)\n",
       data->ext_press[0], data->ext_press[1], data->ext_press[2] ); */
}


void Reset_Simulation_Data( simulation_data* data )
{
    data->E_BE = 0;
    data->E_Ov = 0;
    data->E_Un = 0;
    data->E_Lp = 0;
    data->E_Ang = 0;
    data->E_Pen = 0;
    data->E_Coa = 0;
    data->E_HB = 0;
    data->E_Tor = 0;
    data->E_Con = 0;
    data->E_vdW = 0;
    data->E_Ele = 0;
    data->E_Kin = 0;
}


#ifdef TEST_FORCES
void Reset_Test_Forces( reax_system *system, static_storage *workspace )
{
    memset( workspace->f_ele, 0, system->N * sizeof(rvec) );
    memset( workspace->f_vdw, 0, system->N * sizeof(rvec) );
    memset( workspace->f_bo, 0, system->N * sizeof(rvec) );
    memset( workspace->f_be, 0, system->N * sizeof(rvec) );
    memset( workspace->f_lp, 0, system->N * sizeof(rvec) );
    memset( workspace->f_ov, 0, system->N * sizeof(rvec) );
    memset( workspace->f_un, 0, system->N * sizeof(rvec) );
    memset( workspace->f_ang, 0, system->N * sizeof(rvec) );
    memset( workspace->f_coa, 0, system->N * sizeof(rvec) );
    memset( workspace->f_pen, 0, system->N * sizeof(rvec) );
    memset( workspace->f_hb, 0, system->N * sizeof(rvec) );
    memset( workspace->f_tor, 0, system->N * sizeof(rvec) );
    memset( workspace->f_con, 0, system->N * sizeof(rvec) );
}
#endif


void Reset_Workspace( reax_system *system, static_storage *workspace )
{
    memset( workspace->total_bond_order, 0, system->N * sizeof( real ) );
    memset( workspace->dDeltap_self, 0, system->N * sizeof( rvec ) );

    memset( workspace->CdDelta, 0, system->N * sizeof( real ) );
    //memset( workspace->virial_forces, 0, system->N * sizeof( rvec ) );

#ifdef TEST_FORCES
    memset( workspace->dDelta, 0, sizeof(rvec) * system->N );
    Reset_Test_Forces( system, workspace );
#endif
}


void Reset_Neighbor_Lists( reax_system *system, control_params *control,
        static_storage *workspace, list **lists )
{
    int i, tmp;
    list *bonds = (*lists) + BONDS;
    list *hbonds = (*lists) + HBONDS;

    for ( i = 0; i < system->N; ++i )
    {
        tmp = Start_Index( i, bonds );
        Set_End_Index( i, tmp, bonds );
    }

    //TODO: added for GPU, verify if correct
    memset( bonds->select.bond_list, 0, BOND_DATA_SIZE * bonds->num_intrs );

    if ( control->hb_cut > 0 )
    {
        for ( i = 0; i < system->N; ++i )
        {
            if ( system->reaxprm.sbp[system->atoms[i].type].p_hbond == 1)
            {
                tmp = Start_Index( workspace->hbond_index[i], hbonds );
                Set_End_Index( workspace->hbond_index[i], tmp, hbonds );
                /* fprintf( stderr, "i:%d, hbond: %d-%d\n",
                   i, Start_Index( workspace->hbond_index[i], hbonds ),
                   End_Index( workspace->hbond_index[i], hbonds ) );*/
            }
        }
    }
}


void Reset( reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace, list **lists  )
{
    Reset_Atoms( system );

    Reset_Simulation_Data( data );

    if ( control->ensemble == NPT || control->ensemble == sNPT ||
            control->ensemble == iNPT )
    {
        Reset_Pressures( data );
    }

    Reset_Workspace( system, workspace );

    Reset_Neighbor_Lists( system, control, workspace, lists );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "reset - ");
#endif
}


void Reset_Grid( grid *g )
{
    memset( g->top, 0, INT_SIZE * g->ncell[0]*g->ncell[1]*g->ncell[2] );
}



void Reset_Marks( grid *g, ivec *grid_stack, int grid_top )
{
    int i;

    for ( i = 0; i < grid_top; ++i )
    {
        g->mark[grid_stack[i][0] * g->ncell[1]*g->ncell[2]
            + grid_stack[i][1] * g->ncell[2] + grid_stack[i][2]] = 0;
    }
}
