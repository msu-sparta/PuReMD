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

#include "reset_tools.h"

#include "list.h"
#include "vector.h"


void Reset_Pressures( control_params const * const control, simulation_data * const data )
{
#if defined(_OPENMP)
    int32_t i;
#endif

    rtensor_MakeZero( data->flex_bar.P );
    data->iso_bar.P = 0.0;
    rtensor_MakeZero( data->press );
#if defined(_OPENMP)
    if ( control->ensemble == sNPT || control->ensemble == iNPT
            || control->ensemble == aNPT || control->compute_pressure == TRUE ) {
        for ( i = 0; i < control->num_threads; ++i ) {
            rtensor_MakeZero( data->press_local[i] );
        }
    }
#endif
}


#if defined(TEST_FORCES)
static void Reset_Test_Forces( reax_system const * const system, static_storage * const workspace )
{
    uint32_t i;

#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->dDelta[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_ele[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_vdw[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_be[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_lp[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_ov[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_un[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_ang[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_coa[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_pen[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_hb[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_tor[i] );
    }
#if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,32) private(i) shared(system, workspace)
#endif
    for ( i = 0; i < system->N; i++ ) {
        rvec_MakeZero( workspace->f_con[i] );
    }
}
#endif


void Reset_Atomic_Forces( reax_system * const system, static_storage * const workspace )
{
#if defined(_OPENMP)
    #pragma omp parallel
#endif
    {
        uint32_t i;
#if defined(_OPENMP)
        int32_t tid;
#endif

#if defined(_OPENMP)
        #pragma omp for schedule(dynamic,32)
#endif
        for ( i = 0; i < system->N; ++i ) {
            rvec_MakeZero( system->atoms[i].f );
        }

#if defined(_OPENMP)
        tid = omp_get_thread_num( );

        for ( i = 0; i < system->N; ++i ) {
            rvec_MakeZero( workspace->f_local[(uint32_t) tid * system->N + i] );
        }
#endif
    }
}


void Reset_Energies( simulation_data * const data )
{
    data->E_Tot = 0.0;
    data->E_Kin = 0.0;
    data->E_Pot = 0.0;
    data->E_BE = 0.0;
    data->E_Ov = 0.0;
    data->E_Un = 0.0;
    data->E_Lp = 0.0;
    data->E_Ang = 0.0;
    data->E_Pen = 0.0;
    data->E_Coa = 0.0;
    data->E_HB = 0.0;
    data->E_Tor = 0.0;
    data->E_Con = 0.0;
    data->E_vdW = 0.0;
    data->E_Ele = 0.0;
    data->E_Pol = 0.0;
}


void Reset_Workspace( reax_system const * const system, static_storage * const workspace )
{
#if defined(_OPENMP)
    #pragma omp parallel
#endif
    {
        uint32_t i;

#if defined(_OPENMP)
        #pragma omp for schedule(dynamic,32)
#endif
        for ( i = 0; i < system->N; i++ ) {
            workspace->total_bond_order[i] = 0.0;
        }

#if defined(_OPENMP)
        #pragma omp for schedule(dynamic,32)
#endif
        for ( i = 0; i < system->N; i++ ) {
            rvec_MakeZero( workspace->dDeltap_self[i] );
        }

#if defined(_OPENMP)
        #pragma omp for schedule(dynamic,32)
#endif
        for ( i = 0; i < system->N; i++ ) {
            workspace->CdDelta[i] = 0.0;
        }
    }

#if defined(TEST_FORCES)
    Reset_Test_Forces( system, workspace );
#endif
}


void Reset( reax_system * const system, control_params const * const control,
        simulation_data * const data, static_storage * const workspace )
{
    Reset_Atomic_Forces( system, workspace );

    Reset_Energies( data );

    if ( control->ensemble == sNPT || control->ensemble == iNPT
            || control->ensemble == aNPT || control->compute_pressure == TRUE ) {
        Reset_Pressures( control, data );
    }
}


void Reset_Grid( grid *g )
{
    uint32_t i, j, k;

    for ( i = 0; i < g->ncell[0]; i++ ) {
        for ( j = 0; j < g->ncell[1]; j++ ) {
            for ( k = 0; k < g->ncell[2]; k++ ) {
                g->top[i][j][k] = 0;
            }
        }
    }
}


void Reset_Marks( grid *g, ivec *grid_stack, uint32_t grid_top )
{
    uint32_t i;

    for ( i = 0; i < grid_top; ++i ) {
        g->mark[grid_stack[i][0]][grid_stack[i][1]][grid_stack[i][2]] = FALSE;
    }
}
