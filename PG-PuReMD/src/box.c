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

#include "box.h"

#include "comm_tools.h"
#include "io_tools.h"
#include "system_props.h"
#include "vector.h"


void Make_Consistent( simulation_box * const box )
{
    real one_vol;

    box->V =
        box->box[0][0] * (box->box[1][1] * box->box[2][2] -
                          box->box[2][1] * box->box[2][1]) +
        box->box[0][1] * (box->box[2][0] * box->box[1][2] -
                          box->box[1][0] * box->box[2][2]) +
        box->box[0][2] * (box->box[1][0] * box->box[2][1] -
                          box->box[2][0] * box->box[1][1]);

    one_vol = 1.0 / box->V;
    box->box_inv[0][0] = (box->box[1][1] * box->box[2][2] -
                          box->box[1][2] * box->box[2][1]) * one_vol;
    box->box_inv[0][1] = (box->box[0][2] * box->box[2][1] -
                          box->box[0][1] * box->box[2][2]) * one_vol;
    box->box_inv[0][2] = (box->box[0][1] * box->box[1][2] -
                          box->box[0][2] * box->box[1][1]) * one_vol;

    box->box_inv[1][0] = (box->box[1][2] * box->box[2][0] -
                          box->box[1][0] * box->box[2][2]) * one_vol;
    box->box_inv[1][1] = (box->box[0][0] * box->box[2][2] -
                          box->box[0][2] * box->box[2][0]) * one_vol;
    box->box_inv[1][2] = (box->box[0][2] * box->box[1][0] -
                          box->box[0][0] * box->box[1][2]) * one_vol;

    box->box_inv[2][0] = (box->box[1][0] * box->box[2][1] -
                          box->box[1][1] * box->box[2][0]) * one_vol;
    box->box_inv[2][1] = (box->box[0][1] * box->box[2][0] -
                          box->box[0][0] * box->box[2][1]) * one_vol;
    box->box_inv[2][2] = (box->box[0][0] * box->box[1][1] -
                          box->box[0][1] * box->box[1][0]) * one_vol;


    box->box_norms[0] = SQRT( SQR(box->box[0][0]) + SQR(box->box[0][1]) +
                              SQR(box->box[0][2]) );
    box->box_norms[1] = SQRT( SQR(box->box[1][0]) + SQR(box->box[1][1]) +
                              SQR(box->box[1][2]) );
    box->box_norms[2] = SQRT( SQR(box->box[2][0]) + SQR(box->box[2][1]) +
                              SQR(box->box[2][2]) );

    box->max[0] = box->min[0] + box->box_norms[0];
    box->max[1] = box->min[1] + box->box_norms[1];
    box->max[2] = box->min[2] + box->box_norms[2];


    box->trans[0][0] = box->box[0][0] / box->box_norms[0];
    box->trans[0][1] = box->box[1][0] / box->box_norms[0];
    box->trans[0][2] = box->box[2][0] / box->box_norms[0];

    box->trans[1][0] = box->box[0][1] / box->box_norms[1];
    box->trans[1][1] = box->box[1][1] / box->box_norms[1];
    box->trans[1][2] = box->box[2][1] / box->box_norms[1];

    box->trans[2][0] = box->box[0][2] / box->box_norms[2];
    box->trans[2][1] = box->box[1][2] / box->box_norms[2];
    box->trans[2][2] = box->box[2][2] / box->box_norms[2];


    one_vol = box->box_norms[0] * box->box_norms[1] * box->box_norms[2] * one_vol;
    box->trans_inv[0][0] = (box->trans[1][1] * box->trans[2][2] -
                            box->trans[1][2] * box->trans[2][1]) * one_vol;
    box->trans_inv[0][1] = (box->trans[0][2] * box->trans[2][1] -
                            box->trans[0][1] * box->trans[2][2]) * one_vol;
    box->trans_inv[0][2] = (box->trans[0][1] * box->trans[1][2] -
                            box->trans[0][2] * box->trans[1][1]) * one_vol;

    box->trans_inv[1][0] = (box->trans[1][2] * box->trans[2][0] -
                            box->trans[1][0] * box->trans[2][2]) * one_vol;
    box->trans_inv[1][1] = (box->trans[0][0] * box->trans[2][2] -
                            box->trans[0][2] * box->trans[2][0]) * one_vol;
    box->trans_inv[1][2] = (box->trans[0][2] * box->trans[1][0] -
                            box->trans[0][0] * box->trans[1][2]) * one_vol;

    box->trans_inv[2][0] = (box->trans[1][0] * box->trans[2][1] -
                            box->trans[1][1] * box->trans[2][0]) * one_vol;
    box->trans_inv[2][1] = (box->trans[0][1] * box->trans[2][0] -
                            box->trans[0][0] * box->trans[2][1]) * one_vol;
    box->trans_inv[2][2] = (box->trans[0][0] * box->trans[1][1] -
                            box->trans[0][1] * box->trans[1][0]) * one_vol;


    box->g[0][0] = box->box[0][0] * box->box[0][0] +
                   box->box[0][1] * box->box[0][1] +
                   box->box[0][2] * box->box[0][2];
    box->g[1][0] =
        box->g[0][1] = box->box[0][0] * box->box[1][0] +
                       box->box[0][1] * box->box[1][1] +
                       box->box[0][2] * box->box[1][2];
    box->g[2][0] =
        box->g[0][2] = box->box[0][0] * box->box[2][0] +
                       box->box[0][1] * box->box[2][1] +
                       box->box[0][2] * box->box[2][2];

    box->g[1][1] = box->box[1][0] * box->box[1][0] +
                   box->box[1][1] * box->box[1][1] +
                   box->box[1][2] * box->box[1][2];
    box->g[1][2] =
        box->g[2][1] = box->box[1][0] * box->box[2][0] +
                       box->box[1][1] * box->box[2][1] +
                       box->box[1][2] * box->box[2][2];

    box->g[2][2] = box->box[2][0] * box->box[2][0] +
                   box->box[2][1] * box->box[2][1] +
                   box->box[2][2] * box->box[2][2];
}


/* setup the entire simulation box */
void Setup_Big_Box( real a, real b, real c, real alpha, real beta, real gamma,
        simulation_box * const box )
{
    double c_alpha, c_beta, c_gamma, s_gamma, zi;

    if ( IS_NAN_REAL(a) || IS_NAN_REAL(b) || IS_NAN_REAL(c)
            || IS_NAN_REAL(alpha) || IS_NAN_REAL(beta) || IS_NAN_REAL(gamma) )
    {
        fprintf( stderr, "[ERROR] Invalid simulation box boundaries for big box (NaN). Terminating...\n" );
        exit( INVALID_INPUT );
    }

    c_alpha = COS(DEG2RAD(alpha));
    c_beta  = COS(DEG2RAD(beta));
    c_gamma = COS(DEG2RAD(gamma));
    s_gamma = SIN(DEG2RAD(gamma));
    zi = (c_alpha - c_beta * c_gamma) / s_gamma;

    rvec_MakeZero( box->min );

    box->box[0][0] = a;
    box->box[0][1] = 0.0;
    box->box[0][2] = 0.0;
    box->box[1][0] = b * c_gamma;
    box->box[1][1] = b * s_gamma;
    box->box[1][2] = 0.0;
    box->box[2][0] = c * c_beta;
    box->box[2][1] = c * zi;
    box->box[2][2] = c * SQRT(1.0 - SQR(c_beta) - SQR(zi));

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "box is %8.2f x %8.2f x %8.2f\n",
            box->box[0][0], box->box[1][1], box->box[2][2] );
#endif

    Make_Consistent( box );
}


void Init_Box( rtensor box_tensor, simulation_box * const box )
{
    rvec_MakeZero( box->min );
    rtensor_Copy( box->box, box_tensor );
    Make_Consistent( box );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "box is %8.2f x %8.2f x %8.2f\n",
             box->box[0][0], box->box[1][1], box->box[2][2] );
#endif
}


/* setup my simulation box -- only the region that I own */
void Setup_My_Box( reax_system * const system, control_params * const control )
{
    int d;
    simulation_box * const big_box = &system->big_box;
    simulation_box * const my_box = &system->my_box;

    rtensor_MakeZero( my_box->box );

    for ( d = 0; d < 3; ++d )
    {
        my_box->min[d] = big_box->box_norms[d] * system->my_coords[d] /
            control->procs_by_dim[d];
        my_box->box[d][d] = big_box->box_norms[d] / control->procs_by_dim[d];
        //my_box->max[d] = big_box->box_norms[d] * (system->my_coords[d] + 1) /
        //control->procs_by_dim[d];
        //my_box->box_norms[d] = my_box->max[d] - my_box->min[d];
    }

    Make_Consistent( my_box );
}


/* setup my extended box -- my box together with the ghost regions */
void Setup_My_Ext_Box( reax_system * const system, control_params * const control )
{
    int d;
    ivec native_gcells, ghost_gcells;
    rvec gcell_len;
    boundary_cutoff * const bc = &system->bndry_cuts;
    simulation_box * const my_box = &system->my_box;
    simulation_box * const my_ext_box = &system->my_ext_box;
#if defined(DEBUG_FOCUS)
    simulation_box * const big_box = &system->big_box;
#endif

    rtensor_MakeZero( my_ext_box->box );

    for ( d = 0; d < 3; ++d )
    {
        /* estimate the number of native cells */
        native_gcells[d] = (int)(my_box->box_norms[d] / (control->vlist_cut / 2));
        if ( native_gcells[d] == 0 )
        {
            native_gcells[d] = 1;
        }

        gcell_len[d] = my_box->box_norms[d] / native_gcells[d];
        ghost_gcells[d] = (int) CEIL(bc->ghost_cutoff / gcell_len[d]);

        /* extend my box with the ghost regions */
        my_ext_box->min[d] = my_box->min[d] - ghost_gcells[d] * gcell_len[d];
        my_ext_box->box[d][d] = my_box->box_norms[d] + 2 * ghost_gcells[d] * gcell_len[d];
        //my_ext_box->max[d] = my_box->max[d] + ghost_gcells[d] * gcell_len[d];
        //my_ext_box->box_norms[d] = my_ext_box->max[d] - my_ext_box->min[d];
    }

    Make_Consistent( my_ext_box );
}


/******************** initialize parallel environment ***********************/

void Setup_Boundary_Cutoffs( reax_system * const system, control_params * const control )
{
    boundary_cutoff * const bc = &system->bndry_cuts;

    bc->ghost_nonb = control->nonb_cut;
    bc->ghost_hbond = control->hbond_cut;
    bc->ghost_bond = 2 * control->bond_cut;
    bc->ghost_cutoff = MAX( control->vlist_cut, bc->ghost_bond );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "ghost_nonb: %8.3f\n", bc->ghost_nonb );
    fprintf( stderr, "ghost_hbond: %8.3f\n", bc->ghost_hbond );
    fprintf( stderr, "ghost_bond: %8.3f\n", bc->ghost_bond );
    fprintf( stderr, "ghost_cutoff: %8.3f\n", bc->ghost_cutoff );
#endif
}


void Setup_Environment( reax_system * const system, control_params * const control,
        mpi_datatypes * const mpi_data )
{
    ivec periodic = {1, 1, 1};
#if defined(DEBUG_FOCUS)
    char temp[100] = "";
#endif

    /* initialize communicator - 3D mesh with wrap-arounds = 3D torus */
    MPI_Cart_create( MPI_COMM_WORLD, 3, control->procs_by_dim, periodic, 1,
            &mpi_data->comm_mesh3D );
    MPI_Comm_rank( mpi_data->comm_mesh3D, &system->my_rank );
    MPI_Cart_coords( mpi_data->comm_mesh3D, system->my_rank, 3,
            system->my_coords );

    Setup_Boundary_Cutoffs( system, control );
    Setup_My_Box( system, control );
    Setup_My_Ext_Box( system, control );
    Setup_Comm( system, control, mpi_data );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d coord: %d %d %d\n",
             system->my_rank, system->my_coords[0],
             system->my_coords[1], system->my_coords[2] );
    sprintf( temp, "p%d big_box", system->my_rank );
    Print_Box( &system->big_box, temp, stderr );
    sprintf( temp, "p%d my_box", system->my_rank );
    Print_Box( &system->my_box, temp, stderr );
    sprintf( temp, "p%d ext_box", system->my_rank );
    Print_Box( &system->my_ext_box, temp, stderr );
    MPI_Barrier( MPI_COMM_WORLD );

    fprintf( stderr, "p%d: parallel environment initialized\n",
            system->my_rank );
#endif
}


void Scale_Box( reax_system * const system, control_params * const control,
        simulation_data * const data, mpi_datatypes * const mpi_data )
{
    int i, d;
    real dt, lambda;
    rvec mu = {0.0, 0.0, 0.0};
    reax_atom *atom;

    dt = control->dt;

    /* pressure scaler */
    if ( control->ensemble == iNPT )
    {
        mu[0] = POW( 1.0 + (dt / control->Tau_P[0]) * (data->iso_bar.P - control->P[0]),
                     1. / 3 );

        if ( mu[0] < MIN_dV )
        {
            mu[0] = MIN_dV;
        }
        else if ( mu[0] > MAX_dV )
        {
            mu[0] = MAX_dV;
        }

        mu[1] = mu[0];
        mu[2] = mu[1];
    }
    else if ( control->ensemble == sNPT )
    {
        for ( d = 0; d < 3; ++d )
        {
            mu[d] = POW(1.0 + (dt / control->Tau_P[d]) * (data->tot_press[d] - control->P[d]),
                        1. / 3 );

            if ( mu[d] < MIN_dV )
            {
                mu[d] = MIN_dV;
            }
            else if ( mu[d] > MAX_dV )
            {
                mu[d] = MAX_dV;
            }
        }
    }

    /* temperature scaler */
    lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
    if ( lambda < MIN_dT )
    {
        lambda = MIN_dT;
    }
    else if ( lambda > MAX_dT )
    {
        lambda = MAX_dT;
    }
    lambda = SQRT( lambda );

    /* Scale velocities and positions at t+dt */
    for ( i = 0; i < system->n; ++i )
    {
        atom = &system->my_atoms[i];
        rvec_Scale( atom->v, lambda, atom->v );
        atom->x[0] = mu[0] * atom->x[0];
        atom->x[1] = mu[1] * atom->x[1];
        atom->x[2] = mu[2] * atom->x[2];
    }
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

    /* update box & grid */
    system->big_box.box[0][0] *= mu[0];
    system->big_box.box[1][1] *= mu[1];
    system->big_box.box[2][2] *= mu[2];

    Make_Consistent( &system->big_box );
    Setup_My_Box( system, control );
    Setup_My_Ext_Box( system, control );
    Update_Comm( system );
}


real Metric_Product( rvec x1, rvec x2, simulation_box * const box )
{
    int i, j;
    real dist = 0.0, tmp;

    for ( i = 0; i < 3; i++ )
    {
        tmp = 0.0;
        for ( j = 0; j < 3; j++ )
        {
            tmp += box->g[i][j] * x2[j];
        }
        dist += x1[i] * tmp;
    }

    return dist;
}
