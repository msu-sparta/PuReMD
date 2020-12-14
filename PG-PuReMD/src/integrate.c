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

#include "integrate.h"

#include "allocate.h"
#include "box.h"
#include "comm_tools.h"
#include "forces.h"
#include "grid.h"
#include "io_tools.h"
#include "neighbors.h"
#include "reset_tools.h"
#include "system_props.h"
#include "tool_box.h"
#include "vector.h"


/* Velocity Verlet integrator for microcanonical ensemble. */
int Velocity_Verlet_NVE( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
{
    int i, steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE;
    real inv_m, scalar1, scalar2;
    rvec dx;
    reax_atom *atom;

    ret = SUCCESS;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;
    scalar1 = -0.5 * control->dt * F_CONV;
    scalar2 = -0.5 * SQR( control->dt ) * F_CONV;

    if ( verlet_part1_done == FALSE )
    {
        /* velocity verlet, 1st part */
        for ( i = 0; i < system->n; i++ )
        {
            atom = &system->my_atoms[i];
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;

            /* Compute x(t + dt) */
            rvec_ScaledSum( dx, control->dt, atom->v, scalar2 * inv_m, atom->f );
            rvec_Add( system->my_atoms[i].x, dx );

            /* Compute v(t + dt/2) */
            rvec_ScaledAdd( atom->v, scalar1 * inv_m, atom->f );
        }

        Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

        verlet_part1_done = TRUE;
    }

    Reallocate_Part2( system, control, data, workspace, lists, mpi_data );
        
    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE && gen_nbr_list == FALSE )
    {
        ret = Generate_Neighbor_Lists( system, data, workspace, lists );

        if ( ret == SUCCESS )
        {
            gen_nbr_list = TRUE;
        }
        else
        {
            Estimate_Num_Neighbors( system, data, lists[FAR_NBRS]->format );
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        for ( i = 0; i < system->n; i++ )
        {
            atom = &system->my_atoms[i];
            inv_m = 1.0 / system->reax_param.sbp[ atom->type ].mass;

            /* Compute v(t + dt) */
            rvec_ScaledAdd( atom->v, scalar1 * inv_m, atom->f );
        }

        verlet_part1_done = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


/* Velocity Verlet integrator for constant volume and temperature
 *  with Berendsen thermostat.
 *
 * NOTE: All box dimensions are scaled by the same amount, and
 * there is no change in the angles between axes. */
int Velocity_Verlet_Berendsen_NVT( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
{
    int i, steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE;
    real inv_m, scalar1, scalar2, lambda;
    rvec dx;
    reax_atom *atom;

    ret = SUCCESS;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;
    scalar1 = -0.5 * control->dt * F_CONV;
    scalar2 = -0.5 * SQR( control->dt ) * F_CONV;

    if ( verlet_part1_done == FALSE )
    {
        /* velocity verlet, 1st part */
        for ( i = 0; i < system->n; i++ )
        {
            atom = &system->my_atoms[i];
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;

            /* Compute x(t + dt) */
            rvec_ScaledSum( dx, control->dt, atom->v, scalar2 * inv_m, atom->f );
            rvec_Add( atom->x, dx );

            /* Compute v(t + dt/2) */
            rvec_ScaledAdd( atom->v, scalar1 * inv_m, atom->f );
        }

        Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        if ( renbr == TRUE )
        {
            Update_Grid( system, control, MPI_COMM_WORLD );
        }

        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

        verlet_part1_done = TRUE;
    }

    Reallocate_Part2( system, control, data, workspace, lists, mpi_data );
        
    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE && gen_nbr_list == FALSE )
    {
        ret = Generate_Neighbor_Lists( system, data, workspace, lists );

        if ( ret == SUCCESS )
        {
            gen_nbr_list = TRUE;
        }
        else
        {
            Estimate_Num_Neighbors( system, data, lists[FAR_NBRS]->format );
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        /* velocity verlet, 2nd part */
        for ( i = 0; i < system->n; i++ )
        {
            atom = &system->my_atoms[i];
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;

            /* Compute v(t + dt) */
            rvec_ScaledAdd( atom->v, scalar1 * inv_m, atom->f );
        }

        Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

        /* temperature scaler */
        lambda = 1.0 + ((control->dt * 1.0e-12) / control->Tau_T)
            * (control->T / data->therm.T - 1.0);

        if ( lambda < MIN_dT )
        {
            lambda = MIN_dT;
        }

        lambda = SQRT( lambda );

        if ( lambda > MAX_dT )
        {
            lambda = MAX_dT;
        }

        /* Scale velocities and positions at t+dt */
        for ( i = 0; i < system->n; ++i )
        {
            atom = &system->my_atoms[i];

            rvec_Scale( atom->v, lambda, atom->v );
        }

        /* update kinetic energy and temperature based on new velocities */
        Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );

        verlet_part1_done = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


/* Velocity Verlet integrator for constant volume and constant temperature
 *  with Nose-Hoover thermostat.
 *
 * Reference: Understanding Molecular Simulation, Frenkel and Smit
 *  Academic Press Inc. San Diego, 1996 p. 388-391 */
int Velocity_Verlet_Nose_Hoover_NVT_Klein( reax_system * const system,
        control_params * const control, simulation_data * const data, storage * const workspace,
        reax_list ** const lists, output_controls * const out_control, mpi_datatypes * const mpi_data )
{
    int i, itr, steps, renbr, ret, ret_mpi;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE;
    real inv_m, coef_v;
    real dt, dt_sqr;
    real my_ekin, new_ekin;
    real G_xi_new, v_xi_new, v_xi_old;
    rvec dx;
    thermostat *therm;
    reax_atom *atom;

    ret = SUCCESS;
    dt = control->dt;
    therm = &data->therm;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE )
    {
        dt_sqr = SQR(dt);

        /* velocity verlet, 1st part */
        for ( i = 0; i < system->n; i++ )
        {
            atom = &system->my_atoms[i];
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
            rvec_ScaledSum( dx, dt, atom->v, -0.5 * dt_sqr * F_CONV * inv_m, atom->f );
            rvec_Add( system->my_atoms[i].x, dx );
            rvec_Copy( atom->f_old, atom->f );
        }
    
        /* Compute xi(t + dt) */
        therm->xi += therm->v_xi * dt + 0.5 * dt_sqr * therm->G_xi;

        Reallocate_Part1( system, control, data, workspace, lists, mpi_data );

        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

        verlet_part1_done = TRUE;
    }

    Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE && gen_nbr_list == FALSE )
    {
        ret = Generate_Neighbor_Lists( system, data, workspace, lists );

        if ( ret == SUCCESS )
        {
            gen_nbr_list = TRUE;
        }
        else
        {
            Estimate_Num_Neighbors( system, data, lists[FAR_NBRS]->format );
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        /* Compute iteration constants for each atom's velocity */
        for ( i = 0; i < system->n; ++i )
        {
            atom = &system->my_atoms[i];
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
            rvec_Scale( workspace->v_const[i], 1.0 - 0.5 * dt * therm->v_xi, atom->v );
            rvec_ScaledAdd( workspace->v_const[i], -0.5 * dt * inv_m * F_CONV, atom->f_old );
            rvec_ScaledAdd( workspace->v_const[i], -0.5 * dt * inv_m * F_CONV, atom->f );
        }
    
        v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
        my_ekin = G_xi_new = v_xi_old = 0;
        itr = 0;
        do
        {
            itr++;
    
            /* new values become old in this iteration */
            v_xi_old = v_xi_new;
    
            my_ekin = 0;
            for ( i = 0; i < system->n; ++i )
            {
                atom = &system->my_atoms[i];
                coef_v = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
                rvec_Scale( atom->v, coef_v, workspace->v_const[i] );
                my_ekin += 0.5 * system->reax_param.sbp[atom->type].mass
                        * rvec_Dot( atom->v, atom->v );
            }
    
            ret_mpi = MPI_Allreduce( &my_ekin, &new_ekin, 1, MPI_DOUBLE,
                    MPI_SUM, mpi_data->comm_mesh3D  );
            Check_MPI_Error( ret_mpi, __FILE__, __LINE__ );
    
            G_xi_new = control->Tau_T * ( 2.0 * new_ekin - data->N_f * K_B * control->T );
            v_xi_new = therm->v_xi + 0.5 * dt * ( therm->G_xi + G_xi_new );
        } while ( FABS(v_xi_new - v_xi_old) > 1e-5 );

        therm->v_xi_old = therm->v_xi;
        therm->v_xi = v_xi_new;
        therm->G_xi = G_xi_new;

        verlet_part1_done = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}


/* Velocity Verlet integrator for constant pressure and constant temperature.
 *
 * NOTE: All box dimensions are scaled by the same amount, and
 * there is no change in the angles between axes. */
int Velocity_Verlet_Berendsen_NPT( reax_system * const system, control_params * const control,
        simulation_data * const data, storage * const workspace, reax_list ** const lists,
        output_controls * const out_control, mpi_datatypes * const mpi_data )
{
    int i, steps, renbr, ret;
    static int verlet_part1_done = FALSE, gen_nbr_list = FALSE;
    real inv_m, dt;
    rvec dx;
    reax_atom *atom;

    ret = SUCCESS;
    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = steps % control->reneighbor == 0 ? TRUE : FALSE;

    if ( verlet_part1_done == FALSE )
    {
        /* velocity verlet, 1st part */
        for ( i = 0; i < system->n; i++ )
        {
            atom = &system->my_atoms[i];
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;

            /* Compute x(t + dt) */
            rvec_ScaledSum( dx, dt, atom->v, -0.5 * F_CONV * inv_m * SQR(dt), atom->f );
            rvec_Add( atom->x, dx );

            /* Compute v(t + dt/2) */
            rvec_ScaledAdd( atom->v, -0.5 * F_CONV * inv_m * dt, atom->f );
        }

        if ( renbr == TRUE )
        {
            Update_Grid( system, control, MPI_COMM_WORLD );
        }

        Comm_Atoms( system, control, data, workspace, mpi_data, renbr );

        verlet_part1_done = TRUE;
    }

    Reallocate_Part2( system, control, data, workspace, lists, mpi_data );

    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE && gen_nbr_list == FALSE )
    {
        ret = Generate_Neighbor_Lists( system, data, workspace, lists );

        if ( ret == SUCCESS )
        {
            gen_nbr_list = TRUE;
        }
        else
        {
            Estimate_Num_Neighbors( system, data, lists[FAR_NBRS]->format );
        }
    }

    if ( ret == SUCCESS )
    {
        ret = Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    }

    if ( ret == SUCCESS )
    {
        /* velocity verlet, 2nd part */
        for ( i = 0; i < system->n; i++ )
        {
            atom = &system->my_atoms[i];
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;

            /* Compute v(t + dt) */
            rvec_ScaledAdd( atom->v, -0.5 * dt * F_CONV * inv_m, atom->f );
        }

        Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
        Compute_Pressure( system, control, data, mpi_data );
        Scale_Box( system, control, data, mpi_data );

        verlet_part1_done = FALSE;
        gen_nbr_list = FALSE;
    }

    return ret;
}
