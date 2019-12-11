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

#include "integrate.h"

#include "allocate.h"
#include "box.h"
#include "charges.h"
#include "forces.h"
#include "grid.h"
#include "neighbors.h"
#include "reset_tools.h"
#include "system_props.h"
#include "vector.h"


/* Velocity Verlet integrator for microcanonical ensemble. */
void Velocity_Verlet_NVE(reax_system *system, control_params *control,
        simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, renbr;
    real inv_m, dt, dt_sqr;
    rvec dx;

    dt = control->dt;
    dt_sqr = SQR(dt);
    renbr = ((data->step - data->prev_steps) % control->reneighbor) == 0 ? TRUE : FALSE;

    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        rvec_ScaledSum( dx, dt, system->atoms[i].v,
                0.5 * dt_sqr * -F_CONV * inv_m, system->atoms[i].f );

        control->update_atom_position( system->atoms[i].x, dx, system->atoms[i].rel_map, &system->box );

        rvec_ScaledAdd( system->atoms[i].v,
                0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
    }

    Reallocate( system, control, workspace, lists, renbr );

    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE )
    {
        Generate_Neighbor_Lists( system, control, data, workspace,
                lists, out_control );
    }

    Compute_Forces( system, control, data, workspace, lists, out_control );

    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;
        rvec_ScaledAdd( system->atoms[i].v,
                0.5 * dt * -F_CONV * inv_m, system->atoms[i].f );
    }
}


/* Velocity Verlet integrator for constant volume and constant temperature
 *  with Berendsen thermostat.
 *
 * NOTE: All box dimensions are scaled by the same amount, and
 * there is no change in the angles between axes during scaling. */
void Velocity_Verlet_Berendsen_NVT( reax_system* system,
        control_params* control, simulation_data *data,
        static_storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, renbr;
    real inv_m, dt, lambda;
    rvec dx;
    reax_atom *atom;

    dt = control->dt;
    renbr = ((data->step - data->prev_steps) % control->reneighbor) == 0 ? TRUE : FALSE;

    /* velocity verlet, 1st part */
    for ( i = 0; i < system->N; i++ )
    {
        atom = &system->atoms[i];
        inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;

        /* Compute x(t + dt) */
        rvec_ScaledSum( dx, dt, atom->v, -0.5 * F_CONV * inv_m * SQR(dt), atom->f );

        control->update_atom_position( atom->x, dx, system->atoms[i].rel_map, &system->box );

        /* Compute v(t + dt/2) */
        rvec_ScaledAdd( atom->v, -0.5 * F_CONV * inv_m * dt, atom->f );
    }

    Reallocate( system, control, workspace, lists, renbr );

    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE )
    {
        Generate_Neighbor_Lists( system, control, data, workspace, lists, out_control );
    }

    Compute_Forces( system, control, data, workspace,
            lists, out_control );

    /* velocity verlet, 2nd part */
    for ( i = 0; i < system->N; i++ )
    {
        atom = &system->atoms[i];
        inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
        /* Compute v(t + dt) */
        rvec_ScaledAdd( atom->v, -0.5 * dt * F_CONV * inv_m, atom->f );
    }

    /* temperature scaler */
    Compute_Kinetic_Energy( system, data );
    lambda = 1.0 + ((dt * 1.0e-10) / control->Tau_T) * (control->T / data->therm.T - 1.0);
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
    for ( i = 0; i < system->N; ++i )
    {
        atom = &system->atoms[i];
        rvec_Scale( atom->v, lambda, atom->v );
    }
    Compute_Kinetic_Energy( system, data );
}

/* Velocity Verlet integrator for constant volume and constant temperature
 *  with Nose-Hoover thermostat.
 *
 * Reference: Understanding Molecular Simulation, Frenkel and Smit
 *  Academic Press Inc. San Diego, 1996 p. 388-391 */
void Velocity_Verlet_Nose_Hoover_NVT_Klein(reax_system* system, control_params* control,
        simulation_data *data, static_storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, itr, renbr;
    real inv_m, coef_v, dt, dt_sqr;
    real E_kin_new, G_xi_new, v_xi_new, v_xi_old;
    rvec dx;
    thermostat *therm;

    dt = control->dt;
    dt_sqr = SQR( dt );
    therm = &data->therm;
    renbr = ((data->step - data->prev_steps) % control->reneighbor) == 0 ? TRUE : FALSE;

    /* Compute x(t + dt) and copy old forces */
    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        rvec_ScaledSum( dx, dt - 0.5 * dt_sqr * therm->v_xi, system->atoms[i].v,
                -0.5 * dt_sqr * inv_m * F_CONV, system->atoms[i].f );

        control->update_atom_position( system->atoms[i].x, dx, system->atoms[i].rel_map, &system->box );

        rvec_Copy( workspace->f_old[i], system->atoms[i].f );
    }

    /* Compute xi(t + dt) */
    therm->xi += therm->v_xi * dt + 0.5 * dt_sqr * therm->G_xi;

    Reallocate( system, control, workspace, lists, renbr );

    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE )
    {
        Generate_Neighbor_Lists( system, control, data, workspace,
                lists, out_control );
    }

    /* Calculate Forces at time (t + dt) */
    Compute_Forces( system, control, data, workspace, lists, out_control );

    /* Compute iteration constants for each atom's velocity */
    for ( i = 0; i < system->N; ++i )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        rvec_Scale( workspace->v_const[i],
                1.0 - 0.5 * dt * therm->v_xi, system->atoms[i].v );
        rvec_ScaledAdd( workspace->v_const[i],
                -0.5 * dt * inv_m * F_CONV, workspace->f_old[i] );
        rvec_ScaledAdd( workspace->v_const[i],
                -0.5 * dt * inv_m * F_CONV, system->atoms[i].f );
    }

    v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
    E_kin_new = 0.0;
    G_xi_new = 0.0;
    v_xi_old = 0.0;
    itr = 0;
    do
    {
        itr++;
        v_xi_old = v_xi_new;
        coef_v = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
        E_kin_new = 0.0;

        for ( i = 0; i < system->N; ++i )
        {
            rvec_Scale( system->atoms[i].v, coef_v, workspace->v_const[i] );

            E_kin_new += 0.5 * system->reax_param.sbp[system->atoms[i].type].mass
                    * rvec_Dot( system->atoms[i].v, system->atoms[i].v );
        }

        G_xi_new = control->Tau_T * (2.0 * E_kin_new
                - data->N_f * K_B * control->T);
        v_xi_new = therm->v_xi + 0.5 * dt * (therm->G_xi + G_xi_new);
    }
    while ( FABS( v_xi_new - v_xi_old ) > 1.0e-5 );

    therm->v_xi_old = therm->v_xi;
    therm->v_xi = v_xi_new;
    therm->G_xi = G_xi_new;
}


/* Uses Berendsen-type coupling for both T and P.
 * All box dimensions are scaled by the same amount,
 * there is no change in the angles between axes. */
void Velocity_Verlet_Berendsen_Isotropic_NPT( reax_system* system,
        control_params* control, simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, renbr;
    real inv_m, dt, lambda, mu;
    rvec dx, mu_3;

    dt = control->dt;
    renbr = ((data->step - data->prev_steps) % control->reneighbor) == 0 ? TRUE : FALSE;

    /* velocity verlet, 1st part */
    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        /* Compute x(t + dt) */
        rvec_ScaledSum( dx, dt, system->atoms[i].v,
                F_CONV * inv_m * -0.5 * SQR(dt), system->atoms[i].f );

        control->update_atom_position( system->atoms[i].x, dx, system->atoms[i].rel_map, &system->box );

        /* Compute v(t + dt/2) */
        rvec_ScaledAdd( system->atoms[i].v,
                F_CONV * inv_m * -0.5 * dt, system->atoms[i].f );
    }

    Reallocate( system, control, workspace, lists, renbr );

    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE )
    {
        Update_Grid( system );
        Generate_Neighbor_Lists( system, control, data, workspace,
                lists, out_control );
    }

    Compute_Forces( system, control, data, workspace, lists, out_control );

    /* velocity verlet, 2nd part */
    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;
        /* Compute v(t + dt) */
        rvec_ScaledAdd( system->atoms[i].v,
                -0.5 * dt * F_CONV * inv_m, system->atoms[i].f );
    }

    Compute_Kinetic_Energy( system, data );

    Compute_Pressure_Isotropic( system, control, data, out_control );

    /* pressure scaler */
    for ( i = 0; i < 3; ++i )
    {
        mu_3[i] = POW( 1.0 + (dt / control->Tau_P[i])
                * (data->tot_press[i] - control->P[i]), 1.0 / 3.0 );

        if ( mu_3[i] < MIN_dV )
        {
            mu_3[i] = MIN_dV;
        }
        else if ( mu_3[i] > MAX_dV )
        {
            mu_3[i] = MAX_dV;
        }
    }
    mu = (mu_3[0] + mu_3[1] + mu_3[2]) / 3.0;

    /* temperature scaler */
    lambda = 1.0 + ((dt * 1.0e-10) / control->Tau_T) * (control->T / data->therm.T - 1.0);
    if ( lambda < MIN_dT )
    {
        lambda = MIN_dT;
    }
    else if (lambda > MAX_dT )
    {
        lambda = MAX_dT;
    }
    lambda = SQRT( lambda );

    /* Scale velocities and positions at t+dt */
    for ( i = 0; i < system->N; ++i )
    {
        rvec_Scale( system->atoms[i].v, lambda, system->atoms[i].v );

        /* IMPORTANT: What Adri does with scaling positions first to
         * unit coordinates and then back to cartesian coordinates essentially
         * is scaling the coordinates with mu^2. However, this causes unphysical
         * modifications on the system because box dimensions
         * are being scaled with mu! We need to discuss this with Adri! */
        rvec_Scale( system->atoms[i].x, mu, system->atoms[i].x );
    }

    Compute_Kinetic_Energy( system, data );

    Update_Box_Isotropic( &system->box, mu );
}


/* uses Berendsen-type coupling for both T and P.
 * All box dimensions are scaled by the same amount,
 * there is no change in the angles between axes. */
void Velocity_Verlet_Berendsen_Semi_Isotropic_NPT( reax_system* system,
        control_params* control, simulation_data *data, static_storage *workspace,
        reax_list **lists, output_controls *out_control )
{
    int i, renbr;
    real dt, inv_m, lambda;
    rvec dx, mu;

    dt = control->dt;
    renbr = ((data->step - data->prev_steps) % control->reneighbor) == 0 ? TRUE : FALSE;

    /* velocity verlet, 1st part */
    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        /* Compute x(t + dt) */
        rvec_ScaledSum( dx, dt, system->atoms[i].v,
                -0.5 * F_CONV * inv_m * SQR(dt), system->atoms[i].f );

        control->update_atom_position( system->atoms[i].x, dx, system->atoms[i].rel_map, &system->box );

        /* Compute v(t + dt/2) */
        rvec_ScaledAdd( system->atoms[i].v,
                -0.5 * F_CONV * inv_m * dt, system->atoms[i].f );
    }

    Reallocate( system, control, workspace, lists, renbr );

    Reset( system, control, data, workspace, lists );

    if ( renbr == TRUE )
    {
        Update_Grid( system );
        Generate_Neighbor_Lists( system, control, data, workspace,
                lists, out_control );
    }

    Compute_Forces( system, control, data, workspace, lists, out_control );

    /* velocity verlet, 2nd part */
    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;
        /* Compute v(t + dt) */
        rvec_ScaledAdd( system->atoms[i].v,
                -0.5 * dt * F_CONV * inv_m, system->atoms[i].f );
    }

    Compute_Kinetic_Energy( system, data );

    Compute_Pressure_Isotropic( system, control, data, out_control );

    /* pressure scaler */
    for ( i = 0; i < 3; ++i )
    {
        mu[i] = POW( 1.0 + (dt / control->Tau_P[i])
                * (data->tot_press[i] - control->P[i]), 1.0 / 3.0 );
        if ( mu[i] < MIN_dV )
        {
            mu[i] = MIN_dV;
        }
        else if ( mu[i] > MAX_dV )
        {
            mu[i] = MAX_dV;
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
    for ( i = 0; i < system->N; ++i )
    {
        rvec_Scale( system->atoms[i].v, lambda, system->atoms[i].v );

        /* IMPORTANT: What Adri does with scaling positions first to
         * unit coordinates and then back to cartesian coordinates essentially
         * is scaling the coordinates with mu^2. However, this causes unphysical
         * modifications on the system because box dimensions
         * are being scaled with mu! We need to discuss this with Adri! */
        rvec_Multiply( system->atoms[i].x, mu, system->atoms[i].x );
    }

    Compute_Kinetic_Energy( system, data );

    Update_Box_Semi_Isotropic( &system->box, mu );
}


/************************************************/
/* BELOW FUNCTIONS ARE NOT BEING USED ANYMORE!  */
/************************************************/
#if defined(ANISOTROPIC)
void Velocity_Verlet_Nose_Hoover_NVT( reax_system* system,
        control_params* control, simulation_data *data,
        static_storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i;
    real inv_m;
    real dt = control->dt;
    real dt_sqr = SQR(dt);
    rvec dx;

    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        /* Compute x(t + dt) */
        rvec_ScaledSum( dx, dt, system->atoms[i].v,
                -0.5 * dt_sqr * F_CONV * inv_m, system->atoms[i].f );

        Update_Atom_Position_Triclinic( control, &system->box,
                system->atoms[i].x, dx, system->atoms[i].rel_map );

        /* Compute v(t + dt/2) */
        rvec_ScaledAdd( system->atoms[i].v,
                -0.5 * dt * data->therm.xi, system->atoms[i].v );
        rvec_ScaledAdd( system->atoms[i].v,
                -0.5 * dt * F_CONV * inv_m, system->atoms[i].f );
    }

    /* Compute zeta(t + dt/2), E_Kininetic(t + dt/2)
     * IMPORTANT: What will be the initial value of zeta? and what is g? */
    data->therm.xi += 0.5 * dt * control->Tau_T
        * (2.0 * data->E_Kin - data->N_f * K_B * control->T);

    Reset( system, control, data, workspace );
    fprintf(out_control->log, "reset-");
    fflush( out_control->log );

    Generate_Neighbor_Lists( system, control, data, workspace,
                             lists, out_control );
    fprintf(out_control->log, "nbrs-");
    fflush( out_control->log );

//    Compute_Charges( system, control, workspace, out_control );
//    fprintf( out_control->log, "qeq-" );
//    fflush( out_control->log );

    Compute_Forces( system, control, data, workspace, lists, out_control );
    fprintf(out_control->log, "forces\n");
    fflush( out_control->log );

    Compute_Kinetic_Energy( system, data );

    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        /* compute v(t + dt) */
        rvec_ScaledAdd( system->atoms[i].v,
                -0.5 * dt * data->therm.xi, system->atoms[i].v );
        rvec_ScaledAdd( system->atoms[i].v,
                -0.5 * dt * F_CONV * inv_m, system->atoms[i].f );
    }

    /* Compute zeta(t + dt) */
    data->therm.xi += 0.5 * dt * control->Tau_T
        * (2.0 * data->E_Kin - data->N_f * K_B * control->T);

    fprintf( out_control->log, "Xi: %8.3f %8.3f %8.3f\n",
             data->therm.xi, data->E_Kin, data->N_f * K_B * control->T );
    fflush( out_control->log );
}


void Velocity_Verlet_Isotropic_NPT( reax_system* system,
        control_params* control, simulation_data *data,
        static_storage *workspace, reax_list **lists,
        output_controls *out_control )
{
    int i, itr;
    real deps, v_eps_new = 0, v_eps_old = 0, G_xi_new;
    real dxi, v_xi_new = 0, v_xi_old = 0, a_eps_new;
    real inv_m, exp_deps, inv_3V;
    real E_kin, P_int, P_int_const;
    real coef_v, coef_v_eps;
    real dt = control->dt;
    real dt_sqr = SQR( dt );
    thermostat *therm = &data->therm;
    isotropic_barostat *iso_bar = &data->iso_bar;
    simulation_box *box = &system->box;
    rvec dx, dv;

    // Here we just calculate how much to increment eps, xi, v_eps, v_xi.
    // Commits are done after positions and velocities of atoms are updated
    // because position, velocity updates uses v_eps, v_xi terms;
    // yet we need EXP( deps ) to be able to calculate
    // positions and velocities accurately.
    iso_bar->a_eps = control->Tau_P *
                     ( 3.0 * box->volume * (iso_bar->P - control->P) +
                       6.0 * data->E_Kin / data->N_f ) - iso_bar->v_eps * therm->v_xi;
    deps = dt * iso_bar->v_eps + 0.5 * dt_sqr * iso_bar->a_eps;
    exp_deps = EXP( deps );

    therm->G_xi = control->Tau_T * ( 2.0 * data->E_Kin +
                                     SQR( iso_bar->v_eps ) / control->Tau_P -
                                     (data->N_f + 1) * K_B * control->T );
    dxi = therm->v_xi * dt + 0.5 * therm->G_xi * dt_sqr;

    fprintf(out_control->log, "a: %12.6f   eps: %12.6f   deps: %12.6f\n",
            iso_bar->a_eps, iso_bar->v_eps, iso_bar->eps);
    fprintf(out_control->log, "G: %12.6f   xi : %12.6f   dxi : %12.6f\n",
            therm->G_xi, therm->v_xi, therm->xi );

    // Update positions and velocities
    // NOTE: v_old, v_xi_old, v_eps_old are meant to be the old values
    // in the iteration not the old values at time t or before!
    for ( i = 0; i < system->N; i++ )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        /* Compute x(t + dt) */
        rvec_ScaledSum( workspace->a[i], -1.0 * F_CONV * inv_m, system->atoms[i].f,
                -1.0 * ( (2.0 + 3.0 / data->N_f) * iso_bar->v_eps + therm->v_xi ),
                system->atoms[i].v );

        rvec_ScaledSum( dx, dt, system->atoms[i].v,
                0.5 * dt_sqr, workspace->a[i] );

        control->update_atom_position( system->atoms[i].x, dx, system->atoms[i].rel_map, &system->box );

        rvec_Scale( system->atoms[i].x, exp_deps, system->atoms[i].x );
    }

    // Commit updates
    therm->xi += dxi;
    iso_bar->eps += deps;
    Update_Box_Isotropic( &system->box, EXP( 3.0 * iso_bar->eps ) );

    /* Calculate new forces, f(t + dt) */
    Reset( system, control, data, workspace );
    fprintf(out_control->log, "reset-");
    fflush( out_control->log );

    Generate_Neighbor_Lists( system, control, data, workspace,
                             lists, out_control );
    fprintf( out_control->log, "nbrs-" );
    fflush( out_control->log );

//    Compute_Charges( system, control, workspace, out_control );
//    fprintf( out_control->log, "qeq-" );
//    fflush( out_control->log );

    Compute_Forces( system, control, data, workspace, lists,
            out_control );
    fprintf(out_control->log, "forces\n");
    fflush( out_control->log );

    // Compute iteration constants for each atom's velocity and for P_internal
    // Compute kinetic energy for initial velocities of the iteration
    P_int_const = E_kin = 0;
    for ( i = 0; i < system->N; ++i )
    {
        inv_m = 1.0 / system->reax_param.sbp[system->atoms[i].type].mass;

        rvec_ScaledSum( dv, 0.5 * dt, workspace->a[i],
                -0.5 * dt * F_CONV * inv_m, system->atoms[i].f );
        rvec_Add( dv, system->atoms[i].v );
        rvec_Scale( workspace->v_const[i], exp_deps, dv );

        P_int_const += ( -1.0 * F_CONV * rvec_Dot( system->atoms[i].f, system->atoms[i].x ) );

        E_kin += (0.5 * system->reax_param.sbp[system->atoms[i].type].mass
                * rvec_Dot( system->atoms[i].v, system->atoms[i].v ) );
    }

    // Compute initial p_int
    inv_3V = 1.0 / (3.0 * system->box.volume);
    P_int = inv_3V * ( 2.0 * E_kin + P_int_const );

    v_xi_new = therm->v_xi_old + 2.0 * dt * therm->G_xi;
    v_eps_new = iso_bar->v_eps_old + 2.0 * dt * iso_bar->a_eps;

    itr = 0;
    do
    {
        itr++;
        // new values become old in this iteration
        v_xi_old = v_xi_new;
        v_eps_old = v_eps_new;

        for ( i = 0; i < system->N; ++i )
        {
            coef_v = 1.0 / (1.0 + 0.5 * dt * exp_deps *
                            ( (2.0 + 3.0 / data->N_f) * v_eps_old + v_xi_old ) );
            rvec_Scale( system->atoms[i].v, coef_v, workspace->v_const[i] );
        }


        coef_v_eps = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
        a_eps_new = 3.0 * control->Tau_P *
                    ( system->box.volume * (P_int - control->P) + 2.0 * E_kin / data->N_f );
        v_eps_new = coef_v_eps * ( iso_bar->v_eps +
                                   0.5 * dt * ( iso_bar->a_eps + a_eps_new ) );

        G_xi_new = control->Tau_T * ( 2.0 * E_kin +
                                      SQR( v_eps_old ) / control->Tau_P -
                                      (data->N_f + 1) * K_B * control->T );
        v_xi_new = therm->v_xi + 0.5 * dt * ( therm->G_xi + G_xi_new );

        E_kin = 0;
        for ( i = 0; i < system->N; ++i )
        {
            E_kin += (0.5 * system->reax_param.sbp[system->atoms[i].type].mass *
                      rvec_Dot( system->atoms[i].v, system->atoms[i].v ) );
        }

        P_int = inv_3V * ( 2.0 * E_kin + P_int_const );

        fprintf( out_control->log,
                 "itr %d E_kin: %8.3f veps_n:%8.3f veps_o:%8.3f vxi_n:%8.3f vxi_o: %8.3f\n",
                 itr, E_kin, v_eps_new, v_eps_old, v_xi_new, v_xi_old );
    }
    while ( FABS(v_eps_new - v_eps_old) + FABS(v_xi_new - v_xi_old) > 2e-3 );

    therm->v_xi_old = therm->v_xi;
    therm->v_xi = v_xi_new;
    therm->G_xi = G_xi_new;

    iso_bar->v_eps_old = iso_bar->v_eps;
    iso_bar->v_eps = v_eps_new;
    iso_bar->a_eps = a_eps_new;

    fprintf( out_control->log, "V: %8.3ff\tsides{%8.3f, %8.3f, %8.3f}\n",
             system->box.volume,
             system->box.box[0][0], system->box.box[1][1], system->box.box[2][2] );
    fprintf(out_control->log, "eps:\ta- %8.3f  v- %8.3f  eps- %8.3f\n",
            iso_bar->a_eps, iso_bar->v_eps, iso_bar->eps);
    fprintf(out_control->log, "xi: \tG- %8.3f  v- %8.3f  xi - %8.3f\n",
            therm->G_xi, therm->v_xi, therm->xi);
}
#endif
