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

#ifdef HAVE_CUDA
  #include "cuda_integrate.h"
  #include "cuda_copy.h"
  #include "cuda_neighbors.h"
#endif


int Velocity_Verlet_NVE( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    int i, steps, renbr, ret;
    real inv_m, dt, dt_sqr;
    rvec dx;
    reax_atom *atom;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step %d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    ret = ReAllocate( system, control, data, workspace, lists, mpi_data );

    if ( ret == SUCCESS )
    {
        dt = control->dt;
        dt_sqr = SQR(dt);
        steps = data->step - data->prev_steps;
        renbr = (steps % control->reneighbor == 0);
    
        for ( i = 0; i < system->n; i++ )
        {
            atom = &(system->my_atoms[i]);
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
            rvec_ScaledSum( dx, dt, atom->v, 0.5 * dt_sqr * -F_CONV * inv_m, atom->f );
            rvec_Add( system->my_atoms[i].x, dx );
            rvec_ScaledAdd( atom->v, 0.5 * dt * -F_CONV * inv_m, atom->f );
        }

#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: verlet1 done\n", system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif

        Comm_Atoms( system, control, data, workspace, lists, mpi_data, renbr );
        Reset( system, control, data, workspace, lists );
        if ( renbr )
        {
            Generate_Neighbor_Lists( system, data, workspace, lists );
        }
        Compute_Forces(system, control, data, workspace, lists, out_control, mpi_data);
    
        for ( i = 0; i < system->n; i++ )
        {
            atom = &(system->my_atoms[i]);
            inv_m = 1.0 / system->reax_param.sbp[ atom->type ].mass;
            rvec_ScaledAdd( atom->v, 0.5 * dt * -F_CONV * inv_m, atom->f );
        }

#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: verlet2 done\n", system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    }

    return ret;
}


int Velocity_Verlet_Nose_Hoover_NVT_Klein( reax_system* system,
        control_params* control, simulation_data *data, storage *workspace,
        reax_list **lists, output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, itr, steps, renbr, ret;
    real inv_m, coef_v;
    real dt, dt_sqr;
    real my_ekin, new_ekin;
    real G_xi_new, v_xi_new, v_xi_old;
    rvec dx;
    thermostat *therm;
    reax_atom *atom;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    dt = control->dt;
    dt_sqr = SQR(dt);
    therm = &( data->therm );
    steps = data->step - data->prev_steps;
    renbr = (steps % control->reneighbor == 0);

    for ( i = 0; i < system->n; i++ )
    {
        atom = &(system->my_atoms[i]);
        inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
        rvec_ScaledSum( dx, dt, atom->v, 0.5 * dt_sqr * -F_CONV * inv_m, atom->f );
        rvec_Add( system->my_atoms[i].x, dx );
        rvec_Copy( atom->f_old, atom->f );
    }

    /* Compute xi(t + dt) */
    therm->xi += ( therm->v_xi * dt + 0.5 * dt_sqr * therm->G_xi );

#if defined(DEBUG_FOCUS)
    fprintf(stderr, "p%d @ step%d: verlet1 done\n", system->my_rank, data->step);
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    ret = ReAllocate( system, control, data, workspace, lists, mpi_data );

    if ( ret == SUCCESS )
    {
        Comm_Atoms( system, control, data, workspace, lists, mpi_data, renbr );
        Reset( system, control, data, workspace, lists );
        if ( renbr )
        {
            Generate_Neighbor_Lists( system, data, workspace, lists );
        }
        Compute_Forces( system, control, data, workspace, lists, out_control, mpi_data );
    
        /* Compute iteration constants for each atom's velocity */
        for ( i = 0; i < system->n; ++i )
        {
            atom = &(system->my_atoms[i]);
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
            rvec_Scale( workspace->v_const[i], 1.0 - 0.5 * dt * therm->v_xi, atom->v );
            rvec_ScaledAdd( workspace->v_const[i], 0.5 * dt * inv_m * -F_CONV, atom->f_old );
            rvec_ScaledAdd( workspace->v_const[i], 0.5 * dt * inv_m * -F_CONV, atom->f );
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
                atom = &(system->my_atoms[i]);
                coef_v = 1.0 / (1.0 + 0.5 * dt * v_xi_old);
                rvec_Scale( atom->v, coef_v, workspace->v_const[i] );
                my_ekin +=
                    (0.5 * system->reax_param.sbp[atom->type].mass * rvec_Dot(atom->v, atom->v));
            }
    
            MPI_Allreduce( &my_ekin, &new_ekin, 1, MPI_DOUBLE, MPI_SUM,
                           mpi_data->comm_mesh3D  );
    
            G_xi_new = control->Tau_T * ( 2.0 * new_ekin - data->N_f * K_B * control->T );
            v_xi_new = therm->v_xi + 0.5 * dt * ( therm->G_xi + G_xi_new );
        }
        while ( FABS(v_xi_new - v_xi_old) > 1e-5 );
        therm->v_xi_old = therm->v_xi;
        therm->v_xi = v_xi_new;
        therm->G_xi = G_xi_new;

#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: T-coupling\n", system->my_rank, data->step);
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    }

    return ret;
}


/* uses Berendsen-type coupling for both T and P.
   All box dimensions are scaled by the same amount,
   there is no change in the angles between axes. */
int Velocity_Verlet_Berendsen_NVT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, steps, renbr, ret;
    real inv_m, dt, lambda;
    rvec dx;
    reax_atom *atom;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif
    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = (steps % control->reneighbor == 0);

    /* velocity verlet, 1st part */
    for ( i = 0; i < system->n; i++ )
    {
        atom = &(system->my_atoms[i]);
        inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
        /* Compute x(t + dt) */
        rvec_ScaledSum( dx, dt, atom->v, 0.5 * -F_CONV * inv_m * SQR(dt), atom->f );
        rvec_Add( atom->x, dx );
        /* Compute v(t + dt/2) */
        rvec_ScaledAdd( atom->v, 0.5 * -F_CONV * inv_m * dt, atom->f );
    }

#if defined(DEBUG_FOCUS)
    fprintf(stderr, "p%d @ step%d: verlet1 done\n", system->my_rank, data->step);
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    ret = ReAllocate( system, control, data, workspace, lists, mpi_data );

    if ( ret == SUCCESS )
    {
        if ( renbr )
        {
            Update_Grid( system, control, mpi_data->world );
        }
        Comm_Atoms( system, control, data, workspace, lists, mpi_data, renbr );
        Reset( system, control, data, workspace, lists );
        if ( renbr )
        {
            Generate_Neighbor_Lists( system, data, workspace, lists );
        }
        Compute_Forces( system, control, data, workspace,
                        lists, out_control, mpi_data );
    
        /* velocity verlet, 2nd part */
        for ( i = 0; i < system->n; i++ )
        {
            atom = &(system->my_atoms[i]);
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
            /* Compute v(t + dt) */
            rvec_ScaledAdd( atom->v, 0.5 * dt * -F_CONV * inv_m, atom->f );
        }
    
#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: verlet2 done\n", system->my_rank, data->step);
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    
        /* temperature scaler */
        Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
        lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
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
        for ( i = 0; i < system->n; ++i )
        {
            atom = &(system->my_atoms[i]);
            rvec_Scale( atom->v, lambda, atom->v );
        }
        Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
    
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: scaled velocities\n",
                 system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    }

    return ret;
}


#ifdef HAVE_CUDA
int Cuda_Velocity_Verlet_Berendsen_NVT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, steps, renbr, ret;
    real inv_m, dt, lambda;
    rvec dx;
    reax_atom *atom;

    int *nbr_indices, num_nbrs;
    int *bond_top, *hb_top;
    int Htop, num_3body;
    int total_hbonds, count, total_bonds;
    int bond_cap, cap_3body;

    real t_over_start, t_over_elapsed;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = (steps % control->reneighbor == 0);

    /* velocity verlet, 1st part */
    bNVT_update_velocity_part1( system, dt );

#if defined(DEBUG_FOCUS)
    fprintf(stderr, "p%d @ step%d: verlet1 done\n", system->my_rank, data->step);
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    ret = Cuda_ReAllocate( system, control, data, workspace, lists, mpi_data );

    if ( ret == SUCCESS )
    {
        if ( renbr )
        {
            Update_Grid( system, control, mpi_data->world );
        }
    
        Output_Sync_Atoms( system );
        Comm_Atoms( system, control, data, workspace, lists, mpi_data, renbr );
        Sync_Atoms( system );
    
        /* synch the Grid to the Device here */
        Sync_Grid( &system->my_grid, &system->d_my_grid );
    
        init_blocks( system );
    
#if defined(__CUDA_DEBUG_LOG__)
        fprintf( stderr, "p:%d - Matvec BLocks: %d, blocksize: %d \n",
                system->my_rank, MATVEC_BLOCKS, MATVEC_BLOCK_SIZE );
#endif
    
        //Reset( system, control, data, workspace, lists );
        Cuda_Reset( system, control, data, workspace, lists );
    
        if ( renbr )
        {
#if defined(DEBUG)
            t_over_start  = Get_Time ();
#endif
    
            nbr_indices = (int *) host_scratch;
            memset( nbr_indices, 0, sizeof(int) * system->N );
    
            Cuda_Estimate_Neighbors( system, nbr_indices );
    
            num_nbrs = 0;
            for (i = 0; i < system->N; i++)
            {
                num_nbrs += nbr_indices[i];
            }
    
            num_nbrs = 0;
            for (i = 0; i < system->N; i++)
            {
                nbr_indices[i] = MAX( nbr_indices[i] * SAFE_ZONE, MIN_NBRS );
                num_nbrs += nbr_indices[i];
            }
    
            if (num_nbrs >= (*dev_lists + FAR_NBRS)->num_intrs)
            {
                fprintf( stderr, "p%d: Total neighbors: %d is greater than available entries: %d \n",
                         system->my_rank, num_nbrs, (*dev_lists + FAR_NBRS)->num_intrs );
                exit( 0 );
            }
    
            for (i = 1; i < system->N; i++)
            {
                nbr_indices[i] += nbr_indices[i - 1];
            }
    
            Cuda_Init_Neighbors_Indices( nbr_indices, system->N );
            Cuda_Generate_Neighbor_Lists( system, data, workspace, lists );
    
            /*
            memset (host_scratch, 0, sizeof (int) * (2 * system->total_cap));
            bond_top = (int *) host_scratch;
            hb_top = bond_top + system->total_cap;
            Htop = 0;
            num_3body = 0;
    
            Cuda_Estimate_Storages( system, control, lists, system->local_cap, system->total_cap,
                 &Htop, hb_top, bond_top, &num_3body );
    
            if( control->hbond_cut > 0 ) {
            total_hbonds = 0;
            count = 0;
    
            for( i = 0; i < system->N; ++i ) {
            hb_top [i] = MAX( hb_top[i] * 2, MIN_HBONDS * 2);
            total_hbonds += hb_top[i];
            if (hb_top [i] > 0) count ++;
            }
            total_hbonds = MAX( total_hbonds, MIN_CAP*MIN_HBONDS );
    
            if (total_hbonds >= (*dev_lists + HBONDS)->num_intrs){
                fprintf (stderr, "p%d: Total HBonds: %d and allocated: %d \n",
                                        system->my_rank, total_hbonds, (*dev_lists + HBONDS)->num_intrs);
                exit (0);
            }
            Cuda_Init_HBond_Indices (hb_top, system->N);
            }
    
            // bonds list
            total_bonds = 0;
            for( i = 0; i < system->N; ++i ) {
            num_3body += SQR (bond_top [i]);
            total_bonds += MAX (bond_top[i] * 2, MIN_BONDS);
            }
            bond_cap = MAX( total_bonds, MIN_CAP*MIN_BONDS );
    
            if (total_bonds >= (*dev_lists + BONDS)->num_intrs){
                fprintf (stderr, "p:%d Bonds: %d and allocated: %d \n",
                                        system->my_rank, total_hbonds, (*dev_lists + BONDS)->num_intrs);
                exit (0);
            }
    
            Cuda_Init_Bond_Indices (bond_top, system->N, bond_cap);
            */
    
#if defined(DEBUG)
            t_over_elapsed  = Get_Timing_Info (t_over_start);
            fprintf (stderr, "p%d --> Overhead (Step-%d) %f \n",
                     system->my_rank, data->step, t_over_elapsed);
#endif
        }
    
        //Compute_Forces( system, control, data, workspace,
        //          lists, out_control, mpi_data );
        Cuda_Compute_Forces( system, control, data, workspace,
                lists, out_control, mpi_data );
    
        /* velocity verlet, 2nd part */
        bNVT_update_velocity_part2( system, dt );
    
#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: verlet2 done\n", system->my_rank, data->step);
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    
        /* temperature scaler */
        //Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
        Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
    
        lambda = 1.0 + (dt / control->Tau_T) * (control->T / data->therm.T - 1.0);
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
        bNVT_scale_velocities( system, lambda );
    
        //Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
        Cuda_Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
    
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: scaled velocities\n",
                 system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    }

    return ret;
}
#endif


/* uses Berendsen-type coupling for both T and P.
   All box dimensions are scaled by the same amount,
   there is no change in the angles between axes. */
int Velocity_Verlet_Berendsen_NPT( reax_system* system, control_params* control,
        simulation_data *data, storage *workspace, reax_list **lists,
        output_controls *out_control, mpi_datatypes *mpi_data )
{
    int i, steps, renbr, ret;
    real inv_m, dt;
    rvec dx;
    reax_atom *atom;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d @ step%d\n", system->my_rank, data->step );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    dt = control->dt;
    steps = data->step - data->prev_steps;
    renbr = (steps % control->reneighbor == 0);

    /* velocity verlet, 1st part */
    for ( i = 0; i < system->n; i++ )
    {
        atom = &(system->my_atoms[i]);
        inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
        /* Compute x(t + dt) */
        rvec_ScaledSum( dx, dt, atom->v, 0.5 * -F_CONV * inv_m * SQR(dt), atom->f );
        rvec_Add( atom->x, dx );
        /* Compute v(t + dt/2) */
        rvec_ScaledAdd( atom->v, 0.5 * -F_CONV * inv_m * dt, atom->f );
    }

#if defined(DEBUG_FOCUS)
    fprintf(stderr, "p%d @ step%d: verlet1 done\n", system->my_rank, data->step);
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    ret = ReAllocate( system, control, data, workspace, lists, mpi_data );

    if ( ret == SUCCESS )
    {
        if ( renbr )
        {
            Update_Grid( system, control, mpi_data->world );
        }
        Comm_Atoms( system, control, data, workspace, lists, mpi_data, renbr );
        Reset( system, control, data, workspace, lists );
        if ( renbr )
        {
            Generate_Neighbor_Lists( system, data, workspace, lists );
        }
        Compute_Forces( system, control, data, workspace,
                        lists, out_control, mpi_data );
    
        /* velocity verlet, 2nd part */
        for ( i = 0; i < system->n; i++ )
        {
            atom = &(system->my_atoms[i]);
            inv_m = 1.0 / system->reax_param.sbp[atom->type].mass;
            /* Compute v(t + dt) */
            rvec_ScaledAdd( atom->v, 0.5 * dt * -F_CONV * inv_m, atom->f );
        }
    
#if defined(DEBUG_FOCUS)
        fprintf(stderr, "p%d @ step%d: verlet2 done\n", system->my_rank, data->step);
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    
        Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
        Compute_Pressure( system, control, data, mpi_data );
        Scale_Box( system, control, data, mpi_data );
    
#if defined(DEBUG_FOCUS)
        fprintf( stderr, "p%d @ step%d: scaled box\n", system->my_rank, data->step );
        MPI_Barrier( MPI_COMM_WORLD );
#endif
    }

    return ret;
}
