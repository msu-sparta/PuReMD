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
#include "analyze.h"
#include "comm_tools.h"
#include "control.h"
#include "ffield.h"
#include "forces.h"
#include "geo_tools.h"
#include "init_md.h"
#include "integrate.h"
#include "io_tools.h"
#include "neighbors.h"
#include "reset_tools.h"
#include "restart.h"
#include "system_props.h"
#include "tool_box.h"
#include "traj.h"
#include "vector.h"


void Read_System( char *geo_file, char *ffield_file, char *control_file,
                  reax_system *system, control_params *control,
                  simulation_data *data, storage *workspace,
                  output_controls *out_control, mpi_datatypes *mpi_data )
{
    /* ffield file */
    Read_Force_Field( ffield_file, &(system->reax_param), control );

    /* control file */
    Read_Control_File( control_file, control, out_control );

    /* geo file */
    if ( control->geo_format == CUSTOM )
    {
        Read_Geo( geo_file, system, control, data, workspace, mpi_data );
    }
    else if ( control->geo_format == PDB )
    {
        Read_PDB( geo_file, system, control, data, workspace, mpi_data );
    }
    else if ( control->geo_format == ASCII_RESTART )
    {
        Read_Restart( geo_file, system, control, data, workspace, mpi_data );
        control->restart = 1;
    }
    else if ( control->geo_format == BINARY_RESTART )
    {
        Read_Binary_Restart( geo_file, system, control, data, workspace, mpi_data );
        control->restart = 1;
    }
    else
    {
        fprintf( stderr, "unknown geo file format. terminating!\n" );
        MPI_Abort( MPI_COMM_WORLD, UNKNOWN_OPTION );
    }
}


void Post_Evolve( reax_system* system, control_params* control,
                  simulation_data* data, storage* workspace,
                  reax_list** lists, output_controls *out_control,
                  mpi_datatypes *mpi_data )
{
    int i;
    rvec diff, cross;

    /* remove trans & rot velocity of the center of mass from system */
    if ( control->ensemble != NVE && control->remove_CoM_vel &&
            data->step % control->remove_CoM_vel == 0 )
    {
        /* compute velocity of the center of mass */
        Compute_Center_of_Mass( system, data, mpi_data, mpi_data->comm_mesh3D );

        for ( i = 0; i < system->n; i++ )
        {
            /* remove translational vel */
            rvec_ScaledAdd( system->my_atoms[i].v, -1., data->vcm );

            /* remove rotational */
            rvec_ScaledSum( diff, 1., system->my_atoms[i].x, -1., data->xcm );
            rvec_Cross( cross, data->avcm, diff );
            rvec_ScaledAdd( system->my_atoms[i].v, -1., cross );
        }
    }

    /* compute kinetic energy of the system */
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
}

static void usage(char* argv[])
{
    fprintf(stderr, "usage: ./%s geometry ffield control\n", argv[0]);
}


int main( int argc, char* argv[] )
{
    reax_system *system;
    control_params *control;
    simulation_data *data;
    storage *workspace;
    reax_list **lists;
    output_controls *out_control;
    mpi_datatypes *mpi_data;
    int i;
    real t_start = 0, t_elapsed;

    if ( argc != 4 )
    {
        usage(argv);
        exit( INVALID_INPUT );
    }

    /* allocated main datastructures */
    system = (reax_system *)
             smalloc( sizeof(reax_system), "system", MPI_COMM_WORLD );
    control = (control_params *)
              smalloc( sizeof(control_params), "control", MPI_COMM_WORLD );
    data = (simulation_data *)
           smalloc( sizeof(simulation_data), "data", MPI_COMM_WORLD );

    workspace = (storage *)
                smalloc( sizeof(storage), "workspace", MPI_COMM_WORLD );
    lists = (reax_list **)
            smalloc( LIST_N * sizeof(reax_list*), "lists", MPI_COMM_WORLD );
    for ( i = 0; i < LIST_N; ++i )
    {
        lists[i] = (reax_list *)
                   smalloc( sizeof(reax_list), "lists[i]", MPI_COMM_WORLD );
        lists[i]->allocated = 0;
    }
    out_control = (output_controls *)
                  smalloc( sizeof(output_controls), "out_control", MPI_COMM_WORLD );
    mpi_data = (mpi_datatypes *)
               smalloc( sizeof(mpi_datatypes), "mpi_data", MPI_COMM_WORLD );

    /* setup the parallel environment */
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &(control->nprocs) );
    MPI_Comm_rank( MPI_COMM_WORLD, &(system->my_rank) );
    system->wsize = control->nprocs;
    system->global_offset = (int*)
                            scalloc( system->wsize + 1, sizeof(int), "global_offset", MPI_COMM_WORLD );

    /* read system description files */
    Read_System( argv[1], argv[2], argv[3], system, control,
                 data, workspace, out_control, mpi_data );
#if defined(DEBUG)
    fprintf( stderr, "p%d: read simulation info\n", system->my_rank );
    MPI_Barrier( MPI_COMM_WORLD );
#endif

    /* measure total simulation time after input is read */
    if ( system->my_rank == MASTER_NODE )
        t_start = Get_Time( );

    /* initialize datastructures */
    Initialize( system, control, data, workspace, lists, out_control, mpi_data );
#if defined(DEBUG)
    fprintf( stderr, "p%d: initializated data structures\n", system->my_rank );
    MPI_Barrier( mpi_data->world );
#endif

    /* compute f_0 */
    Comm_Atoms( system, control, data, workspace, lists, mpi_data, 1 );
    Reset( system, control, data, workspace, lists, MPI_COMM_WORLD );
    Generate_Neighbor_Lists( system, data, workspace, lists );
    Compute_Forces( system, control, data, workspace,
                    lists, out_control, mpi_data );
    Compute_Kinetic_Energy( system, data, mpi_data->comm_mesh3D );
    Output_Results( system, control, data, lists, out_control, mpi_data );
#if defined(DEBUG)
    fprintf( stderr, "p%d: computed forces at t0\n", system->my_rank );
    MPI_Barrier( mpi_data->world );
#endif

    /* start the simulation */
    for ( ++data->step; data->step <= control->nsteps; data->step++ )
    {
        if ( control->T_mode )
            Temperature_Control( control, data );

        Evolve( system, control, data, workspace, lists, out_control, mpi_data );
        Post_Evolve(system, control, data, workspace, lists, out_control, mpi_data);
        Output_Results( system, control, data, lists, out_control, mpi_data );
        //Analysis(system, control, data, workspace, lists, out_control, mpi_data);

        /* dump restart info */
        if ( out_control->restart_freq &&
                (data->step - data->prev_steps) % out_control->restart_freq == 0 )
        {
            if ( out_control->restart_format == WRITE_ASCII )
                Write_Restart( system, control, data, out_control, mpi_data );
            else if ( out_control->restart_format == WRITE_BINARY )
                Write_Binary_Restart( system, control, data, out_control, mpi_data );
        }
#if defined(DEBUG)
        fprintf( stderr, "p%d: step%d completed\n", system->my_rank, data->step );
        MPI_Barrier( mpi_data->world );
#endif
    }

    /* end of the simulation, write total simulation time */
    if ( system->my_rank == MASTER_NODE )
    {
        t_elapsed = Get_Timing_Info( t_start );
        fprintf( out_control->out, "Total Simulation Time: %.2f secs\n", t_elapsed );
    }

    // Write_PDB( &system, &(lists[BOND]), &out_control );
    Close_Output_Files( system, control, out_control, mpi_data );

    MPI_Finalize();

    /* de-allocate data structures */
    sfree( system, "system" );
    sfree( control, "control" );
    sfree( data, "data" );
    sfree( workspace, "workspace" );
    sfree( lists, "lists" );
    sfree( out_control, "out_control" );
    sfree( mpi_data, "mpi_data" );

#if defined(TEST_ENERGY) || defined(TEST_FORCES)
//  Integrate_Results(control);
#endif

#if defined(DEBUG)
    fprintf( stderr, "p%d has reached the END\n", system->my_rank );
#endif

    return 0;
}
