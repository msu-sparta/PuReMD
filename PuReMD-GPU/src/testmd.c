/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
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

#include "mytypes.h"

#include "analyze.h"
#include "box.h"
#include "forces.h"
#include "grid.h"
#include "init_md.h"
#include "integrate.h"
#include "neighbors.h"
#include "param.h"
#include "pdb_tools.h"
#include "print_utils.h"
#include "reset_utils.h"
#include "restart.h"
#include "system_props.h"
#include "traj.h"
#include "vector.h"

#include "cuda_environment.h"
#include "cuda_forces.h"
#include "cuda_init_md.h"
#include "cuda_neighbors.h"
#include "cuda_post_evolve.h"
#include "cuda_reset_utils.h"
#include "cuda_system_props.h"

#ifdef __BUILD_DEBUG__
  #include "validation.h"
#endif


interaction_function Interaction_Functions[NO_OF_INTERACTIONS];
lookup_table Exp, Sqrt, Cube_Root, Four_Third_Root, Cos, Sin, ACos;
print_interaction Print_Interactions[NO_OF_INTERACTIONS];

LR_lookup_table *LR;
LR_lookup_table *d_LR;

list *dev_lists;
static_storage *dev_workspace;
reax_timing d_timing;

//TODO
real *testdata;

//Scratch area
void *scratch;
int BLOCKS, BLOCKS_POW_2, BLOCK_SIZE;
int MATVEC_BLOCKS;


void Post_Evolve( reax_system* system, control_params* control, 
        simulation_data* data, static_storage* workspace, 
        list** lists, output_controls *out_control )
{
    int i;
    rvec diff, cross;

    /* if velocity dependent force then
       {
       Generate_Neighbor_Lists( &system, &control, &lists );
       QEq(system, control, workspace, lists[FAR_NBRS]);
       Introduce compute_force here if we are using velocity dependent forces
       Compute_Forces(system,control,data,workspace,lists);
       } */

    /* compute kinetic energy of the system */
    Compute_Kinetic_Energy( system, data );

    /* remove rotational and translational velocity of the center of mass */
    if( control->ensemble != NVE && 
            control->remove_CoM_vel && 
            data->step && data->step % control->remove_CoM_vel == 0 )
    {

        /* compute velocity of the center of mass */
        Compute_Center_of_Mass( system, data, out_control->prs );

        for( i = 0; i < system->N; i++ )
        {
            // remove translational
            rvec_ScaledAdd( system->atoms[i].v, -1., data->vcm ); 

            // remove rotational
            rvec_ScaledSum( diff, 1., system->atoms[i].x, -1., data->xcm );
            rvec_Cross( cross, data->avcm, diff );
            rvec_ScaledAdd( system->atoms[i].v, -1., cross );
        }
    }
}


void Read_System( char *geof, char *ff, char *ctrlf, 
        reax_system *system, control_params *control, 
        simulation_data *data, static_storage *workspace, 
        output_controls *out_control )
{
    FILE *ffield, *ctrl;

    ffield = fopen( ff, "r" );
    ctrl = fopen( ctrlf, "r" );

    /* ffield file */
    Read_Force_Field( ffield, &(system->reaxprm) );

    /* control file */
    Read_Control_File( ctrl, system, control, out_control );

    /* geo file */
    if( control->geo_format == XYZ )
    {
        fprintf( stderr, "xyz input is not implemented yet\n" );
        exit( 1 );
    }
    else if( control->geo_format == PDB ) 
    {
        Read_PDB( geof, system, control, data, workspace );
    }
    else if( control->geo_format == BGF ) 
    {
        Read_BGF( geof, system, control, data, workspace );
    }
    else if( control->geo_format == ASCII_RESTART )
    {
        Read_ASCII_Restart( geof, system, control, data, workspace );
        control->restart = 1;
    }
    else if( control->geo_format == BINARY_RESTART ) {
        Read_Binary_Restart( geof, system, control, data, workspace );
        control->restart = 1;
    }
    else
    {
        fprintf( stderr, "unknown geo file format. terminating!\n" );
        exit( 1 );
    }  

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "input files have been read...\n" );
    Print_Box_Information( &(system->box), stderr );
#endif
}


void Init_Data_Structures( simulation_data *data )
{
    //data->step = 0;
    //data->prev_steps = 0;
    //data->time = 0;

    memset( data, 0, SIMULATION_DATA_SIZE );
}


int main( int argc, char* argv[] )
{
    reax_system system;
    control_params control;
    simulation_data data;
    static_storage workspace;
    list *lists;
    output_controls out_control;
    evolve_function Evolve;
    evolve_function Cuda_Evolve;
    int steps;

    real t_start, t_elapsed;
    real *results = NULL;

    lists = (list*) malloc( sizeof(list) * LIST_N );

    Setup_Cuda_Environment( 0, 1, 1 );

    dev_lists = (list *) malloc (sizeof (list) * LIST_N );
    dev_workspace = (static_storage *) malloc (STORAGE_SIZE);

    //init the nbrs estimate
    dev_workspace->realloc.estimate_nbrs = -1;

    //Cleanup before usage.
    Init_Data_Structures( &data );
    system.init_thblist = FALSE;

    Read_System( argv[1], argv[2], argv[3], &system, &control, 
            &data, &workspace, &out_control );

#ifdef __CUDA_MEM__
    print_device_mem_usage( );
#endif

#ifdef __BUILD_DEBUG__
    Initialize( &system, &control, &data, &workspace, &lists, 
            &out_control, &Evolve );
#endif

    t_start = Get_Time( );
    Cuda_Initialize( &system, &control, &data, &workspace, &lists, 
            &out_control, &Cuda_Evolve );
    t_elapsed = Get_Timing_Info( t_start );

#ifdef __DEBUG_CUDA__
    fprintf( stderr, " Cuda Initialize timing ---> %f \n", t_elapsed );
#endif

#ifdef __CUDA_MEM__
    print_device_mem_usage( );
#endif

#ifdef __BUILD_DEBUG__
    Reset( &system, &control, &data, &workspace, &lists );
#endif

    Cuda_Reset( &system, &control, &data, &workspace, &lists );

#ifdef __BUILD_DEBUG__
    Generate_Neighbor_Lists( &system, &control, &data, &workspace, 
            &lists, &out_control );
#endif

    Cuda_Generate_Neighbor_Lists( &system, &workspace, &control, FALSE );

#ifdef __BUILD_DEBUG__
    Compute_Forces(&system, &control, &data, &workspace, &lists, &out_control);
#endif

    Cuda_Compute_Forces(&system, &control, &data, &workspace, &lists, &out_control);

#ifdef __BUILD_DEBUG__
    Compute_Kinetic_Energy( &system, &data );
#endif

    Cuda_Compute_Kinetic_Energy (&system, &data);

#ifndef __BUILD_DEBUG__
    Cuda_Setup_Output( &system, &data );
    Output_Results(&system, &control, &data, &workspace, &lists, &out_control);
#endif

#ifdef __BUILD_DEBUG__
    if( !validate_device (&system, &data, &workspace, &lists) )
    {
        fprintf (stderr, " Results does not match between Device and host @ step --> %d \n", data.step);
        exit (1);
    }
#endif

#ifdef __DEBUG_CUDA__
    fprintf (stderr, "step -> %d <- done. \n", data.step);
#endif

    ++data.step;

    for( ; data.step <= control.nsteps; data.step++ )
    {
        Cuda_Setup_Evolve( &system, &control, &data, &workspace, &lists, &out_control );

        //fprintf (stderr, "Synched data .... \n");
        if( control.T_mode )
        {
            Temperature_Control( &control, &data, &out_control );
            Cuda_Sync_Temp( &control );
        }
        //fprintf (stderr, "Temp. Control done ... \n");

#ifdef __BUILD_DEBUG__
        Evolve( &system, &control, &data, &workspace, &lists, &out_control );
#endif

        Cuda_Evolve( &system, &control, &data, &workspace, &lists, &out_control );
        //fprintf (stderr, "Evolve done \n");

#ifdef __BUILD_DEBUG__
        Post_Evolve( &system, &control, &data, &workspace, &lists, &out_control );
#endif

        Cuda_Post_Evolve( &system, &control, &data, &workspace, &lists, &out_control );
        //fprintf (stderr, "Post Evolve done \n");

#ifndef __BUILD_DEBUG__
        Cuda_Setup_Output( &system, &data );
        Output_Results( &system, &control, &data, &workspace, &lists, &out_control );

        //Analysis( &system, &control, &data, &workspace, &lists, &out_control );
        steps = data.step - data.prev_steps;
        if( steps && out_control.restart_freq && 
                steps % out_control.restart_freq == 0 )
        {
            Write_Restart( &system, &control, &data, &workspace, &out_control );
        }
#endif

#ifdef __BUILD_DEBUG__
        if (!validate_device (&system, &data, &workspace, &lists) )
        {
            fprintf (stderr, " Results does not match between Device and host @ step --> %d \n", data.step);
            exit (1);
        }

        fprintf (stderr, "step -> %d <- done. \n", data.step);
#endif
    }

    if( out_control.write_steps > 0 ) { 
        fclose( out_control.trj );
        //Write_PDB( &system, &control, &data, &workspace,
        //     &(lists[BONDS]), &out_control );
    }

    data.timing.end = Get_Time( );
    data.timing.elapsed = Get_Timing_Info( data.timing.start );
    fprintf( out_control.log, "total: %.2f secs\n", data.timing.elapsed );

    Cleanup_Cuda_Environment( );

    return 0;
}
