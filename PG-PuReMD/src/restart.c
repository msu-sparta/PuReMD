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

#include "restart.h"

#include "allocate.h"
#include "box.h"
#include "tool_box.h"
#include "vector.h"


void Write_Binary_Restart_File( reax_system *system, control_params *control,
        simulation_data *data, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    int i, me, np, cnt, top;
    char fname[MAX_STR];
    FILE *fres;
    restart_header res_header;
    restart_atom *buffer;
    reax_atom *p_atom;
    MPI_Status status;

    fres = NULL;
    me = system->my_rank;
    np = control->nprocs;

    if ( me == MASTER_NODE )
    {
        /* master handles the restart file */
        sprintf( fname, "%s.res%d", control->sim_name, data->step );
        fres = sfopen( fname, "wb", "Write_Binary_Restart_File::fres" );

        /* master can write the header by itself */
        res_header.step = data->step;
        res_header.bigN = system->bigN;
        res_header.T = data->therm.T;
        res_header.xi = data->therm.xi;
        res_header.v_xi = data->therm.v_xi;
        res_header.v_xi_old = data->therm.v_xi_old;
        res_header.G_xi = data->therm.G_xi;
        rtensor_Copy( res_header.box, system->big_box.box );
        fwrite( &res_header, sizeof(restart_header), 1, fres );

        /* master needs to allocate space for all atoms */
        buffer = scalloc( system->bigN, sizeof(restart_atom),
                "Write_Binary_Restart_File::buffer" );
    }
    else
    {
        buffer = scalloc( system->n, sizeof(restart_atom),
                "Write_Binary_Restart_File::buffer" );
    }

    /* fill in the buffers */
    for ( i = 0 ; i < system->n; ++i )
    {
        p_atom = &(system->my_atoms[i]);
        buffer[i].orig_id = p_atom->orig_id;
        buffer[i].type = p_atom->type;
        strncpy( buffer[i].name, p_atom->name, 8 );
        rvec_Copy( buffer[i].x, p_atom->x );
        rvec_Copy( buffer[i].v, p_atom->v );
    }

    /* gather the buffers at the master node */
    if ( me != MASTER_NODE )
    {
        MPI_Send( buffer, system->n, mpi_data->restart_atom_type, MASTER_NODE,
                  np * RESTART_ATOMS + me, mpi_data->world );
    }
    else
    {
        top = system->n;
        for ( i = 0; i < np; ++i )
        {
            if ( i != MASTER_NODE )
            {
                MPI_Recv( buffer + top, system->bigN - top, mpi_data->restart_atom_type,
                          i, np * RESTART_ATOMS + i, mpi_data->world, &status );
                MPI_Get_count( &status, mpi_data->restart_atom_type, &cnt );
                top += cnt;
            }
        }
    }

    /* master node dumps out the restart file */
    if ( me == MASTER_NODE )
    {
        fwrite( buffer, system->bigN, sizeof(restart_atom), fres );
        sfclose( fres, "Write_Binary_Restart_File::fres" );
    }

    sfree( buffer, "Write_Binary_Restart_File::buffer" );
}


void Write_Restart_File( reax_system *system, control_params *control,
        simulation_data *data, output_controls *out_control,
        mpi_datatypes *mpi_data )
{
    int i, me, np, buffer_len, buffer_req, cnt;
    char fname[MAX_STR];
    FILE *fres;
    char *line;
    char *buffer;
    reax_atom *p_atom;
    MPI_Status status;

    fres = NULL;
    line = smalloc(sizeof(char) * RESTART_LINE_LEN, "restart:line");
    me = system->my_rank;
    np = control->nprocs;

    if ( me == MASTER_NODE )
    {
        sprintf( fname, "%s.res%d", control->sim_name, data->step );
        fres = sfopen( fname, "w", "Write_Restart_File::fres" );

        /* write the header - only master writes it */
        fprintf( fres, RESTART_HEADER,
                 data->step, system->bigN, data->therm.T, data->therm.xi,
                 data->therm.v_xi, data->therm.v_xi_old, data->therm.G_xi,
                 system->big_box.box[0][0], system->big_box.box[0][1],
                 system->big_box.box[0][2],
                 system->big_box.box[1][0], system->big_box.box[1][1],
                 system->big_box.box[1][2],
                 system->big_box.box[2][0], system->big_box.box[2][1],
                 system->big_box.box[2][2]);
        fflush(fres);

        buffer_req = system->bigN * RESTART_LINE_LEN + 1;
    }
    else
    {
        buffer_req = system->n * RESTART_LINE_LEN + 1;
    }

    buffer = smalloc( sizeof(char) * buffer_req, "Write_Restart_File::buffer" );
    line[0] = 0;
    buffer[0] = 0;

    /* fill in the buffers */
    for ( i = 0 ; i < system->n; ++i )
    {
        p_atom = &system->my_atoms[i];

        sprintf( line, RESTART_LINE,
                 p_atom->orig_id, p_atom->type, p_atom->name,
                 p_atom->x[0], p_atom->x[1], p_atom->x[2],
                 p_atom->v[0], p_atom->v[1], p_atom->v[2] );

        strncpy( buffer + i * RESTART_LINE_LEN, line, RESTART_LINE_LEN );
        //was LINE_LEN +1
    }

    /* gather the buffers at the master node */
    if ( me != MASTER_NODE )
    {
        MPI_Send( buffer, buffer_req - 1, MPI_CHAR, MASTER_NODE,
                np * RESTART_LINE_LEN + me, mpi_data->world );
    }
    else
    {
        buffer_len = system->n * RESTART_LINE_LEN;
        for ( i = 0; i < np; ++i )
        {
            if ( i != MASTER_NODE )
            {
                MPI_Recv( buffer + buffer_len, buffer_req - buffer_len,
                        MPI_CHAR, i, np * RESTART_LINE_LEN + i, mpi_data->world, &status );
                MPI_Get_count( &status, MPI_CHAR, &cnt );
                buffer_len += cnt;
            }
        }
        buffer[buffer_len] = 0;
    }

    /* master node dumps out the restart file */
    if ( me == MASTER_NODE )
    {
        fprintf( fres, "%s", buffer );
        sfclose( fres, "Write_Restart_File::fres" );
    }
    sfree( buffer, "Write_Restart_File::buffer" );
    sfree( line, "Write_Restart_File::line" );
}


void Count_Binary_Restart_Atoms( FILE *fres, reax_system *system )
{
    int i;
    restart_atom res_atom;

    system->n = system->N = 0;
    for ( i = 0; i < system->bigN; i++ )
    {
        fread( &res_atom, sizeof(restart_atom), 1, fres );

        /* if the point is inside my_box, add it to my lists */
        Fit_to_Periodic_Box( &(system->big_box), &(res_atom.x) );
        if ( is_Inside_Box(&(system->my_box), res_atom.x) )
        {
            ++system->n;
        }
    }
    system->N = system->n;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "restart: p%d@count atoms:\n", system->my_rank );
    fprintf( stderr, "p%d: bigN = %d\n", system->my_rank, system->bigN );
    fprintf( stderr, "p%d: n = %d\n", system->my_rank, system->n );
    fprintf( stderr, "p%d: N = %d\n\n", system->my_rank, system->N );
#endif
}


void Read_Binary_Restart_File( const char * const res_file, reax_system *system,
        control_params *control, simulation_data *data,
        storage *workspace, mpi_datatypes *mpi_data )
{
    int i, top;
    FILE *fres;
    restart_header res_header;
    restart_atom res_atom;
    reax_atom *p_atom;

    fres = sfopen( res_file, "rb", "Read_Binary_Restart_File::fres" );

    /* first read the header lines */
    fread( &res_header, sizeof(restart_header), 1, fres );
    data->prev_steps = res_header.step;
    system->bigN = res_header.bigN;
    data->therm.T = res_header.T;
    data->therm.xi = res_header.xi;
    data->therm.v_xi = res_header.v_xi;
    data->therm.v_xi_old = res_header.v_xi_old;
    data->therm.G_xi = res_header.G_xi;

    Init_Box( res_header.box, &system->big_box );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "restart info: %d  %d %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n %15.5f%15.5f%15.5f\n %15.5f%15.5f%15.5f\n %15.5f%15.5f%15.5f\n",
             data->prev_steps, system->bigN, data->therm.T, data->therm.xi,
             data->therm.v_xi, data->therm.v_xi_old, data->therm.G_xi,
             system->big_box.box[0][0], system->big_box.box[0][1],
             system->big_box.box[0][2],
             system->big_box.box[1][0], system->big_box.box[1][1],
             system->big_box.box[1][2],
             system->big_box.box[2][0], system->big_box.box[2][1],
             system->big_box.box[2][2]);
#endif

    /* set up box, etc. */
    Setup_Environment( system, control, mpi_data );
    Count_Binary_Restart_Atoms( fres, system );
    PreAllocate_Space( system, control, workspace );

    /* go back to the start of restart file */
    rewind( fres );
    fread( &res_header, sizeof(restart_header), 1, fres );

    /* process atoms */
    top = 0;
    for ( i = 0; i < system->bigN; ++i )
    {
        fread( &res_atom, sizeof(restart_atom), 1, fres );

        /* if the point is inside my_box, add it to my lists */
        Fit_to_Periodic_Box( &(system->big_box), &(res_atom.x) );
        
        if ( is_Inside_Box(&system->my_box, res_atom.x) )
        {
            /* store orig_id, type, name and coord info of the new atom */
            p_atom = &system->my_atoms[top];
            p_atom->orig_id = res_atom.orig_id;
            p_atom->type = res_atom.type;

            strcpy( p_atom->name, res_atom.name );

            rvec_Copy( p_atom->x, res_atom.x );
            rvec_Copy( p_atom->v, res_atom.v );

            top++;
        }
    }

    sfclose( fres, "Read_Binary_Restart_File::fres" );

    data->step = data->prev_steps;
    // nsteps is updated based on the number of steps in the previous run
    control->nsteps += data->prev_steps;
}


void Count_Restart_Atoms( FILE *fres, reax_system *system )
{
    int i = 0;
    /*temporary variable storage*/
    int orig_id_temp, type_temp;
    char name_temp[8];
    rvec x_temp, v_temp;

    system->n = 0;
    system->N = 0;

    for ( i = 0; i < system->bigN; i++)
    {
        fscanf( fres, READ_RESTART_LINE,
                &orig_id_temp, &type_temp, name_temp,
                &x_temp[0], &x_temp[1], &x_temp[2],
                &v_temp[0], &v_temp[1], &v_temp[2] );

        Fit_to_Periodic_Box( &system->big_box, &x_temp );

        /* if the point is inside my_box, add it to my lists */
        if ( is_Inside_Box(&(system->my_box), x_temp) )
        {
            ++system->n;
        }
    }
    system->N = system->n;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "restart: p%d@count atoms:\n", system->my_rank );
    fprintf( stderr, "p%d: bigN = %d\n", system->my_rank, system->bigN );
    fprintf( stderr, "p%d: n = %d\n", system->my_rank, system->n );
    fprintf( stderr, "p%d: N = %d\n\n", system->my_rank, system->N );
#endif
}


void Read_Restart_File( const char * const res_file, reax_system *system,
        control_params *control, simulation_data *data,
        storage *workspace, mpi_datatypes *mpi_data )
{
    int i, c, top;
    FILE *fres;
    char *s, **tmp;
    int orig_id_temp, type_temp;
    char name_temp[8];
    rvec x_temp, v_temp;
    rtensor box;

    fres = sfopen( res_file, "r", "Read_Restart_File::fres" );

    s = smalloc( sizeof(char) * MAX_LINE, "Read_Restart_File::s" );
    tmp = smalloc( sizeof(char*) * MAX_TOKENS, "Read_Restart_File::tmp" );
    for (i = 0; i < MAX_TOKENS; i++)
    {
        tmp[i] = smalloc( sizeof(char) * MAX_LINE, "Read_Restart_File::tmp[i]" );
    }

    //read first header lines
    fgets( s, MAX_LINE, fres );
    c = Tokenize( s, &tmp );

    if ( c != 7 )
    {
        fprintf( stderr, "[ERROR] invalid format in restart file! terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }

    data->prev_steps = atoi(tmp[0]);
    system->bigN = atoi(tmp[1]);
    data->therm.T = atof(tmp[2]);
    data->therm.xi = atof(tmp[3]);
    data->therm.v_xi = atof(tmp[4]);
    data->therm.v_xi_old = atof(tmp[5]);
    data->therm.G_xi = atof(tmp[6]);

    //read box lines
    fgets( s, MAX_LINE, fres );
    c = Tokenize( s, &tmp );

    if ( c != 3 )
    {
        fprintf( stderr, "[ERROR] invalid format in restart file! terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }

    box[0][0] = atof(tmp[0]);
    box[0][1] = atof(tmp[1]);
    box[0][2] = atof(tmp[2]);
    fgets( s, MAX_LINE, fres );
    c = Tokenize( s, &tmp );
    if ( c != 3 )
    {
        fprintf( stderr, "[ERROR] invalid format in restart file! terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }
    box[1][0] = atof(tmp[0]);
    box[1][1] = atof(tmp[1]);
    box[1][2] = atof(tmp[2]);
    fgets( s, MAX_LINE, fres );
    c = Tokenize( s, &tmp );

    if ( c != 3 )
    {
        fprintf( stderr, "[ERROR] invalid format in restart file! terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
    }

    box[2][0] = atof(tmp[0]);
    box[2][1] = atof(tmp[1]);
    box[2][2] = atof(tmp[2]);
    Init_Box( box, &system->big_box );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "restart info: %d  %d %8.3f  %8.3f  %8.3f  %8.3f  %8.3f\n %15.5f%15.5f%15.5f\n %15.5f%15.5f%15.5f\n %15.5f%15.5f%15.5f\n",
             data->prev_steps, system->bigN, data->therm.T, data->therm.xi,
             data->therm.v_xi, data->therm.v_xi_old, data->therm.G_xi,
             system->big_box.box[0][0], system->big_box.box[0][1],
             system->big_box.box[0][2],
             system->big_box.box[1][0], system->big_box.box[1][1],
             system->big_box.box[1][2],
             system->big_box.box[2][0], system->big_box.box[2][1],
             system->big_box.box[2][2]);
#endif

    /* set up the simulation envirionment */
    Setup_Environment( system, control, mpi_data );
    Count_Restart_Atoms(fres, system);
    PreAllocate_Space( system, control, workspace );

    /* go back to the start of file to read actual atom info */
    rewind( fres );
    for (i = 0; i < 4; i++)
    {
        fgets( s, MAX_LINE, fres );
    }

    /*process atoms*/
    top = 0;
    for ( i = 0; i < system->bigN; ++i )
    {
        fgets( s, MAX_LINE, fres );
        c = Tokenize( s, &tmp );

        if ( c != 9 )
        {
            fprintf( stderr, "[ERROR] invalid format in restart file! terminating...\n" );
            MPI_Abort( MPI_COMM_WORLD, INVALID_INPUT );
        }

        orig_id_temp = atoi(tmp[0]);
        type_temp = atoi(tmp[1]);
        strncpy(name_temp, tmp[2], 8);
        x_temp[0] = atof(tmp[3]);
        x_temp[1] = atof(tmp[4]);
        x_temp[2] = atof(tmp[5]);
        v_temp[0] = atof(tmp[6]);
        v_temp[1] = atof(tmp[7]);
        v_temp[2] = atof(tmp[8]);

        Fit_to_Periodic_Box( &system->big_box, &x_temp );

        if ( is_Inside_Box( &system->my_box, x_temp ) )
        {
            /* store orig_id, type, name and coord info of the new atom */
            system->my_atoms[top].orig_id = orig_id_temp;
            system->my_atoms[top].type = type_temp;
            strcpy( system->my_atoms[top].name, name_temp );
            rvec_Copy( system->my_atoms[top].x, x_temp );
            rvec_Copy( system->my_atoms[top].v, v_temp );
            top++;
        }
    }
    sfclose( fres, "Read_Restart_File::fres" );

    /* free memory allocations at the top */
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        sfree( tmp[i], "Read_Restart_File::tmp[i]" );
    }
    sfree( tmp, "Read_Restart_File::tmp" );
    sfree( s, "Read_Restart_File::s" );

    data->step = data->prev_steps;
    // nsteps is updated based on the number of steps in the previous run
    control->nsteps += data->prev_steps;
}
