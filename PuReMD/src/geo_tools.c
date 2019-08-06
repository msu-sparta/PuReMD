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

#include "geo_tools.h"
#include "allocate.h"
#include "box.h"
#include "tool_box.h"
#include "vector.h"


/********************* geo format routines ******************/
void Count_Geo_Atoms( FILE *geo, reax_system *system )
{
    int i, serial;
    rvec x;
    char element[3], name[9], line[MAX_LINE + 1];

    /* total number of atoms */
    fscanf( geo, " %d", &(system->bigN) );

    /* count my own atoms */
    system->n = 0;
    for ( i = 0; i < system->bigN; ++i )
    {
        fscanf( geo, CUSTOM_ATOM_FORMAT,
                &serial, element, name, &x[0], &x[1], &x[2] );
        Fit_to_Periodic_Box( &(system->big_box), &x );

        /* if the point is inside my_box, add it to my lists */
        if ( is_Inside_Box(&(system->my_box), x) )
            ++system->n;
    }

    system->N = system->n;

    fseek( geo, 0, SEEK_SET ); // set the pointer to the beginning of the file
    fgets( line, MAX_LINE, geo );
    fgets( line, MAX_LINE, geo );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d@count atoms:\n", system->my_rank );
    fprintf( stderr, "p%d: bigN = %d\n", system->my_rank, system->bigN );
    fprintf( stderr, "p%d: n = %d\n", system->my_rank, system->n );
    fprintf( stderr, "p%d: N = %d\n\n", system->my_rank, system->N );
#endif
}


char Read_Geo( char* geo_file, reax_system* system, control_params *control,
               simulation_data *data, storage *workspace,
               mpi_datatypes *mpi_data )
{

    FILE *geo;
    char descriptor[9];
    int i, serial, top;
    real box_x, box_y, box_z, alpha, beta, gamma;
    rvec x;
    char element[3], name[9];
    reax_atom *atom;
    MPI_Comm comm;

    comm = MPI_COMM_WORLD;

    /* open the geometry file */
    geo = sfopen( geo_file, "r", "Read_Geo::geo" );

    /* read box information */
    fscanf( geo, CUSTOM_BOXGEO_FORMAT,
            descriptor, &box_x, &box_y, &box_z, &alpha, &beta, &gamma );
    /* initialize the box */
    Setup_Big_Box( box_x, box_y, box_z, alpha, beta, gamma, &(system->big_box) );
    /* initialize the simulation environment */
    Setup_Environment( system, control, mpi_data );

    /* count my atoms & allocate storage */
    Count_Geo_Atoms( geo, system );
    if ( PreAllocate_Space( system, control, workspace, comm ) == FAILURE )
    {
        fprintf( stderr, "PreAllocate_Space: not enough memory!" );
        fprintf( stderr, "terminating...\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }

    /* read in my atom info */
    top = 0;
    for ( i = 0; i < system->bigN; ++i )
    {
        fscanf( geo, CUSTOM_ATOM_FORMAT,
                &serial, element, name, &x[0], &x[1], &x[2] );
        Fit_to_Periodic_Box( &(system->big_box), &x );
#if defined(DEBUG)
        fprintf( stderr, "atom%d: %s %s %f %f %f\n",
                 serial, element, name, x[0], x[1], x[2] );
#endif

        /* if the point is inside my_box, add it to my list */
        if ( is_Inside_Box(&(system->my_box), x) )
        {
            atom = &(system->my_atoms[top]);
            atom->orig_id = serial;
            atom->type = Get_Atom_Type( &(system->reax_param), element, comm );
            strcpy( atom->name, name );
            rvec_Copy( atom->x, x );
            rvec_MakeZero( atom->v );
            rvec_MakeZero( atom->f );
            rvec_MakeZero( atom->f_old );
            atom->q = 0;
            rvec_MakeZero( atom->s );
            rvec_MakeZero( atom->t );

            atom->imprt_id = -1;
            atom->Hindex = -1;
            atom->num_bonds = 0;
            atom->num_hbonds = 0;

            top++;
        }
    }

    sfclose( geo, "Read_Geo::geo" );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: finished reading the geo file\n", system->my_rank );
    //Print_My_Atoms( system );
#endif

    return SUCCESS;
}


int Read_Box_Info( reax_system *system, FILE *geo, int geo_format )
{
    char *cryst;
    char  line[MAX_LINE + 1];
    char  descriptor[9];
    char  s_a[12], s_b[12], s_c[12], s_alpha[12], s_beta[12], s_gamma[12];
    char  s_group[12], s_zValue[12];

    /* initialize variables */
    fseek( geo, 0, SEEK_SET ); // set the pointer to the beginning of the file

    switch ( geo_format )
    {
        case PDB:
            cryst = "CRYST1";
            break;
        default:
            cryst = "BOX";
    }

    /* locate the cryst line in the geo file, read it and
       initialize the big box */
    while ( fgets( line, MAX_LINE, geo ) )
    {
        if ( strncmp( line, cryst, 6 ) == 0 )
        {
            if ( geo_format == PDB )
                sscanf( line, PDB_CRYST1_FORMAT,
                        &descriptor[0],
                        &s_a[0], &s_b[0], &s_c[0],
                        &s_alpha[0], &s_beta[0], &s_gamma[0],
                        &s_group[0], &s_zValue[0] );

            /* compute full volume tensor from the angles */
            Setup_Big_Box( atof(s_a),  atof(s_b), atof(s_c),
                           atof(s_alpha), atof(s_beta), atof(s_gamma),
                           &(system->big_box) );
            return SUCCESS;
        }
    }
    if ( ferror( geo ) )
    {
        return FAILURE;
    }

    return FAILURE;
}


void Count_PDB_Atoms( FILE *geo, reax_system *system )
{
    char *endptr = NULL;
    char line[MAX_LINE + 1];
    char s_x[9], s_y[9], s_z[9];
    rvec x;

    /* initialize variables */
    fseek( geo, 0, SEEK_SET ); /* set the pointer to the beginning of the file */
    system->bigN = system->n = system->N = 0;

    /* increment number of atoms for each line denoting an atom desc */
    while ( fgets( line, MAX_LINE, geo ) )
    {
        if ( strncmp( line, "ATOM", 4 ) == 0 ||
                strncmp( line, "HETATM", 6 ) == 0 )
        {
            system->bigN++;

            strncpy( s_x, line + 30, 8 );
            s_x[8] = 0;
            strncpy( s_y, line + 38, 8 );
            s_y[8] = 0;
            strncpy( s_z, line + 46, 8 );
            s_z[8] = 0;
            Make_Point( strtod( s_x, &endptr ), strtod( s_y, &endptr ),
                        strtod( s_z, &endptr ), &x );
            Fit_to_Periodic_Box( &(system->big_box), &x );

            /* if the point is inside my_box, add it to my lists */
            if ( is_Inside_Box(&(system->my_box), x) )
                ++system->n;
            //if( is_Inside_Box(&(system->my_ext_box), x) )
            //  ++system->N;
        }
    }

    system->N = system->n;

#if defined(DEBUG)
    fprintf( stderr, "p%d@count atoms:\n", system->my_rank );
    fprintf( stderr, "p%d: bigN = %d\n", system->my_rank, system->bigN );
    fprintf( stderr, "p%d: n = %d\n", system->my_rank, system->n );
    fprintf( stderr, "p%d: N = %d\n\n", system->my_rank, system->N );
#endif
}


char Read_PDB( char* pdb_file, reax_system* system, control_params *control,
               simulation_data *data, storage *workspace,
               mpi_datatypes *mpi_data )
{

    FILE  *pdb;
    char **tmp;
    char  *s, *s1;
    char   descriptor[9], serial[9];
    char   atom_name[9], res_name[9], res_seq[9];
    char   s_x[9], s_y[9], s_z[9];
    char   occupancy[9], temp_factor[9];
    char   seg_id[9], element[9], charge[9];
    char   alt_loc, chain_id, icode;
    char  *endptr = NULL;
    int    i, c, c1, pdb_serial, top;
    rvec   x;
    reax_atom *atom;
    MPI_Comm comm;

    comm = MPI_COMM_WORLD;

    /* open pdb file */
    pdb = sfopen( pdb_file, "r", "Read_PDB::pdb" );

    /* allocate memory for tokenizing pdb lines */
    if ( Allocate_Tokenizer_Space( &s, &s1, &tmp ) == FAILURE )
    {
        fprintf( stderr, "Allocate_Tokenizer_Space: not enough memory!" );
        fprintf( stderr, "terminating...\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }

    /* read box information */
    if ( Read_Box_Info( system, pdb, PDB ) == FAILURE )
    {
        fprintf( stderr, "Read_Box_Info: no CRYST line in the pdb file!" );
        fprintf( stderr, "terminating...\n" );
        MPI_Abort( comm, INVALID_GEO );
    }

    Setup_Environment( system, control, mpi_data );
    Count_PDB_Atoms( pdb, system );
    if ( PreAllocate_Space( system, control, workspace, comm ) == FAILURE )
    {
        fprintf( stderr, "PreAllocate_Space: not enough memory!" );
        fprintf( stderr, "terminating...\n" );
        MPI_Abort( comm, INSUFFICIENT_MEMORY );
    }

    /* start reading and processing the pdb file */
#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: starting to read the pdb file\n", system->my_rank );
#endif
    fseek( pdb, 0, SEEK_SET );
    c  = 0;
    c1 = 0;
    top = 0;
    s[0] = 0;

    while ( fgets( s, MAX_LINE, pdb ) )
    {
        /* read new line and tokenize it */
        strncpy( s1, s, MAX_LINE - 1 );
        c1 = Tokenize( s, &tmp );

        /* process new line */
        if ( strncmp(tmp[0], "ATOM", 4) == 0 || strncmp(tmp[0], "HETATM", 6) == 0 )
        {
            if ( strncmp(tmp[0], "ATOM", 4) == 0 )
            {
                strncpy( &descriptor[0], s1, 6 );
                descriptor[6] = 0;
                strncpy( &serial[0], s1 + 6, 5 );
                serial[5] = 0;
                strncpy( &atom_name[0], s1 + 12, 4 );
                atom_name[4] = 0;
                //strncpy( &serial[0], s1+6, 7 );       serial[7] = 0;
                //strncpy( &atom_name[0], s1+13, 3 );   atom_name[3] = 0;
                alt_loc = s1[16];
                strncpy( &res_name[0], s1 + 17, 3 );
                res_name[3] = 0;
                chain_id = s1[21];
                strncpy( &res_seq[0], s1 + 22, 4 );
                res_seq[4] = 0;
                icode = s1[26];
                strncpy( &s_x[0], s1 + 30, 8 );
                s_x[8] = 0;
                strncpy( &s_y[0], s1 + 38, 8 );
                s_y[8] = 0;
                strncpy( &s_z[0], s1 + 46, 8 );
                s_z[8] = 0;
                strncpy( &occupancy[0], s1 + 54, 6 );
                occupancy[6] = 0;
                strncpy( &temp_factor[0], s1 + 60, 6 );
                temp_factor[6] = 0;
                strncpy( &seg_id[0], s1 + 72, 4 );
                seg_id[4] = 0;
                strncpy( &element[0], s1 + 76, 2 );
                element[2] = 0;
                strncpy( &charge[0], s1 + 78, 2 );
                charge[2] = 0;
            }
            else if (strncmp(tmp[0], "HETATM", 6) == 0)
            {
                strncpy( &descriptor[0], s1, 6 );
                descriptor[6] = 0;
                strncpy( &serial[0], s1 + 6, 5 );
                serial[5] = 0;
                strncpy( &atom_name[0], s1 + 12, 4 );
                atom_name[4] = 0;
                //strncpy( &serial[0], s1+6, 7 );       serial[7] = 0;
                //strncpy( &atom_name[0], s1+13, 3 );   atom_name[3] = 0;
                alt_loc = s1[16];
                strncpy( &res_name[0], s1 + 17, 3 );
                res_name[3] = 0;
                chain_id = s1[21];
                strncpy( &res_seq[0], s1 + 22, 4 );
                res_seq[4] = 0;
                icode = s1[26];
                strncpy( &s_x[0], s1 + 30, 8 );
                s_x[8] = 0;
                strncpy( &s_y[0], s1 + 38, 8 );
                s_y[8] = 0;
                strncpy( &s_z[0], s1 + 46, 8 );
                s_z[8] = 0;
                strncpy( &occupancy[0], s1 + 54, 6 );
                occupancy[6] = 0;
                strncpy( &temp_factor[0], s1 + 60, 6 );
                temp_factor[6] = 0;
                //strncpy( &seg_id[0], s1+72, 4 );      seg_id[4] = 0;
                strncpy( &element[0], s1 + 76, 2 );
                element[2] = 0;
                strncpy( &charge[0], s1 + 78, 2 );
                charge[2] = 0;
            }

            /* if the point is inside my_box, add it to my lists */
            Make_Point( strtod( &s_x[0], &endptr ),
                        strtod( &s_y[0], &endptr ),
                        strtod( &s_z[0], &endptr ), &x );

            Fit_to_Periodic_Box( &(system->big_box), &x );

            if ( is_Inside_Box( &(system->my_box), x ) )
            {
                /* store orig_id, type, name and coord info of the new atom */
                atom = &(system->my_atoms[top]);
                pdb_serial = (int) strtod( &serial[0], &endptr );
                atom->orig_id = pdb_serial;

                Trim_Spaces( element );
                atom->type = Get_Atom_Type( &(system->reax_param), element, comm );
                strcpy( atom->name, atom_name );

                rvec_Copy( atom->x, x );
                rvec_MakeZero( atom->v );
                rvec_MakeZero( atom->f );
                rvec_MakeZero( atom->f_old );
                atom->q = 0;
                rvec_MakeZero( atom->s );
                rvec_MakeZero( atom->t );

                atom->imprt_id = -1;
                atom->Hindex = -1;
                atom->num_bonds = 0;
                atom->num_hbonds = 0;

                top++;
                // fprintf( stderr, "p%d: %6d%2d x:%8.3f%8.3f%8.3f"
                //                  "q:%8.3f occ:%s temp:%s seg:%s elmnt:%s\n",
                //       system->my_rank,
                //       c, system->my_atoms[top].type,
                //       system->my_atoms[top].x[0],
                //       system->my_atoms[top].x[1],
                //       system->my_atoms[top].x[2],
                //       system->my_atoms[top].q, occupancy, temp_factor,
                //       seg_id, element );

                //fprintf( stderr, "atom( %8.3f %8.3f %8.3f ) --> p%d\n",
                // system->my_atoms[top].x[0], system->my_atoms[top].x[1],
                // system->my_atoms[top].x[2], system->my_rank );
            }
            c++;
        }

        /* IMPORTANT: We do not check for the soundness of restrictions here.
           When atom2 is on atom1's restricted list, and there is a restriction
           on atom2, then atom1 has to be on atom2's restricted list, too.
           However, we do not check if this is the case in the input file,
           this is upto the user. */
        else if ( strncmp( tmp[0], "CONECT", 6 ) == 0 )
        {
            if ( control->restrict_bonds )
            {
                /* error check */
                // Check_Input_Range( c1 - 2, 0, MAX_RESTRICT,
                // "CONECT line exceeds max num restrictions allowed.\n" );

                /* read bond restrictions */
                // if( is_Valid_Serial( workspace, pdb_serial = atoi(tmp[1]) ) )
                //   ratom = workspace->map_serials[ pdb_serial ];

                // workspace->restricted[ ratom ] = c1 - 2;
                // for( i = 2; i < c1; ++i )
                //  {
                //    if( is_Valid_Serial(workspace, pdb_serial = atoi(tmp[i])) )
                //        workspace->restricted_list[ ratom ][ i-2 ] =
                //          workspace->map_serials[ pdb_serial ];
                //  }

                // fprintf( stderr, "restriction on %d:", ratom );
                // for( i = 0; i < workspace->restricted[ ratom ]; ++i )
                // fprintf( stderr, "  %d",
                //          workspace->restricted_list[ratom][i] );
                // fprintf( stderr, "\n" );
            }
        }

        /* clear previous input line */
        s[0] = 0;
        for ( i = 0; i < c1; ++i )
            tmp[i][0] = 0;
    }
    if ( ferror( pdb ) )
    {
        return FAILURE;
    }

    sfclose( pdb, "Read_PDB::pdb" );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: finished reading the pdb file\n", system->my_rank );
    //Print_My_Atoms( system );
#endif

    return SUCCESS;
}


/* PDB serials are written without regard to the order, we'll see if this
   cause trouble, if so we'll have to rethink this approach
   Also, we do not write connect lines yet.
*/

char Write_PDB(reax_system* system, reax_list** bonds, simulation_data *data,
               control_params *control, mpi_datatypes *mpi_data,
               output_controls *out_control)
{
    int i, cnt, me, np, buffer_req, buffer_len;
    //int j, connect[4];
    char name[8];
    //real bo;
    real alpha, beta, gamma;
    MPI_Status status;
    reax_atom *p_atom;
    char fname[MAX_STR];
    char *line;
    char *buffer;
    FILE *pdb;
    MPI_Comm comm;

    me = system->my_rank;
    np = control->nprocs;
    comm = mpi_data->world;

    /* Allocation*/
    line = (char*)
           smalloc( sizeof(char) * PDB_ATOM_FORMAT_O_LENGTH, "geo:line", comm );
    if ( me == MASTER_NODE )
        buffer_req = system->bigN * PDB_ATOM_FORMAT_O_LENGTH;
    else
        buffer_req = system->n * PDB_ATOM_FORMAT_O_LENGTH;

    buffer = (char*) smalloc( sizeof(char) * buffer_req, "geo:buffer", comm );

    pdb = NULL;
    line[0] = 0;
    buffer[0] = 0;
    /*open pdb and write header*/
    if (me == MASTER_NODE)
    {
        /* Writing Box information */
        gamma = acos( (system->big_box.box[0][0] * system->big_box.box[1][0] +
                       system->big_box.box[0][1] * system->big_box.box[1][1] +
                       system->big_box.box[0][2] * system->big_box.box[1][2]) /
                      (system->big_box.box_norms[0] * system->big_box.box_norms[1]) );
        beta  = acos( (system->big_box.box[0][0] * system->big_box.box[2][0] +
                       system->big_box.box[0][1] * system->big_box.box[2][1] +
                       system->big_box.box[0][2] * system->big_box.box[2][2]) /
                      (system->big_box.box_norms[0] * system->big_box.box_norms[2]) );
        alpha = acos( (system->big_box.box[2][0] * system->big_box.box[1][0] +
                       system->big_box.box[2][1] * system->big_box.box[1][1] +
                       system->big_box.box[2][2] * system->big_box.box[1][2]) /
                      (system->big_box.box_norms[2] * system->big_box.box_norms[1]) );


        sprintf(fname, "%s-%d.pdb", control->sim_name, data->step);
        pdb = sfopen( fname, "w", "Write_PDB::pdb" );
        /*fprintf( pdb, PDB_CRYST1_FORMAT_O,
                 "CRYST1",
                 system->big_box.box_norms[0], system->big_box.box_norms[1],
                 system->big_box.box_norms[2],
                 RAD2DEG(alpha), RAD2DEG(beta), RAD2DEG(gamma), " ", 0 );
        fprintf( out_control->log, "Box written\n" );
        fflush( out_control->log );*/
    }

    /*write atom lines to buffer*/
    for ( i = 0; i < system->n; i++)
    {
        p_atom = &(system->my_atoms[i]);
        strncpy(name, p_atom->name, 8);
        Trim_Spaces(name);
        /*sprintf( line, PDB_ATOM_FORMAT_O,
                 "ATOM  ", p_atom->orig_id, p_atom->name, ' ', "REX", ' ', 1, ' ',
                 p_atom->x[0], p_atom->x[1], p_atom->x[2],
                 1.0, 0.0, "0", name, "  " );*/
        sprintf( line, PDB_ATOM_FORMAT_O,
                 p_atom->orig_id, p_atom->x[0], p_atom->x[1], p_atom->x[2] );
        //fprintf( stderr, "PDB NAME <%s>\n", p_atom->name);
        strncpy( buffer + i * PDB_ATOM_FORMAT_O_LENGTH, line,
                 PDB_ATOM_FORMAT_O_LENGTH );
    }


    if ( me != MASTER_NODE)
    {
        MPI_Send( buffer, buffer_req - 1, MPI_CHAR, MASTER_NODE,
                  np * PDB_ATOM_FORMAT_O_LENGTH + me, mpi_data->world );
    }
    else
    {
        buffer_len = system->n * PDB_ATOM_FORMAT_O_LENGTH;
        for ( i = 0; i < np; ++i )
            if ( i != MASTER_NODE )
            {
                MPI_Recv( buffer + buffer_len, buffer_req - buffer_len,
                          MPI_CHAR, i, np * PDB_ATOM_FORMAT_O_LENGTH + i, mpi_data->world,
                          &status );
                MPI_Get_count( &status, MPI_CHAR, &cnt );
                buffer_len += cnt;
            }
        buffer[buffer_len] = 0;
    }

    if ( me == MASTER_NODE)
    {
        fprintf( pdb, "%s", buffer );
        sfclose( pdb, "Write_PDB::pdb" );
    }

    /* Writing connect information */
    /*
    for(i=0; i < system->N; i++) {
      count = 0;
      for(j = Start_Index(i, bonds); j < End_Index(i, bonds); ++j) {
        bo = bonds->bond_list[j].bo_data.BO;
        if (bo > 0.3) {
          connect[count] = bonds->bond_list[j].nbr+1;
          count++;
        }
      }

      fprintf( out_control->pdb, "%6s%5d", "CONECT", i+1 );
      for( k=0; k < count; k++ )
        fprintf( out_control->pdb, "%5d", connect[k] );
      fprintf( out_control->pdb, "\n" );
    }
    */

    sfree(buffer, "buffer");
    sfree(line, "line");

    return SUCCESS;
}
