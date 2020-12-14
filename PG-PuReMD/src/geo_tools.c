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
#include "comm_tools.h"
#include "tool_box.h"
#include "vector.h"


/********************* geo format routines ******************/
static void Count_Geo_Atoms( FILE *geo, reax_system * const system )
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

        Fit_to_Periodic_Box( &system->big_box, &x );

        /* if the point is inside my_box, add it to my lists */
        if ( is_Inside_Box( &system->my_box, x ) == TRUE )
        {
            ++system->n;
        }
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


void Read_Geo_File( const char * const geo_file, reax_system * const system,
        control_params * const control, simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data )
{
    int i, j, serial, top;
    char descriptor[9];
    real box_x, box_y, box_z, alpha, beta, gamma;
    rvec x;
    char element[3], name[9];
    FILE *geo;
    reax_atom *atom;

    /* open the geometry file */
    geo = sfopen( geo_file, "r", "Read_Geo_File::geo" );

    /* read box information */
    fscanf( geo, CUSTOM_BOXGEO_FORMAT,
            descriptor, &box_x, &box_y, &box_z, &alpha, &beta, &gamma );

    /* initialize the box */
    Setup_Big_Box( box_x, box_y, box_z, alpha, beta, gamma, &system->big_box );

    /* initialize the simulation environment */
    Setup_Environment( system, control, mpi_data );

    /* count my atoms & allocate storage */
    Count_Geo_Atoms( geo, system );
    PreAllocate_Space( system, control, workspace );

    /* read in my atom info */
    top = 0;
    for ( i = 0; i < system->bigN; ++i )
    {
        fscanf( geo, CUSTOM_ATOM_FORMAT,
                &serial, element, name, &x[0], &x[1], &x[2] );
        element[2] = '\0';

        Fit_to_Periodic_Box( &system->big_box, &x );

#if defined(DEBUG_FOCUS)
        fprintf( stderr, "atom%d: %s %s %f %f %f\n",
                 serial, element, name, x[0], x[1], x[2] );
#endif

        /* if the point is inside my_box, add it to my list */
        if ( is_Inside_Box( &system->my_box, x ) == TRUE )
        {
            atom = &system->my_atoms[top];
            atom->orig_id = serial;

            /* strlen safe here as element is NULL-terminated above */
            for ( j = 0; j < strlen(element); j++ )
            {
                element[j] = toupper( element[j] );
            }
            atom->type = Get_Atom_Type( &system->reax_param, element, sizeof(element) );
            strncpy( atom->name, name, sizeof(atom->name) - 1 );
            atom->name[sizeof(atom->name) - 1] = '\0';
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

    sfclose( geo, "Read_Geo_File::geo" );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: finished reading the geo file\n", system->my_rank );
    //Print_My_Atoms( system );
#endif
}


static int Read_Box_Info( reax_system * const system, FILE *geo, int geo_format )
{
    char *cryst;
    char line[MAX_LINE + 1];
    char descriptor[9];
    char s_a[12], s_b[12], s_c[12], s_alpha[12], s_beta[12], s_gamma[12];
    char s_group[12], s_zValue[12];
    int ret;

    ret = FAILURE;

    fseek( geo, 0, SEEK_SET );

    switch ( geo_format )
    {
    case PDB:
        cryst = "CRYST1";
        break;
    case BGF:
        cryst = "CRYSTX";
        break;
    default:
        cryst = "BOX";
        break;
    }

    /* locate the cryst line in the geo file, read it and
     * initialize the big box */
    while ( fgets( line, MAX_LINE, geo ) )
    {
        if ( strncmp( line, cryst, 6 ) == 0 )
        {
            if ( geo_format == PDB )
            {
                sscanf( line, PDB_CRYST1_FORMAT, descriptor,
                        s_a, s_b, s_c, s_alpha, s_beta, s_gamma,
                        s_group, s_zValue );
            }
            else if ( geo_format == BGF )
            {
                sscanf( line, BGF_CRYSTX_FORMAT, descriptor,
                        s_a, s_b, s_c, s_alpha, s_beta, s_gamma );
            }

            /* compute full volume tensor from the angles */
            Setup_Big_Box( atof(s_a), atof(s_b), atof(s_c),
                    atof(s_alpha), atof(s_beta), atof(s_gamma),
                    &system->big_box );

            ret = SUCCESS;

            break;
        }
    }
    if ( ferror( geo ) )
    {
        ret = FAILURE;
    }
    else
    {
        fseek( geo, 0, SEEK_SET );
    }

    return ret;
}


static void Count_PDB_Atoms( FILE *geo, reax_system * const system )
{
    char *endptr = NULL;
    char line[MAX_LINE + 1];
    char s_x[9], s_y[9], s_z[9];
    rvec x;

    fseek( geo, 0, SEEK_SET );
    system->bigN = 0;
    system->n = 0;
    system->N = 0;

    /* increment number of atoms for each line denoting an atom desc */
    while ( fgets( line, MAX_LINE, geo ) )
    {
        if ( strncmp( line, "ATOM", 4 ) == 0
                || strncmp( line, "HETATM", 6 ) == 0 )
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
            Fit_to_Periodic_Box( &system->big_box, &x );

            /* if the point is inside my_box, add it to my lists */
            if ( is_Inside_Box( &system->my_box, x ) == TRUE )
            {
                ++system->n;
            }
        }
    }

    system->N = system->n;

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d@count atoms:\n", system->my_rank );
    fprintf( stderr, "p%d: bigN = %d\n", system->my_rank, system->bigN );
    fprintf( stderr, "p%d: n = %d\n", system->my_rank, system->n );
    fprintf( stderr, "p%d: N = %d\n\n", system->my_rank, system->N );
#endif
}


void Read_PDB_File( const char * const pdb_file, reax_system * const system,
        control_params * const control, simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data )
{
    FILE *pdb;
    char **tmp;
    char *s, *s1;
    char descriptor[9], serial[9];
    char atom_name[9], res_name[9], res_seq[9];
    char s_x[9], s_y[9], s_z[9];
    char occupancy[9], temp_factor[9];
    char seg_id[9], element[9], charge[9];
    char alt_loc, chain_id, icode;
    char *endptr = NULL;
    int i, c, c1, pdb_serial, top;
    rvec x;
    reax_atom *atom;

    pdb = sfopen( pdb_file, "r", "Read_PDB_File::pdb" );

    Allocate_Tokenizer_Space( &s, MAX_LINE, &s1, MAX_LINE,
            &tmp, MAX_TOKENS, MAX_TOKEN_LEN );

    if ( Read_Box_Info( system, pdb, PDB ) == FAILURE )
    {
        fprintf( stderr, "[ERROR] Read_Box_Info: no CRYST line in the pdb file!" );
        fprintf( stderr, " Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_GEO );
    }

    /* initialize the simulation environment */
    Setup_Environment( system, control, mpi_data );

    Count_PDB_Atoms( pdb, system );
    PreAllocate_Space( system, control, workspace );

    /* start reading and processing the pdb file */
    fseek( pdb, 0, SEEK_SET );
    c = 0;
    c1 = 0;
    top = 0;

    s[0] = 0;
    for ( i = 0; i < c1; ++i )
    {
        tmp[i][0] = 0;
    }

    while ( fgets( s, MAX_LINE, pdb ) )
    {
        /* read new line and tokenize it */
        strncpy( s1, s, MAX_LINE - 1 );
        c1 = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        /* process new line */
        if ( strncmp(tmp[0], "ATOM", 4) == 0
                || strncmp(tmp[0], "HETATM", 6) == 0 )
        {
            if ( strncmp(tmp[0], "ATOM", 4) == 0 )
            {
                strncpy( descriptor, s1, 6 );
                descriptor[6] = '\0';
                strncpy( serial, &s1[6], 5 );
                serial[5] = '\0';
                strncpy( atom_name, &s1[12], 4 );
                atom_name[4] = '\0';
                //strncpy( serial, &s1[6], 7 );       serial[7] = 0;
                //strncpy( atom_name, &s1[13], 3 );   atom_name[3] = 0;
                alt_loc = s1[16];
                strncpy( res_name, &s1[17], 3 );
                res_name[3] = '\0';
                chain_id = s1[21];
                strncpy( res_seq, &s1[22], 4 );
                res_seq[4] = '\0';
                icode = s1[26];
                strncpy( s_x, &s1[30], 8 );
                s_x[8] = '\0';
                strncpy( s_y, &s1[38], 8 );
                s_y[8] = '\0';
                strncpy( s_z, &s1[46], 8 );
                s_z[8] = '\0';
                strncpy( occupancy, &s1[54], 6 );
                occupancy[6] = '\0';
                strncpy( temp_factor, &s1[60], 6 );
                temp_factor[6] = '\0';
                strncpy( seg_id, &s1[72], 4 );
                seg_id[4] = '\0';
                strncpy( element, &s1[76], 2 );
                element[2] = '\0';
                strncpy( charge, &s1[78], 2 );
                charge[2] = '\0';
            }
            else if (strncmp(tmp[0], "HETATM", 6) == 0)
            {
                strncpy( descriptor, s1, 6 );
                descriptor[6] = '\0';
                strncpy( serial, &s1[6], 5 );
                serial[5] = '\0';
                strncpy( atom_name, &s1[12], 4 );
                atom_name[4] = '\0';
                //strncpy( serial, &s1[6], 7 );       serial[7] = 0;
                //strncpy( atom_name, &s1[13], 3 );   atom_name[3] = 0;
                alt_loc = s1[16];
                strncpy( res_name, &s1[17], 3 );
                res_name[3] = '\0';
                chain_id = s1[21];
                strncpy( res_seq, &s1[22], 4 );
                res_seq[4] = '\0';
                icode = s1[26];
                strncpy( s_x, &s1[30], 8 );
                s_x[8] = '\0';
                strncpy( s_y, &s1[38], 8 );
                s_y[8] = '\0';
                strncpy( s_z, &s1[46], 8 );
                s_z[8] = '\0';
                strncpy( occupancy, &s1[54], 6 );
                occupancy[6] = '\0';
                strncpy( temp_factor, &s1[60], 6 );
                temp_factor[6] = '\0';
                strncpy( seg_id, &s1[72], 4 );
                seg_id[4] = '\0';
                strncpy( element, &s1[76], 2 );
                element[2] = '\0';
                strncpy( charge, &s1[78], 2 );
                charge[2] = '\0';
            }

            /* if the point is inside my_box, add it to my lists */
            Make_Point( strtod( &s_x[0], &endptr ),
                    strtod( &s_y[0], &endptr ),
                    strtod( &s_z[0], &endptr ), &x );

            Fit_to_Periodic_Box( &system->big_box, &x );

            if ( is_Inside_Box( &system->my_box, x ) == TRUE )
            {
                /* store orig_id, type, name and coord info of the new atom */
                atom = &system->my_atoms[top];
                pdb_serial = (int) strtod( &serial[0], &endptr );
                atom->orig_id = pdb_serial;

                Trim_Spaces( element, sizeof(element) );
                atom->type = Get_Atom_Type( &system->reax_param, element, sizeof(element) );
                strncpy( atom->name, atom_name, sizeof(atom->name) - 1 );
                atom->name[sizeof(atom->name) - 1] = '\0';

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

            c++;
        }

        /* IMPORTANT: We do not check for the soundness of restrictions here.
         * When atom2 is on atom1's restricted list, and there is a restriction
         * on atom2, then atom1 has to be on atom2's restricted list, too.
         * However, we do not check if this is the case in the input file,
         * this is upto the user. */
        else if ( !strncmp( tmp[0], "CONECT", 6 ) )
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
        {
            tmp[i][0] = 0;
        }
    }
    if ( ferror( pdb ) )
    {
        fprintf( stderr, "[ERROR] Read_PDB_FILE: error while reading PDB file!" );
        fprintf( stderr, " Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_GEO );
    }

    Deallocate_Tokenizer_Space( &s, &s1, &tmp, MAX_TOKENS );

    sfclose( pdb, "Read_PDB_File::pdb" );
}


/* PDB serials are written without regard to the order, we'll see if this
 * cause trouble, if so we'll have to rethink this approach
 * Also, we do not write connect lines yet.  */
void Write_PDB_File( reax_system * const system, reax_list * const bond_list,
        simulation_data * const data, control_params * const control,
        mpi_datatypes * const mpi_data, output_controls * const out_control )
{
    int i, cnt, me, np, buffer_req, buffer_len, ret;
    //int j, connect[4];
    char name[8], fname[MAX_STR], *line, *buffer;
    FILE *pdb;
    //real bo;
    real alpha, beta, gamma;
    MPI_Status status;
    reax_atom *p_atom;

    me = system->my_rank;
    np = control->nprocs;

    /* Allocation*/
    line = smalloc( sizeof(char) * PDB_ATOM_FORMAT_O_LENGTH, "Write_PDB_File::line" );
    if ( me == MASTER_NODE )
    {
        buffer_req = system->bigN * PDB_ATOM_FORMAT_O_LENGTH;
    }
    else
    {
        buffer_req = system->n * PDB_ATOM_FORMAT_O_LENGTH;
    }

    buffer = smalloc( sizeof(char) * buffer_req, "Write_PDB_File::buffer" );

    pdb = NULL;
    line[0] = '\0';
    buffer[0] = '\0';

    /* open pdb and write header */
    if ( me == MASTER_NODE )
    {
        /* Writing Box information */
        gamma = ACOS( (system->big_box.box[0][0] * system->big_box.box[1][0]
                    + system->big_box.box[0][1] * system->big_box.box[1][1]
                    + system->big_box.box[0][2] * system->big_box.box[1][2])
                / (system->big_box.box_norms[0] * system->big_box.box_norms[1]) );
        beta  = ACOS( (system->big_box.box[0][0] * system->big_box.box[2][0]
                    + system->big_box.box[0][1] * system->big_box.box[2][1]
                    + system->big_box.box[0][2] * system->big_box.box[2][2])
                / (system->big_box.box_norms[0] * system->big_box.box_norms[2]) );
        alpha = ACOS( (system->big_box.box[2][0] * system->big_box.box[1][0]
                    + system->big_box.box[2][1] * system->big_box.box[1][1]
                    + system->big_box.box[2][2] * system->big_box.box[1][2])
                / (system->big_box.box_norms[2] * system->big_box.box_norms[1]) );

        /* strlen safe here as control->sim_name is NULL-terminated in control.c */
        snprintf( fname, sizeof(fname) - 1, "%.*s-%d.pdb",
                (int) strlen(control->sim_name), control->sim_name, data->step );
        fname[sizeof(fname) - 1] = '\0';
        pdb = sfopen( fname, "w", "Write_PDB_File::pdb" );
        fprintf( pdb, PDB_CRYST1_FORMAT_O,
                 "CRYST1",
                 system->big_box.box_norms[0], system->big_box.box_norms[1],
                 system->big_box.box_norms[2],
                 RAD2DEG(alpha), RAD2DEG(beta), RAD2DEG(gamma), " ", 0 );
        fprintf( out_control->log, "Box written\n" );
#if defined(DEBUG)
        fflush( out_control->log );
#endif
    }

    /* write atom lines to buffer */
    for ( i = 0; i < system->n; i++)
    {
        p_atom = &system->my_atoms[i];

        strncpy( name, p_atom->name, sizeof(name) - 1 );
        name[sizeof(name) - 1] = '\0';

        Trim_Spaces( name, sizeof(name) );

        snprintf( line, sizeof(line) - 1, PDB_ATOM_FORMAT_O,
                 "ATOM  ", p_atom->orig_id, p_atom->name, ' ', "REX", ' ', 1, ' ',
                 p_atom->x[0], p_atom->x[1], p_atom->x[2],
                 1.0, 0.0, "0", name, "  " );
        line[sizeof(line) - 1] = '\0';
        fprintf( stderr, "PDB NAME <%s>\n", p_atom->name );
        strncpy( &buffer[i * PDB_ATOM_FORMAT_O_LENGTH], line,
                 PDB_ATOM_FORMAT_O_LENGTH );
    }


    if ( me != MASTER_NODE)
    {
        ret = MPI_Send( buffer, buffer_req - 1, MPI_CHAR, MASTER_NODE,
                  np * PDB_ATOM_FORMAT_O_LENGTH + me, MPI_COMM_WORLD );
        Check_MPI_Error( ret, __FILE__, __LINE__ );
    }
    else
    {
        buffer_len = system->n * PDB_ATOM_FORMAT_O_LENGTH;
        for ( i = 0; i < np; ++i )
        {
            if ( i != MASTER_NODE )
            {
                ret = MPI_Recv( buffer + buffer_len, buffer_req - buffer_len,
                        MPI_CHAR, i, np * PDB_ATOM_FORMAT_O_LENGTH + i, MPI_COMM_WORLD,
                        &status );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                ret = MPI_Get_count( &status, MPI_CHAR, &cnt );
                Check_MPI_Error( ret, __FILE__, __LINE__ );
                buffer_len += cnt;
            }
        }
        buffer[buffer_len] = '\0';
    }

    if ( me == MASTER_NODE )
    {
        fprintf( pdb, "%s", buffer );
        sfclose( pdb, "Write_PDB_File::pdb" );
    }

    /* Writing connect information */
    /*
    for(i=0; i < system->N; i++) {
      count = 0;
      for(j = Start_Index(i, bond_list); j < End_Index(i, bond_list); ++j) {
        bo = bond_list->bond_list[j].bo_data.BO;
        if (bo > 0.3) {
          connect[count] = bond_list->bond_list[j].nbr+1;
          count++;
        }
      }

      fprintf( out_control->pdb, "%6s%5d", "CONECT", i+1 );
      for( k=0; k < count; k++ )
        fprintf( out_control->pdb, "%5d", connect[k] );
      fprintf( out_control->pdb, "\n" );
    }
    */

    sfree( buffer, "Write_PDB_File::buffer" );
    sfree( line, "Write_PDB_File::line" );
}


void Read_BGF( const char * const bgf_file, reax_system * const system,
        control_params * const control, simulation_data * const data,
        storage * const workspace, mpi_datatypes * const mpi_data )
{
    FILE *bgf;
    char **tokens;
    char *line, *backup;
    char descriptor[7], serial[6];
    char atom_name[6], res_name[4], res_seq[6];
    char s_x[11], s_y[11], s_z[11];
    char occupancy[4], temp_factor[3];
    char element[6], charge[9];
    char chain_id;
    char s_a[12], s_b[12], s_c[12], s_alpha[12], s_beta[12], s_gamma[12];
    char *endptr;
    int i, n, atom_cnt, token_cnt, bgf_serial, ratom;
    rvec x;

    endptr = NULL;
    ratom = 0;

    bgf = sfopen( bgf_file, "r", "Read_BGF::bgf" );

    Allocate_Tokenizer_Space( &line, MAX_LINE, &backup, MAX_LINE,
            &tokens, MAX_TOKENS, MAX_TOKEN_LEN );

    /* count number of atoms in the BGF file */
    n = 0;
    line[0] = '\0';

    if ( Read_Box_Info( system, bgf, BGF ) == FAILURE )
    {
        fprintf( stderr, "[ERROR] Read_Box_Info: no CRYSTX line in the BGF file!" );
        fprintf( stderr, " Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_GEO );
    }

    /* initialize the simulation environment */
    Setup_Environment( system, control, mpi_data );

    while ( fgets( line, MAX_LINE, bgf ) )
    {
        tokens[0][0] = '\0';
        token_cnt = Tokenize( line, &tokens, MAX_TOKEN_LEN );

        if ( strncmp( tokens[0], "ATOM", 4 ) == 0
                || strncmp( tokens[0], "HETATM", 6 ) == 0 )
        {
            ++n;
        }

        line[0] = '\0';
    }
    if ( ferror( bgf ) )
    {
        fprintf( stderr, "[ERROR] Unable to read BGF file. Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_GEO );
    }

    sfclose( bgf, "Read_BGF::bgf" );

    system->n = n;
    system->N = n;
    system->bigN = n;
    PreAllocate_Space( system, control, workspace );

//    workspace->map_serials = scalloc( MAX_ATOM_ID, sizeof(int),
//            "Read_BGF::workspace->map_serials" );
//    for ( i = 0; i < MAX_ATOM_ID; ++i )
//    {
//        workspace->map_serials[i] = -1;
//    }

    bgf = sfopen( bgf_file, "r", "Read_BGF::bgf" );
    atom_cnt = 0;
    token_cnt = 0;

    while ( fgets( line, MAX_LINE, bgf ) )
    {
        /* read new line and tokenize it */
        strncpy( backup, line, MAX_LINE - 1 );
        backup[MAX_LINE - 1] = '\0';
        token_cnt = Tokenize( line, &tokens, MAX_TOKEN_LEN );

        /* process lines with atom info (i.e., begin with keywords HETATM or ATOM),
         * format: "%6s %5d %-5s %3s %c %5s%10.5f%10.5f%10.5f %-5s%3d%2d %8.5f"
         *
         * also, it's common to see the atom info format
         * inlined in MSI BGF files as follows:
         * FORMAT ATOM   (a6,1x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5)
         * */
        if ( strncmp( tokens[0], "ATOM", 4 ) == 0
                || strncmp( tokens[0], "HETATM", 6 ) == 0 )
        {
            strncpy( descriptor, backup, sizeof(descriptor) - 1 );
            descriptor[sizeof(descriptor) - 1] = '\0';
            strncpy( serial, backup + 7, sizeof(serial) - 1 );
            serial[sizeof(serial) - 1] = '\0';
            strncpy( atom_name, backup + 13, sizeof(atom_name) - 1 );
            atom_name[sizeof(atom_name) - 1] = '\0';
            strncpy( res_name, backup + 19, sizeof(res_name) - 1 );
            res_name[sizeof(res_name) - 1] = '\0';
            chain_id = backup[23];
            strncpy( res_seq, backup + 25, sizeof(res_seq) - 1 );
            res_seq[sizeof(res_seq) - 1] = '\0';
            strncpy( s_x, backup + 30, sizeof(s_x) - 1 );
            s_x[sizeof(s_x) - 1] = '\0';
            strncpy( s_y, backup + 40, sizeof(s_y) - 1 );
            s_y[sizeof(s_y) - 1] = '\0';
            strncpy( s_z, backup + 50, sizeof(s_x) - 1 );
            s_z[sizeof(s_x) - 1] = '\0';
            strncpy( element, backup + 61, sizeof(element) - 1 );
            element[sizeof(element) - 1] = '\0';
            strncpy( occupancy, backup + 66, sizeof(occupancy) - 1 );
            occupancy[sizeof(occupancy) - 1] = '\0';
            strncpy( temp_factor, backup + 69, sizeof(temp_factor) - 1 );
            temp_factor[sizeof(temp_factor) - 1] = '\0';
            strncpy( charge, backup + 72, sizeof(charge) - 1 );
            charge[sizeof(charge) - 1] = '\0';

//            sscanf( backup, BGF_ATOM_FORMAT, descriptor, serial, atom_name,
//                    res_name, &chain_id, res_seq, s_x, s_y, s_z, element,
//                    occupancy, temp_factor, charge );

            Make_Point( strtod( s_x, &endptr ), strtod( s_y, &endptr ),
                        strtod( s_z, &endptr ), &x );

            Fit_to_Periodic_Box( &system->big_box, &x );

            /* if the point is inside my_box, add it to my lists */
            if ( is_Inside_Box( &system->my_box, x ) == TRUE )
            {
                /* add to mapping */
                bgf_serial = strtod( serial, &endptr );
                Check_Input_Range( bgf_serial, 0, MAX_ATOM_ID, "Invalid bgf serial" );
//                workspace->map_serials[bgf_serial] = atom_cnt;
                system->my_atoms[atom_cnt].orig_id = bgf_serial;

                /* copy atomic positions */
                system->my_atoms[atom_cnt].x[0] = x[0];
                system->my_atoms[atom_cnt].x[1] = x[1];
                system->my_atoms[atom_cnt].x[2] = x[2];

                /* atom name and type */
                strncpy( system->my_atoms[atom_cnt].name, atom_name,
                        sizeof(system->my_atoms[atom_cnt].name) - 1 );
                system->my_atoms[atom_cnt].name[sizeof(system->my_atoms[atom_cnt].name) - 1] = '\0';
                Trim_Spaces( element, sizeof(element) );
                system->my_atoms[atom_cnt].type =
                    Get_Atom_Type( &system->reax_param, element, sizeof(element) );

#if defined(DEBUG_FOCUS)
                fprintf( stderr,
                        "[INFO] atom_cnt = %5d, atom_type = %3d, x = (%10.5f,%10.5f,%10.5f), q = %10.5f, occ = %s, temp = %s, res_name = %4s, element = %s\n",
                        atom_cnt, system->my_atoms[atom_cnt].type,
                        system->my_atoms[atom_cnt].x[0],
                        system->my_atoms[atom_cnt].x[1],
                        system->my_atoms[atom_cnt].x[2],
                        system->my_atoms[atom_cnt].q, occupancy, temp_factor,
                        res_name, element );
#endif

                atom_cnt++;
            }
        }
        else if ( strncmp( tokens[0], "CONECT", 6 ) == 0 )
        {
//            if ( control->restrict_bonds )
//            {
//                /* check number of restrictions */
//                Check_Input_Range( token_cnt - 2, 0, MAX_RESTRICT,
//                        "CONECT line exceeds max restrictions allowed.\n" );
//
//                /* read bond restrictions */
//                bgf_serial = atoi(tokens[1]);
//                if ( is_Valid_Serial( workspace, bgf_serial ) )
//                {
//                    ratom = workspace->map_serials[ bgf_serial ];
//                }
//
//                workspace->restricted[ ratom ] = token_cnt - 2;
//                for ( i = 2; i < token_cnt; ++i )
//                {
//                    if ( is_Valid_Serial( workspace, bgf_serial = atoi(tokens[i]) ) )
//                    {
//                        workspace->restricted_list[ratom][i - 2] =
//                            workspace->map_serials[bgf_serial];
//                    }
//                }
//            }
        }

        /* clear previous input line */
        line[0] = '\0';

        for ( i = 0; i < token_cnt; ++i )
        {
            tokens[i][0] = '\0';
        }
    }
    if ( ferror( bgf ) )
    {
        fprintf( stderr, "[ERROR] Unable to read BGF file. Terminating...\n" );
        MPI_Abort( MPI_COMM_WORLD, INVALID_GEO );
    }

    system->n = atom_cnt;
    system->N = atom_cnt;

    Deallocate_Tokenizer_Space( &line, &backup, &tokens, MAX_TOKENS );

    sfclose( bgf, "Read_BGF::bgf" );
}
