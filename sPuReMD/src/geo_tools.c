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

#include "geo_tools.h"

#include "allocate.h"
#include "box.h"
#include "tool_box.h"
#include "vector.h"


static void Count_Geo_Atoms( FILE *geo, reax_system *system )
{
    int i, serial;
    rvec x;
    char element[3], name[9], line[MAX_LINE + 1];

    /* total number of atoms */
    fscanf( geo, " %d", &(system->N) );

    /* count atoms */
    for ( i = 0; i < system->N; ++i )
    {
        fscanf( geo, CUSTOM_ATOM_FORMAT,
                &serial, element, name, &x[0], &x[1], &x[2] );
        Fit_to_Periodic_Box( &(system->box), &x );
    }

    fseek( geo, 0, SEEK_SET );
    fgets( line, MAX_LINE, geo );
    fgets( line, MAX_LINE, geo );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "N = %d\n\n", system->N );
#endif
}


static int Read_Box_Info( reax_system *system, FILE *geo, int geo_format )
{
    int ret, cryst_len;
    char *cryst;
    char line[MAX_LINE + 1];
    char descriptor[9];
    char s_a[12], s_b[12], s_c[12], s_alpha[12], s_beta[12], s_gamma[12];
    char s_group[12], s_zValue[12];

    /* set the pointer to the beginning of the file */
    fseek( geo, 0, SEEK_SET );

    /* initialize variables */
    ret = FAILURE;

    switch ( geo_format )
    {
        case PDB:
            cryst = "CRYST1";
            cryst_len = 6;
            break;

        default:
            cryst = "BOX";
            cryst_len = 3;
            break;
    }

    /* locate the cryst line in the geo file, read it and
       initialize the big box */
    while ( fgets( line, MAX_LINE, geo ) )
    {
        if ( strncmp( line, cryst, cryst_len ) == 0 )
        {
            if ( geo_format == PDB )
            {
                sscanf( line, PDB_CRYST1_FORMAT,
                        &descriptor[0],
                        &s_a[0], &s_b[0], &s_c[0],
                        &s_alpha[0], &s_beta[0], &s_gamma[0],
                        &s_group[0], &s_zValue[0] );

                /* compute full volume tensor from the angles */
                Setup_Box( atof(s_a),  atof(s_b), atof(s_c),
                        atof(s_alpha), atof(s_beta), atof(s_gamma),
                        &system->box );

                ret = SUCCESS;
                break;
            }
        }
    }
    if ( ferror( geo ) )
    {
        ret = FAILURE;
    }

    return ret;
}


void Read_Geo( const char * const geo_file, reax_system* system, control_params *control,
        simulation_data *data, static_storage *workspace )
{

    FILE *geo;
    char descriptor[9];
    int i, serial, top;
    real box_x, box_y, box_z, alpha, beta, gamma;
    rvec x;
    char element[3], name[9];
    reax_atom *atom;

    /* open the geometry file */
    geo = sfopen( geo_file, "r" );

    /* read box information */
    fscanf( geo, CUSTOM_BOXGEO_FORMAT,
            descriptor, &box_x, &box_y, &box_z, &alpha, &beta, &gamma );
    /* initialize the box */
    Setup_Box( box_x, box_y, box_z, alpha, beta, gamma, &(system->box) );

    /* count my atoms & allocate storage */
    Count_Geo_Atoms( geo, system );
    PreAllocate_Space( system, control, workspace );

    /* read in my atom info */
    top = 0;
    for ( i = 0; i < system->N; ++i )
    {
        fscanf( geo, CUSTOM_ATOM_FORMAT,
                &serial, element, name, &x[0], &x[1], &x[2] );
        Fit_to_Periodic_Box( &(system->box), &x );
#if defined(DEBUG)
        fprintf( stderr, "atom%d: %s %s %f %f %f\n",
                 serial, element, name, x[0], x[1], x[2] );
#endif

        atom = &(system->atoms[top]);
        workspace->orig_id[i] = serial;
        atom->type = Get_Atom_Type( &(system->reaxprm), element );
        strncpy( atom->name, name, 8 );
        rvec_Copy( atom->x, x );
        rvec_MakeZero( atom->v );
        rvec_MakeZero( atom->f );
        atom->q = 0.;

        top++;
    }

    sfclose( geo, "Read_Geo::geo" );
}


static void Count_PDB_Atoms( FILE *geo, reax_system *system )
{
    char *endptr = NULL;
    char line[MAX_LINE + 1];
    char s_x[9], s_y[9], s_z[9];
    rvec x;

    fseek( geo, 0, SEEK_SET );
    system->N = 0;

    /* increment number of atoms for each line denoting an atom desc */
    while ( fgets( line, MAX_LINE, geo ) )
    {
        if ( strncmp( line, "ATOM", 4 ) == 0
                || strncmp( line, "HETATM", 6 ) == 0 )
        {
            system->N++;

            strncpy( s_x, line + 30, 8 );
            s_x[8] = 0;
            strncpy( s_y, line + 38, 8 );
            s_y[8] = 0;
            strncpy( s_z, line + 46, 8 );
            s_z[8] = 0;

            Make_Point( strtod( s_x, &endptr ), strtod( s_y, &endptr ),
                    strtod( s_z, &endptr ), &x );

            Fit_to_Periodic_Box( &(system->box), &x );
        }
    }

#if defined(DEBUG)
    fprintf( stderr, "count atoms:\n" );
    fprintf( stderr, "N = %d\n\n", system->N );
#endif
}


void Read_PDB( const char * const pdb_file, reax_system* system, control_params *control,
        simulation_data *data, static_storage *workspace )
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

    /* open pdb file */
    pdb = sfopen( pdb_file, "r" );

    /* allocate memory for tokenizing pdb lines */
    Allocate_Tokenizer_Space( &s, &s1, &tmp );

    /* read box information */
    if ( Read_Box_Info( system, pdb, PDB ) == FAILURE )
    {
        fprintf( stderr, "[ERROR] Read_Box_Info: no CRYST line in the pdb file!" );
        fprintf( stderr, " Terminating...\n" );
        exit( INVALID_GEO );
    }

    Count_PDB_Atoms( pdb, system );
    PreAllocate_Space( system, control, workspace );

    /* start reading and processing the pdb file */
    fseek( pdb, 0, SEEK_SET );
    c = 0;
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
            else if ( strncmp(tmp[0], "HETATM", 6) == 0 )
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
            Make_Point( strtod( &s_x[0], &endptr ), strtod( &s_y[0], &endptr ),
                    strtod( &s_z[0], &endptr ), &x );

            Fit_to_Periodic_Box( &system->box, &x );

            if ( is_Inside_Box( &system->box, x ) )
            {
                /* store orig_id, type, name and coord info of the new atom */
                atom = &system->atoms[top];
                pdb_serial = (int) strtod( &serial[0], &endptr );
                workspace->orig_id[top] = pdb_serial;

                Trim_Spaces( element, 9 );
                atom->type = Get_Atom_Type( &system->reaxprm, element );
                strncpy( atom->name, atom_name, 8 );

                rvec_Copy( atom->x, x );
                rvec_MakeZero( atom->v );
                rvec_MakeZero( atom->f );
                atom->q = 0;

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
        fprintf( stderr, "[ERROR] Unable to read PDB file. Terminating...\n" );
        exit( INVALID_INPUT );
    }

    sfclose( pdb, "Read_PDB::pdb" );

    Deallocate_Tokenizer_Space( &s, &s1, &tmp );
} 


/* PDB serials are written without regard to the order, we'll see if this
 * cause trouble, if so we'll have to rethink this approach
 * Also, we do not write connect lines yet.
 * */
void Write_PDB( reax_system* system, reax_list* bonds, simulation_data *data,
        control_params *control, static_storage *workspace, output_controls *out_control )
{
    int i; 
    char name[8];
    real alpha, beta, gamma;
    rvec x;
    reax_atom *p_atom;
    char fname[MAX_STR];
    char buffer[PDB_ATOM_FORMAT_O_LENGTH + 1];
    FILE *pdb;

    /* Writing Box information */
    gamma = ACOS( (system->box.box[0][0] * system->box.box[1][0] +
                system->box.box[0][1] * system->box.box[1][1] +
                system->box.box[0][2] * system->box.box[1][2]) /
                  (system->box.box_norms[0] * system->box.box_norms[1]) );
    beta  = ACOS( (system->box.box[0][0] * system->box.box[2][0] +
                system->box.box[0][1] * system->box.box[2][1] +
                system->box.box[0][2] * system->box.box[2][2]) /
            (system->box.box_norms[0] * system->box.box_norms[2]) );
    alpha = ACOS( (system->box.box[2][0] * system->box.box[1][0] +
                system->box.box[2][1] * system->box.box[1][1] +
                system->box.box[2][2] * system->box.box[1][2]) /
            (system->box.box_norms[2] * system->box.box_norms[1]) );

    /* open pdb and write header */
    snprintf( fname, MAX_STR + 9, "%s-%d.pdb", control->sim_name, data->step );
    pdb = sfopen( fname, "w" );
    fprintf( pdb, PDB_CRYST1_FORMAT_O,
             "CRYST1",
             system->box.box_norms[0], system->box.box_norms[1],
             system->box.box_norms[2],
             RAD2DEG(alpha), RAD2DEG(beta), RAD2DEG(gamma), " ", 0 );

    /* write atom lines to buffer */
    for ( i = 0; i < system->N; i++)
    {
        p_atom = &(system->atoms[i]);

        strncpy( name, p_atom->name, 8 );
        Trim_Spaces( name, 8 );

        memcpy( x, p_atom->x, 3 * sizeof(real) );
        Fit_to_Periodic_Box( &(system->box), &x );

        snprintf( buffer, PDB_ATOM_FORMAT_O_LENGTH, PDB_ATOM_FORMAT_O,
                "ATOM  ", workspace->orig_id[i], p_atom->name, ' ', "REX", ' ', 1, ' ',
                x[0], x[1], x[2],
                1.0, 0.0, "0", name, "  " );

        buffer[PDB_ATOM_FORMAT_O_LENGTH] = '\n';

        fprintf( pdb, "%s\n", buffer );
    }
    
    if ( ferror( pdb ) )
    {
        fprintf( stderr, "[ERROR] Unable to write PDB file. Terminating...\n" );
        exit( INVALID_INPUT );
    }

    sfclose( pdb, "Write_PDB::pdb" );
}


void Read_BGF( const char * const bgf_file, reax_system* system, control_params *control,
        simulation_data *data, static_storage *workspace )
{
    FILE *bgf;
    char **tokens;
    char *line, *backup;
    char descriptor[10], serial[10];
    char atom_name[10], res_name[10], res_seq[10];
    char s_x[12], s_y[12], s_z[12];
    char occupancy[10], temp_factor[10];
    char element[10], charge[10];
    char chain_id;
    char s_a[12], s_b[12], s_c[12], s_alpha[12], s_beta[12], s_gamma[12];
    char *endptr = NULL;
    int i, atom_cnt, token_cnt, bgf_serial, ratom = 0;

    /* open biograf file */
    bgf = sfopen( bgf_file, "r" );

    /* allocate memory for tokenizing biograf file lines */
    line = (char*) smalloc( sizeof(char)  * MAX_LINE,
           "Read_BGF::line" );
    backup = (char*) smalloc( sizeof(char)  * MAX_LINE,
           "Read_BGF::backup" );
    tokens = (char**) smalloc( sizeof(char*) * MAX_TOKENS,
           "Read_BGF::tokens" );
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        tokens[i] = (char*) smalloc( sizeof(char) * MAX_TOKEN_LEN,
               "Read_BGF::tokens[i]" );
    }

    /* count number of atoms in the pdb file */
    system->N = 0;
    line[0] = 0;

    while ( fgets( line, MAX_LINE, bgf ) )
    {
        tokens[0][0] = 0;
        token_cnt = Tokenize( line, &tokens );

        if ( !strncmp( tokens[0], "ATOM", MAX_TOKEN_LEN )
                || !strncmp( tokens[0], "HETATM", MAX_TOKEN_LEN ) )
        {
            ++system->N;
        }

        line[0] = 0;
    }
    if ( ferror( bgf ) )
    {
        fprintf( stderr, "[ERROR] Unable to read BGF file. Terminating...\n" );
        exit( INVALID_INPUT );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "system->N: %d\n", system->N );
#endif

    sfclose( bgf, "Read_BGF::bgf" );

    /* memory allocations for atoms, atom maps, bond restrictions */
    system->atoms = (reax_atom*) scalloc( system->N, sizeof(reax_atom),
            "Read_BGF::system->atoms" );

    workspace->map_serials = (int*) scalloc( MAX_ATOM_ID, sizeof(int),
            "Read_BGF::workspace->map_serials" );
    for ( i = 0; i < MAX_ATOM_ID; ++i )
    {
        workspace->map_serials[i] = -1;
    }

    workspace->orig_id = (int*) scalloc( system->N, sizeof(int),
            "Read_BGF::workspace->orig_id" );
    workspace->restricted  = (int*) scalloc( system->N, sizeof(int),
            "Read_BGF::workspace->restricted" );
    workspace->restricted_list = (int**) scalloc( system->N, sizeof(int*),
            "Read_BGF::workspace->restricted_list" );
    for ( i = 0; i < system->N; ++i )
    {
        workspace->restricted_list[i] = (int*) scalloc( MAX_RESTRICT, sizeof(int),
                "Read_BGF::workspace->restricted_list[i]" );
    }

    /* start reading and processing bgf file */
    bgf = sfopen( bgf_file, "r" );
    atom_cnt = 0;
    token_cnt = 0;

    while ( fgets( line, MAX_LINE, bgf ) )
    {
        /* read new line and tokenize it */
        strncpy( backup, line, MAX_LINE - 1 );
        token_cnt = Tokenize( line, &tokens );

        /* process new line */
        if ( !strncmp(tokens[0], "ATOM", 4) || !strncmp(tokens[0], "HETATM", 6) )
        {
            if ( !strncmp(tokens[0], "ATOM", 4) )
            {
                strncpy( &descriptor[0], backup, 6 );
                descriptor[6] = 0;
                strncpy( &serial[0], backup + 7, 5 );
                serial[5] = 0;
                strncpy( &atom_name[0], backup + 13, 5 );
                atom_name[5] = 0;
                strncpy( &res_name[0], backup + 19, 3 );
                res_name[3] = 0;
                chain_id = backup[23];
                strncpy( &res_seq[0], backup + 25, 5 );
                res_seq[5] = 0;
                strncpy( &s_x[0], backup + 30, 10 );
                s_x[10] = 0;
                strncpy( &s_y[0], backup + 40, 10 );
                s_y[10] = 0;
                strncpy( &s_z[0], backup + 50, 10 );
                s_z[10] = 0;
                strncpy( &element[0], backup + 61, 5 );
                element[5] = 0;
                strncpy( &occupancy[0], backup + 66, 3 );
                occupancy[3] = 0;
                strncpy( &temp_factor[0], backup + 69, 2 );
                temp_factor[2] = 0;
                strncpy( &charge[0], backup + 72, 8 );
                charge[8] = 0;
            }
            else if ( !strncmp(tokens[0], "HETATM", 6) )
            {
                /* bgf hetatm:
                   (7x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5) */
                strncpy( &descriptor[0], backup, 6 );
                descriptor[6] = 0;
                strncpy( &serial[0], backup + 7, 5 );
                serial[5] = 0;
                strncpy( &atom_name[0], backup + 13, 5 );
                atom_name[5] = 0;
                strncpy( &res_name[0], backup + 19, 3 );
                res_name[3] = 0;
                chain_id = backup[23];
                strncpy( &res_seq[0], backup + 25, 5 );
                res_seq[5] = 0;
                strncpy( &s_x[0], backup + 30, 10 );
                s_x[10] = 0;
                strncpy( &s_y[0], backup + 40, 10 );
                s_y[10] = 0;
                strncpy( &s_z[0], backup + 50, 10 );
                s_z[10] = 0;
                strncpy( &element[0], backup + 61, 5 );
                element[5] = 0;
                strncpy( &occupancy[0], backup + 66, 3 );
                occupancy[3] = 0;
                strncpy( &temp_factor[0], backup + 69, 2 );
                temp_factor[2] = 0;
                strncpy( &charge[0], backup + 72, 8 );
                charge[8] = 0;
            }

            /* add to mapping */
            bgf_serial = strtod( &serial[0], &endptr );
            Check_Input_Range( bgf_serial, 0, MAX_ATOM_ID, "Invalid bgf serial" );
            workspace->map_serials[ bgf_serial ] = atom_cnt;
            workspace->orig_id[ atom_cnt ] = bgf_serial;
            // fprintf( stderr, "map %d --> %d\n", bgf_serial, atom_cnt );

            /* copy atomic positions */
            system->atoms[atom_cnt].x[0] = strtod( &s_x[0], &endptr );
            system->atoms[atom_cnt].x[1] = strtod( &s_y[0], &endptr );
            system->atoms[atom_cnt].x[2] = strtod( &s_z[0], &endptr );

            /* atom name and type */
            strncpy( system->atoms[atom_cnt].name, atom_name, 8 );
            Trim_Spaces( element, 10 );
            system->atoms[atom_cnt].type =
                Get_Atom_Type( &(system->reaxprm), element );

            /* fprintf( stderr,
            "a:%3d(%1d) c:%10.5f%10.5f%10.5f q:%10.5f occ:%s temp:%s seg_id:%s element:%s\n",
             atom_cnt, system->atoms[ atom_cnt ].type,
             system->atoms[ atom_cnt ].x[0],
             system->atoms[ atom_cnt ].x[1], system->atoms[ atom_cnt ].x[2],
             system->atoms[ atom_cnt ].q, occupancy, temp_factor,
             seg_id, element ); */

            atom_cnt++;
        }
        else if (!strncmp( tokens[0], "CRYSTX", 6 ))
        {
            sscanf( backup, BGF_CRYSTX_FORMAT, &descriptor[0],
                    &s_a[0], &s_b[0], &s_c[0],
                    &s_alpha[0], &s_beta[0], &s_gamma[0] );

            /* Compute full volume tensor from the angles */
            Setup_Box( atof(s_a), atof(s_b), atof(s_c),
                    atof(s_alpha), atof(s_beta), atof(s_gamma),
                    &(system->box) );
        }
        else if (!strncmp( tokens[0], "CONECT", 6 ))
        {
            /* check number of restrictions */
            Check_Input_Range( token_cnt - 2, 0, MAX_RESTRICT,
                    "CONECT line exceeds max restrictions allowed.\n" );

            /* read bond restrictions */
            if ( is_Valid_Serial( workspace, bgf_serial = atoi(tokens[1]) ) )
            {
                ratom = workspace->map_serials[ bgf_serial ];
            }

            workspace->restricted[ ratom ] = token_cnt - 2;
            for ( i = 2; i < token_cnt; ++i )
            {
                if ( is_Valid_Serial( workspace, bgf_serial = atoi(tokens[i]) ) )
                {
                    workspace->restricted_list[ ratom ][ i - 2 ] =
                        workspace->map_serials[ bgf_serial ];
                }
            }

            /* fprintf( stderr, "restriction on %d:", ratom );
            for( i = 0; i < workspace->restricted[ ratom ]; ++i )
             fprintf( stderr, "  %d", workspace->restricted_list[ratom][i] );
             fprintf( stderr, "\n" ); */
        }

        /* clear previous input line */
        line[0] = 0;

        for ( i = 0; i < token_cnt; ++i )
        {
            tokens[i][0] = 0;
        }
    }
    if ( ferror( bgf ) )
    {
        fprintf( stderr, "[ERROR] Unable to read BGF file. Terminating...\n" );
        exit( INVALID_INPUT );
    }

    sfclose( bgf, "Read_BGF::bgf" );
}
