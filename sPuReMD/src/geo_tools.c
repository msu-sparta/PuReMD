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
    char element[3], name[9], line[MAX_LINE];

    /* total number of atoms */
    fscanf( geo, " %d", &system->N );

    /* count atoms */
    for ( i = 0; i < system->N; ++i )
    {
        fscanf( geo, CUSTOM_ATOM_FORMAT,
                &serial, element, name, &x[0], &x[1], &x[2] );
        Fit_to_Periodic_Box( &system->box, x );
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
    char line[MAX_LINE];
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
     * initialize the big box */
    while ( fgets( line, MAX_LINE, geo ) )
    {
        if ( strncmp( line, cryst, cryst_len ) == 0 )
        {
            if ( geo_format == PDB )
            {
                sscanf( line, PDB_CRYST1_FORMAT,
                        descriptor, s_a, s_b, s_c,
                        s_alpha, s_beta, s_gamma, s_group, s_zValue );

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
        Fit_to_Periodic_Box( &system->box, x );

#if defined(DEBUG)
        fprintf( stderr, "atom%d: %s %s %f %f %f\n",
                 serial, element, name, x[0], x[1], x[2] );
#endif

        atom = &system->atoms[top];
        workspace->orig_id[i] = serial;
        atom->type = Get_Atom_Type( &system->reax_param, element, sizeof(element) );
        strncpy( atom->name, name, sizeof(atom->name) - 1 );
        atom->name[sizeof(atom->name) - 1] = '\0';
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
    char line[MAX_LINE];
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

            strncpy( s_x, line + 30, sizeof(s_x) - 1 );
            s_x[sizeof(s_x) - 1] = '\0';
            strncpy( s_y, line + 38, sizeof(s_y) - 1 );
            s_y[sizeof(s_y) - 1] = '\0';
            strncpy( s_z, line + 46, sizeof(s_z) - 1 );
            s_z[sizeof(s_z) - 1] = '\0';

            Make_Point( strtod( s_x, &endptr ), strtod( s_y, &endptr ),
                    strtod( s_z, &endptr ), &x );

            Fit_to_Periodic_Box( &(system->box), x );
        }
    }

#if defined(DEBUG)
    fprintf( stderr, "[INFO] count atoms:\n" );
    fprintf( stderr, "N = %d\n\n", system->N );
#endif
}


void Read_PDB( const char * const pdb_file, reax_system* system, control_params *control,
        simulation_data *data, static_storage *workspace )
{
    int i, c, c1, pdb_serial, top;
    FILE *pdb;
    char **tmp;
    char *s, *s1;
    char descriptor[7], serial[6];
    char atom_name[5], res_name[4], res_seq[5];
    char s_x[9], s_y[9], s_z[9];
    char occupancy[7], temp_factor[7];
    char seg_id[5], element[3], charge[3];
    char alt_loc, chain_id, icode;
    char *endptr;
    rvec x;
    reax_atom *atom;

    endptr = NULL;

    /* open pdb file */
    pdb = sfopen( pdb_file, "r" );

    /* allocate memory for tokenizing pdb lines */
    Allocate_Tokenizer_Space( &s, MAX_LINE, &s1, MAX_LINE,
            &tmp, MAX_TOKENS, MAX_TOKEN_LEN );

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
        s1[MAX_LINE - 1] = '\0';
        c1 = Tokenize( s, &tmp, MAX_TOKEN_LEN );

        /* process new line */
        if ( strncmp( tmp[0], "ATOM", 4 ) == 0
                || strncmp( tmp[0], "HETATM", 6 ) == 0 )
        {
            if ( strncmp( tmp[0], "ATOM", 4 ) == 0 )
            {
                strncpy( descriptor, s1, sizeof(descriptor) - 1 );
                descriptor[sizeof(descriptor) - 1] = '\0';
                strncpy( serial, s1 + 6, sizeof(serial) - 1 );
                serial[sizeof(serial) - 1] = '\0';
                strncpy( atom_name, s1 + 12, sizeof(atom_name) - 1 );
                atom_name[sizeof(atom_name) - 1] = '\0';
                alt_loc = s1[16];
                strncpy( res_name, s1 + 17, sizeof(res_name) - 1 );
                res_name[sizeof(res_name) - 1] = '\0';
                chain_id = s1[21];
                strncpy( res_seq, s1 + 22, sizeof(res_seq) - 1 );
                res_seq[sizeof(res_seq) - 1] = '\0';
                icode = s1[26];
                strncpy( s_x, s1 + 30, sizeof(s_x) - 1 );
                s_x[sizeof(s_x) - 1] = '\0';
                strncpy( s_y, s1 + 38, sizeof(s_y) - 1 );
                s_y[sizeof(s_y) - 1] = '\0';
                strncpy( s_z, s1 + 46, sizeof(s_z) - 1 );
                s_z[sizeof(s_z) - 1] = '\0';
                strncpy( occupancy, s1 + 54, sizeof(occupancy) - 1 );
                occupancy[sizeof(occupancy) - 1] = '\0';
                strncpy( temp_factor, s1 + 60, sizeof(temp_factor) - 1 );
                temp_factor[sizeof(temp_factor) - 1] = '\0';
                strncpy( seg_id, s1 + 72, sizeof(seg_id) - 1 );
                seg_id[sizeof(seg_id) - 1] = '\0';
                strncpy( element, s1 + 76, sizeof(element) - 1 );
                element[sizeof(element) - 1] = '\0';
                strncpy( charge, s1 + 78, sizeof(charge) - 1 );
                charge[sizeof(charge) - 1] = '\0';
            }
            else if ( strncmp( tmp[0], "HETATM", 6 ) == 0 )
            {
                strncpy( descriptor, s1, sizeof(descriptor) - 1 );
                descriptor[sizeof(descriptor) - 1] = '\0';
                strncpy( serial, s1 + 6, sizeof(serial) - 1 );
                serial[sizeof(serial) - 1] = '\0';
                strncpy( atom_name, s1 + 12, sizeof(atom_name) - 1 );
                atom_name[sizeof(atom_name) - 1] = '\0';
                alt_loc = s1[16];
                strncpy( res_name, s1 + 17, sizeof(res_name) - 1 );
                res_name[sizeof(res_name) - 1] = '\0';
                chain_id = s1[21];
                strncpy( res_seq, s1 + 22, sizeof(res_seq) - 1 );
                res_seq[sizeof(res_seq) - 1] = '\0';
                icode = s1[26];
                strncpy( s_x, s1 + 30, sizeof(s_x) - 1 );
                s_x[sizeof(s_x) - 1] = '\0';
                strncpy( s_y, s1 + 38, sizeof(s_y) - 1 );
                s_y[sizeof(s_y) - 1] = '\0';
                strncpy( s_z, s1 + 46, sizeof(s_z) - 1 );
                s_z[sizeof(s_z) - 1] = '\0';
                strncpy( occupancy, s1 + 54, sizeof(occupancy) - 1 );
                occupancy[sizeof(occupancy) - 1] = '\0';
                strncpy( temp_factor, s1 + 60, sizeof(temp_factor) - 1 );
                temp_factor[sizeof(temp_factor) - 1] = '\0';
                strncpy( element, s1 + 76, sizeof(element) - 1 );
                element[sizeof(element) - 1] = '\0';
                strncpy( charge, s1 + 78, sizeof(charge) - 1 );
                charge[sizeof(charge) - 1] = '\0';
            }

            /* if the point is inside my_box, add it to my lists */
            Make_Point( strtod( s_x, &endptr ), strtod( s_y, &endptr ),
                    strtod( s_z, &endptr ), &x );

            Fit_to_Periodic_Box( &system->box, x );

            if ( is_Inside_Box( &system->box, x ) )
            {
                /* store orig_id, type, name and coord info of the new atom */
                atom = &system->atoms[top];
                pdb_serial = (int) strtod( serial, &endptr );
                workspace->orig_id[top] = pdb_serial;

                Trim_Spaces( element, sizeof(element) );
                atom->type = Get_Atom_Type( &system->reax_param, element, sizeof(element) );
                strncpy( atom->name, atom_name, sizeof(atom->name) - 1 );
                atom->name[sizeof(atom->name) - 1] = '\0';

                rvec_Copy( atom->x, x );
                rvec_MakeZero( atom->v );
                rvec_MakeZero( atom->f );
                atom->q = 0;

                top++;

#if defined(DEBUG)
                fprintf( stderr, "[INFO] atom: id = %d, name = %s, serial = %d, type = %d, ", top, atom->name, pdb_serial, atom->type );
                fprintf( stderr, "x = %7.3f, %7.3f, %7.3f\n", atom->x[0], atom->x[1], atom->x[2] );
                fflush( stderr );
#endif
            }
            c++;
        }

        /* IMPORTANT: We do not check for the soundness of restrictions here.
         * When atom2 is on atom1's restricted list, and there is a restriction
         * on atom2, then atom1 has to be on atom2's restricted list, too.
         * However, we do not check if this is the case in the input file,
         * this is upto the user. */
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
    char name[3];
    real alpha, beta, gamma;
    rvec x;
    reax_atom *p_atom;
    char fname[MAX_STR];
    char buffer[PDB_ATOM_FORMAT_O_LENGTH + 1];
    FILE *pdb;

    /* Writing Box information */
    gamma = ACOS( (system->box.box[0][0] * system->box.box[1][0]
                + system->box.box[0][1] * system->box.box[1][1]
                + system->box.box[0][2] * system->box.box[1][2])
            / (system->box.box_norms[0] * system->box.box_norms[1]) );
    beta  = ACOS( (system->box.box[0][0] * system->box.box[2][0]
                + system->box.box[0][1] * system->box.box[2][1]
                + system->box.box[0][2] * system->box.box[2][2])
            / (system->box.box_norms[0] * system->box.box_norms[2]) );
    alpha = ACOS( (system->box.box[2][0] * system->box.box[1][0]
                + system->box.box[2][1] * system->box.box[1][1]
                + system->box.box[2][2] * system->box.box[1][2])
            / (system->box.box_norms[2] * system->box.box_norms[1]) );

    /* write header */
    snprintf( fname, MAX_STR + 9, "%s-%d.pdb", control->sim_name, data->step );
    pdb = sfopen( fname, "w" );
    fprintf( pdb, PDB_CRYST1_FORMAT_O,
             "CRYST1",
             system->box.box_norms[0], system->box.box_norms[1],
             system->box.box_norms[2],
             RAD2DEG(alpha), RAD2DEG(beta), RAD2DEG(gamma), " ", 0 );

    /* write atom lines to buffer */
    for ( i = 0; i < system->N; i++ )
    {
        p_atom = &system->atoms[i];

        strncpy( name, p_atom->name, sizeof(name) - 1 );
        name[sizeof(name) - 1] = '\0';
        Trim_Spaces( name, sizeof(name) );

        memcpy( x, p_atom->x, 3 * sizeof(real) );
        Fit_to_Periodic_Box( &system->box, x );

        snprintf( buffer, PDB_ATOM_FORMAT_O_LENGTH, PDB_ATOM_FORMAT_O,
                "ATOM  ", workspace->orig_id[i], p_atom->name, ' ', "REX", ' ', 1, ' ',
                x[0], x[1], x[2], 1.0, 0.0, "0", name, "  " );

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
    char descriptor[7], serial[6];
    char atom_name[6], res_name[4], res_seq[6];
    char s_x[11], s_y[11], s_z[11];
    char occupancy[4], temp_factor[3];
    char element[6], charge[9];
    char chain_id;
    char s_a[12], s_b[12], s_c[12], s_alpha[12], s_beta[12], s_gamma[12];
    char *endptr;
    int i, atom_cnt, token_cnt, bgf_serial, ratom, crystx_found;

    endptr = NULL;
    ratom = 0;
    crystx_found = FALSE;

    bgf = sfopen( bgf_file, "r" );

    line = smalloc( sizeof(char) * MAX_LINE, "Read_BGF::line" );
    backup = smalloc( sizeof(char) * MAX_LINE, "Read_BGF::backup" );
    tokens = smalloc( sizeof(char*) * MAX_TOKENS, "Read_BGF::tokens" );
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        tokens[i] = smalloc( sizeof(char) * MAX_TOKEN_LEN,
               "Read_BGF::tokens[i]" );
    }

    /* count number of atoms in the BGF file */
    system->N = 0;
    line[0] = 0;

    while ( fgets( line, MAX_LINE, bgf ) )
    {
        tokens[0][0] = 0;
        token_cnt = Tokenize( line, &tokens, MAX_TOKEN_LEN );

        if ( strncmp( tokens[0], "ATOM", 4 ) == 0
                || strncmp( tokens[0], "HETATM", 6 ) == 0 )
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

    sfclose( bgf, "Read_BGF::bgf" );

    system->atoms = scalloc( system->N, sizeof(reax_atom),
            "Read_BGF::system->atoms" );
    workspace->map_serials = scalloc( MAX_ATOM_ID, sizeof(int),
            "Read_BGF::workspace->map_serials" );
    for ( i = 0; i < MAX_ATOM_ID; ++i )
    {
        workspace->map_serials[i] = -1;
    }

    workspace->orig_id = scalloc( system->N, sizeof(int),
            "Read_BGF::workspace->orig_id" );
    workspace->restricted  = scalloc( system->N, sizeof(int),
            "Read_BGF::workspace->restricted" );
    workspace->restricted_list = scalloc( system->N, sizeof(int*),
            "Read_BGF::workspace->restricted_list" );
    for ( i = 0; i < system->N; ++i )
    {
        workspace->restricted_list[i] = scalloc( MAX_RESTRICT, sizeof(int),
                "Read_BGF::workspace->restricted_list[i]" );
    }

    bgf = sfopen( bgf_file, "r" );
    atom_cnt = 0;
    token_cnt = 0;

    while ( fgets( line, MAX_LINE, bgf ) )
    {
        /* read new line and tokenize it */
        strncpy( backup, line, MAX_LINE - 1 );
        backup[MAX_LINE - 1] = '\0';
        token_cnt = Tokenize( line, &tokens, MAX_TOKEN_LEN );

        /* process new line */
        if ( strncmp( tokens[0], "ATOM", 4 ) == 0
                || strncmp( tokens[0], "HETATM", 6 ) == 0 )
        {
            if ( strncmp( tokens[0], "ATOM", 4 ) == 0 )
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
            }
            else if ( strncmp( tokens[0], "HETATM", 6 ) == 0 )
            {
                /* bgf hetatm:
                   (7x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5) */
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
                strncpy( s_z, backup + 50, sizeof(s_z) - 1 );
                s_z[sizeof(s_z) - 1] = '\0';
                strncpy( element, backup + 61, sizeof(element) - 1 );
                element[sizeof(element) - 1] = '\0';
                strncpy( occupancy, backup + 66, sizeof(occupancy) - 1 );
                occupancy[sizeof(occupancy) - 1] = '\0';
                strncpy( temp_factor, backup + 69, sizeof(temp_factor) - 1 );
                temp_factor[sizeof(temp_factor) - 1] = '\0';
                strncpy( charge, backup + 72, sizeof(charge) - 1 );
                charge[sizeof(charge) - 1] = '\0';
            }

            /* add to mapping */
            bgf_serial = strtod( serial, &endptr );
            Check_Input_Range( bgf_serial, 0, MAX_ATOM_ID, "[ERROR] Invalid bgf serial" );
            workspace->map_serials[ bgf_serial ] = atom_cnt;
            workspace->orig_id[ atom_cnt ] = bgf_serial;
            // fprintf( stderr, "map %d --> %d\n", bgf_serial, atom_cnt );

            /* copy atomic positions */
            system->atoms[atom_cnt].x[0] = strtod( s_x, &endptr );
            system->atoms[atom_cnt].x[1] = strtod( s_y, &endptr );
            system->atoms[atom_cnt].x[2] = strtod( s_z, &endptr );

            /* atom name and type */
            strncpy( system->atoms[atom_cnt].name, atom_name,
                    sizeof(system->atoms[atom_cnt].name) - 1 );
            system->atoms[atom_cnt].name[sizeof(system->atoms[atom_cnt].name) - 1] = '\0';
            Trim_Spaces( element, sizeof(element) );
            system->atoms[atom_cnt].type =
                Get_Atom_Type( &system->reax_param, element, sizeof(element) );

            /* fprintf( stderr,
            "a:%3d(%1d) c:%10.5f%10.5f%10.5f q:%10.5f occ:%s temp:%s seg_id:%s element:%s\n",
             atom_cnt, system->atoms[ atom_cnt ].type,
             system->atoms[ atom_cnt ].x[0],
             system->atoms[ atom_cnt ].x[1], system->atoms[ atom_cnt ].x[2],
             system->atoms[ atom_cnt ].q, occupancy, temp_factor,
             seg_id, element ); */

            atom_cnt++;
        }
        else if ( strncmp( tokens[0], "CRYSTX", 6 ) == 0 )
        {
            sscanf( backup, BGF_CRYSTX_FORMAT, descriptor,
                    s_a, s_b, s_c, s_alpha, s_beta, s_gamma );

            /* Compute full volume tensor from the angles */
            Setup_Box( atof(s_a), atof(s_b), atof(s_c),
                    atof(s_alpha), atof(s_beta), atof(s_gamma),
                    &system->box );

            crystx_found = TRUE;
        }
        else if ( strncmp( tokens[0], "CONECT", 6 ) == 0 )
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
                    workspace->restricted_list[ratom][i - 2] =
                        workspace->map_serials[bgf_serial];
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

    if ( crystx_found == FALSE )
    {
        fprintf( stderr, "[ERROR] improperly formatted BGF file (no CRYSTX keyword found). Terminating...\n" );
        exit( INVALID_INPUT );
    }

    sfree( line, "Read_BGF::line" );
    sfree( backup, "Read_BGF::backup" );
    for ( i = 0; i < MAX_TOKENS; i++ )
    {
        sfree( tokens[i], "Read_BGF::tokens[i]" );
    }
    sfree( tokens, "Read_BGF::tokens" );

    sfclose( bgf, "Read_BGF::bgf" );
}
