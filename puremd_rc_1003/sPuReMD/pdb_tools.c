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

#include "pdb_tools.h"
#include "box.h"
#include "list.h"
#include "param.h"
#include "restart.h"
#include "ctype.h"


int is_Valid_Serial( static_storage *workspace, int serial )
{
  if( workspace->map_serials[ serial ] < 0 )
    {
      fprintf( stderr, "CONECT line includes invalid pdb serial number %d.\n", 
	       serial );
      fprintf( stderr, "Please correct the input file.Terminating...\n" );
      exit( INVALID_INPUT );
    }
  
  return 1;
}


int Check_Input_Range( int val, int lo, int hi, char *message )
{
  if( val < lo || val > hi )
    {
      fprintf( stderr, "%s\nInput %d - Out of range %d-%d. Terminating...\n", 
	       message, val, lo, hi );
      exit( INVALID_INPUT );
    }

  return 1;
}


void Trim_Spaces( char *element )
{
  int i, j;

  for( i = 0; element[i] == ' '; ++i );  // skip initial space chars
  
  for( j = i; j < strlen(element) && element[j] != ' '; ++j )
    element[j-i] = toupper( element[j] ); // make uppercase, move to beginning
  element[j-i] = 0; // finalize the string
}


char Read_PDB( char* pdb_file, reax_system* system, control_params *control, 
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
  char s_a[10], s_b[10], s_c[10], s_alpha[9], s_beta[9], s_gamma[9];
  char s_group[12], s_zValue[9];
  char *endptr = NULL;
  int  i, c, c1, pdb_serial, ratom = 0;
  /* open pdb file */
  if ( (pdb = fopen(pdb_file, "r")) == NULL ) {
    fprintf( stderr, "Error opening the pdb file!\n" );
    exit( FILE_NOT_FOUND_ERR );
  }


  /* allocate memory for tokenizing pdb lines */
  s =   (char*)  malloc( sizeof(char)  * MAX_LINE );
  s1 =  (char*)  malloc( sizeof(char)  * MAX_LINE );
  tmp = (char**) malloc( sizeof(char*) * MAX_TOKENS );
  for( i = 0; i < MAX_TOKENS; i++ )
    tmp[i] = (char*) malloc( sizeof(char) * MAX_TOKEN_LEN );


  /* count number of atoms in the pdb file */
  system->N = 0;
  while (!feof(pdb)) {
    s[0] = 0;
    fgets( s, MAX_LINE, pdb );
    
    tmp[0][0] = 0;
    c = Tokenize( s, &tmp );
    
    if( strncmp( tmp[0], "ATOM", 4 ) == 0 || 
	strncmp( tmp[0], "HETATM", 6 ) == 0 )
      (system->N)++;
  }
  fclose(pdb);
#if defined(DEBUG_FOCUS)
  fprintf( stderr, "system->N: %d\n", system->N );
#endif

  /* memory allocations for atoms, atom maps, bond restrictions */
  system->atoms = (reax_atom*) calloc( system->N, sizeof(reax_atom) );

  workspace->map_serials = (int*) calloc( MAX_ATOM_ID, sizeof(int) );
  for( i = 0; i < MAX_ATOM_ID; ++i )
    workspace->map_serials[i] = -1;

  workspace->orig_id = (int*) calloc( system->N, sizeof(int) );
  workspace->restricted  = (int*) calloc( system->N, sizeof(int) );
  workspace->restricted_list = (int**) calloc( system->N, sizeof(int*) );
  for( i = 0; i < system->N; ++i )
    workspace->restricted_list[i] = (int*) calloc( MAX_RESTRICT, sizeof(int) );
  

  /* start reading and processing pdb file */
  pdb = fopen(pdb_file,"r");
  c = 0;
  c1 = 0;

  while (!feof(pdb)) {
    /* clear previous input line */
    s[0] = 0;
    for( i = 0; i < c1; ++i )
      tmp[i][0] = 0;
    
    /* read new line and tokenize it */
    fgets( s, MAX_LINE, pdb );
    strncpy( s1, s, MAX_LINE-1 );
    c1 = Tokenize( s, &tmp );
    
    /* process new line */
    if( strncmp(tmp[0],"ATOM",4) == 0 || strncmp(tmp[0],"HETATM", 6) == 0 ) {
      if( strncmp(tmp[0],"ATOM",4) == 0 ) {  
	strncpy( &descriptor[0], s1, 6 );     descriptor[6] = 0;
	strncpy( &serial[0], s1+6, 5 );       serial[5] = 0;
	strncpy( &atom_name[0], s1+12, 4 );   atom_name[4] = 0;
	alt_loc = s1[16];
	strncpy( &res_name[0], s1+17, 3 );    res_name[3] = 0;
	chain_id = s1[21];
	strncpy( &res_seq[0], s1+22, 4 );     res_seq[4] = 0;
	icode = s1[26];
	strncpy( &s_x[0], s1+30, 8 );         s_x[8] = 0;
	strncpy( &s_y[0], s1+38, 8 );         s_y[8] = 0;
	strncpy( &s_z[0], s1+46, 8 );         s_z[8] = 0;
	strncpy( &occupancy[0], s1+54, 6 );   occupancy[6] = 0;
	strncpy( &temp_factor[0], s1+60, 6 ); temp_factor[6] = 0;
	strncpy( &seg_id[0], s1+72, 4 );      seg_id[4] = 0;
	strncpy( &element[0], s1+76, 2 );     element[2] = 0;
	strncpy( &charge[0], s1+78, 2 );      charge[2] = 0;	
      }
      else if (strncmp(tmp[0],"HETATM", 6) == 0) {	  
	strncpy( &descriptor[0], s1, 6 );     descriptor[6] = 0;
	strncpy( &serial[0], s1+6, 5 );       serial[5] = 0;
	strncpy( &atom_name[0], s1+12, 4 );   atom_name[4] = 0;
	alt_loc = s1[16];
	strncpy( &res_name[0], s1+17, 3 );    res_name[3] = 0;
	chain_id = s1[21];
	strncpy( &res_seq[0], s1+22, 4 );     res_seq[4] = 0;
	icode = s1[26];
	strncpy( &s_x[0], s1+30, 8 );         s_x[8] = 0;
	strncpy( &s_y[0], s1+38, 8 );         s_y[8] = 0;
	strncpy( &s_z[0], s1+46, 8 );         s_z[8] = 0;
	strncpy( &occupancy[0], s1+54, 6 );   occupancy[6] = 0;
	strncpy( &temp_factor[0], s1+60, 6 ); temp_factor[6] = 0;
	//strncpy( &seg_id[0], s1+72, 4 );      seg_id[4] = 0;
	strncpy( &element[0], s1+76, 2 );     element[2] = 0;
	strncpy( &charge[0], s1+78, 2 );      charge[2] = 0;
      }


      /* add to mapping */
      pdb_serial = strtod( &serial[0], &endptr );
      Check_Input_Range( pdb_serial, 0, MAX_ATOM_ID, "Invalid pdb_serial" );
      workspace->map_serials[ pdb_serial ] = c;
      workspace->orig_id[ c ] = pdb_serial;
      // fprintf( stderr, "map %d --> %d\n", pdb_serial, c );


      /* copy atomic positions */
      system->atoms[c].x[0] = strtod( &s_x[0], &endptr );
      system->atoms[c].x[1] = strtod( &s_y[0], &endptr );
      system->atoms[c].x[2] = strtod( &s_z[0], &endptr );
	  
      /* atom name and type */
      strcpy( system->atoms[c].name, atom_name );
      Trim_Spaces( element );
      system->atoms[c].type = Get_Atom_Type( &(system->reaxprm), element );
	  	  	 
      /* fprintf( stderr, 
	 "%d%8.3f%8.3f%8.3fq:%8.3f occ:%s temp:%s seg_id:%s element:%s\n", 
	 system->atoms[c].type, 
	 system->atoms[c].x[0], system->atoms[c].x[1], system->atoms[c].x[2],
	 system->atoms[c].q, occupancy, temp_factor, seg_id, element ); */
      c++;
    }
    else if(!strncmp( tmp[0], "CRYST1", 6 )) {
      sscanf( s1, PDB_CRYST1_FORMAT,
	      &descriptor[0],
	      &s_a[0],
	      &s_b[0],
	      &s_c[0],
	      &s_alpha[0],
	      &s_beta[0],
	      &s_gamma[0],
	      &s_group[0],
	      &s_zValue[0] );

      /* Compute full volume tensor from the angles */
      Init_Box_From_CRYST( atof(s_a),  atof(s_b), atof(s_c), 
			   atof(s_alpha), atof(s_beta), atof(s_gamma), 
			   &(system->box) );
    }

    /* IMPORTANT: We do not check for the soundness of restrictions here. 
       When atom2 is on atom1's restricted list, and there is a restriction on
       atom2, then atom1 has to be on atom2's restricted list, too. However, 
       we do not check if this is the case in the input file, 
       this is upto the user. */
    else if(!strncmp( tmp[0], "CONECT", 6 )) {
      /* error check */
      //fprintf(stderr, "CONECT: %d\n", c1 );
      Check_Input_Range( c1 - 2, 0, MAX_RESTRICT,
			 "CONECT line exceeds max restrictions allowed.\n" );

      /* read bond restrictions */
      if( is_Valid_Serial( workspace, pdb_serial = atoi(tmp[1]) ) )
	ratom = workspace->map_serials[ pdb_serial ];

      workspace->restricted[ ratom ] = c1 - 2;
      for( i = 2; i < c1; ++i )
	{
	  if( is_Valid_Serial( workspace, pdb_serial = atoi(tmp[i]) ) )
	    workspace->restricted_list[ ratom ][ i-2 ] = 
	      workspace->map_serials[ pdb_serial ];
	}
	  
      /* fprintf( stderr, "restriction on %d:", ratom );
	 for( i = 0; i < workspace->restricted[ ratom ]; ++i )
	 fprintf( stderr, "  %d", workspace->restricted_list[ratom][i] );
	 fprintf( stderr, "\n" ); */
    }      
  }

  fclose(pdb);

#if defined(DEBUG_FOCUS)
  fprintf( stderr, "pdb file read\n" );
#endif

  return 1;
}


char Write_PDB( reax_system* system, control_params *control, 
		simulation_data *data, static_storage *workspace, 
		list* bonds, output_controls *out_control )
{
  int  i, j, k, count;
  int  connect[4];
  char temp[MAX_STR], name[10];
  real bo;
  real alpha, beta, gamma;
  

  /* open output pdb file */
  sprintf( temp, "%s%d.pdb", control->sim_name, data->step );
  out_control->pdb = fopen( temp, "w" );


  /* Writing Box information */
  /* Write full volume tensor from the angles (as soon as possible) TODO_SOON */
  gamma = acos( (system->box.box[0][0] * system->box.box[1][0] +  
		 system->box.box[0][1] * system->box.box[1][1] + 
		 system->box.box[0][2] * system->box.box[1][2]) / 
		(system->box.box_norms[0]*system->box.box_norms[1]));
  beta  = acos( (system->box.box[0][0] * system->box.box[2][0] +  
		 system->box.box[0][1] * system->box.box[2][1] + 
		 system->box.box[0][2] * system->box.box[2][2]) / 
		(system->box.box_norms[0]*system->box.box_norms[2]));
  alpha = acos( (system->box.box[2][0] * system->box.box[1][0] +  
		 system->box.box[2][1] * system->box.box[1][1] + 
		 system->box.box[2][2] * system->box.box[1][2]) / 
		(system->box.box_norms[2]*system->box.box_norms[1]));

  fprintf(out_control->pdb,PDB_CRYST1_FORMAT_O,
	  "CRYST1",
	  system->box.box_norms[0],
	  system->box.box_norms[1],
	  system->box.box_norms[2],
	  RAD2DEG(alpha),
	  RAD2DEG(beta),
	  RAD2DEG(gamma),
	  " ",
	  0);
  fprintf( out_control->log, "Box written\n" ); fflush( out_control->log );

  /* Writing atom information */
  for (i=0; i < system->N; i++) {
    strncpy( name, system->reaxprm.sbp[system->atoms[i].type].name, 2 );
    name[2] = '\0';
    fprintf( out_control->pdb,PDB_ATOM_FORMAT_O,
	     "ATOM  ",
	     workspace->orig_id[i],
	     name,
	     ' ',
	     "REX",
	     ' ',
	     1,
	     ' ',
	     system->atoms[i].x[0],
	     system->atoms[i].x[1],
	     system->atoms[i].x[2],
	     1.0, 
	     0.0,
	     "0",
	     name,
	     "  " );
  }

  fprintf( out_control->log, "ATOM written\n" ); fflush( out_control->log );
  
  /* Writing connect information */
  for(i=0; i < system->N; i++) {
    count = 0;

    for(j = Start_Index(i, bonds); j < End_Index(i, bonds); ++j) {
      bo = bonds->select.bond_list[j].bo_data.BO;
      if (bo > 0.3)
	{
	  connect[count]=workspace->orig_id[bonds->select.bond_list[j].nbr];
	  count++;
	}	
    }

    fprintf( out_control->pdb, "%6s%6d", "CONECT", workspace->orig_id[i] );
    for( k=0; k < count; k++ )
      fprintf( out_control->pdb, "%6d", connect[k] );
    fprintf( out_control->pdb, "\n" );
  }

  fprintf( out_control->pdb, "END\n" );

  fclose( out_control->pdb );

  return 1;
}


char Read_BGF( char* bgf_file, reax_system* system, control_params *control, 
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
  int  i, atom_cnt, token_cnt, bgf_serial, ratom = 0;

  /* open biograf file */
  if ( (bgf = fopen( bgf_file, "r" )) == NULL )
    {
      fprintf( stderr, "Error opening the bgf file!\n" );
      exit( FILE_NOT_FOUND_ERR );
    }


  /* allocate memory for tokenizing biograf file lines */
  line   = (char*)  malloc( sizeof(char)  * MAX_LINE );
  backup = (char*)  malloc( sizeof(char)  * MAX_LINE );
  tokens = (char**) malloc( sizeof(char*) * MAX_TOKENS );
  for( i = 0; i < MAX_TOKENS; i++ )
    tokens[i] = (char*) malloc( sizeof(char) * MAX_TOKEN_LEN );


  /* count number of atoms in the pdb file */
  system->N = 0;
  while( !feof( bgf ) ) {
    line[0] = 0;
    fgets( line, MAX_LINE, bgf );
      
    tokens[0][0] = 0;
    token_cnt = Tokenize( line, &tokens );
      
    if( !strcmp( tokens[0], "ATOM" ) || !strcmp( tokens[0], "HETATM" ) )
      (system->N)++;
  }
  //fprintf( stderr, "system->N: %d\n", system->N );
  fclose( bgf );


  /* memory allocations for atoms, atom maps, bond restrictions */
  system->atoms = (reax_atom*) calloc( system->N, sizeof(reax_atom) );

  workspace->map_serials = (int*) calloc( MAX_ATOM_ID, sizeof(int) );
  for( i = 0; i < MAX_ATOM_ID; ++i )
    workspace->map_serials[i] = -1;

  workspace->orig_id = (int*) calloc( system->N, sizeof(int) );
  workspace->restricted  = (int*) calloc( system->N, sizeof(int) );
  workspace->restricted_list = (int**) calloc( system->N, sizeof(int*) );
  for( i = 0; i < system->N; ++i )
    workspace->restricted_list[i] = (int*) calloc( MAX_RESTRICT, sizeof(int) );
  

  /* start reading and processing pdb file */
  bgf = fopen( bgf_file, "r" );
  atom_cnt = 0;
  token_cnt = 0;

  while( !feof( bgf ) ) {
    /* clear previous input line */
    line[0] = 0;
    for( i = 0; i < token_cnt; ++i )
      tokens[i][0] = 0;
	  
    /* read new line and tokenize it */
    fgets( line, MAX_LINE, bgf );
    strncpy( backup, line, MAX_LINE-1 );
    token_cnt = Tokenize( line, &tokens );

    /* process new line */
    if( !strncmp(tokens[0], "ATOM", 4) || !strncmp(tokens[0], "HETATM", 6) ) {
      if( !strncmp(tokens[0], "ATOM", 4) ) {
	strncpy( &descriptor[0], backup, 6 );     descriptor[6] = 0;
	strncpy( &serial[0], backup+7, 5 );       serial[5] = 0;
	strncpy( &atom_name[0], backup+13, 5 );   atom_name[5] = 0;
	strncpy( &res_name[0], backup+19, 3 );    res_name[3] = 0;
	chain_id = backup[23];
	strncpy( &res_seq[0], backup+25, 5 );     res_seq[5] = 0;	      
	strncpy( &s_x[0], backup+30, 10 );        s_x[10] = 0;
	strncpy( &s_y[0], backup+40, 10 );        s_y[10] = 0;
	strncpy( &s_z[0], backup+50, 10 );        s_z[10] = 0;
	strncpy( &element[0], backup+61, 5 );     element[5] = 0;
	strncpy( &occupancy[0], backup+66, 3 );   occupancy[3] = 0;
	strncpy( &temp_factor[0], backup+69, 2 ); temp_factor[2] = 0;
	strncpy( &charge[0], backup+72, 8 );      charge[8] = 0;
      }
      else if( !strncmp(tokens[0],"HETATM", 6) ) {
	/* bgf hetatm:
	   (7x,i5,1x,a5,1x,a3,1x,a1,1x,a5,3f10.5,1x,a5,i3,i2,1x,f8.5) */
	strncpy( &descriptor[0], backup, 6 );     descriptor[6] = 0;
	strncpy( &serial[0], backup+7, 5 );       serial[5] = 0;
	strncpy( &atom_name[0], backup+13, 5 );   atom_name[5] = 0;
	strncpy( &res_name[0], backup+19, 3 );    res_name[3] = 0;
	chain_id = backup[23];
	strncpy( &res_seq[0], backup+25, 5 );     res_seq[5] = 0;	      
	strncpy( &s_x[0], backup+30, 10 );        s_x[10] = 0;
	strncpy( &s_y[0], backup+40, 10 );        s_y[10] = 0;
	strncpy( &s_z[0], backup+50, 10 );        s_z[10] = 0;
	strncpy( &element[0], backup+61, 5 );     element[5] = 0;
	strncpy( &occupancy[0], backup+66, 3 );   occupancy[3] = 0;
	strncpy( &temp_factor[0], backup+69, 2 ); temp_factor[2] = 0;
	strncpy( &charge[0], backup+72, 8 );      charge[8] = 0;
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
      strcpy( system->atoms[atom_cnt].name, atom_name );
      Trim_Spaces( element );
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
    else if(!strncmp( tokens[0], "CRYSTX", 6 )) {
      sscanf( backup, BGF_CRYSTX_FORMAT,
	      &descriptor[0],
	      &s_a[0],
	      &s_b[0],
	      &s_c[0],
	      &s_alpha[0],
	      &s_beta[0],
	      &s_gamma[0] );

      /* Compute full volume tensor from the angles */
      Init_Box_From_CRYST( atof(s_a),  atof(s_b), atof(s_c), 
			   atof(s_alpha), atof(s_beta), atof(s_gamma), 
			   &(system->box) );
    }
    else if(!strncmp( tokens[0], "CONECT", 6 )) {
      /* check number of restrictions */
      Check_Input_Range( token_cnt - 2, 0, MAX_RESTRICT,
			 "CONECT line exceeds max restrictions allowed.\n" );

      /* read bond restrictions */
      if( is_Valid_Serial( workspace, bgf_serial = atoi(tokens[1]) ) )
	ratom = workspace->map_serials[ bgf_serial ];

      workspace->restricted[ ratom ] = token_cnt - 2;
      for( i = 2; i < token_cnt; ++i )
	if( is_Valid_Serial( workspace, bgf_serial = atoi(tokens[i]) ) )
	  workspace->restricted_list[ ratom ][ i-2 ] = 
	    workspace->map_serials[ bgf_serial ];
	  
      /* fprintf( stderr, "restriction on %d:", ratom );
	 for( i = 0; i < workspace->restricted[ ratom ]; ++i )
	 fprintf( stderr, "  %d", workspace->restricted_list[ratom][i] );
	 fprintf( stderr, "\n" ); */
    }      
  }

  fclose( bgf );

#if defined(DEBUG_FOCUS)
  fprintf( stderr, "bgf file read\n" );
#endif

  return 1;
} 
