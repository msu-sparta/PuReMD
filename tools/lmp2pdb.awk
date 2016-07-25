BEGIN{
    box_flag = 0;
    natoms = -1;
    ntypes = -1;
    atom_style = "charge";
}
{
    # num atoms
    if( $2 == "atoms" )
	natoms = $1;
    # num atom types
    else if( $2 == "atom" && $3 == "types" )
	ntypes = $1;
    # box geometry 
    else if( $3 == "xlo" && $4 == "xhi" ) {
	box[0] = $2 - $1;
	++box_flag;
    }
    else if( $3 == "ylo" && $4 == "yhi" ) {
	box[1] = $2 - $1;
	++box_flag;
    }
    else if( $3 == "zlo" && $4 == "zhi" ) {
	box[2] = $2 - $1;
	++box_flag;
    }
    # atom masses
    else if( $1 == "Masses" ) {
	if( ntypes <= 0 ) {
	    printf( "number of atom types can not be %d!\n", ntypes ); 
	    exit;
	}
	
	getline; # skip the empty line
	for( i = 0; i < ntypes; ++i ) {
	    getline;
	    if( NF != 2 ) { # expect one integer, one float
		printf( "unexpected mass line format: %s!\n", $0 );
		exit;
	    }
	    
	    # record the atom type
	    if( $2 == 12.0000 )
		types[$1] = "C";
	    else if( $2 == 1.0080 )
		types[$1] = "H";
	    else if( $2 == 15.9990 )
		types[$1] = "O";
	    else if( $2 == 14.0000 )
		types[$1] = "N";
	    else {
		printf( "unknown atom type!\n" );
		exit;
	    }
	}
    }
    # atom info
    else if( $1 == "Atoms" ) {
	if( natoms <= 0 ) {
	    printf( "number of atoms can not be %d!\n", natoms ); 
	    exit;
	}
	
	getline; # skip the empty line
	for( i = 0; i < natoms; ++i ) {
	    getline;
	    if( NF != 6 ) { # expect 3 ints, 3 floats
		printf( "unexpected atom line format: %s!\n", $0 );
		exit;
	    }
	    
	    atoms[i,"serial"] = $1;
	    atoms[i,"type"] = types[$2];
	    atoms[i,"q"] = $3;
	    atoms[i,"x"] = $4;
	    atoms[i,"y"] = $5;
	    atoms[i,"z"] = $6;
	}
    }
    else if( $1 == "#" )
	1; # skip the comment
    else if( NF == 0 )
	1; # skip the empty line
    else {
	printf( "unexpected line: %s\n", $0 );
	exit;
    }
}
END{
    if( box_flag != 3 ) {
	printf( "incorrect box geometry!\n" );
	exit;
    }
    
    printf( "%6s%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f%11s%4d\n",
	    "CRYST1", box[0], box[1], box[2], 90, 90, 90, "P", 1 );
    
    for( i = 0; i < natoms; ++i ) {
	printf( "%-6s%5d%5s%c%3s %c%4d%c   %8.3f%8.3f%8.3f%6.2f%6.2f      "\
		"%-4s%2s%2s\n", "ATOM", atoms[i,"serial"], atoms[i,"type"], " ",
		"REX", " ", 1, " ", atoms[i,"x"], atoms[i,"y"],	atoms[i,"z"], 
		1.0, 0.0, "0", atoms[i,"type"], "  " );
    }
    
    printf( "END\n" );
}