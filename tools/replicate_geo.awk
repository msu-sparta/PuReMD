# replicates an orthogonal box as many times as given
# by replicate_x, replicate_y, replicate_z parameters
# while reading the pdb file, some problems may arise
# if the fields are not separated as expected. 
# make sure to check for those if output is not as expected.

BEGIN{
    replicate_x = 1;
    replicate_y = 1;
    replicate_z = 1;
    
    num_atoms = 0;
}
{
    # collect atom info
    if( $1 == "BOXGEO" ) {
	box[0] = $2;
	box[1] = $3;
	box[2] = $4;
	
	box_ang[0] = $5;
	box_ang[1] = $6;
	box_ang[2] = $7;
    }
    else if( NF == 6 ) {
	atom_id[num_atoms] = $1; 
	atom_type[num_atoms] = $2;
	atom_name[num_atoms] = $3;
	atom_pos[num_atoms, 0] = $4;
	atom_pos[num_atoms, 1] = $5;
	atom_pos[num_atoms, 2] = $6;
	
	++num_atoms;
    }
}
END{
    new_atoms = 0;

    printf( "%6s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f\n",
	    "BOXGEO", box[0]*replicate_x, box[1]*replicate_y, box[2]*replicate_z, 
	    box_ang[0], box_ang[1], box_ang[2] );

    for( i = 0; i < replicate_x; ++i )
	for( j = 0; j < replicate_y; ++j )
	    for( k = 0; k < replicate_z; ++k ){
		base_x = i * box[0];
		base_y = j * box[1];
		base_z = k * box[2];

                for( n = 0; n < num_atoms; ++n ) {
		    ref = new_atoms % num_atoms;
		    printf( "%d %s %s %8.3f %8.3f %8.3f\n",\
			    new_atoms+1, atom_type[ref], atom_name[ref], 
			    base_x + atom_pos[ref,0], 
			    base_y + atom_pos[ref,1], 
			    base_z + atom_pos[ref,2] );
		    ++new_atoms;
		}
	    }
}
