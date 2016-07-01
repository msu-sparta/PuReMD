BEGIN{
  comment = "water strong scaling test";
  ntypes = 2;
  masses[1] = 1.0080;
  masses[2] = 15.990;
  types["H"] = 1;
  types["O"] = 2;
  box[0] = box[1] = box[2] = 0;
}
{
  if( $1 == "BOXGEO" ) {
    box[0] = $2;
    box[1] = $3;
    box[2] = $4;
  }
  else if( NF == 1 ) {
    natoms = $1;
    # print the header
    print "#", comment, "\n";
    
    print natoms, "atoms";
    print ntypes, "atom types\n";
    
    print "0", box[0], "xlo", "xhi";
    print "0", box[1], "ylo", "yhi";
    print "0", box[2], "zlo", "zhi\n";
    
    print "Masses\n";
    for( i = 1; i <= ntypes; ++i )
      print i, masses[i];
    
    print "\nAtoms\n";
  }
  else{
    # print the atom info
    print $1, types[$2], "0", $4, $5, $6;
  }
}