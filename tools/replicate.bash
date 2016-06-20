#!/bin/bash

function usage
{
	echo -e "\tusage: $0 format x_dim y_dim z_dim in_file [out_file]"
	exit 1
}


function repl_geo
{
	if [ -z "${5:+x}" ]; then
		FILE=$(basename "$1")
		FILE="${FILE%.*}.geo"
	else
		FILE="$5"
	fi

	# 1. Convert the pdb file to custom format which is an intermediate format
	"$AWK" '{if($1=="CRYST1") print "BOXGEO", $2, $3, $4, $5, $6, $7; if($1=="ATOM") print $2, $3, $12, $6, $7, $8;}' \
	       	"$1" >& "$FILE.cus"

	# 2. Convert the custom format to geo format with the following command
	#     replicate_geo.awk script can also be used to replicate the box in 
	#     any of the x, y, z directions. They are set as 1, 1, 1 for now. 
	#     If you want to replicate it, just change these numbers at the beginning
	#    of the replicate_geo.awk file.
	"$SED" -e "s/replicate_x = .*/replicate_x = $2;/" \
	       -e "s/replicate_y = .*/replicate_y = $3;/" \
       	       -e "s/replicate_z = .*/replicate_z = $4;/" \
	       -i "$GEO"
	"$AWK" -f "$GEO" "$FILE.cus" > "$FILE"
	rm "$FILE.cus"

	# 3. Open the geo file and insert a line below the BOXGEO line
	#     BOXGEO is the very first line which contains the BOX geometry
	#    The new line to be inserted contains the number of atoms in the
	#     newly created geo file. This number should be equal to the 
	#     total number of lines in the geo file - (2). one is the first line
	#     which is the BOXGEO line and second one is for the newly inserted line.

	LC=$(($(cat "$FILE" | wc -l)-1))
	"$SED" -e "s/BOXGEO.*/&\n$LC/" -i "$FILE"
}


function repl_pdb
{
	#TODO
	exit
}


AWK="/usr/bin/awk"
SED="/usr/bin/sed"

GEO="$(dirname $0)/replicate_geo.awk"

if [ "$#" -ne 5 -a "$#" -ne 6 ]; then
	usage
fi

if [ "$1" == "geo" ]; then
	repl_geo "$5" "$2" "$3" "$4" "$6"
elif [ "$1" == "pdb" ]; then
	repl_pdb "$5" "$2" "$3" "$4" "$6"
fi
