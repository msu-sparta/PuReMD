#!/bin/bash          
LAMMPS_DIR=/homes/haktulga/lammps/lammps-5Jun10/src_new/REAX_C
diff -u reax_types.h $LAMMPS_DIR/reax_types.h > reax_types.patch

cp reax_types.h $LAMMPS_DIR/reax_types.h
cp reax_defs.h $LAMMPS_DIR/reax_defs.h
cp allocate.h $LAMMPS_DIR/reax_allocate.h 
cp allocate.c $LAMMPS_DIR/reax_allocate.cpp 
cp basic_comm.h $LAMMPS_DIR/reax_basic_comm.h
cp basic_comm.c $LAMMPS_DIR/reax_basic_comm.cpp
cp bond_orders.h $LAMMPS_DIR/reax_bond_orders.h
cp bond_orders.c $LAMMPS_DIR/reax_bond_orders.cpp
cp bonds.h $LAMMPS_DIR/reax_bonds.h
cp bonds.c $LAMMPS_DIR/reax_bonds.cpp
cp control.h $LAMMPS_DIR/reax_control.h
cp control.c $LAMMPS_DIR/reax_control.cpp
cp ffield.h $LAMMPS_DIR/reax_ffield.h
cp ffield.c $LAMMPS_DIR/reax_ffield.cpp
cp forces.h $LAMMPS_DIR/reax_forces.h
cp forces.c $LAMMPS_DIR/reax_forces.cpp
cp hydrogen_bonds.h $LAMMPS_DIR/reax_hydrogen_bonds.h
cp hydrogen_bonds.c $LAMMPS_DIR/reax_hydrogen_bonds.cpp
cp init_md.h $LAMMPS_DIR/reax_init_md.h
cp init_md.c $LAMMPS_DIR/reax_init_md.cpp
cp io_tools.h $LAMMPS_DIR/reax_io_tools.h
cp io_tools.c $LAMMPS_DIR/reax_io_tools.cpp
cp list.h $LAMMPS_DIR/reax_list.h
cp list.c $LAMMPS_DIR/reax_list.cpp
cp lookup.h $LAMMPS_DIR/reax_lookup.h
cp lookup.c $LAMMPS_DIR/reax_lookup.cpp
cp multi_body.h $LAMMPS_DIR/reax_multi_body.h
cp multi_body.c $LAMMPS_DIR/reax_multi_body.cpp
cp nonbonded.h $LAMMPS_DIR/reax_nonbonded.h
cp nonbonded.c $LAMMPS_DIR/reax_nonbonded.cpp
cp reset_tools.h $LAMMPS_DIR/reax_reset_tools.h
cp reset_tools.c $LAMMPS_DIR/reax_reset_tools.cpp
cp system_props.h $LAMMPS_DIR/reax_system_props.h
cp system_props.c $LAMMPS_DIR/reax_system_props.cpp
cp tool_box.h $LAMMPS_DIR/reax_tool_box.h
cp tool_box.c $LAMMPS_DIR/reax_tool_box.cpp
cp torsion_angles.h $LAMMPS_DIR/reax_torsion_angles.h
cp torsion_angles.c $LAMMPS_DIR/reax_torsion_angles.cpp
cp traj.h $LAMMPS_DIR/reax_traj.h
cp traj.c $LAMMPS_DIR/reax_traj.cpp
cp valence_angles.h $LAMMPS_DIR/reax_valence_angles.h
cp valence_angles.c $LAMMPS_DIR/reax_valence_angles.cpp
cp vector.h $LAMMPS_DIR/reax_vector.h
cp vector.c $LAMMPS_DIR/reax_vector.cpp

patch -d $LAMMPS_DIR < reax_types.patch
