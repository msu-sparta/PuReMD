ACLOCAL_AMFLAGS = -I ../m4

bin_PROGRAMS = bin/puremd
bin_puremd_SOURCES = src/allocate.c src/basic_comm.c src/ffield.c src/grid.c src/list.c src/lookup.c \
      src/io_tools.c src/reset_tools.c src/restart.c src/random.c src/tool_box.c src/traj.c \
      src/vector.c src/analyze.c src/box.c src/system_props.c src/control.c src/comm_tools.c \
      src/geo_tools.c src/linear_solvers.c src/neighbors.c src/qEq.c src/bond_orders.c src/multi_body.c \
      src/bonds.c src/valence_angles.c src/hydrogen_bonds.c src/torsion_angles.c src/nonbonded.c src/forces.c \
      src/integrate.c src/init_md.c src/parallelreax.c \
      src/reax_defs.h src/reax_types.h \
      src/allocate.h src/basic_comm.h src/ffield.h src/grid.h src/list.h src/lookup.h \
      src/io_tools.h src/reset_tools.h src/restart.h src/random.h src/tool_box.h src/traj.h \
      src/vector.h src/analyze.h src/box.h src/system_props.h src/control.h src/comm_tools.h \
      src/geo_tools.h src/linear_solvers.h src/neighbors.h src/qEq.h src/bond_orders.h src/multi_body.h \
      src/bonds.h src/valence_angles.h src/hydrogen_bonds.h src/torsion_angles.h src/nonbonded.h src/forces.h \
      src/integrate.h src/init_md.h

bin_puremd_CFLAGS = $(MPI_CFLAGS)
bin_puremd_LDADD = $(MPI_LIBS)
