AM_CFLAGS = -Wall -O3 -funroll-loops -fstrict-aliasing #-finline-functions -finline-limit=15 -qhot

bin_PROGRAMS = bin/puremd
bin_puremd_SOURCES = src/allocate.c src/basic_comm.c src/ffield.c src/grid.c src/list.c src/lookup.c \
      src/io_tools.c src/reset_tools.c src/restart.c src/random.c src/tool_box.c src/traj.c \
      src/vector.c src/analyze.c src/box.c src/system_props.c src/control.c src/comm_tools.c \
      src/geo_tools.c src/linear_solvers.c src/neighbors.c src/qEq.c src/bond_orders.c src/multi_body.c \
      src/bonds.c src/valence_angles.c src/hydrogen_bonds.c src/torsion_angles.c src/nonbonded.c src/forces.c \
      src/integrate.c src/init_md.c src/parallelreax.c

bin_puremd_CFLAGS = $(AM_CFLAGS) $(MPI_CFLAGS)
bin_puremd_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS)
