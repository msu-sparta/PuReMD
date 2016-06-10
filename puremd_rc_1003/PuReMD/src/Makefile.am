AM_CFLAGS = -Wall -O3 -funroll-loops -fstrict-aliasing #-finline-functions -finline-limit=15 -qhot

bin_PROGRAMS = puremd
puremd_SOURCES = allocate.c basic_comm.c ffield.c grid.c list.c lookup.c \
      io_tools.c reset_tools.c restart.c random.c tool_box.c traj.c \
      vector.c analyze.c box.c system_props.c control.c comm_tools.c \
      geo_tools.c linear_solvers.c neighbors.c qEq.c bond_orders.c multi_body.c \
      bonds.c valence_angles.c hydrogen_bonds.c torsion_angles.c nonbonded.c forces.c \
      integrate.c init_md.c parallelreax.c

puremd_CFLAGS = $(AM_CFLAGS) $(MPI_CFLAGS)
puremd_LDFLAGS = $(AM_LDFLAGS) $(MPI_LDFLAGS)
