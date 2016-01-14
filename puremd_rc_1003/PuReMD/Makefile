#CC      = cc
CC	= mpicc
CDEBUG  = -g
CNOOPT  = -O2
COPT    = -O3 -funroll-loops -fstrict-aliasing #-finline-functions -finline-limit=15 -qhot
#COPT    = -fast -Mipa=fast,inline
#COPT    = -fast

LIBS    = -lz -lm #-lmass
CDEFS   = -I. #-Wall -W -Wno-uninitialized -ldl -rdynamic -pg -g #-qlanglvl=extc99 -qsuppress=1500-036 -qarch=auto -qtune=auto
CFLAGS  = $(CDEFS) $(COPT)
FLAG    = $(CFLAGS) $(INCS) $(LIBS)
EXE     = puremd

OBJ     =  allocate.o basic_comm.o ffield.o grid.o list.o lookup.o io_tools.o reset_tools.o restart.o random.o tool_box.o traj.o vector.o analyze.o box.o system_props.o control.o comm_tools.o geo_tools.o linear_solvers.o neighbors.o qEq.o bond_orders.o multi_body.o bonds.o valence_angles.o hydrogen_bonds.o torsion_angles.o nonbonded.o forces.o integrate.o init_md.o parallelreax.o

all:    $(EXE)

puremd: $(OBJ) Makefile
	$(CC) $(OBJ) $(FLAG) -o $(EXE)

# level 8
parallelreax.o: parallelreax.c reax_types.h analyze.o comm_tools.o control.o ffield.o forces.o geo_tools.o init_md.o integrate.o io_tools.o neighbors.o reset_tools.o restart.o system_props.o tool_box.o traj.o vector.o
	$(CC) $(CFLAGS) -c parallelreax.c

# level 7
init_md.o: init_md.c init_md.h reax_types.h allocate.o box.o comm_tools.o forces.o grid.o integrate.o io_tools.o list.o lookup.o neighbors.o random.o reset_tools.o system_props.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c init_md.c

#level 6
integrate.o: integrate.c integrate.h reax_types.h box.o comm_tools.o forces.o grid.o io_tools.o neighbors.o reset_tools.o system_props.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c integrate.c

# level 5
forces.o: forces.c forces.h reax_types.h reax_defs.h bond_orders.o bonds.o basic_comm.o hydrogen_bonds.o io_tools.o list.o lookup.o multi_body.o nonbonded.o qEq.o tool_box.o torsion_angles.o valence_angles.o vector.o
	$(CC) $(CFLAGS) -c forces.c

geo_tools.o: geo_tools.c geo_tools.h reax_types.h allocate.o box.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c geo_tools.c

restart.o: restart.c restart.h reax_types.h allocate.o box.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c restart.c

# level 4
box.o: box.c box.h reax_types.h comm_tools.o io_tools.o system_props.o vector.o
	$(CC) $(CFLAGS) -c box.c

comm_tools.o: comm_tools.c comm_tools.h grid.o reset_tools.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c comm_tools.c

linear_solvers.o: linear_solvers.c linear_solvers.h reax_types.h basic_comm.o io_tools.o vector.o
	$(CC) $(CFLAGS) -c linear_solvers.c

neighbors.o: neighbors.c neighbors.h reax_types.h io_tools.o list.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c neighbors.c

qEq.o: qEq.c qEq.h reax_types.h allocate.o basic_comm.o io_tools.o linear_solvers.o
	$(CC) $(CFLAGS) -c qEq.c

# level 3
bonds.o: bonds.c bonds.h reax_types.h bond_orders.o list.o tool_box.o vector.o 
	$(CC) $(CFLAGS) -c bonds.c

grid.o: grid.c grid.h reax_types.h allocate.o reset_tools.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c grid.c

hydrogen_bonds.o: hydrogen_bonds.c hydrogen_bonds.h reax_types.h bond_orders.o list.o valence_angles.o vector.o
	$(CC) $(CFLAGS) -c hydrogen_bonds.c

io_tools.o: io_tools.c io_tools.h basic_comm.o list.o reset_tools.o system_props.o tool_box.o traj.o vector.o
	$(CC) $(CFLAGS) -c io_tools.c

lookup.o: lookup.c lookup.h reax_types.h nonbonded.o tool_box.o
	$(CC) $(CFLAGS) -c lookup.c

multi_body.o: multi_body.c multi_body.h reax_types.h bond_orders.o list.o vector.o
	$(CC) $(CFLAGS) -c multi_body.c

torsion_angles.o: torsion_angles.c torsion_angles.h reax_types.h tool_box.o bond_orders.o list.o vector.o
	$(CC) $(CFLAGS) -c torsion_angles.c

valence_angles.o: valence_angles.c valence_angles.h reax_types.h bond_orders.o list.o vector.o
	$(CC) $(CFLAGS) -c valence_angles.c

# level 2
allocate.o: allocate.c allocate.h list.o reset_tools.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c allocate.c

analyze.o: analyze.c analyze.h reax_types.h list.o vector.o
	$(CC) $(CFLAGS) -c analyze.c

basic_comm.o: basic_comm.c basic_comm.h reax_types.h vector.o
	$(CC) $(CFLAGS) -c basic_comm.c

bond_orders.o: bond_orders.c bond_orders.h reax_types.h list.o vector.o
	$(CC) $(CFLAGS) -c bond_orders.c

nonbonded.o: nonbonded.c nonbonded.h reax_types.h list.o vector.o 
	$(CC) $(CFLAGS) -c nonbonded.c

reset_tools.o: reset_tools.c reset_tools.h reax_types.h list.o tool_box.o vector.o
	$(CC) $(CFLAGS) -c reset_tools.c

system_props.o: system_props.c system_props.h reax_types.h tool_box.o vector.o
	$(CC) $(CFLAGS) -c system_props.c

# level 1
control.o: control.c control.h reax_types.h tool_box.o
	$(CC) $(CFLAGS) -c control.c

ffield.o: ffield.c ffield.h reax_types.h tool_box.o
	$(CC) $(CFLAGS) -c ffield.c

list.o: list.c list.h reax_types.h tool_box.o
	$(CC) $(CFLAGS) -c list.c

traj.o: traj.c traj.h reax_types.h list.o tool_box.o
	$(CC) $(CFLAGS) -c traj.c

vector.o: vector.c vector.h reax_types.h reax_defs.h random.o
	$(CC) $(CFLAGS) -c vector.c

# level 0 - no external dependencies
random.o: random.c random.h reax_types.h 
	$(CC) $(CFLAGS) -c random.c

tool_box.o: tool_box.c tool_box.h reax_types.h reax_defs.h
	$(CC) $(CFLAGS) -c tool_box.c

clean:
	rm -f *.o *~ core $(EXE) 
