CC      = gcc 

LIBS    = -lm -lz
CFLAGS  = -I.  -Wall -O3 -funroll-loops -fstrict-aliasing
#CFLAGS  = -I.  -Wall -funroll-loops -fstrict-aliasing
#-finline-functions  -finline-limit=15 -g#-DTEST -pg -ldl -rdynamic -g
DEFS    = $(CFLAGS)
FLAG    = $(DEFS) $(INCS) $(LIBS)

OBJ     =  grid.o list.o lookup.o print_utils.o reset_utils.o restart.o random.o traj.o vector.o allocate.o analyze.o box.o system_props.o param.o pdb_tools.o neighbors.o GMRES.o QEq.o bond_orders.o single_body_interactions.o two_body_interactions.o three_body_interactions.o four_body_interactions.o forces.o integrate.o init_md.o testmd.o

#all:    SerialReax
#SerialReax: $(OBJ) Makefile
#	$(CC) $(OBJ) $(FLAG) -o SerialReax

all:   spuremd 
spuremd: $(OBJ) Makefile
	$(CC) $(OBJ) $(FLAG) -o spuremd 

testmd.o: testmd.c mytypes.h forces.o analyze.o init_md.o integrate.o neighbors.o param.o pdb_tools.o QEq.o system_props.o traj.o vector.o
	$(CC) $(DEFS) -c testmd.c

init_md.o: init_md.c init_md.h mytypes.h box.o forces.o grid.o integrate.o list.o lookup.o print_utils.o reset_utils.o system_props.o traj.o vector.o
	$(CC) $(DEFS) -c init_md.c

integrate.o: integrate.c integrate.h mytypes.h box.o forces.o neighbors.o QEq.o reset_utils.o system_props.o vector.o
	$(CC) $(DEFS) -c integrate.c

forces.o: forces.c forces.h mytypes.h bond_orders.o single_body_interactions.o two_body_interactions.o three_body_interactions.o four_body_interactions.o print_utils.o vector.o
	$(CC) $(DEFS) -c forces.c

four_body_interactions.o: four_body_interactions.c four_body_interactions.h mytypes.h bond_orders.o list.o lookup.o vector.o
	$(CC) $(DEFS) -c four_body_interactions.c

three_body_interactions.o: three_body_interactions.c three_body_interactions.h mytypes.h bond_orders.o list.o lookup.o vector.o
	$(CC) $(DEFS) -c three_body_interactions.c

two_body_interactions.o: two_body_interactions.c two_body_interactions.h mytypes.h bond_orders.o list.o lookup.o vector.o 
	$(CC) $(DEFS) -c two_body_interactions.c

single_body_interactions.o: single_body_interactions.c single_body_interactions.h mytypes.h bond_orders.o list.o lookup.o vector.o
	$(CC) $(DEFS) -c single_body_interactions.c

bond_orders.o: bond_orders.c bond_orders.h mytypes.h list.o lookup.o vector.o
	$(CC) $(DEFS) -c bond_orders.c

QEq.o: QEq.c QEq.h mytypes.h GMRES.o list.o print_utils.o
	$(CC) $(DEFS) -c QEq.c

GMRES.o: GMRES.c GMRES.h mytypes.h list.o vector.o
	$(CC) $(DEFS) -c GMRES.c

neighbors.o: neighbors.c neighbors.h mytypes.h box.o grid.o list.o reset_utils.o system_props.o vector.o
	$(CC) $(DEFS) -c neighbors.c

grid.o: grid.c grid.h mytypes.h reset_utils.o vector.o
	$(CC) $(DEFS) -c grid.c

pdb_tools.o: pdb_tools.c pdb_tools.h mytypes.h box.o list.o param.o
	$(CC) $(DEFS) -c pdb_tools.c

print_utils.o: print_utils.c print_utils.h mytypes.h list.o pdb_tools.o restart.o system_props.o vector.o
	$(CC) $(DEFS) -c print_utils.c

param.o: param.c param.h mytypes.h traj.o
	$(CC) $(DEFS) -c param.c

reset_utils.o: reset_utils.c reset_utils.h mytypes.h list.o vector.o
	$(CC) $(DEFS) -c reset_utils.c

system_props.o: system_props.c system_props.h mytypes.h vector.o
	$(CC) $(DEFS) -c system_props.c

allocate.o: allocate.c allocate.h mytypes.h list.o
	$(CC) $(DEFS) -c allocate.c

analyze.o: analyze.c analyze.h mytypes.h list.o vector.o
	$(CC) $(DEFS) -c analyze.c

box.o: box.c box.h mytypes.h vector.o
	$(CC) $(DEFS) -c box.c


list.o: list.c list.h mytypes.h 
	$(CC) $(DEFS) -c list.c

lookup.o: lookup.c lookup.h mytypes.h 
	$(CC) $(DEFS) -c lookup.c

random.o: random.c random.h mytypes.h 
	$(CC) $(DEFS) -c random.c

restart.o: restart.c restart.h mytypes.h
	$(CC) $(DEFS) -c restart.c

traj.o: traj.c traj.h mytypes.h 
	$(CC) $(DEFS) -c traj.c

vector.o: vector.c vector.h mytypes.h 
	$(CC) $(DEFS) -c vector.c

clean:
	rm -f *.o *~ core 
