BINDIR=./bin

CC	 	= g++
NVCC	 = nvcc

LIBS    = -L/opt/cuda/lib64 -lm -lz -lcuda -lcudart -lcublas -lcusparse -m64
#LIBS    = -L/opt/cuda/lib64 -lm -lz -lcuda -lcudart

#Debug Options
#__CUDA_MEM__ Prints the memory allocated to the executable
#__DEBUG_CUDA__ Prints debug information in CUDA
#__BUILD_DEBUG__ builds a debug version, which runs serial and gpu code at the same time for 
#					  every time step and validates the results 

NVCCFLAGS  = -I.  -I/opt/cuda/include -D__USE_GPU__ -arch=sm_20 -O3 -Xptxas="-O3"
CFLAGS  = -I.  -Wall -funroll-loops -fstrict-aliasing -O3

DEFS    = $(CFLAGS)
NVCCDEFS = $(NVCCFLAGS)

FLAG    = $(DEFS) $(INCS) $(LIBS)
NVCCFLAG = $(NVCCDEFS) $(LIBS)

OBJ		= GMRES.o QEq.o allocate.o analyze.o bond_orders.o \
			box.o forces.o four_body_interactions.o \
			grid.o init_md.o integrate.o list.o \
			lookup.o neighbors.o param.o pdb_tools.o \
			print_utils.o reset_utils.o \
			restart.o single_body_interactions.o \
			system_props.o three_body_interactions.o \
			traj.o two_body_interactions.o vector.o \
			testmd.o \
			cuda_utils.o cuda_copy.o cuda_init.o reduction.o \
			center_mass.o helpers.o validation.o matvec.o

all:  gpu 

#
#build the executable here
#

gpu: $(OBJ) Makefile
	$(NVCC) $(NVCCFLAG) $(OBJ) -o puremd-gpu 

#
#Existing C code
#
#

param.o: param.c param.h mytypes.h traj.o
	$(CC) $(DEFS) -c param.c

restart.o: restart.c restart.h mytypes.h box.h
	$(CC) $(DEFS) -c restart.c

pdb_tools.o: pdb_tools.c pdb_tools.h mytypes.h box.h list.h restart.h param.h
	$(CC) $(DEFS) -c pdb_tools.c

analyze.o: analyze.h analyze.c
	$(CC) $(DEFS) -c analyze.c

print_utils.o: print_utils.h print_utils.c
	$(CC) $(DEFS) -c print_utils.c


#
# Code converted to cuda
#

traj.o: traj.cu traj.h mytypes.h 
	$(NVCC) $(NVCCDEFS) -c traj.cu

grid.o: grid.h grid.cu
	$(NVCC) $(NVCCDEFS) -c grid.cu

integrate.o: integrate.h integrate.cu
	$(NVCC) $(NVCCDEFS) -c integrate.cu

vector.o: vector.h vector.cu
	$(NVCC) $(NVCCDEFS) -c vector.cu

system_props.o: system_props.h system_props.cu
	$(NVCC) $(NVCCDEFS) -c system_props.cu

cuda_copy.o: cuda_copy.h cuda_copy.cu
	$(NVCC) $(NVCCDEFS) -c cuda_copy.cu

cuda_utils.o: cuda_utils.h cuda_utils.cu
	$(NVCC) $(NVCCDEFS) -c cuda_utils.cu

cuda_init.o: cuda_init.h cuda_init.cu
	$(NVCC) $(NVCCDEFS) -c cuda_init.cu

reduction.o: reduction.h reduction.cu
	$(NVCC) $(NVCCDEFS) -c reduction.cu

center_mass.o: center_mass.h center_mass.cu
	$(NVCC) $(NVCCDEFS) -c center_mass.cu

box.o: box.h
	$(NVCC) $(NVCCDEFS) -c box.cu

init_md.o: init_md.h init_md.cu
	$(NVCC) $(NVCCDEFS) -c init_md.cu

helpers.o: helpers.h helpers.cu
	$(NVCC) $(NVCCDEFS) -c helpers.cu

GMRES.o: GMRES.h GMRES.cu
	$(NVCC) $(NVCCDEFS) -c GMRES.cu

neighbors.o: neighbors.cu neighbors.h
	$(NVCC) $(NVCCDEFS) -c neighbors.cu

reset_utils.o: reset_utils.h reset_utils.cu mytypes.h list.h vector.h
	$(NVCC) $(NVCCDEFS) -c reset_utils.cu

list.o: list.h 
	$(NVCC) $(NVCCDEFS) -c list.cu

forces.o: forces.h forces.cu
	$(NVCC) $(NVCCDEFS) -c forces.cu

allocate.o: allocate.h allocate.cu
	$(NVCC) $(NVCCDEFS) -c allocate.cu

lookup.o: lookup.h lookup.cu
	$(NVCC) $(NVCCDEFS) -c lookup.cu

two_body_interactions.o: two_body_interactions.h two_body_interactions.cu
	$(NVCC) $(NVCCDEFS) -c two_body_interactions.cu

bond_orders.o: bond_orders.h bond_orders.cu
	$(NVCC) $(NVCCDEFS) -c bond_orders.cu

single_body_interactions.o: single_body_interactions.h single_body_interactions.cu
	$(NVCC) $(NVCCDEFS) -c single_body_interactions.cu

three_body_interactions.o: three_body_interactions.h three_body_interactions.cu
	$(NVCC) $(NVCCDEFS) -c three_body_interactions.cu

four_body_interactions.o: four_body_interactions.h four_body_interactions.cu
	$(NVCC) $(NVCCDEFS) -c four_body_interactions.cu

validation.o: validation.h validation.cu
	$(NVCC) $(NVCCDEFS) -c validation.cu

QEq.o: QEq.h QEq.cu
	$(NVCC) $(NVCCDEFS) -c QEq.cu

matvec.o: matvec.h matvec.cu
	$(NVCC) $(NVCCDEFS) -c matvec.cu

testmd.o: testmd.cu mytypes.h param.o traj.o restart.o pdb_tools.o list.o vector.o 
	$(NVCC) $(NVCCDEFS) -c testmd.cu

#
# Clean up here
#

clean:
	rm -f *.o *~ core 
