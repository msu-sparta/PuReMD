ACLOCAL_AMFLAGS = -I ../m4

if USE_CUDA
SUFFIXES = .cu
include ../cuda.am
endif

AM_CFLAGS = -Wall -O3 -funroll-loops -fstrict-aliasing -m64
AM_CPPFLAGS =
AM_LDFLAGS =

# default CUDA nvcc flags
#   Note: cc 13 for Tesla
#   Note: cc 20 for Fermi
#   Note: cc 30 for Kepler K10
#   Note: cc 35 for Kepler K20
NVCCFLAGS += -use_fast_math 
NVCCFLAGS += -gencode arch=compute_35,code=sm_35
NVCCFLAGS += --compiler-options "$(DEFS) -D__SM_35__ -O3 -funroll-loops -fstrict-aliasing"
#NVCCFLAGS += -Xcompiler -fPIC -dc
#NVCCFLAGS += --ptxas-options -v

bin_PROGRAMS = bin/spuremd
bin_spuremd_SOURCES = src/analyze.c src/print_utils.c \
	src/restart.c src/tool_box.c src/control.c src/ffield.c \
	src/geo_tools.c src/box.c \
	src/lin_alg.c src/qeq.c src/allocate.c src/bond_orders.c \
	src/forces.c src/four_body_interactions.c \
	src/grid.c src/init_md.c src/integrate.c src/list.c \
	src/lookup.c src/neighbors.c \
	src/reset_utils.c src/single_body_interactions.c \
	src/system_props.c src/three_body_interactions.c \
	src/traj.c src/two_body_interactions.c src/vector.c \
	src/testmd.c
include_HEADERS = src/mytypes.h src/analyze.h src/print_utils.h \
        src/restart.h src/tool_box.c src/control.h src/ffield.c \
	src/geo_tools.h src/box.h \
	src/lin_alg.h src/qeq.h src/allocate.h src/bond_orders.h \
	src/forces.h src/four_body_interactions.h \
	src/grid.h src/init_md.h src/integrate.h src/list.h \
	src/lookup.h src/neighbors.h \
	src/reset_utils.h src/single_body_interactions.h \
	src/system_props.h src/three_body_interactions.h \
	src/traj.h src/two_body_interactions.h src/vector.h

if USE_CUDA
bin_spuremd_SOURCES += src/cuda_utils.cu src/cuda_copy.cu \
	src/cuda_init.cu src/cuda_reduction.cu \
	src/cuda_center_mass.cu src/cuda_box.cu src/cuda_validation.cu \
        src/cuda_allocate.cu src/cuda_bond_orders.cu \
	src/cuda_lin_alg.cu src/cuda_qeq.cu \
        src/cuda_forces.cu src/cuda_four_body_interactions.cu \
	src/cuda_grid.cu src/cuda_init_md.cu src/cuda_integrate.cu src/cuda_list.cu \
	src/cuda_lookup.cu src/cuda_neighbors.cu \
	src/cuda_reset_utils.cu src/cuda_single_body_interactions.cu \
        src/cuda_system_props.cu src/cuda_three_body_interactions.cu \
	src/cuda_two_body_interactions.cu src/cuda_environment.cu \
	src/cuda_post_evolve.cu
include_HEADERS += src/cuda_utils.h src/cuda_copy.h \
	src/cuda_init.h src/cuda_reduction.h \
	src/cuda_center_mass.h src/cuda_box.h src/cuda_validation.h \
        src/cuda_allocate.h src/cuda_bond_orders.h \
	src/cuda_lin_alg.h src/cuda_qeq.h \
        src/cuda_forces.h src/cuda_four_body_interactions.h \
	src/cuda_grid.h src/cuda_init_md.h src/cuda_integrate.h src/cuda_list.h \
	src/cuda_lookup.h src/cuda_neighbors.h \
	src/cuda_reset_utils.h src/cuda_single_body_interactions.h \
        src/cuda_system_props.h src/cuda_three_body_interactions.h \
	src/cuda_two_body_interactions.h src/cuda_environment.h \
	src/cuda_post_evolve.h

# dummy source to cause C linking
nodist_EXTRA_bin_spuremd_SOURCES = src/dummy.c

endif


bin_spuremd_CFLAGS = $(AM_CFLAGS) $(CFLAGS)
bin_spuremd_CPPFLAGS = $(AM_CPPFLAGS) $(CPPFLAGS)
bin_spuremd_LDFLAGS = $(AM_LDFLAGS) $(LDFLAGS)

if USE_CUDA
bin_spuremd_CFLAGS += $(CUDA_CFLAGS)
bin_spuremd_LDFLAGS += $(CUDA_LIBS)
endif
