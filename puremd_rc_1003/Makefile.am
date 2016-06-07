ACLOCAL_AMFLAGS = -I m4

if BUILD_S_OMP
  S_OMP=sPuReMD
endif
if BUILD_MPI
  MPI=PuReMD
endif
if BUILD_SINGLE_GPU
  SINGLE_GPU=PuReMD-GPU
endif
if BUILD_MULTI_GPU
  MULTI_GPU=PG-PuReMD
endif

SUBDIRS = $(S_OMP) $(MPI) $(SINGLE_GPU) $(MULTI_GPU)

EXTRA_DIST = doc/README.txt doc/manual.pdf

dist-hook: rm -rf `find $(distdir) -name .git`
