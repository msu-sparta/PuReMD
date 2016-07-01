ACLOCAL_AMFLAGS = -I m4

SUBDIRS =

if BUILD_S_OMP
SUBDIRS += sPuReMD
endif
if BUILD_MPI
SUBDIRS += PuReMD
endif
if BUILD_GPU
SUBDIRS += PuReMD-GPU
endif
if BUILD_MPI_GPU
SUBDIRS += PG-PuReMD
endif

if BUILD_DOC
SUBDIRS += doc
endif

dist-hook: rm -rf `find $(distdir) -name .git`
