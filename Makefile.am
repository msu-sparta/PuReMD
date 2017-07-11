ACLOCAL_AMFLAGS = -I m4

SUBDIRS =
DIST_SUBDIRS =
EXTRA_DIST = data environ tools

if BUILD_S_OMP
SUBDIRS += PuReMD-GPU
DIST_SUBDIRS += PuReMD-GPU
endif
if BUILD_MPI
SUBDIRS += PuReMD
DIST_SUBDIRS += PuReMD
endif
if BUILD_GPU
SUBDIRS += PuReMD-GPU
DIST_SUBDIRS += PuReMD-GPU
endif
if BUILD_MPI_GPU
SUBDIRS += PG-PuReMD
DIST_SUBDIRS += PG-PuReMD
EXTRA_DIST += PG-PuReMD/src/cub
endif

if BUILD_DOC
SUBDIRS += doc
DIST_SUBDIRS += doc
endif

dist-hook:
	rm -rf `find $(distdir) -name .git`
