ACLOCAL_AMFLAGS = -I m4

SUBDIRS =
DIST_SUBDIRS =

if BUILD_S_OMP
SUBDIRS += sPuReMD
DIST_SUBDIRS += sPuReMD
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
endif

if BUILD_DOC
SUBDIRS += doc
DIST_SUBDIRS += doc
endif

EXTRA_DIST = data environ tools

dist-hook:
	rm -rf `find $(distdir) -name .git`
