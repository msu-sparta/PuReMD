#ifndef __INDEX_UTILS_H_
#define __INDEX_UTILS_H_

#include "reax_types.h"


static inline CUDA_HOST_DEVICE int index_grid_3d( int i, int j, int k, grid *g )
{
    return  (i * g->ncells[1] * g->ncells[2]) +
            (j * g->ncells[2]) +
            k;
}

static inline CUDA_HOST_DEVICE int index_grid_nbrs( int i, int j, int k, int l, grid *g )
{
    return  (i * g->ncells[1] * g->ncells[2] * g->max_nbrs) +
            (j * g->ncells[2] * g->max_nbrs) +
            (k * g->max_nbrs) +
            l;
}

static inline CUDA_HOST_DEVICE int index_grid_atoms( int i, int j, int k, int l, grid *g )
{
    return  (i * g->ncells[1] * g->ncells[2] * g->max_atoms) +
            (j * g->ncells[2] * g->max_atoms) +
            (k * g->max_atoms) +
            l;
}

static inline CUDA_HOST_DEVICE int index_wkspace_sys( int i, int j, int N )
{
    return (i * N) + j;
}

static inline CUDA_HOST_DEVICE int index_wkspace_res( int i, int j )
{
    return (i * (RESTART + 1)) + j;
}

static inline CUDA_HOST_DEVICE int index_tbp( int i, int j, int num_atom_types )
{
    return (i * num_atom_types) + j;
}

static inline CUDA_HOST_DEVICE int index_thbp( int i, int j, int k, int num_atom_types )
{
    return  (i * num_atom_types * num_atom_types ) +
            (j * num_atom_types ) +
            k;
}

static inline CUDA_HOST_DEVICE int index_hbp( int i, int j, int k, int num_atom_types )
{
    return  (i * num_atom_types * num_atom_types ) +
            (j * num_atom_types ) +
            k;
}

static inline CUDA_HOST_DEVICE int index_fbp( int i, int j, int k, int l, int num_atom_types )
{
    return  (i * num_atom_types * num_atom_types * num_atom_types ) +
            (j * num_atom_types * num_atom_types ) +
            (k * num_atom_types ) +
            l;
}

static inline CUDA_HOST_DEVICE int index_lr( int i, int j, int num_atom_types )
{
    return (i * num_atom_types) + j;
}

#endif
