/*----------------------------------------------------------------------
  PuReMD-GPU - Reax Force Field Simulator

  Copyright (2014) Purdue University
  Sudhir Kylasa, skylasa@purdue.edu
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Ananth Y Grama, ayg@cs.purdue.edu

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License as
  published by the Free Software Foundation; either version 2 of
  the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details:
  <http://www.gnu.org/licenses/>.
  ----------------------------------------------------------------------*/

#ifndef __INDEX_UTILS_H_
#define __INDEX_UTILS_H_

#include "mytypes.h"


extern inline HOST_DEVICE int index_grid_3d (int i, int j, int k, grid *g)
{
    return  (i * g->ncell[1] * g->ncell[2]) +
            (j * g->ncell[2]) +
            k;
}


extern inline HOST_DEVICE int index_grid_nbrs (int i, int j, int k, int l, grid *g)
{
    return  (i * g->ncell[1] * g->ncell[2] * g->max_nbrs) +
            (j * g->ncell[2] * g->max_nbrs) +
            (k * g->max_nbrs) +
            l;
}


extern inline HOST_DEVICE int index_grid_atoms (int i, int j, int k, int l, grid *g)
{
    return  (i * g->ncell[1] * g->ncell[2] * g->max_atoms) +
            (j * g->ncell[2] * g->max_atoms) +
            (k * g->max_atoms) +
            l;
}


extern inline HOST_DEVICE int index_wkspace_sys (int i, int j, int N)
{
    return (i * N) + j;
}


extern inline HOST_DEVICE int index_wkspace_res (int i, int j )
{
    return (i * (RESTART + 1)) + j;
}


extern inline HOST_DEVICE int index_tbp (int i, int j, int num_atom_types)
{
    return (i * num_atom_types) + j;
}


extern inline HOST_DEVICE int index_thbp (int i, int j, int k, int num_atom_types)
{
    return  (i * num_atom_types * num_atom_types ) +
            (j * num_atom_types ) +
            k;
}


extern inline HOST_DEVICE int index_hbp (int i, int j, int k, int num_atom_types)
{
    return  (i * num_atom_types * num_atom_types ) +
            (j * num_atom_types ) +
            k;
}


extern inline HOST_DEVICE int index_fbp (int i, int j, int k, int l, int num_atom_types)
{
    return  (i * num_atom_types * num_atom_types * num_atom_types ) +
            (j * num_atom_types * num_atom_types ) +
            (k * num_atom_types ) +
            l;
}


extern inline HOST_DEVICE int index_lr (int i, int j, int num_atom_types )
{
    return (i * num_atom_types) + j;
}

#endif
