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

#ifndef __VALIDATION_H__
#define __VALIDATION_H__

#include "mytypes.h"

int check_zero (real , real );
int check_zero (rvec , rvec );
int check_same (ivec , ivec );


int validate_box (simulation_box *host, simulation_box *dev);
int validate_atoms (reax_system *, list **);
int validate_grid (reax_system *);

int validate_bonds (reax_system *, static_storage *, list **);
int validate_hbonds (reax_system *, static_storage *, list **);
int validate_sym_dbond_indices (reax_system *, static_storage *, list **);
int validate_three_bodies (reax_system *, static_storage *, list **);
void count_three_bodies (reax_system *system, static_storage *workspace, list **lists);

int bin_three_bodies (reax_system *, static_storage *, list **);

int validate_sort_matrix (reax_system *, static_storage *);
int validate_sparse_matrix (reax_system *, static_storage *);
int validate_lu (static_storage *);
void print_sparse_matrix (reax_system *, static_storage *);
void print_bond_list (reax_system *, static_storage *, list **);

int validate_workspace (reax_system *, static_storage *, list **);
int validate_neighbors (reax_system *, list **lists);

int validate_data (reax_system *, simulation_data *);

int analyze_hbonds (reax_system *, static_storage *, list **);

void Print_Matrix (sparse_matrix *);
void Print_Matrix_L (sparse_matrix *);
void print_far_neighbors (reax_system *, list **);
void print_atoms (reax_system *);
void print_sys_atoms (reax_system *);
void print_grid (reax_system *);
#endif
