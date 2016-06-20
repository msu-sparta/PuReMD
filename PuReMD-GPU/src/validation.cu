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


#include "validation.h"

#include "cuda_utils.h"
#include "list.h"

#include "sort.h"
#include "index_utils.h"

bool check_zero (real p1, real p2)
{
	if (abs (p1 - p2) >= GPU_TOLERANCE)
		return true;
	else 
		return false;
}

bool check_zero (rvec p1, rvec p2)
{

	if (((abs (p1[0] - p2[0])) >= GPU_TOLERANCE) ||
			((abs (p1[1] - p2[1])) >= GPU_TOLERANCE) ||
			((abs (p1[2] - p2[2])) >= GPU_TOLERANCE ))
		return true;
	else return false;
}

bool check_same (ivec p1, ivec p2)
{
	if ( (p1[0] == p2[0]) || (p1[1] == p2[1]) || (p1[2] == p2[2]) )
		return true;
	else 
		return false;
}

bool validate_box (simulation_box *host, simulation_box *dev)
{

	simulation_box test;

	copy_host_device (&test, dev, SIMULATION_BOX_SIZE, cudaMemcpyDeviceToHost, RES_SYSTEM_SIMULATION_BOX );

	if (memcmp (&test, host, SIMULATION_BOX_SIZE)) {
		fprintf (stderr, " Simulation box is not in synch between host and device \n");
		return false;
	}

	fprintf (stderr, " Simulation box is in **synch** between host and device \n");
	return true;
}

bool validate_atoms (reax_system *system, list **lists)
{

	int start, end, index, count, miscount;
	reax_atom *test = (reax_atom *) malloc (REAX_ATOM_SIZE * system->N);
	copy_host_device (test, system->d_atoms, REAX_ATOM_SIZE * system->N, cudaMemcpyDeviceToHost, RES_SYSTEM_ATOMS );

	/*
	   int *d_start, *d_end;
	   bond_data *d_bond_data;
	   list *d_bonds = dev_lists + BONDS;
	   list *bonds = *lists + BONDS;

	   d_end = (int *)malloc (sizeof (int) * system->N);
	   d_start = (int *) malloc (sizeof (int) * system->N );
	   d_bond_data = (bond_data *) malloc (BOND_DATA_SIZE * system->num_bonds );

	   copy_host_device (d_start, d_bonds->index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
	   copy_host_device (d_end, d_bonds->end_index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
	   copy_host_device (d_bond_data, d_bonds->select.bond_list, BOND_DATA_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);


	   count = 0;
	   miscount = 0;
	   for (int i = 0; i < 1; i++) {

	   for (int j = d_start[i]; j < d_end[i]; j++) {
	   bond_data *src, *tgt;
	   src = &d_bond_data[j];
	   tgt = &d_bond_data[ src->dbond_index ];

	   fprintf (stderr, "Atom %d f neighbor %d vector (%e %e %e) thbh count %d \n", i, src->nbr, tgt->f[0], tgt->f[1], tgt->f[2], src->scratch );
	   }
	   }
	   exit (-1);
	 */

	//if (memcmp (test, system->atoms, REAX_ATOM_SIZE * system->N)) {
	count = miscount = 0;
	for (int i = 0; i < system->N; i++) 
	{
		if (test[i].type != system->atoms[i].type) {
			fprintf (stderr, " Type does not match (%d %d) @ index %d \n", system->atoms[i].type, test[i].type, i);
			exit (-1);
		}

		if ( 	check_zero (test[i].x, system->atoms[i].x) )
		{
			fprintf (stderr, "Atom :%d x --> host (%f %f %f) device (%f %f %f) \n", i,
					system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2], 
					test[i].x[0], test[i].x[1], test[i].x[2] );
			miscount ++;
			exit (-1);
		}
		if (		check_zero (test[i].v, system->atoms[i].v) )
		{
			fprintf (stderr, "Atom :%d v --> host (%6.10f %6.10f %6.10f) device (%6.10f %6.10f %6.10f) \n", i,
					system->atoms[i].v[0], system->atoms[i].v[1], system->atoms[i].v[2], 
					test[i].v[0], test[i].v[1], test[i].v[2] );
			miscount ++;
			exit (-1);
		}
		if (		check_zero (test[i].f, system->atoms[i].f) )
		{
			fprintf (stderr, "Atom :%d f --> host (%6.10f %6.10f %6.10f) device (%6.10f %6.10f %6.10f) \n", i,
					system->atoms[i].f[0], system->atoms[i].f[1], system->atoms[i].f[2], 
					test[i].f[0], test[i].f[1], test[i].f[2] );
			miscount ++;
			exit (-1);
		}

		if (		check_zero (test[i].q, system->atoms[i].q) )
		{
			fprintf (stderr, "Atom :%d q --> host (%f) device (%f) \n", i,
					system->atoms[i].q, test[i].q );
			miscount ++;
			exit (-1);
		}

		count ++;
	}

	//fprintf (stderr, "Reax Atoms DOES **match** between host and device --> %d miscount --> %d \n", count, miscount);

	free (test);
	return true;
}

void Print_Matrix( sparse_matrix *A )
{
	int i, j;
	for( i = 0; i < 10; ++i ) { 
		fprintf( stderr, "i:%d  j(val):", i );

		for( j = A->start[i]; j < A->end[i]; ++j )
			fprintf( stderr, "%d(%.4f) ", A->entries[j].j, A->entries[j].val );

		fprintf( stderr, "\n" );
	}
}

void Print_Matrix_L( sparse_matrix *A )
{
	int i, j;
	for( i = 0; i < 10; ++i ) { 
		fprintf( stderr, "i:%d  j(val):", i );

		for( j = A->start[i]; j < A->start[i+1]; ++j )
			fprintf( stderr, "%d(%.4f) ", A->entries[j].j, A->entries[j].val );

		fprintf( stderr, "\n" );
	}
}


bool validate_sort_matrix (reax_system *system, static_storage *workspace)
{
	sparse_matrix test;
	int index, count;
	test.start = (int *) malloc (INT_SIZE * (system->N + 1));
	test.end = (int *) malloc (INT_SIZE * (system->N + 1));

	test.entries = (sparse_matrix_entry *) malloc (SPARSE_MATRIX_ENTRY_SIZE * (system->N * system->max_sparse_matrix_entries));
	memset (test.entries, 0xFF, SPARSE_MATRIX_ENTRY_SIZE * system->N * system->max_sparse_matrix_entries);

	copy_host_device ( test.entries, dev_workspace->H.entries, SPARSE_MATRIX_ENTRY_SIZE * system->N * system->max_sparse_matrix_entries, 
			cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.start, dev_workspace->H.start, INT_SIZE * (system->N + 1), cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.end , dev_workspace->H.end, INT_SIZE * (system->N + 1), cudaMemcpyDeviceToHost, __LINE__ );

	//Print_Matrix ( &test );

	for (int i = 0; i < system->N; i++)
	{
		int start = test.start[i];
		int end = test.end [i];

		//d_quick_sort ( & (test.entries[start]), 0, end - start - 1 );
		for (int x = start; x < end-1; x++)
			if (test.entries[x].j > test.entries[x+1].j) {
				fprintf (stderr, "Matrix is not sorted for the entri %d \n", i );
				exit (-1);
			}
	}
	fprintf (stderr, " Done sorting with all the entries in the sparse matrix \n");

	free (test.start);
	free (test.end);
	free (test.entries);
}


bool validate_sparse_matrix( reax_system *system, static_storage *workspace )
{
	sparse_matrix test;
	int index, count;
	test.start = (int *) malloc (INT_SIZE * (system->N + 1));
	test.end = (int *) malloc (INT_SIZE * (system->N + 1));

	test.entries = (sparse_matrix_entry *) malloc (SPARSE_MATRIX_ENTRY_SIZE * (system->N * system->max_sparse_matrix_entries));

	memset (test.entries, 0xFF, SPARSE_MATRIX_ENTRY_SIZE * system->N * system->max_sparse_matrix_entries);
	copy_host_device ( test.entries, dev_workspace->H.entries, SPARSE_MATRIX_ENTRY_SIZE * system->N * system->max_sparse_matrix_entries, 
			cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.start, dev_workspace->H.start, INT_SIZE * (system->N + 1), cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.end , dev_workspace->H.end, INT_SIZE * (system->N + 1), cudaMemcpyDeviceToHost, __LINE__ );

	/*
	   for (int i = 0 ; i < system->N; i++) {
	   if ((test.end[i] - test.start[i]) != (workspace->H.start[i+1] - workspace->H.start[i])){
	//if ((test.end[i] - test.start[i]) < 32 ){
	fprintf (stderr, "Sparse Matrix gpu (%d %d) cpu (%d %d)\n", 
	test.start[i], test.end[i], 
	workspace->H.start[i], workspace->H.start[i+1]);
	exit (-1);
	}
	}
	 */
	//fprintf (stderr, "Sparse Matrix COUNT matches between HOST and DEVICE \n");

	count = 0;
	for (int i = 0; i < system->N; i++) {
		for (int j = workspace->H.start[i]; j < workspace->H.start[i+1]; j++) {
			sparse_matrix_entry *src = &workspace->H.entries[j];

			for (int k = test.start[i]; k < test.end[i]; k++) {
				sparse_matrix_entry *tgt = &test.entries [k];
				if (src->j == tgt->j){
					if ( check_zero (src->val, tgt->val)) {
						index = test.start [i];
						/*
						   fprintf (stderr, " i-1 (%d %d ) (%d %d) \n", 
						   test.start[i-1], test.end[i-1], 
						   workspace->H.start[i-1], workspace->H.start[i]);
						   fprintf (stderr, " Sparse matrix entry does not match for atom %d at index %d (%d %d) (%d %d) \n", 
						   i, k, test.start[i], test.end[i], 
						   workspace->H.start[i], workspace->H.start[i+1]);
						   for (int x = workspace->H.start[i]; x < workspace->H.start[i+1]; x ++)
						   {
						   src = &workspace->H.entries[x];
						   tgt = &test.entries [index];
						   fprintf (stderr, " cpu (%d %f)**** <--> gpu (%d %f) index %d \n", src->j, src->val, tgt->j, tgt->val, index);
						   index ++;
						   }
						 */
						fprintf (stderr, "Sparse Matrix DOES NOT match between device and host \n");
						exit (-1);
						count++;
					} else break;
				}
			}
		}
	}

	//fprintf (stderr, "Sparse Matrix mismatch count %d  \n", count);
	free (test.start);
	free (test.end);
	free (test.entries);
	return true;
}

bool validate_lu (static_storage *workspace)
{
	sparse_matrix test;
	int index, count;

	test.start = (int *) malloc (INT_SIZE * (dev_workspace->L.n + 1));
	test.end = (int *) malloc (INT_SIZE * (dev_workspace->L.n + 1));
	test.entries = (sparse_matrix_entry *) malloc (SPARSE_MATRIX_ENTRY_SIZE * (dev_workspace->L.m));

	memset (test.entries, 0xFF, SPARSE_MATRIX_ENTRY_SIZE * dev_workspace->L.m);
	copy_host_device ( test.entries, dev_workspace->L.entries, SPARSE_MATRIX_ENTRY_SIZE * dev_workspace->L.m, cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.start, dev_workspace->L.start, INT_SIZE * (dev_workspace->L.n + 1), cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.end , dev_workspace->L.end, INT_SIZE * (dev_workspace->L.n + 1), cudaMemcpyDeviceToHost, __LINE__ );

	count = 0;
	for (int i = 0; i < workspace->L.n; i ++)
	{
		if (workspace->L.start[i] != test.start[i]){
			fprintf (stderr, "L -- Count does not match for index %d \n", i);
			exit (-1);
		}

		for (int j = workspace->L.start[i]; j < workspace->L.start[i+1]; j++) 
		{
			if (check_zero (workspace->L.entries [j].val, test.entries[j].val) || 
					workspace->L.entries[j].j != test.entries [j].j)
			{
				fprintf (stderr, "L -- J or value does not match for the index %d \n", i);
				count ++;
				exit (-1);
			}
		}
	}

	test.start = (int *) malloc (INT_SIZE * (dev_workspace->U.n + 1));
	test.end = (int *) malloc (INT_SIZE * (dev_workspace->U.n + 1));
	test.entries = (sparse_matrix_entry *) malloc (SPARSE_MATRIX_ENTRY_SIZE * (dev_workspace->U.m));

	memset (test.entries, 0xFF, SPARSE_MATRIX_ENTRY_SIZE * dev_workspace->U.m);
	copy_host_device ( test.entries, dev_workspace->U.entries, SPARSE_MATRIX_ENTRY_SIZE * dev_workspace->U.m, cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.start, dev_workspace->U.start, INT_SIZE * (dev_workspace->U.n + 1), cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.end , dev_workspace->U.end, INT_SIZE * (dev_workspace->U.n + 1), cudaMemcpyDeviceToHost, __LINE__ );

	count = 0;
	for (int i = 0; i < workspace->U.n; i ++)
	{
		if (workspace->U.start[i] != test.start[i]){
			fprintf (stderr, "U -- Count does not match for index %d \n", i);
			exit (-1);
		}

		for (int j = workspace->U.start[i]; j < workspace->U.start[i+1]; j++) 
		{
			if (check_zero (workspace->U.entries [j].val, test.entries[j].val) || 
					workspace->U.entries[j].j != test.entries [j].j)
			{
				fprintf (stderr, "U -- J or value does not match for the index %d \n", i);
				count ++;
				exit (-1);
			}
		}
	}

	//fprintf (stderr, "L and U match on device and host \n");
	return true;
}

void print_sparse_matrix (reax_system *system, static_storage *workspace)
{
	sparse_matrix test;
	int index, count;

	test.start = (int *) malloc (INT_SIZE * (system->N + 1));
	test.end = (int *) malloc (INT_SIZE * (system->N + 1));

	test.entries = (sparse_matrix_entry *) malloc (SPARSE_MATRIX_ENTRY_SIZE * (system->N * system->max_sparse_matrix_entries));
	memset (test.entries, 0xFF, SPARSE_MATRIX_ENTRY_SIZE * system->N * system->max_sparse_matrix_entries);

	test.j = (int *)  malloc (INT_SIZE * (system->N * system->max_sparse_matrix_entries));
	test.val = (real *)  malloc (REAL_SIZE * (system->N * system->max_sparse_matrix_entries));

	copy_host_device ( test.entries, dev_workspace->H.entries, 
			SPARSE_MATRIX_ENTRY_SIZE * system->N * system->max_sparse_matrix_entries, cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.start, dev_workspace->H.start, INT_SIZE * (system->N + 1), cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.end , dev_workspace->H.end, INT_SIZE * (system->N + 1), cudaMemcpyDeviceToHost, __LINE__ );

	copy_host_device ( test.j , dev_workspace->H.j, INT_SIZE * (system->N * system->max_sparse_matrix_entries), cudaMemcpyDeviceToHost, __LINE__ );
	copy_host_device ( test.val , dev_workspace->H.val, REAL_SIZE * (system->N * system->max_sparse_matrix_entries), cudaMemcpyDeviceToHost, __LINE__ );

	count = 0;
	for (int i = 0; i < 1; i++) {
		//for (int j = workspace->H.start[i]; j < workspace->H.start[i+1]; j++) {
		//	sparse_matrix_entry *src = &workspace->H.entries[j];
		//	fprintf (stderr, " cpu (%d %f) \n", src->j, src->val);
		//}
		//fprintf (stderr, " start: %d -- end: %d  ------- count %d\n", test.start[i], test.end[i], test.end[i] - test.start[i]);
		for (int j = test.start[i]; j < test.end[i]; j++) {
			//sparse_matrix_entry *src = &test.entries[j];
			//fprintf (stderr, "Row:%d:%d:%f\n", i, src->j, src->val);
			fprintf (stderr, "Row:%d:%d:%f\n", i, test.j[j], test.val[j]);
		}

		//if (test.end[i] - test.start[i] > 500 )
		//	fprintf (stderr, " Row -- %d,  count %d \n", i, test.end[i] - test.start[i] );
	}
	fprintf (stderr, "--------------- ");

	free (test.start);
	free (test.end);
	free (test.entries);
	free (test.j);
	free (test.val);
}


bool validate_bonds (reax_system *system, static_storage *workspace, list **lists)
{
	int start, end, index, count, miscount;
	int *d_start, *d_end;
	bond_data *d_bond_data;
	list *d_bonds = dev_lists + BONDS;
	list *bonds = *lists + BONDS;

	d_end = (int *)malloc (sizeof (int) * system->N);
	d_start = (int *) malloc (sizeof (int) * system->N );
	d_bond_data = (bond_data *) malloc (BOND_DATA_SIZE * system->num_bonds );
	//fprintf (stderr, "Num bonds copied from device to host is --> %d \n", system->num_bonds );

	copy_host_device (d_start, d_bonds->index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
	copy_host_device (d_end, d_bonds->end_index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
	copy_host_device (d_bond_data, d_bonds->select.bond_list, BOND_DATA_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);

	count = 0;
	for (int i = 0; i < system->N; i++) {
		start = Start_Index (i, bonds);
		end = End_Index (i, bonds);

		count += end - start;
		if ((end-start) != (d_end[i]-d_start[i])){
			fprintf (stderr, "Entries does NOT match --> atom %d: cpu (%d %d) gpu (%d %d) \n", 
					i, start, end, d_start[i], d_end[i]);
			exit (-1);
		}

	}
	fprintf (stderr, "BOND LIST COUNT match on device and host  count %d \n", count);

	for (int i = 0; i < system->N-1; i++) {
		if ( d_end[i] >= d_start[i+1] ){
			fprintf (stderr, "Bonds list check Overwrite @ index --> %d \n", i);
			exit (-1);
		}
	}
	//fprintf (stderr, " BOND LIST Overwrite *PASSED* \n");

	count = 0;
	miscount = 0;
	for (int i = 0; i < system->N; i++) {

		for (int j = d_start[i]; j < d_end[i]; j++) {
			bond_data *src, *tgt;
			src = &d_bond_data[j];
			bond_data *src_sym = & d_bond_data[ src->sym_index ];

			//Previously this was commented out. Thats why it was working.
			//if (i >= src->nbr) continue;

			int k = 0;
			for (k = Start_Index (i, bonds); k < End_Index (i, bonds); k++) {
				tgt = & (bonds->select.bond_list[k]);
				bond_data *tgt_sym = &(bonds->select.bond_list [tgt->sym_index]);

				if ((src->nbr == tgt->nbr) && !check_zero (src->d,tgt->d) && 
						!check_zero (src->dvec,tgt->dvec) && check_same (src->rel_box, tgt->rel_box)) {

					bond_order_data *s, *t;
					s = &(src->bo_data);
					t = &(tgt->bo_data);

					/*
					   if (i == 45){
					   fprintf (stderr, " Host %e for %d\n", t->BO, tgt->nbr);
					   fprintf (stderr, " Device %e for %d\n", s->BO, src->nbr);
					   }
					 */

					if (	!check_zero (s->BO,t->BO) && 
							!check_zero (s->BO_s,t->BO_s) && 
							!check_zero(s->BO_pi,t->BO_pi)  && 
							!check_zero (s->BO_pi2,t->BO_pi2) &&
							!check_zero (s->Cdbo,t->Cdbo) && !check_zero (s->Cdbopi,t->Cdbopi) && !check_zero (s->Cdbopi2,t->Cdbopi2) &&
							!check_zero (s->C1dbo,t->C1dbo) && !check_zero (s->C2dbo,t->C2dbo) && !check_zero (s->C3dbo,t->C3dbo) &&
							!check_zero(s->C1dbopi,t->C1dbopi) && !check_zero(s->C2dbopi,t->C2dbopi) && !check_zero(s->C3dbopi,t->C3dbopi) && !check_zero(s->C4dbopi,t->C4dbopi) &&
							!check_zero(s->C1dbopi2,t->C1dbopi2) && !check_zero(s->C2dbopi2,t->C2dbopi2) &&!check_zero(s->C3dbopi2,t->C3dbopi2) &&!check_zero(s->C4dbopi2,t->C4dbopi2) &&
							!check_zero (s->dln_BOp_s, t->dln_BOp_s ) && 
							!check_zero (s->dln_BOp_pi, t->dln_BOp_pi ) && 
							!check_zero (s->dln_BOp_pi2, t->dln_BOp_pi2 ) && 
							!check_zero (s->dBOp, t->dBOp )) {
						count ++;

						//Check the sym index and dbond index here for double checking
						// bond_ij on both device and hosts are matched now. 
						bond_order_data *ss, *ts;
						ss = & (src_sym->bo_data );
						ts = & (tgt_sym->bo_data );

						if ((src_sym->nbr != tgt_sym->nbr) || check_zero (src_sym->d,tgt_sym->d) || 
								check_zero (src_sym->dvec,tgt_sym->dvec) || !check_same (src_sym->rel_box, tgt_sym->rel_box)
								|| check_zero (ss->Cdbo, ts->Cdbo)){

							fprintf (stderr, " Sym Index information does not match for atom %d \n", i);
							fprintf (stderr, " atom --> %d \n", i);
							fprintf (stderr, " nbr --> %d %d\n", src->nbr, tgt->nbr );
							fprintf (stderr, " d --> %f %f \n", src_sym->d, tgt_sym->d );
							fprintf (stderr, " sym Index nbr --> %d %d \n", src_sym->nbr, tgt_sym->nbr );
							fprintf (stderr, " dvec (%f %f %f) (%f %f %f) \n", 
									src_sym->dvec[0], src_sym->dvec[1], src_sym->dvec[2], 
									tgt_sym->dvec[0], tgt_sym->dvec[1], tgt_sym->dvec[2] );
							fprintf (stderr, " ivec (%d %d %d) (%d %d %d) \n", 
									src_sym->rel_box[0], src_sym->rel_box[1], src_sym->rel_box[2], 
									tgt_sym->rel_box[0], tgt_sym->rel_box[1], tgt_sym->rel_box[2] );

							fprintf (stderr, " sym index Cdbo (%4.10e %4.10e) \n", ss->Cdbo,ts->Cdbo );
							exit (-1);
						}

						break;
					}
					fprintf (stderr, " d --> %f %f \n", src->d, tgt->d );
					fprintf (stderr, " dvec (%f %f %f) (%f %f %f) \n", 
							src->dvec[0], src->dvec[1], src->dvec[2], 
							tgt->dvec[0], tgt->dvec[1], tgt->dvec[2] );
					fprintf (stderr, " ivec (%d %d %d) (%d %d %d) \n", 
							src->rel_box[0], src->rel_box[1], src->rel_box[2], 
							tgt->rel_box[0], tgt->rel_box[1], tgt->rel_box[2] );

					fprintf (stderr, "Bond_Order_Data does not match for atom %d neighbor (%d %d) BO (%e %e) BO_s (%e %e) BO_pi (%e %e) BO_pi2 (%e %e) \n", i, 
							src->nbr, tgt->nbr, 
							s->BO, t->BO, 
							s->BO_s, t->BO_s, 
							s->BO_pi, t->BO_pi, 
							s->BO_pi2, t->BO_pi2
						);
					fprintf (stderr, " dBOp (%e %e %e) (%e %e %e) \n", s->dBOp[0], s->dBOp[1], s->dBOp[2], 
							t->dBOp[0], t->dBOp[1], t->dBOp[2] );

					fprintf (stderr, " Cdbo (%4.10e %4.10e) \n", s->Cdbo,t->Cdbo );
					fprintf (stderr, " Cdbopi (%e %e) \n", s->Cdbopi,t->Cdbopi );
					fprintf (stderr, " Cdbopi2 (%e %e) \n", s->Cdbopi2,t->Cdbopi2 );
					fprintf (stderr, " C1dbo (%e %e %e)(%e %e %e) \n", s->C1dbo,s->C2dbo,s->C3dbo, t->C1dbo,t->C2dbo,t->C3dbo );
					fprintf (stderr, " C1dbopi (%e %e %e %e) (%e %e %e %e)\n", s->C1dbopi,s->C2dbopi,s->C3dbopi,s->C4dbopi, t->C1dbopi,t->C2dbopi,t->C3dbopi,t->C4dbopi);
					fprintf (stderr, " C1dbopi2 (%e %e %e %e) (%e %e %e %e)\n", s->C1dbopi2,s->C2dbopi2,s->C3dbopi2,s->C4dbopi2, t->C1dbopi2,t->C2dbopi2,t->C3dbopi2,t->C4dbopi2);
					fprintf (stderr, " dln_BOp_s (%e %e %e ) (%e %e %e) \n", 
							s->dln_BOp_s[0], s->dln_BOp_s[1], s->dln_BOp_s[2],
							t->dln_BOp_s[0], t->dln_BOp_s[1], t->dln_BOp_s[2] );
					fprintf (stderr, " dln_BOp_pi (%e %e %e ) (%e %e %e) \n", 
							s->dln_BOp_pi[0], s->dln_BOp_pi[1], s->dln_BOp_pi[2],
							t->dln_BOp_pi[0], t->dln_BOp_pi[1], t->dln_BOp_pi[2] );
					fprintf (stderr, " dln_BOp_pi2 (%e %e %e ) (%e %e %e) \n", 
							s->dln_BOp_pi2[0], s->dln_BOp_pi2[1], s->dln_BOp_pi2[2],
							t->dln_BOp_pi2[0], t->dln_BOp_pi2[1], t->dln_BOp_pi2[2] );

					//exit (-1);
				} 
			}

			if (k >= End_Index (i, bonds)) {
				miscount ++;
				fprintf (stderr, " We have a problem with the atom %d and bond entry %d \n", i, j);
				exit (-1);
			}
		}
	}

	fprintf (stderr, " Total bond order matched count %d miscount %d (%d) \n", count, miscount, (count+miscount));

	/*
	   for (int i = 5423; i < 5424; i++) {
	   start = Start_Index (i, bonds);
	   end = End_Index (i, bonds);

	   index = d_start[i];

	   fprintf (stderr, "Bond Count %d \n", end-start);
	   for (int j = start; j < end; j++)
	   {
	   bond_data src, tgt;
	   src = bonds->select.bond_list[j];
	   tgt = d_bond_data[index];
	   index ++;

	//compare here
	if ((src.nbr != tgt.nbr) || (src.d != tgt.d) ||
	memcmp (src.rel_box, tgt.rel_box, IVEC_SIZE) || 
	memcmp (src.dvec, tgt.dvec, RVEC_SIZE) ) {
	fprintf (stderr, "Entries does not MATCH with bond data at atom %d index %d \r\n src ( %d %f (%d %d %d) (%f %f %f) )  tgt (%d %f (%d %d %d) (%f %f %f))\n",
	i, j, 
	src.nbr, src.d, src.rel_box[0], src.rel_box[1], src.rel_box[2], 
	src.dvec[0], src.dvec[1], src.dvec[2],
	tgt.nbr, tgt.d, tgt.rel_box[0], tgt.rel_box[1], tgt.rel_box[2], 
	tgt.dvec[0], tgt.dvec[1], tgt.dvec[2] );
	}
	}
	}
	 */

	//fprintf (stderr, "BOND LIST match on device and host \n");

	free (d_start);
	free (d_end);
	free (d_bond_data);
	return true;
}

bool validate_sym_dbond_indices (reax_system *system, static_storage *workspace, list **lists)
{
	int start, end, index, count, miscount;
	int *d_start, *d_end;
	bond_data *d_bond_data;
	list *d_bonds = dev_lists + BONDS;
	list *bonds = *lists + BONDS;

	d_end = (int *)malloc (sizeof (int) * system->N);
	d_start = (int *) malloc (sizeof (int) * system->N );
	d_bond_data = (bond_data *) malloc (BOND_DATA_SIZE * system->num_bonds );
	//fprintf (stderr, "Num bonds copied from device to host is --> %d \n", system->num_bonds );

	copy_host_device (d_start, d_bonds->index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
	copy_host_device (d_end, d_bonds->end_index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
	copy_host_device (d_bond_data, d_bonds->select.bond_list, BOND_DATA_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);

	count = 0;
	miscount = 0;
	for (int i = 0; i < system->N; i++) {

		for (int j = d_start[i]; j < d_end[i]; j++) {
			bond_data *src, *tgt;
			src = &d_bond_data[j];

			tgt = &d_bond_data[ src->sym_index ];	

			if ((src->dbond_index == tgt->dbond_index) )
				count ++;
			else 
				miscount ++;
		}
	}
	fprintf (stderr, "Sym and dbond indexes done count(device) --> %d  (%d)\n", count, miscount);

	count = 0;
	miscount = 0;
	for (int i = 0; i < system->N; i++) {

		for (int j = Start_Index (i, bonds); j < End_Index(i, bonds); j++) {
			bond_data *src, *tgt;
			src = &bonds->select.bond_list [j];

			tgt = &bonds->select.bond_list [ src->sym_index ];	

			if ((src->dbond_index == tgt->dbond_index) )
				count ++;
			else 
				miscount ++;
		}
	}
	fprintf (stderr, "Sym and dbond indexes done count (host) --> %d  (%d)\n", count, miscount);

	free (d_start);
	free (d_end);
	free (d_bond_data);
	return true;
}

bool analyze_hbonds (reax_system *system, static_storage *workspace, list **lists)
{
	int hindex, nbr_hindex;
	int pj, hj, hb_start_j, hb_end_j, j, nbr;
	far_neighbor_data *nbr_pj;

	list *far_nbrs = *lists + FAR_NBRS;	
	list *hbonds = *lists + HBONDS;
	hbond_data *src, *tgt, *h_bond_data;
	int i, k, l;

	for (i = 0; i < system->N; i ++)
		for (pj = Start_Index (i, far_nbrs); pj < End_Index (i, far_nbrs); pj ++)
		{
			// check if the neighbor is of h_type
			nbr_pj = &( far_nbrs->select.far_nbr_list[pj] );
			j = nbr_pj->nbr;

			if (workspace->hbond_index [j] != -1)
			{    
				hb_start_j = Start_Index( workspace->hbond_index[j], hbonds );
				hb_end_j   = End_Index  ( workspace->hbond_index[j], hbonds );

				if (hb_start_j == hb_end_j) fprintf (stderr, "start == end \n");

				for ( hj = hb_start_j; hj < hb_end_j; hj ++ ) 
				{    
					h_bond_data = &( hbonds->select.hbond_list [hj] );
					nbr = h_bond_data->nbr;

					if (nbr == i) 
						fprintf (stderr, "found it for atom %d and neighbor %d neighbor %d \n", i, j , nbr);
					if (Start_Index (workspace->hbond_index [nbr], hbonds) == End_Index (workspace->hbond_index [nbr], hbonds))
						fprintf (stderr, " neighbor start == end \n");

					for ( k = Start_Index (workspace->hbond_index [nbr], hbonds);
							k < End_Index (workspace->hbond_index [nbr], hbonds);
							k ++)  
					{    
						if (hbonds->select.hbond_list [k].nbr == i) { 
							fprintf (stderr, "found it for atom %d and neighbor %d \n", i, j);
						}    
					}    
				}    
			}    
			else fprintf (stderr, "hbond index in workspace is -1\n");
		}


	for (i = 0; i < system->N; i++) 
	{
		hindex = workspace->hbond_index [i];
		if (hindex != -1) 
		{
			for (j = Start_Index ( hindex, hbonds ); j < End_Index ( hindex, hbonds ); j ++)
			{
				src = &hbonds->select.hbond_list [j];

				nbr_hindex = workspace->hbond_index [src->nbr];
				if (nbr_hindex == -1) {
					fprintf (stderr, " HBonds are NOT symmetric atom %d, neighbor %d\n", i, src->nbr);
					exit (-1);
				}

				for (k = Start_Index ( nbr_hindex, hbonds ); k < End_Index ( nbr_hindex, hbonds ); k++)
				{
					tgt = &hbonds->select.hbond_list [k];
					if ((tgt->nbr == i) && (src->scl == tgt->scl)) 
					{
						break;
					}
				}

				if ( k >= End_Index (nbr_hindex, hbonds)) {
					fprintf (stderr, " Could not find the other half of the hbonds \n");
					exit (-1);
				}
			}
		}
	}

	fprintf (stderr, "HBONDS list is symmetric \n");
}


bool validate_hbonds (reax_system *system, static_storage *workspace, list **lists)
{
	int *hbond_index, count;
	int *d_start, *d_end, index, d_index;
	hbond_data *data, src, tgt;
	list *d_hbonds = dev_lists + HBONDS;
	list *hbonds = *lists + HBONDS;

	hbond_index = (int *) malloc (INT_SIZE * system->N);
	copy_host_device (hbond_index, dev_workspace->hbond_index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);

	d_end = (int *)malloc (INT_SIZE * system->N);
	d_start = (int *) malloc (INT_SIZE * system->N );

	copy_host_device (d_start, d_hbonds->index, INT_SIZE * dev_workspace->num_H, cudaMemcpyDeviceToHost, __LINE__);
	copy_host_device (d_end, d_hbonds->end_index, INT_SIZE * dev_workspace->num_H, cudaMemcpyDeviceToHost, __LINE__);

	//fprintf (stderr, "Copying hbonds to host %d \n", system->num_hbonds);
	data = (hbond_data *) malloc (HBOND_DATA_SIZE * system->num_hbonds);
	copy_host_device (data, d_hbonds->select.hbond_list, HBOND_DATA_SIZE * system->num_hbonds, cudaMemcpyDeviceToHost, __LINE__);

	/*
	   Now the hbonds list is symmetric. will not work any longer

	   for (int i = 0; i < system->N; i++)
	   if (hbond_index[i] != workspace->hbond_index[i]) {
	   fprintf (stderr, "hbond index does not match for atom %d (%d %d)\n", 
	   i, workspace->hbond_index[i], hbond_index[i]);
	   exit (-1);
	   }

	 */

	//fprintf (stderr, "hbond_index match between host and device \n");

	for (int i = 0; i < system->N; i++) {

		if ( system->reaxprm.sbp[ system->atoms[i].type ].p_hbond == 1 )
		{
			if (hbond_index[i] >= 0) {
				if ((d_end[ hbond_index[i]] - d_start[hbond_index[i]])	 != 
						(End_Index (workspace->hbond_index[i], hbonds) - Start_Index (workspace->hbond_index[i], hbonds))) {
					fprintf (stderr, "%d %d - d(%d  %d) c(%d %d) \n",hbond_index[i], workspace->hbond_index[i],
							d_start[hbond_index[i]], d_end[ hbond_index[i]], 
							Start_Index (workspace->hbond_index[i], hbonds), 
							End_Index (workspace->hbond_index[i], hbonds) );
					exit (-1);
				}
			}
		}
	}
	//fprintf (stderr, "hbonds count match between host and device \n");

	count = 0;
	for (int i = 0; i < system->N; i++) {

		int d = workspace->hbond_index[i];
		if (d == -1) continue;

		d_index = hbond_index[i];
		/*
		   fprintf (stderr, " Count cpu %d gpu %d \n", 
		   End_Index (workspace->hbond_index[i], hbonds) - index, 
		   d_end[d_index] - d_start[d_index]);
		 */
		for (int j = d_start[d_index]; j < d_end[d_index]; j++ )
		{
			tgt = data[j];

			int k = 0;
			for (k = Start_Index (workspace->hbond_index[i], hbonds); 
					k < End_Index (workspace->hbond_index[i], hbonds); k++) {
				src = hbonds->select.hbond_list[k];

				if ((src.nbr == tgt.nbr) || (src.scl == tgt.scl)) {
					/*
					   fprintf (stderr, "Mismatch  at atom %d index %d (%d %d) -- (%d %d) \n", i, k,
					   src.nbr, src.scl, 
					   tgt.nbr, tgt.scl);
					 */
					count ++;
					break;
				}
			}

			/*
			   if ( 	((End_Index (workspace->hbond_index[i], hbonds) - index) != index ) && 
			   (k >= End_Index (workspace->hbond_index[i], hbonds))) {
			   fprintf (stderr, "Hbonds does not match for atom %d hbond_Index %d \n", i, d_index );
			   exit (-1);
			   }
			 */

			if ( k >= (End_Index (workspace->hbond_index[i], hbonds) )){
				fprintf (stderr, "Hbonds does not match for atom %d hbond_Index %d \n", i, j);
				exit (-1);
			}
		}

		if ((End_Index (workspace->hbond_index[i], hbonds)- Start_Index(workspace->hbond_index[i], hbonds)) != (d_end[d_index] - d_start[d_index])){
			fprintf (stderr, "End index does not match between device and host \n");
			exit (-1);
		}
	}

	//fprintf (stderr, "HBONDs match on device and Host count --> %d\n", count);

	free (d_start);
	free (d_end);
	free (data);
	return true;
}

bool validate_neighbors (reax_system *system, list **lists)
{
	list *far_nbrs = *lists + FAR_NBRS;
	list *d_nbrs = dev_lists + FAR_NBRS;
	far_neighbor_data gpu, cpu;
	int index, count, jicount;

	int *end = (int *)malloc (sizeof (int) * system->N);
	int *start = (int *) malloc (sizeof (int) * system->N );

	//fprintf (stderr, "numnbrs %d \n", system->num_nbrs);

	copy_host_device (start, d_nbrs->index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, 1);
	copy_host_device (end, d_nbrs->end_index, INT_SIZE * system->N, cudaMemcpyDeviceToHost, 2);

	far_neighbor_data *data = (far_neighbor_data *) malloc (FAR_NEIGHBOR_SIZE * system->num_nbrs);
	copy_host_device (data, d_nbrs->select.far_nbr_list, FAR_NEIGHBOR_SIZE * system->num_nbrs, cudaMemcpyDeviceToHost, 3);

	int cpu_count = 0;
	int gpu_count = 0;

	for (int i = 0; i < system->N; i++){
		cpu_count += Num_Entries (i, far_nbrs);
		gpu_count += end[i] - start[i];
	}

	//fprintf (stderr, " Nbrs count cpu: %d -- gpu: %d \n", cpu_count, gpu_count );
	for (int i = 0; i < system->N-1; i++){
		if (end [i] > start [i+1])
		{
			fprintf (stderr, " Far Neighbors index over write  @ index %d\n", i);
			exit (-1);
		}
	}



	for (int i = 0; i < system->N; i++){
		index = Start_Index (i, far_nbrs);

		for (int j = start[i]; j < end[i]; j++){
			gpu = data[j];

			if (i < data[j].nbr) {
				int src = data[j].nbr;
				int dest = i;
				int x;


				for (x = start[src]; x < end[src]; x++) {
					if (data[x].nbr != dest) continue;

					gpu = data[x];
					cpu = data[j];

					if (  (gpu.d != cpu.d) ||
							(cpu.dvec[0] != gpu.dvec[0]) || (cpu.dvec[1] != gpu.dvec[1]) || (cpu.dvec[2] != gpu.dvec[2]) ||
							(cpu.rel_box[0] != gpu.rel_box[0]) || (cpu.rel_box[1] != gpu.rel_box[1]) || (cpu.rel_box[2] != gpu.rel_box[2])) {
						fprintf (stderr, " atom %d neighbor %d  (%f, %d, %d, %d - %f %f %f) \n", i, data[j].nbr, 
								data[j].d, 
								data[j].rel_box[0],
								data[j].rel_box[1],
								data[j].rel_box[2],
								data[j].dvec[0], 
								data[j].dvec[1], 
								data[j].dvec[2] 
							);
						fprintf (stderr, " atom %d neighbor %d  (%f, %d, %d, %d - %f %f %f) \n", data[j].nbr, data[x].nbr,
								data[x].d,
								data[x].rel_box[0],
								data[x].rel_box[1],
								data[x].rel_box[2],
								data[x].dvec[0],
								data[x].dvec[1],
								data[x].dvec[2]
							);
						jicount++;

						fprintf (stderr, " Far Neighbors DOES NOT match between Deivce and Host \n");
						exit (-1);
					}
					break;
				}

				if (x >= end[src]) {
					fprintf (stderr, "could not find the neighbor duplicate data for ij (%d %d)\n", i, src );
					exit (-1);
				}

				continue;
			}

			cpu = far_nbrs->select.far_nbr_list[index];
			//if (  (gpu.d != cpu.d) || (gpu.nbr != cpu.nbr) ||
			//     (cpu.dvec[0] != gpu.dvec[0]) || (cpu.dvec[1] != gpu.dvec[1]) || (cpu.dvec[2] != gpu.dvec[2]) ||
			//    (cpu.rel_box[0] != gpu.rel_box[0]) || (cpu.rel_box[1] != gpu.rel_box[1]) || (cpu.rel_box[2] != gpu.rel_box[2])) {
			//if (memcmp (&gpu, &cpu, FAR_NEIGHBOR_SIZE)) {
			if (  check_zero (gpu.d, cpu.d) || 
					(gpu.nbr != cpu.nbr) ||
					check_zero (cpu.dvec, gpu.dvec) || 
					!check_same (cpu.rel_box, gpu.rel_box)) {

				fprintf (stderr, "GPU:atom --> %d (s: %d , e: %d, i: %d )\n", i, start[i], end[i], j );
				fprintf (stderr, "CPU:atom --> %d (s: %d , e: %d, i: %d )\n", i, Start_Index(i, far_nbrs), End_Index (i, far_nbrs), index);
				fprintf (stdout, "Far neighbors does not match atom: %d \n", i );
				fprintf (stdout, "neighbor %d ,  %d \n",  cpu.nbr, gpu.nbr);
				fprintf (stdout, "d %f ,  %f \n", cpu.d, data[j].d);
				fprintf (stdout, "dvec (%f %f %f) (%f %f %f) \n", 
						cpu.dvec[0], cpu.dvec[1], cpu.dvec[2],
						gpu.dvec[0], gpu.dvec[1], gpu.dvec[2] );

				fprintf (stdout, "rel_box (%d %d %d) (%d %d %d) \n", 
						cpu.rel_box[0], cpu.rel_box[1], cpu.rel_box[2],
						gpu.rel_box[0], gpu.rel_box[1], gpu.rel_box[2] );

				fprintf (stderr, " Far Neighbors DOES NOT match between Deivce and Host  **** \n");
				exit (-1);
				count ++;
			}
			index ++;
		}    

		if (index != End_Index (i, far_nbrs))
		{    
			fprintf (stderr, "End index does not match for atom --> %d end index (%d) Cpu (%d, %d ) gpu (%d, %d)\n", i, index, Start_Index (i, far_nbrs), End_Index(i, far_nbrs),
					start[i], end[i]);
			exit (10);
		}    
		}

		//fprintf (stderr, "FAR Neighbors match between device and host \n");
		free (start);
		free (end);
		free (data);
		return true;
		}

		bool validate_workspace (reax_system *system, static_storage *workspace, list **lists) 
		{
			real *total_bond_order;
			int count, tcount;

			total_bond_order = (real *) malloc ( system->N * REAL_SIZE );
			copy_host_device (total_bond_order, dev_workspace->total_bond_order, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < system->N; i++) {

				//if (abs (workspace->total_bond_order[i] - total_bond_order[i]) >= GPU_TOLERANCE){
				if ( check_zero (workspace->total_bond_order[i], total_bond_order[i])){
					fprintf (stderr, "Total bond order does not match for atom %d (%4.15e %4.15e)\n",
							i, workspace->total_bond_order[i], total_bond_order[i]);
					exit (-1);
					count ++;
				}
			}
			free (total_bond_order);
			//fprintf (stderr, "TOTAL Bond Order mismatch count %d\n", count);


			rvec *dDeltap_self;
			dDeltap_self = (rvec *) calloc (system->N, RVEC_SIZE);
			copy_host_device (dDeltap_self, dev_workspace->dDeltap_self, system->N * RVEC_SIZE, cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < system->N; i++ )
			{
				if (check_zero (workspace->dDeltap_self[i], dDeltap_self[i]))
				{
					fprintf (stderr, "index: %d c (%f %f %f) g (%f %f %f )\n", i, 
							workspace->dDeltap_self[i][0],
							workspace->dDeltap_self[i][1],
							workspace->dDeltap_self[i][2],
							dDeltap_self[3*i+0],
							dDeltap_self[3*i+1],
							dDeltap_self[3*i+2] );
					exit (-1);
					count ++;
				}
			}
			free (dDeltap_self);
			//fprintf (stderr, "dDeltap_self mismatch count %d\n", count);

			//exit for init_forces

			real *test;
			test = (real *) malloc (system->N * REAL_SIZE);

			copy_host_device (test, dev_workspace->Deltap, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ )
			{
				if (check_zero (workspace->Deltap[i], test[i]))
				{
					fprintf (stderr, "Deltap: Mismatch index --> %d (%f %f) \n", i, workspace->Deltap[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "Deltap mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->Deltap_boc, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ )
			{
				if (check_zero (workspace->Deltap_boc[i], test[i]))
				{
					fprintf (stderr, "Deltap_boc: Mismatch index --> %d (%f %f) \n", i, workspace->Deltap_boc[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "dDeltap_boc mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->Delta, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->Delta[i], test[i])) {
					fprintf (stderr, "Delta: Mismatch index --> %d (%f %f) \n", i, workspace->Delta[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "Delta mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->Delta_e, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->Delta_e[i], test[i])) {
					fprintf (stderr, "Delta_e: Mismatch index --> %d (%f %f) \n", i, workspace->Delta_e[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "Delta_e mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->vlpex, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->vlpex[i], test[i])) {
					fprintf (stderr, "vlpex: Mismatch index --> %d (%f %f) \n", i, workspace->vlpex[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "vlpex mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->nlp, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->nlp[i], test[i])) {
					fprintf (stderr, "nlp: Mismatch index --> %d (%f %f) \n", i, workspace->nlp[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "nlp mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->Delta_lp, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->Delta_lp[i], test[i])) {
					fprintf (stderr, "Delta_lp: Mismatch index --> %d (%f %f) \n", i, workspace->Delta_lp[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "Delta_lp mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->Clp, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->Clp[i], test[i])) {
					fprintf (stderr, "Clp: Mismatch index --> %d (%f %f) \n", i, workspace->Clp[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "Clp mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->dDelta_lp, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->dDelta_lp[i], test[i])) {
					fprintf (stderr, "dDelta_lp: Mismatch index --> %d (%f %f) \n", i, workspace->dDelta_lp[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "dDelta_lp mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->nlp_temp, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->nlp_temp[i], test[i])) {
					fprintf (stderr, "nlp_temp: Mismatch index --> %d (%f %f) \n", i, workspace->nlp_temp[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "nlp_temp mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->Delta_lp_temp, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->Delta_lp_temp[i], test[i])) {
					fprintf (stderr, "Delta_lp_temp: Mismatch index --> %d (%f %f) \n", i, workspace->Delta_lp_temp[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "Delta_lp_temp mismatch count %d\n", count);

			copy_host_device (test, dev_workspace->dDelta_lp_temp, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->dDelta_lp_temp[i], test[i])) {
					fprintf (stderr, "dDelta_lp_temp: Mismatch index --> %d (%f %f) \n", i, workspace->dDelta_lp_temp[i], test[i]);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "dDelta_lp_temp mismatch count %d\n", count);

			//exit for Bond order calculations


			copy_host_device (test, dev_workspace->CdDelta, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->CdDelta[i], test[i])) {
					fprintf (stderr, " CdDelta does NOT match (%f %f) for atom  %d \n", workspace->CdDelta[i], test[i], i);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "CdDelta mismatch count %d\n", count);
			//exit for Bond Energy calculations

			/*
			   copy_host_device (test, dev_workspace->droptol, system->N * REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);
			   count = 0;
			   for (int i = 0; i < system->N; i++ ) {
			   if (check_zero (workspace->droptol[i], test[i])) {
			   fprintf (stderr, " Droptol Does not match (%f %f) \n", workspace->droptol[i], test[i]);
			   exit (-1);
			   count ++;
			   }
			   }
			//fprintf (stderr, "droptol mismatch count %d\n", count);
			 */


			//exit for  QEa calculations
			/*
			   real *t_s;

			   t_s = (real *) malloc (REAL_SIZE * (system->N * 2) );
			   copy_host_device (t_s, dev_workspace->b_prm, REAL_SIZE * (system->N * 2), cudaMemcpyDeviceToHost, __LINE__);

			   count = 0;
			   for (int i = 0; i < (system->N * 2); i++ ) {
			   if (check_zero (workspace->b_prm[i], t_s[i])) {
			   fprintf (stderr, " (%f %f) \n", workspace->b_prm[i], t_s[i]);
			   exit (-1);
			   count ++;
			   }
			   }
			//fprintf (stderr, "b_prm mismatch count %d\n", count);

			t_s = (real *) malloc (REAL_SIZE * 5 * system->N);
			copy_host_device (t_s, dev_workspace->s, system->N * REAL_SIZE * 5, cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < 5*system->N; i++ ) {
			if (check_zero (workspace->s[i], t_s[i])) {
			//fprintf (stderr, " (%f %f)  @ index %d \n", workspace->s[i], t_s[i], i);
			count ++;
			}
			}
			fprintf (stderr, "s mismatch count %d\n", count);


			t_s = (real *) malloc (REAL_SIZE * 5 * system->N);
			copy_host_device (t_s, dev_workspace->t, system->N * REAL_SIZE * 5, cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < 5*system->N; i++ ) {
			if (check_zero (workspace->t[i], t_s[i])) {
			//fprintf (stderr, " (%f %f) @ index : %d\n", workspace->t[i], t_s[i], i);
			count ++;
			}
			}
			fprintf (stderr, "t mismatch count %d\n", count);


			t_s = (real *) malloc (REAL_SIZE * (RESTART+1) * system->N);
			copy_host_device (t_s, dev_workspace->v, system->N * REAL_SIZE * (RESTART+1), cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < (RESTART + 1)*system->N; i++ ) {
			if (check_zero (workspace->v[i], t_s[i])) {
			//fprintf (stderr, " (%f %f) @ index %d \n", workspace->v[i], t_s[i], i);
			count ++;
			}
			}
			fprintf (stderr, "v mismatch count %d\n", count);

			t_s = (real *) malloc (REAL_SIZE * (RESTART+1) );
			copy_host_device (t_s, dev_workspace->y, REAL_SIZE * (RESTART+1), cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < (RESTART + 1); i++ ) {
			if (check_zero (workspace->y[i], t_s[i])) {
			//fprintf (stderr, " (%f %f) \n", workspace->y[i], t_s[i]);
			count ++;
			}
			}
			fprintf (stderr, "y mismatch count %d\n", count);

			t_s = (real *) malloc (REAL_SIZE * (RESTART+1) );
			copy_host_device (t_s, dev_workspace->hc, REAL_SIZE * (RESTART+1), cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < (RESTART + 1); i++ ) {
			if (check_zero (workspace->hc[i], t_s[i])) {
				//fprintf (stderr, " (%f %f) \n", workspace->hc[i], t_s[i]);
				count ++;
			}
			}
			fprintf (stderr, "hc mismatch count %d\n", count);

			t_s = (real *) malloc (REAL_SIZE * (RESTART+1) );
			copy_host_device (t_s, dev_workspace->hs, REAL_SIZE * (RESTART+1), cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < (RESTART + 1); i++ ) {
				if (check_zero (workspace->hs[i], t_s[i])) {
					//fprintf (stderr, " (%f %f) \n", workspace->hs[i], t_s[i]);
					count ++;
				}
			}
			fprintf (stderr, "hs mismatch count %d\n", count);

			t_s = (real *) malloc (REAL_SIZE * (RESTART+1) * (RESTART+1) );
			copy_host_device (t_s, dev_workspace->h, REAL_SIZE * (RESTART+1)*(RESTART+1), cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < (RESTART+1)*(RESTART+1); i++ ) {
				if (check_zero (workspace->h[i], t_s[i])) {
					//fprintf (stderr, " (%f %f) \n", workspace->h[i], t_s[i]);
					count ++;
				}
			}
			fprintf (stderr, "h mismatch count %d\n", count);

			t_s = (real *) malloc (REAL_SIZE * (RESTART+1) );
			copy_host_device (t_s, dev_workspace->g, REAL_SIZE * (RESTART+1), cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < (RESTART + 1); i++ ) {
				if (check_zero (workspace->g[i], t_s[i])) {
					//fprintf (stderr, " (%f %f) @ index %d\n", workspace->g[i], t_s[i], i);
					count ++;
				}
			}
			fprintf (stderr, "g mismatch count %d\n", count);
			*/

				rvec *r_s = (rvec *) malloc (RVEC_SIZE * system->N );
			copy_host_device (r_s, dev_workspace->v_const, RVEC_SIZE * system->N,  cudaMemcpyDeviceToHost, __LINE__);

			count = 0;
			for (int i = 0; i < system->N; i++ ) {
				if (check_zero (workspace->v_const[i], r_s[i])) {
					fprintf (stderr, " v_const (%f %f %f) (%f %f %f) @ index %d\n", 
							workspace->v_const[i][0], 
							workspace->v_const[i][1], 
							workspace->v_const[i][2], 
							r_s[i][0], 
							r_s[i][1], 
							r_s[i][2], 
							i);
					exit (-1);
					count ++;
				}
			}
			//fprintf (stderr, "v_const mismatch count %d\n", count);

			free (test);
			free (r_s);
			return true;
			}

			bool validate_data (reax_system *system, simulation_data *host)
			{
				simulation_data device;

				copy_host_device (&device, host->d_simulation_data, SIMULATION_DATA_SIZE, cudaMemcpyDeviceToHost, __LINE__);

				if (check_zero (host->E_BE, device.E_BE)){
					fprintf (stderr, "E_BE does not match (%4.15e %4.15e) \n", host->E_BE, device.E_BE);
					exit (-1);
				}

				if (check_zero (host->E_Lp, device.E_Lp)){
					fprintf (stderr, "E_Lp does not match (%4.10e %4.10e) \n", host->E_Lp, device.E_Lp);
					exit (-1);
				}

				if (check_zero (host->E_Ov, device.E_Ov)){
					fprintf (stderr, "E_Ov does not match (%4.10e %4.10e) \n", host->E_Ov, device.E_Ov);
					exit (-1);
				}

				if (check_zero (host->E_Un, device.E_Un)){
					fprintf (stderr, "E_Un does not match (%4.10e %4.10e) \n", host->E_Un, device.E_Un);
					exit (-1);
				}

				if (check_zero (host->E_Tor, device.E_Tor)) {
					fprintf (stderr, "E_Tor does not match (%4.10e %4.10e) \n", host->E_Tor, device.E_Tor);
					exit (-1);
				}

				if (check_zero (host->E_Con, device.E_Con)) {
					fprintf (stderr, "E_Con does not match (%4.10e %4.10e) \n", host->E_Con, device.E_Con);
					exit (-1);
				}

				if (check_zero (host->ext_press, device.ext_press)) {
					fprintf (stderr, "ext_press does not match (%4.10e %4.10e) \n", host->ext_press, device.ext_press);
					exit (-1);
				}

				if (check_zero (host->E_HB, device.E_HB)) {
					fprintf (stderr, "E_Hb does not match (%4.10e %4.10e) \n", host->E_HB, device.E_HB);
					exit (-1);
				}

				if (check_zero (host->E_Ang, device.E_Ang)) {
					fprintf (stderr, "E_Ang does not match (%4.10e %4.10e) \n", host->E_Ang, device.E_Ang);
					exit (-1);
				}

				if (check_zero (host->E_Pen, device.E_Pen)) {
					fprintf (stderr, "E_Pen does not match (%4.10e %4.10e) \n", host->E_Pen, device.E_Pen);
					exit (-1);
				}

				if (check_zero (host->E_Coa, device.E_Coa)) {
					fprintf (stderr, "E_Coa does not match (%4.10e %4.10e) \n", host->E_Coa, device.E_Coa);
					exit (-1);
				}

				if (check_zero (host->E_vdW, device.E_vdW)) {
					fprintf (stderr, "E_vdW does not match (%4.20e %4.20e) \n", host->E_vdW, device.E_vdW);
					exit (-1);
				}

				if (check_zero (host->E_Ele, device.E_Ele)) {
					fprintf (stderr, "E_Ele does not match (%4.20e %4.20e) \n", host->E_Ele, device.E_Ele);
					exit (-1);
				}

				if (check_zero (host->E_Pol, device.E_Pol)) {
					fprintf (stderr, "E_Pol does not match (%4.10e %4.10e) \n", host->E_Pol, device.E_Pol);
					exit (-1);
				}


				//fprintf (stderr, "Simulation Data match between host and device \n");
				return true;
			}

			void print_bond_data (bond_order_data *s)
			{
				/*
				   fprintf (stderr, "Bond_Order_Data BO (%f ) BO_s (%f ) BO_pi (%f ) BO_pi2 (%f ) ", 
				   s->BO, 
				   s->BO_s, 
				   s->BO_pi,
				   s->BO_pi2 );
				 */
				fprintf (stderr, " Cdbo (%e) ", s->Cdbo );
				fprintf (stderr, " Cdbopi (%e) ", s->Cdbopi );
				fprintf (stderr, " Cdbopi2 (%e) ", s->Cdbopi2 );
			}

			void print_bond_list (reax_system *system, static_storage *workspace, list **lists)
			{
				list *bonds = *lists + BONDS;

				for (int i = 1; i < 2; i++)
				{
					fprintf (stderr, "Atom %d Bond_data ( nbrs \n", i);
					for (int j = Start_Index (i, bonds); j < End_Index (i, bonds); j++) 
					{
						bond_data *data = &bonds->select.bond_list [j];
						fprintf (stderr, "  %d, ", data->nbr );
						print_bond_data (&data->bo_data);
						fprintf (stderr, ")\n");
					}
				}

				int *b_start = (int *) malloc (INT_SIZE * system->N);
				int *b_end = (int *) malloc (INT_SIZE * system->N);
				list *d_bonds = dev_lists + BONDS;
				bond_data *d_bond_data;

				d_bond_data = (bond_data *) malloc (BOND_DATA_SIZE * system->num_bonds );

				copy_host_device ( b_start, d_bonds->index, 
						INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( b_end, d_bonds->end_index, 
						INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device (d_bond_data, d_bonds->select.bond_list, BOND_DATA_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);
				for (int i = 0; i < 2; i++)
				{
					fprintf (stderr, "Atom %d Bond_data ( nbrs \n", i);
					for (int j = b_start[i]; j < b_end[i]; j ++) {
						bond_data *src = &d_bond_data[j];
						fprintf (stderr, "  %d, ", src->nbr );
						print_bond_data (&src->bo_data);
						fprintf (stderr, ")\n");
					}
				}
			}



			void count_three_bodies (reax_system *system, static_storage *workspace, list **lists)
			{
				list *three = *lists + THREE_BODIES;
				list *bonds = *lists + BONDS;

				list *d_three = dev_lists + THREE_BODIES;
				list *d_bonds = dev_lists + BONDS;
				bond_data *d_bond_data;
				real *test;

				three_body_interaction_data *data = (three_body_interaction_data *) 
					malloc ( sizeof (three_body_interaction_data) * system->num_thbodies);
				int *start = (int *) malloc (INT_SIZE * system->num_bonds);
				int *end = (int *) malloc (INT_SIZE * system->num_bonds);

				int *b_start = (int *) malloc (INT_SIZE * system->N);
				int *b_end = (int *) malloc (INT_SIZE * system->N);
				int count;
				int hcount, dcount;

				copy_host_device ( start, d_three->index, 
						INT_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( end, d_three->end_index, 
						INT_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( data, d_three->select.three_body_list, 
						sizeof (three_body_interaction_data) * system->num_thbodies, 
						cudaMemcpyDeviceToHost, __LINE__);

				d_bond_data = (bond_data *) malloc (BOND_DATA_SIZE * system->num_bonds );

				copy_host_device ( b_start, d_bonds->index, 
						INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( b_end, d_bonds->end_index, 
						INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device (d_bond_data, d_bonds->select.bond_list, BOND_DATA_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);

				count = 0;
				hcount = dcount = 0;
				for (int i = 0; i < system->N; i++)
				{
					for (int j = b_start[i]; j < b_end[i]; j ++) {
						dcount += end[j] - start[j];
					}
				}

				fprintf (stderr, "Total Actual Three Body Count ---> %d \n", dcount);

				free (data);
				free (start);
				free (end);
				free (b_start);
				free (b_end);
				free (d_bond_data);
			}



			bool validate_three_bodies (reax_system *system, static_storage *workspace, list **lists)
			{
				list *three = *lists + THREE_BODIES;
				list *bonds = *lists + BONDS;

				list *d_three = dev_lists + THREE_BODIES;
				list *d_bonds = dev_lists + BONDS;
				bond_data *d_bond_data;
				real *test;

				three_body_interaction_data *data = (three_body_interaction_data *) 
					malloc ( sizeof (three_body_interaction_data) * system->num_thbodies);
				int *start = (int *) malloc (INT_SIZE * system->num_bonds);
				int *end = (int *) malloc (INT_SIZE * system->num_bonds);

				int *b_start = (int *) malloc (INT_SIZE * system->N);
				int *b_end = (int *) malloc (INT_SIZE * system->N);
				int count;
				int hcount, dcount;



				copy_host_device ( start, d_three->index, 
						INT_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( end, d_three->end_index, 
						INT_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( data, d_three->select.three_body_list, 
						sizeof (three_body_interaction_data) * system->num_thbodies, 
						cudaMemcpyDeviceToHost, __LINE__);

				d_bond_data = (bond_data *) malloc (BOND_DATA_SIZE * system->num_bonds );

				copy_host_device ( b_start, d_bonds->index, 
						INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( b_end, d_bonds->end_index, 
						INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device (d_bond_data, d_bonds->select.bond_list, BOND_DATA_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);

				//test = (real *) malloc (REAL_SIZE * system->num_bonds);
				//memset (test, 0, REAL_SIZE * system->num_bonds);
				//copy_host_device (test, testdata, REAL_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);

				count = 0;
				for (int i = 0; i < system->N; i++)
				{
					//for (int j = bonds->index[i]; j < bonds->end_index[i]; j ++)

					hcount = dcount = 0;
					for (int j = b_start[i]; j < b_end[i]; j ++) {
						dcount += end[j] - start[j];
						hcount += Num_Entries (j, three);

						/*
						   if ((end[j] - start[j]) != (End_Index (j, three) - Start_Index (j, three)))
						   {
						   fprintf (stderr, " Three body count does not match between host and device\n");
						   fprintf (stderr, " Host count : (%d, %d)\n", Start_Index (j, three), End_Index (j, three));
						   fprintf (stderr, " Device count: (%d, %d)\n", start[j], end[j]);
						   }
						 */
					}


					if ((dcount != hcount)) {

						fprintf (stderr, " Three body count does not match for the bond %d - %d \n", hcount, dcount); 

						for (int j = b_start[i]; j < b_end[i]; j ++) {
							bond_order_data *src = &d_bond_data[j].bo_data;
							dcount = end[j] - start[j];
							hcount = Num_Entries (j, three);
							fprintf (stderr, "device \n");
							print_bond_data (src);

							fprintf (stderr, "\n");
							src = &bonds->select.bond_list[j].bo_data;
							fprintf (stderr, "host \n");
							print_bond_data (src);
							fprintf (stderr, "\n");

							//fprintf (stderr, "--- Device bo is %f \n", test[j]);
							fprintf (stderr, "Device %d %d bonds (%d %d) - Host %d %d bonds (%d %d) \n", start[j], end[j],b_start[i], b_end[i],  
									Start_Index (j, three), End_Index (j, three), Start_Index (i, bonds), End_Index (i, bonds));
							fprintf (stderr, "Host %d Device %d -- atom %d index %d \n", hcount, dcount, i, j);
							fprintf (stderr, "------\n");
						}
						fprintf (stderr, " Three Bodies count does not match between host and device \n");
						exit (-1);
					}
				}

				//fprintf (stderr, "Three body count on DEVICE %d  HOST %d \n", dcount, hcount);

				count = 0;
				for (int i = 0; i < system->N; i++)
				{
					int x, y, z;
					for (x = b_start[i]; x < b_end[i]; x++)
					{
						int t_start = start[x];
						int t_end = end[x];

						bond_data *dev_bond = &d_bond_data [x];
						bond_data *host_bond;
						for (z = Start_Index (i, bonds); z < End_Index (i, bonds); z++)
						{
							host_bond = &bonds->select.bond_list [z];
							if ((dev_bond->nbr == host_bond->nbr) &&
									check_same (dev_bond->rel_box, host_bond->rel_box) && 
									!check_zero (dev_bond->dvec, host_bond->dvec) &&
									!check_zero (dev_bond->d, host_bond->d) )
							{
								break;
							}
						}
						if (z >= End_Index (i, bonds)){
							fprintf (stderr, "Could not find the matching bond on host and device \n");
							exit (-1);
						}

						//find this bond in the bonds on the host side.

						for (y = t_start; y < t_end; y++)
						{

							three_body_interaction_data *device = data + y;
							three_body_interaction_data *host;

							//fprintf (stderr, "Device thb %d pthb %d \n", device->thb, device->pthb);

							int xx;	
							for (xx = Start_Index (z, three); xx < End_Index (z, three); xx++)
							{
								host = &three->select.three_body_list [xx];
								//fprintf (stderr, "Host thb %d pthb %d \n", host->thb, host->pthb);
								//if ((host->thb == device->thb) && (host->pthb == device->pthb))
								if ((host->thb == device->thb) && !check_zero (host->theta, device->theta))
								{
									count ++;
									break;
								}
							}

							if ( xx >= End_Index (z, three) ) {
								fprintf (stderr, " Could not match for atom %d bonds %d (%d) Three body(%d %d) (%d %d) \n", i, x, z, 
										Start_Index (z, three), End_Index (z, three), start[x], end[x] );
								exit (-1);
							}// else fprintf (stderr, "----------------- \n");
						}
					}
				}
				free (data);
				free (start);
				free (end);
				free (b_start);
				free (b_end);
				free (d_bond_data);

				//fprintf (stderr, "Three Body Interaction Data MATCH on device and HOST --> %d \n", count);
				return true;
			}

			bool bin_three_bodies (reax_system *system, static_storage *workspace, list **lists)
			{
				list *d_three = dev_lists + THREE_BODIES;
				list *d_bonds = dev_lists + BONDS;
				list *three = *lists + THREE_BODIES;
				list *bonds = *lists + BONDS;
				bond_data *d_bond_data;

				three_body_interaction_data *data = (three_body_interaction_data *) 
					malloc ( sizeof (three_body_interaction_data) * system->num_thbodies);
				int *start = (int *) malloc (INT_SIZE * system->num_bonds);
				int *end = (int *) malloc (INT_SIZE * system->num_bonds);

				int *b_start = (int *) malloc (INT_SIZE * system->N);
				int *b_end = (int *) malloc (INT_SIZE * system->N);

				int *a = (int *) malloc (2 * INT_SIZE * system->N );
				int *b = (int *) malloc (2 * INT_SIZE * system->N );
				int *c = (int *) malloc (2 * INT_SIZE * system->N );
				int *d = (int *) malloc (2 * INT_SIZE * system->N );

				for (int i = 0; i < 2 * system->N; i++)
					a[i] = b[i] = c[i] = d[i] = -1;

				int count;
				int hcount, dcount;
				int index_a, index_b, index_c, index_d;
				index_a = index_b = index_c = index_d = 0;

				copy_host_device ( start, d_three->index, 
						INT_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( end, d_three->end_index, 
						INT_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( data, d_three->select.three_body_list, 
						sizeof (three_body_interaction_data) * system->num_thbodies, 
						cudaMemcpyDeviceToHost, __LINE__);

				d_bond_data = (bond_data *) malloc (BOND_DATA_SIZE * system->num_bonds );

				copy_host_device ( b_start, d_bonds->index, 
						INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device ( b_end, d_bonds->end_index, 
						INT_SIZE * system->N, cudaMemcpyDeviceToHost, __LINE__);
				copy_host_device (d_bond_data, d_bonds->select.bond_list, BOND_DATA_SIZE * system->num_bonds, cudaMemcpyDeviceToHost, __LINE__);

				count = 0;
				hcount = dcount = 0;

				/*
				   for (int i = 0; i < 20; i++)
				   {
				   for (int j = Start_Index (i, bonds); j < End_Index (i, bonds); j++)
				   {
				   for ( int k = Start_Index (j, three); k < End_Index (j, three); k ++)
				   {
				   three_body_interaction_data *host = &three->select.three_body_list [k];
				   fprintf (stderr, " atom %d bond (%d %d) -- %d,  (%d %d)\n", 
				   i, Start_Index (i, bonds), End_Index (i, bonds), j, host->thb, host->pthb );

				   }
				   }
				   }
				   exit (-1);
				 */

				count = 0;
				for (int i = 0; i < system->N; i++)
				{
					for (int j = b_start[i]; j < b_end[i]; j ++) {

						/*
						   bond_data *src;
						   src = &d_bond_data[j];
						   fprintf (stderr, " atom %d Neighbor %d \n", i, src->nbr );
						 */

						for (int x = start[j]; x < end[j]; x ++)
						{
							three_body_interaction_data *device = data + x;

							int center = device->j;
							int d_i = device->i;
							int d_k = device->k;


							//fprintf (stderr, " atom %d bond (%d %d) -- %d, (%d %d %d) -- (%d %d)\n", 
							//i, b_start[i], b_end[i], j, center, d_i, d_k, device->thb, device->pthb);

							if ((a[system->N + center] != -1)) {
								a[d_i] = a[d_k] = 1;
								continue;
							} else if ((b[system->N + center] != -1)) {
								b[d_i] = b[d_k] = 1;
								continue;
							} else if ((c[system->N + center] != -1)) {
								c[d_i] = c[d_k] = 1;
								continue;
							} else if ((d[system->N + center] != -1)) {
								d[d_i] = d[d_k] = 1;
								continue;
							}

							if ((a[center] == -1) && (a[d_i] == -1) && (a[d_k] == -1)) {
								a[center] = a[d_i] = a[d_k] = 1;
								a[system->N + center] = 1;
							} else if ((b[center] == -1) && (b[d_i] == -1) && (b[d_k] == -1)) {
								b[center] =  b[d_i] = b[d_k] = 1;
								b[system->N + center] = 1;
							} else if ((c[center] == -1) && (c[d_i] == -1) && (c[d_k] == -1)) {
								c[center] =  c[d_i] = c[d_k] = 1;
								c[system->N + center] = 1;
							} else if ((d[center] == -1) && (d[d_i] == -1) && (d[d_k] == -1)) {
								d[center] =  d[d_i] = d[d_k] = 1;
								d[system->N + center]= 1;
							}
							else {
								count ++;
								break;
								fprintf (stderr, "We have a problem with the four bins atom %d bond (%d %d) -- %d, (%d %d %d)\n", 
										i, b_start[i], b_end[i], j, center, d_i, d_k);
								fprintf (stderr, "A's contents %d %d %d (%d %d %d)\n", 
										a[system->N + center], a[system->N + d_i], a[system->N + d_k], a[center], a[d_i], a[d_k]);
								fprintf (stderr, "B's contents %d %d %d (%d %d %d)\n", 
										b[system->N + center], b[system->N + d_i], b[system->N + d_k], b[center], b[d_i], b[d_k]);
								fprintf (stderr, "C's contents %d %d %d (%d %d %d)\n", 
										c[system->N + center], c[system->N + d_i], c[system->N + d_k], c[center], c[d_i], c[d_k]);
								fprintf (stderr, "D's contents %d %d %d (%d %d %d)\n", 
										d[system->N + center], d[system->N + d_i], d[system->N + d_k], d[center], d[d_i], d[d_k]);

							}
						}
					}
				}
				fprintf (stderr, "Miscount is %d \n", count);
				exit (-1);

				count = 0;
				for (int i = 0; i < system->N; i++)
				{
					if (a[system->N + i] != -1) count ++;
					if (b[system->N + i] != -1) count ++;
					if (c[system->N + i] != -1) count ++;
					if (d[system->N + i] != -1) count ++;
				}

				fprintf (stderr, "binned so many atoms --> %d \n", count );
			}

			bool validate_grid (reax_system *system)
			{
				int total = system->g.ncell[0] * system->g.ncell[1] * system->g.ncell[2];
				int count = 0;

				int *dtop = (int *) malloc (INT_SIZE * total );
				copy_host_device (dtop, system->d_g.top, INT_SIZE * total, cudaMemcpyDeviceToHost, __LINE__);

				for (int i = 0; i < total; i++){
					if (system->g.top[i] != dtop[i]){
						fprintf (stderr, " top count does not match (%d %d) @ index %d \n", system->g.top[i], dtop[i], i );
						exit (-1);
					}
				}
				free (dtop);

				int *datoms = (int *) malloc (INT_SIZE * total * system->d_g.max_atoms);
				copy_host_device (datoms, system->d_g.atoms, INT_SIZE * total * system->d_g.max_atoms, cudaMemcpyDeviceToHost, __LINE__);
				for (int i = 0; i < total*system->d_g.max_atoms; i++){
					if (system->g.atoms[i] != datoms[i]){
						fprintf (stderr, " atoms count does not match (%d %d) @ index %d \n", system->g.atoms[i], datoms[i], i );
						exit (-1);
					}
				}
				free (datoms);

				ivec *dnbrs = (ivec *) malloc (IVEC_SIZE * total * system->d_g.max_nbrs);
				copy_host_device (dnbrs, system->d_g.nbrs, IVEC_SIZE * total * system->d_g.max_nbrs, cudaMemcpyDeviceToHost, __LINE__);
				for (int i = 0; i < total*system->d_g.max_nbrs; i++){
					if (!check_same (system->g.nbrs[i], dnbrs[i])){
						fprintf (stderr, " nbrs count does not match @ index %d \n", i );
						exit (-1);
					}
				}
				free (dnbrs);

				rvec *dnbrs_cp = (rvec *) malloc (RVEC_SIZE * total * system->d_g.max_nbrs);
				copy_host_device (dnbrs_cp, system->d_g.nbrs_cp, RVEC_SIZE * total * system->d_g.max_nbrs, cudaMemcpyDeviceToHost, __LINE__);
				for (int i = 0; i < total*system->d_g.max_nbrs; i++){
					if (check_zero (system->g.nbrs_cp[i], dnbrs_cp[i])){
						fprintf (stderr, " nbrs_cp count does not match @ index %d \n", i );
						exit (-1);
					}
				}
				free (dnbrs_cp);

				//fprintf (stderr, " Grid match between device and host \n");
				return true;
			}

			void print_atoms (reax_system *system)
			{
				int start, end, index;

				reax_atom *test = (reax_atom *) malloc (REAX_ATOM_SIZE * system->N);
				copy_host_device (test, system->d_atoms, REAX_ATOM_SIZE * system->N, cudaMemcpyDeviceToHost, RES_SYSTEM_ATOMS );

				//for (int i = 0; i < system->N; i++) 
				for (int i = 0; i < 10; i++) 
				{
					fprintf (stderr, "Atom:%d: Type:%d", i, test[i].type);
					fprintf (stderr, " x(%6.10f %6.10f %6.10f)", test[i].x[0], test[i].x[1], test[i].x[2] );
					fprintf (stderr, " v(%6.10f %6.10f %6.10f)", test[i].v[0], test[i].v[1], test[i].v[2] );
					fprintf (stderr, " f(%6.10f %6.10f %6.10f)", test[i].f[0], test[i].f[1], test[i].f[2] );
					fprintf (stderr, " q(%6.10f) \n", test[i].q );
				}
			}

			void print_sys_atoms (reax_system *system)
			{
				for (int i = 0; i < 10; i++) 
				{
					fprintf (stderr, "Atom:%d: Type:%d", i, system->atoms[i].type);
					fprintf (stderr, " x(%6.10f %6.10f %6.10f)",system->atoms[i].x[0], system->atoms[i].x[1], system->atoms[i].x[2] );
					fprintf (stderr, " v(%6.10f %6.10f %6.10f)",system->atoms[i].v[0], system->atoms[i].v[1], system->atoms[i].v[2] );
					fprintf (stderr, " f(%6.10f %6.10f %6.10f)", system->atoms[i].f[0], system->atoms[i].f[1], system->atoms[i].f[2] );
					fprintf (stderr, " q(%6.10f) \n", system->atoms[i].q );
				}
			}


			void print_grid (reax_system *system)
			{
				int i, j, k, x;
				grid *g = &system->g;

				for( i = 0; i < g->ncell[0]; i++ )
					for( j = 0; j < g->ncell[1]; j++ )
						for( k = 0; k < g->ncell[2]; k++ ){
							fprintf (stderr, "Cell [%d,%d,%d]--(", i, j, k);
							for (x = 0; x < g->top[index_grid_3d (i,j,k,g) ]; x++){
								fprintf (stderr, "%d,", g->atoms[ index_grid_atoms (i,j,k,x,g) ]);
							}
							fprintf (stderr, ")\n");
						}
			}


