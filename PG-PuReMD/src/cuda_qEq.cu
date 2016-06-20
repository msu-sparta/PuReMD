/*----------------------------------------------------------------------
  PuReMD - Purdue ReaxFF Molecular Dynamics Program

  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
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

#include "cuda_qEq.h"

#include "reax_types.h"
#include "reduction.h"
#include "cuda_utils.h"

#include "validation.h"

CUDA_GLOBAL void ker_init_matvec( 	reax_atom *my_atoms, 
		single_body_parameters *sbp, 
		storage p_workspace, int n  )
{
	storage *workspace = &( p_workspace );
	reax_atom *atom;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	//for( i = 0; i < system->n; ++i ) {
	atom = &( my_atoms[i] );

	/* init pre-conditioner for H and init solution vectors */
	workspace->Hdia_inv[i] = 1. / sbp[ atom->type ].eta;
	workspace->b_s[i] = -sbp[ atom->type ].chi;
	workspace->b_t[i] = -1.0;
	workspace->b[i][0] = -sbp[ atom->type ].chi;
	workspace->b[i][1] = -1.0;

	workspace->x[i][1] = atom->t[2] + 3 * ( atom->t[0] - atom->t[1] );

	/* cubic extrapolation for s and t */
	workspace->x[i][0] = 4*(atom->s[0]+atom->s[2])-(6*atom->s[1]+atom->s[3]);
	//}
}

void Cuda_Init_MatVec ( reax_system *system, storage *workspace )
{
	int blocks;

	blocks = system->n / DEF_BLOCK_SIZE + 
		(( system->n % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

	ker_init_matvec <<< blocks, DEF_BLOCK_SIZE >>>
		( system->d_my_atoms, system->reax_param.d_sbp, 
		  *dev_workspace, system->n );
	cudaThreadSynchronize ();
	cudaCheckError ();
}

void cuda_charges_x (reax_system *system, rvec2 my_sum)
{
	int blocks;
	rvec2 *output = (rvec2 *) scratch;
	cuda_memset (output, 0, sizeof (rvec2) * 2 * system->n, "cuda_charges_x:q");

	blocks = system->n / DEF_BLOCK_SIZE + 
		(( system->n % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);

	k_reduction_rvec2 <<< blocks, DEF_BLOCK_SIZE, sizeof (rvec2) * DEF_BLOCK_SIZE >>>
		( dev_workspace->x, output, system->n );
	cudaThreadSynchronize ();
	cudaCheckError ();

	k_reduction_rvec2 <<< 1, BLOCKS_POW_2, sizeof (rvec2) * BLOCKS_POW_2 >>>
		( output, output + system->n, blocks );
	cudaThreadSynchronize ();
	cudaCheckError ();

	copy_host_device (my_sum, output + system->n, sizeof (rvec2), cudaMemcpyDeviceToHost, "charges:x");
}

CUDA_GLOBAL void ker_calculate_st (reax_atom *my_atoms, storage p_workspace, 
		real u, real *q, int n)
{
	storage *workspace = &( p_workspace );
	reax_atom *atom;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;

	//for( i = 0; i < system->n; ++i ) {
	atom = &( my_atoms[i] );

	//atom->q = workspace->s[i] - u * workspace->t[i];
	q[i] = atom->q = workspace->x[i][0] - u * workspace->x[i][1];

	atom->s[3] = atom->s[2];
	atom->s[2] = atom->s[1];
	atom->s[1] = atom->s[0];
	//atom->s[0] = workspace->s[i];
	atom->s[0] = workspace->x[i][0];

	atom->t[3] = atom->t[2];
	atom->t[2] = atom->t[1];
	atom->t[1] = atom->t[0];
	//atom->t[0] = workspace->t[i];
	atom->t[0] = workspace->x[i][1];
	//}
}

//TODO if we use the function argument (output), we are getting 
//TODO Address not mapped/Invalid permissions error with segmentation fault
//TODO so using the local argument, which is a global variable anyways. 
//TODO NEED TO INVESTIGATE MORE ON THIS ISSUE
//TODO
//TODO
//TODO

extern "C" void cuda_charges_st (reax_system *system, storage *workspace, real *output, real u)
{
	int blocks;
	real *tmp = (real *) scratch;
	real *tmp_output = (real *) host_scratch;

	cuda_memset (tmp, 0, sizeof (real) * system->n, "charges:q");
	memset (tmp_output, 0, sizeof (real) * system->n);

	blocks = system->n / DEF_BLOCK_SIZE + 
		(( system->n % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	ker_calculate_st <<< blocks, DEF_BLOCK_SIZE >>>
		( system->d_my_atoms, *dev_workspace, u, tmp, system->n);
	cudaThreadSynchronize ();
	cudaCheckError ();

	copy_host_device (output, tmp, sizeof (real) * system->n, 
			cudaMemcpyDeviceToHost, "charges:q");
}
//TODO
//TODO
//TODO
//TODO
//TODO
//TODO
//TODO

CUDA_GLOBAL void ker_update_q (reax_atom *my_atoms, real *q, int n, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (N-n)) return;

	//for( i = system->n; i < system->N; ++i )
	my_atoms[i + n].q = q[i + n];
}

void cuda_charges_updateq (reax_system *system, real *q) 
{
	int blocks;
	real *dev_q = (real *) scratch;
	copy_host_device (q, dev_q, system->N * sizeof (real), 
			cudaMemcpyHostToDevice, "charges:q");
	blocks = (system->N - system->n) / DEF_BLOCK_SIZE + 
		(( (system->N - system->n) % DEF_BLOCK_SIZE == 0 ) ? 0 : 1);
	ker_update_q <<< blocks, DEF_BLOCK_SIZE >>>
		( system->d_my_atoms, dev_q, system->n, system->N);
	cudaThreadSynchronize ();
	cudaCheckError ();
}
