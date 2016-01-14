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




#include "center_mass.h"
#include "vector.h"

GLOBAL void center_of_mass_blocks (single_body_parameters *sbp, reax_atom *atoms,
							rvec *res_xcm, 
							rvec *res_vcm, 
							rvec *res_amcm, 
							size_t n)
{
  extern __shared__ rvec xcm[];
  extern __shared__ rvec vcm[];
  extern __shared__ rvec amcm[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int xcm_id = threadIdx.x;
  unsigned int vcm_id = blockDim.x;
  unsigned int amcm_id = 2 *(blockDim.x);

  unsigned int index = 0;
  rvec tmp;
  real m;

  rvec_MakeZero (xcm [threadIdx.x]);
  rvec_MakeZero (vcm [vcm_id + threadIdx.x]);
  rvec_MakeZero (amcm[amcm_id + threadIdx.x]);
  rvec_MakeZero (tmp);

  if (i < n){
    m = sbp [ atoms[i].type ].mass;
  	rvec_ScaledAdd (xcm [threadIdx.x], m, atoms [i].x);
	rvec_ScaledAdd (vcm [vcm_id + threadIdx.x], m, atoms [i].v);
	rvec_Cross (tmp, atoms[i].x, atoms [i].v);
	rvec_ScaledAdd (amcm[amcm_id + threadIdx.x], m, tmp);
  }
  __syncthreads ();

  for( int offset = blockDim.x / 2; offset > 0; offset >>= 1 ) { 

  	if ((threadIdx.x < offset)) {
    	index = threadIdx.x + offset;
		rvec_Add (xcm [threadIdx.x], xcm[index]);
		rvec_Add (vcm [vcm_id  + threadIdx.x], vcm[vcm_id + index]);
		rvec_Add (amcm[amcm_id + threadIdx.x], amcm[amcm_id + index]);
	} 
	__syncthreads ();
  }
  
  if ((threadIdx.x == 0)){
  	rvec_Copy (res_xcm[blockIdx.x], xcm[0]);
	rvec_Copy (res_vcm[blockIdx.x], vcm[vcm_id]);
	rvec_Copy (res_amcm[blockIdx.x], amcm[amcm_id]);
  }
}

GLOBAL void center_of_mass (rvec *xcm, 
							rvec *vcm, 
							rvec *amcm, 
							rvec *res_xcm,
							rvec *res_vcm,
							rvec *res_amcm,
							size_t n)
{
  extern __shared__ rvec sh_xcm[];
  extern __shared__ rvec sh_vcm[];
  extern __shared__ rvec sh_amcm[];

  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  unsigned int xcm_id = threadIdx.x;
  unsigned int vcm_id = blockDim.x;
  unsigned int amcm_id = 2 * (blockDim.x);

  unsigned int index = 0;
  rvec t_xcm, t_vcm, t_amcm;

  rvec_MakeZero (t_xcm);
  rvec_MakeZero (t_vcm);
  rvec_MakeZero (t_amcm);

  if (i < n){
  	rvec_Copy ( t_xcm, xcm[threadIdx.x]);
	rvec_Copy ( t_vcm, vcm[threadIdx.x]);
	rvec_Copy ( t_amcm, amcm[threadIdx.x]);
  }

  rvec_Copy (sh_xcm[xcm_id], t_xcm);
  rvec_Copy (sh_vcm[vcm_id + threadIdx.x], t_vcm);
  rvec_Copy (sh_amcm[amcm_id + threadIdx.x], t_amcm);

  __syncthreads ();

  for( int offset = blockDim.x / 2; offset > 0; offset >>= 1 ) { 

  	if (threadIdx.x < offset) {
    	index = threadIdx.x + offset;
		rvec_Add (sh_xcm [threadIdx.x], sh_xcm[index]);
		rvec_Add (sh_vcm [vcm_id + threadIdx.x], sh_vcm[vcm_id + index]);
		rvec_Add (sh_amcm [amcm_id + threadIdx.x], sh_amcm[amcm_id + index]);
	} 
	__syncthreads ();
  }

  if (threadIdx.x == 0){
  	rvec_Copy (res_xcm[blockIdx.x], sh_xcm[0]);
	rvec_Copy (res_vcm[blockIdx.x], sh_vcm[vcm_id]);
	rvec_Copy (res_amcm[blockIdx.x], sh_amcm[amcm_id]);
  }
}

GLOBAL void compute_center_mass (single_body_parameters *sbp, 
								reax_atom *atoms,
								real *results, 
								real xcm0, real xcm1, real xcm2,
								size_t n)
{
	extern __shared__ real xx[];
	extern __shared__ real xy[];
	extern __shared__ real xz[];
	extern __shared__ real yy[];
	extern __shared__ real yz[];
	extern __shared__ real zz[];

	unsigned int xx_i = threadIdx.x;
	unsigned int xy_i = blockDim.x;
	unsigned int xz_i = 2 * blockDim.x;
	unsigned int yy_i = 3 * blockDim.x;
	unsigned int yz_i = 4 * blockDim.x;
	unsigned int zz_i = 5 * blockDim.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int index = 0;

	rvec diff, xcm;
	real m = 0;
	rvec_MakeZero (diff);
	xcm[0] = xcm0;
	xcm[1] = xcm1;
	xcm[2] = xcm2;


	xx[xx_i] = xy [xy_i + threadIdx.x] = xz[xz_i + threadIdx.x] = 
	yy[yy_i + threadIdx.x] = yz[yz_i + threadIdx.x] = zz[zz_i + threadIdx.x] = 0;

	if (i < n){
      	m = sbp[ atoms[i].type ].mass;
		rvec_ScaledSum( diff, 1., atoms[i].x, -1., xcm );
		xx[ xx_i ] = diff[0] * diff[0] * m;
		xy[ xy_i + threadIdx.x ] = diff[0] * diff[1] * m;
		xz[ xz_i + threadIdx.x ] = diff[0] * diff[2] * m;
		yy[ yy_i + threadIdx.x ] = diff[1] * diff[1] * m;
		yz[ yz_i + threadIdx.x ] = diff[1] * diff[2] * m;
		zz[ zz_i + threadIdx.x ] = diff[2] * diff[2] * m;    
	}
	__syncthreads ();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1){
		if (threadIdx.x < offset){
			index = threadIdx.x + offset;
			xx[ threadIdx.x ] += xx[ index ];
			xy[ xy_i + threadIdx.x ] += xy [ xy_i + index ];
			xz[ xz_i + threadIdx.x ] += xz [ xz_i + index ];
			yy[ yy_i + threadIdx.x ] += yy [ yy_i + index ];
			yz[ yz_i + threadIdx.x ] += yz [ yz_i + index ];
			zz[ zz_i + threadIdx.x ] += zz [ zz_i + index ];
		}
		__syncthreads ();
	}

	if (threadIdx.x == 0) {
		results [ blockIdx.x*6 ] = xx [ 0 ];
		results [ blockIdx.x*6 + 1 ] = xy [ xy_i + 0 ];
		results [ blockIdx.x*6 + 2 ] = xz [ xz_i + 0 ];
		results [ blockIdx.x*6 + 3 ] = yy [ yy_i + 0 ];
		results [ blockIdx.x*6 + 4 ] = yz [ yz_i + 0 ];
		results [ blockIdx.x*6 + 5 ] = zz [ zz_i + 0 ];
	}
}

GLOBAL void compute_center_mass (real *input, real *output, size_t n)
{
	extern __shared__ real xx[];
	extern __shared__ real xy[];
	extern __shared__ real xz[];
	extern __shared__ real yy[];
	extern __shared__ real yz[];
	extern __shared__ real zz[];

	unsigned int xx_i = threadIdx.x;
	unsigned int xy_i = blockDim.x;
	unsigned int xz_i = 2 * blockDim.x;
	unsigned int yy_i = 3 * blockDim.x;
	unsigned int yz_i = 4 * blockDim.x;
	unsigned int zz_i = 5 * blockDim.x;

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int index = 0;

	xx[xx_i] = xy [xy_i + threadIdx.x] = xz[xz_i + threadIdx.x] = 
	yy[yy_i + threadIdx.x] = yz[yz_i + threadIdx.x] = zz[zz_i + threadIdx.x] = 0;

	if (i < n)
	{
		xx [ xx_i ] = input [ threadIdx.x*6 + 0 ];
		xy [ xy_i + threadIdx.x ] = input [ threadIdx.x*6 + 1 ];
		xz [ xz_i + threadIdx.x ] = input [ threadIdx.x*6 + 2 ];
		yy [ yy_i + threadIdx.x ] = input [ threadIdx.x*6 + 3 ];
		yz [ yz_i + threadIdx.x ] = input [ threadIdx.x*6 + 4 ];
		zz [ zz_i + threadIdx.x ] = input [ threadIdx.x*6 + 5 ];
	}
	__syncthreads ();

	for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if (threadIdx.x < offset )
		{
			index = threadIdx.x + offset;
			xx [ threadIdx.x ] += xx [ index ];
			xy [ xy_i + threadIdx.x ] += xy [ xy_i + index ];
			xz [ xz_i + threadIdx.x ] += xz [ xz_i + index ];
			yy [ yy_i + threadIdx.x ] += yy [ yy_i + index ];
			yz [ yz_i + threadIdx.x ] += yz [ yz_i + index ];
			zz [ zz_i + threadIdx.x ] += zz [ zz_i + index ];
		}
		__syncthreads ();
	}

	if (threadIdx.x == 0)
	{
		output[0] = xx[0];
		output[1] = xy[xy_i];
		output[2] = xz[xz_i];
		output[3] = xz[yy_i];
		output[4] = xz[yz_i];
		output[5] = xz[zz_i];
	}
}
