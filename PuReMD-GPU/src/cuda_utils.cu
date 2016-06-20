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




#include "cuda_utils.h"
#include "mytypes.h"

void cuda_malloc (void **ptr, int size, int memset, int err_code) {

	cudaError_t retVal = cudaSuccess;

	//fprintf (stderr, "&ptr --. %ld \n", &ptr);
	//fprintf (stderr, "ptr --> %ld \n", ptr );

	retVal = cudaMalloc (ptr, size);
	if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to allocate memory on device for the res: %d...  exiting with code: %d size: %d \n", 
				err_code, retVal, size);
		exit (err_code);
	}  

	//fprintf (stderr, "&ptr --. %ld \n", &ptr);
	//fprintf (stderr, "ptr --> %ld \n", ptr );

	if (memset) {
		retVal = cudaMemset (*ptr, 0, size);
		if (retVal != cudaSuccess) {
			fprintf (stderr, "Failed to memset memory on device... exiting with code %d\n", 
					err_code);
			exit (err_code);
		}
	}  
}

void cuda_free (void *ptr, int err_code) {

	cudaError_t retVal = cudaSuccess;
	if (!ptr) return;

	retVal = cudaFree (ptr);

	if (retVal != cudaSuccess) {
		fprintf (stderr, "Failed to release memory on device for res %d... exiting with code %d -- Address %ld\n", 
				err_code, retVal, ptr);
		return;
	}  
}
void cuda_memset (void *ptr, int data, size_t count, int err_code){
	cudaError_t retVal = cudaSuccess;

	retVal = cudaMemset (ptr, data, count);
	if (retVal != cudaSuccess) {
		fprintf (stderr, "ptr passed is %ld, value: %ld \n", ptr, &ptr);
		fprintf (stderr, " size to memset: %d \n", count);
		fprintf (stderr, " target data is : %d \n", data);
		fprintf (stderr, "Failed to memset memory on device... exiting with code %d, cuda code %d\n", 
				err_code, retVal);
		exit (err_code);
	}
}

void copy_host_device (void *host, void *dev, int size, enum cudaMemcpyKind dir, int resid)
{
	cudaError_t	retVal = cudaErrorNotReady;

	if (dir == cudaMemcpyHostToDevice)
		retVal = cudaMemcpy (dev, host, size, cudaMemcpyHostToDevice);
	else
		retVal = cudaMemcpy (host, dev, size, cudaMemcpyDeviceToHost);

	if (retVal != cudaSuccess) {
		fprintf (stderr, "could not copy resource %d from host to device: reason %d \n",
				resid, retVal);
		exit (resid);
	}
}

void copy_device (void *dest, void *src, int size, int resid)
{
	cudaError_t	retVal = cudaErrorNotReady;

	retVal = cudaMemcpy (dest, src, size, cudaMemcpyDeviceToDevice);
	if (retVal != cudaSuccess) {
		fprintf (stderr, "could not copy resource %d from host to device: reason %d \n",
				resid, retVal);
		exit (resid);
	}
}

void compute_blocks ( int *blocks, int *block_size, int count )
{
	*block_size = CUDA_BLOCK_SIZE;
	*blocks = (count / CUDA_BLOCK_SIZE ) + (count % CUDA_BLOCK_SIZE == 0 ? 0 : 1);
}

void compute_nearest_pow_2 (int blocks, int *result)
{
	int power = 1;
	while (power < blocks) power *= 2;

	*result = power;
}


void print_device_mem_usage ()
{
	size_t total, free;
	cudaMemGetInfo (&free, &total);
	if (cudaGetLastError () != cudaSuccess )
	{
		fprintf (stderr, "Error on the memory call \n");
		return;
	}

	fprintf (stderr, "Total %ld Mb %ld gig %ld , free %ld, Mb %ld , gig %ld \n", 
			total, total/(1024*1024), total/ (1024*1024*1024), 
			free, free/(1024*1024), free/ (1024*1024*1024) );
}
