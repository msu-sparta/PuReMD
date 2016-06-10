
#include "matvec.h"
#include "cuda_shuffle.h"

//one thread per row
CUDA_GLOBAL void k_dual_matvec(sparse_matrix H, rvec2 *vec, rvec2 *results, int rows)
{
	rvec2 results_row;
	int col;
	real val;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i >= rows) return;

	results_row [0] = results_row[1] = 0;

	for (int c = H.start[i]; c < H.end[i]; c++)
	{
		col = H.entries [c].j;
		val = H.entries[c].val;

		results_row[0] += val * vec [col][0];
		results_row[1] += val * vec [col][1];
	}

	results [i][0] = results_row[0];
	results [i][1] = results_row[1];
}

//32 thread warp per matrix row.
//invoked as follows
// <<< system->N, 32 >>>
//CUDA_GLOBAL void __launch_bounds__(384, 8) k_dual_matvec_csr(sparse_matrix H, rvec2 *vec, rvec2 *results, int num_rows)
CUDA_GLOBAL void  k_dual_matvec_csr(sparse_matrix H, rvec2 *vec, rvec2 *results, int num_rows)
{
#if defined(__SM_35__)

	rvec2 vals;
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int warp_id = thread_id / MATVEC_KER_THREADS_PER_ROW;
	int lane = thread_id & (MATVEC_KER_THREADS_PER_ROW - 1);

	int row_start;
	int row_end;

	// one warp per row
	int row = warp_id;

	vals[0] = 0;
	vals[1] = 0;
		
	if (row < num_rows) {
		row_start = H.start[row];
		row_end = H.end[row];

		for(int jj = row_start + lane; jj < row_end; jj += MATVEC_KER_THREADS_PER_ROW) {
			vals[0] += H.entries[jj].val * vec [ H.entries[jj].j ][0];
			vals[1] += H.entries[jj].val * vec [ H.entries[jj].j ][1];
		}
	}

	for (int s = MATVEC_KER_THREADS_PER_ROW >> 1; s >= 1; s /= 2){
		vals[0] += shfl( vals[0], s);
		vals[1] += shfl( vals[1], s);
	}

	if (lane == 0 && row < num_rows){
		results[row][0] = vals[0];
		results[row][1] = vals[1];
	}

#else


	extern __shared__ rvec2 vals [];
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
	int warp_id = thread_id / 32;
	int lane = thread_id & (32 - 1);

	int row_start;
	int row_end;

	// one warp per row
	//int row = warp_id;
	int row = warp_id;
	//if (row < num_rows)
	{
		vals[threadIdx.x][0] = 0;
		vals[threadIdx.x][1] = 0;
		
		if (row < num_rows) {
			row_start = H.start[row];
			row_end = H.end[row];

			// compute running sum per thread
			for(int jj = row_start + lane; jj < row_end; jj += 32) {
				vals[threadIdx.x][0] += H.entries[jj].val * vec [ H.entries[jj].j ][0];
				vals[threadIdx.x][1] += H.entries[jj].val * vec [ H.entries[jj].j ][1];
			}
		}

		__syncthreads ();

		// parallel reduction in shared memory
		//SIMD instructions with a WARP are synchronous -- so we do not need to synch here
		if (lane < 16) {
			vals[threadIdx.x][0] += vals[threadIdx.x + 16][0]; 
			vals[threadIdx.x][1] += vals[threadIdx.x + 16][1]; 
		}
		__syncthreads();
		if (lane < 8) {
			vals[threadIdx.x][0] += vals[threadIdx.x + 8][0]; 
			vals[threadIdx.x][1] += vals[threadIdx.x + 8][1]; 
		}
		__syncthreads ();
		if (lane < 4) {
			vals[threadIdx.x][0] += vals[threadIdx.x + 4][0]; 
			vals[threadIdx.x][1] += vals[threadIdx.x + 4][1]; 
		}
		__syncthreads ();
		if (lane < 2) {
			vals[threadIdx.x][0] += vals[threadIdx.x + 2][0]; 
			vals[threadIdx.x][1] += vals[threadIdx.x + 2][1]; 
		}
		__syncthreads ();
		if (lane < 1) {
			vals[threadIdx.x][0] += vals[threadIdx.x + 1][0]; 
			vals[threadIdx.x][1] += vals[threadIdx.x + 1][1]; 
		}
		__syncthreads ();

		// first thread writes the result
		if (lane == 0 && row < num_rows) {
			results[row][0] = vals[threadIdx.x][0];
			results[row][1] = vals[threadIdx.x][1];
		}
	}

#endif
}
