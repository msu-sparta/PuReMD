
#include "cuda_environment.h"

#include "cuda_utils.h"

extern "C" void Setup_Cuda_Environment (int rank, int nprocs, int gpus_per_node)
{

	int deviceCount = 0;
	cudaGetDeviceCount (&deviceCount);

	//Calculate the # of GPUs per processor
	//and assign the GPU for each process

	//hpcc changes
	//if (gpus_per_node == 2) {
		cudaSetDevice ( (rank % (deviceCount)) );
		//cudaSetDevice( 1 );
		fprintf( stderr, "p:%d is using GPU: %d \n", rank, (rank % deviceCount));
	//} else {
	//	cudaSetDevice ( 0 );
	//}

	///////////////////////////////////////////////
	///////////////////////////////////////////////
	///////////////////////////////////////////////
	// CHANGE ORIGINAL/////////////////////////////
	///////////////////////////////////////////////
	///////////////////////////////////////////////
	///////////////////////////////////////////////
	//cudaDeviceSetLimit ( cudaLimitStackSize, 8192 );
	//cudaDeviceSetCacheConfig ( cudaFuncCachePreferL1 );
	//cudaCheckError ();
	///////////////////////////////////////////////
	///////////////////////////////////////////////
	///////////////////////////////////////////////
	///////////////////////////////////////////////
	///////////////////////////////////////////////

}

extern "C" void Cleanup_Cuda_Environment ()
{
	cudaDeviceReset ();
	cudaDeviceSynchronize ();
}