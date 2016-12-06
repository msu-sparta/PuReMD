#include "cuda_environment.h"

#include "cuda_utils.h"


extern "C" void Setup_Cuda_Environment(int rank, int nprocs, int gpus_per_node)
{

    int deviceCount;
    cudaError_t flag;
    
    flag = cudaGetDeviceCount(&deviceCount);

    if ( flag != cudaSuccess )
    {
        fprintf( stderr, "ERROR: no CUDA capable device(s) found. Terminating...\n" );
        exit( CANNOT_INITIALIZE );
    }

    //Calculate the # of GPUs per processor
    //and assign the GPU for each process
    //TODO: handle condition where # CPU procs > # GPUs
    cudaSetDevice( (rank % (deviceCount)) );

#if defined(__CUDA_DEBUG__)
    fprintf( stderr, "p:%d is using GPU: %d \n", rank, (rank % deviceCount));
#endif

    //CHANGE ORIGINAL
    //cudaDeviceSetLimit( cudaLimitStackSize, 8192 );
    //cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    //cudaCheckError();
}


extern "C" void Cleanup_Cuda_Environment()
{
    cudaDeviceReset();
    cudaDeviceSynchronize();
}
