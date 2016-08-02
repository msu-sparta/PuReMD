#include "cuda_utils.h"


extern "C" void cuda_malloc (void **ptr, int size, int mem_set, char *msg) {

    cudaError_t retVal = cudaSuccess;

    retVal = cudaMalloc (ptr, size);

    if (retVal != cudaSuccess)
    {
        fprintf (stderr, "Failed to allocate memory on device for the res: %s...  exiting with code: %d size: %d \n", 
                msg, retVal, size);
        exit (-1);
    }  

    if (mem_set)
    {
        retVal = cudaMemset (*ptr, 0, size);

        if (retVal != cudaSuccess) {
            fprintf (stderr, "Failed to memset memory on device for resource %s\n", 
                    msg);
            exit (-1);
        }
    }  
}


extern "C" void cuda_free (void *ptr, char *msg) {

    cudaError_t retVal = cudaSuccess;

    if ( !ptr )
    {
        return;
    }  

    retVal = cudaFree (ptr);

    if (retVal != cudaSuccess)
    {
        fprintf (stderr, "Failed to release memory on device for res %s... exiting with code %d -- Address %ld\n", 
                msg, retVal, (long int) ptr);
        return;
    }  
}


extern "C" void cuda_memset (void *ptr, int data, size_t count, char *msg){
    cudaError_t retVal = cudaSuccess;

    retVal = cudaMemset (ptr, data, count);

    if (retVal != cudaSuccess)
    {
        fprintf (stderr, "Failed to memset memory on device for %s, cuda code %d\n", 
                msg, retVal);
        exit (-1);
    }
}


extern "C" void copy_host_device (void *host, void *dev, int size, enum cudaMemcpyKind dir, char *msg)
{
    cudaError_t retVal = cudaErrorNotReady;

    if (dir == cudaMemcpyHostToDevice)
    {
        retVal = cudaMemcpy (dev, host, size, cudaMemcpyHostToDevice);
    }
    else
    {
        retVal = cudaMemcpy (host, dev, size, cudaMemcpyDeviceToHost);
    }

    if (retVal != cudaSuccess)
    {
        fprintf (stderr, "could not copy resource %s from host to device: reason %d \n",
                msg, retVal);
        exit (-1);
    }
}


extern "C" void copy_device (void *dest, void *src, int size, char *msg)
{
    cudaError_t retVal = cudaErrorNotReady;

    retVal = cudaMemcpy (dest, src, size, cudaMemcpyDeviceToDevice);
    if (retVal != cudaSuccess) {
        fprintf (stderr, "could not copy resource %s from device to device: reason %d \n",
                msg, retVal);
        exit (-1);
    }
}


extern "C" void compute_blocks ( int *blocks, int *block_size, int count )
{
    *block_size = CUDA_BLOCK_SIZE;
    *blocks = (int) CEIL((double) count / CUDA_BLOCK_SIZE);
}


extern "C" void compute_matvec_blocks ( int *blocks, int count )
{

    *blocks = (int) CEIL((double) count * MATVEC_KER_THREADS_PER_ROW / MATVEC_BLOCK_SIZE);
}


extern "C" void compute_nearest_pow_2 (int blocks, int *result)
{

  *result = (int) EXP2( CEIL( LOG2((double) blocks) ) );
}


void print_info ()
{
    size_t total, free;

    cudaMemGetInfo (&free, &total);

    if ( cudaGetLastError () != cudaSuccess )
    {
        fprintf (stderr, "Error on the memory call \n");
        return;
    }

    fprintf (stderr, "Total %ld Mb %ld gig %ld , free %ld, Mb %ld , gig %ld \n", 
            total, total/(1024*1024), total/ (1024*1024*1024), 
            free, free/(1024*1024), free/ (1024*1024*1024) );
}


extern "C" void print_device_mem_usage ()
{
    print_info ();
}
