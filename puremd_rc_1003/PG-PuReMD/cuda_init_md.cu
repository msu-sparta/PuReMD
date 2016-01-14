
#include "cuda_init_md.h"

#include "reax_types.h"
#include "cuda_utils.h"

void Cuda_Init_ScratchArea ()
{
	cuda_malloc ((void **)& scratch, SCRATCH_SIZE, 1, "Device:Scratch");

	host_scratch = (void *)malloc (HOST_SCRATCH_SIZE );
}
