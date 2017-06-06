
#include "cuda_init_md.h"

#include "reax_types.h"
#include "cuda_utils.h"

#include "tool_box.h"

void Cuda_Init_ScratchArea( )
{
    cuda_malloc( (void **)&scratch, DEVICE_SCRATCH_SIZE, TRUE, "device:scratch" );

    host_scratch = (void *) smalloc( HOST_SCRATCH_SIZE, "host:scratch" );
}
