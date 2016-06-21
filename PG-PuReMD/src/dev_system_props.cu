
#include "dev_system_props.h"

#include "reduction.h"
#include "cuda_utils.h"
#include "center_mass.h"
#include "cuda_copy.h"

#include "vector.h"
#include "cuda_shuffle.h"

CUDA_GLOBAL void k_compute_total_mass (single_body_parameters *sbp, reax_atom *my_atoms, 
        real *block_results, int n)
{
#if defined(__SM_35__)

    extern __shared__ real my_sbp[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real    sdata = 0;

    if (i < n)
        sdata = sbp [ my_atoms [i].type ].mass;
    __syncthreads ();

    for(int z = 16; z >=1; z/=2)
        sdata += shfl ( sdata, z);

    if (threadIdx.x % 32 == 0)
        my_sbp[threadIdx.x >> 5] = sdata;

    __syncthreads ();

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1) {
        if(threadIdx.x < offset)
            my_sbp[threadIdx.x] += my_sbp[threadIdx.x + offset];

        __syncthreads();
    }

    if(threadIdx.x == 0)
        block_results[blockIdx.x] = my_sbp[0];


#else

    extern __shared__ real sdata [];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real    x = 0;

    if (i < n)
        x = sbp [ my_atoms [i].type ].mass;

    sdata[ threadIdx.x ] = x;
    __syncthreads ();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1){
        if (threadIdx.x < offset)
            sdata [threadIdx.x] += sdata [threadIdx.x + offset];

        __syncthreads ();
    }

    if (threadIdx.x == 0)
        block_results[ blockIdx.x] = sdata [0];

#endif
}

extern "C" void dev_compute_total_mass (reax_system *system, real *local_val)
{
    real *block_mass = (real *) scratch;
    cuda_memset (block_mass, 0, sizeof (real) * (1 + BLOCKS_POW_2), "total_mass:tmp");

    k_compute_total_mass <<<BLOCKS, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>
        (system->reax_param.d_sbp, system->d_my_atoms, block_mass, system->n);
    cudaThreadSynchronize ();
    cudaCheckError ();

    k_reduction <<<1, BLOCKS_POW_2, sizeof (real) * BLOCKS_POW_2 >>>
        (block_mass, block_mass + BLOCKS_POW_2, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    copy_host_device (local_val, block_mass + BLOCKS_POW_2, sizeof (real), 
            cudaMemcpyDeviceToHost, "total_mass:tmp");
}

CUDA_GLOBAL void k_compute_kinetic_energy (single_body_parameters *sbp, reax_atom *my_atoms, 
        real *block_results, int n)
{

#if defined(__SM_35__)

    extern __shared__ real my_sbpdot[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real    sdata = 0;
    rvec p;

    if (i < n) {
        sdata = sbp [ my_atoms [i].type ].mass;
        rvec_Scale( p, sdata, my_atoms[ i ].v );
        sdata = 0.5 * rvec_Dot( p, my_atoms[ i ].v );
    }

    __syncthreads ();

    for(int z = 16; z >=1; z/=2)
        sdata += shfl ( sdata, z);

    if (threadIdx.x % 32 == 0)
        my_sbpdot[threadIdx.x >> 5] = sdata;

    __syncthreads ();

    for (int offset = blockDim.x >> 6; offset > 0; offset >>= 1){
        if (threadIdx.x < offset)
            my_sbpdot[threadIdx.x] += my_sbpdot[threadIdx.x + offset];

        __syncthreads ();
    }

    if (threadIdx.x == 0)
        block_results[ blockIdx.x] = my_sbpdot[0];

#else


    extern __shared__ real sdata [];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real    m = 0;
    rvec p;

    if (i < n) {
        m = sbp [ my_atoms [i].type ].mass;
        rvec_Scale( p, m, my_atoms[ i ].v );
        m = 0.5 * rvec_Dot( p, my_atoms[ i ].v );
    }

    sdata[ threadIdx.x ] = m;
    __syncthreads ();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1){
        if (threadIdx.x < offset)
            sdata [threadIdx.x] += sdata [threadIdx.x + offset];

        __syncthreads ();
    }

    if (threadIdx.x == 0)
        block_results[ blockIdx.x] = sdata [0];

#endif
}

extern "C" void dev_compute_kinetic_energy (reax_system *system, simulation_data *data, real *local_val)
{
    real *block_energy = (real *) scratch;
    cuda_memset (block_energy, 0, sizeof (real) * (BLOCKS_POW_2 + 1), "kinetic_energy:tmp");

    k_compute_kinetic_energy <<<BLOCKS, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>
        (system->reax_param.d_sbp, system->d_my_atoms, block_energy, system->n);
    cudaThreadSynchronize ();
    cudaCheckError ();

    k_reduction <<<1, BLOCKS_POW_2, sizeof (real) * BLOCKS_POW_2 >>>
        (block_energy, block_energy + BLOCKS_POW_2, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    copy_host_device (local_val, block_energy + BLOCKS_POW_2,
            //copy_host_device (local_val, &((simulation_data *)data->d_simulation_data)->my_en.e_kin, 
            sizeof (real), cudaMemcpyDeviceToHost, "kinetic_energy:tmp");
            //copy_device (block_energy + BLOCKS_POW_2, &((simulation_data *)data->d_simulation_data)->my_en.e_kin,
            //        sizeof (real), "kinetic_energy");
            }

            extern "C" void dev_compute_momentum (reax_system *system, rvec xcm, 
                rvec vcm, rvec amcm)
            {
            rvec *l_xcm, *l_vcm, *l_amcm;
            rvec *r_scratch = (rvec *)scratch;

#if defined( __SM_35__)
            // xcm
            cuda_memset( scratch, 0,  sizeof (rvec) * (BLOCKS_POW_2 + 1), "momentum:tmp");
            l_xcm = r_scratch;

            center_of_mass_blocks_xcm <<<BLOCKS_POW_2,BLOCK_SIZE,(sizeof (rvec) * BLOCK_SIZE) >>>
            (system->reax_param.d_sbp, system->d_my_atoms, l_xcm, system->n );
            cudaThreadSynchronize ();
            cudaCheckError ();

            k_reduction_rvec <<<1, BLOCKS_POW_2, (sizeof (rvec) * BLOCKS_POW_2) >>>
                (l_xcm, l_xcm + BLOCKS_POW_2, BLOCKS_POW_2);
            cudaThreadSynchronize ();
            cudaCheckError ();
            copy_host_device (xcm, l_xcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost, "momentum:xcm");

            // vcm
            cuda_memset( scratch, 0,  sizeof (rvec) * (BLOCKS_POW_2 + 1), "momentum:tmp");
            l_vcm = r_scratch;

            center_of_mass_blocks_vcm <<<BLOCKS_POW_2,BLOCK_SIZE,(sizeof (rvec) * BLOCK_SIZE) >>>
                (system->reax_param.d_sbp, system->d_my_atoms, l_vcm, system->n );
            cudaThreadSynchronize ();
            cudaCheckError ();

            k_reduction_rvec <<<1, BLOCKS_POW_2, (sizeof (rvec) * BLOCKS_POW_2) >>>
                (l_vcm, l_vcm + BLOCKS_POW_2, BLOCKS_POW_2);
            cudaThreadSynchronize ();
            cudaCheckError ();
            copy_host_device (vcm, l_vcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost, "momentum:vcm");

            // amcm
            cuda_memset( scratch, 0,  sizeof (rvec) * (BLOCKS_POW_2 + 1), "momentum:tmp");
            l_amcm = r_scratch;

            center_of_mass_blocks_amcm <<<BLOCKS_POW_2,BLOCK_SIZE,(sizeof (rvec) * BLOCK_SIZE) >>>
                (system->reax_param.d_sbp, system->d_my_atoms, l_amcm, system->n );
            cudaThreadSynchronize ();
            cudaCheckError ();

            k_reduction_rvec <<<1, BLOCKS_POW_2, (sizeof (rvec) * BLOCKS_POW_2) >>>
                (l_amcm, l_amcm + BLOCKS_POW_2, BLOCKS_POW_2);
            cudaThreadSynchronize ();
            cudaCheckError ();
            copy_host_device (amcm, l_amcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost, "momemtum:amcm");

#else
            cuda_memset ( scratch, 0, 3 * sizeof (rvec) * (BLOCKS_POW_2 + 1), "momentum:tmp");

            l_xcm = r_scratch;
            l_vcm = r_scratch + (BLOCKS_POW_2 + 1); 
            l_amcm = r_scratch + 2 * (BLOCKS_POW_2 + 1); 

            center_of_mass_blocks <<<BLOCKS_POW_2, BLOCK_SIZE, 3 * (sizeof (rvec) * BLOCK_SIZE) >>> 
                (system->reax_param.d_sbp, system->d_my_atoms, l_xcm, l_vcm, l_amcm, system->n);
            cudaThreadSynchronize (); 
            cudaCheckError (); 

            center_of_mass <<<1, BLOCKS_POW_2, 3 * (sizeof (rvec) * BLOCKS_POW_2) >>> 
                (l_xcm, l_vcm, l_amcm,
                 l_xcm + BLOCKS_POW_2, 
                 l_vcm + BLOCKS_POW_2, 
                 l_amcm + BLOCKS_POW_2, 
                 BLOCKS_POW_2);
            cudaThreadSynchronize (); 
            cudaCheckError ();

            copy_host_device (xcm, l_xcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost, "momemtum:xcm" );
            copy_host_device (vcm, l_vcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost, "momentum:vcm" );
            copy_host_device (amcm, l_amcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost,"momentum:amcm" );
#endif
            }

extern "C" void dev_compute_inertial_tensor (reax_system *system, real *local_results, rvec my_xcm)
{
#if defined(__SM_35__)
    real *partial_results = (real *) scratch;
    cuda_memset (partial_results, 0, sizeof (real) * 6 * (BLOCKS_POW_2 + 1), "tensor:tmp");

    compute_center_mass_xx_xy <<<BLOCKS_POW_2, BLOCK_SIZE, 2 * (sizeof (real) * BLOCK_SIZE) >>>
        (system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n);
    cudaThreadSynchronize ();
    cudaCheckError ();

    compute_center_mass_xz_yy <<<BLOCKS_POW_2, BLOCK_SIZE, 2 * (sizeof (real) * BLOCK_SIZE) >>>
        (system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n);
    cudaThreadSynchronize ();
    cudaCheckError ();

    compute_center_mass_yz_zz <<<BLOCKS_POW_2, BLOCK_SIZE, 2 * (sizeof (real) * BLOCK_SIZE) >>>
        (system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n);
    cudaThreadSynchronize ();
    cudaCheckError ();

    compute_center_mass <<<1, BLOCKS_POW_2, 6 * (sizeof (real) * BLOCKS_POW_2) >>>
        (partial_results, partial_results + (BLOCKS_POW_2 * 6), BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    copy_host_device (local_results, partial_results + 6 * BLOCKS_POW_2, sizeof (real) * 6, cudaMemcpyDeviceToHost, "tensor:local_results");

#else

    real *partial_results = (real *) scratch;
    //real *local_results;

    cuda_memset (partial_results, 0, sizeof (real) * 6 * (BLOCKS_POW_2 + 1), "tensor:tmp");
    //local_results = (real *) malloc (sizeof (real) * 6 *(BLOCKS_POW_2+ 1));

    compute_center_mass <<<BLOCKS_POW_2, BLOCK_SIZE, 6 * (sizeof (real) * BLOCK_SIZE) >>>
        (system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n);
    cudaThreadSynchronize ();
    cudaCheckError ();

    compute_center_mass <<<1, BLOCKS_POW_2, 6 * (sizeof (real) * BLOCKS_POW_2) >>>
        (partial_results, partial_results + (BLOCKS_POW_2 * 6), BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    copy_host_device (local_results, partial_results + 6 * BLOCKS_POW_2, 
            sizeof (real) * 6, cudaMemcpyDeviceToHost, "tensor:local_results");
#endif
}

extern "C" void dev_sync_simulation_data (simulation_data *data)
{
    Output_Sync_Simulation_Data (data, (simulation_data *)data->d_simulation_data );
}
/*
   CUDA_GLOBAL void ker_kinetic_energy (reax_atom *my_atoms, 
   single_body_parameters *sbp, 
   int n, real *block_results)
   {
   extern __shared__ real sken[];
   rvec p;
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
   real x = 0;

   if(i < n)
   {
   m = sbp[my_atoms[i].type].mass;
   rvec_Scale( p, m, my_atoms[i].v );
   x = 0.5 * rvec_Dot( p, my_atoms[i].v );
   }
   sken[threadIdx.x] = x;
   __syncthreads();

   for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
   {
   if(threadIdx.x < offset)
   {   
   sken[threadIdx.x] += sken[threadIdx.x + offset];
   }   

   __syncthreads();
   }

   if(threadIdx.x == 0)
   {
   per_block_results[blockIdx.x] = sken[0];
   }
   }

   void dev_compute_kinetic_energy (reax_system *system, simulation_data *data, real *p_ekin)
   {
   real *spad = (real *) scratch;
   cuda_memset (spad, 0, sizeof (real) * 2 * system->n, "kinetic_energy");

   ker_kinetic_energy <<<BLOCKS, BLOCK_SIZE, sizeof (real) * BLOCK_SIZE >>>
   (spad, spad + system->n,  system->n);
   cudaThreadSynchronize (); 
   cudaCheckError (); 

   k_reduction <<<1, BLOCKS_POW_2, sizeof (real) * BLOCKS_POW_2 >>> 
   (spad + system->n, &((simulation_data *)data->d_simulation_data)->my_en.e_kin, BLOCKS);
   cudaThreadSynchronize (); 
   cudaCheckError (); 

   copy_host_device (p_ekin, &((simulation_data *)data->d_simulation_data)->my_en.e_kin, 
   sizeof (real), cudaMemcpyDeviceToHost, "kinetic_energy");
   }
 */
