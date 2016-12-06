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

#include "cuda_system_props.h"

#include "box.h"
#include "vector.h"

#include "cuda_center_mass.h"
#include "cuda_copy.h"
#include "cuda_utils.h"
#include "cuda_reduction.h"


GLOBAL void k_Compute_Total_Mass(single_body_parameters *, reax_atom *, real *, size_t );
GLOBAL void k_Compute_Kinetic_Energy(single_body_parameters *, reax_atom *, unsigned int , real *);
GLOBAL void k_Kinetic_Energy_Reduction(simulation_data *, real *, int);


void prep_dev_system (reax_system *system) 
{
    //copy the system atoms to the device
    Sync_Host_Device_Sys( system, cudaMemcpyHostToDevice );
}


void Cuda_Compute_Total_Mass( reax_system *system, simulation_data *data )
{
    real    *partial_sums = (real *) scratch;
    //data->M = 0;

    //cuda_malloc ((void **)&partial_sums, sizeof (real) * (blocks + 1), 1, 0);
    cuda_memset (partial_sums, 0, REAL_SIZE * (BLOCKS_POW_2 + 1), RES_SCRATCH );

    k_Compute_Total_Mass <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE >>> 
        (system->reaxprm.d_sbp, system->d_atoms, partial_sums, system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    Cuda_reduction <<<1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2>>> 
        (partial_sums, partial_sums + BLOCKS_POW_2, BLOCKS_POW_2);
    //(partial_sums, &((simulation_data *)data->d_simulation_data)->M, BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //#ifdef __BUILD_DEBUG__
    //    validate_data ( system, data );
    //#endif

    //copy_host_device (&data->M, &((simulation_data *)data->d_simulation_data)->M, 
    //#ifdef __BUILD_DEBUG__
    //    t_data_M = data->M;
    //#endif
    copy_host_device (&data->M, partial_sums + BLOCKS_POW_2, 
            REAL_SIZE, cudaMemcpyDeviceToHost, __LINE__);

    //#ifdef __BUILD_DEBUG__
    //    if (check_zero (t_data, data->M))
    //    {
    //        fprintf (stderr, "SimulationData:M does not match on host and device (%f %f) \n", t_data, data->M );
    //        exit (0);
    //    }
    //#endif
    data->inv_M = 1. / data->M;    
}


void Cuda_Compute_Center_of_Mass( reax_system *system, simulation_data *data, 
        FILE *fout )
{
    int i;
    real m, xx, xy, xz, yy, yz, zz, det;
    rvec tvec, diff;
    rtensor mat, inv;

    int blocks;
    int block_size;
    rvec *l_xcm, *l_vcm, *l_amcm;
    real t_start, t_end;

    rvec t_xcm, t_vcm, t_amcm;

    rvec *r_scratch = (rvec *)scratch;

    //rvec_MakeZero( data->xcm );  // position of CoM
    //rvec_MakeZero( data->vcm );  // velocity of CoM
    //rvec_MakeZero( data->amcm ); // angular momentum of CoM
    //rvec_MakeZero( data->avcm ); // angular velocity of CoM

    //cuda_malloc ((void **)&l_xcm, RVEC_SIZE * (blocks + 1), 1, 0);
    //cuda_malloc ((void **)&l_vcm, RVEC_SIZE * (blocks + 1), 1, 0);
    //cuda_malloc ((void **)&l_amcm, RVEC_SIZE * (blocks + 1), 1, 0);

    cuda_memset ( scratch, 0, 3 * RVEC_SIZE * (BLOCKS_POW_2 + 1), RES_SCRATCH );
    l_xcm = r_scratch;
    l_vcm = r_scratch + (BLOCKS_POW_2 + 1);
    l_amcm = r_scratch + 2 * (BLOCKS_POW_2 + 1);

    k_center_of_mass_blocks<<<BLOCKS_POW_2, BLOCK_SIZE, 3 * (RVEC_SIZE * BLOCK_SIZE) >>> 
        (system->reaxprm.d_sbp, system->d_atoms, l_xcm, l_vcm, l_amcm, system->N);
    cudaThreadSynchronize ();
    cudaCheckError ();

    k_center_of_mass<<<1, BLOCKS_POW_2, 3 * (RVEC_SIZE * BLOCKS_POW_2) >>> 
        (l_xcm, l_vcm, l_amcm, 
         l_xcm + BLOCKS_POW_2, 
         l_vcm + BLOCKS_POW_2, 
         l_amcm + BLOCKS_POW_2, 
         BLOCKS_POW_2);
    cudaThreadSynchronize ();
    cudaCheckError ();

    //#ifdef __BUILD_DEBUG
    //    validate_data ( system, data );
    //#endif

    //#ifdef __BUILD_DEBUG__
    //    rvec_MakeZero (t_xcm);
    //    rvec_MakeZero (t_vcm);
    //    rvec_MakeZero (t_amcm);
    //
    //    rvec_Copy (t_xcm, data->xcm);
    //    rvec_Copy (t_vcm, data->vcm);
    //    rvec_Copy (t_amcm, data->amcm);
    //#endif

    copy_host_device (data->xcm, l_xcm + BLOCKS_POW_2, RVEC_SIZE, cudaMemcpyDeviceToHost, __LINE__);
    copy_host_device (data->vcm, l_vcm + BLOCKS_POW_2, RVEC_SIZE, cudaMemcpyDeviceToHost, __LINE__);
    copy_host_device (data->amcm, l_amcm + BLOCKS_POW_2, RVEC_SIZE, cudaMemcpyDeviceToHost, __LINE__);

    rvec_Scale( data->xcm, data->inv_M, data->xcm );
    rvec_Scale( data->vcm, data->inv_M, data->vcm );

    rvec_Cross( tvec, data->xcm, data->vcm );
    rvec_ScaledAdd( data->amcm, -data->M, tvec );

    //#ifdef __BUILD_DEBUG__
    //    if (check_zero (t_xcm, data->xcm) || 
    //        check_zero (t_vcm, data->vcm) ||
    //        check_zero (t_amcm, data->amcm)){
    //            fprintf (stderr, "SimulationData (xcm, vcm, amcm) does not match between device and host \n");
    //            exit (0);
    //        }
    //#endif

    data->etran_cm = 0.5 * data->M * rvec_Norm_Sqr( data->vcm );

    /* Calculate and then invert the inertial tensor */
    xx = xy = xz = yy = yz = zz = 0;

#ifdef __BUILD_DEBUG__

    for( i = 0; i < system->N; ++i ) {
        m = system->reaxprm.sbp[ system->atoms[i].type ].mass;

        rvec_ScaledSum( diff, 1., system->atoms[i].x, -1., data->xcm );
        xx += diff[0] * diff[0] * m;
        xy += diff[0] * diff[1] * m;
        xz += diff[0] * diff[2] * m;
        yy += diff[1] * diff[1] * m;
        yz += diff[1] * diff[2] * m;
        zz += diff[2] * diff[2] * m;      
    }

#endif

    real *partial_results = (real *) scratch;
    real *local_results;

    //cuda_malloc ((void **)&partial_results, 6 * sizeof (real) * (blocks + 1), 1, 0);
    cuda_memset (partial_results, 0, REAL_SIZE * 6 * (BLOCKS_POW_2 + 1), RES_SCRATCH );
    local_results = (real *) malloc (REAL_SIZE * 6 *(BLOCKS_POW_2+ 1));

    k_compute_center_mass_sbp<<<BLOCKS_POW_2, BLOCK_SIZE, 6 * (REAL_SIZE * BLOCK_SIZE) >>> 
        (system->reaxprm.d_sbp, system->d_atoms, partial_results, 
         data->xcm[0], data->xcm[1], data->xcm[2], system->N);
    cudaThreadSynchronize( );
    cudaCheckError( );

    k_compute_center_mass<<<1, BLOCKS_POW_2, 6 * (REAL_SIZE * BLOCKS_POW_2) >>> 
        (partial_results, partial_results + (BLOCKS_POW_2 * 6), BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( local_results, partial_results + 6 * BLOCKS_POW_2, REAL_SIZE * 6, cudaMemcpyDeviceToHost, __LINE__ );

#ifdef __BUILD_DEBUG__
    if (check_zero (local_results[0],xx) ||
            check_zero (local_results[1],xy) ||
            check_zero (local_results[2],xz) ||
            check_zero (local_results[3],yy) ||
            check_zero (local_results[4],yz) ||
            check_zero (local_results[5],zz) )
    {
        fprintf (stderr, " xx (%4.15f %4.15f) \n", xx, local_results[0]);
        fprintf (stderr, " xy (%4.15f %4.15f) \n", xy, local_results[1]);
        fprintf (stderr, " xz (%4.15f %4.15f) \n", xz, local_results[2]);
        fprintf (stderr, " yy (%4.15f %4.15f) \n", yy, local_results[3]);
        fprintf (stderr, " yz (%4.15f %4.15f) \n", yz, local_results[4]);
        fprintf (stderr, " zz (%4.15f %4.15f) \n", zz, local_results[5]);
        fprintf (stderr, " Failed to compute the center of mass \n");
        exit (1);
    }
#endif

    xx = local_results[0];
    xy = local_results[1];
    xz = local_results[2];
    yy = local_results[3];
    yz = local_results[4];
    zz = local_results[5];

    mat[0][0] = yy + zz;     
    mat[0][1] = mat[1][0] = -xy;
    mat[0][2] = mat[2][0] = -xz;
    mat[1][1] = xx + zz;
    mat[2][1] = mat[1][2] = -yz;
    mat[2][2] = xx + yy;

    /* invert the inertial tensor */
    det = ( mat[0][0] * mat[1][1] * mat[2][2] + 
            mat[0][1] * mat[1][2] * mat[2][0] + 
            mat[0][2] * mat[1][0] * mat[2][1] ) -
        ( mat[0][0] * mat[1][2] * mat[2][1] + 
          mat[0][1] * mat[1][0] * mat[2][2] + 
          mat[0][2] * mat[1][1] * mat[2][0] );

    inv[0][0] = mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1];
    inv[0][1] = mat[0][2] * mat[2][1] - mat[0][1] * mat[2][2];
    inv[0][2] = mat[0][1] * mat[1][2] - mat[0][2] * mat[1][1];
    inv[1][0] = mat[1][2] * mat[2][0] - mat[1][0] * mat[2][2];
    inv[1][1] = mat[0][0] * mat[2][2] - mat[0][2] * mat[2][0];
    inv[1][2] = mat[0][2] * mat[1][0] - mat[0][0] * mat[1][2];
    inv[2][0] = mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1];
    inv[2][1] = mat[2][0] * mat[0][1] - mat[0][0] * mat[2][1];
    inv[2][2] = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];

    if( fabs(det) > ALMOST_ZERO )
    {
        rtensor_Scale( inv, 1./det, inv );
    }
    else 
    {
        rtensor_MakeZero( inv );
    }

    /* Compute the angular velocity about the centre of mass */
    rtensor_MatVec( data->avcm, inv, data->amcm );  
    data->erot_cm = 0.5 * E_CONV * rvec_Dot( data->avcm, data->amcm );

    free( local_results );

#if defined(DEBUG)
    fprintf( stderr, "xcm:  %24.15e %24.15e %24.15e\n",  
            data->xcm[0], data->xcm[1], data->xcm[2] );
    fprintf( stderr, "vcm:  %24.15e %24.15e %24.15e\n", 
            data->vcm[0], data->vcm[1], data->vcm[2] );
    fprintf( stderr, "amcm: %24.15e %24.15e %24.15e\n", 
            data->amcm[0], data->amcm[1], data->amcm[2] );
    /* fprintf( fout, "mat:  %f %f %f\n     %f %f %f\n     %f %f %f\n",
       mat[0][0], mat[0][1], mat[0][2], 
       mat[1][0], mat[1][1], mat[1][2], 
       mat[2][0], mat[2][1], mat[2][2] );
       fprintf( fout, "inv:  %g %g %g\n     %g %g %g\n     %g %g %g\n",
       inv[0][0], inv[0][1], inv[0][2], 
       inv[1][0], inv[1][1], inv[1][2], 
       inv[2][0], inv[2][1], inv[2][2] );
       fflush( fout ); */
    fprintf( stderr, "avcm:  %24.15e %24.15e %24.15e\n", 
            data->avcm[0], data->avcm[1], data->avcm[2] );
#endif
}


void Cuda_Compute_Kinetic_Energy( reax_system *system, simulation_data *data )
{
    real *results = (real *) scratch;
    cuda_memset (results, 0, REAL_SIZE * BLOCKS_POW_2, RES_SCRATCH);
    k_Compute_Kinetic_Energy <<<BLOCKS_POW_2, BLOCK_SIZE, REAL_SIZE * BLOCK_SIZE>>>
        (system->reaxprm.d_sbp, system->d_atoms, system->N, (real *) results);
    cudaThreadSynchronize (); 
    cudaCheckError ();

    k_Kinetic_Energy_Reduction <<< 1, BLOCKS_POW_2, REAL_SIZE * BLOCKS_POW_2 >>>
        ((simulation_data *)data->d_simulation_data, results, BLOCKS_POW_2);
    cudaThreadSynchronize (); 
    cudaCheckError ();
}


GLOBAL void k_Compute_Total_Mass (single_body_parameters *sbp, reax_atom *atoms, real *per_block_results, size_t n) 
{
    extern __shared__ real sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0; 

    if(i < n) 
        x = sbp [ atoms[ i ].type ].mass;

    sdata[threadIdx.x] = x; 
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) 
    {  
        if(threadIdx.x < offset)
        {  
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }  
        __syncthreads();
    }  

    if(threadIdx.x == 0) 
    {  
        per_block_results[blockIdx.x] = sdata[0];
    }
}


GLOBAL void k_Compute_Kinetic_Energy( single_body_parameters* sbp, reax_atom* atoms, 
        unsigned int N, real *output)
{
    extern __shared__ real sh_ekin[];
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    rvec p;
    real m, tmp;

    tmp = 0;
    m = 0;
    if (index < N) {
        m = sbp[atoms[index].type].mass;
        rvec_Scale( p, m, atoms[index].v );
        tmp = 0.5 * rvec_Dot( p, atoms[index].v );
    }
    sh_ekin[threadIdx.x] = tmp;
    __syncthreads ();

    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset ) {
            index = threadIdx.x + offset;
            sh_ekin[threadIdx.x] += sh_ekin[ index ];
        }
        __syncthreads ();
    }

    if (threadIdx.x == 0) {
        output [ blockIdx.x ] = sh_ekin [ 0 ];
    }
}


GLOBAL void k_Kinetic_Energy_Reduction (simulation_data *data, real *input, int n)
{
    extern __shared__ real sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if(i < n)
    {
        x = input[i];
    }
    sdata[threadIdx.x] = x;
    __syncthreads();

    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {   
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }   

        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        //per_block_results[blockIdx.x] = sdata[0];
        data->E_Kin = sdata[0];
        data->therm.T = (2. * data->E_Kin) / (data->N_f * K_B);

        if ( fabs(data->therm.T) < ALMOST_ZERO ) // avoid T being an absolute zero! 
            data->therm.T = ALMOST_ZERO;
    }
}


/*
   GLOBAL void Compute_Kinetic_Energy( single_body_parameters* sbp, reax_atom* atoms, 
   unsigned int N, simulation_data *data, 
   real *output)
   {
   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
   rvec p;
   real m, tmp;

   extern __shared__ real sh_ekin[];

   tmp = 0;
   m = 0;
   if (index < N) {
   m = sbp[atoms[index].type].mass;
   rvec_Scale( p, m, atoms[index].v );
   tmp = 0.5 * rvec_Dot( p, atoms[index].v );
   }
   sh_ekin[threadIdx.x] = tmp;
   __syncthreads ();

   for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
   if (threadIdx.x < offset ) {
   index = threadIdx.x + offset;
   sh_ekin[threadIdx.x] += sh_ekin[ index ];
   }
   __syncthreads ();
   }

   if (threadIdx.x == 0) {
   output [ blockIdx.x ] = sh_ekin [ 0 ];
   }
   __syncthreads ();

//if ((blockIdx.x == 0) && (threadIdx.x < gridDim.x)) {
//    sh_ekin [ threadIdx.x ] = output [ threadIdx.x ];
//}
//__syncthreads ();


//gridDim indicates number of blocks configured for this invokation.
// in this case it will be BLOCKS_POW_2 == 16
for (int offset = gridDim.x/2; offset > 0; offset >>= 1) {
if ((threadIdx.x < offset ) && (blockIdx.x == 0)) {
index = threadIdx.x + offset;
//sh_ekin[threadIdx.x] += sh_ekin[ index ];
output [threadIdx.x] += output [index];
}
__syncthreads ();
}
__syncthreads ();

if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
data->E_Kin = output[0];
data->therm.T = (2. * data->E_Kin) / (data->N_f * K_B);

if ( fabs(data->therm.T) < ALMOST_ZERO ) // avoid T being an absolute zero! 
data->therm.T = ALMOST_ZERO;
}
}
 */
