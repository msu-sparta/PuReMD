
#include "cuda_system_props.h"

#include "cuda_utils.h"
#include "cuda_reduction.h"
#include "cuda_copy.h"
#include "cuda_shuffle.h"

#include "../vector.h"


CUDA_GLOBAL void center_of_mass_blocks( single_body_parameters *sbp, reax_atom *atoms,
        rvec *res_xcm, rvec *res_vcm, rvec *res_amcm, size_t n )
{
    extern __shared__ rvec xcm[];
    extern __shared__ rvec vcm[];
    extern __shared__ rvec amcm[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    //unsigned int xcm_id = threadIdx.x;
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


#if defined( __SM_35__)
CUDA_GLOBAL void center_of_mass_blocks_xcm( single_body_parameters *sbp, reax_atom *atoms,
        rvec *res_xcm, size_t n )
{
    extern __shared__ rvec my_xcm[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int xcm_id = threadIdx.x;
    unsigned int index = 0;
    rvec xcm;
    real m;

    rvec_MakeZero (xcm);

    if (i < n){
        m = sbp [ atoms[i].type ].mass;
        rvec_ScaledAdd (xcm , m, atoms [i].x);
    }
    __syncthreads ();

    for (int z = 16; z >= 1; z /= 2){
        xcm[0] += shfl( xcm[0], z);
        xcm[1] += shfl( xcm[1], z);
        xcm[2] += shfl( xcm[2], z);
    }
    __syncthreads ();

    if (threadIdx.x % 32 == 0)
        rvec_Copy( my_xcm[ threadIdx.x >> 5], xcm );
    __syncthreads ();

    for( int offset = blockDim.x >> 6; offset > 0; offset >>= 1 ) {

        if ((threadIdx.x < offset)) {
            index = threadIdx.x + offset;
            rvec_Add (my_xcm [threadIdx.x], my_xcm[index]);
        }
        __syncthreads ();
    }

    if ((threadIdx.x == 0))
        rvec_Copy (res_xcm[blockIdx.x], my_xcm[0]);
}


CUDA_GLOBAL void center_of_mass_blocks_vcm( single_body_parameters *sbp, reax_atom *atoms,
        rvec *res_vcm, size_t n )
{
    extern __shared__ rvec my_vcm[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = 0;
    rvec vcm;
    real m;

    rvec_MakeZero (vcm);

    if (i < n){
        m = sbp [ atoms[i].type ].mass;
        rvec_ScaledAdd (vcm , m, atoms [i].v);
    }
    __syncthreads ();

    for (int z = 16; z >= 1; z /= 2){
        vcm[0] += shfl( vcm[0], z);
        vcm[1] += shfl( vcm[1], z);
        vcm[2] += shfl( vcm[2], z);
    }
    __syncthreads ();

    if (threadIdx.x % 32 == 0)
        rvec_Copy( my_vcm[ threadIdx.x >> 5], vcm );
    __syncthreads ();

    for( int offset = blockDim.x >> 6; offset > 0; offset >>= 1 ) {

        if ((threadIdx.x < offset)) {
            index = threadIdx.x + offset;
            rvec_Add (my_vcm [threadIdx.x], my_vcm[index]);
        }
        __syncthreads ();
    }

    if ((threadIdx.x == 0))
        rvec_Copy (res_vcm[blockIdx.x], my_vcm[0]);
}


CUDA_GLOBAL void center_of_mass_blocks_amcm( single_body_parameters *sbp, reax_atom *atoms,
        rvec *res_amcm, size_t n )
{
    extern __shared__ rvec my_amcm[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = 0;
    rvec amcm;
    real m;
    rvec tmp;

    rvec_MakeZero (amcm);
    rvec_MakeZero( tmp );

    if (i < n){
        m = sbp [ atoms[i].type ].mass;
        rvec_Cross (tmp, atoms[i].x, atoms [i].v);
        rvec_ScaledAdd (amcm, m, tmp);
    }
    __syncthreads ();

    for (int z = 16; z >= 1; z /= 2){
        amcm[0] += shfl( amcm[0], z);
        amcm[1] += shfl( amcm[1], z);
        amcm[2] += shfl( amcm[2], z);
    }
    __syncthreads ();

    if (threadIdx.x % 32 == 0)
        rvec_Copy( my_amcm[ threadIdx.x >> 5], amcm );
    __syncthreads ();


    for( int offset = blockDim.x >> 6; offset > 0; offset >>= 1 ) {

        if ((threadIdx.x < offset)) {
            index = threadIdx.x + offset;
            rvec_Add (my_amcm[threadIdx.x], my_amcm[index]);
        }
        __syncthreads ();
    }

    if ((threadIdx.x == 0)){
        rvec_Copy (res_amcm[blockIdx.x], my_amcm[0]);
    }
}
#endif


CUDA_GLOBAL void center_of_mass( rvec *xcm, rvec *vcm, rvec *amcm, 
        rvec *res_xcm, rvec *res_vcm, rvec *res_amcm, size_t n )
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


CUDA_GLOBAL void compute_center_mass( single_body_parameters *sbp, 
        reax_atom *atoms, real *results, real xcm0, real xcm1, real xcm2,
        size_t n )
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


CUDA_GLOBAL void compute_center_mass( real *input, real *output, size_t n )
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


#if defined( __SM_35__)
CUDA_GLOBAL void compute_center_mass_xx_xy( single_body_parameters *sbp,
        reax_atom *atoms, real *results, real xcm0, real xcm1, real xcm2,
        size_t n )
{
    extern __shared__ real my_results_xx[];
    extern __shared__ real my_results_xy[];

    unsigned int xx_i = threadIdx.x;
    unsigned int xy_i = blockDim.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = 0;
    real xx = 0;
    real xy = 0;

    rvec diff, xcm;
    real m = 0;
    rvec_MakeZero (diff);
    xcm[0] = xcm0;
    xcm[1] = xcm1;
    xcm[2] = xcm2;


    if (i < n){
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1., atoms[i].x, -1., xcm );
        xx = diff[0] * diff[0] * m;
        xy = diff[0] * diff[1] * m;
    }
    __syncthreads ();

    for (int z = 16; z <= 1; z++){
        xx += shfl( xx, z);
        xy += shfl( xy, z);
    }
    __syncthreads ();

    if (threadIdx.x % 32 == 0){
        my_results_xx[threadIdx.x >> 5] = xx;    
        my_results_xy[threadIdx.x >> 5] = xy;    
    }
    __syncthreads ();

    for (int offset = blockDim.x >> 6; offset > 0; offset >>= 1){
        if (threadIdx.x < offset){
            index = threadIdx.x + offset;
            my_results_xx[ threadIdx.x ] += my_results_xx[ index ];
            my_results_xy[ xy_i + threadIdx.x ] += my_results_xy [ xy_i + index ];
        }
        __syncthreads ();
    }

    if (threadIdx.x == 0) {
        results [ blockIdx.x*6 ] = my_results_xx [ 0 ];
        results [ blockIdx.x*6 + 1 ] = my_results_xy [ xy_i + 0 ];
    }
}


CUDA_GLOBAL void compute_center_mass_xz_yy( single_body_parameters *sbp,
        reax_atom *atoms, real *results, real xcm0, real xcm1, real xcm2,
        size_t n )
{
    extern __shared__ real my_results_xz[];
    extern __shared__ real my_results_yy[];

    unsigned int yy_i = blockDim.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = 0;
    real xz = 0;
    real yy = 0;

    rvec diff, xcm;
    real m = 0;
    rvec_MakeZero (diff);
    xcm[0] = xcm0;
    xcm[1] = xcm1;
    xcm[2] = xcm2;

    if (i < n){
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1., atoms[i].x, -1., xcm );
        xz = diff[0] * diff[2] * m;
        yy = diff[1] * diff[1] * m;
    }
    __syncthreads ();

    for (int z = 16; z <= 1; z++){
        xz += shfl( xz, z);
        yy += shfl( yy, z);
    }
    __syncthreads ();

    if (threadIdx.x % 32 == 0){
        my_results_xz[threadIdx.x >> 5] = xz;    
        my_results_yy[threadIdx.x >> 5] = yy;    
    }
    __syncthreads ();

    for (int offset = blockDim.x >> 6; offset > 0; offset >>= 1){
        if (threadIdx.x < offset){
            index = threadIdx.x + offset;
            my_results_xz[ threadIdx.x ] += my_results_xz [ index ];
            my_results_yy[ yy_i + threadIdx.x ] += my_results_yy [ yy_i + index ];
        }
        __syncthreads ();
    }

    if (threadIdx.x == 0) {
        results [ blockIdx.x*6 + 2 ] = my_results_xz [ 0 ];
        results [ blockIdx.x*6 + 3 ] = my_results_yy [ yy_i + 0 ];
    }
}


CUDA_GLOBAL void compute_center_mass_yz_zz( single_body_parameters *sbp,
        reax_atom *atoms, real *results, real xcm0, real xcm1, real xcm2,
        size_t n )
{
    extern __shared__ real my_results_yz[];
    extern __shared__ real my_results_zz[];

    unsigned int zz_i = blockDim.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = 0;
    real yz = 0;
    real zz = 0;

    rvec diff, xcm;
    real m = 0;
    rvec_MakeZero (diff);
    xcm[0] = xcm0;
    xcm[1] = xcm1;
    xcm[2] = xcm2;

    if (i < n)
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1., atoms[i].x, -1., xcm );
        yz = diff[1] * diff[2] * m;
        zz = diff[2] * diff[2] * m;
    }
    __syncthreads ();

    for (int z = 16; z <= 1; z++){
        yz += shfl( yz, z);
        zz += shfl( zz, z);
    }
    __syncthreads ();

    if (threadIdx.x % 32 == 0){
        my_results_yz[threadIdx.x >> 5] = yz;    
        my_results_zz[threadIdx.x >> 5] = zz;    
    }
    __syncthreads ();

    for (int offset = blockDim.x >> 6; offset > 0; offset >>= 1){
        if (threadIdx.x < offset){
            index = threadIdx.x + offset;
            my_results_yz[ threadIdx.x ] += my_results_yz [ index ];
            my_results_zz[ zz_i + threadIdx.x ] += my_results_zz [ zz_i + index ];
        }
        __syncthreads ();
    }

    if (threadIdx.x == 0) {
        results [ blockIdx.x*6 + 4 ] = my_results_yz [ 0 ];
        results [ blockIdx.x*6 + 5 ] = my_results_zz [ zz_i + 0 ];
    }
}
#endif


CUDA_GLOBAL void k_compute_total_mass( single_body_parameters *sbp, reax_atom *my_atoms, 
        real *block_results, int n )
{
#if defined(__SM_35__)
    extern __shared__ real my_sbp[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real sdata = 0;

    if (i < n)
    {
        sdata = sbp[ my_atoms[i].type ].mass;
    }
    __syncthreads( );

    for(int z = 16; z >=1; z/=2)
    {
        sdata += shfl( sdata, z);
    }

    if (threadIdx.x % 32 == 0)
    {
        my_sbp[threadIdx.x >> 5] = sdata;
    }

    __syncthreads( );

    for(int offset = blockDim.x >> 6; offset > 0; offset >>= 1)
    {
        if(threadIdx.x < offset)
        {
            my_sbp[threadIdx.x] += my_sbp[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if(threadIdx.x == 0)
    {
        block_results[blockIdx.x] = my_sbp[0];
    }

#else
    extern __shared__ real sdata[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real x = 0;

    if (i < n)
    {
        x = sbp[ my_atoms[i].type ].mass;
    }

    sdata[ threadIdx.x ] = x;
    __syncthreads( );

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if (threadIdx.x == 0)
    {
        block_results[ blockIdx.x] = sdata [0];
    }

#endif
}


extern "C" void dev_compute_total_mass( reax_system *system, real *local_val )
{
    real *block_mass = (real *) scratch;
    cuda_memset( block_mass, 0, sizeof(real) * (1 + BLOCKS_POW_2), "total_mass:tmp" );

    k_compute_total_mass <<<BLOCKS, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE >>>
        (system->reax_param.d_sbp, system->d_my_atoms, block_mass, system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );

    k_reduction <<<1, BLOCKS_POW_2, sizeof(real) * BLOCKS_POW_2 >>>
        (block_mass, block_mass + BLOCKS_POW_2, BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device (local_val, block_mass + BLOCKS_POW_2, sizeof(real), 
            cudaMemcpyDeviceToHost, "total_mass:tmp");
}


CUDA_GLOBAL void k_compute_kinetic_energy( single_body_parameters *sbp, reax_atom *my_atoms, 
        real *block_results, int n )
{
#if defined(__SM_35__)
    extern __shared__ real my_sbpdot[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real sdata = 0;
    rvec p;

    if (i < n)
    {
        sdata = sbp[ my_atoms[i].type ].mass;
        rvec_Scale( p, sdata, my_atoms[ i ].v );
        sdata = 0.5 * rvec_Dot( p, my_atoms[ i ].v );
    }

    __syncthreads( );

    for(int z = 16; z >=1; z/=2)
    {
        sdata += shfl( sdata, z);
    }

    if (threadIdx.x % 32 == 0)
    {
        my_sbpdot[threadIdx.x >> 5] = sdata;
    }

    __syncthreads( );

    for (int offset = blockDim.x >> 6; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            my_sbpdot[threadIdx.x] += my_sbpdot[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if (threadIdx.x == 0)
    {
        block_results[blockIdx.x] = my_sbpdot[0];
    }

#else
    extern __shared__ real sdata [];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    real    m = 0;
    rvec p;

    if (i < n)
    {
        m = sbp[ my_atoms[i].type ].mass;
        rvec_Scale( p, m, my_atoms[ i ].v );
        m = 0.5 * rvec_Dot( p, my_atoms[ i ].v );
    }

    sdata[ threadIdx.x ] = m;
    __syncthreads( );

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
        if (threadIdx.x < offset)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if (threadIdx.x == 0)
    {
        block_results[blockIdx.x] = sdata[0];
    }
#endif
}

extern "C" void dev_compute_kinetic_energy( reax_system *system,
        simulation_data *data, real *local_val )
{
    real *block_energy = (real *) scratch;
    cuda_memset( block_energy, 0, sizeof(real) * (BLOCKS_POW_2 + 1), "kinetic_energy:tmp" );

    k_compute_kinetic_energy <<<BLOCKS, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE >>>
        (system->reax_param.d_sbp, system->d_my_atoms, block_energy, system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );

    k_reduction <<<1, BLOCKS_POW_2, sizeof(real) * BLOCKS_POW_2 >>>
        (block_energy, block_energy + BLOCKS_POW_2, BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( local_val, block_energy + BLOCKS_POW_2,
            //copy_host_device (local_val, &((simulation_data *)data->d_simulation_data)->my_en.e_kin, 
            sizeof(real), cudaMemcpyDeviceToHost, "kinetic_energy:tmp" );
            //copy_device (block_energy + BLOCKS_POW_2, &((simulation_data *)data->d_simulation_data)->my_en.e_kin,
            //        sizeof (real), "kinetic_energy");
}


extern "C" void dev_compute_momentum( reax_system *system, rvec xcm, 
        rvec vcm, rvec amcm )
{
    rvec *l_xcm, *l_vcm, *l_amcm;
    rvec *r_scratch = (rvec *)scratch;

#if defined( __SM_35__)
    // xcm
    cuda_memset( scratch, 0, sizeof(rvec) * (BLOCKS_POW_2 + 1), "momentum:tmp" );
    l_xcm = r_scratch;
    
    center_of_mass_blocks_xcm <<< BLOCKS_POW_2,BLOCK_SIZE,(sizeof(rvec) * BLOCK_SIZE) >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, l_xcm, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<<1, BLOCKS_POW_2, (sizeof(rvec) * BLOCKS_POW_2) >>>
            (l_xcm, l_xcm + BLOCKS_POW_2, BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );
    copy_host_device( xcm, l_xcm + BLOCKS_POW_2,
            sizeof(rvec), cudaMemcpyDeviceToHost, "momentum:xcm" );
    
    // vcm
    cuda_memset( scratch, 0, sizeof(rvec) * (BLOCKS_POW_2 + 1), "momentum:tmp" );
    l_vcm = r_scratch;
    
    center_of_mass_blocks_vcm <<< BLOCKS_POW_2,BLOCK_SIZE,(sizeof(rvec) * BLOCK_SIZE) >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, l_vcm, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<<1, BLOCKS_POW_2, (sizeof(rvec) * BLOCKS_POW_2) >>>
        (l_vcm, l_vcm + BLOCKS_POW_2, BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );
    copy_host_device( vcm, l_vcm + BLOCKS_POW_2, sizeof(rvec),
        cudaMemcpyDeviceToHost, "momentum:vcm" );
    
    // amcm
    cuda_memset( scratch, 0,  sizeof (rvec) * (BLOCKS_POW_2 + 1), "momentum:tmp");
    l_amcm = r_scratch;
    
    center_of_mass_blocks_amcm <<< BLOCKS_POW_2,BLOCK_SIZE,(sizeof(rvec) * BLOCK_SIZE) >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, l_amcm, system->n );
    cudaThreadSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<<1, BLOCKS_POW_2, (sizeof(rvec) * BLOCKS_POW_2) >>>
        (l_amcm, l_amcm + BLOCKS_POW_2, BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );
    copy_host_device( amcm, l_amcm + BLOCKS_POW_2, sizeof(rvec),
        cudaMemcpyDeviceToHost, "momemtum:amcm" );

#else
    cuda_memset( scratch, 0, 3 * sizeof (rvec) * (BLOCKS_POW_2 + 1), "momentum:tmp" );
    
    l_xcm = r_scratch;
    l_vcm = r_scratch + (BLOCKS_POW_2 + 1); 
    l_amcm = r_scratch + 2 * (BLOCKS_POW_2 + 1); 
    
    center_of_mass_blocks <<< BLOCKS_POW_2, BLOCK_SIZE, 3 * (sizeof (rvec) * BLOCK_SIZE) >>> 
        ( system->reax_param.d_sbp, system->d_my_atoms, l_xcm, l_vcm, l_amcm, system->n );
    cudaThreadSynchronize( ); 
    cudaCheckError( ); 
    
    center_of_mass <<< 1, BLOCKS_POW_2, 3 * (sizeof (rvec) * BLOCKS_POW_2) >>> 
        ( l_xcm, l_vcm, l_amcm, l_xcm + BLOCKS_POW_2, l_vcm + BLOCKS_POW_2,
          l_amcm + BLOCKS_POW_2, BLOCKS_POW_2 );
    cudaThreadSynchronize( ); 
    cudaCheckError( );
    
    copy_host_device( xcm, l_xcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost, "momemtum:xcm" );
    copy_host_device( vcm, l_vcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost, "momentum:vcm" );
    copy_host_device( amcm, l_amcm + BLOCKS_POW_2, sizeof (rvec), cudaMemcpyDeviceToHost,"momentum:amcm" );
#endif
}


extern "C" void dev_compute_inertial_tensor( reax_system *system, real *local_results, rvec my_xcm )
{
#if defined(__SM_35__)
    real *partial_results = (real *) scratch;
    cuda_memset( partial_results, 0, sizeof (real) * 6 * (BLOCKS_POW_2 + 1), "tensor:tmp" );

    compute_center_mass_xx_xy <<<BLOCKS_POW_2, BLOCK_SIZE, 2 * (sizeof (real) * BLOCK_SIZE) >>>
        (system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );

    compute_center_mass_xz_yy <<<BLOCKS_POW_2, BLOCK_SIZE, 2 * (sizeof (real) * BLOCK_SIZE) >>>
        (system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );

    compute_center_mass_yz_zz <<<BLOCKS_POW_2, BLOCK_SIZE, 2 * (sizeof (real) * BLOCK_SIZE) >>>
        (system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );

    compute_center_mass <<<1, BLOCKS_POW_2, 6 * (sizeof (real) * BLOCKS_POW_2) >>>
        (partial_results, partial_results + (BLOCKS_POW_2 * 6), BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device( local_results, partial_results + 6 * BLOCKS_POW_2,
        sizeof(real) * 6, cudaMemcpyDeviceToHost, "tensor:local_results" );

#else
    real *partial_results = (real *) scratch;
    //real *local_results;

    cuda_memset (partial_results, 0, sizeof (real) * 6 * (BLOCKS_POW_2 + 1), "tensor:tmp");
    //local_results = (real *) malloc (sizeof (real) * 6 *(BLOCKS_POW_2+ 1));

    compute_center_mass <<<BLOCKS_POW_2, BLOCK_SIZE, 6 * (sizeof (real) * BLOCK_SIZE) >>>
        (system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n);
    cudaThreadSynchronize( );
    cudaCheckError( );

    compute_center_mass <<<1, BLOCKS_POW_2, 6 * (sizeof (real) * BLOCKS_POW_2) >>>
        (partial_results, partial_results + (BLOCKS_POW_2 * 6), BLOCKS_POW_2);
    cudaThreadSynchronize( );
    cudaCheckError( );

    copy_host_device (local_results, partial_results + 6 * BLOCKS_POW_2, 
            sizeof(real) * 6, cudaMemcpyDeviceToHost, "tensor:local_results");
#endif
}


extern "C" void dev_sync_simulation_data( simulation_data *data )
{
    Output_Sync_Simulation_Data( data, (simulation_data *)data->d_simulation_data );
}


void Cuda_Compute_Kinetic_Energy( reax_system* system, simulation_data* data,
        MPI_Comm comm )
{
    int i;
    rvec p;
    real m;

    data->my_en.e_kin = 0.0;

    dev_compute_kinetic_energy( system, data, &data->my_en.e_kin );

    MPI_Allreduce( &data->my_en.e_kin,  &data->sys_en.e_kin,
            1, MPI_DOUBLE, MPI_SUM, comm );

    data->therm.T = (2. * data->sys_en.e_kin) / (data->N_f * K_B);

    // avoid T being an absolute zero, might cause F.P.E!
    if ( FABS(data->therm.T) < ALMOST_ZERO )
    {
        data->therm.T = ALMOST_ZERO;
    }
}


void Cuda_Compute_Total_Mass( reax_system *system, simulation_data *data,
        MPI_Comm comm  )
{
    int  i;
    real tmp;

    //compute local total mass of the system
    dev_compute_total_mass( system, &tmp );

    MPI_Allreduce( &tmp, &data->M, 1, MPI_DOUBLE, MPI_SUM, comm );

    data->inv_M = 1. / data->M;
}


void Cuda_Compute_Center_of_Mass( reax_system *system, simulation_data *data,
        mpi_datatypes *mpi_data, MPI_Comm comm )
{
    int i;
    real m, det; //xx, xy, xz, yy, yz, zz;
    real tmp_mat[6], tot_mat[6];
    rvec my_xcm, my_vcm, my_amcm, my_avcm;
    rvec tvec, diff;
    rtensor mat, inv;

    rvec_MakeZero( my_xcm );  // position of CoM
    rvec_MakeZero( my_vcm );  // velocity of CoM
    rvec_MakeZero( my_amcm ); // angular momentum of CoM
    rvec_MakeZero( my_avcm ); // angular velocity of CoM

    /* Compute the position, vel. and ang. momentum about the centre of mass */
    dev_compute_momentum ( system, my_xcm, my_vcm, my_amcm );

    MPI_Allreduce( my_xcm, data->xcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( my_vcm, data->vcm, 3, MPI_DOUBLE, MPI_SUM, comm );
    MPI_Allreduce( my_amcm, data->amcm, 3, MPI_DOUBLE, MPI_SUM, comm );

    rvec_Scale( data->xcm, data->inv_M, data->xcm );
    rvec_Scale( data->vcm, data->inv_M, data->vcm );
    rvec_Cross( tvec, data->xcm, data->vcm );
    rvec_ScaledAdd( data->amcm, -data->M, tvec );
    data->etran_cm = 0.5 * data->M * rvec_Norm_Sqr( data->vcm );

    /* Calculate and then invert the inertial tensor */
    for ( i = 0; i < 6; ++i )
    {
        tmp_mat[i] = 0;
    }

    dev_compute_inertial_tensor( system, tmp_mat, my_xcm );

    MPI_Reduce( tmp_mat, tot_mat, 6, MPI_DOUBLE, MPI_SUM, MASTER_NODE, comm );

    if ( system->my_rank == MASTER_NODE )
    {
        mat[0][0] = tot_mat[3] + tot_mat[5];  // yy + zz;
        mat[0][1] = mat[1][0] = -tot_mat[1];  // -xy;
        mat[0][2] = mat[2][0] = -tot_mat[2];  // -xz;
        mat[1][1] = tot_mat[0] + tot_mat[5];  // xx + zz;
        mat[2][1] = mat[1][2] = -tot_mat[4];  // -yz;
        mat[2][2] = tot_mat[0] + tot_mat[3];  // xx + yy;

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

        if ( det > ALMOST_ZERO )
        {
            rtensor_Scale( inv, 1. / det, inv );
        }
        else
        {
            rtensor_MakeZero( inv );
        }

        /* Compute the angular velocity about the centre of mass */
        rtensor_MatVec( data->avcm, inv, data->amcm );
    }

    MPI_Bcast( data->avcm, 3, MPI_DOUBLE, MASTER_NODE, comm );

    /* Compute the rotational energy */
    data->erot_cm = 0.5 * E_CONV * rvec_Dot( data->avcm, data->amcm );

#if defined(DEBUG)
    fprintf( stderr, "xcm:  %24.15e %24.15e %24.15e\n",
             data->xcm[0], data->xcm[1], data->xcm[2] );
    fprintf( stderr, "vcm:  %24.15e %24.15e %24.15e\n",
             data->vcm[0], data->vcm[1], data->vcm[2] );
    fprintf( stderr, "amcm: %24.15e %24.15e %24.15e\n",
             data->amcm[0], data->amcm[1], data->amcm[2] );
    /* fprintf( stderr, "mat:  %f %f %f\n     %f %f %f\n     %f %f %f\n",
       mat[0][0], mat[0][1], mat[0][2],
       mat[1][0], mat[1][1], mat[1][2],
       mat[2][0], mat[2][1], mat[2][2] );
       fprintf( stderr, "inv:  %g %g %g\n     %g %g %g\n     %g %g %g\n",
       inv[0][0], inv[0][1], inv[0][2],
       inv[1][0], inv[1][1], inv[1][2],
       inv[2][0], inv[2][1], inv[2][2] ); */
    fprintf( stderr, "avcm: %24.15e %24.15e %24.15e\n",
             data->avcm[0], data->avcm[1], data->avcm[2] );
#endif
}


