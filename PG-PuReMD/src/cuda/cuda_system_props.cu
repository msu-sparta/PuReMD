
#include "cuda_system_props.h"

#include "cuda_copy.h"
#include "cuda_utils.h"
#include "cuda_random.h"
#include "cuda_reduction.h"
#include "cuda_shuffle.h"
#include "cuda_vector.h"

#include "../tool_box.h"
#include "../vector.h"


CUDA_GLOBAL void k_center_of_mass_blocks( single_body_parameters *sbp, reax_atom *atoms,
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

    rvec_MakeZero( xcm[threadIdx.x] );
    rvec_MakeZero( vcm[vcm_id + threadIdx.x] );
    rvec_MakeZero( amcm[amcm_id + threadIdx.x] );
    rvec_MakeZero( tmp );

    if ( i < n )
    {
        m = sbp [ atoms[i].type ].mass;
        rvec_ScaledAdd( xcm [threadIdx.x], m, atoms[i].x );
        rvec_ScaledAdd( vcm [vcm_id + threadIdx.x], m, atoms[i].v );
        rvec_Cross( tmp, atoms[i].x, atoms[i].v );
        rvec_ScaledAdd( amcm[amcm_id + threadIdx.x], m, tmp );
    }
    __syncthreads( );

    for ( int offset = blockDim.x / 2; offset > 0; offset >>= 1 )
    { 
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            rvec_Add( xcm[threadIdx.x], xcm[index] );
            rvec_Add( vcm[vcm_id  + threadIdx.x], vcm[vcm_id + index] );
            rvec_Add( amcm[amcm_id + threadIdx.x], amcm[amcm_id + index] );
        } 
        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        rvec_Copy( res_xcm[blockIdx.x], xcm[0] );
        rvec_Copy( res_vcm[blockIdx.x], vcm[vcm_id] );
        rvec_Copy( res_amcm[blockIdx.x], amcm[amcm_id] );
    }
}


#if defined( __SM_35__)
CUDA_GLOBAL void k_center_of_mass_blocks_xcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *res_xcm, size_t n )
{
    extern __shared__ rvec my_xcm[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int xcm_id = threadIdx.x;
    unsigned int index = 0;
    rvec xcm;
    real m;

    rvec_MakeZero( xcm );

    if ( i < n )
    {
        m = sbp [ atoms[i].type ].mass;
        rvec_ScaledAdd( xcm, m, atoms[i].x );
    }
    __syncthreads( );

    for ( int z = 16; z >= 1; z /= 2 )
    {
        xcm[0] += shfl( xcm[0], z);
        xcm[1] += shfl( xcm[1], z);
        xcm[2] += shfl( xcm[2], z);
    }
    __syncthreads( );

    if ( threadIdx.x % 32 == 0 )
    {
        rvec_Copy( my_xcm[ threadIdx.x >> 5 ], xcm );
    }
    __syncthreads( );

    for ( int offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            rvec_Add( my_xcm[threadIdx.x], my_xcm[index] );
        }

        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        rvec_Copy( res_xcm[blockIdx.x], my_xcm[0] );
    }
}


CUDA_GLOBAL void k_center_of_mass_blocks_vcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *res_vcm, size_t n )
{
    extern __shared__ rvec my_vcm[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = 0;
    rvec vcm;
    real m;

    rvec_MakeZero( vcm );

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledAdd( vcm, m, atoms[i].v );
    }
    __syncthreads( );

    for ( int z = 16; z >= 1; z /= 2 )
    {
        vcm[0] += shfl( vcm[0], z );
        vcm[1] += shfl( vcm[1], z );
        vcm[2] += shfl( vcm[2], z );
    }
    __syncthreads( );

    if ( threadIdx.x % 32 == 0 )
    {
        rvec_Copy( my_vcm[ threadIdx.x >> 5 ], vcm );
    }
    __syncthreads( );

    for ( int offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            rvec_Add( my_vcm[threadIdx.x], my_vcm[index] );
        }
        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        rvec_Copy( res_vcm[blockIdx.x], my_vcm[0] );
    }
}


CUDA_GLOBAL void k_center_of_mass_blocks_amcm( single_body_parameters *sbp,
        reax_atom *atoms, rvec *res_amcm, size_t n )
{
    extern __shared__ rvec my_amcm[];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int index = 0;
    rvec amcm;
    real m;
    rvec tmp;

    rvec_MakeZero( amcm );
    rvec_MakeZero( tmp );

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_Cross( tmp, atoms[i].x, atoms [i].v );
        rvec_ScaledAdd( amcm, m, tmp );
    }
    __syncthreads( );

    for ( int z = 16; z >= 1; z /= 2 )
    {
        amcm[0] += shfl( amcm[0], z );
        amcm[1] += shfl( amcm[1], z );
        amcm[2] += shfl( amcm[2], z );
    }
    __syncthreads( );

    if ( threadIdx.x % 32 == 0 )
    {
        rvec_Copy( my_amcm[ threadIdx.x >> 5 ], amcm );
    }
    __syncthreads( );


    for ( int offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            rvec_Add( my_amcm[threadIdx.x], my_amcm[index] );
        }
        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        rvec_Copy( res_amcm[blockIdx.x], my_amcm[0] );
    }
}
#endif


CUDA_GLOBAL void k_center_of_mass( rvec *xcm, rvec *vcm, rvec *amcm, 
        rvec *res_xcm, rvec *res_vcm, rvec *res_amcm, size_t n )
{
    extern __shared__ rvec sh_xcm[];
    extern __shared__ rvec sh_vcm[];
    extern __shared__ rvec sh_amcm[];

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    unsigned int xcm_id = threadIdx.x;
    unsigned int vcm_id = blockDim.x;
    unsigned int amcm_id = 2 * blockDim.x;

    unsigned int index = 0;
    rvec t_xcm, t_vcm, t_amcm;

    rvec_MakeZero( t_xcm );
    rvec_MakeZero( t_vcm );
    rvec_MakeZero( t_amcm );

    if ( i < n )
    {
        rvec_Copy( t_xcm, xcm[threadIdx.x] );
        rvec_Copy( t_vcm, vcm[threadIdx.x] );
        rvec_Copy( t_amcm, amcm[threadIdx.x] );
    }

    rvec_Copy( sh_xcm[xcm_id], t_xcm );
    rvec_Copy( sh_vcm[vcm_id + threadIdx.x], t_vcm );
    rvec_Copy( sh_amcm[amcm_id + threadIdx.x], t_amcm );

    __syncthreads( );

    for ( int offset = blockDim.x / 2; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            rvec_Add( sh_xcm[threadIdx.x], sh_xcm[index] );
            rvec_Add( sh_vcm[vcm_id + threadIdx.x], sh_vcm[vcm_id + index] );
            rvec_Add( sh_amcm[amcm_id + threadIdx.x], sh_amcm[amcm_id + index] );
        } 
        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        rvec_Copy( res_xcm[blockIdx.x], sh_xcm[0] );
        rvec_Copy( res_vcm[blockIdx.x], sh_vcm[vcm_id] );
        rvec_Copy( res_amcm[blockIdx.x], sh_amcm[amcm_id] );
    }
}


CUDA_GLOBAL void k_compute_center_mass( single_body_parameters *sbp, 
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


CUDA_GLOBAL void k_compute_center_mass_opt( real *input, real *output, size_t n )
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
    int offset;

    xx[xx_i] = 0.0;
    xy[xy_i + threadIdx.x] = 0.0;
    xz[xz_i + threadIdx.x] = 0.0;
    yy[yy_i + threadIdx.x] = 0.0;
    yz[yz_i + threadIdx.x] = 0.0;
    zz[zz_i + threadIdx.x] = 0.0;

    if ( i < n )
    {
        xx[ xx_i ] = input [ threadIdx.x * 6 + 0 ];
        xy[ xy_i + threadIdx.x ] = input[ threadIdx.x * 6 + 1 ];
        xz[ xz_i + threadIdx.x ] = input[ threadIdx.x * 6 + 2 ];
        yy[ yy_i + threadIdx.x ] = input[ threadIdx.x * 6 + 3 ];
        yz[ yz_i + threadIdx.x ] = input[ threadIdx.x * 6 + 4 ];
        zz[ zz_i + threadIdx.x ] = input[ threadIdx.x * 6 + 5 ];
    }
    __syncthreads( );

    for ( offset = blockDim.x / 2; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            xx[ threadIdx.x ] += xx[ index ];
            xy[ xy_i + threadIdx.x ] += xy[ xy_i + index ];
            xz[ xz_i + threadIdx.x ] += xz[ xz_i + index ];
            yy[ yy_i + threadIdx.x ] += yy[ yy_i + index ];
            yz[ yz_i + threadIdx.x ] += yz[ yz_i + index ];
            zz[ zz_i + threadIdx.x ] += zz[ zz_i + index ];
        }
        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
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
CUDA_GLOBAL void k_compute_center_mass_xx_xy( single_body_parameters *sbp,
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

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1., atoms[i].x, -1., xcm );
        xx = diff[0] * diff[0] * m;
        xy = diff[0] * diff[1] * m;
    }
    __syncthreads( );

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


CUDA_GLOBAL void k_compute_center_mass_xz_yy( single_body_parameters *sbp,
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


CUDA_GLOBAL void k_compute_center_mass_yz_zz( single_body_parameters *sbp,
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
    rvec_MakeZero( diff );
    xcm[0] = xcm0;
    xcm[1] = xcm1;
    xcm[2] = xcm2;

    if ( i < n )
    {
        m = sbp[ atoms[i].type ].mass;
        rvec_ScaledSum( diff, 1.0, atoms[i].x, -1.0, xcm );
        yz = diff[1] * diff[2] * m;
        zz = diff[2] * diff[2] * m;
    }
    __syncthreads( );

    for ( int z = 16; z <= 1; z++ )
    {
        yz += shfl( yz, z );
        zz += shfl( zz, z );
    }
    __syncthreads( );

    if ( threadIdx.x % 32 == 0 )
    {
        my_results_yz[threadIdx.x >> 5] = yz;    
        my_results_zz[threadIdx.x >> 5] = zz;    
    }
    __syncthreads( );

    for ( int offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            index = threadIdx.x + offset;
            my_results_yz[ threadIdx.x ] += my_results_yz[ index ];
            my_results_zz[ zz_i + threadIdx.x ] += my_results_zz[ zz_i + index ];
        }
        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        results [ blockIdx.x * 6 + 4 ] = my_results_yz[ 0 ];
        results [ blockIdx.x * 6 + 5 ] = my_results_zz[ zz_i + 0 ];
    }
}
#endif


CUDA_GLOBAL void k_compute_total_mass( single_body_parameters *sbp, reax_atom *my_atoms, 
        real *block_results, int n )
{
#if defined(__SM_35__)
    extern __shared__ real my_sbp[];
    unsigned int i;
    int z, offset;
    real sdata;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        sdata = sbp[ my_atoms[i].type ].mass;
    }
    else
    {
        sdata = 0.0;
    }
    __syncthreads( );

    for ( z = 16; z >= 1; z /= 2 )
    {
        sdata += shfl( sdata, z );
    }

    if ( threadIdx.x % 32 == 0 )
    {
        my_sbp[threadIdx.x >> 5] = sdata;
    }

    __syncthreads( );

    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            my_sbp[threadIdx.x] += my_sbp[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        block_results[blockIdx.x] = my_sbp[0];
    }

#else
    extern __shared__ real sdata[];
    unsigned int i;
    int offset;
    real x;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        x = sbp[ my_atoms[i].type ].mass;
    }
    else
    {
        x = 0.0;
    }
    __syncthreads( );

    sdata[ threadIdx.x ] = x;
    __syncthreads( );

    for ( offset = blockDim.x / 2; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        block_results[ blockIdx.x] = sdata[0];
    }
#endif
}


CUDA_GLOBAL void k_compute_kinetic_energy( single_body_parameters *sbp, reax_atom *my_atoms, 
        real *block_results, int n )
{
#if defined(__SM_35__)
    extern __shared__ real my_sbpdot[];
    unsigned int i;
    int offset, z;
    real sdata;
    rvec p;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        rvec_Scale( p, sbp[ my_atoms[i].type ].mass, my_atoms[ i ].v );
        sdata = 0.5 * rvec_Dot( p, my_atoms[ i ].v );
    }
    else
    {
        sdata = 0;
    }

    __syncthreads( );

    for ( z = 16; z >= 1; z /= 2 )
    {
        sdata += shfl( sdata, z );
    }

    if ( threadIdx.x % 32 == 0 )
    {
        my_sbpdot[threadIdx.x >> 5] = sdata;
    }

    __syncthreads( );

    for ( offset = blockDim.x >> 6; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            my_sbpdot[threadIdx.x] += my_sbpdot[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        block_results[blockIdx.x] = my_sbpdot[0];
    }

#else
    extern __shared__ real sdata[];
    unsigned int i;
    int offset;
    real m;
    rvec p;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < n )
    {
        rvec_Scale( p, sbp[ my_atoms[i].type ].mass, my_atoms[ i ].v );
        m = 0.5 * rvec_Dot( p, my_atoms[ i ].v );
    }
    else
    {
        m = 0;
    }

    sdata[ threadIdx.x ] = m;
    __syncthreads( );

    for ( offset = blockDim.x / 2; offset > 0; offset >>= 1 )
    {
        if ( threadIdx.x < offset )
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }

        __syncthreads( );
    }

    if ( threadIdx.x == 0 )
    {
        block_results[blockIdx.x] = sdata[0];
    }
#endif
}


CUDA_GLOBAL void k_generate_initial_velocities( single_body_parameters *sbp, reax_atom *my_atoms, 
        real T, int n )
{
    int i;
    real m, scale, norm;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    if ( T <= 0.1 )
    {
        rvec_MakeZero( my_atoms[i].v );
    }
    else
    {
        cuda_rvec_Random( my_atoms[i].v );

        norm = rvec_Norm_Sqr( my_atoms[i].v );
        m = sbp[ my_atoms[i].type ].mass;
        scale = SQRT( m * norm / (3.0 * K_B * T) );

        rvec_Scale( my_atoms[i].v, 1. / scale, my_atoms[i].v );
    }
}


CUDA_GLOBAL void k_compute_pressure( reax_atom *my_atoms, simulation_box *big_box,
        rvec *int_press, int n )
{
    reax_atom *p_atom;
    rvec tx;
    int i;

    i = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= n )
    {
        return;
    }

    p_atom = &my_atoms[i];
    rvec_MakeZero( int_press[i] );

    /* transform x into unit box coordinates, store in tx */
    Transform_to_UnitBox( p_atom->x, big_box, 1, tx );

    /* this atom's contribution to internal pressure */
    rvec_Multiply( int_press[i], p_atom->f, tx );
}


static void Cuda_Compute_Momentum( reax_system *system, control_params *control,
        storage *workspace, rvec xcm, rvec vcm, rvec amcm )
{
    rvec *l_xcm, *l_vcm, *l_amcm, *r_scratch;

    r_scratch = (rvec *) workspace->scratch;

#if defined( __SM_35__)
    // xcm
    cuda_memset( workspace->scratch, 0, sizeof(rvec) * (control->blocks_pow_2 + 1), "momentum:tmp" );
    l_xcm = r_scratch;
    
    k_center_of_mass_blocks_xcm <<< control->blocks_pow_2, control->block_size, sizeof(rvec) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, l_xcm, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<< 1, control->blocks_pow_2, sizeof(rvec) * control->blocks_pow_2 >>>
            (l_xcm, l_xcm + control->blocks_pow_2, control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    copy_host_device( xcm, l_xcm + control->blocks_pow_2,
            sizeof(rvec), cudaMemcpyDeviceToHost, "momentum:xcm" );
    
    // vcm
    cuda_memset( workspace->scratch, 0, sizeof(rvec) * (control->blocks_pow_2 + 1), "momentum:tmp" );
    l_vcm = r_scratch;
    
    k_center_of_mass_blocks_vcm <<< control->blocks_pow_2, control->block_size, sizeof(rvec) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, l_vcm, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<< 1, control->blocks_pow_2, sizeof(rvec) * control->blocks_pow_2 >>>
        ( l_vcm, l_vcm + control->blocks_pow_2, control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    copy_host_device( vcm, l_vcm + control->blocks_pow_2, sizeof(rvec),
        cudaMemcpyDeviceToHost, "momentum:vcm" );
    
    // amcm
    cuda_memset( workspace->scratch, 0,  sizeof (rvec) * (control->blocks_pow_2 + 1), "momentum:tmp");
    l_amcm = r_scratch;
    
    k_center_of_mass_blocks_amcm <<< control->blocks_pow_2, control->block_size, sizeof(rvec) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, l_amcm, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    
    k_reduction_rvec <<< 1, control->blocks_pow_2, sizeof(rvec) * control->blocks_pow_2 >>>
        ( l_amcm, l_amcm + control->blocks_pow_2, control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );
    copy_host_device( amcm, l_amcm + control->blocks_pow_2, sizeof(rvec),
        cudaMemcpyDeviceToHost, "momemtum:amcm" );

#else
    cuda_memset( workspace->scratch, 0, 3 * sizeof(rvec) * (control->blocks_pow_2 + 1), "momentum:tmp" );
    
    l_xcm = r_scratch;
    l_vcm = r_scratch + (control->blocks_pow_2 + 1); 
    l_amcm = r_scratch + 2 * (control->blocks_pow_2 + 1); 
    
    k_center_of_mass_blocks <<< control->blocks_pow_2, control->block_size, 3 * sizeof(rvec) * control->block_size >>> 
        ( system->reax_param.d_sbp, system->d_my_atoms, l_xcm, l_vcm, l_amcm, system->n );
    cudaDeviceSynchronize( ); 
    cudaCheckError( ); 
    
    k_center_of_mass <<< 1, control->blocks_pow_2, sizeof(rvec) * 3 * control->blocks_pow_2 >>> 
        ( l_xcm, l_vcm, l_amcm, l_xcm + control->blocks_pow_2, l_vcm + control->blocks_pow_2,
          l_amcm + control->blocks_pow_2, control->blocks_pow_2 );
    cudaDeviceSynchronize( ); 
    cudaCheckError( );
    
    copy_host_device( xcm, l_xcm + control->blocks_pow_2, sizeof(rvec),
            cudaMemcpyDeviceToHost, "momemtum:xcm" );
    copy_host_device( vcm, l_vcm + control->blocks_pow_2, sizeof(rvec),
            cudaMemcpyDeviceToHost, "momentum:vcm" );
    copy_host_device( amcm, l_amcm + control->blocks_pow_2, sizeof(rvec),
            cudaMemcpyDeviceToHost,"momentum:amcm" );
#endif
}


static void Cuda_Compute_Inertial_Tensor( reax_system *system, control_params *control,
        storage *workspace, real *local_results, rvec my_xcm )
{
#if defined(__SM_35__)
    real *partial_results = (real *) workspace->scratch;
    cuda_memset( partial_results, 0, sizeof(real) * 6 * (control->blocks_pow_2 + 1), "tensor:tmp" );

    k_compute_center_mass_xx_xy <<< control->blocks_pow_2, control->block_size, sizeof(real) * 2 * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_compute_center_mass_xz_yy <<< control->blocks_pow_2, control->block_size, sizeof(real) * 2 * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, partial_results,
         my_xcm[0], my_xcm[1], my_xcm[2], system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_compute_center_mass_yz_zz <<< control->blocks_pow_2, control->block_size, sizeof(real) * 2 * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, partial_results,
          my_xcm[0], my_xcm[1], my_xcm[2], system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_compute_center_mass_opt <<< 1, control->blocks_pow_2, sizeof(real) * 6 * control->blocks_pow_2 >>>
        ( partial_results, partial_results + (control->blocks_pow_2 * 6), control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( local_results, partial_results + 6 * control->blocks_pow_2,
        sizeof(real) * 6, cudaMemcpyDeviceToHost, "tensor:local_results" );

#else
    real *partial_results = (real *) workspace->scratch;
    //real *local_results;

    cuda_memset( partial_results, 0, sizeof(real) * 6 * (control->blocks_pow_2 + 1), "tensor:tmp" );
    //local_results = (real *) malloc( sizeof(real) * 6 * (control->blocks_pow_2 + 1) );

    k_compute_center_mass <<< control->blocks_pow_2, control->block_size, sizeof(real) * 6 * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, partial_results,
          my_xcm[0], my_xcm[1], my_xcm[2], system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_compute_center_mass_opt <<< 1, control->blocks_pow_2, sizeof(real) * 6 * control->blocks_pow_2 >>>
        ( partial_results, partial_results + (control->blocks_pow_2 * 6), control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( local_results, partial_results + 6 * control->blocks_pow_2, 
            sizeof(real) * 6, cudaMemcpyDeviceToHost, "tensor:local_results" );
#endif
}


extern "C" void Cuda_Sync_Simulation_Data( simulation_data *data )
{
    Output_Sync_Simulation_Data( data, (simulation_data *)data->d_simulation_data );
}


void Cuda_Generate_Initial_Velocities( reax_system *system, real T )
{
    int blocks;

    blocks = system->n / DEF_BLOCK_SIZE + 
        ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

    if ( T > 0.1 )
    {
        Cuda_Randomize( );
    }

    k_generate_initial_velocities <<< blocks, DEF_BLOCK_SIZE >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, T, system->n );
}


void Cuda_Compute_Kinetic_Energy( reax_system* system, control_params *control,
        storage *workspace, simulation_data* data, MPI_Comm comm )
{
    real *block_energy;

    block_energy = (real *) workspace->scratch;
    data->my_en.e_kin = 0.0;

    cuda_memset( block_energy, 0, sizeof(real) * (control->blocks_pow_2 + 1), "kinetic_energy:tmp" );

    k_compute_kinetic_energy <<< control->blocks, control->block_size, sizeof(real) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, block_energy, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_reduction <<< 1, control->blocks_pow_2, sizeof(real) * control->blocks_pow_2 >>>
        ( block_energy, block_energy + control->blocks_pow_2, control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    //copy_host_device( &data->my_en.e_kin, &((simulation_data *)data->d_simulation_data)->my_en.e_kin, 
    copy_host_device( &data->my_en.e_kin, block_energy + control->blocks_pow_2,
            sizeof(real), cudaMemcpyDeviceToHost, "kinetic_energy:tmp" );
    //copy_device( block_energy + control->blocks_pow_2, &((simulation_data *)data->d_simulation_data)->my_en.e_kin,
    //        sizeof(real), "kinetic_energy" );

    MPI_Allreduce( &data->my_en.e_kin, &data->sys_en.e_kin,
            1, MPI_DOUBLE, MPI_SUM, comm );

    data->therm.T = (2.0 * data->sys_en.e_kin) / (data->N_f * K_B);

    /* avoid T being an absolute zero, might cause F.P.E! */
    if ( FABS(data->therm.T) < ALMOST_ZERO )
    {
        data->therm.T = ALMOST_ZERO;
    }
}


void Check_Energy( simulation_data* data )
{
    if ( IS_NAN_REAL(data->my_en.e_kin) )
    {
        fprintf( stderr, "[ERROR] NaN detected for kinetic energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }

    if ( IS_NAN_REAL(data->my_en.e_pot) )
    {
        fprintf( stderr, "[ERROR] NaN detected for potential energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }

    if ( IS_NAN_REAL(data->my_en.e_tot) )
    {
        fprintf( stderr, "[ERROR] NaN detected for total energy. Terminating...\n" );
        exit( NUMERIC_BREAKDOWN );
    }
}


void Cuda_Compute_Total_Mass( reax_system *system, control_params *control,
        storage *workspace, simulation_data *data, MPI_Comm comm  )
{
    real tmp;
    real *block_mass;

    block_mass = (real * ) workspace->scratch;

    cuda_memset( block_mass, 0, sizeof(real) * (1 + control->blocks_pow_2),
            "Cuda_Compute_Total_Mass::block_mass" );

    k_compute_total_mass <<< control->blocks, control->block_size, sizeof(real) * control->block_size >>>
        ( system->reax_param.d_sbp, system->d_my_atoms, block_mass, system->n );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    k_reduction <<< 1, control->blocks_pow_2, sizeof(real) * control->blocks_pow_2 >>>
        ( block_mass, block_mass + control->blocks_pow_2, control->blocks_pow_2 );
    cudaDeviceSynchronize( );
    cudaCheckError( );

    copy_host_device( &tmp, block_mass + control->blocks_pow_2, sizeof(real), 
            cudaMemcpyDeviceToHost, "total_mass:tmp" );

    MPI_Allreduce( &tmp, &data->M, 1, MPI_DOUBLE, MPI_SUM, comm );

    data->inv_M = 1.0 / data->M;
}


void Cuda_Compute_Center_of_Mass( reax_system *system, control_params *control,
        storage *workspace, simulation_data *data, mpi_datatypes *mpi_data, MPI_Comm comm )
{
    int i;
    real det; //xx, xy, xz, yy, yz, zz;
    real tmp_mat[6], tot_mat[6];
    rvec my_xcm, my_vcm, my_amcm, my_avcm;
    rvec tvec;
    rtensor mat, inv;

    rvec_MakeZero( my_xcm );  // position of CoM
    rvec_MakeZero( my_vcm );  // velocity of CoM
    rvec_MakeZero( my_amcm ); // angular momentum of CoM
    rvec_MakeZero( my_avcm ); // angular velocity of CoM

    /* Compute the position, vel. and ang. momentum about the centre of mass */
    Cuda_Compute_Momentum( system, control, workspace, my_xcm, my_vcm, my_amcm );

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

    Cuda_Compute_Inertial_Tensor( system, control, workspace, tmp_mat, my_xcm );

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

#if defined(DEBUG_FOCUS)
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


/* IMPORTANT: This function assumes that current kinetic energy
 * the system is already computed
 *
 * IMPORTANT: In Klein's paper, it is stated that a dU/dV term needs
 *  to be added when there are long-range interactions or long-range
 *  corrections to short-range interactions present.
 *  We may want to add that for more accuracy.
 */
void Cuda_Compute_Pressure( reax_system* system, control_params *control,
        storage *workspace, simulation_data* data, mpi_datatypes *mpi_data )
{
    int blocks, block_size, blocks_n, blocks_pow_2_n;
    rvec *rvec_spad;
    rvec int_press;
    simulation_box *big_box;
    
    rvec_spad = (rvec *) workspace->scratch;
    big_box = &system->big_box;

    /* 0: both int and ext, 1: ext only, 2: int only */
    if ( control->press_mode == 0 || control->press_mode == 2 )
    {
        blocks = system->n / DEF_BLOCK_SIZE + 
            ((system->n % DEF_BLOCK_SIZE == 0) ? 0 : 1);

        compute_blocks( &blocks_n, &block_size, system->n );
        compute_nearest_pow_2( blocks_n, &blocks_pow_2_n );

        k_compute_pressure <<< blocks, DEF_BLOCK_SIZE >>>
            ( system->d_my_atoms, system->d_big_box, rvec_spad,
              system->n );

        k_reduction_rvec <<< blocks_n, block_size, sizeof(rvec) * block_size >>>
            ( rvec_spad, rvec_spad + system->n,  system->n );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        k_reduction_rvec <<< 1, blocks_pow_2_n, sizeof(rvec) * blocks_pow_2_n >>>
            ( rvec_spad + system->n, rvec_spad + system->n + blocks_n, blocks_n );
        cudaDeviceSynchronize( );
        cudaCheckError( );

        copy_host_device( &int_press, rvec_spad + system->n + blocks_n, sizeof(rvec), 
                cudaMemcpyDeviceToHost, "Cuda_Compute_Pressure::d_int_press" );
    }

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d:p_int(%10.5f %10.5f %10.5f)p_ext(%10.5f %10.5f %10.5f)\n",
            system->my_rank, int_press[0], int_press[1], int_press[2],
            data->my_ext_press[0], data->my_ext_press[1], data->my_ext_press[2] );
#endif

    /* sum up internal and external pressure */
    MPI_Allreduce( int_press, data->int_press,
            3, MPI_DOUBLE, MPI_SUM, mpi_data->comm_mesh3D );
    MPI_Allreduce( data->my_ext_press, data->ext_press,
            3, MPI_DOUBLE, MPI_SUM, mpi_data->comm_mesh3D );

#if defined(DEBUG_FOCUS)
    fprintf( stderr, "p%d: %10.5f %10.5f %10.5f\n",
             system->my_rank,
             data->int_press[0], data->int_press[1], data->int_press[2] );
    fprintf( stderr, "p%d: %10.5f %10.5f %10.5f\n",
             system->my_rank,
             data->ext_press[0], data->ext_press[1], data->ext_press[2] );
#endif

    /* kinetic contribution */
    data->kin_press = 2.0 * (E_CONV * data->sys_en.e_kin)
        / (3.0 * big_box->V * P_CONV);

    /* Calculate total pressure in each direction */
    data->tot_press[0] = data->kin_press -
        (( data->int_press[0] + data->ext_press[0] ) /
         ( big_box->box_norms[1] * big_box->box_norms[2] * P_CONV ));

    data->tot_press[1] = data->kin_press -
        (( data->int_press[1] + data->ext_press[1] ) /
         ( big_box->box_norms[0] * big_box->box_norms[2] * P_CONV ));

    data->tot_press[2] = data->kin_press -
        (( data->int_press[2] + data->ext_press[2] ) /
         ( big_box->box_norms[0] * big_box->box_norms[1] * P_CONV ));

    /* Average pressure for the whole box */
    data->iso_bar.P =
        ( data->tot_press[0] + data->tot_press[1] + data->tot_press[2] ) / 3.0;
}
