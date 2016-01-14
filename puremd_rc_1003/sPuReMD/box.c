/*----------------------------------------------------------------------
  SerialReax - Reax Force Field Simulator
      
  Copyright (2010) Purdue University
  Hasan Metin Aktulga, haktulga@cs.purdue.edu
  Joseph Fogarty, jcfogart@mail.usf.edu
  Sagar Pandit, pandit@usf.edu
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

#include "box.h"
#include "vector.h"


void Init_Box_From_CRYST(real a, real b, real c, 
			 real alpha, real beta, real gamma, 
			 simulation_box* box )
{
  double c_alpha, c_beta, c_gamma, s_gamma, zi;

  c_alpha = cos(DEG2RAD(alpha));
  c_beta  = cos(DEG2RAD(beta));
  c_gamma = cos(DEG2RAD(gamma));
  s_gamma = sin(DEG2RAD(gamma));

  zi = (c_alpha - c_beta * c_gamma)/s_gamma; 

  box->box[0][0] = a; 
  box->box[0][1] = 0.0; 
  box->box[0][2] = 0.0;
  
  box->box[1][0] = b * c_gamma; 
  box->box[1][1] = b * s_gamma; 
  box->box[1][2] = 0.0; 

  box->box[2][0] = c * c_beta;
  box->box[2][1] = c * zi;
  box->box[2][2] = c * SQRT(1.0 - SQR(c_beta) - SQR(zi));

  Make_Consistent( box );

#if defined(DEBUG_FOCUS)
  fprintf( stderr, "box is %8.2f x %8.2f x %8.2f\n", 
	   box->box[0][0], box->box[1][1], box->box[2][2] );
#endif
}


void Update_Box( rtensor box_tensor, simulation_box* box )
{
  int i, j;

  for (i=0; i < 3; i++)
    for (j=0; j < 3; j++)
      box->box[i][j] = box_tensor[i][j];

  Make_Consistent( box );
}


void Update_Box_Isotropic( simulation_box *box, real mu )
{
  /*box->box[0][0] = 
    POW( V_new / ( box->side_prop[1] * box->side_prop[2] ), 1.0/3.0 );
  box->box[1][1] = box->box[0][0] * box->side_prop[1];
  box->box[2][2] = box->box[0][0] * box->side_prop[2]; 
  */
  rtensor_Copy( box->old_box, box->box );
  box->box[0][0] *= mu;
  box->box[1][1] *= mu;
  box->box[2][2] *= mu;
  
  box->volume = box->box[0][0]*box->box[1][1]*box->box[2][2];
  Make_Consistent(box/*, periodic*/);
}


void Update_Box_SemiIsotropic( simulation_box *box, rvec mu )
{
  /*box->box[0][0] = 
    POW( V_new / ( box->side_prop[1] * box->side_prop[2] ), 1.0/3.0 );
  box->box[1][1] = box->box[0][0] * box->side_prop[1];
  box->box[2][2] = box->box[0][0] * box->side_prop[2]; */
  rtensor_Copy( box->old_box, box->box );
  box->box[0][0] *= mu[0];
  box->box[1][1] *= mu[1];
  box->box[2][2] *= mu[2];
  
  box->volume = box->box[0][0]*box->box[1][1]*box->box[2][2];
  Make_Consistent(box);
}


void Make_Consistent(simulation_box* box)
{
  real one_vol;

  box->volume = 
    box->box[0][0] * (box->box[1][1]*box->box[2][2] - 
		       box->box[2][1]*box->box[2][1]) +
    box->box[0][1] * (box->box[2][0]*box->box[1][2] -
		       box->box[1][0]*box->box[2][2]) +
    box->box[0][2] * (box->box[1][0]*box->box[2][1] -
		       box->box[2][0]*box->box[1][1]);

  one_vol = 1.0/box->volume;

  box->box_inv[0][0] = (box->box[1][1]*box->box[2][2] -
			 box->box[1][2]*box->box[2][1]) * one_vol;
  box->box_inv[0][1] = (box->box[0][2]*box->box[2][1] -
			 box->box[0][1]*box->box[2][2]) * one_vol;
  box->box_inv[0][2] = (box->box[0][1]*box->box[1][2] -
			 box->box[0][2]*box->box[1][1]) * one_vol;

  box->box_inv[1][0] = (box->box[1][2]*box->box[2][0] -
			 box->box[1][0]*box->box[2][2]) * one_vol;
  box->box_inv[1][1] = (box->box[0][0]*box->box[2][2] -
			 box->box[0][2]*box->box[2][0]) * one_vol;
  box->box_inv[1][2] = (box->box[0][2]*box->box[1][0] -
			 box->box[0][0]*box->box[1][2]) * one_vol;

  box->box_inv[2][0] = (box->box[1][0]*box->box[2][1] -
			 box->box[1][1]*box->box[2][0]) * one_vol;
  box->box_inv[2][1] = (box->box[0][1]*box->box[2][0] -
			 box->box[0][0]*box->box[2][1]) * one_vol;
  box->box_inv[2][2] = (box->box[0][0]*box->box[1][1] -
			 box->box[0][1]*box->box[1][0]) * one_vol;

  box->box_norms[0] = SQRT( SQR(box->box[0][0]) +
			     SQR(box->box[0][1]) +
			     SQR(box->box[0][2]) );
  box->box_norms[1] = SQRT( SQR(box->box[1][0]) +
			     SQR(box->box[1][1]) +
			     SQR(box->box[1][2]) );
  box->box_norms[2] = SQRT( SQR(box->box[2][0]) +
			     SQR(box->box[2][1]) +
			     SQR(box->box[2][2]) );

  box->trans[0][0] = box->box[0][0]/box->box_norms[0]; 
  box->trans[0][1] = box->box[1][0]/box->box_norms[0];
  box->trans[0][2] = box->box[2][0]/box->box_norms[0];

  box->trans[1][0] = box->box[0][1]/box->box_norms[1]; 
  box->trans[1][1] = box->box[1][1]/box->box_norms[1];
  box->trans[1][2] = box->box[2][1]/box->box_norms[1];

  box->trans[2][0] = box->box[0][2]/box->box_norms[2]; 
  box->trans[2][1] = box->box[1][2]/box->box_norms[2];
  box->trans[2][2] = box->box[2][2]/box->box_norms[2];

  one_vol = box->box_norms[0]*box->box_norms[1]*box->box_norms[2]*one_vol;

  box->trans_inv[0][0] = (box->trans[1][1]*box->trans[2][2] -
			   box->trans[1][2]*box->trans[2][1]) * one_vol;
  box->trans_inv[0][1] = (box->trans[0][2]*box->trans[2][1] -
			   box->trans[0][1]*box->trans[2][2]) * one_vol;
  box->trans_inv[0][2] = (box->trans[0][1]*box->trans[1][2] -
			   box->trans[0][2]*box->trans[1][1]) * one_vol;
  
  box->trans_inv[1][0] = (box->trans[1][2]*box->trans[2][0] -
			   box->trans[1][0]*box->trans[2][2]) * one_vol;
  box->trans_inv[1][1] = (box->trans[0][0]*box->trans[2][2] -
			   box->trans[0][2]*box->trans[2][0]) * one_vol;
  box->trans_inv[1][2] = (box->trans[0][2]*box->trans[1][0] -
			   box->trans[0][0]*box->trans[1][2]) * one_vol;

  box->trans_inv[2][0] = (box->trans[1][0]*box->trans[2][1] -
			   box->trans[1][1]*box->trans[2][0]) * one_vol;
  box->trans_inv[2][1] = (box->trans[0][1]*box->trans[2][0] -
			   box->trans[0][0]*box->trans[2][1]) * one_vol;
  box->trans_inv[2][2] = (box->trans[0][0]*box->trans[1][1] -
			   box->trans[0][1]*box->trans[1][0]) * one_vol;

//   for (i=0; i < 3; i++)
//     {
//       for (j=0; j < 3; j++)
// 	fprintf(stderr,"%lf\t",box->trans[i][j]);
//       fprintf(stderr,"\n");
//     }
//   fprintf(stderr,"\n");
//   for (i=0; i < 3; i++)
//     {
//       for (j=0; j < 3; j++)
// 	fprintf(stderr,"%lf\t",box->trans_inv[i][j]);
//       fprintf(stderr,"\n");
//     }


  box->g[0][0] = box->box[0][0] * box->box[0][0] +
                  box->box[0][1] * box->box[0][1] +
                  box->box[0][2] * box->box[0][2];
  box->g[1][0] = 
  box->g[0][1] = box->box[0][0] * box->box[1][0] +
                  box->box[0][1] * box->box[1][1] +
                  box->box[0][2] * box->box[1][2];
  box->g[2][0] =
  box->g[0][2] = box->box[0][0] * box->box[2][0] +
                  box->box[0][1] * box->box[2][1] +
                  box->box[0][2] * box->box[2][2];

  box->g[1][1] = box->box[1][0] * box->box[1][0] +
                  box->box[1][1] * box->box[1][1] +
                  box->box[1][2] * box->box[1][2];
  box->g[1][2] =
  box->g[2][1] = box->box[1][0] * box->box[2][0] +
                  box->box[1][1] * box->box[2][1] +
                  box->box[1][2] * box->box[2][2];

  box->g[2][2] = box->box[2][0] * box->box[2][0] +
                  box->box[2][1] * box->box[2][1] +
                  box->box[2][2] * box->box[2][2];

  // These proportions are only used for isotropic_NPT!
  box->side_prop[0] = box->box[0][0] / box->box[0][0];
  box->side_prop[1] = box->box[1][1] / box->box[0][0];
  box->side_prop[2] = box->box[2][2] / box->box[0][0];
}


void Transform( rvec x1, simulation_box *box, char flag, rvec x2 )
{
  int i, j;
  real tmp;

  //  printf(">x1: (%lf, %lf, %lf)\n",x1[0],x1[1],x1[2]);
	  
  if (flag > 0) {
    for (i=0; i < 3; i++) {
      tmp = 0.0;
      for (j=0; j < 3; j++)
	tmp += box->trans[i][j]*x1[j]; 
      x2[i] = tmp;
    }
  }
  else {
    for (i=0; i < 3; i++) {
      tmp = 0.0;
      for (j=0; j < 3; j++)
	tmp += box->trans_inv[i][j]*x1[j]; 
      x2[i] = tmp;
    }
  }
  //  printf(">x2: (%lf, %lf, %lf)\n", x2[0], x2[1], x2[2]);  
}


void Transform_to_UnitBox( rvec x1, simulation_box *box, char flag, rvec x2 )
{
  Transform( x1, box, flag, x2 );
  
  x2[0] /= box->box_norms[0];
  x2[1] /= box->box_norms[1];
  x2[2] /= box->box_norms[2];
}


void Inc_on_T3( rvec x, rvec dx, simulation_box *box )
{
  int i;
  real tmp;

  for (i=0; i < 3; i++) { 
    tmp = x[i] + dx[i];      
    if( tmp <= -box->box_norms[i] || tmp >= box->box_norms[i] )
      tmp = fmod( tmp, box->box_norms[i] );

    if( tmp < 0 ) tmp += box->box_norms[i];
    x[i] = tmp;
  }
}


real Sq_Distance_on_T3(rvec x1, rvec x2, simulation_box* box, rvec r)
{
  real norm=0.0;
  real d, tmp;
  int i;
  
  for (i=0; i < 3; i++) {
    d = x2[i] - x1[i];
    tmp = SQR(d);

    if( tmp >= SQR( box->box_norms[i] / 2.0 ) ) {
      if (x2[i] > x1[i])
	d -= box->box_norms[i];
      else
	d += box->box_norms[i];
      
      r[i] = d;
      norm += SQR(d);
    }
    else {
      r[i] = d;
      norm += tmp;
    } 
  }
  
  return norm;
}


void Distance_on_T3_Gen( rvec x1, rvec x2, simulation_box* box, rvec r )
{
  rvec xa, xb, ra;

  Transform( x1, box, -1, xa );
  Transform( x2, box, -1, xb );

  //printf(">xa: (%lf, %lf, %lf)\n",xa[0],xa[1],xa[2]);
  //printf(">xb: (%lf, %lf, %lf)\n",xb[0],xb[1],xb[2]);
  
  Sq_Distance_on_T3( xa, xb, box, ra );

  Transform( ra, box, 1, r );
}


void Inc_on_T3_Gen( rvec x, rvec dx, simulation_box* box )
{
  rvec xa, dxa;

  Transform( x, box, -1, xa );
  Transform( dx, box, -1, dxa );

  //printf(">xa: (%lf, %lf, %lf)\n",xa[0],xa[1],xa[2]);
  //printf(">dxa: (%lf, %lf, %lf)\n",dxa[0],dxa[1],dxa[2]);
  
  Inc_on_T3( xa, dxa, box );

  //printf(">new_xa: (%lf, %lf, %lf)\n",xa[0],xa[1],xa[2]);

  Transform( xa, box, 1, x );
}


real Metric_Product( rvec x1, rvec x2, simulation_box* box )
{
  int i, j;
  real dist=0.0, tmp;

  for( i = 0; i < 3; i++ )
    {
      tmp = 0.0;
      for( j = 0; j < 3; j++ )
	tmp += box->g[i][j] * x2[j];
      dist += x1[i] * tmp;
    }

  return dist;
}



int Are_Far_Neighbors( rvec x1, rvec x2, simulation_box *box, 
		       real cutoff, far_neighbor_data *data )
{
  real norm_sqr, d, tmp;
  int i;
  
  norm_sqr = 0;
  
  for( i = 0; i < 3; i++ ) {
    d = x2[i] - x1[i];
    tmp = SQR(d);
        
    if( tmp >= SQR( box->box_norms[i] / 2.0 ) ) {	
      if( x2[i] > x1[i] ) {
	d -= box->box_norms[i];
	data->rel_box[i] = -1;
      }
      else {
	d += box->box_norms[i];
	data->rel_box[i] = +1;
      }
      
      data->dvec[i] = d;
      norm_sqr += SQR(d);
    }
    else {
      data->dvec[i] = d;
      norm_sqr += tmp;
      data->rel_box[i] = 0;
    } 
  }
  
  if( norm_sqr <= SQR(cutoff) ){
    data->d = sqrt(norm_sqr);
    return 1;
  }
  
  return 0;
}



/* Determines if the distance between x1 and x2 is < vlist_cut. 
   If so, this neighborhood is added to the list of far neighbors.
   Periodic boundary conditions do not apply. */
void Get_NonPeriodic_Far_Neighbors( rvec x1, rvec x2, simulation_box *box, 
				    control_params *control, 
				    far_neighbor_data *new_nbrs, int *count )
{
  real norm_sqr;
	
  rvec_ScaledSum( new_nbrs[0].dvec, 1.0, x2, -1.0, x1 );

  norm_sqr = rvec_Norm_Sqr( new_nbrs[0].dvec );

  if( norm_sqr <= SQR( control->vlist_cut ) ) {
    *count = 1;
    new_nbrs[0].d = SQRT( norm_sqr );
    
    ivec_MakeZero( new_nbrs[0].rel_box );
    // rvec_MakeZero( new_nbrs[0].ext_factor );
  }
  else *count = 0;
}


/* Finds periodic neighbors in a 'big_box'. Here 'big_box' means:
   the current simulation box has all dimensions > 2 *vlist_cut.
   If the periodic distance between x1 and x2 is than vlist_cut, this 
   neighborhood is added to the list of far neighbors. */
void Get_Periodic_Far_Neighbors_Big_Box( rvec x1, rvec x2, simulation_box *box, 
					 control_params *control, 
					 far_neighbor_data *periodic_nbrs, 
					 int *count )
{
  real norm_sqr, d, tmp;
  int i;
  
  norm_sqr = 0;

  for( i = 0; i < 3; i++ ) {
    d = x2[i] - x1[i];
    tmp = SQR(d);
    // fprintf(out,"Inside Sq_Distance_on_T3, %d, %lf, %lf\n",
    // i,tmp,SQR(box->box_norms[i]/2.0));
    
    if( tmp >= SQR( box->box_norms[i] / 2.0 ) ) {	
      if( x2[i] > x1[i] ) {
	d -= box->box_norms[i];
	periodic_nbrs[0].rel_box[i] = -1;
	// periodic_nbrs[0].ext_factor[i] = +1;
      }
      else {
	d += box->box_norms[i];
	periodic_nbrs[0].rel_box[i] = +1;
	// periodic_nbrs[0].ext_factor[i] = -1;
      }
      
      periodic_nbrs[0].dvec[i] = d;
      norm_sqr += SQR(d);
    }
    else {
      periodic_nbrs[0].dvec[i] = d;
      norm_sqr += tmp;
      periodic_nbrs[0].rel_box[i]   = 0;
      // periodic_nbrs[0].ext_factor[i] = 0;
    } 
  }
  
  if( norm_sqr <= SQR( control->vlist_cut ) ) {
    *count = 1;
    periodic_nbrs[0].d = SQRT( norm_sqr );
  }
  else *count = 0;
}


/* Finds all periodic far neighborhoods between x1 and x2 
   ((dist(x1, x2') < vlist_cut, periodic images of x2 are also considered).
   Here the box is 'small' meaning that at least one dimension is < 2*vlist_cut.
   IMPORTANT: This part might need some improvement. In NPT, the simulation box 
   might get too small (such as <5 A!). In this case we have to consider the 
   periodic images of x2 that are two boxs away!!!
*/
void Get_Periodic_Far_Neighbors_Small_Box( rvec x1, rvec x2, simulation_box *box,
					   control_params *control, 
					   far_neighbor_data *periodic_nbrs, 
					   int *count )
{
  int i, j, k;
  int imax, jmax, kmax;
  real sqr_norm, d_i, d_j, d_k;
  
  *count = 0;
  /* determine the max stretch of imaginary boxs in each direction
     to handle periodic boundary conditions correctly. */
  imax = (int)(control->vlist_cut / box->box_norms[0] + 1);
  jmax = (int)(control->vlist_cut / box->box_norms[1] + 1);
  kmax = (int)(control->vlist_cut / box->box_norms[2] + 1);
  /*if( imax > 1 || jmax > 1 || kmax > 1 )
    fprintf( stderr, "box %8.3f x %8.3f x %8.3f --> %2d %2d %2d\n",
    box->box_norms[0], box->box_norms[1], box->box_norms[2],
    imax, jmax, kmax ); */


  for( i = -imax; i <= imax; ++i )
    if(fabs(d_i=((x2[0]+i*box->box_norms[0])-x1[0]))<=control->vlist_cut) {
      for( j = -jmax; j <= jmax; ++j )
	if(fabs(d_j=((x2[1]+j*box->box_norms[1])-x1[1]))<=control->vlist_cut) {
	  for( k = -kmax; k <= kmax; ++k )
	    if(fabs(d_k=((x2[2]+k*box->box_norms[2])-x1[2]))<=control->vlist_cut) {
	      sqr_norm = SQR(d_i) + SQR(d_j) + SQR(d_k);
	      if( sqr_norm <= SQR(control->vlist_cut) ) {
		periodic_nbrs[ *count ].d = SQRT( sqr_norm );
		
		periodic_nbrs[ *count ].dvec[0] = d_i;
		periodic_nbrs[ *count ].dvec[1] = d_j;
		periodic_nbrs[ *count ].dvec[2] = d_k;
		
		periodic_nbrs[ *count ].rel_box[0] = i;
		periodic_nbrs[ *count ].rel_box[1] = j;
		periodic_nbrs[ *count ].rel_box[2] = k;
		
		/* if( i || j || k ) {
		   fprintf(stderr, "x1: %.2f %.2f %.2f\n", x1[0], x1[1], x1[2]);
		   fprintf(stderr, "x2: %.2f %.2f %.2f\n", x2[0], x2[1], x2[2]);
		   fprintf( stderr, "d : %8.2f%8.2f%8.2f\n\n", d_i, d_j, d_k );
		   } */
		
		/* if(i) periodic_nbrs[*count].ext_factor[0] = (real)i/-abs(i);
		   else  periodic_nbrs[*count].ext_factor[0] = 0;
		   
		   if(j) periodic_nbrs[*count].ext_factor[1] = (real)j/-abs(j);
		   else  periodic_nbrs[*count].ext_factor[1] = 0;

		   if(k) periodic_nbrs[*count].ext_factor[2] = (real)k/-abs(k);
		   else  periodic_nbrs[*count].ext_factor[2] = 0; */
			 
		
		/* if( i == 0 && j == 0 && k == 0 )
		 *  periodic_nbrs[ *count ].imaginary = 0;
		 *  else periodic_nbrs[ *count ].imaginary = 1;
		 */
		++(*count);
	      }
	    }
	}
    }
}

 
/* Returns the mapping for the neighbor box pointed by (ix,iy,iz) */
/*int Get_Nbr_Box( simulation_box *box, int ix, int iy, int iz )
{
  return (9 * ix + 3 * iy + iz + 13);  
  // 13 is to handle negative indexes properly
}*/


/* Returns total pressure vector for the neighbor box pointed by (ix,iy,iz) */
/*rvec Get_Nbr_Box_Press( simulation_box *box, int ix, int iy, int iz )
{
  int map;

  map = 9 * ix + 3 * iy + iz + 13;  
  // 13 is to adjust -1,-1,-1 correspond to index 0

  return box->nbr_box_press[map];
}*/


/* Increments total pressure vector for the nbr box pointed by (ix,iy,iz) */
/*void Inc_Nbr_Box_Press( simulation_box *box, int ix, int iy, int iz, rvec v )
  {
  int map;
  
  map = 9 * ix + 3 * iy + iz + 13;  
  // 13 is to adjust -1,-1,-1 correspond to index 0
  
  rvec_Add( box->nbr_box_press[map], v );
}*/


/* Increments the total pressure vector for the neighbor box mapped to 'map' */
/*void Inc_Nbr_Box_Press( simulation_box *box, int map, rvec v )
{
  rvec_Add( box->nbr_box_press[map], v );
}*/


void Print_Box_Information( simulation_box* box, FILE *out )
{
  int i, j;

  fprintf( out, "box: {" );
  for( i = 0; i < 3; ++i )
    {
      fprintf( out, "{" );
      for( j = 0; j < 3; ++j )
	fprintf( out, "%8.3f ", box->box[i][j] );
      fprintf( out, "}" );
    }
  fprintf( out, "}\n" );

  fprintf( out, "V: %8.3f\tdims: {%8.3f, %8.3f, %8.3f}\n", 
	   box->volume, 
	   box->box_norms[0], box->box_norms[1], box->box_norms[2] );

  fprintf( out, "box_trans: {" );
  for( i = 0; i < 3; ++i )
    {
      fprintf( out, "{" );
      for( j = 0; j < 3; ++j )
	fprintf( out, "%8.3f ", box->trans[i][j] );
      fprintf( out, "}" );
    }
  fprintf( out, "}\n" );

  fprintf( out, "box_trinv: {" );
  for( i = 0; i < 3; ++i )
    {
      fprintf( out, "{" );
      for( j = 0; j < 3; ++j )
	fprintf( out, "%8.3f ", box->trans_inv[i][j] );
      fprintf( out, "}" );
    }
  fprintf( out, "}\n" );
}
