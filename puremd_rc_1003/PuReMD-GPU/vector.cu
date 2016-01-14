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

#include "vector.h"


int Vector_isZero( real* v, int k )
{
  for( --k; k>=0; --k )
    if( fabs( v[k] ) > ALMOST_ZERO )
      return 0;
  
  return 1;
}


void Vector_MakeZero( real *v, int k )
{
  for( --k; k>=0; --k )
    v[k] = 0;
}


void Vector_Copy( real* dest, real* v, int k )
{
  for( --k; k>=0; --k )
    dest[k] = v[k];
}


void Vector_Print( FILE *fout, char *vname, real *v, int k )
{
  int i;

  fprintf( fout, "%s:\n", vname );
  for( i = 0; i < k; ++i )
    fprintf( fout, "%24.15e\n", v[i] );
  fprintf( fout, "\n" );
}


real Norm( real* v1, int k )
{
  real ret = 0;
  
  for( --k; k>=0; --k )
    ret +=  SQR( v1[k] );

  return SQRT( ret );
}


void rvec_Sum( rvec ret, rvec v1 ,rvec v2 )
{
  ret[0] = v1[0] + v2[0];
  ret[1] = v1[1] + v2[1];
  ret[2] = v1[2] + v2[2];
}


real rvec_ScaledDot( real c1, rvec v1, real c2, rvec v2 )
{
  return (c1*c2) * (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]);
}


void rvec_Multiply( rvec r, rvec v1, rvec v2 )
{
  r[0] = v1[0] * v2[0];
  r[1] = v1[1] * v2[1];
  r[2] = v1[2] * v2[2];
}


void rvec_Divide( rvec r, rvec v1, rvec v2 )
{
  r[0] = v1[0] / v2[0];
  r[1] = v1[1] / v2[1];
  r[2] = v1[2] / v2[2];
}


void rvec_iDivide( rvec r, rvec v1, ivec v2 )
{
  r[0] = v1[0] / v2[0];
  r[1] = v1[1] / v2[1];
  r[2] = v1[2] / v2[2];
}


void rvec_Invert( rvec r, rvec v )
{
  r[0] = 1. / v[0];
  r[1] = 1. / v[1];
  r[2] = 1. / v[2];
}


void rvec_OuterProduct( rtensor r, rvec v1, rvec v2 )
{
  int i, j;

  for( i = 0; i < 3; ++i )
    for( j = 0; j < 3; ++j )
      r[i][j] = v1[i] * v2[j];
}



int rvec_isZero( rvec v )
{
  if( fabs(v[0]) > ALMOST_ZERO || 
      fabs(v[1]) > ALMOST_ZERO || 
      fabs(v[2]) > ALMOST_ZERO )
    return 0;
  return 1;
}


void rtensor_Multiply( rtensor ret, rtensor m1, rtensor m2 )
{
  int i, j, k;
  rtensor temp;

  // check if the result matrix is the same as one of m1, m2.
  // if so, we cannot modify the contents of m1 or m2, so 
  // we have to use a temp matrix.
  if( ret == m1 || ret == m2 )
    {
      for( i = 0; i < 3; ++i )
	for( j = 0; j < 3; ++j )
	  {
	    temp[i][j] = 0;	    
	    for( k = 0; k < 3; ++k )
	      temp[i][j] += m1[i][k] * m2[k][j];
	  }
      
      for( i = 0; i < 3; ++i )
	for( j = 0; j < 3; ++j )
	  ret[i][j] = temp[i][j];	
    }
  else
    {
      for( i = 0; i < 3; ++i )
	for( j = 0; j < 3; ++j )
	  {
	    ret[i][j] = 0;	    
	    for( k = 0; k < 3; ++k )
	      ret[i][j] += m1[i][k] * m2[k][j];
	  }
    }
}


void rtensor_MatVec( rvec ret, rtensor m, rvec v )
{
  int i;
  rvec temp;

  // if ret is the same vector as v, we cannot modify the 
  // contents of v until all computation is finished.
  if( ret == v )
    {
      for( i = 0; i < 3; ++i )
	temp[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];

      for( i = 0; i < 3; ++i )
	ret[i] = temp[i];
    }
  else
    {
      for( i = 0; i < 3; ++i )
	ret[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2];
    }
}


void rtensor_Scale( rtensor ret, real c, rtensor m )
{
  int i, j;

  for( i = 0; i < 3; ++i )
    for( j = 0; j < 3; ++j )
      ret[i][j] = c * m[i][j];
}


void rtensor_Add( rtensor ret, rtensor t )
{
  int i, j;

  for( i = 0; i < 3; ++i )
    for( j = 0; j < 3; ++j )
      ret[i][j] += t[i][j];
}


void rtensor_ScaledAdd( rtensor ret, real c, rtensor t )
{
  int i, j;

  for( i = 0; i < 3; ++i )
    for( j = 0; j < 3; ++j )
      ret[i][j] += c * t[i][j];
}


void rtensor_Sum( rtensor ret, rtensor t1, rtensor t2 )
{
  int i, j;

  for( i = 0; i < 3; ++i )
    for( j = 0; j < 3; ++j )
      ret[i][j] = t1[i][j] + t2[i][j];
}


void rtensor_ScaledSum( rtensor ret, real c1, rtensor t1, 
			       real c2, rtensor t2 )
{
  int i, j;

  for( i = 0; i < 3; ++i )
    for( j = 0; j < 3; ++j )
      ret[i][j] = c1 * t1[i][j] + c2 * t2[i][j];
}


void rtensor_Copy( rtensor ret, rtensor t )
{
  int i, j;

  for( i = 0; i < 3; ++i )
    for( j = 0; j < 3; ++j )
      ret[i][j] = t[i][j];
}


void rtensor_Identity( rtensor t )
{
  t[0][0] = t[1][1] = t[2][2] = 1;
  t[0][1] = t[0][2] = t[1][0] = t[1][2] = t[2][0] = t[2][1] = ZERO;
}


void rtensor_MakeZero( rtensor t )
{
  t[0][0] = t[0][1] = t[0][2] = ZERO;
  t[1][0] = t[1][1] = t[1][2] = ZERO;
  t[2][0] = t[2][1] = t[2][2] = ZERO;
}


void rtensor_Transpose( rtensor ret, rtensor t )
{
  ret[0][0] = t[0][0], ret[1][1] = t[1][1], ret[2][2] = t[2][2];
  ret[0][1] = t[1][0], ret[0][2] = t[2][0];
  ret[1][0] = t[0][1], ret[1][2] = t[2][1];
  ret[2][0] = t[0][2], ret[2][1] = t[1][2];
}


real rtensor_Det( rtensor t )
{
  return ( t[0][0] * (t[1][1] * t[2][2] - t[1][2] * t[2][1] ) +
           t[0][1] * (t[1][2] * t[2][0] - t[1][0] * t[2][2] ) +
           t[0][2] * (t[1][0] * t[2][1] - t[1][1] * t[2][0] ) );
}


real rtensor_Trace( rtensor t )
{
  return (t[0][0] + t[1][1] + t[2][2]);
}


void Print_rTensor(FILE* fp, rtensor t)
{
  int i, j;

  for (i=0; i < 3; i++)
    {
      fprintf(fp,"[");
      for (j=0; j < 3; j++)
	fprintf(fp,"%8.3f,\t",t[i][j]);
      fprintf(fp,"]\n");
    }
}


void ivec_MakeZero( ivec v )
{
  v[0] = v[1] = v[2] = 0;
}


void ivec_rScale( ivec dest, real C, rvec src )
{
  dest[0] = (int)(C * src[0]);
  dest[1] = (int)(C * src[1]);
  dest[2] = (int)(C * src[2]);
}


int ivec_isZero( ivec v )
{
  if( v[0]==0 && v[1]==0 && v[2]==0 )
    return 1;
  return 0;
}


int ivec_isEqual( ivec v1, ivec v2 )
{
  if( v1[0]==v2[0] && v1[1]==v2[1] && v1[2]==v2[2] )
    return 1;

  return 0;
}


