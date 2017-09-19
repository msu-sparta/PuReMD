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

#ifndef __QEq_H_
#define __QEq_H_

#include "mytypes.h"


void QEq( reax_system* const, control_params* const, simulation_data* const,
          static_storage* const, const list* const,
          const output_controls* const );


//static inline HOST_DEVICE void swap( sparse_matrix_entry *array,
//        int index1, int index2 ) 
//{
//    sparse_matrix_entry temp = array[index1];
//    array[index1] = array[index2];
//    array[index2] = temp;
//}
//
//
//static inline HOST_DEVICE void quick_sort( sparse_matrix_entry *array,
//        int start, int end )
//{
//    int i = start;
//    int k = end; 
//
//    if (end - start >= 1)  
//    {  
//        int pivot = array[start].j;
//
//        while (k > i) 
//        {  
//            while ((array[i].j <= pivot) && (i <= end) && (k > i))
//            {
//                i++;
//            }
//            while ((array[k].j > pivot) && (k >= start) && (k >= i))
//            {
//                k--;
//            }
//            if (k > i)
//            {
//                swap( array, i, k );
//            }
//        }  
//        swap( array, start, k );
//        quick_sort( array, start, k - 1 );
//        quick_sort( array, k + 1, end );
//    }  
//}


#endif
