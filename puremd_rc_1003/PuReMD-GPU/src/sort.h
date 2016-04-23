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

#ifndef __SORT_H__
#define __SORT_H__

#include "mytypes.h"

HOST_DEVICE inline void h_swap(sparse_matrix_entry *array, int index1, int index2) 
{
	sparse_matrix_entry temp = array[index1];
	array[index1] = array[index2];
	array[index2] = temp;
}

HOST_DEVICE inline void h_quick_sort(sparse_matrix_entry *array, int start, int end)
{
	int i = start;
	int k = end; 

	if (end - start >= 1)  
	{
		int pivot = array[start].j;

		while (k > i)
		{
			while ((array[i].j <= pivot) && (i <= end) && (k > i)) i++;
			while ((array[k].j > pivot) && (k >= start) && (k >= i)) k--;
			if (k > i) h_swap(array, i, k);
		}
		h_swap(array, start, k);
		h_quick_sort(array, start, k - 1);
		h_quick_sort(array, k + 1, end);
	}
}

inline void d_swap(sparse_matrix_entry *array, int index1, int index2) 
{
	sparse_matrix_entry temp = array[index1];
	array[index1] = array[index2];
	array[index2] = temp;
}

inline void d_quick_sort(sparse_matrix_entry *array, int start, int end)
{
	int i = start;
	int k = end; 

	//fprintf (stderr, " start %d end %d \n", start, end);
	if (end - start >= 1)  
	{
		int pivot = array[start].j;

		while (k > i)
		{
			while (array[i].j <= pivot && i <= end && k > i) i++;
			while (array[k].j > pivot && k >= start && k >= i) k--;
			if (k > i) d_swap(array, i, k);
		}
		//fprintf (stderr, "Swapping %d %d \n", array[start].j, array[k].j);
		d_swap(array, start, k);
		d_quick_sort(array, start, k - 1);
		d_quick_sort(array, k + 1, end);
	}
}



#endif
