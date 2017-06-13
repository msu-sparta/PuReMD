#include "cuda_validation.h"

#include "cuda_utils.h"
#include "list.h"
#include "reax_types.h"

#include "index_utils.h"
#include "vector.h"


bool check_zero (real p1, real p2)
{
    if (abs (p1 - p2) >= GPU_TOLERANCE)
        return true;
    else 
        return false;
}


bool check_zero (rvec p1, rvec p2)
{

    if (((abs (p1[0] - p2[0])) >= GPU_TOLERANCE) ||
            ((abs (p1[1] - p2[1])) >= GPU_TOLERANCE) ||
            ((abs (p1[2] - p2[2])) >= GPU_TOLERANCE ))
        return true;
    else return false;
}


bool check_zero_rvec2 (rvec2 p1, rvec2 p2)
{

    if (((abs (p1[0] - p2[0])) >= GPU_TOLERANCE) ||
            ((abs (p1[1] - p2[1])) >= GPU_TOLERANCE ))
        return true;
    else return false;
}


bool check_same (ivec p1, ivec p2)
{
    if ( (p1[0] == p2[0]) || (p1[1] == p2[1]) || (p1[2] == p2[2]) )
        return true;
    else 
        return false;
}


void print_bond_data (bond_order_data *s)
{
    /*   
         fprintf (stderr, "Bond_Order_Data BO (%f ) BO_s (%f ) BO_pi (%f ) BO_pi2 (%f ) ", 
         s->BO, 
         s->BO_s, 
         s->BO_pi,
         s->BO_pi2 );
     */
    fprintf (stderr, " Cdbo (%e) ", s->Cdbo );
    fprintf (stderr, " Cdbopi (%e) ", s->Cdbopi );
    fprintf (stderr, " Cdbopi2 (%e) ", s->Cdbopi2 );
}


int validate_neighbors (reax_system *system, reax_list **lists)
{
    reax_list *far_nbrs = *lists + FAR_NBRS;
    reax_list *d_nbrs = *dev_lists + FAR_NBRS;
    far_neighbor_data gpu, cpu;
    int index, count, jicount;
    int hostcount, dijcount, djicount;
    int i;

    int *end = (int *)malloc (sizeof (int) * system->N);
    int *start = (int *) malloc (sizeof (int) * system->N );

    copy_host_device (start, d_nbrs->index, 
            sizeof (int) * system->N, cudaMemcpyDeviceToHost, "far_nbrs:index");
    copy_host_device (end, d_nbrs->end_index, 
            sizeof (int) * system->N, cudaMemcpyDeviceToHost, "far_nbrs:end_index");

    far_neighbor_data *data = (far_neighbor_data *) 
        malloc (sizeof (far_neighbor_data)* d_nbrs->num_intrs);
    copy_host_device (data, d_nbrs->select.far_nbr_list, 
            sizeof (far_neighbor_data) * d_nbrs->num_intrs, cudaMemcpyDeviceToHost, "far_nbr_list");

    hostcount = dijcount = djicount = 0;

    for (i= 0; i < system->N-1; i++){
        if (end [i] > start [i+1])
        {
            fprintf (stderr, " Far Neighbors index over write  @ index %d (%d, %d) and (%d %d)\n", 
                    i, start[i], end[i], start[i+1], end[i+1]);
            return FAILURE;
        }
        hostcount += end[i] - start[i];
    }
    hostcount += end[i] - start[i];
    fprintf (stderr, "Total Neighbors count: %d \n", hostcount);
    hostcount = 0;

    return 0;

    /*
       for (int i = 0; i < 2; i++) {
       for (int j = start[i]; j < end[i]; j++){
       gpu = data[j];
       fprintf (stderr, " atom %d neighbor %d  (%f, %d, %d, %d - %f %f %f) - %d \n", i, data[j].nbr,
       data[j].d,
       data[j].rel_box[0],
       data[j].rel_box[1],
       data[j].rel_box[2],
       data[j].dvec[0],
       data[j].dvec[1],
       data[j].dvec[2], 
       j
       );
       }
       }

       return SUCCESS;
     */

    for (int i = 0; i < system->N; i++){
        index = Start_Index (i, far_nbrs);

        for (int j = start[i]; j < end[i]; j++){


            if (i > data[j].nbr) {

                int src = data[j].nbr;
                int dest = i;
                int x;


                for (x = start[src]; x < end[src]; x++) {
                    if (data[x].nbr != dest) continue;

                    gpu = data[x];
                    cpu = data[j];

                    if (  (gpu.d != cpu.d) ||
                            (cpu.dvec[0] != gpu.dvec[0]) || (cpu.dvec[1] != gpu.dvec[1]) || (cpu.dvec[2] != gpu.dvec[2]) ||
                            (cpu.rel_box[0] != gpu.rel_box[0]) || (cpu.rel_box[1] != gpu.rel_box[1]) || (cpu.rel_box[2] != gpu.rel_box[2])) {
                        fprintf (stderr, " atom %d neighbor %d  (%f, %d, %d, %d - %f %f %f) - %d \n", i, data[j].nbr,
                                data[j].d,
                                data[j].rel_box[0],
                                data[j].rel_box[1],
                                data[j].rel_box[2],
                                data[j].dvec[0],
                                data[j].dvec[1],
                                data[j].dvec[2], 
                                j
                            );
                        fprintf (stderr, " atom %d neighbor %d  (%f, %d, %d, %d - %f %f %f) - %d \n", data[j].nbr, data[x].nbr,
                                data[x].d,
                                data[x].rel_box[0],
                                data[x].rel_box[1],
                                data[x].rel_box[2],
                                data[x].dvec[0],
                                data[x].dvec[1],
                                data[x].dvec[2], 
                                x
                            );
                        jicount++;

                        fprintf (stderr, " Far Neighbors DOES NOT match between Deivce and Host \n");
                        exit (-1);
                    }
                    djicount ++;
                    break;
                }

                if (x >= end[src]) {
                    fprintf (stderr, "could not find the neighbor duplicate data for ij (%d %d)\n", i, src );
                    exit (-1);
                }
                continue;
            }

            gpu = data[j];
            cpu = far_nbrs->select.far_nbr_list[index];
            if (  check_zero (gpu.d, cpu.d) ||
                    (gpu.nbr != cpu.nbr) ||
                    check_zero (cpu.dvec, gpu.dvec) ||
                    !check_same (cpu.rel_box, gpu.rel_box)) {

                fprintf (stderr, "GPU:atom --> %d (s: %d , e: %d, i: %d )\n", i, start[i], end[i], j );
                fprintf (stderr, "CPU:atom --> %d (s: %d , e: %d, i: %d )\n", i, Start_Index(i, far_nbrs), End_Index (i, far_nbrs), index);
                fprintf (stdout, "Far neighbors does not match atom: %d \n", i );
                fprintf (stdout, "neighbor %d ,  %d \n",  cpu.nbr, gpu.nbr);
                fprintf (stdout, "d %f ,  %f \n", cpu.d, data[j].d);
                fprintf (stdout, "dvec (%f %f %f) (%f %f %f) \n",
                        cpu.dvec[0], cpu.dvec[1], cpu.dvec[2],
                        gpu.dvec[0], gpu.dvec[1], gpu.dvec[2] );

                fprintf (stdout, "rel_box (%d %d %d) (%d %d %d) \n",
                        cpu.rel_box[0], cpu.rel_box[1], cpu.rel_box[2],
                        gpu.rel_box[0], gpu.rel_box[1], gpu.rel_box[2] );

                fprintf (stderr, " Far Neighbors DOES NOT match between Deivce and Host  **** \n");
                return FAILURE;
                count ++;
            }
            index ++;
            hostcount ++;
            dijcount ++;
        }

        if (index != End_Index (i, far_nbrs))
        {
            fprintf (stderr, "End index does not match for atom --> %d end index (%d) Cpu (%d, %d ) gpu (%d, %d)\n", 
                    i, index, Start_Index (i, far_nbrs), End_Index(i, far_nbrs), start[i], end[i]);
            return FAILURE;
        }
    }

    fprintf (stderr, "FAR Neighbors match between device and host host:%d, device:%d dji: %d \n", 
            hostcount, dijcount, djicount);
    free (start);
    free (end);
    free (data);
    return SUCCESS;
}


int validate_sym_dbond_indices (reax_system *system, storage *workspace, reax_list **lists)
{
    int start, end, index, count, miscount;
    int hostcount, devicecount, h, d;
    int *d_start, *d_end;
    bond_data *d_bond_data;
    reax_list *d_bonds = *dev_lists + BONDS;
    reax_list *bonds = *lists + BONDS;

    d_end = (int *)malloc (sizeof (int) * system->N);
    d_start = (int *) malloc (sizeof (int) * system->N );
    d_bond_data = (bond_data *) malloc (sizeof (bond_data) * d_bonds->num_intrs);
    //fprintf (stderr, "Num bonds copied from device to host is --> %d \n", system->num_bonds );

    copy_host_device (d_start, d_bonds->index, sizeof (int) * system->N, cudaMemcpyDeviceToHost, "index");
    copy_host_device (d_end, d_bonds->end_index, sizeof (int) * system->N, cudaMemcpyDeviceToHost, "index");
    copy_host_device (d_bond_data, d_bonds->select.bond_list, sizeof (bond_data) * d_bonds->num_intrs, cudaMemcpyDeviceToHost, "bond_data");

    count = 0; 
    miscount = 0; 
    hostcount = 0;
    devicecount = 0;

    for (int i = 0; i < system->N; i++) {
        h= End_Index (i, bonds) - Start_Index (i, bonds);
        d= d_end[i] - d_start[i];
        //if (h != d) 
        //    fprintf (stderr, "Count does not match atom:%d, host:%d, device:%d \n", 
        //                    i, h, d);
        hostcount += h;
        devicecount += d;
    }
    fprintf (stderr, "Bonds count: host: %d device: %d \n", hostcount, devicecount);

    for (int i = 0; i < system->N; i++) {

        for (int j = d_start[i]; j < d_end[i]; j++) {
            bond_data *src, *tgt;
            src = &d_bond_data[j];

            tgt = &d_bond_data[ src->sym_index ];   

            if ((src->dbond_index == tgt->dbond_index) )
                count ++;
            else
                miscount ++;
        }    
    }
    fprintf (stderr, "Sym and dbond indexes done count(device) --> %d  (%d)\n", count, miscount);

    count = 0; 
    miscount = 0; 
    for (int i = 0; i < system->N; i++) {

        for (int j = Start_Index (i, bonds); j < End_Index(i, bonds); j++) {
            bond_data *src, *tgt;
            src = &bonds->select.bond_list [j]; 

            tgt = &bonds->select.bond_list [ src->sym_index ]; 

            if ((src->dbond_index == tgt->dbond_index) )
                count ++;
            else
                miscount ++;
        }    
    }
    fprintf (stderr, "Sym and dbond indexes done count (host) --> %d  (%d)\n", count, miscount);

    free (d_end);
    free (d_start);
    free (d_bond_data);

    return SUCCESS;
}


int validate_sparse_matrix( reax_system *system, storage *workspace )
{
    sparse_matrix test;
    int index, count, total;
    test.start = (int *) malloc (sizeof (int) * (system->N));
    test.end = (int *) malloc (sizeof (int) * (system->N));

    test.entries = (sparse_matrix_entry *) malloc 
        (sizeof (sparse_matrix_entry) * (dev_workspace->H.m));

    memset (test.entries, 0xFF, 
            sizeof (sparse_matrix_entry) * dev_workspace->H.m);
    copy_host_device ( test.entries, dev_workspace->H.entries, 
            sizeof (sparse_matrix_entry)* dev_workspace->H.m, 
            cudaMemcpyDeviceToHost, "sparse_matrix_entries");
    copy_host_device ( test.start, dev_workspace->H.start, sizeof (int) * (system->N), cudaMemcpyDeviceToHost, "start");
    copy_host_device ( test.end , dev_workspace->H.end, sizeof (int) * (system->N), cudaMemcpyDeviceToHost, "end");

    for (int i = 0 ; i < system->N; i++) {
        if ((test.end[i] >= dev_workspace->H.m)) {
            fprintf (stderr, " exceeding number of entries for atom: %d \n", i);
            exit (-1);
        }

        if (( i < (system->N-1)) && (test.end[i] >= test.start[i+1]))
        {
            fprintf (stderr, " Index exceeding for atom : %d \n", i );
            fprintf (stderr, "end(i): %d \n", test.end[i]);
            fprintf (stderr, "start(i+1): %d \n", test.start[i+1]);
            exit (-1);
        }
    }
    fprintf (stderr, "Sparse Matrix Boundary Check PASSED !!!\n");

    //TODO
    //TODO
    //TODO
    return SUCCESS;

    count = 0;
    for (int i = 0 ; i < system->N; i++) 
        count += test.end[i] - test.start[i];
    fprintf (stderr, " Total number of entries : %d \n", count);

    fprintf (stderr, " ALlocated memeory for entries : %d\n", dev_workspace->H.m);

    ////////////////////////////
    //for (int i = workspace->H.start[0]; i < workspace->H.end[0]; i++) {
    //    fprintf (stderr, "Row: 0, col: %d val: %f \n", workspace->H.entries[i].j, workspace->H.entries[i].val );
    //}
    //////////////////////////////

    count = 0;
    total = 0;
    for (int i = 0; i < system->n; i++) {
        for (int j = workspace->H.start[i]; j < workspace->H.end[i]; j++) {
            sparse_matrix_entry *src = &workspace->H.entries[j];

            for (int k = test.start[i]; k < test.end[i]; k++) {
                sparse_matrix_entry *tgt = &test.entries [k];
                if (src->j == tgt->j){
                    if ( check_zero (src->val, tgt->val)) {
                        index = test.start [i];
                        /*
                           fprintf (stderr, " i-1 (%d %d ) (%d %d) \n", 
                           test.start[i-1], test.end[i-1], 
                           workspace->H.start[i-1], workspace->H.start[i]);
                         */
                        fprintf (stderr, " Sparse matrix entry does not match for atom %d at index %d (%d %d) (%d %d) \n", 
                                i, k, test.start[i], test.end[i], 
                                workspace->H.start[i], workspace->H.end[i]);
                        for (int x = workspace->H.start[i]; x < workspace->H.end[i]; x ++)
                        {
                            src = &workspace->H.entries[x];
                            tgt = &test.entries [index];
                            fprintf (stderr, " cpu (%d %f)**** <--> gpu (%d %f) index %d \n", src->j, src->val, tgt->j, tgt->val, index);
                            index ++;
                        }
                        fprintf (stderr, "Sparse Matrix DOES NOT match between device and host \n");
                        exit (-1);
                        count++;
                    } else 
                    {
                        total ++;
                        if (i == tgt->j)  continue;
                        //if (tgt->j >= system->n) continue;

                        //success case here. check for row - k and column i;
                        for (int x = test.start[tgt->j]; x < test.end[tgt->j]; x++){
                            sparse_matrix_entry *rtgt = &test.entries [x];
                            if (i == rtgt->j) {
                                if (check_zero (tgt->val, rtgt->val)) {
                                    fprintf (stderr, "symmetric entry not matching for (%d, %d) \n", i, tgt->j);
                                    fprintf (stderr, "row: %d col: %d val: %f \n", i, tgt->j, tgt->val);
                                    fprintf (stderr, "row: %d col: %d val: %f \n", tgt->j, rtgt->j, rtgt->val);
                                    exit (-1);
                                } else {
                                    total ++;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fprintf( stderr, "Sparse Matrix mismatch total: %d, miscount %d  \n",
            total, count );
    free( test.start );
    free( test.end );
    free( test.entries );
    return SUCCESS;
}


void print_hbonds( int *d_start, int *d_end, int i, hbond_data *data )
{
    int j;
    hbond_data src, tgt; 

    fprintf( stderr, " start %d end %d count ---> %d \n",
            d_start[i], d_end[i], d_end[i] - d_start[i] );

    for ( j = d_start[i]; j < d_end[i]; j++ )
    {
        fprintf( stderr, "Atom : %d , Hbond Info . nbr: %d scl: %d index:%d\n",
                i, data[j].nbr, data[j].scl );
    }
    fprintf( stderr, " ========================================= \n" );
}


int validate_hbonds( reax_system *system, storage *workspace,
        reax_list **lists )
{
    int count, nbr, sym_count, dev_count;
    int *d_start, *d_end, index, d_index;
    hbond_data *data, src, tgt;
    reax_list *d_hbonds = *dev_lists + HBONDS;
    reax_list *hbonds = *lists + HBONDS;

    d_end = (int *)malloc (sizeof (int)* d_hbonds->n);
    d_start = (int *) malloc (sizeof (int) * d_hbonds->n );
    fprintf (stderr, "Total index values: %d \n", d_hbonds->n);

    copy_host_device (d_start, d_hbonds->index, sizeof (int)* d_hbonds->n, cudaMemcpyDeviceToHost, "start");
    copy_host_device (d_end, d_hbonds->end_index, sizeof (int) * d_hbonds->n, cudaMemcpyDeviceToHost, "end");

    //fprintf (stderr, "Copying hbonds to host %d \n", system->num_hbonds);
    data = (hbond_data *) malloc (sizeof (hbond_data) * d_hbonds->num_intrs);
    copy_host_device (data, d_hbonds->select.hbond_list, sizeof (hbond_data) * d_hbonds->num_intrs, 
            cudaMemcpyDeviceToHost, "hbond_data");

    count = 0;
    dev_count = 0;
    sym_count = 0;
    for (int i = 0; i < system->n; i++) {

        if ( system->reax_param.sbp[ system->my_atoms[i].type ].p_hbond == 1 )
        {
            count += End_Index (i, hbonds) - Start_Index (i, hbonds);
            dev_count += d_end [i] - d_start[i];

            if ((d_end[ i] - d_start[i])    !=
                    (End_Index (i, hbonds) - Start_Index (i, hbonds))) {
                fprintf (stderr, "%d %d - d(%d  %d) c(%d %d) \n",i, i,
                        d_start[i], d_end[ i],
                        Start_Index (i, hbonds),
                        End_Index (i, hbonds) );
                print_hbonds( d_start, d_end, i, data );
                print_hbonds( hbonds->index, hbonds->end_index, i, hbonds->select.hbond_list );
                exit (-1);
            }
        }
        else {
            sym_count += d_end[ i] - d_start[i];
        }
    }
    fprintf (stderr, "hbonds count match between host: %d and device: %d (%d) \n", count,dev_count, sym_count);
    sym_count = 0;

    for (int i = system->n; i < system->N; i++) {
        //if (system->reax_param.sbp[ system->my_atoms[i].type].p_hbond == 2)
        {
            sym_count += d_end[i] - d_start[i];
        }
    }
    fprintf (stderr, "Sym count outside 'n' : %d \n", sym_count );
    //print_hbonds( d_start, d_end, 0, data );


    count = 0;
    for (int i = 0; i < system->n; i++) {

        d_index = i; 
        /*
           fprintf (stderr, " Count cpu %d gpu %d \n", 
           End_Index (workspace->hbond_index[i], hbonds) - index, 
           d_end[d_index] - d_start[d_index]);
         */

        if ( system->reax_param.sbp[ system->my_atoms[i].type ].p_hbond != 1 )
        {
            /*
               int x;
               for (int j = d_start[d_index]; j < d_end[d_index]; j++ )
               {
               tgt = data [j];
               nbr = tgt.nbr;
               for (x = d_start[nbr]; x < d_end[nbr]; x++) 
               {
               src = data [x];
               if (src.nbr == i) {
               break;
               }
               }
               if (x >= d_end[nbr]) {
               fprintf (stderr, "HBONDS is NOT SYMMETRIC \n");
               fprintf (stderr, "Atom: %d, nbr: %d (%d)\n", i, nbr);
               fprintf (stderr, "Atom: %d, start: %d end: %d \n", nbr, d_start[nbr], d_end[nbr]);
               for (x = d_start[nbr]; x < d_end[nbr]; x++) 
               {
               src = data [x];
               fprintf (stderr, "Atom: %d, nbr: %d \n", nbr, src.nbr);
               }

               exit (1);
               }
               }
             */

            for (int j = d_start[d_index]; j < d_end[d_index]; j++ )
            {
                tgt = data[j];
                nbr = tgt.sym_index;

                if (nbr >= d_hbonds->num_intrs || nbr < 0){
                    fprintf (stderr, "Index out of range for atom: %d sym_index:%d Hbond index: %d, nbr: %d\n", i, nbr, j, data[j].nbr);
                    fprintf (stderr, "atom type: %d \n", system->reax_param.sbp[ system->my_atoms [ data[j].nbr ].type].p_hbond);
                    exit (1);
                }

                if (data[nbr].sym_index != j) {
                    fprintf (stderr, "Sym Index for hydrogen bonds does not match \n");
                    exit (1);
                }
            }
            continue;
        }

        for (int j = d_start[d_index]; j < d_end[d_index]; j++ )
        {
            tgt = data[j];

            int k = 0;
            for (k = Start_Index (i, hbonds);
                    k < End_Index (i, hbonds); k++) {
                src = hbonds->select.hbond_list[k];

                if ((src.nbr == tgt.nbr) && (src.scl == tgt.scl)) {
                    /*
                       fprintf (stderr, "Mismatch  at atom %d index %d (%d %d) -- (%d %d) \n", i, k,
                       src.nbr, src.scl, 
                       tgt.nbr, tgt.scl);
                     */
                    count ++;
                    break;
                }
            }

            /*
               if (  ((End_Index (workspace->hbond_index[i], hbonds) - index) != index ) && 
               (k >= End_Index (workspace->hbond_index[i], hbonds))) {
               fprintf (stderr, "Hbonds does not match for atom %d hbond_Index %d \n", i, d_index );
               exit (-1);
               }
             */

            if ( k >= (End_Index (i, hbonds) )){
                fprintf (stderr, "Hbonds does not match for atom %d hbond_Index %d \n", i, j);
                fprintf (stderr, " ==========Host============ \n");
                print_hbonds( hbonds->index, hbonds->end_index,
                        i, hbonds->select.hbond_list );
                fprintf (stderr, " ==========Device============ \n");
                print_hbonds( d_start, d_end,
                        i, data );
                exit (-1);
            }
        }

        if ((End_Index (i, hbonds)- Start_Index(i, hbonds)) != (d_end[i] - d_start[i])){
            fprintf (stderr, "End index does not match between device and host \n");
            fprintf (stderr, " Atom: %d Host: %d %d \n", i, Start_Index (i, hbonds), End_Index (i, hbonds));
            fprintf (stderr, " Device: %d %d \n", d_start[i], d_end[i]);
            exit (-1);
        }
    }

    fprintf (stderr, "HBONDs match on device and Host count --> %d\n", count);

    free (d_start);
    free (d_end);
    free (data);
    return SUCCESS;
}

int validate_bonds (reax_system *system, storage *workspace, reax_list **lists)
{
    int start, end, index, count, miscount;
    int *d_start, *d_end;
    bond_data *d_bond_data;
    reax_list *d_bonds = *dev_lists + BONDS;
    reax_list *bonds = *lists + BONDS;

    d_end = (int *)malloc (sizeof (int) * system->N);
    d_start = (int *) malloc (sizeof (int) * system->N );
    d_bond_data = (bond_data *) malloc (sizeof (bond_data) * d_bonds->num_intrs);
    //fprintf (stderr, "Num bonds copied from device to host is --> %d \n", system->num_bonds );

    copy_host_device (d_start, d_bonds->index, sizeof (int) * system->N, cudaMemcpyDeviceToHost, "start");
    copy_host_device (d_end, d_bonds->end_index, sizeof (int) * system->N, cudaMemcpyDeviceToHost, "end");
    copy_host_device (d_bond_data, d_bonds->select.bond_list, sizeof (bond_data) * d_bonds->num_intrs, 
            cudaMemcpyDeviceToHost, "bond_data");

    count = 0;
    for (int i = 0; i < system->N; i++) {
        start = Start_Index (i, bonds);
        end = End_Index (i, bonds);

        count += end - start;
        if ((end-start) != (d_end[i]-d_start[i])){
            fprintf (stderr, "Entries does NOT match --> atom %d: cpu (%d %d) gpu (%d %d) \n",
                    i, start, end, d_start[i], d_end[i]);
            exit (-1);
        }

    }
    fprintf (stderr, "BOND LIST COUNT match on device and host  count %d \n", count);

    for (int i = 0; i < system->N-1; i++) {
        if ( d_end[i] >= d_start[i+1] ){
            fprintf (stderr, "Bonds list check Overwrite @ index --> %d \n", i);
            exit (-1);
        }
    }
    //fprintf (stderr, " BOND LIST Overwrite *PASSED* \n");
    count = 0;
    miscount = 0;
    for (int i = 0; i < system->N; i++) {

        for (int j = d_start[i]; j < d_end[i]; j++) {
            bond_data *src, *tgt;
            src = &d_bond_data[j];
            bond_data *src_sym = & d_bond_data[ src->sym_index ];

            //Previously this was commented out. Thats why it was working.
            //if (i >= src->nbr) continue;

            int k = 0;
            for (k = Start_Index (i, bonds); k < End_Index (i, bonds); k++) {
                tgt = & (bonds->select.bond_list[k]);
                bond_data *tgt_sym = &(bonds->select.bond_list [tgt->sym_index]);

                if ((src->nbr == tgt->nbr) && !check_zero (src->d,tgt->d) &&
                        !check_zero (src->dvec,tgt->dvec) && check_same (src->rel_box, tgt->rel_box)) {

                    bond_order_data *s, *t;
                    s = &(src->bo_data);
                    t = &(tgt->bo_data);

                    if (  !check_zero (s->BO,t->BO) &&
                            !check_zero (s->BO_s,t->BO_s) &&
                            !check_zero(s->BO_pi,t->BO_pi)  &&
                            !check_zero (s->BO_pi2,t->BO_pi2) &&
                            !check_zero (s->Cdbo,t->Cdbo) && !check_zero (s->Cdbopi,t->Cdbopi) && !check_zero (s->Cdbopi2,t->Cdbopi2) &&
                            !check_zero (s->C1dbo,t->C1dbo) && !check_zero (s->C2dbo,t->C2dbo) && !check_zero (s->C3dbo,t->C3dbo) &&
                            !check_zero(s->C1dbopi,t->C1dbopi) && !check_zero(s->C2dbopi,t->C2dbopi) && !check_zero(s->C3dbopi,t->C3dbopi) && !check_zero(s->C4dbopi,t->C4dbopi) &&
                            !check_zero(s->C1dbopi2,t->C1dbopi2) && !check_zero(s->C2dbopi2,t->C2dbopi2) &&!check_zero(s->C3dbopi2,t->C3dbopi2) &&!check_zero(s->C4dbopi2,t->C4dbopi2) &&
                            !check_zero (s->dln_BOp_s, t->dln_BOp_s ) &&
                            !check_zero (s->dln_BOp_pi, t->dln_BOp_pi ) &&
                            !check_zero (s->dln_BOp_pi2, t->dln_BOp_pi2 ) &&
                            !check_zero (s->dBOp, t->dBOp )) {
                        count ++;

                        //Check the sym index and dbond index here for double checking
                        // bond_ij on both device and hosts are matched now. 
                        /*
                           bond_order_data *ss, *ts;
                           ss = & (src_sym->bo_data );
                           ts = & (tgt_sym->bo_data );

                           if ((src_sym->nbr != tgt_sym->nbr) || check_zero (src_sym->d,tgt_sym->d) ||
                           check_zero (src_sym->dvec,tgt_sym->dvec) || !check_same (src_sym->rel_box, tgt_sym->rel_box)
                           || check_zero (ss->Cdbo, ts->Cdbo)){

                           fprintf (stderr, " Sym Index information does not match for atom %d \n", i);
                           fprintf (stderr, " atom --> %d \n", i);
                           fprintf (stderr, " nbr --> %d %d\n", src->nbr, tgt->nbr );
                           fprintf (stderr, " d --> %f %f \n", src_sym->d, tgt_sym->d );
                           fprintf (stderr, " sym Index nbr --> %d %d \n", src_sym->nbr, tgt_sym->nbr );
                           fprintf (stderr, " dvec (%f %f %f) (%f %f %f) \n",
                           src_sym->dvec[0], src_sym->dvec[1], src_sym->dvec[2],
                           tgt_sym->dvec[0], tgt_sym->dvec[1], tgt_sym->dvec[2] );
                           fprintf (stderr, " ivec (%d %d %d) (%d %d %d) \n",
                           src_sym->rel_box[0], src_sym->rel_box[1], src_sym->rel_box[2],
                           tgt_sym->rel_box[0], tgt_sym->rel_box[1], tgt_sym->rel_box[2] );

                           fprintf (stderr, " sym index Cdbo (%4.10e %4.10e) \n", ss->Cdbo,ts->Cdbo );
                           exit (-1);
                           }
                         */

                        break;
                    }
                    fprintf (stderr, " d --> %f %f \n", src->d, tgt->d );
                    fprintf (stderr, " dvec (%f %f %f) (%f %f %f) \n",
                            src->dvec[0], src->dvec[1], src->dvec[2],
                            tgt->dvec[0], tgt->dvec[1], tgt->dvec[2] );
                    fprintf (stderr, " ivec (%d %d %d) (%d %d %d) \n",
                            src->rel_box[0], src->rel_box[1], src->rel_box[2],
                            tgt->rel_box[0], tgt->rel_box[1], tgt->rel_box[2] );

                    fprintf (stderr, "Bond_Order_Data does not match for atom %d neighbor (%d %d) BO (%e %e) BO_s (%e %e) BO_pi (%e %e) BO_pi2 (%e %e) \n", i,
                            src->nbr, tgt->nbr,
                            s->BO, t->BO,
                            s->BO_s, t->BO_s,
                            s->BO_pi, t->BO_pi,
                            s->BO_pi2, t->BO_pi2
                        );
                    fprintf (stderr, " dBOp (%e %e %e) (%e %e %e) \n", s->dBOp[0], s->dBOp[1], s->dBOp[2],
                            t->dBOp[0], t->dBOp[1], t->dBOp[2] );

                    fprintf (stderr, " Cdbo (%4.10e %4.10e) \n", s->Cdbo,t->Cdbo );
                    fprintf (stderr, " Cdbopi (%e %e) \n", s->Cdbopi,t->Cdbopi );
                    fprintf (stderr, " Cdbopi2 (%e %e) \n", s->Cdbopi2,t->Cdbopi2 );
                    fprintf (stderr, " C1dbo (%e %e %e)(%e %e %e) \n", s->C1dbo,s->C2dbo,s->C3dbo, t->C1dbo,t->C2dbo,t->C3dbo );
                    fprintf (stderr, " C1dbopi (%e %e %e %e) (%e %e %e %e)\n", s->C1dbopi,s->C2dbopi,s->C3dbopi,s->C4dbopi, t->C1dbopi,t->C2dbopi,t->C3dbopi,t->C4dbopi);
                    fprintf (stderr, " C1dbopi2 (%e %e %e %e) (%e %e %e %e)\n", s->C1dbopi2,s->C2dbopi2,s->C3dbopi2,s->C4dbopi2, t->C1dbopi2,t->C2dbopi2,t->C3dbopi2,t->C4dbopi2);
                    fprintf (stderr, " dln_BOp_s (%e %e %e ) (%e %e %e) \n",
                            s->dln_BOp_s[0], s->dln_BOp_s[1], s->dln_BOp_s[2],
                            t->dln_BOp_s[0], t->dln_BOp_s[1], t->dln_BOp_s[2] );
                    fprintf (stderr, " dln_BOp_pi (%e %e %e ) (%e %e %e) \n",
                            s->dln_BOp_pi[0], s->dln_BOp_pi[1], s->dln_BOp_pi[2],
                            t->dln_BOp_pi[0], t->dln_BOp_pi[1], t->dln_BOp_pi[2] );
                    fprintf (stderr, " dln_BOp_pi2 (%e %e %e ) (%e %e %e) \n",
                            s->dln_BOp_pi2[0], s->dln_BOp_pi2[1], s->dln_BOp_pi2[2],
                            t->dln_BOp_pi2[0], t->dln_BOp_pi2[1], t->dln_BOp_pi2[2] );

                    //miscount ++;
                    //break;
                    exit (-1);
                }
            }

            if (k >= End_Index (i, bonds)) {
                miscount ++;
                fprintf (stderr, " We have a problem with the atom %d and bond entry %d \n", i, j);
                exit (-1);
            }
        }
    }

    fprintf (stderr, " BONDS matched count %d miscount %d (%d) \n", count, miscount, (count+miscount));
    free (d_start);
    free (d_end);
    free (d_bond_data);
    return SUCCESS;
}

int validate_workspace (reax_system *system, storage *workspace)
{
    int miscount;
    int count, tcount;

    ///////////////////////
    //INIT FORCES
    ///////////////////////

    // bond_mark
    int *bond_mark = (int *)malloc (sizeof (int) * system->N);
    copy_host_device (bond_mark, dev_workspace->bond_mark, sizeof (int) * system->N, 
            cudaMemcpyDeviceToHost, "bond_mark");
    miscount = 0;
    for (int i = 0; i < system->N; i++) {
        if (workspace->bond_mark [i] != bond_mark [i])  {
            fprintf (stderr, "Bond_mark atom:%d -- %d:%d \n", i, bond_mark [i], workspace->bond_mark [i]);
            miscount ++;
        }
    }
    free (bond_mark);
    fprintf (stderr, " Bond Mark : %d \n", miscount );

    //total_bond_order
    real *total_bond_order = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (total_bond_order, dev_workspace->total_bond_order, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "total_bond_order");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->total_bond_order[i], total_bond_order[i])){
            fprintf (stderr, "Total bond order does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->total_bond_order[i], total_bond_order[i]);
            exit (-1);
            count ++;
        }    
    }
    free (total_bond_order);
    fprintf (stderr, "TOTAL Bond Order mismatch count %d\n", count);

    //////////////////////////////
    //BOND ORDERS 
    //////////////////////////////

    //deltap
    real *deltap= (real *) malloc ( system->N * sizeof (real));
    copy_host_device (deltap, dev_workspace->Deltap, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "deltap");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->Deltap[i], deltap[i])){
            fprintf (stderr, "deltap does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->Deltap[i], deltap[i]);
            exit (-1);
            count ++;
        }    
    }
    free (deltap);
    fprintf (stderr, "Deltap mismatch count %d\n", count);

    //deltap_boc
    real *deltap_boc = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (deltap_boc, dev_workspace->Deltap_boc, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "deltap_boc");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->Deltap_boc[i], deltap_boc[i])){
            fprintf (stderr, "deltap_boc does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->Deltap_boc[i], deltap_boc[i]);
            exit (-1);
            count ++;
        }    
    }
    free (deltap_boc);
    fprintf (stderr, "Deltap_boc mismatch count %d\n", count);


    rvec *dDeltap_self;
    dDeltap_self = (rvec *) calloc (system->N, sizeof (rvec) );
    copy_host_device (dDeltap_self, dev_workspace->dDeltap_self, system->N * sizeof (rvec), cudaMemcpyDeviceToHost, "ddeltap_self");

    count = 0; 
    for (int i = 0; i < system->N; i++ )
    {
        if (check_zero (workspace->dDeltap_self[i], dDeltap_self[i]))
        {    
            fprintf (stderr, "index: %d c (%f %f %f) g (%f %f %f )\n", i, 
                    workspace->dDeltap_self[i][0],
                    workspace->dDeltap_self[i][1],
                    workspace->dDeltap_self[i][2],
                    dDeltap_self[3*i+0],
                    dDeltap_self[3*i+1],
                    dDeltap_self[3*i+2] );
            exit (-1);
            count ++;
        }    
    }
    free (dDeltap_self);
    fprintf (stderr, "dDeltap_self mismatch count %d\n", count);

    //Delta
    real *delta = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (delta, dev_workspace->Delta, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "Delta");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->Delta[i], delta[i])){
            fprintf (stderr, "delta does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->Delta[i], delta[i]);
            exit (-1);
            count ++;
        }    
    }
    free (delta);
    fprintf (stderr, "Delta mismatch count %d\n", count);

    //Delta_e
    real *deltae = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (deltae, dev_workspace->Delta_e, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "Deltae");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->Delta_e[i], deltae[i])){
            fprintf (stderr, "deltae does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->Delta_e[i], deltae[i]);
            exit (-1);
            count ++;
        }    
    }
    free (deltae);
    fprintf (stderr, "Delta_e mismatch count %d\n", count);

    //vlpex
    real *vlpex= (real *) malloc ( system->N * sizeof (real));
    copy_host_device (vlpex, dev_workspace->vlpex, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "vlpex");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->vlpex[i], vlpex[i])){
            fprintf (stderr, "vlpex does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->vlpex[i], vlpex[i]);
            exit (-1);
            count ++;
        }    
    }
    free (vlpex);
    fprintf (stderr, "vlpex mismatch count %d\n", count);

    //nlp
    real *nlp = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (nlp, dev_workspace->nlp, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->nlp[i], nlp[i])){
            fprintf (stderr, "nlp does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->nlp[i], nlp[i]);
            exit (-1);
            count ++;
        }    
    }
    free (nlp);
    fprintf (stderr, "nlp mismatch count %d\n", count);

    //delta_lp
    real *Delta_lp = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (Delta_lp , dev_workspace->Delta_lp , system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "Delta_lp ");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->Delta_lp [i], Delta_lp [i])){
            fprintf (stderr, "Delta_lp  does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->Delta_lp [i], Delta_lp [i]);
            exit (-1);
            count ++;
        }    
    }
    free (Delta_lp );
    fprintf (stderr, "Delta_lp  mismatch count %d\n", count);

    //Clp
    real *Clp = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (Clp, dev_workspace->Clp, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "Clp");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->Clp[i], Clp[i])){
            fprintf (stderr, "Clp does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->Clp[i], Clp[i]);
            exit (-1);
            count ++;
        }    
    }
    free (Clp);
    fprintf (stderr, "Clp mismatch count %d\n", count);

    //dDelta_lp
    real *dDelta_lp = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (dDelta_lp, dev_workspace->dDelta_lp, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "dDelta_lp");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->dDelta_lp[i], dDelta_lp[i])){
            fprintf (stderr, "dDelta_lp does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->dDelta_lp[i], dDelta_lp[i]);
            exit (-1);
            count ++;
        }    
    }
    free (dDelta_lp);
    fprintf (stderr, "dDelta_lp mismatch count %d\n", count);

    //nlp_temp
    real *nlp_temp = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (nlp_temp, dev_workspace->nlp_temp, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "nlp_temp");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->nlp_temp[i], nlp_temp[i])){
            fprintf (stderr, "nlp_temp does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->nlp_temp[i], nlp_temp[i]);
            exit (-1);
            count ++;
        }    
    }
    free (nlp_temp);
    fprintf (stderr, "nlp_temp mismatch count %d\n", count);

    //Delta_lp_temp
    real *Delta_lp_temp = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (Delta_lp_temp, dev_workspace->Delta_lp_temp, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "Delta_lp_temp");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->Delta_lp_temp[i], Delta_lp_temp[i])){
            fprintf (stderr, "Delta_lp_temp does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->Delta_lp_temp[i], Delta_lp_temp[i]);
            exit (-1);
            count ++;
        }    
    }
    free (Delta_lp_temp);
    fprintf (stderr, "Delta_lp_temp mismatch count %d\n", count);


    //dDelta_lp_temp
    real *dDelta_lp_temp = (real *) malloc ( system->N * sizeof (real));
    copy_host_device (dDelta_lp_temp, dev_workspace->dDelta_lp_temp, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "dDelta_lp_temp");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->dDelta_lp_temp[i], dDelta_lp_temp[i])){
            fprintf (stderr, "dDelta_lp_temp does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->dDelta_lp_temp[i], dDelta_lp_temp[i]);
            exit (-1);
            count ++;
        }    
    }
    free (dDelta_lp_temp);
    fprintf (stderr, "dDelta_lp_temp mismatch count %d\n", count);

    //////////////////////////////
    //BONDS
    //////////////////////////////

    //CdDelta
    real *CdDelta= (real *) malloc ( system->N * sizeof (real));
    copy_host_device (CdDelta, dev_workspace->CdDelta, system->N * sizeof (real), 
            cudaMemcpyDeviceToHost, "CdDelta");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->CdDelta[i], CdDelta[i])){
            fprintf (stderr, "CdDelta does not match for atom %d (%4.15e %4.15e)\n",
                    i, workspace->CdDelta[i], CdDelta[i]);
            exit (-1);
            count ++;
        }    
    }
    free (CdDelta);
    fprintf (stderr, "CdDelta mismatch count %d\n", count);


    //////////////////////////////////
    //ATOM ENERGY
    //////////////////////////////////

    //////////////////////////////////
    //VALENCE ANGLES
    //////////////////////////////////
    rvec *f= (rvec *) malloc ( system->N * sizeof (rvec));
    copy_host_device (f, dev_workspace->f, system->N * sizeof (rvec), 
            cudaMemcpyDeviceToHost, "f");
    count = 0; 
    for (int i = 0; i < system->N; i++) {

        if ( check_zero (workspace->f[i], f[i])){
            fprintf (stderr, "f does not match for atom %d (%4.15e %4.15e, %4.15e) (%4.15e %4.15e, %4.15e)\n",
                    i, 
                    workspace->f[i][0], workspace->f[i][1], workspace->f[i][2], 
                    f[i][0], f[i][1], f[i][2]);
            //exit (-1);
            count ++;
        }    
    }
    free (f);
    fprintf (stderr, "f mismatch count %d\n", count);

    /////////////////////////////////////////////////////
    //QEq part
    /////////////////////////////////////////////////////
    compare_rvec2( workspace->d2, dev_workspace->d2, system->N, "d2" );

    compare_rvec2( workspace->q2, dev_workspace->q2, system->N, "q2" );

    compare_rvec2( workspace->x, dev_workspace->x, system->N, "x" );

    compare_rvec2( workspace->b, dev_workspace->b, system->N, "b" );

    return SUCCESS;
}

void compare_rvec2( rvec2 *host, rvec2 *device, int N, const char *msg)
{
    int count = 0;
    int miscount = 0;
    rvec2 *tmp = (rvec2 *) host_scratch;
    copy_host_device (tmp, device, sizeof (rvec2) * N, cudaMemcpyDeviceToHost, msg);

    for (int i = 0; i < N; i++)
    {
        if (check_zero_rvec2 (host [i], tmp [i])) {
            fprintf (stderr, " %s does not match at index: %d (%f %f) - (%f %f) \n", 
                    msg, i, host[i][0], host[i][1], tmp[i][0], tmp[i][1] );
            // exit (-1);
            miscount ++;
        }
        count ++;
    }
    fprintf (stderr, "%s match between host and device (%d - %d) \n", msg, count, miscount);
}

void compare_array( real *host, real *device, int N, const char *msg )
{
    int count = 0;
    int miscount = 0;
    real *tmp = (real *) host_scratch;
    copy_host_device (tmp, device, sizeof (real) * N, cudaMemcpyDeviceToHost, msg);

    for (int i = 0; i < N; i++)
    {
        if (check_zero (host [i], tmp [i])) {
            fprintf (stderr, " %s does not match at index: %d (%f) - (%f) \n", 
                    msg, i, host[i], tmp[i] );
            // exit (-1);
            miscount ++;
        }
        count ++;
    }
    fprintf (stderr, "%s match between host and device (%d - %d) \n", msg, count, miscount);
}


int validate_data (reax_system *system, simulation_data *host)
{
    simulation_data device;

    copy_host_device (&device, host->d_simulation_data, sizeof (simulation_data), 
            cudaMemcpyDeviceToHost, "simulation_data");

    if (check_zero (host->my_en.e_bond, device.my_en.e_bond)){
        fprintf (stderr, "E_BE does not match (%4.15e %4.15e) \n", host->my_en.e_bond, device.my_en.e_bond);
        exit (-1);
    }

    if (check_zero (host->my_en.e_lp, device.my_en.e_lp)){
        fprintf (stderr, "E_Lp does not match (%4.10e %4.10e) \n", host->my_en.e_lp, device.my_en.e_lp);
        exit (-1);
    }

    if (check_zero (host->my_en.e_ov, device.my_en.e_ov)){
        fprintf (stderr, "E_Ov does not match (%4.10e %4.10e) \n", host->my_en.e_ov, device.my_en.e_ov);
        exit (-1);
    }

    if (check_zero (host->my_en.e_un, device.my_en.e_un)){
        fprintf (stderr, "E_Un does not match (%4.10e %4.10e) \n", host->my_en.e_un, device.my_en.e_un);
        exit (-1);
    }

    if (check_zero (host->my_en.e_tor, device.my_en.e_tor)) {
        fprintf (stderr, "E_Tor does not match (%4.10e %4.10e) \n", host->my_en.e_tor, device.my_en.e_tor);
        exit (-1);
    }

    if (check_zero (host->my_en.e_con, device.my_en.e_con)) {
        fprintf (stderr, "E_Con does not match (%4.10e %4.10e) \n", host->my_en.e_con, device.my_en.e_con);
        exit (-1);
    }

    fprintf (stderr, "E_Hb does not match (%4.10e %4.10e) \n", host->my_en.e_hb, device.my_en.e_hb);
    if (check_zero (host->my_en.e_hb, device.my_en.e_hb)) {
        fprintf (stderr, "E_Hb does not match (%4.10e %4.10e) \n", host->my_en.e_hb, device.my_en.e_hb);
        exit (-1);
    }

    if (check_zero (host->my_en.e_ang, device.my_en.e_ang)) {
        fprintf (stderr, "E_Ang does not match (%4.10e %4.10e) \n", host->my_en.e_ang, device.my_en.e_ang);
        exit (-1);
    }

    if (check_zero (host->my_en.e_pen, device.my_en.e_pen)) {
        fprintf (stderr, "E_Pen does not match (%4.10e %4.10e) \n", host->my_en.e_pen, device.my_en.e_pen);
        exit (-1);
    }

    if (check_zero (host->my_en.e_coa, device.my_en.e_coa)) {
        fprintf (stderr, "E_Coa does not match (%4.10e %4.10e) \n", host->my_en.e_coa, device.my_en.e_coa);
        exit (-1);
    }

    if (check_zero (host->my_en.e_vdW, device.my_en.e_vdW)) {
        fprintf (stderr, "E_vdW does not match (%4.20e %4.20e) \n", host->my_en.e_vdW, device.my_en.e_vdW);
        exit (-1);
    }

    if (check_zero (host->my_en.e_pol, device.my_en.e_pol)) {
        fprintf (stderr, "E_Pol does not match (%4.10e %4.10e) \n", host->my_en.e_pol, device.my_en.e_pol);
        //exit (-1);
    }

    if (check_zero (host->my_en.e_kin, device.my_en.e_kin)) {
        fprintf (stderr, "E_Kin does not match (%4.10e %4.10e) \n", host->my_en.e_kin, device.my_en.e_kin);
        //exit (-1);
    }

    if (check_zero (host->my_en.e_ele, device.my_en.e_ele)) {
        fprintf (stderr, "E_Ele does not match (%4.20e %4.20e) \n", host->my_en.e_ele, device.my_en.e_ele);
        //exit (-1);
    }

    fprintf (stderr, "Simulation Data match between host and device \n");
    return SUCCESS;
}

int validate_grid (reax_system *system)
{
    int  x,i, j, k,l, itr; //, tmp, tested;
    int itr_nbr,itr_11, miscount;
    ivec src, dest;
    grid *g;
    grid_cell *gci, *gcj, *gcj_nbr;
    int found = 0;

    int *tmp = (int *) host_scratch;
    int total;

    g = &( system->my_grid );
    miscount = 0;

    total = g->ncells[0] * g->ncells[1] * g->ncells[2];

    copy_host_device (tmp, system->d_my_grid.str, sizeof(int) * total, cudaMemcpyDeviceToHost, "grid:str");
    copy_host_device (tmp + total, system->d_my_grid.end, sizeof(int) * total, cudaMemcpyDeviceToHost, "grid:end");

    real *cutoff = (real *) (tmp + 2 * total);
    copy_host_device (cutoff, system->d_my_grid.cutoff, sizeof (real) * total, cudaMemcpyDeviceToHost, "grid:cutoff");

    for( i = 0; i < g->ncells[0]; i++ )
        for( j = 0; j < g->ncells[1]; j++ )
            for( k = 0; k < g->ncells[2]; k++ ) 
            {
                if ((g->str [index_grid_3d (i, j, k, g)] != tmp [index_grid_3d (i, j, k, g)]) || 
                        (g->end [index_grid_3d (i, j, k, g)] != tmp[total + index_grid_3d (i, j, k, g)]) ||
                        (cutoff [index_grid_3d (i, j, k, g)] != g->cutoff [index_grid_3d (i, j, k, g)]))
                {
                    fprintf (stderr, "we have a problem here \n");
                    exit (0);
                }
                /*
                   fprintf (stderr, " %d %d %d - str: %d end: %d  (%d %d) ( %f %f)\n", 
                   i, j, k, g->str [index_grid_3d (i, j, k, g)], g->end [index_grid_3d (i, j, k, g)], 
                   tmp [index_grid_3d (i, j, k, g)], tmp[total + index_grid_3d (i, j, k, g)], 
                   cutoff [index_grid_3d (i, j, k, g)], g->cutoff [index_grid_3d (i, j, k, g)]);
                 */
            }

    rvec *tmpvec = (rvec *) host_scratch;
    copy_host_device (tmpvec, system->d_my_grid.nbrs_cp, sizeof (rvec) * total * g->max_nbrs, 
            cudaMemcpyDeviceToHost, "grid:nbrs_cp");

    ivec *tivec = (ivec *) (((rvec *)host_scratch) + total * g->max_nbrs);
    copy_host_device (tivec, system->d_my_grid.nbrs_x, sizeof (ivec) * total * g->max_nbrs, 
            cudaMemcpyDeviceToHost, "grid:nbrs_x");


    for( i = 0; i < g->ncells[0]; i++ )
        for( j = 0; j < g->ncells[1]; j++ )
            for( k = 0; k < g->ncells[2]; k++ ) 
                for (l = 0; l < g->max_nbrs; l++) {

                    if (( g->nbrs_cp[index_grid_nbrs(i, j, k, l, g)][0] != tmpvec[index_grid_nbrs(i, j, k, l, g)][0]) ||
                            (g->nbrs_cp[index_grid_nbrs(i, j, k, l, g)][1] != tmpvec[index_grid_nbrs(i, j, k, l, g)][1]) || 
                            (g->nbrs_cp[index_grid_nbrs(i, j, k, l, g)][2] != tmpvec[index_grid_nbrs(i, j, k, l, g)][2]) || 
                            (g->nbrs_x[index_grid_nbrs(i, j, k, l, g)][0] != tivec[index_grid_nbrs(i, j, k, l, g)][0]) ||
                            (g->nbrs_x[index_grid_nbrs(i, j, k, l, g)][1] != tivec[index_grid_nbrs(i, j, k, l, g)][1]) || 
                            (g->nbrs_x[index_grid_nbrs(i, j, k, l, g)][2] != tivec[index_grid_nbrs(i, j, k, l, g)][2] )) 
                    {
                        fprintf (stderr, "we have a big problem here \n");
                        exit (0);
                    }

                    if ((g->nbrs_cp[index_grid_nbrs(i, j, k, l, g)][0] > NEG_INF) &&
                            (g->nbrs_cp[index_grid_nbrs(i, j, k, l, g)][1] > NEG_INF) &&
                            (g->nbrs_cp[index_grid_nbrs(i, j, k, l, g)][2] > NEG_INF) )
                        ;/*
                            fprintf (stderr, "%d %d %d %d ---- %d %d %d - %d %d %d \n", 
                    //fprintf (stderr, "%d %d %d %d ---- (%3.2f %3.2f %3.2f) - (%3.2f %3.2f %3.2f) \n", 
                    i, j, k, l, 
                    g->nbrs_x[index_grid_nbrs(i, j, k, l, g)][0], 
                    g->nbrs_x[index_grid_nbrs(i, j, k, l, g)][1], 
                    g->nbrs_x[index_grid_nbrs(i, j, k, l, g)][2], 
                    tivec[index_grid_nbrs(i, j, k, l, g)][0], 
                    tivec[index_grid_nbrs(i, j, k, l, g)][1], 
                    tivec[index_grid_nbrs(i, j, k, l, g)][2]
                    );
                          */
                }

    return 0;

    //  for( i = 0; i < g->ncells[0]; i++ )
    //    for( j = 0; j < g->ncells[1]; j++ )
    //      for( k = 0; k < g->ncells[2]; k++ ) 
    //        { 
    //         gci = &(g->cells[ index_grid_3d (i, j, k, g) ]); 
    //            //for (x = 0; x < g->max_nbrs; x++)
    //            //    fprintf (stderr, "(%d, %d, %d) - (%d, %d, %d) \n", 
    //            //                            i, j, k, 
    //            //                            gci->nbrs_x[x][0],
    //            //                            gci->nbrs_x[x][1],
    //            //                            gci->nbrs_x[x][2] );
    //            //exit (0);
    //
    //           itr = 0;
    //           while( (gcj=gci->nbrs[itr]) != NULL ) 
    //           {
    //                    //iterate through the neighbors of gcj and find (i, j, k)
    //                    itr_nbr = 0;
    //                    found = 0;
    //                    while ( (gcj_nbr=gcj->nbrs [itr_nbr]) != NULL )
    //                    {
    //                        ivec_Copy (dest, gcj_nbr->nbrs_x[itr_nbr] );
    //
    //                        if ( (i == dest[0]) && (j == dest[1]) && (k == dest[2]))
    //                        {
    //                            found = 1;
    //                            break;
    //                        }
    //                        itr_nbr ++;
    //                    }
    //
    //                    if (found == 0) {
    //                        fprintf (stderr, "we have a problem here: (%d, %d, %d): (%d, %d, %d) type: (%d, %d) \n",
    //                                            i, j, k, 
    //                                            gci->nbrs_x[itr][0],
    //                                            gci->nbrs_x[itr][1],
    //                                            gci->nbrs_x[itr][2],
    //                                            gci->type, 
    //                                            gcj->type);
    //                        itr_11 = 0;
    //                        while ( (gcj_nbr=gcj->nbrs [itr_11]) != NULL )
    //                        {
    //                            ivec_Copy (dest, gcj_nbr->nbrs_x[itr_11] );
    //                            fprintf (stderr, "%d, %d, %d \n", dest[0], dest[1], dest[2]);
    //                            itr_11 ++;
    //                        }
    //                        exit (0);
    //                        miscount ++;
    //                    }
    //
    //                    itr ++;
    //              }
    //        }
    //
    //        fprintf (stderr, " cell miscount: %d \n", miscount);
}

int validate_three_bodies (reax_system *system, storage *workspace, reax_list **lists)
{
    reax_list *three = *lists + THREE_BODIES;
    reax_list *bonds = *lists + BONDS;

    reax_list *d_three = *dev_lists + THREE_BODIES;
    reax_list *d_bonds = *dev_lists + BONDS;
    bond_data *d_bond_data;
    real *test;

    three_body_interaction_data *data = (three_body_interaction_data *)
        malloc ( sizeof (three_body_interaction_data) * d_three->num_intrs);
    int *start = (int *) malloc (sizeof (int) * d_three->n);
    int *end = (int *) malloc (sizeof (int) * d_three->n);

    int *b_start = (int *) malloc (sizeof (int) * d_bonds->n);
    int *b_end = (int *) malloc (sizeof (int) * d_bonds->n);
    int count;
    int hcount, dcount;


    copy_host_device ( start, d_three->index,
            sizeof (int) * d_three->n, cudaMemcpyDeviceToHost, "three:start");
    copy_host_device ( end, d_three->end_index,
            sizeof (int) * d_three->n, cudaMemcpyDeviceToHost, "three:end");
    copy_host_device ( data, d_three->select.three_body_list,
            sizeof (three_body_interaction_data) * d_three->num_intrs,
            cudaMemcpyDeviceToHost, "three:data");

    d_bond_data = (bond_data *) malloc (sizeof (bond_data)* d_bonds->num_intrs);

    copy_host_device ( b_start, d_bonds->index,
            sizeof (int) * d_bonds->n, cudaMemcpyDeviceToHost, "bonds:start");
    copy_host_device ( b_end, d_bonds->end_index,
            sizeof (int) * d_bonds->n, cudaMemcpyDeviceToHost, "bonds:end");
    copy_host_device (d_bond_data, d_bonds->select.bond_list, sizeof (bond_data) *  d_bonds->num_intrs, 
            cudaMemcpyDeviceToHost, "bonds:data");

    count = 0;
    hcount = dcount = 0;
    for (int i = 0; i < system->N; i++)
    {

        int x, y, z;
        for (x = b_start[i]; x < b_end[i]; x++)
        {
            int t_start = start[x];
            int t_end = end[x];

            bond_data *dev_bond = &d_bond_data [x];
            bond_data *host_bond;
            for (z = Start_Index (i, bonds); z < End_Index (i, bonds); z++)
            {
                host_bond = &bonds->select.bond_list [z];
                if ((dev_bond->nbr == host_bond->nbr) &&
                        check_same (dev_bond->rel_box, host_bond->rel_box) &&
                        !check_zero (dev_bond->dvec, host_bond->dvec) &&
                        !check_zero (dev_bond->d, host_bond->d) )
                {
                    break;
                }
            }
            if (z >= End_Index (i, bonds)){
                fprintf (stderr, "Could not find the matching bond on host and device \n");
                exit (-1);
            }

            dcount += end[x] - start[x];
            hcount += Num_Entries (z, three);

            if ((end[x] - start[x]) != (End_Index (z, three) - Start_Index (z, three)))
            {
                count ++;
                /*
                   fprintf (stderr, " Three body count does not match between host and device\n");
                   fprintf (stderr, " Host count : (%d, %d)\n", Start_Index (z, three), End_Index (z, three));
                   fprintf (stderr, " atom: %d - bond: %d Device count: (%d, %d)\n", i, x, start[x], end[x]);
                 */
            }
        }

        /*
           if ((dcount != hcount)) {

           fprintf (stderr, " Three body count does not match for the bond %d - %d \n", hcount, dcount);

           for (int j = b_start[i]; j < b_end[i]; j ++) {
           bond_order_data *src = &d_bond_data[j].bo_data;
           dcount = end[j] - start[j];
           hcount = Num_Entries (j, three);
           fprintf (stderr, "device \n");
           print_bond_data (src);

           fprintf (stderr, "\n");
           src = &bonds->select.bond_list[j].bo_data;
           fprintf (stderr, "host \n");
           print_bond_data (src);
           fprintf (stderr, "\n");

        //fprintf (stderr, "--- Device bo is %f \n", test[j]);
        fprintf (stderr, "Device %d %d bonds (%d %d) - Host %d %d bonds (%d %d) \n", start[j], end[j],b_start[i], b_end[i],
        Start_Index (j, three), End_Index (j, three), Start_Index (i, bonds), End_Index (i, bonds));
        fprintf (stderr, "Host %d Device %d -- atom %d index %d \n", hcount, dcount, i, j);
        fprintf (stderr, "------\n");
        }
        fprintf (stderr, " Three Bodies count does not match between host and device \n");
        exit (-1);
        }
         */
    }
    fprintf (stderr, "Three body count on DEVICE %d  HOST %d -- miscount: %d\n", dcount, hcount, count);

    count = 0;
    for (int i = 0; i < system->N; i++)
    {
        int x, y, z;
        for (x = b_start[i]; x < b_end[i]; x++)
        {
            int t_start = start[x];
            int t_end = end[x];

            bond_data *dev_bond = &d_bond_data [x];
            bond_data *host_bond;
            for (z = Start_Index (i, bonds); z < End_Index (i, bonds); z++)
            {
                host_bond = &bonds->select.bond_list [z];
                if ((dev_bond->nbr == host_bond->nbr) &&
                        check_same (dev_bond->rel_box, host_bond->rel_box) &&
                        !check_zero (dev_bond->dvec, host_bond->dvec) &&
                        !check_zero (dev_bond->d, host_bond->d) )
                {
                    break;
                }
            }
            if (z >= End_Index (i, bonds)){
                fprintf (stderr, "Could not find the matching bond on host and device \n");
                exit (-1);
            }

            //find this three-body in the bonds on the host side.
            for (y = t_start; y < t_end; y++)
            {
                three_body_interaction_data *device = data + y;
                three_body_interaction_data *host;

                //fprintf (stderr, "Device thb %d pthb %d \n", device->thb, device->pthb);

                int xx;
                for (xx = Start_Index (z, three); xx < End_Index (z, three); xx++)
                {
                    host = &three->select.three_body_list [xx];
                    //fprintf (stderr, "Host thb %d pthb %d \n", host->thb, host->pthb);
                    //if ((host->thb == device->thb) && (host->pthb == device->pthb))
                    if ((host->thb == device->thb) && !check_zero (host->theta, device->theta))
                    {
                        count ++;
                        break;
                    }
                }

                if ( xx >= End_Index (z, three) ) {
                    fprintf (stderr, " Could not match for atom %d bonds %d (%d) Three body(%d %d) (%d %d) \n", i, x, z,
                            Start_Index (z, three), End_Index (z, three), start[x], end[x] );
                    exit (-1);
                }// else fprintf (stderr, "----------------- \n");
            }
        }
    }
    free (data);
    free (start);
    free (end);
    free (b_start);
    free (b_end);
    free (d_bond_data);

    fprintf (stderr, "Three Body Interaction Data MATCH on device and HOST --> %d \n", count);
    return SUCCESS;
}


int validate_atoms (reax_system *system, reax_list **lists)
{

    int start, end, index, count, miscount;
    reax_atom *test = (reax_atom *) malloc (sizeof (reax_atom)* system->N);
    copy_host_device (test, system->d_my_atoms, sizeof (reax_atom) * system->N, cudaMemcpyDeviceToHost, "atoms");

    /*
       for (int i = system->n; i < system->n + 10; i++)
       {
       fprintf (stderr, " Atom: %d HIndex: %d \n", i, test[i].Hindex);
       }
     */

    count = miscount = 0; 
    for (int i = 0; i < system->N; i++) 
    {
        if (test[i].type != system->my_atoms[i].type) {
            fprintf (stderr, " Type does not match (%d %d) @ index %d \n", system->my_atoms[i].type, test[i].type, i);
            exit (-1);
        }    

        if (  check_zero (test[i].x, system->my_atoms[i].x) )
        {    
            fprintf (stderr, "Atom :%d x --> host (%f %f %f) device (%f %f %f) \n", i,
                    system->my_atoms[i].x[0], system->my_atoms[i].x[1], system->my_atoms[i].x[2], 
                    test[i].x[0], test[i].x[1], test[i].x[2] );
            miscount ++;
            exit (-1);
        }
        if (     check_zero (test[i].v, system->my_atoms[i].v) )
        {
            fprintf (stderr, "Atom :%d v --> host (%6.10f %6.10f %6.10f) device (%6.10f %6.10f %6.10f) \n", i,
                    system->my_atoms[i].v[0], system->my_atoms[i].v[1], system->my_atoms[i].v[2],
                    test[i].v[0], test[i].v[1], test[i].v[2] );
            miscount ++;
            exit (-1);
        }
        if (     check_zero (test[i].f, system->my_atoms[i].f) )
        {
            fprintf (stderr, "Atom :%d f --> host (%6.10f %6.10f %6.10f) device (%6.10f %6.10f %6.10f) \n", i,
                    system->my_atoms[i].f[0], system->my_atoms[i].f[1], system->my_atoms[i].f[2],
                    test[i].f[0], test[i].f[1], test[i].f[2] );
            miscount ++;
            exit (-1);
        }

        if (     check_zero (test[i].q, system->my_atoms[i].q) )
        {
            fprintf (stderr, "Atom :%d q --> host (%f) device (%f) \n", i,
                    system->my_atoms[i].q, test[i].q );
            miscount ++;
            exit (-1);
        }

        count ++;
    }

    fprintf (stderr, "Reax Atoms DOES **match** between host and device --> %d miscount --> %d \n", count, miscount);

    free (test);
    return true;
}


int print_sparse_matrix (sparse_matrix *H)
{
    sparse_matrix test;
    int index, count;

    test.start = (int *) malloc (sizeof (int) * (H->cap)); 
    test.end = (int *) malloc (sizeof (int) * (H->cap)); 

    test.entries = (sparse_matrix_entry *) malloc (sizeof (sparse_matrix_entry) * (H->m));
    memset (test.entries, 0xFF, sizeof (sparse_matrix_entry) * H->m);

    copy_host_device ( test.entries, dev_workspace->H.entries, 
            sizeof (sparse_matrix_entry) * H->m, cudaMemcpyDeviceToHost, "H:m");
    copy_host_device ( test.start, dev_workspace->H.start, sizeof (int)* (H->cap), cudaMemcpyDeviceToHost, "H:start");
    copy_host_device ( test.end , dev_workspace->H.end, sizeof (int) * (H->cap), cudaMemcpyDeviceToHost, "H:end");

    count = 0; 
    for (int i = 0; i < 1; i++) {
        for (int j = test.start[i]; j < test.end[i]; j++) {
            sparse_matrix_entry *src = &test.entries[j];
            fprintf (stderr, "Row:%d:%d:%f\n", i, src->j, src->val);
        }    
    }
    fprintf (stderr, "--------------- ");

    free (test.start);
    free (test.end);
    free (test.entries);

    return SUCCESS;
}

int print_sparse_matrix_host (sparse_matrix *H)
{
    int index, count;

    count = 0; 
    for (int i = 0; i < 1; i++) {
        for (int j = H->start[i]; j < H->end[i]; j++) {
            sparse_matrix_entry *src = &H->entries[j];
            fprintf (stderr, "Row:%d:%d:%f\n", i, src->j, src->val);
        }    
    }
    fprintf (stderr, "--------------- ");
    return SUCCESS;
}

int print_host_rvec2 (rvec2 *a, int n)
{
    for (int i = 0; i < n; i++)
        fprintf (stderr, "a[%f][%f] \n", a[i][0], a[i][1]);
    fprintf (stderr, " ---------------------------------\n");

    return SUCCESS;
}

int print_device_rvec2 (rvec2 *b, int n)
{
    rvec2 *a = (rvec2 *) host_scratch;    

    copy_host_device (a, b, sizeof (rvec2) * n, cudaMemcpyDeviceToHost, "rvec2");

    return print_host_rvec2 (a, n);
}


int print_host_array( real *a, int n )
{
    int i;

    for ( i = 0; i < n; i++ )
    {
        fprintf( stderr," a[%d] = %f \n", i, a[i] );
    }
    fprintf( stderr, " ----------------------------------\n" );

    return SUCCESS;
}


int print_device_array( real *a, int n )
{
    real *b = (real *) host_scratch;
    copy_host_device( b, a, sizeof(real) * n,
            cudaMemcpyDeviceToHost, "real");
    print_host_array( b, n );

    return SUCCESS;
}


int check_zeros_host( rvec2 *host, int n, const char *msg )
{
    int i, count, count1;

    count = 0;
    count1 = 0;

    for ( i = 0; i < n; i++ )
    {
        if (host[i][0] == 0)
        {
            count++;
        }
        if (host[i][1] == 0)
        {
            count1++;
        }
    }

    fprintf( stderr, "%s has %d, %d zero elements \n",
            msg, count, count1 );

    return SUCCESS;
}


int check_zeros_device( rvec2 *device, int n, const char *msg )
{
    rvec2 *a = (rvec2 *) host_scratch;    

    copy_host_device( a, device, sizeof(rvec2) * n,
            cudaMemcpyDeviceToHost, msg );

    check_zeros_host( a, n, msg );

    return SUCCESS;
}
