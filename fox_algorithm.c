#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include<stdlib.h>
#include<time.h>
#define N 288

void rearrangeMatrix(double matrix[][N], int submatrixSize, int num_blocks, double new_matrix[N*N]) {
    int count=0;

    for(int i=0; i<sqrt(num_blocks); i++){
        for(int j=0; j<sqrt(num_blocks); j++){
            for(int k=i*submatrixSize; k<(i+1)*submatrixSize; k++){
                for(int w=j*submatrixSize; w<(j+1)*submatrixSize; w++){
                    new_matrix[count] = matrix[k][w];
                    count++;
                }
            }
        }
    }    
}

void multiply_matrices(double *matrix1, double *matrix2, double *result, int size) {
    for (int i = 0; i < size; i++) {
        for(int j=0; j<size; j++){
            result[i*size + j] = 0;
            for(int k=0; k<size; k++){
                result[i*size + j] += matrix1[i*size + k]*matrix2[j + k*size];
            }
        }
    }
}


int main(int argc, char *argv[]) {

    int rank, num_procs, i, provided;
    double A[N][N], B[N][N], C[N][N];
    double newA[N*N], newB[N*N];
    int tile_size;
    int start_time, end_time;

    //definition of the matrices
    for(int j = 0; j < N; j++)
        for(int k = 0; k < N; k++)
            A[j][k] = j + k;

    for(int j = 0; j < N; j++)
        for(int k = 0; k < N; k++)
            B[j][k] = j - k;

    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int sqrt_size = sqrt(num_procs);
    tile_size = N / sqrt_size; //size of each sub_matrix

    double local_subA[tile_size*tile_size], local_subB[tile_size*tile_size], local_subC[tile_size*tile_size];
    double current_A[tile_size*tile_size], current_B[tile_size*tile_size], current_C[tile_size*tile_size];


    for(int j=0; j<tile_size*tile_size; j++)
        local_subC[j] = 0;

    if(rank == 0) { //we make process 0 reordering the matrices
        rearrangeMatrix(A, tile_size, num_procs, newA);
        rearrangeMatrix(B, tile_size, num_procs, newB);
    }

    //broadcasting of the matrices chunck to the different processes
    MPI_Scatter(newA, tile_size*tile_size, MPI_DOUBLE, local_subA, tile_size*tile_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); 
    MPI_Scatter(newB, tile_size*tile_size, MPI_DOUBLE, local_subB, tile_size*tile_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    

    //process coordinates
    int row = rank / sqrt_size;
    int col = rank % sqrt_size;

    MPI_Comm row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, rank, &row_comm); //splitting into row-based communicators

    MPI_Comm col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, col, rank, &col_comm); //splitting into column-based communicators

    int row_rank;
    MPI_Comm_rank(row_comm, &row_rank);

    int col_rank;
    MPI_Comm_rank(col_comm, &col_rank);


    start_time = MPI_Wtime();

    for(int count=0, index=0; count<sqrt_size; count++){

        //the following if statement handles which process will broadcast its submatrix to its row
        if(count==0){
            if(row==col){
                index = row;
                for(int w=0; w<tile_size*tile_size; w++)
                    current_A[w] = local_subA[w];
            }
            MPI_Bcast(&index, 1, MPI_INT, row, row_comm); //broadcasting index so that each process know who is sending its submatrix
        }else{
            if(row_rank == index){
                for(int w=0; w<tile_size*tile_size; w++)
                    current_A[w] = local_subA[w];
            }
            MPI_Bcast(&index, 1, MPI_INT, index, row_comm);
        }

        MPI_Bcast(current_A, tile_size*tile_size, MPI_DOUBLE, index, row_comm);
        index = (index+1)%sqrt_size; //next process to send

        multiply_matrices(current_A, local_subB, current_C, tile_size);        

        for(int j=0; j<tile_size*tile_size; j++)
            local_subC[j] += current_C[j];

        //vertical shift of matrix B
        for (int t = 0; t < sqrt_size; t++) {
            int dest_rank = (col_rank - 1 + sqrt_size) % sqrt_size; 
            int source_rank = (col_rank + 1) % sqrt_size; 
            MPI_Sendrecv(local_subB, tile_size*tile_size, MPI_DOUBLE, dest_rank, 0, current_B, tile_size*tile_size, MPI_DOUBLE, source_rank, 0, col_comm, MPI_STATUS_IGNORE);
        }

        for(int t=0; t<tile_size*tile_size; t++)
            local_subB[t] = current_B[t];
            
        MPI_Barrier(col_comm);
        
    }


    end_time = MPI_Wtime () ;

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);

    printf("process %d A\n", rank);
    for(int j=0; j<tile_size; j++){
        for(int k=0; k<tile_size; k++)
            printf("%f ", local_subA[j*tile_size+k]);
        printf("\n");
    }

    printf("process %d B\n", rank);
    for(int j=0; j<tile_size; j++){
        for(int k=0; k<tile_size; k++)
            printf("%f ", local_subB[j*tile_size+k]);
        printf("\n");
    }

    printf("process %d C\n", rank);
    for(int j=0; j<tile_size; j++){
        for(int k=0; k<tile_size; k++)
            printf("%f ", local_subC[j*tile_size+k]);
        printf("\n");
    }

    if(rank==0)
        printf("Execution time %f\n", end_time-start_time);

    MPI_Finalize();
    return 0;
}