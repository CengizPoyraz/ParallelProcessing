#include <mpi.h>      // Include MPI parallel computing library
#include <stdio.h>    // Include standard input/output library
#include <stdlib.h>   // Include standard library for memory allocation, conversions
#include <math.h>     // Include mathematical functions library

// Function to compute the exact solution for the Poisson problem
// This is used for boundary conditions and error checking
double exact(double x, double y) {
    return x * y;  // Simple exact solution: u(x,y) = xy
}

// Function to compute the source term f(x,y) for the Poisson equation
// For the given exact solution (u(x,y) = xy), this returns 0
double f(double x, double y) {
    // For u(x,y) = xy, the Laplacian (uxx + uyy) = 0
    return 0.0;
}

// Compute an optimal 2D grid decomposition for parallel processes
void compute_2d_grid(int num_procs, int* rows, int* cols) {
    int sqrt_num_procs = sqrt(num_procs);  // Compute square root of total processes
    
    // Find the largest divisor less than or equal to sqrt(num_procs)
    for (int i = sqrt_num_procs; i >= 1; i--) {
        if (num_procs % i == 0) {
            *rows = i;               // Set number of process rows
            *cols = num_procs / i;   // Set number of process columns
            return;
        }
    }
    
    // Fallback in case no divisors found (shouldn't happen)
    *rows = 1;
    *cols = num_procs;
}

// Perform one Jacobi iteration for a local subdomain
double jacobi_iteration(double **u, double **u_new, int local_rows, int local_cols, 
                        int global_rows, int global_cols, double h, 
                        int start_row, int start_col, 
                        double *top_recv_buffer, double *bottom_recv_buffer, 
                        double *left_recv_buffer, double *right_recv_buffer) {
    double local_max_diff = 0.0;  // Track maximum difference in this iteration

    // Update interior points of local subdomain
    for (int i = 1; i < local_rows - 1; i++) {
        for (int j = 1; j < local_cols - 1; j++) {
            // Compute Jacobi iteration update using 4-point stencil
            u_new[i][j] = 0.25 * (u[i+1][j] + u[i-1][j] + 
                                   u[i][j+1] + u[i][j-1] - 
                                   h * h * f((start_row + i) * h, (start_col + j) * h));
            
            // Update maximum difference
            local_max_diff = fmax(local_max_diff, fabs(u_new[i][j] - u[i][j]));
        }
    }

    // Handle top ghost row (if a top neighbor exists)
    if (top_recv_buffer) {
        for (int j = 1; j < local_cols - 1; j++) {
            u_new[0][j] = 0.25 * (u[1][j] + top_recv_buffer[j] + 
                                   u[0][j+1] + u[0][j-1] - 
                                   h * h * f((start_row) * h, (start_col + j) * h));
            local_max_diff = fmax(local_max_diff, fabs(u_new[0][j] - u[0][j]));
        }
    }

    // Handle bottom ghost row (if a bottom neighbor exists)
    if (bottom_recv_buffer) {
        for (int j = 1; j < local_cols - 1; j++) {
            u_new[local_rows-1][j] = 0.25 * (bottom_recv_buffer[j] + u[local_rows-2][j] + 
                                              u[local_rows-1][j+1] + u[local_rows-1][j-1] - 
                                              h * h * f((start_row + local_rows - 1) * h, (start_col + j) * h));
            local_max_diff = fmax(local_max_diff, fabs(u_new[local_rows-1][j] - u[local_rows-1][j]));
        }
    }

    // Handle left ghost column (if a left neighbor exists)
    if (left_recv_buffer) {
        for (int i = 1; i < local_rows - 1; i++) {
            u_new[i][0] = 0.25 * (u[i+1][0] + u[i-1][0] + 
                                   u[i][1] + left_recv_buffer[i] - 
                                   h * h * f((start_row + i) * h, (start_col) * h));
            local_max_diff = fmax(local_max_diff, fabs(u_new[i][0] - u[i][0]));
        }
    }

    // Handle right ghost column (if a right neighbor exists)
    if (right_recv_buffer) {
        for (int i = 1; i < local_rows - 1; i++) {
            u_new[i][local_cols-1] = 0.25 * (u[i+1][local_cols-1] + u[i-1][local_cols-1] + 
                                              right_recv_buffer[i] + u[i][local_cols-2] - 
                                              h * h * f((start_row + i) * h, (start_col + local_cols - 1) * h));
            local_max_diff = fmax(local_max_diff, fabs(u_new[i][local_cols-1] - u[i][local_cols-1]));
        }
    }

    return local_max_diff;  // Return maximum difference for convergence check
}

int main(int argc, char *argv[]) {
    // Initialize MPI parallel environment
    MPI_Init(&argc, &argv);

    int rank, num_procs;  // Process rank and total number of processes
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);      // Get current process rank
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // Get total number of processes

    // Check command-line argument (grid size)
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: %s N\n", argv[0]);  // Print usage if incorrect arguments
        }
        MPI_Finalize();  // Finalize MPI
        return 1;
    }

    int N = atoi(argv[1]);           // Grid size from command line
    double h = 1.0 / (N - 1);         // Grid spacing
    double tolerance = 1e-6;          // Convergence tolerance
    int max_iterations = 10000;       // Maximum number of iterations

    // Compute 2D grid decomposition for parallel processes
    int grid_rows, grid_cols;
    compute_2d_grid(num_procs, &grid_rows, &grid_cols);

    // Determine local grid dimensions with load balancing
    int rows_per_proc = (N - 2) / grid_rows;
    int extra_rows = (N - 2) % grid_rows;

    int local_row = rank / grid_cols;  // Compute local row in the process grid
    int local_col = rank % grid_cols;  // Compute local column in the process grid
    
    // Calculate local grid dimensions, accounting for uneven distribution
    int local_rows = rows_per_proc + 2 + (local_row < extra_rows ? 1 : 0);
    int local_cols = rows_per_proc + 2 + (local_col < extra_rows ? 1 : 0);
    
    // Calculate global starting indices for this process
    int start_row = local_row * rows_per_proc + 
                    (local_row < extra_rows ? local_row : extra_rows) + 1;
    int start_col = local_col * rows_per_proc + 
                    (local_col < extra_rows ? local_col : extra_rows) + 1;

    // Allocate local grid with ghost points
    double **u = (double **)malloc(local_rows * sizeof(double *));
    double **u_new = (double **)malloc(local_rows * sizeof(double *));
    for (int i = 0; i < local_rows; i++) {
        u[i] = (double *)malloc(local_cols * sizeof(double));
        u_new[i] = (double *)malloc(local_cols * sizeof(double));
    }

    // Initialize local grid with boundary conditions
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < local_cols; j++) {
            int global_row = start_row + i - 1;
            int global_col = start_col + j - 1;

            // Set boundary conditions using exact solution
            if (global_row == 0 || global_row == N-1 || 
                global_col == 0 || global_col == N-1) {
                u[i][j] = u_new[i][j] = exact(global_row * h, global_col * h);
            } else {
                u[i][j] = u_new[i][j] = 0.0;  // Initialize interior points to zero
            }
        }
    }

    // Determine neighboring process ranks
    int top_rank = (local_row > 0) ? rank - grid_cols : MPI_PROC_NULL;
    int bottom_rank = (local_row < grid_rows - 1) ? rank + grid_cols : MPI_PROC_NULL;
    int left_rank = (local_col > 0) ? rank - 1 : MPI_PROC_NULL;
    int right_rank = (local_col < grid_cols - 1) ? rank + 1 : MPI_PROC_NULL;

    // Allocate communication buffers for ghost point exchange
    double *top_send_buffer = (double *)malloc(local_cols * sizeof(double));
    double *bottom_send_buffer = (double *)malloc(local_cols * sizeof(double));
    double *left_send_buffer = (double *)malloc(local_rows * sizeof(double));
    double *right_send_buffer = (double *)malloc(local_rows * sizeof(double));
    
    double *top_recv_buffer = (double *)malloc(local_cols * sizeof(double));
    double *bottom_recv_buffer = (double *)malloc(local_cols * sizeof(double));
    double *left_recv_buffer = (double *)malloc(local_rows * sizeof(double));
    double *right_recv_buffer = (double *)malloc(local_rows * sizeof(double));

    // Start timing
    double start_time = MPI_Wtime();
    int iteration = 0;
    double global_max_diff;

    // Main iterative solver loop
    do {
        // Prepare send buffers with boundary points
        for (int j = 0; j < local_cols; j++) {
            top_send_buffer[j] = u[1][j];
            bottom_send_buffer[j] = u[local_rows-2][j];
        }
        for (int i = 0; i < local_rows; i++) {
            left_send_buffer[i] = u[i][1];
            right_send_buffer[i] = u[i][local_cols-2];
        }

        // Exchange ghost points with neighboring processes
        // Top-bottom exchange
        MPI_Sendrecv(top_send_buffer, local_cols, MPI_DOUBLE, top_rank, 0,
                     bottom_recv_buffer, local_cols, MPI_DOUBLE, bottom_rank, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(bottom_send_buffer, local_cols, MPI_DOUBLE, bottom_rank, 1,
                     top_recv_buffer, local_cols, MPI_DOUBLE, top_rank, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Left-right exchange
        MPI_Sendrecv(left_send_buffer, local_rows, MPI_DOUBLE, left_rank, 2,
                     right_recv_buffer, local_rows, MPI_DOUBLE, right_rank, 2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(right_send_buffer, local_rows, MPI_DOUBLE, right_rank, 3,
                     left_recv_buffer, local_rows, MPI_DOUBLE, left_rank, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Perform Jacobi iteration with ghost point exchanges
        double local_max_diff = jacobi_iteration(u, u_new, local_rows, local_cols, 
                                                 N, N, h, start_row, start_col,
                                                 top_rank == MPI_PROC_NULL ? NULL : top_recv_buffer,
                                                 bottom_rank == MPI_PROC_NULL ? NULL : bottom_recv_buffer,
                                                 left_rank == MPI_PROC_NULL ? NULL : left_recv_buffer,
                                                 right_rank == MPI_PROC_NULL ? NULL : right_recv_buffer);

        // Copy u_new to u for next iteration
        for (int i = 0; i < local_rows; i++) {
            for (int j = 0; j < local_cols; j++) {
                u[i][j] = u_new[i][j];
            }
        }

        // Reduce maximum difference across all processes to check convergence
        MPI_Allreduce(&local_max_diff, &global_max_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        iteration++;
    } while (global_max_diff > tolerance && iteration < max_iterations);

    // Stop timing
    double end_time = MPI_Wtime();

    // Calculate global maximum error
    double local_max_error = 0.0;
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < local_cols; j++) {
            int global_row = start_row + i - 1;
            int global_col = start_col + j - 1;
            
            // Skip boundary points
            if (global_row == 0 || global_row == N-1 || 
                global_col == 0 || global_col == N-1) continue;
            
            // Compute maximum absolute error
            double error = fabs(u[i][j] - exact(global_row * h, global_col * h));
            local_max_error = fmax(local_max_error, error);
        }
    }

    // Reduce maximum error across all processes
    double global_max_error;
    MPI_Reduce(&local_max_error, &global_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Print results from rank 0 process
    if (rank == 0) {
        printf("Grid size: %d x %d\n", N, N);
        printf("Number of processes: %d\n", num_procs);
        printf("Grid decomposition: %d x %d\n", grid_rows, grid_cols);
        printf("Iterations: %d\n", iteration);
        printf("Max error: %e\n", global_max_error);
        printf("Execution time: %.6f seconds\n", end_time - start_time);
    }

    // Cleanup
    for (int i = 0; i < local_rows; i++) {
        free(u[i]);
        free(u_new[i]);
    }
    free(u);
    free(u_new);

    // Free communication buffers
    free(top_send_buffer);
    free(bottom_send_buffer);
    free(left_send_buffer);
    free(right_send_buffer);
    free(top_recv_buffer);
    free(bottom_recv_buffer);
    free(left_recv_buffer);
    free(right_recv_buffer);

    MPI_Finalize();
    return 0;
}