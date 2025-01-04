#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

// Function to get exact solution (used for boundary conditions and error checking)
double exact(double x, double y, double z) {
    return x * y * z;
}

// Function to get f(x,y,z) based on the exact solution
double f(double x, double y, double z) {
    // For u(x,y,z) = xyz, the Laplacian (uxx + uyy + uzz) = 0
    return 0.0;
}

// Initialize grid with boundary conditions
void initialize_grid(double ***u, int n, double h) {
    #pragma omp parallel for collapse(3)
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            for (int k = 0; k <= n; k++) {
                // Check if point is on boundary
                if (i == 0 || i == n || j == 0 || j == n || k == 0 || k == n) {
                    u[i][j][k] = exact(i*h, j*h, k*h);
                } else {
                    u[i][j][k] = 0.0; // Initial guess for interior points
                }
            }
        }
    }
}

// Perform one Jacobi iteration
double jacobi_iteration(double ***u, double ***u_new, int n, double h) {
    double max_diff = 0.0;
    
    #pragma omp parallel
    {
        double local_max_diff = 0.0;
        
        #pragma omp for collapse(3)
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                for (int k = 1; k < n; k++) {
                    double new_value = (u[i+1][j][k] + u[i-1][j][k] +
                                      u[i][j+1][k] + u[i][j-1][k] +
                                      u[i][j][k+1] + u[i][j][k-1]) / 6.0 -
                                      (h * h * f(i*h, j*h, k*h)) / 6.0;
                    
                    u_new[i][j][k] = new_value;
                    double diff = fabs(new_value - u[i][j][k]);
                    local_max_diff = fmax(local_max_diff, diff);
                }
            }
        }
        
        #pragma omp critical
        {
            max_diff = fmax(max_diff, local_max_diff);
        }
    }
    
    return max_diff;
}

// Copy grid values
void copy_grid(double ***dest, double ***src, int n) {
    #pragma omp parallel for collapse(3)
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < n; j++) {
            for (int k = 1; k < n; k++) {
                dest[i][j][k] = src[i][j][k];
            }
        }
    }
}

// Calculate maximum error compared to exact solution
double calculate_error(double ***u, int n, double h) {
    double max_error = 0.0;
    
    #pragma omp parallel
    {
        double local_max_error = 0.0;
        
        #pragma omp for collapse(3)
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < n; j++) {
                for (int k = 1; k < n; k++) {
                    double error = fabs(u[i][j][k] - exact(i*h, j*h, k*h));
                    local_max_error = fmax(local_max_error, error);
                }
            }
        }
        
        #pragma omp critical
        {
            max_error = fmax(max_error, local_max_error);
        }
    }
    
    return max_error;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s n num_threads\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    double h = 1.0 / n;
    double tolerance = 1e-6;
    int max_iterations = 10000;
    
    omp_set_num_threads(num_threads);

    // Allocate 3D arrays
    double ***u = (double ***)malloc((n+1) * sizeof(double **));
    double ***u_new = (double ***)malloc((n+1) * sizeof(double **));
    for (int i = 0; i <= n; i++) {
        u[i] = (double **)malloc((n+1) * sizeof(double *));
        u_new[i] = (double **)malloc((n+1) * sizeof(double *));
        for (int j = 0; j <= n; j++) {
            u[i][j] = (double *)malloc((n+1) * sizeof(double));
            u_new[i][j] = (double *)malloc((n+1) * sizeof(double));
        }
    }

    // Initialize grid
    initialize_grid(u, n, h);
    initialize_grid(u_new, n, h);

    // Main iteration loop
    double start_time = omp_get_wtime();
    int iteration = 0;
    double max_diff;
    
    do {
        max_diff = jacobi_iteration(u, u_new, n, h);
        copy_grid(u, u_new, n);
        iteration++;
    } while (max_diff > tolerance && iteration < max_iterations);
    
    double end_time = omp_get_wtime();
    double total_time = end_time - start_time;

    // Calculate and print results
    double max_error = calculate_error(u, n, h);
    printf("Grid size: %d x %d x %d\n", n+1, n+1, n+1);
    printf("Number of threads: %d\n", num_threads);
    printf("Iterations: %d\n", iteration);
    printf("Max error: %e\n", max_error);
    printf("Execution time: %.6f seconds\n", total_time);

    // Free memory
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            free(u[i][j]);
            free(u_new[i][j]);
        }
        free(u[i]);
        free(u_new[i]);
    }
    free(u);
    free(u_new);

    return 0;
}