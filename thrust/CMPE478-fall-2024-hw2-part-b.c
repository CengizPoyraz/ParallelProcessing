#include <thrust/host_vector.h>        // Thrust host-side vector container
#include <thrust/device_vector.h>       // Thrust device-side vector container
#include <thrust/copy.h>                // Thrust copy algorithms
#include <thrust/transform.h>           // Thrust transform algorithms for parallel operations
#include <thrust/iterator/zip_iterator.h> // Iterator that combines multiple iterators
#include <thrust/functional.h>          // Thrust functional utilities
#include <cuda_runtime.h>               // CUDA runtime API
#include <iostream>                     // Input/output stream
#include <cmath>                        // Mathematical functions

// Exact solution function for boundary conditions and error checking
// __host__ __device__ allows function to be called from both CPU and GPU
__host__ __device__ double exact(double x, double y) {
    // For this problem, exact solution is simply the product of x and y
    return x * y;
}

// Source term function for the Poisson equation
// For u(x,y) = xy, the Laplacian (uxx + uyy) = 0, so source term is 0
__host__ __device__ double source_term(double x, double y) {
    return 0.0;
}

// Functor for performing Jacobi iteration on the grid
struct JacobiIterationFunctor {
    const double h;        // Grid spacing
    const int N;           // Grid size
    
    // Constructor to initialize grid spacing and size
    JacobiIterationFunctor(double grid_spacing, int grid_size) 
        : h(grid_spacing), N(grid_size) {}

    // Operator to compute new value for a grid point
    // Uses tuple to access neighboring points and global index
    __host__ __device__ 
    double operator()(thrust::tuple<double, double, double, double, double, double> t) const {
        // Extract values from the input tuple
        double center = thrust::get<0>(t);    // Current grid point
        double top = thrust::get<1>(t);       // Top neighboring point
        double bottom = thrust::get<2>(t);    // Bottom neighboring point
        double left = thrust::get<3>(t);      // Left neighboring point
        double right = thrust::get<4>(t);     // Right neighboring point
        int global_index = thrust::get<5>(t); // Global index of the point

        // Compute global (x,y) coordinates from linear index
        int row = global_index / N;
        int col = global_index % N;

        // Apply boundary conditions
        if (row == 0 || row == N-1 || col == 0 || col == N-1) {
            return exact(row * h, col * h);
        }

        // Jacobi iteration formula
        // Average of neighboring points minus source term
        return 0.25 * (top + bottom + left + right - 
                       h * h * source_term(row * h, col * h));
    }
};

// Functor for calculating error between numerical and exact solutions
struct ErrorFunctor {
    const double h;    // Grid spacing
    const int N;       // Grid size
    
    // Constructor to initialize grid spacing and size
    ErrorFunctor(double grid_spacing, int grid_size) 
        : h(grid_spacing), N(grid_size) {}

    // Compute error for a single grid point
    __host__ __device__
    double operator()(thrust::tuple<double, int> t) const {
        // Extract numerical solution value and global index
        double value = thrust::get<0>(t);
        int global_index = thrust::get<1>(t);

        // Compute global (x,y) coordinates
        int row = global_index / N;
        int col = global_index % N;

        // Skip boundary points (already known exact values)
        if (row == 0 || row == N-1 || col == 0 || col == N-1) {
            return 0.0;
        }

        // Compute absolute error
        return std::abs(value - exact(row * h, col * h));
    }
};

// Main Poisson solver class using Thrust and GPU
class PoissonSolverThrustGPU {
private:
    thrust::device_vector<double> grid;       // Current grid state
    thrust::device_vector<double> grid_new;   // Next iteration grid
    int N;                                    // Grid size
    double h;                                 // Grid spacing
    double tolerance;                         // Convergence tolerance
    int max_iterations;                       // Maximum iteration limit

public:
    // Constructor to initialize solver parameters
    PoissonSolverThrustGPU(int grid_size, double tolerance_val = 1e-6, 
                            int max_iter = 10000) 
        : N(grid_size), h(1.0 / (grid_size - 1)), 
          tolerance(tolerance_val), max_iterations(max_iter) {
        // Resize device vectors for grid storage
        grid.resize(N * N);
        grid_new.resize(N * N);
        
        // Initialize grid with boundary conditions
        initializeGrid();
    }

    // Initialize grid with boundary conditions
    void initializeGrid() {
        // Create a transform iterator to set initial grid values
        auto grid_begin = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            [this](int index) {
                // Compute row and column from linear index
                int row = index / N;
                int col = index % N;
                
                // Set boundary points to exact solution, interior to zero
                return (row == 0 || row == N-1 || col == 0 || col == N-1) ?
                    exact(row * h, col * h) : 0.0;
            }
        );
        
        // Copy initialized values to device grid
        thrust::copy(grid_begin, grid_begin + N * N, grid.begin());
        
        // Initialize new grid with same values
        grid_new = grid;
    }

    // Solve Poisson equation using Jacobi iteration
    double solve() {
        double max_diff = tolerance * 2; // Ensure first iteration
        int iterations = 0;

        while (max_diff > tolerance && iterations < max_iterations) {
            // Create zip iterator to access grid points and their neighborhoods
            auto zip_iter = thrust::make_zip_iterator(
                thrust::make_tuple(
                    grid.begin(),           // center points
                    grid.begin() + N,       // top row
                    grid.begin() + N * (N-1), // bottom row
                    grid.begin() + 1,       // left column
                    grid.begin() + N - 1,   // right column
                    thrust::make_counting_iterator(0) // global indices
                )
            );

            // Perform Jacobi iteration using parallel transform
            thrust::transform(
                zip_iter, 
                zip_iter + N * N,
                grid_new.begin(),
                JacobiIterationFunctor(h, N)
            );

            // Compute maximum difference between old and new grid
            auto diff_iter = thrust::make_transform_iterator(
                thrust::make_zip_iterator(
                    thrust::make_tuple(grid.begin(), grid_new.begin())
                ),
                [](thrust::tuple<double, double> t) {
                    // Compute absolute difference
                    return std::abs(thrust::get<0>(t) - thrust::get<1>(t));
                }
            );
            
            // Find maximum difference
            max_diff = *thrust::max_element(diff_iter, diff_iter + N * N);

            // Update grid for next iteration
            grid = grid_new;
            iterations++;
        }

        // Compute and return maximum error
        return computeError();
    }

    // Compute maximum error between numerical and exact solutions
    double computeError() {
        // Create iterator to compute errors for each grid point
        auto error_iter = thrust::make_transform_iterator(
            thrust::make_zip_iterator(
                thrust::make_tuple(grid.begin(), 
                                   thrust::make_counting_iterator(0))
            ),
            ErrorFunctor(h, N)
        );
        
        // Find and return maximum error
        return *thrust::max_element(error_iter, error_iter + N * N);
    }

    // Optional method to print or access results (stub implementation)
    void printResults() {
        // Copy results back to host for printing
        thrust::host_vector<double> host_grid = grid;
        // Implement printing logic if needed
    }
};

// Main function to demonstrate solver usage
int main() {
    // Set grid size
    int grid_size = 129; // 129x129 grid

    // Create Poisson solver instance
    PoissonSolverThrustGPU solver(grid_size);

    // Time the solve operation
    clock_t start = clock();
    double max_error = solver.solve();
    clock_t end = clock();

    // Print results
    std::cout << "Grid size: " << grid_size << " x " << grid_size << std::endl;
    std::cout << "Max Error: " << max_error << std::endl;
    std::cout << "Execution Time: " 
              << (double)(end - start) / CLOCKS_PER_SEC << " seconds" << std::endl;

    return 0;
}