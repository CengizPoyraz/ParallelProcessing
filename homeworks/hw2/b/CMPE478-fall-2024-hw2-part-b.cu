// Include necessary Thrust headers for GPU operations
#include <thrust/device_vector.h>    // For GPU memory management
#include <thrust/host_vector.h>      // For CPU memory management
#include <thrust/transform.h>        // For parallel transformations
#include <thrust/functional.h>       // For basic functors like plus, minus
#include <thrust/extrema.h>          // For finding maximum values
#include <thrust/iterator/counting_iterator.h>  // For generating sequences
#include <thrust/tuple.h>            // For handling multiple values together
#include <cmath>                     // For mathematical functions
#include <iostream>                  // For input/output operations

// Absolute value to transform  
struct AbsoluteValue {
    __host__ __device__
    double operator()(double x) const {
        return fabs(x);
    }
};

// Functor to compute the exact solution u(x,y) = xy
struct ExactSolution {
    double h;      // Grid spacing
    int n;         // Number of grid points in one dimension
    
    // Constructor to initialize grid parameters
    ExactSolution(double _h, int _n) : h(_h), n(_n) {}
    
    // Operator to compute exact solution at each point
    __host__ __device__    // Function can run on both CPU and GPU
    double operator()(int idx) const {
        int i = idx / (n + 1);        // Calculate row index
        int j = idx % (n + 1);        // Calculate column index
        return (i * h) * (j * h);     // Return xy value at this point
    }
};

// Functor to perform one Jacobi iteration
struct JacobiIteration {
    double h;                         // Grid spacing
    int n;                           // Number of grid points in one dimension
    thrust::device_ptr<double> u;    // Pointer to current solution array
    
    // Constructor to initialize iteration parameters
    JacobiIteration(double _h, int _n, thrust::device_ptr<double> _u) 
        : h(_h), n(_n), u(_u) {}
    
    // Operator to perform one Jacobi iteration at each point
    __host__ __device__
    double operator()(int idx) const {
        // Convert linear index to 2D coordinates
        int i = idx / (n - 1) + 1;    // Row index (offset by 1 for interior points)
        int j = idx % (n - 1) + 1;    // Column index (offset by 1 for interior points)
        
        // Check if point is on boundary
        if (i == 0 || i == n || j == 0 || j == n) {
            return u[i * (n + 1) + j];  // Return boundary value unchanged
        }
        
        // Get values from neighboring points
        double up = u[(i - 1) * (n + 1) + j];      // Point above
        double down = u[(i + 1) * (n + 1) + j];    // Point below
        double left = u[i * (n + 1) + (j - 1)];    // Point to right
        double right = u[i * (n + 1) + (j + 1)];   // Point to left
        
        // Return average of neighbors (f(x,y) = 0 in this case)
        return (up + down + left + right) / 4.0;
    }
};

// Functor to compute error between numerical and exact solutions
struct ComputeError {
    double h;                         // Grid spacing
    int n;                           // Number of grid points in one dimension
    thrust::device_ptr<double> u;    // Pointer to current solution array
    
    // Constructor to initialize error computation parameters
    ComputeError(double _h, int _n, thrust::device_ptr<double> _u) 
        : h(_h), n(_n), u(_u) {}
    
    // Operator to compute error at each point
    __host__ __device__
    double operator()(int idx) const {
        // Convert linear index to 2D coordinates
        int i = idx / (n - 1) + 1;    // Row index
        int j = idx % (n - 1) + 1;    // Column index
        
        // Compute exact solution at this point
        double exact = (i * h) * (j * h);
        // Return absolute difference between numerical and exact solutions
        return fabs(u[i * (n + 1) + j] - exact);
    }
};

int main(int argc, char* argv[]) {
    // Check command line arguments
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " n num_threads\n";
        return 1;
    }

    // Parse command line arguments
    int n = atoi(argv[1]);           // Number of grid points in each dimension
    int num_threads = atoi(argv[2]); // Number of threads (not used in Thrust)
    double h = 1.0 / n;              // Grid spacing
    double tolerance = 1e-6;         // Convergence tolerance
    int max_iterations = 10000;      // Maximum number of iterations
    
    // Calculate grid dimensions
    int total_size = (n + 1) * (n + 1);       // Total number of grid points
    int interior_size = (n - 1) * (n - 1);    // Number of interior points
    
    // Allocate device memory for current and next iteration
    thrust::device_vector<double> d_u(total_size);
    thrust::device_vector<double> d_u_new(total_size);
    
    // Initialize boundary conditions using exact solution
    thrust::counting_iterator<int> index_begin(0);
    thrust::transform(index_begin,                    // Start of index range
                     index_begin + total_size,        // End of index range
                     d_u.begin(),                     // Output iterator
                     ExactSolution(h, n));           // Functor to compute values
    d_u_new = d_u;                                  // Copy initial values to new array
    
    // Start timing
    float start_time = clock();
    int iteration = 0;
    double max_diff;
    
    // Main iteration loop
    do {
        // Perform Jacobi iteration on interior points
        thrust::transform(
            thrust::make_counting_iterator(0),        // Start index
            thrust::make_counting_iterator(interior_size), // End index
            d_u_new.begin(),                         // Output iterator for new values
            JacobiIteration(h, n, d_u.data())        // Functor for iteration
        );
        
        // Compute maximum difference between iterations
        thrust::device_vector<double> d_diff(interior_size);
        thrust::transform(d_u_new.begin(),           // First input start
                         d_u_new.end(),             // First input end
                         d_u.begin(),               // Second input start
                         d_diff.begin(),            // Output start
                         thrust::minus<double>());   // Subtraction operator
        
        // Find maximum absolute difference
        max_diff = thrust::transform_reduce(
            d_diff.begin(),                         // Input start
            d_diff.end(),                          // Input end
            thrust::abs<double>(),                 // Transformation operator
            0.0,                                   // Initial value
            thrust::maximum<double>()              // Reduction operator
        );
        
        // Update solution for next iteration
        d_u = d_u_new;
        iteration++;
    } while (max_diff > tolerance && iteration < max_iterations);
    
    // Stop timing and calculate total time
    float end_time = clock();
    float total_time = (end_time - start_time) / CLOCKS_PER_SEC;
    
    // Calculate error compared to exact solution
    thrust::device_vector<double> d_error(interior_size);
    thrust::transform(
        thrust::make_counting_iterator(0),          // Start index
        thrust::make_counting_iterator(interior_size), // End index
        d_error.begin(),                           // Output iterator
        ComputeError(h, n, d_u.data())            // Error computation functor
    );
    
    // Find maximum error
    double max_error = thrust::reduce(
        d_error.begin(),                          // Input start
        d_error.end(),                           // Input end
        0.0,                                     // Initial value
        thrust::maximum<double>()                // Reduction operator
    );
    
    // Print results
    std::cout << "Grid size: " << n + 1 << " x " << n + 1 << std::endl;
    std::cout << "Iterations: " << iteration << std::endl;
    std::cout << "Max error: " << max_error << std::endl;
    std::cout << "Execution time: " << total_time << " seconds" << std::endl;
    
    return 0;
}