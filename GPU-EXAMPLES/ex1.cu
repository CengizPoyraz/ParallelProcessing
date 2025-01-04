#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void MatAdd(float A[], float B[], float C[], int n) {

   int ij = threadIdx.x*n + threadIdx.y ; 

   if ( threadIdx.x < n && threadIdx.y < n  ) 
     C[ij] = A[ij] + B[ij] ; 
}  

int   n = 3 ; 
float h_A[3][3] = { {1,2,3}, {1,2,3}, {1,2,3} } ; 
float h_B[3][3] = { {1,2,3}, {1,2,3}, {1,2,3} } ; 
float h_C[3][3] ; 

/* Host code */
int main(int argc, char* argv[]) {

   size_t size;
   float *d_A, *d_B, *d_C ; 

   size = n*n*sizeof(float) ; 
   
   /* Allocate matrices in device memory */
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, size);
   cudaMalloc(&d_C, size);

   /* Copy matrices from host memory to device memory */
   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

   /* Invoke kernel using 1 block with n x n threads    */
   dim3 threadsPerBlock(n, n); 
   MatAdd<<<1, threadsPerBlock>>>(d_A, d_B, d_C, n);

   /* Wait for the kernel to complete */
   cudaThreadSynchronize();

   /* Copy result from device memory to host memory */
   cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);


   /* Free device memory */
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   for(int i=0 ; i < n ; i++) {
      for(int j=0 ; j < n ; j++) {
        printf("%g ",h_C[i][j]) ;
      }
      printf("\n") ;
   }

   return 0;
}  

