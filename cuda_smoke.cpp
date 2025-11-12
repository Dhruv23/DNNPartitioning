#include <cuda_runtime.h>
#include <iostream>
int main(){
  int n=0;
  auto e = cudaGetDeviceCount(&n);
  std::cerr << "cudaGetDeviceCount -> " << e << " (" << cudaGetErrorString(e) << "), devices=" << n << "\n";
  return e==cudaSuccess?0:1;
}
