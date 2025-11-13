#include <stdio.h>

__device__ int calcularBrillo(int galaxia, int estrella)
{
  return (galaxia * galaxia + estrella * 5 + galaxia * estrella * 3) % 10;
}

__global__ void universo()
{
  int galaxia = blockIdx.x;
  int estrella = threadIdx.x;
  int brillo = calcularBrillo(galaxia, estrella);

  printf("Galaxia %d - Estrella %d -> Brillo: %d\n", galaxia, estrella, brillo);
}

int main()
{
  int galaxias = 2;
  int estrellasp_galaxia = 2;

  universo<<<galaxias, estrellasp_galaxia>>>();
  cudaDeviceSynchronize();
  return 0;
}