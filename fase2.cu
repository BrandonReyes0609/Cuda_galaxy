#include <stdio.h>

__device__ int calcularBrillo(int galaxia, int estrella)
{
  return (galaxia * galaxia + estrella * 5 + galaxia * estrella * 3) % 10;
}

__global__ void universo(int galaxia)
{
  int estrella = threadIdx.x;
  __shared__ int brillos[1024]; // guardar el brillo de estrella

  int brillo = calcularBrillo(galaxia, estrella);
  brillos[estrella] = brillo;

  __syncthreads(); // sincronizarlos

  if (estrella == 0)
  {
    printf(">>> Galaxia %d completa:\n", galaxia);
    for (int i = 0; i < blockDim.x; i++)
    {
      printf("Estrella %d -> Brillo: %d\n", i, brillos[i]);
    }
  }
}

int main()
{
  int galaxias = 2;
  int estrellasp_galaxia = 4;

  for (int g = 0; g < galaxias; g++) // esperar a que galaxias terminen
  {
    universo<<<1, estrellasp_galaxia>>>(g);
    cudaDeviceSynchronize();
  }

  return 0;
}