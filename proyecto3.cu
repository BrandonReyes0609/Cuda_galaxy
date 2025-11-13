#include <stdio.h>
#include <cuda_runtime.h>

__device__ int calcularBrillo(int galaxia, int estrella)
{
  return (galaxia * galaxia + estrella * 5 + galaxia * estrella * 3) % 10;
}

__global__ void universo(int galaxia, float *promedios)
{
  int estrella = threadIdx.x;
  __shared__ int brillos[1024]; // guardar el brillo de estrella

  int brillo = calcularBrillo(galaxia, estrella);
  brillos[estrella] = brillo;

  __syncthreads(); // sincronizarlos

  if (estrella == 0)
  {
    float suma = 0;
    for (int i = 0; i < blockDim.x; i++)
    {
      suma += brillos[i];
    }
    float promedio = suma / blockDim.x;
    promedios[galaxia] = promedio;
    printf(">>> Galaxia %d - Brillo promedio: %.1f\n", galaxia, promedio);
  }
}

int main()
{
  int galaxias = 3;
  int estrellasp_galaxia = 4;

  float *d_promedios;
  float h_promedios[3];

  cudaMalloc(&d_promedios, galaxias * sizeof(float));

  for (int g = 0; g < galaxias; g++)
  {
    universo<<<1, estrellasp_galaxia>>>(g, d_promedios);
    cudaDeviceSynchronize();
  }

  cudaMemcpy(h_promedios, d_promedios, galaxias * sizeof(float), cudaMemcpyDeviceToHost);

  int galaxia_max = 0;
  float brillo_max = h_promedios[0];

  for (int g = 1; g < galaxias; g++)
  {
    if (h_promedios[g] > brillo_max)
    {
      brillo_max = h_promedios[g];
      galaxia_max = g;
    }
  }

  printf("Galaxia mas brillante: %d con brillo promedio: %.1f\n", galaxia_max, brillo_max);

  cudaFree(d_promedios);

  return 0;
}