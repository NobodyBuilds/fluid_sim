#include"compute.h"
#include"D:\visual_studio\fluid_sim\struct.h"
#include<atomic>
#include<cuda.h>
#include <cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<curand_kernel.h>
#include<device_launch_parameters.h>
#include<iostream>
#include<math_constants.h>
#include<math_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


#define BLOCKS(n) ((n + 255) / 256)
#define THREADS 256
#define MAX_PARTICLES_PER_CELL 256
//

;
int HASH_TABLE_SIZE;     // 2^18 - adjust based on particle count


//__host__ __device__ inline float3 make_float3(float x, float y, float z) {
//    return { x, y, z };
//}
__host__ __device__ inline float clamp(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
__host__ __device__ inline float3 operator+(float3 a, float3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
__host__ __device__ inline float3 operator*(float3 a, float s) { return { a.x * s, a.y * s, a.z * s }; }
__host__ __device__ inline float3 operator/(float3 a, float s) { return { a.x / s, a.y / s, a.z / s }; }
__host__ __device__ inline float3 operator*(float s, float3 a) {
    return { a.x * s, a.y * s, a.z * s };
}
__host__ __device__ inline float3 operator-(float3 v) {
    return { -v.x, -v.y, -v.z };
}
__host__ __device__ inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__host__ __device__ inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
__host__ __device__ inline float3& operator*=(float3& v, float s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
__host__ __device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}
__host__ __device__ inline float length(float3 v) {
    return sqrt(dot(v, v));
}
__host__ __device__ inline float3 normalize(float3 v) {
    float l = length(v);
    return (l > 0.0f) ? v / l : float3{ 0,0,0 };
}
__device__ __forceinline__ float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float* dposx = nullptr;
float* dposy = nullptr;
float* dposz = nullptr;

float* daclx = nullptr; 
float* dacly = nullptr;
float* daclz = nullptr;

float* dvelx = nullptr;
float* dvely = nullptr;
float* dvelz = nullptr;

float* dSize = nullptr;
float* dMass = nullptr;
int* dIscenter = nullptr;

int* dr = nullptr;
int* dg = nullptr;
int* db = nullptr;

float* dHeat = nullptr;
float* dDensity = nullptr;
float* dnearDensity = nullptr;
float* dPressure = nullptr;
float* dnearPressure = nullptr;



extern"C" void initgpu(int count) {


    cudaMalloc(&dposx, count * sizeof(float));
    cudaMalloc(&dposy, count * sizeof(float));
    cudaMalloc(&dposz, count * sizeof(float));
    cudaMalloc(&daclx, count * sizeof(float));
    cudaMalloc(&dacly, count * sizeof(float));
    cudaMalloc(&daclz, count * sizeof(float));
   
    cudaMalloc(&dvelx, count * sizeof(float));
    cudaMalloc(&dvely, count * sizeof(float));
    cudaMalloc(&dvelz, count * sizeof(float));
  
    cudaMalloc(&dSize, count * sizeof(float));
    cudaMalloc(&dMass, count * sizeof(float));
    cudaMalloc(&dIscenter, count * sizeof(int));
    cudaMalloc(&dr, count * sizeof(int));
    cudaMalloc(&db, count * sizeof(int));
    cudaMalloc(&dg, count * sizeof(int));
  
    cudaMalloc(&dHeat, count * sizeof(float));
    cudaMalloc(&dDensity, count * sizeof(float));
    cudaMalloc(&dnearDensity, count * sizeof(float));
    cudaMalloc(&dPressure, count * sizeof(float));
    cudaMalloc(&dnearPressure, count * sizeof(float));


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: CUDA mem allocation failed: %s\n", cudaGetErrorString(err));
        return;
    }




}
extern "C" void freegpu() {
    cudaFree(dposx);
    cudaFree(dposy);
    cudaFree(dposz);
    cudaFree(daclx);
    cudaFree(dacly);
    cudaFree(daclz);
 
     cudaFree(dvelx );
     cudaFree(dvely );
     cudaFree(dvelz );
    
     cudaFree(dSize);
     cudaFree(dMass);
     cudaFree(dIscenter);
     cudaFree(dr);
     cudaFree(dg);
     cudaFree(db);
    
     cudaFree(dHeat);
     cudaFree(dDensity);
     cudaFree(dnearDensity);
     cudaFree(dPressure);
     cudaFree(dnearPressure);
};

extern"C" void updatearray(int count, float* px, float* py, float* pz, float* size, int* r, int* g, int* b) {
    cudaMemcpy(px, dposx, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(py, dposy, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(pz, dposz, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(size, dSize, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(r, dr, count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, db, count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(g, dg, count * sizeof(int), cudaMemcpyDeviceToHost);
 

   

}

/// ///////////////////////////
//sph

struct HashCell {
    int count;                               // Number of particles in this cell
    int particles[MAX_PARTICLES_PER_CELL];   // Particle indices
};
static HashCell* d_hashTable = nullptr;
static int* d_cellStart = nullptr;
static int* d_cellEnd = nullptr;
static int* d_particleHash = nullptr;
static int* d_particleIndex = nullptr;

__device__ __host__ inline unsigned int spatialHash(int ix, int iy, int iz, int HASH_TABLE_SIZE) {

    const unsigned int p1 = 73856093;
    const unsigned int p2 = 19349663;
    const unsigned int p3 = 83492791;

    unsigned int hash = ((unsigned int)ix * p1) ^
        ((unsigned int)iy * p2) ^
        ((unsigned int)iz * p3);

    return hash % HASH_TABLE_SIZE;
}

__device__ __host__ inline void getCell(float x, float y, float z,
    float cellSize,
    int& ix, int& iy, int& iz) {
    ix = (int)floorf(x / cellSize);
    iy = (int)floorf(y / cellSize);
    iz = (int)floorf(z / cellSize);
}

__device__ __host__ inline unsigned int getHashFromPos(float x, float y, float z,
    float cellSize, int hs) {
    int ix, iy, iz;
    getCell(x, y, z, cellSize, ix, iy, iz);
    return spatialHash(ix, iy, iz, hs);
}

extern "C" void initDynamicGrid(int maxParticles) {
    HASH_TABLE_SIZE = maxParticles * 2;
  //  size_t hashTableBytes = HASH_TABLE_SIZE * sizeof(HashCell);

    printf("\n=== INITIALIZING DYNAMIC GRID ===\n");
    printf("Hash table size: %d buckets\n", HASH_TABLE_SIZE);
    printf("Max particles per cell: %d\n", MAX_PARTICLES_PER_CELL);
   // printf("Memory for hash table: %.2f MB\n", hashTableBytes / (1024.0f * 1024.0f));
    printf("Max particles: %d\n", maxParticles);


   // cudaMalloc(&d_hashTable, hashTableBytes);
    cudaMalloc(&d_cellStart, HASH_TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_cellEnd, HASH_TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_particleHash, maxParticles * sizeof(int));
    cudaMalloc(&d_particleIndex, maxParticles * sizeof(int));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: CUDA allocation failed: %s\n", cudaGetErrorString(err));
        return;
    }
    // Initialize
  //  cudaMemset(d_hashTable, 0, hashTableBytes);
    cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
    cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));

    printf("Dynamic grid initialized: %d hash buckets, max %d particles/cell\n",
        HASH_TABLE_SIZE, MAX_PARTICLES_PER_CELL);
}
extern "C" void freeDynamicGrid() {
   // cudaFree(d_hashTable);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    cudaFree(d_particleHash);
    cudaFree(d_particleIndex);

    d_cellStart = nullptr;
    d_cellEnd = nullptr;
    d_particleHash = nullptr;
    d_particleIndex = nullptr;
}

__global__ void computeHashKernel(
    int numParticles,
    float cellSize,
    const float* px,
    const float* py,
    const float* pz,
     int* particleHash,
    int* particleIndex,
    int hs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float x = px[i];
    float y = py[i];
    float z = pz[i];

    // Check for NaN
    if (isnan(x) || isnan(y) || isnan(z)) {
        particleHash[i] = 0xFFFFFFFF;  // Invalid hash
        particleIndex[i] = i;
        return;
    }

    // Compute hash
    unsigned int hash = getHashFromPos(x, y, z, cellSize, hs);

    particleHash[i] = hash;
    particleIndex[i] = i;  // Store original index
}

//__device__ void bitonicSort(unsigned int* keys, int* values, int n) {
//    // For production, use thrust::sort_by_key or similar
//    // This is a placeholder - implement proper sorting
//}

__global__ void findCellBoundariesKernel(
    int numParticles,
     int* particleHash,
    int* cellStart,
    int* cellEnd
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    unsigned int hash = particleHash[i];
    unsigned int prevHash = (i > 0) ? particleHash[i - 1] : 0xFFFFFFFF;

    if (hash != prevHash) {
        // Start of new cell
        cellStart[hash] = i;
        if (i > 0) {
            cellEnd[prevHash] = i;
        }
    }

    if (i == numParticles - 1) {
        cellEnd[hash] = numParticles;
    }
}

__global__ void buildHashTableKernel(
    int numParticles,
    float cellSize,
    const float* px,
    const float* py,
    const float* pz,
    HashCell* hashTable, int hs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float x = px[i];
    float y = py[i];
    float z = pz[i];

    // Check for NaN
    if (isnan(x) || isnan(y) || isnan(z)) {
        if (i < 5) printf("WARNING: Particle %d has NaN position\n", i);
        return;
    }

    // Get hash
    int ix, iy, iz;
    getCell(x, y, z, cellSize, ix, iy, iz);


    unsigned int hash = spatialHash(ix, iy, iz, hs);

    /* if (i < 10) {
         printf("BUILD: Particle %d: pos=(%.3f, %.3f, %.3f) cell=(%d, %d, %d) hash=%u\n",
             i, x, y, z, ix, iy, iz, hash);
     }*/
     // Atomically add to hash table
    int slot = atomicAdd(&hashTable[hash].count, 1);

    if (slot < MAX_PARTICLES_PER_CELL) {
        hashTable[hash].particles[slot] = i;
        /* if (i < 10) {
             printf("  -> Inserted at slot %d in bucket %u\n", slot, hash);
         }*/
    }
    else {
        // Cell overflow - reduce count back
        atomicSub(&hashTable[hash].count, 1);
        if (slot == MAX_PARTICLES_PER_CELL) {
            printf("WARNING: Cell hash %u overflow (particle %d)\n", hash, i);
        }
    }
}




void buildDynamicGrid(
    int numParticles,
    float cellSize,
    const float* d_px,
    const float* d_py,
    const float* d_pz
) {
    if (numParticles <= 0) {
        printf("WARNING: buildDynamicGrid called with %d particles\n", numParticles);
        return;
    }

    /*printf("\n=== BUILDING DYNAMIC GRID ===\n");
    printf("Particles: %d, CellSize: %.3f\n", numParticles, cellSize);*/

    // Clear hash table
  
 //   cudaMemset(d_hashTable, 0, HASH_TABLE_SIZE * sizeof(HashCell));

    cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
    cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));

    // Build hash table directly
    int blocks = (numParticles + THREADS - 1) / THREADS;
    /*printf("Launching kernel: %d blocks, %d threads/block\n", blocks, THREADS);*/

    /*buildHashTableKernel << <blocks, THREADS >> > (
        numParticles, cellSize,
        d_px, d_py, d_pz,
        d_hashTable, HASH_TABLE_SIZE
        );*/
    computeHashKernel << < blocks, THREADS >> > (numParticles,cellSize,d_px,d_py,d_pz,d_particleHash,d_particleIndex,HASH_TABLE_SIZE);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Grid build failed: %s\n", cudaGetErrorString(err));
    }
    //sorting
    thrust::sort_by_key(thrust::device, d_particleHash, d_particleHash + numParticles, d_particleIndex);


    if (d_particleHash == nullptr || d_particleIndex == nullptr) {
        printf("ERROR: Null pointers in grid sort!\n");
        return;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Grid sort failed: %s\n", cudaGetErrorString(err));
    }

    findCellBoundariesKernel << <blocks, THREADS >> > (numParticles, d_particleHash, d_cellStart, d_cellEnd);
     err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Grid bound failed: %s\n", cudaGetErrorString(err));
    }

}



__device__ float smoothingkernel(float r, float h,float h9) {
    if (r >= 0.0f && r < h) {
        float polycoeff = 315.0f / (64.0f * CUDART_PI_F * h9);
        float v = h * h - r * r;
        return  v * v * v * polycoeff;

         /*float v = CUDART_PI * powf(h, 8) / 4;
         float value = fmaxf(0.0f, h * h - r * r);
         return value * value / v;*/
    }
    return 0.0f;
}
__device__ float spikyKernel(float r, float h) {
    if (r >= 0.0f && r < h) {
        float coeff = 15.0f / (CUDART_PI_F * powf(h, 6));
        float x = h - r;
        return coeff * x * x * x;
    }
    return 0.0f;
}
__device__ float densitykernel(float dst, float radius,float h9) {

    return smoothingkernel(dst, radius,h9);
    //  return spikyKernel(dst, radius);
}
__device__ float neardensitykernel(float r, float h) {
   return spikyKernel(r, h);
}
__device__ float spikyGrad(float r, float h,float h6) {
    if (r > 0.0f && r < h) {
        float v = h - r;
        return -45.0f / (CUDART_PI_F * h6) * v*v;


    }
    return 0.0f;
}
__device__ float PressureFromDensity(float density, float pressure, float rest_density)
{
    float p= pressure * (density - rest_density);

    
    return p;

    // Adiabatic index

          /* float p=  pressure * (powf(rest_density / density, gamma) - 1.0f);
   return fmaxf(0.0f,p);
           */



           //float ratio = density / rest_density;

           ////// Soft transition around rest density
           //if (ratio < 1.0f) {
           //    // Under compression: very soft pressure
           //    return pressure * 0.1f * (powf(ratio, gamma) - 1.0f);
           //}
           //else {
           //    // Compression: normal pressure
           //    return pressure * (powf(ratio, gamma) - 1.0f);
           //}
}
__device__ float nearpressurefromdensity(float d, float k) {
    return d * k;
}

__device__ float viscosityKernel(float r, float h,float h9) {
    float h2 = h * h;
    if (r < h) {
        return 45.0f / (CUDART_PI_F * (h2 * h2 * h2)) * (h - r);
    }
    return 0.0f;
   
}

//functions
__global__ void self_density(int n, float h, float* density, float* mass,float* neardensity,float sdensity,float ndensity) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bool debug = (i == 0);
    debug = 0;
    float r = 0.0f;
    
   
    float value = 0.0f;
    float nvalue = 0.0f;
    //density kernel
    if(r>=0.0f && r< h){

        value = sdensity; //precompute selfdensity based on h as r is zero      
    }
    //near density kernel
    if(r>=0.0f && r< h){

       
        nvalue = ndensity;//precompute near density based on h as r is zero 
    }

     float n_d = mass[i] * nvalue;
    float  d = mass[i] * value;
    neardensity[i] = n_d;
    density[i] = d;
    if (debug)printf("self density :%6f\n", d);
    if (debug)printf("near density :%6f\n", n_d);

}
__global__ void self_pressure(int n,  float k, float rest_density, float* density,float* pressure,float* nearpressure,float* neardensity,float k_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool debug = 0;
   // bool debug = (i == 0);
    float p = k * (density[i] - rest_density);;
    float n_p = neardensity[i] * k_;
    nearpressure[i] = n_p;
    pressure[i] = p;
    if (debug && p < 0)printf("pressure value from pressurefromdensity kernel is negative\n");
    if (debug && p > 0)printf("pressure value from pressurefromdensity kernel is postive\n");

}
//for optimization still in progrress
__global__ void sphCompute(
    int n, float h, float cellSize,
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float* ax, float* ay, float* az,
    float* density, float* pressure,
    float* mass, int* cellstart, int* cellend,
    int* particleindex, float K, float r_density,
    float visc, float h2, float h6, float h9, int hs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float xi = px[i];
    float yi = py[i];
    float zi = pz[i];
   
    float3 force={0.0f,0.0f,0.0f};
    float3 vforce = { 0.0f,0.0f,0.0f };
    float3 deltaV = { 0.0f,0.0f,0.0f };
    float epsilon = 0.3f;
    float p_i = pressure[i];
    float rho_i = density[i];
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

    // Search 27 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);

                int start = cellstart[hash];
                int end = cellend[hash];
                if (start == -1) continue;

                for (int k = start; k < end; k++) {
                    int j = particleindex[k];

                    if (j == i) continue;

                    float dx_val = xi - px[j];
                    float dy_val = yi - py[j];
                    float dz_val = zi - pz[j];
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2 && r2 > 1e-9f) {
                        float r = sqrtf(r2);
                        //density
                        float D= mass[j] * densitykernel(r, h, h9);
                        density[i] = D;
                        //pressure
                        float p_j = pressure[j];
                        float3 dir= { dx_val / r, dy_val / r, dz_val / r };
                        float rho_j = density[j];
                        float3 ri = make_float3(xi, yi, zi);
                        float3 rj = make_float3(px[j], py[j], pz[j]);
                        float3 rij = ri - rj;
                        float rl = length(rij);
                        float3 gradW =(rij/rl) * spikyGrad(r, h, h6);
                        float pressureterm = (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));

                        force += mass[j] * pressureterm * gradW;//* dir;
                        //viscosity
                        float3 vi = make_float3(vx[i], vy[i], vz[i]);
                        float3 vj = make_float3(vx[j], vy[j], vz[j]);
                        float3 vij = vj - vi;

                        float lapW = viscosityKernel(r, h,h9);
                        float viscosityCoeff = visc;
                        vforce += viscosityCoeff
                            * mass[j]
                            * vij
                            / density[j]
                            * lapW;
                        //xsph
                        float W = smoothingkernel(r, h, h9);
                        float factor = (mass[j] / density[j]) * W;

                        deltaV.x += factor * (vx[j] - vx[i]);
                        deltaV.y += factor * (vy[j] - vy[i]);
                        deltaV.z += factor * (vz[j] - vz[i]);

                    }

                }
            }
        }
    }
    ax[i] = (force.x + vforce.x) /*mass[i]*/;
    ay[i] = (force.y + vforce.y) /*mass[i]*/;
    az[i] = (force.z + vforce.z) /*mass[i]*/;
    vx[i] += epsilon * deltaV.x;
    vy[i] += epsilon * deltaV.y;
    vz[i] += epsilon * deltaV.z;

}

__global__ void computeDensity(
    int numParticles,
    float h,
    float cellSize,
    const float* px,
    const float* py,
    const float* pz,
    const float* mass,
    float* density,
    int hs,
    float rest_density,
    float h2,
   
    int* cellstart,
    int* cellend,
    int* particleindex,
    float K_,float* neardensity,float pollycoef6,float spikycoef
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float xi = px[i];
    float yi = py[i];
    float zi = pz[i];


    float rho = density[i];
    float rhon = neardensity[i];
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

    
    int neighborCount = 0;
    int cellsChecked = 0;
    int cellsWithParticles = 0;

    // Debug for first particle
   // bool debug = (i == 0);
   bool debug = 0;

    //if (debug) {
    //    printf("\n=== DENSITY: Particle %d ===\n", i);
    //    printf("Position: (%.3f, %.3f, %.3f)\n", xi, yi, zi);
    //    printf("Cell: (%d, %d, %d)\n", cx, cy, cz);
    //    printf("Search radius h: %.3f\n", h);
    //}

    // Search 27 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);

                int start = cellstart[hash];
                int end = cellend[hash];
                if (start == -1) continue;

                
                cellsChecked++;

                if (start > 0) cellsWithParticles++;

                /*  if (debug && count > 0) {
                     printf("  Neighbor cell (%d,%d,%d) hash=%u: %d particles\n",
                          cx + dx, cy + dy, cz + dz, hash, count);
                  }*/

                for (int k = start; k < end; k++) {
                    int j = particleindex[k];

                    if (j == i) continue;

                    float dx_val = xi - px[j];
                    float dy_val = yi - py[j];
                    float dz_val = zi - pz[j];
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2) {
                        float r = sqrtf(r2);
                        float v = h2 - r * r;
                        float vcube = v * v * v;
                        float d = pollycoef6 * vcube;//precomputed pollycoef6
                        rho += mass[j] * d;
                        float x = h - r;
                        float nd = spikycoef * x * x * x;
                        rhon += mass[j] * nd;
                        neighborCount++;

                        /* if (debug && neighborCount <= 5) {
                             printf("    Neighbor %d: dist=%.3f (within h=%.3f)\n", j, r, h);
                         }
                         if (debug) {
                             printf("Cell size: %.3f, Search radius h: %.3f\n", cellSize, h);
                             printf("Max possible neighbor distance in 27 cells: %.3f\n",
                                 cellSize * sqrtf(3.0f) * 1.5f);
                             printf("Ratio (should be > 1.0): %.3f\n",
                                 (cellSize * sqrtf(3.0f) * 1.5f) / h);
                         }
                     }*/
                    }
                }
            }
        }
    }

    density[i] = fmaxf(rho, 1e-6f);
    neardensity[i] = fmaxf(rhon, 1e-6f);
    /*if (i == 0) {
        float W_zero = densitykernel(0.0f, h);
        printf("\n=== DENSITY DEBUG ===\n");
        printf("h = %.6f\n", h);
        printf("mass = %.6f\n", mass[0]);
        printf("W(0,h) = %.6f\n", W_zero);
        printf("Self contribution = mass * W(0,h) = %.6f\n", mass[0] * W_zero);
        printf("Neighbors found: %d\n", neighborCount);
        printf("Final density: %.6f\n", density[0]);
        printf("rest_density expected: %.6f\n", rest_density);
        printf("===================\n");
    }*/

    if (debug) {
        printf("Total: checked %d cells, %d had particles, found %d neighbors\n",
           cellsChecked, cellsWithParticles, neighborCount);
        printf("h: %2f , restdensity: %6f, k: %2f\n", h, rest_density, K_);
        printf("final density after density function: %.6f\n", rho);
       // printf("=== END DENSITY ===\n\n");
    }
}

__global__ void computePressure(
    int numParticles,
    float h,
    float cellSize,
    float k_,
    float restDensity,
    float* pressure,
    
    float* px,
    float* py,
    float* pz,
    float* density,
    float* mass,
    float* ax,
    float* ay,
    float* az,
    float* vx,
    float* vy,
    float* vz,
   
    float st,
    int hs,float h2
    ,int* cellstart,int* cellend,
    int* particleIndex,float* nearpressure,float* neardensity,float spikyGradv,float viscK,float pollycoef6

) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float xi = px[i];
    float yi = py[i];
    float zi = pz[i];

    
    float p_i = pressure[i];
    float pn_i=nearpressure[i];
    float3 force = { 0.0f, 0.0f, 0.0f };
    float3 visc = { 0.0f, 0.0f, 0.0f };
    float3 deltaV = { 0.0f,0.0f,0.0f };
   
    float epsilon = 0.1f;
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

   
    int neighborCount = 0;
    bool debug = 0;
   // bool debug = (i==0);
                        float rho_i = density[i];
                        float nrho_i = neardensity[i];

    /* if (debug) {
         printf("\n=== PRESSURE: Particle %d ===\n", i);
         printf("Position: (%.3f, %.3f, %.3f)\n", xi, yi, zi);
         printf("Density: %.6f, Pressure: %.6f\n", rho_i, p_i);
     }*/

     
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);

                int start = cellstart[hash];
                int end = cellend[hash];
                if (start == -1) continue;

                for (int k = start; k < end ; k++) {
                    int j = particleIndex[k];

                    if (j == i) continue;

                    float dx_val = xi - px[j];
                    float dy_val = yi - py[j];
                    float dz_val = zi - pz[j];
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2 && r2 > 1e-9f) {
                        float r = sqrtf(r2);
                      
                        float p_j = pressure[j];
                        float np_j = nearpressure[j];

                        float3 dir = { dx_val / r, dy_val / r, dz_val / r };

                        neighborCount++;

                      /*  if (debug && neighborCount <= 3) {
                            printf("  Neighbor %d: dist=%.3f rho=%.6f p=%.6f\n",
                                j, r, density[j], p_j);
                        }*/
                        float rho_j = density[j];
                        float nrho_j = neardensity[j];
                        float x = h - r;
                        float gradW = spikyGradv *x*x;//precomputed -gradw
                        float ngradW = spikyGradv*x*x;
                      
                        float pressureterm = (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));
                        float npressureterm = (pn_i / (nrho_i * nrho_i) + np_j / (nrho_j * nrho_j));
                       // float pressureterm = (p_i + p_j)/2;
                       

                        // pressure accumulation difference with pos and neg signs
                        // setting where they act kinda like water
                        // 
                        // 
                        // 
                        // -sign settings
                        //r-d=0.1
                        //k=1000
                        //nk=2000
                        //mass=1
                        // as h increases particle act like thick fluid
                        //as rest density is increase fluid becomes thick and attracts
                        //high k result is thick fluid
                        //motion is smooth and very thick like honey


                        //no - sign settings 
                        // only - for near pressure, postive attracts
                        //r-d=1
                        //k=100
                        //nk=2000
                        //mass=10
                        //as h increaases particle becomes jittery they repel 
                        //as rest density is incrreases particle distance themselves from others repulsion dominates
                        //high k is very jittery ,even explodes,low k is kinda okkay but jitter at surface
                        force += -mass[j] * pressureterm * gradW * dir;
                        force += -mass[j] * npressureterm * ngradW * dir;
                        float3 vi = make_float3(vx[i], vy[i], vz[i]);
                        float3 vj = make_float3(vx[j], vy[j], vz[j]);
                        float3 vij = vj - vi;

                        float lapW = viscK * x;
                        float viscosityCoeff = st;
                        visc += viscosityCoeff
                            * mass[j]
                            * vij
                            / density[j]
                            * lapW;
                        //combined pressure and xsph
                        float v = h2 - r * r;
                        float W = pollycoef6 * v*v*v;
                        float factor = (mass[j] / density[j]) * W;

                        deltaV.x += factor * (vx[j] - vx[i]);
                        deltaV.y += factor * (vy[j] - vy[i]);
                        deltaV.z += factor * (vz[j] - vz[i]);



                    }
                }
            }
        }
    }
    if (debug) {
        printf(" pressure forces-\n x: %6f \n y: %6f \n z: %6f \n viscosity \n x: %6f \n y: %6f \n z %6f\n", force.x, force.y, force.z, visc.x, visc.y, visc.z);
        printf("epsilon: %3f\n", epsilon);
        printf("xsph values in deltaV \n x: %5f \n y: %5f \n y: %5f \n", deltaV.x,deltaV.y,deltaV.z);
    }
    //apply pressure
    ax[i] += (force.x + visc.x );
    ay[i] += (force.y + visc.y );
    az[i] += (force.z + visc.z );
    //apply xsph
    vx[i] += epsilon * deltaV.x;
    vy[i] += epsilon * deltaV.y;
    vz[i] += epsilon * deltaV.z;
   
}

__global__ void applyXSPH(
    int numParticles,
    float h,
    float cellSize,

    
    const float* px, const float* py, const float* pz,
    const float* density,
    const float* mass,
    float* vx, float* vy, float* vz, int hs,float h2,float h9, int* cellstart, int* cellend,
    int* particleIndex
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float xi = px[i];
    float yi = py[i];
    float zi = pz[i];
    float epsilon = 0.1f;
    float3 deltaV = { 0.0f, 0.0f, 0.0f };

    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

  

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);
                int start = cellstart[hash];
                int end = cellend[hash];

                for (int k = start; k < end ; k++) {
                    int j = particleIndex[k];
                    if (j == i) continue;

                    float dx_val = xi - px[j];
                    float dy_val = yi - py[j];
                    float dz_val = zi - pz[j];
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2) {
                        float r = sqrtf(r2);
                        float W = smoothingkernel(r, h,h9);
                        float factor = (mass[j] / density[j]) * W;

                        deltaV.x += factor * (vx[j] - vx[i]);
                        deltaV.y += factor * (vy[j] - vy[i]);
                        deltaV.z += factor * (vz[j] - vz[i]);
                    }
                }
            }
        }
    }
    /*if (i == 0) { 
        printf("epsilon: %3f\n", epsilon);
        printf("xsph values in deltaV \n x: %5f \n y: %5f \n y: %5f \n", deltaV.x); }*/
    vx[i] += epsilon * deltaV.x;
    vy[i] += epsilon * deltaV.y;
    vz[i] += epsilon * deltaV.z;
}

extern "C"
void stepsph(int totalBodies, float dt, float h, float pressure, float rest_density, float mx, float my, float mz, float maxx, float maxy, float maxz, float visc,float k_,float h2,
float pollycoef6,float spikycoef,float gradv,float viscK,float sdensity,float ndensity
    ) {


    

    cudaError_t err;

    float d_cellsize = h * 1.0f;//tweaak it gng

    buildDynamicGrid(totalBodies, d_cellsize, dposx, dposy, dposz);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Grid build failed: %s\n", cudaGetErrorString(err));
    }

    int B = (totalBodies + THREADS - 1) / THREADS;

    self_density << <B, THREADS >> > (totalBodies, h, dDensity, dMass,dnearDensity,sdensity,ndensity);
   
        computeDensity << <B, THREADS >> > (totalBodies, h, d_cellsize, dposx, dposy, dposz, dMass, dDensity, HASH_TABLE_SIZE, rest_density, h2, d_cellStart, d_cellEnd, d_particleIndex,k_,dnearDensity,pollycoef6,spikycoef);
       
    self_pressure << <B, THREADS >> > (totalBodies, pressure, rest_density, dDensity, dPressure,dnearPressure,dnearDensity,k_);
   
        computePressure << <B, THREADS >> > (totalBodies, h, d_cellsize, pressure, rest_density, dPressure, dposx, dposy, dposz, dDensity, dMass, daclx, dacly, daclz, dvelx, dvely, dvelz, visc, HASH_TABLE_SIZE, h2, d_cellStart, d_cellEnd, d_particleIndex,dnearPressure,dnearDensity,gradv,viscK,pollycoef6);
       

       // applyXSPH << <B, THREADS >> > (totalBodies, h, d_cellsize, dposx, dposy, dposz, dDensity, dMass, dvelx, dvely, dvelz, HASH_TABLE_SIZE, h2, h9, d_cellStart, d_cellEnd, d_particleIndex);
        

  
        //sphCompute << <B, THREADS >> > (totalBodies,h,d_cellsize,dposx,dposy,dposz,dvelx,dvely,dvelz,daclx,dacly,daclz,dDensity,dPressure,dMass,d_cellStart,d_cellEnd,d_particleIndex,pressure,rest_density,visc,h2,h6,h9,HASH_TABLE_SIZE);
    
        
    
   

}
/////////////////////////////////////////
//heating
__global__ void computeheat(int totalbodies, float dt, float hmulti, float cold, float* h, float* mass, int* r, int* g, int* b,  float* vx, float* vy, float* vz,int rr,int gg,int bbbb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalbodies) return;
   
    float3 v = { vx[i],vy[i],vz[i] };
    float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    float heat = speed * 0.5f;
   // heat += mass[i];
    h[i] += (heat * hmulti) * dt;

    h[i] *= expf(-cold * dt);
    h[i] = clamp(h[i], 0.0f, 100.0f);
    float t = clamp(h[i] / 100.0f, 0.0f, 1.0f);
    t = pow(t, 0.95f);
    float bbr = (float)rr;
    float bbb = (float)gg;
    float bbg = (float)bbbb;

    float R = lerp(bbr, 255.0f, t);
    float G = lerp(bbg, 0.0f, t);
    float B = lerp(bbb, 0.0f, t);

    r[i] = clamp((int)R, 0, 255);
    g[i] = clamp((int)G, 0, 255);
    b[i] = clamp((int)B, 0, 255);


}
extern"C"
void heating(int totalbodies, float dt, float hmulti, float cold,int r,int g,int b) {
    int B = (totalbodies + THREADS - 1) / THREADS;
    computeheat << <B, THREADS >> > (totalbodies, dt, hmulti, cold, dHeat, dMass, dr, dg, db, dvelx, dvely, dvelz,r,g,b);
  
}


//update
__global__ void updateKernel(float dt, int count, float cold,  float* px, float* py, float* pz, float* vx, float* vy, float* vz, float* ax, float* ay, float* az,
    float minX, float maxX, float minY, float maxY, float minZ, float maxZ, float restitution, float downf
) {
    // Vec3 acc_new;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;



    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += (az[i]) * dt;
    vz[i] -= downf;


    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;


    ax[i] = 0;
    ay[i] = 0;
    az[i] = 0;



    const float friction = 0.1f;        // Surface friction coefficient
    const float damping = 0.1f;        // Velocity damping on contact


    // ---- X bounds ----
    if (px[i] <= minX) {
        px[i] = minX;
        float normalVel = vx[i];

        if (normalVel < 0.0f) {
            // Normal component (bounce)
            vx[i] = -normalVel * restitution;

            // Tangential friction
            float tangentSpeed = sqrtf(vy[i] * vy[i] + vz[i] * vz[i]);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vy[i] *= (1.0f - frictionMag / tangentSpeed);
                vz[i] *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }
    else if (px[i] >= maxX) {
        px[i] = maxX;
        float normalVel = vx[i];

        if (normalVel > 0.0f) {
            vx[i] = -normalVel * restitution;

            float tangentSpeed = sqrtf(vy[i] * vy[i] + vz[i] * vz[i]);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vy[i] *= (1.0f - frictionMag / tangentSpeed);
                vz[i] *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }

    // ---- Y bounds ----
    if (py[i] <= minY) {
        py[i] = minY;
        float normalVel = vy[i];

        if (normalVel < 0.0f) {
            vy[i] = -normalVel * restitution;

            float tangentSpeed = sqrtf(vx[i] * vx[i] + vz[i] * vz[i]);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vx[i] *= (1.0f - frictionMag / tangentSpeed);
                vz[i] *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }
    else if (py[i] >= maxY) {
        py[i] = maxY;
        float normalVel = vy[i];

        if (normalVel > 0.0f) {
            vy[i] = -normalVel * restitution;

            float tangentSpeed = sqrtf(vx[i] * vx[i] + vz[i] * vz[i]);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vx[i] *= (1.0f - frictionMag / tangentSpeed);
                vz[i] *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }

    // ---- Z bounds (floor/ceiling) ----
    if (pz[i] <= minZ) {
        pz[i] = minZ;
        float normalVel = vz[i];

        if (normalVel < 0.0f) {
            vz[i] = -normalVel * restitution;

            // Extra friction on floor (prevents sliding)
            float tangentSpeed = sqrtf(vx[i] * vx[i] + vy[i] * vy[i]);
            if (tangentSpeed > 1e-6f) {
                float floorFriction = friction * 2.0f; // Stronger floor friction
                float frictionMag = fminf(floorFriction * fabsf(normalVel), tangentSpeed);
                vx[i] *= (1.0f - frictionMag / tangentSpeed);
                vy[i] *= (1.0f - frictionMag / tangentSpeed);
            }

            // Stop tiny vibrations (resting particles)
            if (fabsf(vz[i]) < 0.1f) {
                vz[i] = 0.0f;
                vx[i] *= damping;
                vy[i] *= damping;
            }
        }
    }
    else if (pz[i] >= maxZ) {
        pz[i] = maxZ;
        float normalVel = vz[i];

        if (normalVel > 0.0f) {
            vz[i] = -normalVel * restitution;

            float tangentSpeed = sqrtf(vx[i] * vx[i] + vy[i] * vy[i]);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vx[i] *= (1.0f - frictionMag / tangentSpeed);
                vy[i] *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }

}





extern "C"
void updatebodies(float dt, int count, float cold, float MAX_HEAT, float minx, float maxx, float miny, float maxy, float minz, float maxz, float res, float downf) {

    int B = (count + THREADS - 1) / THREADS;
    updateKernel << < B, THREADS >> > (dt, count, cold,  dposx, dposy, dposz, dvelx, dvely, dvelz, daclx, dacly, daclz, minx, maxx, miny, maxy, minz, maxz, res, downf);
   


}

__global__ void registerKernel(int n,float h,
    float Size,float Mass,
    int R,int G,int B,
    float maxX,float maxY,float maxz,
    float minX,float minY,float minZ,
    float* Px, float* Py, float* Pz,
    float* vx, float* vy, float* vz,
    float* ax, float* ay, float* az,
    float* size,
    float* mass,
    int* is,
    int* r, int* g, int* b,
    float* heat,
    float* density,
    float* pressure

) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x, y, z;
    float particle_spacing = h * 0.6f;

    int particles_per_side = (int)ceil(cbrt((float)n));
    int maxXcount = (int)((maxX - minX) / particle_spacing);
    int maxYcount = (int)((maxY - minY) / particle_spacing);
    int maxZcount = (int)((maxz - minZ) / particle_spacing);

    // Use cubic grid but respect physical limits
    int nx = fminf(particles_per_side, fmaxf(1, maxXcount));
    int ny = fminf(particles_per_side, fmaxf(1, maxYcount));
    int nz = fminf(particles_per_side, fmaxf(1, maxZcount));

   //int nx = max(1, int((maxX - minX) / particle_spacing));
   //int ny = max(1, int((maxY - minY) / particle_spacing));
   //int nz = ceil(float(n) / (nx * ny));

    
    //// Convert flattened index to 3D grid coordinates
    int ix = i % nx;
    int iy = (i / nx) % ny;
    int iz = i / (nx * ny);

    // Calculate grid dimensions
    float gridSizeX = (nx - 1) * particle_spacing;
    float gridSizeY = (ny - 1) * particle_spacing;
    float gridSizeZ = (nz - 1) * particle_spacing;

    // Calculate starting position with half particle spacing offset from edges
    float startX = minX + (maxX - minX - gridSizeX) * 0.5f;
    float startY = minY + (maxY - minY - gridSizeY) * 0.5f;
    float startZ = maxz - particle_spacing * 1.5f;  // Offset from top

    // Generate grid
    x = startX + ix * particle_spacing;
    y = startY + iy * particle_spacing;
    z = startZ - iz * particle_spacing;  // Start at maxZ with offset, go downward

    Px[i] = x;
    Py[i] = y;
    Pz[i] = z;

    vx[i] = 0.0f;
    vy[i] = 0.0f;
    vz[i] = 0.0f;

    ax[i] = 0.0f;
    ay[i] = 0.0f;
    az[i] = 0.0f;

    size[i] = Size;
    mass[i] = Mass;
    density[i] = 0.0f;
    pressure[i] = 0.0f;
    is[i] = 0;
    r[i] = R;
    g[i] = G;
    b[i] = B;

    heat[i] = 0.0f;
}
extern "C" void registerBodies(int n,float h,float Size,float Mass,int R,int G,int B, float maxX, float maxY, float maxz, float minX, float minY, float minZ ) {
    int Block = (n + THREADS - 1) / THREADS;
    registerKernel << < Block, THREADS >> > (n,h,Size,Mass,R,G,B,maxX,maxY,maxz,minX,minY,minZ,
                                                 dposx,dposy,dposz,dvelx,dvely,dvelz,daclx,dacly,daclz,dSize,dMass,dIscenter,dr,dg,db,dHeat,dDensity,dPressure
        );
   
}