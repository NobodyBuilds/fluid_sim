#include "cuda.h"
#include <cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<iostream>
#include"compute.h"

#include<curand_kernel.h>
#include<math_constants.h>
#include<atomic>
#include<math_functions.h>
#include"D:\visual_studio\fluid_sim\struct.h"


#define BLOCKS(n) ((n + 255) / 256)
#define THREADS 256
#define MAX_PARTICLES_PER_CELL 256
//
int HASH_TABLE_SIZE ;     // 2^18 - adjust based on particle count


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
__host__ __device__ inline float3& operator-=(float3& a,  float3& b) {
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

float* dposx= nullptr;
float* dposy= nullptr;
float* dposz= nullptr;
float* daclx = nullptr;
float* dacly = nullptr;
float* daclz = nullptr;
float* dold_aclx= nullptr;
float* dold_acly= nullptr;
float* dold_aclz= nullptr;
float* dvelx= nullptr;
float* dvely= nullptr;
float* dvelz= nullptr;
float* dforcex=nullptr;
float* dforcey=nullptr;
float* dforcez=nullptr;
float* dSize = nullptr;
float* dMass = nullptr;
int* dIscenter = nullptr;
int* dr= nullptr;
int* dg= nullptr;
int* db= nullptr;
int* dbr= nullptr;
int* dbg= nullptr;
int* dbb= nullptr;
float* dHeat = nullptr;
float* dDensity = nullptr;
float* dPressure = nullptr;



    extern"C" void initgpu(int count) {
       
       
        cudaMalloc(&dposx, count * sizeof(float));
		cudaMalloc(&dposy, count * sizeof(float));
		cudaMalloc(&dposz, count * sizeof(float));
		cudaMalloc(&daclx, count * sizeof(float));
		cudaMalloc(&dacly, count * sizeof(float));
		cudaMalloc(&daclz, count * sizeof(float));
		cudaMalloc(&dold_aclx, count * sizeof(float));
		cudaMalloc(&dold_acly, count * sizeof(float));
		cudaMalloc(&dold_aclz, count * sizeof(float));
		cudaMalloc(&dvelx, count * sizeof(float));
		cudaMalloc(&dvely, count * sizeof(float));
		cudaMalloc(&dvelz, count * sizeof(float));
		cudaMalloc(&dforcex, count * sizeof(float));
		cudaMalloc(&dforcey, count * sizeof(float));
		cudaMalloc(&dforcez, count * sizeof(float));
		cudaMalloc(&dSize, count * sizeof(float));
		cudaMalloc(&dMass, count * sizeof(float));
		cudaMalloc(&dIscenter, count * sizeof(int));
		cudaMalloc(&dr, count * sizeof(int));
		cudaMalloc(&db, count * sizeof(int));
		cudaMalloc(&dg, count * sizeof(int));
		cudaMalloc(&dbr, count * sizeof(int));
		cudaMalloc(&dbb, count * sizeof(int));
		cudaMalloc(&dbg, count * sizeof(int));
		cudaMalloc(&dHeat, count * sizeof(float));
		cudaMalloc(&dDensity, count * sizeof(float));
		cudaMalloc(&dPressure, count * sizeof(float));

       
       
       
		
       
       
}
extern"C" void copyarray( int count, float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float* ax, float* ay, float* az,
    float* ox, float* oy, float* oz,
    float* fx, float* fy, float* fz,
    float* size,
    float* mass,
    int* is,
    int* r, int* g, int* b,
    int* br, int* bg, int* bb,
    float* heat,
    float* density,
    float* pressure) {

    
	cudaMemcpy(dposx,px , count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dposy, py, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dposz, pz, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dvelx, vx, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dvely, vy, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dvelz, vz, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(daclx, ax, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dacly, ay, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(daclz, az, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dold_aclx, ox, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dold_acly, oy, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dold_aclz, oz, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dforcex, fx, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dforcey, fy, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dforcez, fz, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dSize, size, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dMass, mass, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dIscenter, is, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dr, r, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dg, g, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(db, b, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dbr, br, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dbg, bg, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dbb, bb, count * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dHeat, heat, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dDensity, density, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dPressure, pressure, count * sizeof(float), cudaMemcpyHostToDevice);

}
extern"C" void updatearray(int count,float* px,float* py, float* pz,float* size,int* r,int* g, int* b,float* heat) {
	cudaMemcpy(px, dposx, count * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(py, dposy, count * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(pz, dposz, count * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(size, dSize, count * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(r, dr, count * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(b, db, count * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(g, dg, count * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(heat, dHeat, count * sizeof(float), cudaMemcpyDeviceToHost);
	

	
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

__device__ __host__ inline unsigned int spatialHash(int ix, int iy, int iz,int HASH_TABLE_SIZE) {
   
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
    float cellSize,int hs) {
    int ix, iy, iz;
    getCell(x, y, z, cellSize, ix, iy, iz);
    return spatialHash(ix, iy, iz,hs);
}

extern "C" void initDynamicGrid(int maxParticles) {
    HASH_TABLE_SIZE = maxParticles * 2;
    size_t hashTableBytes = HASH_TABLE_SIZE * sizeof(HashCell);

    printf("\n=== INITIALIZING DYNAMIC GRID ===\n");
    printf("Hash table size: %d buckets\n", HASH_TABLE_SIZE);
    printf("Max particles per cell: %d\n", MAX_PARTICLES_PER_CELL);
    printf("Memory for hash table: %.2f MB\n", hashTableBytes / (1024.0f * 1024.0f));
    printf("Max particles: %d\n", maxParticles);


    cudaMalloc(&d_hashTable, hashTableBytes);
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
    cudaMemset(d_hashTable, 0, hashTableBytes);
    cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
    cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));

    printf("Dynamic grid initialized: %d hash buckets, max %d particles/cell\n",
        HASH_TABLE_SIZE, MAX_PARTICLES_PER_CELL);
}
extern "C" void freeDynamicGrid() {
    cudaFree(d_hashTable);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    cudaFree(d_particleHash);
    cudaFree(d_particleIndex);
}

__global__ void computeHashKernel(
    int numParticles,
    float cellSize,
    const float* px,
    const float* py,
    const float* pz,
    unsigned int* particleHash,
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
    unsigned int hash = getHashFromPos(x, y, z, cellSize,hs);

    particleHash[i] = hash;
    particleIndex[i] = i;  // Store original index
}

//__device__ void bitonicSort(unsigned int* keys, int* values, int n) {
//    // For production, use thrust::sort_by_key or similar
//    // This is a placeholder - implement proper sorting
//}

__global__ void findCellBoundariesKernel(
    int numParticles,
    const unsigned int* particleHash,
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
    HashCell* hashTable,int hs
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


    unsigned int hash = spatialHash(ix, iy, iz,hs);

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

__global__ void getGridStatsKernel(
    const HashCell* hashTable,
    int* maxCellCount,
    int* totalOccupiedCells,int HASH_TABLE_SIZE
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HASH_TABLE_SIZE) return;

    int count = hashTable[i].count;
    if (count > 0) {
        atomicMax(maxCellCount, count);
        atomicAdd(totalOccupiedCells, 1);
    }
}


void printGridStats() {
    int* d_maxCount;
    int* d_occupiedCells;
    cudaMalloc(&d_maxCount, sizeof(int));
    cudaMalloc(&d_occupiedCells, sizeof(int));
    cudaMemset(d_maxCount, 0, sizeof(int));
    cudaMemset(d_occupiedCells, 0, sizeof(int));

    int blocks = (HASH_TABLE_SIZE + THREADS - 1) / THREADS;
    getGridStatsKernel << <blocks, THREADS >> > (d_hashTable, d_maxCount, d_occupiedCells,HASH_TABLE_SIZE);
    cudaDeviceSynchronize();

    int maxCount, occupiedCells;
    cudaMemcpy(&maxCount, d_maxCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&occupiedCells, d_occupiedCells, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n--- Grid Statistics ---\n");
    printf("Total hash buckets: %d\n", HASH_TABLE_SIZE);
    printf("Occupied buckets: %d (%.1f%%)\n",
        occupiedCells, 100.0f * occupiedCells / HASH_TABLE_SIZE);
    printf("Max particles in single cell: %d\n", maxCount);
    printf("Cell capacity: %d\n", MAX_PARTICLES_PER_CELL);

    if (maxCount >= MAX_PARTICLES_PER_CELL) {
        printf("WARNING: Some cells are at capacity! Consider:\n");
        printf("  - Increasing MAX_PARTICLES_PER_CELL\n");
        printf("  - Increasing cellSize\n");
        printf("  - Increasing HASH_TABLE_SIZE\n");
    }
   
    if (occupiedCells > HASH_TABLE_SIZE * 0.7f) {
        printf("WARNING: Hash table >70%% full, may have collisions\n");
        printf("  - Consider increasing HASH_TABLE_SIZE\n");
    }
     /*
    printf("--- End Statistics ---\n");*/

    cudaFree(d_maxCount);
    cudaFree(d_occupiedCells);
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
    cudaMemset(d_hashTable, 0, HASH_TABLE_SIZE * sizeof(HashCell));

    // Build hash table directly
    int blocks = (numParticles + THREADS - 1) / THREADS;
    /*printf("Launching kernel: %d blocks, %d threads/block\n", blocks, THREADS);*/

    buildHashTableKernel << <blocks, THREADS >> > (
        numParticles, cellSize,
        d_px, d_py, d_pz,
        d_hashTable,HASH_TABLE_SIZE
        );
    /*cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Grid build failed: %s\n", cudaGetErrorString(err));
    }
    
    else {
        printf("Grid build completed successfully\n");
    }*/

     //Print statistics
   /*printGridStats();
    printf("=== GRID BUILD COMPLETE ===\n\n");*/

    cudaDeviceSynchronize();
}



__device__ float smoothingkernel(float r, float h) {
    if (r >= 0.0f && r < h) {
		float polycoeff = 315.0f / (64.0f * CUDART_PI_F * powf(h, 9));
        float v = h * h - r * r;
        return polycoeff * v * v * v;

       /* float v = CUDART_PI * powf(h, 8) / 4;
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
__device__ float densitykernel(float dst, float radius) {

    return smoothingkernel(dst, radius);
   //  return spikyKernel(dst, radius);
}
__device__ float spikyGrad(float r, float h) {
    if (r > 0.0f && r <h) {
        float v = h - r;
        return -45.0f / (CUDART_PI_F * powf(h, 6)) * v * v;
        

    }
    return 0.0f;
}
__device__ float PressureFromDensity(float density,float pressure,float rest_density,float gamma)
{
     return pressure * (density - rest_density);

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
__device__ float cohesionKernel(float r, float h) {
    if (r >= 0.0f && r <= h / 2.0f) {
        float coeff = 32.0f / (CUDART_PI_F * powf(h, 9));
        return coeff * powf(h - r, 3) * r * r * r;
    }
    else if (r > h / 2.0f && r < h) {
        float coeff = 32.0f / (CUDART_PI_F * powf(h, 9));
        return coeff * (2.0f * powf(h - r, 3) * r * r * r - powf(h, 6) / 64.0f);
    }
    return 0.0f;
}
__device__ float cohesionGradient(float r, float h) {
    if (r > 0.0f && r < h) {
        float coeff = 32.0f / (CUDART_PI_F * powf(h, 9));
        if (r <= h / 2.0f) {
            return coeff * (3.0f * powf(h - r, 2) * r * r * r / r -
                3.0f * powf(h - r, 3) * r * r);
        }
        else {
            return coeff * (6.0f * powf(h - r, 2) * r * r * r / r -
                6.0f * powf(h - r, 3) * r * r);
        }
    }
    return 0.0f;
}
__device__ float viscosityKernel(float r, float h) {
    float h2 = h * h;
    if (r < h) {
        return 45.0f / (CUDART_PI_F * (h2*h2*h2)) * (h - r);
    }
    return 0.0f;
}
__device__ float artificialViscosity(int i, int j, float r, float h,float pressure,float rest_density,float gamma,float alpha_visc,float beta_visc,float* px,float* py,float* pz,float* vx,float* vy,float* vz,float* dens) {
    float3 ivel = { vx[i],vy[i],vz[i] };
    float3 jvel = { vx[j],vy[j],vz[j] };

    float3 ipos = { px[i],py[i],pz[i] };
    float3 jpos = { px[j],py[j],pz[j] };

    float3 vij = ivel - jvel;
    float3 rij = ipos - jpos;

    float vdotr = dot(vij, rij);
    if (vdotr >= 0.0f) return 0.0f; // particles moving apart

    float rho_avg = 0.5f * (dens[i] + dens[j]);
   float c_s = sqrtf(gamma * pressure / rho_avg);

    float mu = h * vdotr / (r * r + 0.01f * h * h);

    return (-alpha_visc * c_s * mu + beta_visc * mu * mu) / rho_avg;
}
//functions
__global__ void computeDensity(
    int numParticles,
    float h,
    float cellSize,
    const HashCell* hashTable,
    const float* px,
    const float* py,
    const float* pz,
    const float* mass,
    float* density,
    int hs,
    float rest_density
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float xi = px[i];
    float yi = py[i];
    float zi = pz[i];

   
    float rho = mass[i] * densitykernel(0.0f, h);

    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

    float h2 = h * h;
    int neighborCount = 0;
    int cellsChecked = 0;
    int cellsWithParticles = 0;

    // Debug for first particle
    bool debug = (i==0);
    //bool debug = 0;

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
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz,hs);

                const HashCell& cell = hashTable[hash];
                int count = cell.count;
                cellsChecked++;

                if (count > 0) cellsWithParticles++;

              /*  if (debug && count > 0) {
                   printf("  Neighbor cell (%d,%d,%d) hash=%u: %d particles\n",
                        cx + dx, cy + dy, cz + dz, hash, count);
                }*/

                for (int k = 0; k < count && k < MAX_PARTICLES_PER_CELL; k++) {
                    int j = cell.particles[k];

                    if (j == i) continue;

                    float dx_val = xi - px[j];
                    float dy_val = yi - py[j];
                    float dz_val = zi - pz[j];
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2) {
                        float r = sqrtf(r2);
                        rho += mass[j] * densitykernel(r, h);
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
  printf("Final density: %.6f\n", rho);
        printf("Total: checked %d cells, %d had particles, found %d neighbors\n",
            cellsChecked, cellsWithParticles, neighborCount);
        printf("=== END DENSITY ===\n\n");
    }
}

__global__ void computePressure(
    int numParticles,
    float h,
    float cellSize,
    float k_,
    float restDensity,
    float gamma,
    const HashCell* hashTable,
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
    float alpha_visc,
    float beta_visc,
    float st,
    int hs

) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float xi = px[i];
    float yi = py[i];
    float zi = pz[i];

    float rho_i = density[i];
    float p_i = PressureFromDensity(rho_i, k_, restDensity, gamma);

    float3 force = { 0.0f, 0.0f, 0.0f };
    float3 artificalvisc = { 0.0f, 0.0f, 0.0f };
    float3 cohesionForce = { 0, 0, 0 };  

    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

    float h2 = h * h;
    int neighborCount = 0;

    bool debug = 0;

   /* if (debug) {
        printf("\n=== PRESSURE: Particle %d ===\n", i);
        printf("Position: (%.3f, %.3f, %.3f)\n", xi, yi, zi);
        printf("Density: %.6f, Pressure: %.6f\n", rho_i, p_i);
    }*/

    // Search 27 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz,hs);

                const HashCell& cell = hashTable[hash];
                int count = cell.count;

                for (int k = 0; k < count && k < MAX_PARTICLES_PER_CELL; k++) {
                    int j = cell.particles[k];

                    if (j == i) continue;

                    float dx_val = xi - px[j];
                    float dy_val = yi - py[j];
                    float dz_val = zi - pz[j];
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2 && r2 > 1e-9f) {
                        float r = sqrtf(r2);
                        float rho_j = density[j];
                        float p_j = PressureFromDensity(rho_j, k_, restDensity, gamma);

                        float3 dir = { dx_val / r, dy_val / r, dz_val / r };

                        neighborCount++;

                         /*if (debug && neighborCount <= 3) {
                             printf("  Neighbor %d: dist=%.3f rho=%.6f p=%.6f\n",
                                 j, r, rho_j, p_j);
                         }*/

                        float gradW = spikyGrad(r, h);
                        float pressureterm = (p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j));
                        force += mass[j] * pressureterm * gradW * dir;
                        float pi_ij = artificialViscosity(i, j, r, h, k_, restDensity, gamma, alpha_visc, beta_visc, px, py, pz, vx, vy, vz, density);
                        artificalvisc += mass[j] * pi_ij * gradW * dir;
                        //xsph
                       /* float vv = (mass[j] / density[j]) * smoothingkernel(r, h);
                        float esp2 = 0.5f;
                        vx[i] += (vv * (vx[j] - vx[i])) * esp2;
                        vy[i] += (vv * (vy[j] - vy[i])) * esp2;
                        vz[i] += (vv * (vz[j] - vz[i])) * esp2;*/

                        float cohesion = cohesionGradient(r, h);
                        cohesionForce -= st * mass[j] * cohesion * dir;
                        



                    }
                }
            }
        }
    }

    ax[i] =(force.x + artificalvisc.x + cohesionForce.x) /*mass[i]*/;
    ay[i] =(force.y + artificalvisc.y + cohesionForce.y) /*mass[i]*/;
    az[i] =(force.z + artificalvisc.z + cohesionForce.z) /*mass[i]*/;

    if (debug) {
      printf("Neighbors: %d\n", neighborCount);
        printf("pressure accelrated Force: (%.6f, %.6f, %.6f)\n", ax[i], ay[i], az[i]);
        printf("artificalForce: (%.6f, %.6f, %.6f)\n", artificalvisc.x, artificalvisc.y, artificalvisc.z);
        printf("=== END PRESSURE ===\n\n");
    }
}

__global__ void applyXSPH(
    int numParticles,
    float h,
    float cellSize,
    
    const HashCell* hashTable,
    const float* px, const float* py, const float* pz,
    const float* density,
    const float* mass,
    float* vx, float* vy, float* vz,int hs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float xi = px[i];
    float yi = py[i];
    float zi = pz[i];
    float epsilon = 0.5f;
    float3 deltaV = { 0.0f, 0.0f, 0.0f };

    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

    float h2 = h * h;

    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz,hs);
                const HashCell& cell = hashTable[hash];
                int count = cell.count;

                for (int k = 0; k < count && k < MAX_PARTICLES_PER_CELL; k++) {
                    int j = cell.particles[k];
                    if (j == i) continue;

                    float dx_val = xi - px[j];
                    float dy_val = yi - py[j];
                    float dz_val = zi - pz[j];
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2) {
                        float r = sqrtf(r2);
                        float W = smoothingkernel(r, h);
                        float factor = (mass[j] / density[j]) * W;

                        deltaV.x += factor * (vx[j] - vx[i]);
                        deltaV.y += factor * (vy[j] - vy[i]);
                        deltaV.z += factor * (vz[j] - vz[i]);
                    }
                }
            }
        }
    }

    vx[i] += epsilon * deltaV.x;
    vy[i] += epsilon * deltaV.y;
    vz[i] += epsilon * deltaV.z;
}
__global__ void applyAdhesion(
    int numParticles,
    float* px, float* py, float* pz,
    float* vx, float* vy, float* vz,
    float minX, float maxX, float minY, float maxY, float minZ, float maxZ,
    float adhesion_strength,
    float h
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float adhesion_dist = h * 0.3f;  // Activate near walls

    // X walls
    if (px[i] - minX < adhesion_dist) {
        float dist = px[i] - minX;
        float factor = 1.0f - dist / adhesion_dist;
        vx[i] *= (1.0f - adhesion_strength * factor);
        vy[i] *= (1.0f - adhesion_strength * factor * 0.5f);
        vz[i] *= (1.0f - adhesion_strength * factor * 0.5f);
    }
    if (maxX - px[i] < adhesion_dist) {
        float dist = maxX - px[i];
        float factor = 1.0f - dist / adhesion_dist;
        vx[i] *= (1.0f - adhesion_strength * factor);
        vy[i] *= (1.0f - adhesion_strength * factor * 0.5f);
        vz[i] *= (1.0f - adhesion_strength * factor * 0.5f);
    }

    // Y walls (similar)
    if (py[i] - minY < adhesion_dist) {
        float dist = py[i] - minY;
        float factor = 1.0f - dist / adhesion_dist;
        vy[i] *= (1.0f - adhesion_strength * factor);
        vx[i] *= (1.0f - adhesion_strength * factor * 0.5f);
        vz[i] *= (1.0f - adhesion_strength * factor * 0.5f);
    }
    if (maxY - py[i] < adhesion_dist) {
        float dist = maxY - py[i];
        float factor = 1.0f - dist / adhesion_dist;
        vy[i] *= (1.0f - adhesion_strength * factor);
        vx[i] *= (1.0f - adhesion_strength * factor * 0.5f);
        vz[i] *= (1.0f - adhesion_strength * factor * 0.5f);
    }

    // Z floor (strongest adhesion)
    if (pz[i] - minZ < adhesion_dist) {
        float dist = pz[i] - minZ;
        float factor = 1.0f - dist / adhesion_dist;
        vz[i] *= (1.0f - adhesion_strength * factor * 2.0f);  // Stronger
        vx[i] *= (1.0f - adhesion_strength * factor);
        vy[i] *= (1.0f - adhesion_strength * factor);
    }
}
extern "C"
void stepsph(int totalBodies,float dt,float h,float pressure,float rest_density,float gamma,float aplha_visc,float beta_visc,float mx,float my,float mz,float maxx,float maxy,float maxz,float st,float dm) {
   

  

   

	float d_cellsize = h*1.5f;
	
    buildDynamicGrid(totalBodies, d_cellsize, dposx, dposy, dposz);
   

	int B = (totalBodies + THREADS - 1) / THREADS;
	
    
   computeDensity<<<B,THREADS>>>( totalBodies, h,d_cellsize,d_hashTable,dposx,dposy,dposz,dMass,dDensity,HASH_TABLE_SIZE,rest_density);


 
    computePressure<<<B,THREADS>>>(totalBodies, h,d_cellsize, pressure, rest_density, gamma, d_hashTable,dposx,dposy,dposz,dDensity,dMass,daclx,dacly,daclz,dvelx,dvely,dvelz,aplha_visc,beta_visc,st,HASH_TABLE_SIZE);
    applyXSPH << <B, THREADS >> > (totalBodies, h, d_cellsize, d_hashTable, dposx, dposy, dposz, dDensity, dMass, dvelx, dvely, dvelz,HASH_TABLE_SIZE);
   
    cudaDeviceSynchronize();

   
}
////////////////////////////////////////////
//heating
__global__ void computeheat(int totalbodies,float dt,float hmulti,float cold,float* h,float* mass,int* r,int* g,int* b,int* br,int* bg,int* bb,float* vx,float* vy,float* vz) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= totalbodies) return;
	
    float3 v = { vx[i],vy[i],vz[i]};
    float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    float heat = speed * 0.5f;
    heat += mass[i];
    h[i] += (heat * hmulti) * dt;

    h[i] *= expf(-cold * dt);
    h[i] = clamp(h[i], 0.0f, 100.0f);
    float t = clamp( h[i] / 100.0f, 0.0f, 1.0f);
    t = pow(t, 0.95f);
    float bbr=(float)br[i];
    float bbb=(float)bb[i];
    float bbg=(float)bg[i];

    float R = lerp(bbr, 255.0f, t);
    float G = lerp(bbg, 0.0f, t);
    float B = lerp(bbb, 0.0f, t);

	r[i] = clamp((int)R, 0, 255);
	g[i] = clamp((int)G, 0, 255);
	b[i] = clamp((int)B, 0, 255);


}
extern"C"
void heating(int totalbodies,float dt,float hmulti,float cold){
    int B = (totalbodies + THREADS - 1) / THREADS;
    computeheat << <B, THREADS >> > (totalbodies, dt, hmulti, cold,dHeat,dMass,dr,dg,db,dbr,dbg,dbb,dvelx,dvely,dvelz);
    cudaDeviceSynchronize();
}


//update
__global__ void updateKernel(float dt,int count,float cold,float MAX_HEAT,float* px,float* py,float* pz, float* vx, float* vy, float* vz, float* ax, float* ay, float* az, float* fx, float* fy, float* fz,float* mass,float* dens,float* pres,
float minX,float maxX,float minY,float maxY,float minZ,float maxZ,float restitution,float downf,float* ox,float* oy,float* oz
) {
   // Vec3 acc_new;
   
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
       
       
       

        vx[i] += ax[i] * dt;
        vy[i] += ay[i] * dt;
        vz[i] += (az[i] -downf) * dt;
      


        px[i] += vx[i] * dt;
        py[i] += vy[i] * dt;
        pz[i] += vz[i] * dt;

        fx[i] = 0;
        fy[i] = 0;
        fz[i] = 0;
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

   
    
    
}
extern "C"
void updatebodies(float dt,int count,float cold,float MAX_HEAT,float minx,float maxx,float miny,float maxy,float minz,float maxz,float res ,float downf) {
	
	int B = (count + THREADS - 1) / THREADS;
	updateKernel << < B, THREADS >> > (dt,  count, cold,MAX_HEAT,dposx,dposy,dposz,dvelx,dvely,dvelz,daclx,dacly,daclz,dforcex,dforcey,dforcez,dMass,dDensity,dPressure,minx,maxx,miny,maxy,minz,maxz,res,downf,dold_aclx,dold_acly,dold_aclz);
    cudaDeviceSynchronize();
   
    
}

