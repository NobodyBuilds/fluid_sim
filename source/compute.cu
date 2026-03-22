#include"compute.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include"D:\visual_studio\fluid_sim\struct.h"
#include<atomic>
#include<cuda.h>
#include <cuda_runtime.h>

#include<cuda_gl_interop.h>
#include<cuda_runtime_api.h>
#include<curand_kernel.h>
#include<device_launch_parameters.h>
#include<iostream>

#include<math_constants.h>
#include<math_functions.h>
#include"\visual_studio\fluid_sim\fluid_sim\settings.h"

#include <cub/cub.cuh>


#define BLOCKS(n) ((n + 255) / 256)
#define THREADS 256
#define MAX_PARTICLES_PER_CELL 256
//param settings;
__constant__ param s;

static void* d_sortTempStorage = nullptr;
static size_t sortTempBytes = 0;
static int* d_particleHash_alt = nullptr;   // double buffer
static int* d_particleIndex_alt = nullptr;
int HASH_TABLE_SIZE;     // 2^18 - adjust based on particle count
int d_count;
//device helpers
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
__host__ __device__ inline float4& operator+=(float4& a, const float4& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
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

extern "C" void syncSettings() {
	cudaMemcpyToSymbol(&s, &settings, sizeof(param));
}

// arrays to store particle data

//float4 for better cache
//compact array system with 124bit load 4X better cache loading
//uses 68bytes per particle ,before was 80bytes

float4* positions = nullptr;  //contains- x,y,x,particlemass

float4* velocity = nullptr; //vx,vy,vz,particlesize

float4* accelration = nullptr;//ax,ay,az,particletemp in heat

float4* fluidProp = nullptr;//density,neardensity,pressure,nearpressure

uchar4* color = nullptr;//contains rgb values and iscenter for future implemetation

float4* pridictedPos = nullptr; //for vorticity confinement and XSPH viscosity, not used in update kernel, only for force compuation, so we can keep it unsorted and avoid extra copy

//stroage for sorting
float4* positions_sorted = nullptr;
float4* velocity_sorted = nullptr;
float4* accelration_sorted = nullptr;
float4* fluidProp_sorted = nullptr;


extern"C" void initgpu(int count) {

    cudaMalloc(&positions, count * sizeof(float4));
    cudaMalloc(&pridictedPos, count * sizeof(float4));
    cudaMalloc(&velocity, count * sizeof(float4));
    cudaMalloc(&accelration, count * sizeof(float4));
    cudaMalloc(&fluidProp, count * sizeof(float4));
    cudaMalloc(&color, count * sizeof(uchar4));

    cudaMalloc(&positions_sorted, count * sizeof(float4));
    cudaMalloc(&velocity_sorted, count * sizeof(float4));
    cudaMalloc(&accelration_sorted, count * sizeof(float4));
    cudaMalloc(&fluidProp_sorted, count * sizeof(float4));
  

   
   
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: CUDA mem allocation failed: %s\n", cudaGetErrorString(err));
        return;
    }

}
extern "C" void freegpu() {
    cudaFree(positions);   cudaFree(positions_sorted);
    cudaFree(velocity);    cudaFree(velocity_sorted);
    cudaFree(accelration); cudaFree(accelration_sorted);
    cudaFree(fluidProp);   cudaFree(fluidProp_sorted);
    cudaFree(color);      
    cudaFree(pridictedPos);

};

 struct GLVertex {
     float px, py, pz;
     float radius;
     float cr, cg, cb, ca;
     float ox, oy;
     float wx, xy, xz;
 };

 static cudaGraphicsResource* g_vboResource = nullptr;

 extern "C" void registerGLBuffer(unsigned int vboId)
 {
     cudaError_t err = cudaGraphicsGLRegisterBuffer(
         &g_vboResource, vboId, cudaGraphicsMapFlagsWriteDiscard);
     if (err != cudaSuccess)
         printf("ERROR: cudaGraphicsGLRegisterBuffer: %s\n", cudaGetErrorString(err));
     else
         printf("INFO: GL VBO %u registered with CUDA\n", vboId);
 }

 extern "C" void unregisterGLBuffer()
 {
     if (g_vboResource) {
         cudaGraphicsUnregisterResource(g_vboResource);
         g_vboResource = nullptr;
     }
 }

 __global__ void packToVBOKernel(
     int n,
     const float4* __restrict__ pos,
     const float4* __restrict__ vel,
     const uchar4* __restrict__ color,
     GLVertex* vbo)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if (i >= n) return;
     float4 p = __ldg(&pos[i]);
     float4 v = __ldg(&vel[i]);
     uchar4 c = __ldg(&color[i]);

     float fpx = p.x, 
           fpy = p.y, 
           fpz = p.z;
     float rad = v.w;//size
     float fcr = c.x * (1.0f / 255.0f);
     float fcg = c.y * (1.0f / 255.0f);
     float fcb = c.z * (1.0f / 255.0f);

     // Matches the offsets used in the old CPU drawAll() loop
     const float ox[3] = { -1.0f,  3.0f, -1.0f };
     const float oy[3] = { -1.0f, -1.0f,  3.0f };

     int base = i * 3;
     for (int k = 0; k < 3; k++) {
         GLVertex& vtx = vbo[base + k];
         vtx.px = fpx;  vtx.py = fpy;  vtx.pz = fpz;
         vtx.radius = rad;
         vtx.cr = fcr;  vtx.cg = fcg;  vtx.cb = fcb;  vtx.ca = 1.0f;
         vtx.ox = ox[k]; vtx.oy = oy[k];
         vtx.wx = 0.0f;  vtx.xy = 0.0f;  vtx.xz = 0.0f;
     }
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

    return hash & (HASH_TABLE_SIZE - 1);
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
    HASH_TABLE_SIZE = 1;
    while (HASH_TABLE_SIZE < maxParticles * 2)
        HASH_TABLE_SIZE <<= 1;
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

    cudaMalloc(&d_particleHash_alt, maxParticles * sizeof(int));
    cudaMalloc(&d_particleIndex_alt, maxParticles * sizeof(int));

    // correct dry run — all nullptr, just getting the size
    cub::DeviceRadixSort::SortPairs(
        nullptr, sortTempBytes,
        (int*)nullptr, (int*)nullptr,
        (int*)nullptr, (int*)nullptr,
        maxParticles);

    cudaMalloc(&d_sortTempStorage, sortTempBytes);
    printf("CUB sort temp buffer: %zu bytes\n", sortTempBytes);
}
extern "C" void freeDynamicGrid() {
   // cudaFree(d_hashTable);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    cudaFree(d_particleHash);
    cudaFree(d_particleIndex);

    cudaFree(d_particleHash_alt);
    cudaFree(d_particleIndex_alt);
    cudaFree(d_sortTempStorage);
    d_particleHash_alt = nullptr;
    d_particleIndex_alt = nullptr;
    d_sortTempStorage = nullptr;
    sortTempBytes = 0;

    d_cellStart = nullptr;
    d_cellEnd = nullptr;
    d_particleHash = nullptr;
    d_particleIndex = nullptr;
}

__global__ void computeHashKernel(
    int numParticles,
    float cellSize,
    const float4* __restrict__ pos,
     int* particleHash,
    int* particleIndex,
    int hs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float x = pos[i].x;
    float y = pos[i].y;
    float z = pos[i].z;

    // Check for NaN
    if (isnan(x) || isnan(y) || isnan(z)) {
        particleHash[i] = 0xFFFFFFFF;  // Invalid hash
        particleIndex[i] = i;
        printf("WARNING nan positions\n");
        return;
    }


    // Compute hash
    unsigned int hash = getHashFromPos(x, y, z, cellSize, hs);

    particleHash[i] = hash;
    particleIndex[i] = i;  // Store original index
}


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
__global__ void reorderParticlesKernel(
    int n,
    const int* __restrict__ sortedIndex,   // d_particleIndex (after CUB sort)
    const float4* __restrict__ posIn,
    const float4* __restrict__ velIn,
    const float4* __restrict__ aclIn,
    const float4* __restrict__ fluidIn,
 
    float4* posOut,
    float4* velOut,
    float4* aclOut,
    float4* fluidOut
   )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int src = sortedIndex[i];   // where this particle CAME from in the original array

    posOut[i] = posIn[src];
    velOut[i] = velIn[src];
    aclOut[i] = aclIn[src];
    fluidOut[i] = fluidIn[src];
  
}

__global__ void clearActiveCellsKernel(
    int numParticles,
    const int* __restrict__ particleHash,  // sorted hashes from LAST frame
    int* cellStart,
    int* cellEnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    // Each thread clears its own hash bucket.
    // Duplicate writes (multiple particles same cell) are harmless — idempotent.
    unsigned int h = (unsigned int)particleHash[i];
    cellStart[h] = -1;
    cellEnd[h] = -1;
}


void buildDynamicGrid(
    int numParticles,
    float cellSize,
    const float4* __restrict__ pos
    
) {
    int blocks = (numParticles + THREADS - 1) / THREADS;
    if (numParticles <= 0) {
        printf("WARNING: buildDynamicGrid called with %d particles\n", numParticles);
        return;
    }

    
    static bool firstFrame = true;
    if (firstFrame) {
        cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
        cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));
        firstFrame = false;
    }
    else {
        // d_particleHash still holds last frame's sorted hashes — perfect
        clearActiveCellsKernel << <blocks, THREADS >> > (
            numParticles, d_particleHash, d_cellStart, d_cellEnd);
    }
  
    computeHashKernel << < blocks, THREADS >> > (numParticles,cellSize,pos,d_particleHash,d_particleIndex,HASH_TABLE_SIZE);
    
  
    //sorting
  
    cub::DeviceRadixSort::SortPairs(
        d_sortTempStorage, sortTempBytes,
        d_particleHash, d_particleHash_alt,
        d_particleIndex, d_particleIndex_alt,
        numParticles);
    std::swap(d_particleHash, d_particleHash_alt);
    std::swap(d_particleIndex, d_particleIndex_alt);

    if (d_particleHash == nullptr || d_particleIndex == nullptr) {
        printf("ERROR: Null pointers in grid sort!\n");
        return;
    }

  

    findCellBoundariesKernel << <blocks, THREADS >> > (numParticles, d_particleHash, d_cellStart, d_cellEnd);

    reorderParticlesKernel << <blocks, THREADS >> > (
        numParticles,
        d_particleIndex,        // tells us: sorted slot i came from original slot src
        positions, velocity, accelration, fluidProp,       // source
        positions_sorted, velocity_sorted, accelration_sorted,
        fluidProp_sorted);

    

}


__global__ void scatterForcesKernel(
    int n,
    const int* __restrict__ sortedIndex,    // d_particleIndex
    const float4* __restrict__ aclSorted,      // accelration_sorted — forces computed here
    float4* aclCanonical)   // accelration — updateKernel reads here
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int orig = sortedIndex[i];   // which original particle this sorted slot belongs to
    float4 a = aclSorted[i];

    // Only scatter xyz forces. Leave .w alone — updateKernel manages heat on canonical.
    aclCanonical[orig].x = a.x;
    aclCanonical[orig].y = a.y;
    aclCanonical[orig].z = a.z;
    // aclCanonical[orig].w is NOT touched here
}


//sph-functions



__global__ void computeDensity(
    int numParticles,
    float h,
    float cellSize,
    const float4* __restrict__ pos,
   
    float4* fluidProp,
    int hs,
    float rest_density,
    float h2,
   
    int* cellstart,
    int* cellend,
    int* particleindex,
	float K_, float k, float pollycoef6, float spikycoef, float sdensity, float ndensity, bool clamping
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float4 p = __ldg(&pos[i]);//is it good using ldg?? idk
    float xi = p.x;
    float yi = p.y;
    float zi = p.z;
   

  //  float rho = density[i];
    //float rhon = neardensity[i];
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

    
    int neighborCount = 0;
    int cellsChecked = 0;
    int cellsWithParticles = 0;

    // Debug for first particle
    bool Debug = (i == 0||i == 5||i == 10||i == 50||i == 100||i == 500||i == 1000);
 /*  bool debug = 0;*/

   float m_i = p.w;
  
   float rhon = m_i * ndensity;
   float rho = m_i * sdensity;
   float mindensity = rho * 0.5f;

   

    // Search 27 neighboring cells
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                /*int manhattanDist = abs(dx) + abs(dy) + abs(dz);
                if (manhattanDist > 2) continue;*/
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
					int j = k; // particleindex[k]; // index of neighbor particle in sorted array

                    if (j == i) continue;
                    float4 pj = __ldg(&pos[j]);
                    float dx_val = xi - pj.x;
                    float dy_val = yi - pj.y;
                    float dz_val = zi - pj.z;
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2) {
                        float invR = rsqrtf(r2+ 1e-12f);
                        float r = r2 * invR;
                        float v = h2 - r2;
                        
                       float vcube = v * v * v;
                        
                            float d = pollycoef6 * vcube;//precomputed pollycoef6
                        
                        float m_j = pj.w;//mass
                        rho += m_j * d;
                        float x = h - r;
                        float nd = spikycoef * x * x * x;
                        rhon += m_j * nd;
                        neighborCount++;

                    }
                }
            }
        }
    }
    float pp, n_p;
    if(clamping){
		pp = fmaxf(k * (rho - rest_density), 0.0f);// x=density)
    }
    else {
         pp = k * (rho - rest_density);
    }
     n_p = rhon * K_;//y= near density
   
    fluidProp[i] = make_float4(fmaxf(rho, mindensity), fmaxf(rhon, mindensity* 0.1f),pp, n_p);
    
}

__global__ void computePressure(
    int numParticles,
    float h,
    float cellSize,
    float k_,
    float restDensity,
    const float4* __restrict__ pos,
    float4*  acl,
    float4*  vel,
    float4* fluidProp,

   
    float st,
    int hs,float h2
    ,int* cellstart,int* cellend,
    int* particleIndex,float spikyGradv,float viscK,float pollycoef6,float minZ,float minX,float minY,float maxX,float maxY,int maxz,int pressuremode,float rep,float dst

) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float4 p = __ldg(&pos[i]);
    float xi = p.x;
    float yi = p.y;
    float zi = p.z;

    float3 force = { 0.0f, 0.0f, 0.0f };
    float wallForce = rep; // tune this
    float wallDst = dst * h;
    if (xi - minX < wallDst) force.x += wallForce * (1.0f - (xi - minX) / wallDst);
    if (maxX - xi < wallDst) force.x -= wallForce * (1.0f - (maxX - xi) / wallDst);
    if (yi - minY < wallDst) force.y += wallForce * (1.0f - (yi - minY) / wallDst);
    if (maxY - yi < wallDst) force.y -= wallForce * (1.0f - (maxY - yi) / wallDst);
    if (zi - minZ < wallDst) force.z += wallForce * (1.0f - (zi - minZ) / wallDst);
    if (maxz - zi < wallDst) force.z -= wallForce * (1.0f - (maxz - zi) / wallDst);

   

    float4 fluid = __ldg(&fluidProp[i]);
    float p_i = fluid.z;
    float pn_i= fluid.w;
   
    float3 visc = { 0.0f, 0.0f, 0.0f };
    float3 deltaV = { 0.0f,0.0f,0.0f };
    float4 v = __ldg(&vel[i]);
    float3 vi = make_float3(v.x, v.y, v.z);
    float epsilon = 0.3f;
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

   
    int neighborCount = 0;
    int cellsChecked = 0;
    int cellsWithParticles = 0;
   bool debug = 0;
    //bool debug = (i==0);
                        float rho_i =  fluid.x;
                        float nrho_i = fluid.y;

                        float pressuretermRho_i = p_i / (rho_i * rho_i);
                        float NpressuretermRho_i = pn_i / (nrho_i * nrho_i);

   

     
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
               /* int manhattanDist = abs(dx) + abs(dy) + abs(dz);
				if (manhattanDist > 2) continue;*///TRY 3 INSTEAD OF 2 TO FIX JITTERYNESS
              
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);

                int start = cellstart[hash];
                int end = cellend[hash];    
                if (start == -1) continue;

                cellsChecked++;

                if (start > 0) cellsWithParticles++;


                for (int k = start; k < end ; k++) {
					int j = k; // particleIndex[k]; // index of neighbor particle in sorted array

                    if (j == i) continue;
                    float4 pj = __ldg(&pos[j]);
                    float dx_val = xi - pj.x;
                    float dy_val = yi - pj.y;
                    float dz_val = zi - pj.z;
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2 && r2 > 1e-9f) {
                        float4 fluidj = __ldg(&fluidProp[j]);
                        float invR = rsqrtf(r2+ 1e-12f);
                        float r = r2 * invR;
                      
                        float p_j =  fluidj.z;
                        float np_j = fluidj.w;

                        float3 dir = { dx_val *invR, dy_val * invR, dz_val * invR };

                        neighborCount++;

                      /*  if (debug && neighborCount <= 3) {
                            printf("  Neighbor %d: dist=%.3f rho=%.6f p=%.6f\n",
                                j, r, density[j], p_j);
                        }*/
                        float rho_j =  fluidj.x;
                        float nrho_j = fluidj.y;
                        float x = h - r;
                        float gradW = spikyGradv *x*x;//precomputed -gradw
                       
                        
                      
                        float pressureterm = pressuretermRho_i + p_j / (rho_j * rho_j);
                        float npressureterm =NpressuretermRho_i + np_j / (nrho_j * nrho_j);
                       // float pressureterm = (p_i + p_j)/2;
                     
                        float m_j = pj.w;//particle mass stored in pos.w
                        if (pressuremode == 0) {
                            force += -m_j * pressureterm * gradW * dir;
                        }
                        else if (pressuremode == 1) {
                            force += m_j * pressureterm * gradW * dir;
                        }
                        force += -m_j* npressureterm * gradW * dir;
                        float4 v2 = __ldg(&vel[j]);
                        float3 vj = make_float3(v2.x, v2.y, v2.z);
                        float3 vij = vj - vi;

                        float lapW = viscK * x;
                        float viscosityCoeff = st;
                        visc += viscosityCoeff
                            * m_j
                            * vij
                            / rho_j
                            * lapW;
                        


                    }
                }
            }
        }
    }
    if (debug) {
        printf("pressure Total: checked %d cells, %d had particles, found %d neighbors\n",
            cellsChecked, cellsWithParticles, neighborCount);
      
    }
    //apply pressure
    float4 accl;

    accl.x = (force.x + visc.x );
    accl.y = (force.y + visc.y );
    accl.z = (force.z + visc.z );
    accl.w = 0.0f;
    acl[i]+= accl;
  
   
}

//testing commits 22223w63


//update
__global__ void updateKernel(float dt, int count, float cold, float4* pos,float4* vel ,float4* acl,
    float minX, float maxX, float minY, float maxY, float minZ, float maxZ, float restitution, float downf,int star,float heatMultipler,float heatDecay,uchar4* color,bool heat
) {
    // Vec3 acc_new;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float4 p  = __ldg(&pos[i]);
    float4 vl = __ldg(&vel[i]);
    float4 a  = __ldg(&acl[i]);

    if (heat) {
        float3 v = { vl.x,vl.y,vl.z };
        float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

        float heat = speed * 0.5f;
        // heat += mass[i];
        a.w += (heat * heatMultipler) * dt; //acl.w=heat;

        a.w *= expf(-heatDecay * dt);
        a.w = clamp(a.w, 0.0f, 100.0f);
        uchar4 col = color[i];   // load once to get col.w
        float t = clamp(a.w / 100.0f, 0.0f, 1.0f);
        t = pow(t, 0.95f);

        float R, G, B;

        if (t < 0.35f) {
            float tt = t / 0.25f;
            R = 0.0f;
            G = tt * 255.0f;
            B = 255.0f;
        }
        else if (t < 0.55f) {
            float tt = (t - 0.25f) / 0.25f;
            R = 0.0f;
            G = 255.0f;
            B = (1.0f - tt) * 255.0f;
        }
        else if (t < 0.85f) {
            float tt = (t - 0.5f) / 0.25f;
            R = tt * 255.0f;
            G = 255.0f;
            B = 0.0f;
        }
        else {
            float tt = (t - 0.75f) / 0.25f;
            R = 255.0f;
            G = (1.0f - tt) * 255.0f;
            B = 0.0f;
        }
        color[i] = make_uchar4(
            (unsigned char)clamp((int)R, 0, 255),
            (unsigned char)clamp((int)G, 0, 255),
            (unsigned char)clamp((int)B, 0, 255),
            col.w);
    }

      
    
    vl.x += a.x * dt;
    vl.y += a.y * dt;
    vl.z += a.z * dt;
    vl.z -= downf * dt;

    p.x += vl.x * dt;
    p.y += vl.y * dt;
    p.z += vl.z * dt;


    a.x = 0;
    a.y = 0;
    a.z = 0;



    const float friction = 0.1f;        // Surface friction coefficient
    const float damping = 0.1f;        // Velocity damping on contact


    // ---- X bounds ----
    if (p.x <= minX) {
        p.x = minX;
        float normalVel = vl.x;

        if (normalVel < 0.0f) {
            // Normal component (bounce)
            vl.x = -normalVel * restitution;

            // Tangential friction
            float tangentSpeed = sqrtf(vl.y * vl.y + vl.z * vl.z);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.y *= (1.0f - frictionMag / tangentSpeed);
                vl.z *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }
    else if (p.x >= maxX) {
        p.x = maxX;
        float normalVel = vl.x;

        if (normalVel  > 0.0f) {
            // Normal component (bounce)
            vl.x = -normalVel * restitution;

            // Tangential friction
            float tangentSpeed = sqrtf(vl.y * vl.y + vl.z * vl.z);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.y *= (1.0f - frictionMag / tangentSpeed);
                vl.z *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }

    // ---- Y bounds ----
    if (p.y <= minY) {
        p.y = minY;
        float normalVel = vl.y;

        if (normalVel < 0.0f) {
            vl.y = -normalVel * restitution;

            float tangentSpeed = sqrtf(vl.x * vl.x + vl.z * vl.z);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.x *= (1.0f - frictionMag / tangentSpeed);
                vl.z *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }
    else if (p.y >= maxY) {
             p.y = maxY;
        float normalVel = vl.y;

        if (normalVel > 0.0f) {
            vl.y = -normalVel * restitution;

            float tangentSpeed = sqrtf(vl.x * vl.x + vl.z* vl.z);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.x *= (1.0f - frictionMag / tangentSpeed);
                vl.z *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }

    // ---- Z bounds (floor/ceiling) ----
    if (p.z <= minZ) {
        p.z = minZ;
        float normalVel = vl.z;

        if (normalVel < 0.0f) {
            vl.z = -normalVel * restitution;

            // Extra friction on floor (prevents sliding)
            float tangentSpeed = sqrtf(vl.x * vl.x + vl.y * vl.y);
            if (tangentSpeed > 1e-6f) {
              //  float floorFriction = friction * 2.0f; // Stronger floor friction
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.x *= (1.0f - frictionMag / tangentSpeed);
                vl.y *= (1.0f - frictionMag / tangentSpeed);
            }

            //// Stop tiny vibrations (resting particles)
            //if (fabsf(vl.z) < 0.1f) {
            //    vl.z = 0.0f;
            //    vl.x *= damping;
            //    vl.y *= damping;
            //}
        }
    }
    else if (p.z >= maxZ) {
        p.z = maxZ;
        float normalVel = vl.z;

        if (normalVel > 0.0f) {
            vl.z = -normalVel * restitution;

            float tangentSpeed = sqrtf(vl.x * vl.x + vl.y * vl.y);
            if (tangentSpeed > 1e-6f) {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.x *= (1.0f - frictionMag / tangentSpeed);
                vl.y *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }

    

    pos[i] = p;
    vel[i] = vl;
    acl[i] = a;
}

__global__ void addparticles(int n, float h,
    float Size, float Mass,
    int R, int G, int B,
    float maxX, float maxY, float maxz,
    float minX, float minY, float minZ,
    float4* position, float4* velocity, float4* accelration, float4* fluidProp, uchar4* color,int flowcount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= flowcount) return;
    int k = n + i;

    float x, y, z;
    float particle_spacing = h * 1.1f;

    int particles_per_side = (int)ceil(cbrt((float)flowcount));
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
    float startZ = minZ + (maxz - minZ - gridSizeZ );  // Offset from top

    // Generate grid
    x = startX + ix * particle_spacing;
    y = startY + iy * particle_spacing;
    z = startZ - iz * particle_spacing;  // Start at maxZ with offset, go downward

    position[k].x = x;
    position[k].y = y;
    position[k].z = z;

    position[k].w = Mass;//w used for particle mass

    velocity[k].x = 0.0f;
    velocity[k].y = 0.0f;
    velocity[k].z = 0.0f;

    velocity[k].w = Size;// w used for particle size

    accelration[k].x = 0.0f;
    accelration[k].y = 0.0f;
    accelration[k].z = 0.0f;

    accelration[k].w = 0.0f;//heat

    fluidProp[k].x = 0.0f;//density
    fluidProp[k].y = 0.0f;//neardensity
    fluidProp[k].z = 0.0f;//pressure
    fluidProp[k].w = 0.0f;//nearpressure

    color[k].x = R;
    color[k].y = G;
    color[k].z = B;
    color[k].w = 0;

}
__global__ void registerKernel(int n,float h,
    float Size,float Mass,
    int R,int G,int B,
    float maxX,float maxY,float maxz,
    float minX,float minY,float minZ,
    float4* position,float4* velocity,float4* accelration,float4* fluidProp,uchar4* color

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
    float startZ = minZ + (maxz - minZ - gridSizeZ/2);  // Offset from top

    // Generate grid
    x = startX + ix * particle_spacing;
    y = startY + iy * particle_spacing;
    z = startZ - iz * particle_spacing;  // Start at maxZ with offset, go downward
    
    position[i].x = x;
    position[i].y = y;
    position[i].z = z;

    position[i].w = Mass;//w used for particle mass

    velocity[i].x = 0.0f;
    velocity[i].y = 0.0f;
    velocity[i].z = 0.0f;

    velocity[i].w = Size;// w used for particle size

    accelration[i].x = 0.0f;
    accelration[i].y = 0.0f;
    accelration[i].z = 0.0f;

    accelration[i].w = 0.0f;//heat
 
    fluidProp[i].x = 0.0f;//density
    fluidProp[i].y= 0.0f;//neardensity
    fluidProp[i].z = 0.0f;//pressure
    fluidProp[i].w = 0.0f;//nearpressure
    
    color[i].x = R;
    color[i].y = G;
    color[i].z = B;
    color[i].w = 0; //iscenter
}
extern "C" void registerBodies() {
    int Block = (settings.count + THREADS - 1) / THREADS;
    registerKernel << < Block, THREADS >> > (settings.count, settings.h, settings.size, settings.particleMass, settings.rc, settings.gc, settings.bc, settings.maxX, settings.maxY, settings.maxz, settings.minX, settings.minY, settings.minZ,
                                                 positions,velocity,accelration,fluidProp,color);  
}

__global__ void pridictedPositions(const int n, const float4* __restrict__ pos,const float4* __restrict__ vel,float4* ppos,const float dt){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

	float4 p = __ldg(&pos[i]);
	float4 v = __ldg(&vel[i]);
    
	ppos[i].x = p.x + v.x * dt; 
	ppos[i].y = p.y + v.y * dt;
	ppos[i].z = p.z + v.z * dt;
	ppos[i].w = p.w;//mass

}

extern "C" void computephysics(float dt) {
    int blocks = (settings.count + THREADS - 1) / THREADS;
    int totalBodies = settings.count;
    cudaError_t err;
    float d_cellsize = settings.h * 1.0f;//tweaak it gng
    
    float subdt = settings.fixedDt / settings.substeps;
	float deltaTime = dt / settings.substeps;
   
    cudaGraphicsUnmapResources(1, &g_vboResource, 0);
   
    if (settings.nopause) {
        for (int i = 0; i < settings.substeps; i++) {

            if (settings.colisionFun) {

                if (settings.pridectedpos) {
                    pridictedPositions << <blocks, THREADS >> > (totalBodies, positions, velocity, pridictedPos, subdt);
                    buildDynamicGrid(settings.count, d_cellsize, pridictedPos);
                }
                else {
                    buildDynamicGrid(settings.count, d_cellsize, positions);
                }



                computeDensity << <blocks, THREADS >> > (totalBodies, settings.h, d_cellsize, positions_sorted, fluidProp_sorted, HASH_TABLE_SIZE, settings.rest_density, settings.h2, d_cellStart, d_cellEnd, d_particleIndex, settings.nearpressure, settings.pressure, settings.pollycoef6, settings.spikycoef, settings.Sdensity, settings.ndensity, settings.pressureClamp);

                computePressure << <blocks, THREADS >> > (totalBodies, settings.h, d_cellsize, settings.pressure, settings.rest_density, positions_sorted, accelration_sorted, velocity_sorted, fluidProp_sorted, settings.visc, HASH_TABLE_SIZE, settings.h2, d_cellStart, d_cellEnd, d_particleIndex, settings.spikygradv, settings.viscosity, settings.pollycoef6, settings.minZ, settings.minX, settings.minY, settings.maxX, settings.maxY, settings.maxz, settings.pressureMode,settings.wallrep,settings.walldst);

                scatterForcesKernel << <blocks, THREADS >> > (totalBodies, d_particleIndex, accelration_sorted, accelration);
            }

            updateKernel << < blocks, THREADS >> > (deltaTime, settings.count, settings.cold, positions, velocity, accelration, settings.minX, settings.maxX, settings.minY, settings.maxY, settings.minZ, settings.maxz, settings.restitution, settings.downf, settings.star, settings.heatMultiplier, settings.cold, color, settings.heateffect);


        }


        if (settings.addParticle == true) {
            addparticles << <blocks, THREADS >> > (settings.count, settings.h, settings.size, settings.particleMass, settings.rc, settings.gc, settings.bc, settings.maxX, settings.maxY, settings.maxz, settings.minX, settings.minY, settings.minZ,
                positions, velocity, accelration, fluidProp, color, settings.flowcount);
            d_count += settings.flowcount;
            settings.samplecount = d_count;

        }
    }
   // cudaEventRecord(start);
    if (g_vboResource) {
        cudaGraphicsMapResources(1, &g_vboResource, 0);

        GLVertex* d_vbo = nullptr;
        size_t    nbytes = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &nbytes, g_vboResource);

        packToVBOKernel << <blocks, THREADS >> > (
            settings.count, positions,velocity,color, d_vbo);

       
    }
    
}