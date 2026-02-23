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
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include<math_constants.h>
#include<math_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>


#define BLOCKS(n) ((n + 255) / 256)
#define THREADS 256
#define MAX_PARTICLES_PER_CELL 256

;
int HASH_TABLE_SIZE;     // 2^18 - adjust based on particle count
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
// arrays to store particle data

//float4 for better cache
//compact array system with 124bit load 4X better cache loading
//uses 68bytes per particle ,before was 80bytes

float4* positions = nullptr;  //contains- x,y,x,particlemass

float4* velocity = nullptr; //vx,vy,vz,particlesize

float4* accelration = nullptr;//ax,ay,az,particletemp in heat

float4* fluidProp = nullptr;//density,neardensity,pressure,nearpressure

uchar4* color = nullptr;//contains rgb values and iscenter for future implemetation



extern"C" void initgpu(int count) {

    cudaMalloc(&positions, count * sizeof(float4));
    cudaMalloc(&velocity, count * sizeof(float4));
    cudaMalloc(&accelration, count * sizeof(float4));
    cudaMalloc(&fluidProp, count * sizeof(float4));
    cudaMalloc(&color, count * sizeof(uchar4));

   
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: CUDA mem allocation failed: %s\n", cudaGetErrorString(err));
        return;
    }

}
extern "C" void freegpu() {
    cudaFree(positions);
    cudaFree(velocity);
    cudaFree(accelration);
    cudaFree(fluidProp);
    cudaFree(color);

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
    while (HASH_TABLE_SIZE < maxParticles * 4)
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




void buildDynamicGrid(
    int numParticles,
    float cellSize,
    const float4* __restrict__ pos
    
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
    computeHashKernel << < blocks, THREADS >> > (numParticles,cellSize,pos,d_particleHash,d_particleIndex,HASH_TABLE_SIZE);
    
   /* cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Grid build failed: %s\n", cudaGetErrorString(err));
    }*/
    //sorting
    thrust::sort_by_key(thrust::device, d_particleHash, d_particleHash + numParticles, d_particleIndex);


    if (d_particleHash == nullptr || d_particleIndex == nullptr) {
        printf("ERROR: Null pointers in grid sort!\n");
        return;
    }

   /* err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Grid sort failed: %s\n", cudaGetErrorString(err));
    }*/

    findCellBoundariesKernel << <blocks, THREADS >> > (numParticles, d_particleHash, d_cellStart, d_cellEnd);
    /* err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: Grid bound failed: %s\n", cudaGetErrorString(err));
    }*/

}





//sph-functions

__global__ void self_pressure(int n,  float k, float rest_density,float4* fluidProp,float k_) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    bool debug = 0;
   // bool debug = (i == 0);
    float4 fluid = __ldg(&fluidProp[i]); 
    float p = k * (fluid.x - rest_density);// x=density
    float n_p = fluid.y * k_;//y= near density
    fluidProp[i] = make_float4(fluid.x, fluid.y, p, n_p);
    if (debug && p < 0)printf("pressure value from pressurefromdensity kernel is negative\n");
    if (debug && p > 0)printf("pressure value from pressurefromdensity kernel is postive\n");

}


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
    float K_,float pollycoef6,float spikycoef,float sdensity,float ndensity
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
   // bool debug = (i == 0);
   bool debug = 0;

   float m_i = p.w;
  
   float rhon = m_i * ndensity;
   float rho = m_i * sdensity;


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
                    float4 pj = __ldg(&pos[j]);
                    float dx_val = xi - pj.x;
                    float dy_val = yi - pj.y;
                    float dz_val = zi - pj.z;
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < h2) {
                        float invR = rsqrtf(r2+ 1e-12f);
                        float r = r2 * invR;
                        float v = h2 - r * r;
                        float vcube = v * v * v;
                        float d = pollycoef6 * vcube;//precomputed pollycoef6
                        float m_j = pj.w;//mass
                        rho += m_j * d;
                        float x = h - r;
                        float nd = spikycoef * x * x * x;
                        rhon += m_j * nd;
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

   
    fluidProp[i] = make_float4(fmaxf(rho, 1e-6f), fmaxf(rhon, 1e-6f), 0.0f, 0.0f);
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
    const float4* __restrict__ pos,
    float4*  acl,
    float4*  vel,
    float4* fluidProp,

   
    float st,
    int hs,float h2
    ,int* cellstart,int* cellend,
    int* particleIndex,float spikyGradv,float viscK,float pollycoef6,float minZ

) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles) return;

    float4 p = __ldg(&pos[i]);
    float xi = p.x;
    float yi = p.y;
    float zi = p.z;
    float floorBias = (zi <= minZ + 0.01f) ? h * 0.1f : 0.0f;
    zi += floorBias;

    float4 fluid = __ldg(&fluidProp[i]);
    float p_i = fluid.z;
    float pn_i= fluid.w;
    float3 force = { 0.0f, 0.0f, 0.0f };
    float3 visc = { 0.0f, 0.0f, 0.0f };
    float3 deltaV = { 0.0f,0.0f,0.0f };
    float4 v = __ldg(&vel[i]);
    float3 vi = make_float3(v.x, v.y, v.z);
    float epsilon = 0.3f;
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

   
    int neighborCount = 0;
    bool debug = 0;
   // bool debug = (i==0);
                        float rho_i =  fluid.x;
                        float nrho_i = fluid.y;

                        float pressuretermRho_i = p_i / (rho_i * rho_i);
                        float NpressuretermRho_i = pn_i / (nrho_i * nrho_i);

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
                        float m_j = pj.w;//particle mass stored in pos.w
                        force += -m_j * pressureterm * gradW * dir;
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
                        //combined pressure and xsph
                        /*float v = h2 - r * r;
                        float W = pollycoef6 * v*v*v;
                        float factor = (mass[j] / density[j]) * W;

                        deltaV.x += factor * (vx[j] - vx[i]);
                        deltaV.y += factor * (vy[j] - vy[i]);
                        deltaV.z += factor * (vz[j] - vz[i]);*/



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
    float4 accl;

    accl.x = (force.x + visc.x );
    accl.y = (force.y + visc.y );
    accl.z = (force.z + visc.z );
    accl.w = 0.0f;
    acl[i]+= accl;
    //apply xsph
   /* vx[i] += epsilon * deltaV.x;
    vy[i] += epsilon * deltaV.y;
    vz[i] += epsilon * deltaV.z;*/
   
}




//update
__global__ void updateKernel(float dt, int count, float cold, float4* pos,float4* vel ,float4* acl,
    float minX, float maxX, float minY, float maxY, float minZ, float maxZ, float restitution, float downf,int star,float heatMultipler,float heatDecay,float initial_r,float initial_b,float initial_g,uchar4* color
) {
    // Vec3 acc_new;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    float4 p  = __ldg(&pos[i]);
    float4 vl = __ldg(&vel[i]);
    float4 a  = __ldg(&acl[i]);


    float3 v = { vl.x,vl.y,vl.z };
    float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    float heat = speed * 0.5f;
    // heat += mass[i];
    a.w += (heat * heatMultipler) * dt; //acl.w=heat;

    a.w *= expf(-heatDecay * dt);
    a.w = clamp(a.w, 0.0f, 100.0f);
    float t = clamp(a.w / 100.0f, 0.0f, 1.0f);
    t = pow(t, 0.95f);
    float floatr = (float)initial_r;
    float floatb = (float)initial_b;
    float floatg = (float)initial_g;

    float R = lerp(floatr, 255.0f, t);
    float G = lerp(floatg, 0.0f, t);
    float B = lerp(floatb, 0.0f, t);

    uchar4 col = color[i];   // load once to get col.w
    color[i] = make_uchar4(
        (unsigned char)clamp((int)R, 0, 255),
        (unsigned char)clamp((int)G, 0, 255),
        (unsigned char)clamp((int)B, 0, 255),
        col.w);


      
    
    vl.x += a.x * dt;
    vl.y += a.y * dt;
    vl.z += a.z * dt;
    vl.z -= downf;

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
extern "C" void registerBodies(int n,float h,float Size,float Mass,int R,int G,int B, float maxX, float maxY, float maxz, float minX, float minY, float minZ ) {
    int Block = (n + THREADS - 1) / THREADS;
    registerKernel << < Block, THREADS >> > (n,h,Size,Mass,R,G,B,maxX,maxY,maxz,minX,minY,minZ,
                                                 positions,velocity,accelration,fluidProp,color);  
}

extern "C" void computephysics(int n, float dt, float h, float h2, float pollycoef6, float spikycoef, float gradv, float viscK, float sdensity, float ndensity, float rest_density, float pressure, float k_,
    float hmulti, float cold, float br, float bg, float bb, float maxX, float maxY, float maxZ, float minX, float minY, float minZ, float restitution, float downwardforce, int isstar, float visc) {
    int blocks = (n + THREADS - 1) / THREADS;
    int totalBodies = n;
    cudaError_t err;
    float d_cellsize = h * 1.0f;//tweaak it gng

    /* cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);*/

    float ms = 0;
    //sph//////////////
    //builg gird
   /* printf("count %d\n", n);
    cudaEventRecord(start);*/
    buildDynamicGrid(n, d_cellsize, positions);
    /* cudaEventRecord(stop);
     cudaEventSynchronize(stop);
     cudaEventElapsedTime(&ms, start, stop);
     printf("Grid build: %.2f ms\n", ms);*/

     //density
   //  cudaEventRecord(start);
    computeDensity << <blocks, THREADS >> > (totalBodies, h, d_cellsize, positions, fluidProp, HASH_TABLE_SIZE, rest_density, h2, d_cellStart, d_cellEnd, d_particleIndex, k_, pollycoef6, spikycoef, sdensity, ndensity);
    /*cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Density: %.2f ms\n", ms);*/

    //pfd
   // cudaEventRecord(start);
    self_pressure << <blocks, THREADS >> > (totalBodies, pressure, rest_density,fluidProp, k_);
    //pressure
    computePressure << <blocks, THREADS >> > (totalBodies, h, d_cellsize, pressure, rest_density,positions,accelration,velocity,fluidProp, visc, HASH_TABLE_SIZE, h2, d_cellStart, d_cellEnd, d_particleIndex, gradv, viscK, pollycoef6,minZ);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&ms, start, stop);
    // printf("Pressure: %.2f ms\n", ms);

     //intigration
    // cudaEventRecord(start);
    updateKernel << < blocks, THREADS >> > (dt, totalBodies, cold, positions,velocity,accelration, minX, maxX, minY, maxY, minZ, maxZ, restitution, downwardforce, isstar,  hmulti, cold, br, bb, bg, color);
    //////////////////////
   /* cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Update: %.2f ms\n", ms);*/

    if (g_vboResource) {
        cudaGraphicsMapResources(1, &g_vboResource, 0);

        GLVertex* d_vbo = nullptr;
        size_t    nbytes = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &nbytes, g_vboResource);

        packToVBOKernel << <blocks, THREADS >> > (
            n, positions,velocity,color, d_vbo);

        cudaGraphicsUnmapResources(1, &g_vboResource, 0);

        // updatearray(n,px,py,pz,size,r,g,b);
         /*cudaEventRecord(start);
         cudaEventRecord(stop);
         cudaEventSynchronize(stop);
         cudaEventElapsedTime(&ms, start, stop);
         printf("Update array: %.2f ms\n", ms);*/
         /*cudaEventDestroy(start);
         cudaEventDestroy(stop);*/
    }
}