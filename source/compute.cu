#include "compute.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "D:\visual_studio\fluid_sim\struct.h"
#include <atomic>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iostream>

#include <math_constants.h>
#include <math_functions.h>
#include <vector>
#include "\visual_studio\fluid_sim\fluid_sim\settings.h"

#include <cub/cub.cuh>

#define BLOCKS(n) ((n + 255) / 256)
#define THREADS 256
#define MAX_PARTICLES_PER_CELL 256
// param settings;

static void *d_sortTempStorage = nullptr;
static size_t sortTempBytes = 0;
static int *d_particleHash_alt = nullptr; // double buffer
static int *d_particleIndex_alt = nullptr;
int HASH_TABLE_SIZE; // 2^18 - adjust based on particle count
int d_count;
size_t free_mem, total_mem;
// debug
__device__ int min_nb, max_nb, avg_nb = 0;
__device__ float min_Density, max_Density, avg_Density = 0;
__device__ float min_nearDensity, max_nearDensity, avg_nearDensity = 0;

// device helpers
__host__ __device__ inline float clamp(float x, float lo, float hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}
__host__ __device__ inline float3 operator+(float3 a, float3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ inline float3 operator*(float3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__host__ __device__ inline float3 operator+(float3 a, float s) { return {a.x + s, a.y + s, a.z + s}; }
__host__ __device__ inline float3 operator/(float3 a, float s) { return {a.x / s, a.y / s, a.z / s}; }
__host__ __device__ inline float3 operator*(float s, float3 a)
{
    return {a.x * s, a.y * s, a.z * s};
}
__host__ __device__ inline float3 operator-(float3 v)
{
    return {-v.x, -v.y, -v.z};
}
__host__ __device__ inline float3 &operator+=(float3 &a, const float3 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}
__host__ __device__ inline float4 &operator+=(float4 &a, const float4 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__host__ __device__ inline float3 &operator-=(float3 &a, const float3 &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}
__host__ __device__ inline float3 &operator*=(float3 &v, float s)
{
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
__host__ __device__ inline float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline float3 cross(Vec3 a, Vec3 b)
{
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}
__host__ __device__ inline float length(float3 v)
{
    return sqrt(dot(v, v));
}
__host__ __device__ inline float3 normalize(float3 v)
{
    float l = length(v);
    return (l > 0.0f) ? v / l : float3{0, 0, 0};
}
__device__ __forceinline__ float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

// arrays to store particle data

// float4 for better cache
// compact array system with 124bit load 4X better cache loading
// uses 68bytes per particle ,before was 80bytes

float4 *positions = nullptr; // contains- x,y,x,density

float4 *velocity = nullptr; // vx,vy,vz,neardensity

float4 *accelration = nullptr; // ax,ay,az,heat in heat

int *ncount = nullptr; // saves nighbor count per particle to apply airdrag later

// stroage for sorting
float4 *positions_sorted = nullptr;
float4 *velocity_sorted = nullptr;
float4* accelration_sorted = nullptr;
float3* xsph_delta = nullptr;



// constant struct for kernels to reduce memory occupancy and register load, also to avoid passing too many parameters to kernels which can cause register spilling and performance 

__constant__ data d;
    data h;

    //helpers
	int* d_cob = nullptr;

    //vs
    float* voxelgrid = nullptr;
    
extern "C" void syncstruct() {
    h.dt = settings.fixedDt;
    h.downf = settings.gravityforce;
	h.particlemass = settings.particleMass;
	h.minX = settings.minX;
	h.minY = settings.minY;
	h.minZ = settings.minZ;
	h.maxX = settings.maxX;
	h.maxY = settings.maxY;
	h.maxZ = settings.maxz;
	h.mx = settings.mx;
	h.nx = settings.nx;
	h.my = settings.my;
	h.ny = settings.ny;
	h.mz = settings.mz;
	h.nz = settings.nz;
	h.nearrestdensity = settings.nearRestDensity;
	h.restitution = settings.restitution;
	h.h = settings.h;
	h.spacing = settings.spacing;
	h.ndensity = settings.ndensity;
	h.sdensity = settings.Sdensity;
	h.h2 = settings.h2;
	h.pollycoef6 = settings.pollycoef6;
	h.spikycoef = settings.spikycoef;
	h.rep = settings.wallrep;
	h.dst = settings.walldst;
	h.pressure = settings.pressure;
	h.nearpressure = settings.nearpressure;
	h.restDensity = settings.rest_density;
	h.spikyGradv = settings.spikygradv;
	h.viscK = settings.visc;
	h.viscstrength = settings.viscosity;
	h.viscK = settings.visc;
    h.count = settings.count;
	h.flowcount = settings.flowcount;
	h.epsilon = settings.epsilon;
    h.scale = settings.scale;
    h.neargrad = settings.neargrad;
   
	h.densityoffset = settings.densityoffset;
    h.stepsize = settings.stepsize;
	h.depth = settings.depth;
    h.extinctionR = settings.extinctionR;
    h.extinctionG = settings.extinctionG;
    h.extinctionB = settings.extinctionB;
	h.variationStrength = settings.variationStrength;
	h.tilesize = settings.tilesize;
	h.col1 = make_float3(settings.color1R, settings.color1G, settings.color1B);
	h.col2 = make_float3(settings.color2R, settings.color2G, settings.color2B);
	h.col3 = make_float3(settings.color3R, settings.color3G, settings.color3B);
	h.col4 = make_float3(settings.color4R, settings.color4G, settings.color4B);
	h.centerx = (settings.floorbounx+(settings.floorboun_x))/2.0f;
	h.centerz = (settings.floorbounz+(settings.floorboun_z))/2.0f;
    
	cudaMemcpyToSymbol(d, &h, sizeof(data));

}

extern "C" bool initgpu(int count)
{
   
    cudaMalloc(&positions, count * sizeof(float4));
    cudaMalloc(&velocity, count * sizeof(float4));
    cudaMalloc(&accelration, count * sizeof(float4));

    cudaMalloc(&positions_sorted, count * sizeof(float4));
    cudaMalloc(&velocity_sorted, count * sizeof(float4));
	cudaMalloc(&accelration_sorted, count * sizeof(float4));
    cudaMalloc(&ncount, count * sizeof(int));
	cudaMalloc(&d_cob, count * sizeof(int));
	cudaMalloc(&xsph_delta, count * sizeof(float3));

	cudaMemset(velocity_sorted, 0, count * sizeof(float4));

   // if (settings.shaderType == 2) {
      

        size_t voxelBytes = (size_t)settings.x * (size_t)settings.y * (size_t)settings.z * sizeof(float);
        cudaMalloc(&voxelgrid, voxelBytes);
        cudaMemset(voxelgrid, 0, voxelBytes);
    

    printf("Total particle mem allocated: %.2f MB\n", (count * (6 * sizeof(float4) +  sizeof(float3) + ( 2* sizeof(int)))) / (1024.0 * 1024.0)); // prints the mem size for total allocation with maxpartiucles buffer

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: CUDA mem allocation failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
extern "C" void freegpu()
{
    cudaFree(positions);
    cudaFree(positions_sorted);
    cudaFree(velocity);
    cudaFree(velocity_sorted);
    cudaFree(accelration);
    cudaFree(accelration_sorted);

	cudaFree(ncount);
	cudaFree(xsph_delta);
    cudaFree(d_cob);
	cudaFree(voxelgrid);
	voxelgrid = nullptr;
	d_cob = nullptr;
    positions = nullptr;
    velocity = nullptr;
    accelration = nullptr;
    accelration_sorted = nullptr;
	ncount = nullptr;
};

struct GLVertex
{
    float px, py, pz;
    float radius;
    float cr, cg, cb, ca;
    float ox, oy;
    float wx, xy, xz;
};

static cudaGraphicsResource *g_vboResource = nullptr;

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
    if (g_vboResource)
    {
        cudaGraphicsUnregisterResource(g_vboResource);
        g_vboResource = nullptr;
    }
}

__global__ void packToVBOKernel(
    int n,
    const float4* __restrict__ pos,
    const float4* __restrict__ vel,
    float4* acl,

    GLVertex* vbo, bool heat, float heatMultipler, float dt, float heatDecay, float size,
    float R, float G, float B
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    float4 p = __ldg(&pos[i]);
    float4 v = __ldg(&vel[i]);
    float4 a = acl[i];

    int3 c = {0, 0, 0};

    if (heat)
    {

        float speed = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

        float heat = speed * 0.5f;
        // heat += mass[i];
        a.w += (heat * heatMultipler) * dt; // acl.w=heat;

        a.w *= expf(-heatDecay * dt);
        a.w = clamp(a.w, 0.0f, 100.0f);
        // load once to get col.w
        float t = clamp(a.w / 100.0f, 0.0f, 1.0f);
        t = pow(t, 0.95f);
        acl[i].w = a.w;

        if (t < 0.35f)
        {
            float tt = t / 0.25f;
            c.x = 0.0f;
            c.y = tt * 255.0f;
            c.z = 255.0f;
        }
        else if (t < 0.55f)
        {
            float tt = (t - 0.25f) / 0.25f;
            c.x = 0.0f;
            c.y = 255.0f;
            c.z = (1.0f - tt) * 255.0f;
        }
        else if (t < 0.85f)
        {
            float tt = (t - 0.5f) / 0.25f;
            c.x = tt * 255.0f;
            c.y = 255.0f;
            c.z = 0.0f;
        }
        else
        {
            float tt = (t - 0.75f) / 0.25f;
            c.x = 255.0f;
            c.y = (1.0f - tt) * 255.0f;
            c.z = 0.0f;
        }
    }
    else {
        c.x = R*255.0f;
        c.y = G*255.0f;
        c.z = B*255.0f;
    }

    float fpx = p.x,
          fpy = p.y,
          fpz = p.z;
    float rad = size; // size
    float fcr = c.x * (1.0f / 255.0f);
    float fcg = c.y * (1.0f / 255.0f);
    float fcb = c.z * (1.0f / 255.0f);

    
    const float ox[3] = {-1.0f, 3.0f, -1.0f};
    const float oy[3] = {-1.0f, -1.0f, 3.0f};

    int base = i * 3;
    for (int k = 0; k < 3; k++)
    {
        GLVertex &vtx = vbo[base + k];
        vtx.px = fpx;
        vtx.py = fpy;
        vtx.pz = fpz;
        vtx.radius = rad;
        vtx.cr = fcr;
        vtx.cg = fcg;
        vtx.cb = fcb;
        vtx.ca = 1.0f;
        vtx.ox = ox[k];
        vtx.oy = oy[k];
        vtx.wx = 0.0f;
        vtx.xy = 0.0f;
        vtx.xz = 0.0f;
    }
}

/// ///////////////////////////
// sph

struct HashCell
{
    int count;                             // Number of particles in this cell
    int particles[MAX_PARTICLES_PER_CELL]; // Particle indices
};
static int *d_cellStart = nullptr;
static int *d_cellEnd = nullptr;
static int *d_particleHash = nullptr;
static int *d_particleIndex = nullptr;

__device__ __host__ inline unsigned int spatialHash(int ix, int iy, int iz, int HASH_TABLE_SIZE)
{

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
                                        int &ix, int &iy, int &iz)
{
    ix = (int)floorf(x / cellSize);
    iy = (int)floorf(y / cellSize);
    iz = (int)floorf(z / cellSize);
}

__device__ __host__ inline unsigned int getHashFromPos(float x, float y, float z,
                                                       float cellSize, int hs)
{
    int ix, iy, iz;
    getCell(x, y, z, cellSize, ix, iy, iz);
    return spatialHash(ix, iy, iz, hs);
}

extern "C" bool initDynamicGrid(int maxParticles)
{
    // using maxpartcles which are 2-5X total particles for emiiter to work and dynamic add or remove particles
    HASH_TABLE_SIZE = 1;
    while (HASH_TABLE_SIZE < maxParticles * 2)
        HASH_TABLE_SIZE <<= 1;
    //  size_t hashTableBytes = HASH_TABLE_SIZE * sizeof(HashCell);

    // cudaMalloc(&d_hashTable, hashTableBytes);
    cudaMalloc(&d_cellStart, HASH_TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_cellEnd, HASH_TABLE_SIZE * sizeof(int));
    cudaMalloc(&d_particleHash, maxParticles * sizeof(int));
    cudaMalloc(&d_particleIndex, maxParticles * sizeof(int));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: CUDA allocation failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    // Initialize
    //  cudaMemset(d_hashTable, 0, hashTableBytes);
    cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
    cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));

    cudaMalloc(&d_particleHash_alt, maxParticles * sizeof(int));
    cudaMalloc(&d_particleIndex_alt, maxParticles * sizeof(int));

    // correct dry run — all nullptr, just getting the size
    cub::DeviceRadixSort::SortPairs(
        nullptr, sortTempBytes,
        (int *)nullptr, (int *)nullptr,
        (int *)nullptr, (int *)nullptr,
        maxParticles);

    cudaMalloc(&d_sortTempStorage, sortTempBytes);

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("ERROR: CUDA sort temp allocation failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    return true;
}
extern "C" void freeDynamicGrid()
{
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
    const float4 *__restrict__ pos,
    int *particleHash,
    int *particleIndex,
    int hs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;

    float x = pos[i].x;
    float y = pos[i].y;
    float z = pos[i].z;

    // Check for NaN
    if (isnan(x) || isnan(y) || isnan(z))
    {
        particleHash[i] = 0xFFFFFFFF; // Invalid hash
        particleIndex[i] = i;
        printf("WARNING nan positions\n");
        return;
    }

    // Compute hash
    unsigned int hash = getHashFromPos(x, y, z, cellSize, hs);

    particleHash[i] = hash;
    particleIndex[i] = i; // Store original index
}

__global__ void findCellBoundariesKernel(
    int numParticles,
    int *particleHash,
    int *cellStart,
    int *cellEnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;

    unsigned int hash = particleHash[i];
    
    unsigned int prevHash = (i > 0) ? particleHash[i - 1] : 0xFFFFFFFF;

    if (hash != prevHash)
    {
        // Start of new cell
        cellStart[hash] = i;
        if (i > 0)
        {
            cellEnd[prevHash] = i;
        }
    }

    if (i == numParticles - 1)
    {
        cellEnd[hash] = numParticles;
    }
}
__global__ void reorderParticlesKernel(
    int n, float dt,
    const int *__restrict__ sortedIndex, // d_particleIndex (after CUB sort)
    const float4 *__restrict__ posIn,
    const float4 *__restrict__ velIn,
	const float4* __restrict__ aclIn,

    float4 *posOut,
    float4 *velOut,
	float4* aclOut

)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int src = sortedIndex[i]; // where this particle CAME from in the original array
    float4 pi = __ldg(&posIn[src]);
    float4 vi = __ldg(&velIn[src]);
	float4 ai = __ldg(&aclIn[src]);
    // using pridicted positiopn into sorted arrays for density and pressure kernel  and help in stability
    // directly writeing to sorted arrays to avoid extra copy and also we will be using predicted position for density and pressure calculation which will help in stability
    float px = pi.x + vi.x * dt;
    float py = pi.y + vi.y * dt;
    float pz = pi.z + vi.z * dt;

    posOut[i] = {px, py, pz, pi.w};
    velOut[i] = vi;
	aclOut[i] = ai;
}

__global__ void clearActiveCellsKernel(
    int numParticles,
    const int *__restrict__ particleHash, // sorted hashes from LAST frame
    int *cellStart,
    int *cellEnd)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;

    // Each thread clears its own hash bucket.
    // Duplicate writes (multiple particles same cell) are harmless — idempotent.
    unsigned int h = (unsigned int)particleHash[i];
    cellStart[h] = -1;
    cellEnd[h] = -1;
}

void buildDynamicGrid(
 
    float cellSize,
    const float4 *__restrict__ pos, float dt

)
{
    int blocks = (settings.count + THREADS - 1) / THREADS;
	int numParticles = settings.count;
    if (numParticles <= 0)
    {
        printf("WARNING: buildDynamicGrid called with %d particles\n", numParticles);
        return;
    }

    if (settings.ff)
    {
        cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
        cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));
        settings.ff = false;
    }
    else
    {
        // d_particleHash still holds last frame's sorted hashes — perfect
        clearActiveCellsKernel<<<blocks, THREADS>>>(
            numParticles, d_particleHash, d_cellStart, d_cellEnd);
    }
    computeHashKernel<<<blocks, THREADS>>>(numParticles, cellSize, pos, d_particleHash, d_particleIndex, HASH_TABLE_SIZE);

    // sorting

    cub::DeviceRadixSort::SortPairs(
        d_sortTempStorage, sortTempBytes,
        d_particleHash, d_particleHash_alt,
        d_particleIndex, d_particleIndex_alt,
        settings.count);
    std::swap(d_particleHash, d_particleHash_alt);
    std::swap(d_particleIndex, d_particleIndex_alt);

    if (d_particleHash == nullptr || d_particleIndex == nullptr)
    {
        printf("ERROR: Null pointers in grid sort!\n");
        return;
    }

    findCellBoundariesKernel<<<blocks, THREADS>>>(numParticles, d_particleHash, d_cellStart, d_cellEnd);
    // sorteding arrays
    reorderParticlesKernel<<<blocks, THREADS>>>(
        numParticles, dt,
        d_particleIndex,     // tells us: sorted slot i came from original slot src
        positions, velocity,accelration, // source
        positions_sorted, velocity_sorted, accelration_sorted);


}

// sph-functions

__global__ void computeDensity(float cellSize,float4 *pos,float4 *vel,int hs,const int *__restrict__ cellstart,const int *__restrict__ cellend,const int *__restrict__ particleindex)
{
    // no shared memory because the arrays are sorted and coalesced access is good enough, also we are doing more computation per neighbor which helps hide latency.
    // shared memory has been tried and got no difference
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;

    float4 p = __ldg(&pos[i]); // is it good using ldg?? idk
    float xi = p.x;
    float yi = p.y;
    float zi = p.z;

    //  float rho = density[i];
    // float rhon = neardensity[i];
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

    int neighborCount = 0;
    int cellsChecked = 0;
    int cellsWithParticles = 0;

    float m_i = d.particlemass;

    float rhon = m_i * d.ndensity;
    float rho = m_i * d.sdensity;
    float mindensity = rho * 0.5f;

    // Search 27 neighboring cells
#pragma unroll 3

    for (int dz = -1; dz <= 1; dz++)
    {
#pragma unroll 3

        for (int dy = -1; dy <= 1; dy++)
        {
#pragma unroll 3

            for (int dx = -1; dx <= 1; dx++)
            {
                /*int manhattanDist = abs(dx) + abs(dy) + abs(dz);
                if (manhattanDist > 2) continue;*/
                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);

                int start = cellstart[hash];
                int end = cellend[hash];
                if (start == -1)
                    continue;

                cellsChecked++;

                if (start > 0)
                    cellsWithParticles++;

                /*  if (debug && count > 0) {
                     printf("  Neighbor cell (%d,%d,%d) hash=%u: %d particles\n",
                          cx + dx, cy + dy, cz + dz, hash, count);
                  }*/

                for (int k = start; k < end; k++)
                {
                    int j = k; // particleindex[k]; // index of neighbor particle in sorted array

                    if (j == i)
                        continue;
                    float4 pj = __ldg(&pos[j]);
                    float dx_val = xi - pj.x;
                    float dy_val = yi - pj.y;
                    float dz_val = zi - pj.z;
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < d.h2)
                    {
                        float invR = rsqrtf(r2 + 1e-12f);
                        float r = r2 * invR;
                        float v = d.h2 - r2;
                         // float v2 = h - r;
                        float vcube = v * v * v ;

                        float D = d.pollycoef6 * vcube; // precomputed pollycoef6
                        // float d = spikycoef2 * v2 * v2;
                        float m_j = d.particlemass; // mass
                        rho += m_j * D;
                        float x = d.h - r;
                        float nd = d.spikycoef * x * x * x ;
                        rhon += m_j * nd;
                        neighborCount++;
                    }
                }
            }
        }
    }
    /* if (debug) {
         printf("density:%5f \n",
              rho);
     }*/

    pos[i].w = fmaxf(rho, 0.0001f);   // just prevent div by zero
    vel[i].w = fmaxf(rhon, 0.0001f);
    //pos[i].w = rho;
    //vel[i].w = rhon;
}

__global__ void computePressure( float cellSize, const float4 *__restrict__ pos,float4 *acl,float4 *vel, int hs, int *cellstart, int *cellend,int *particleIndex, int *ncount, float3 *xsph)
{

    // no shared memory because the arrays are sorted and coalesced access is good enough, also we are doing more computation per neighbor which helps hide latency.
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;

    float4 p = __ldg(&pos[i]);
    float4 v = __ldg(&vel[i]);
    float xi = p.x;
    float yi = p.y;
    float zi = p.z;

    float3 force = {0.0f, 0.0f, 0.0f};
    float wallForce = d.rep; // tune this
    float wallDst = d.dst * d.h;
    if (xi - d.minX < wallDst)
        force.x += wallForce * (1.0f - (xi - d.minX) / wallDst);
    if (d.maxX - xi < wallDst)
        force.x -= wallForce * (1.0f - (d.maxX - xi) / wallDst);
    if (yi - d.minY < wallDst)
        force.y += wallForce * (1.0f - (yi - d.minY) / wallDst);
    if (d.maxY - yi < wallDst)
        force.y -= wallForce * (1.0f - (d.maxY - yi) / wallDst);
    if (zi - d.minZ < wallDst)
        force.z += wallForce * (1.0f - (zi - d.minZ) / wallDst);
    if (d.maxZ - zi < wallDst)
        force.z -= wallForce * (1.0f - (d.maxZ - zi) / wallDst);

    float p_i = d.pressure * (p.w - d.restDensity);
    float pn_i = d.nearpressure *( v.w -d.restDensity) ;
   
    float3 visc = {0.0f, 0.0f, 0.0f};
    float3 delta = {0.0f, 0.0f, 0.0f};

    float3 vi = make_float3(v.x, v.y, v.z);
    int cx, cy, cz;
    getCell(xi, yi, zi, cellSize, cx, cy, cz);

    int neighborCount = 0;
    int cellsChecked = 0;
    int cellsWithParticles = 0;
    float rho_i = p.w;
    float nrho_i = v.w;


    float pressuretermRho_i = p_i / (rho_i * rho_i);
    float NpressuretermRho_i = (pn_i * p_i)*0.5f;
#pragma unroll 3
    for (int dz = -1; dz <= 1; dz++)
    {
#pragma unroll 3

        for (int dy = -1; dy <= 1; dy++)
        {
#pragma unroll 3

            for (int dx = -1; dx <= 1; dx++)
            {
                /* int manhattanDist = abs(dx) + abs(dy) + abs(dz);
                 if (manhattanDist > 2) continue;*/
                // TRY 3 INSTEAD OF 2 TO FIX JITTERYNESS

                unsigned int hash = spatialHash(cx + dx, cy + dy, cz + dz, hs);

                int start = cellstart[hash];
                int end = cellend[hash];
                if (start == -1)
                    continue;

                cellsChecked++;

                if (start > 0)
                    cellsWithParticles++;

                for (int k = start; k < end; k++)
                {
                    int j = k; // particleIndex[k]; // index of neighbor particle in sorted array

                    if (j == i)
                        continue;
                    float4 pj = __ldg(&pos[j]);
                    float4 vj = __ldg(&vel[j]);
                    float dx_val = xi - pj.x;
                    float dy_val = yi - pj.y;
                    float dz_val = zi - pj.z;
                    float r2 = dx_val * dx_val + dy_val * dy_val + dz_val * dz_val;

                    if (r2 < d.h2 && r2 > 1e-9f)
                    {

                        float invR = rsqrtf(r2 + 1e-12f);
                        float r = r2 * invR;

                        float p_j = d.pressure * (pj.w - d.restDensity);
                        float np_j = d.nearpressure * vj.w;

                        float3 dir = {dx_val * invR, dy_val * invR, dz_val * invR};

                        neighborCount++;

                        /*  if (debug && neighborCount <= 3) {
                              printf("  Neighbor %d: dist=%.3f rho=%.6f p=%.6f\n",
                                  j, r, density[j], p_j);
                          }*/
                        float rho_j = pj.w;
                        float nrho_j = vj.w;
                        float x = d.h - r;
                        float gradW = d.spikyGradv * x * x; // precomputed gradw in negative value
                        float ngrad = d.neargrad * x * x * x;
                        float pressureterm = pressuretermRho_i + p_j / (rho_j * rho_j);
                        float npressureterm = NpressuretermRho_i + np_j / (nrho_j * nrho_j);
                        // float pressureterm = (p_i + p_j)/2.0f;
                        // float npressureterm = (pn_i + np_j)/2.0f;

                        float m_j = d.particlemass; // particle mass

                        force += -m_j*pressureterm * gradW * dir;

                        force += m_j*npressureterm * ngrad * dir;
                        
                        
                            float3 vxj = make_float3(vj.x, vj.y, vj.z);
                            float3 vij = (vxj - vi);

                            float lapW = d.viscK * x;
                            float viscosityCoeff = d.viscstrength;
                            visc += viscosityCoeff * m_j * vij / rho_j * lapW;

                            float v = d.h2 - r2;
                            float W = d.pollycoef6 * v * v * v;

                            

                            float coeff = ( d.particlemass / rho_j) * W;
                            delta.x += coeff * (vj.x - vi.x);
                            delta.y += coeff * (vj.y - vi.y);
                            delta.z += coeff * (vj.z - vi.z);

                      
                    }
                }
            }
        }
    }

    float4 accl;

    accl.z = (force.z + visc.z)/rho_i;
    accl.x = (force.x + visc.x)/rho_i;
    accl.y = (force.y + visc.y)/rho_i;
    accl.w = 0.0f;

	float eps = d.epsilon;
    xsph[i] = eps * delta;

    
	
   // int org = particleIndex[i]; // where this particle came from in the original unsorted array
                                // velocity written to org idx ,using swaps or memcpy caused visuals errors and performance heavy
    ncount[i] = neighborCount;
    acl[i] += accl; // write back to original slot in acl array
    // velocity verlet intigration fisrt step
    /*velocity[org].x += accl.x * dt * 0.5;
    velocity[org].y += accl.y * dt * 0.5;
    velocity[org].z += accl.z * dt * 0.5;*/
}


__global__ void scatterarray(int numParticles,float dt,float4* aclin,float4* aclout,float4* vel,int* particleindex,float3* xsph) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;
    int org = particleindex[i]; // where this particle came from in the original unsorted array
  float4 accl=__ldg(&aclin[i]);
  float3 vx = xsph[i];
  aclout[org] = accl;
  float4 v;
  v.x =  accl.x * dt * 0.5;
  v.y =   (accl.y - d.downf) * dt * 0.5;
  v.z =  accl.z * dt * 0.5;

  v.x += vx.x;
  v.y += vx.y;
  v.z += vx.z;

  vel[org] += v;
  

}

__device__ void atomicMinFloat(float *addr, float val)
{
    int *addr_i = (int *)addr;
    int old = *addr_i, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) <= val)
            break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__device__ void atomicMaxFloat(float *addr, float val)
{
    int *addr_i = (int *)addr;
    int old = *addr_i, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) >= val)
            break;
        old = atomicCAS(addr_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void debug(int n, float4 *pos, float4 *vel, int *ncount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    int nc = ncount[i];
    float d = pos[i].w;
    float nd = vel[i].w;

    atomicMin(&min_nb, nc);
    atomicMax(&max_nb, nc);
    atomicAdd(&avg_nb, nc);

    atomicMinFloat(&min_Density, d);
    atomicMaxFloat(&max_Density, d);
    atomicAdd(&avg_Density, d);

    atomicMinFloat(&min_nearDensity, nd);
    atomicMaxFloat(&max_nearDensity, nd);
    atomicAdd(&avg_nearDensity, nd);
}

__global__ void checkOutOfBounds(int n, float4* pos, int* cob)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 p = __ldg(&pos[i]);

    if (p.x < d.minX || p.x > d.maxX ||
        p.y < d.minY || p.y > d.maxY ||
        p.z < d.minZ || p.z > d.maxZ)
    {
        atomicOr(cob, 1);  // safe, no race
    }
}
// update
__global__ void updateKernel( float dt,float4 *pos, float4 *vel, float4 *acl,float4 target,float radius,bool t,float omega,float steer
    )
{
    // Vec3 acc_new;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;

    float4 p = __ldg(&pos[i]);
    float4 vl = __ldg(&vel[i]);
    float4 a = __ldg(&acl[i]);
    vl.x += a.x * dt * 0.5f;
    vl.y += (a.y - d.downf) * dt * 0.5f;
    vl.z += a.z * dt * 0.5f;
   
    

    /*if (ncount[i] < 5)
    {
        float drag = expf(-coeff * dt);
        vl.x *= drag;
        vl.y *= drag;
        vl.z *= drag;
    }*/

    p.x +=   vl.x* dt;
    p.y +=   vl.y* dt;
    p.z +=   vl.z* dt;

    if (t)
    {
        float dx = p.x - target.x;
        float dz = p.z - target.z;
        float r2 = dx * dx + dz * dz;
        float rr = radius * radius;
        if (r2 < rr && r2 > 1e-6f &&
            p.y >= 0.0f && p.y <= target.w)
        {
            float invR = rsqrtf(r2);   // fast, already have r2

            float tx = -dz * invR;
            float tz = dx * invR;

            float curTang = vl.x * tx + vl.z * tz;
            float targetTang = omega * sqrtf(r2);
            float deltaV = (targetTang - curTang) * steer * dt;

            vl.x += tx * deltaV;
            vl.z += tz * deltaV;
        }
    }
    a.x = 0;
    a.y = 0;
    a.z = 0;

    const float friction = 0.1f; // Surface friction coefficient

    // ---- X bounds ----
    if (p.x <= d.minX)
    {
        p.x = d.minX;
        float normalVel = vl.x;

        if (normalVel < 0.0f)
        {
            // Normal component (bounce)
            vl.x = -normalVel * d.restitution;

            // Tangential friction
            float tangentSpeed = sqrtf(vl.y * vl.y + vl.z * vl.z);
            if (tangentSpeed > 1e-6f)
            {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.y *= (1.0f - frictionMag / tangentSpeed);
                vl.z *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }
    else if (p.x >= d.maxX)
    {
        p.x = d.maxX;
        float normalVel = vl.x;

        if (normalVel > 0.0f)
        {
            // Normal component (bounce)
            vl.x = -normalVel * d.restitution;

            // Tangential friction
            float tangentSpeed = sqrtf(vl.y * vl.y + vl.z * vl.z);
            if (tangentSpeed > 1e-6f)
            {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.y *= (1.0f - frictionMag / tangentSpeed);
                vl.z *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }

    // ---- Y bounds ----
    if (p.y <= d.minY)
    {
        p.y = d.minY;
        float normalVel = vl.y;

        if (normalVel < 0.0f)
        {
            vl.y = -normalVel * d.restitution;

            float tangentSpeed = sqrtf(vl.x * vl.x + vl.z * vl.z);
            if (tangentSpeed > 1e-6f)
            {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.x *= (1.0f - frictionMag / tangentSpeed);
                vl.z *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }
    else if (p.y >= d.maxY)
    {
        p.y = d.maxY;
        float normalVel = vl.y;

        if (normalVel > 0.0f)
        {
            vl.y = -normalVel * d.restitution;

            float tangentSpeed = sqrtf(vl.x * vl.x + vl.z * vl.z);
            if (tangentSpeed > 1e-6f)
            {
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.x *= (1.0f - frictionMag / tangentSpeed);
                vl.z *= (1.0f - frictionMag / tangentSpeed);
            }
        }
    }

    // ---- Z bounds (floor/ceiling) ----
    if (p.z <= d.minZ)
    {
        p.z = d.minZ;
        float normalVel = vl.z;

        if (normalVel < 0.0f)
        {
            vl.z = -normalVel * d.restitution;

            // Extra friction on floor (prevents sliding)
            float tangentSpeed = sqrtf(vl.x * vl.x + vl.y * vl.y);
            if (tangentSpeed > 1e-6f)
            {
                //  float floorFriction = friction * 2.0f; // Stronger floor friction
                float frictionMag = fminf(friction * fabsf(normalVel), tangentSpeed);
                vl.x *= (1.0f - frictionMag / tangentSpeed);
                vl.y *= (1.0f - frictionMag / tangentSpeed);
            }

            //// Stop tiny vibrations (resting particles)
            // if (fabsf(vl.z) < 0.1f) {
            //     vl.z = 0.0f;
            //     vl.x *= damping;
            //     vl.y *= damping;
            // }
        }
    }
    else if (p.z >= d.maxZ)
    {
        p.z = d.maxZ;
        float normalVel = vl.z;

        if (normalVel > 0.0f)
        {
            vl.z = -normalVel * d.restitution;

            float tangentSpeed = sqrtf(vl.x * vl.x + vl.y * vl.y);
            if (tangentSpeed > 1e-6f)
            {
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

__device__ inline float randf(unsigned int seed)
{
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    return (float)(seed & 0xFFFFFF) / (float)0xFFFFFF;
}
// emitter
__global__ void addparticles(float4 *position, float4 *velocity, float4 *accelration)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= d.flowcount)
        return;
    int k = d.count + i;
    float x, y, z;
    float particle_spacing = d.h * d.spacing;
    int nx = (int)ceilf(sqrtf((float)d.flowcount));
    int ix = i % nx;
    int iz = i / nx;

    float gridSizeX = (nx - 1) * particle_spacing;
    float gridSizeZ = (nx - 1) * particle_spacing;

    x = (d.maxX + d.minX) * 0.5f - gridSizeX * 0.5f + ix * particle_spacing;
    z = (d.maxZ + d.minZ) * 0.5f - gridSizeZ * 0.5f + iz * particle_spacing;
    y = d.maxY - particle_spacing;

    position[k].x = x;
    position[k].y = y;
    position[k].z = z;

    position[k].w = 0.0f; // w used for particle density

    velocity[k].x = 0.0f;
    velocity[k].z = 0.0f;
    velocity[k].y = -98.0f;

    velocity[k].w = 0.0f; // w used for particle neardensity

    accelration[k].x = 0.0f;
    accelration[k].y = 0.0f;
    accelration[k].z = 0.0f;

    accelration[k].w = 0.0f; // heat
}

__global__ void registerKernel( float4 *position, float4 *velocity, float4 *accelration)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)
        return;

    float x, y, z;
    float particle_spacing = d.h * 0.6f;
    float Lx = d.mx - d.nx;
    float Ly = d.my - d.ny;
    float Lz = d.mz - d.nz;
    // Distribute particles proportionally to box shape
    float cbrtN = cbrtf((float)d.count);
    float scale = cbrtf(Lx * Ly * Lz); // geometric mean volume
    int nx = max(1, (int)roundf(cbrtN * (Lx / scale)));
    int ny = max(1, (int)roundf(cbrtN * (Ly / scale)));
    int nz = max(1, (int)ceilf((float)d.count / (nx * ny))); // nz fills remainder
    // Clamp to physical box limits
    nx = min(nx, max(1, (int)floorf(Lx) + 1));
    ny = min(ny, max(1, (int)floorf(Ly) + 1));
    nz = min(nz, max(1, (int)floorf(Lz) + 1));
    // 3D 
    int ix = i % nx;
    int iy = (i / nx) % ny;
    int iz = i / (nx * ny);
    // Actual grid footprint
    float gridSizeX = (nx - 1) * particle_spacing;
    float gridSizeY = (ny - 1) * particle_spacing;
    float gridSizeZ = (nz - 1) * particle_spacing;
    // Center the grid in the box on ALL 3 axes
    float cx = (d.nx + d.mx) * 0.5f;
    float cy = (d.ny + d.my) * 0.5f;
    float cz = (d.nz + d.mz) * 0.5f;
    float startX = cx - gridSizeX * 0.5f;
    float startY = cy - gridSizeY * 0.5f;
    float startZ = cz - gridSizeZ * 0.5f;
    x = startX + ix * particle_spacing;
    y = startY + iy * particle_spacing;
    z = startZ + iz * particle_spacing; // Start at maxZ with offset, go downward

    position[i].x = x;
    position[i].y = y;
    position[i].z = z;

    position[i].w = 0.0f; // w used for particle density

    velocity[i].x = 0.0f;
    velocity[i].y = 0.0f;
    velocity[i].z = 0.0f;

    velocity[i].w = 0.0f; // w used for particle neardensity

    accelration[i].x = 0.0f;
    accelration[i].y = 0.0f;
    accelration[i].z = 0.0f;

    accelration[i].w = 0.0f; // heat
}


extern "C" void registerBodies()
{
    int Block = (settings.count + THREADS - 1) / THREADS;
    registerKernel<<<Block, THREADS>>>(
                                       positions, velocity, accelration);
}



__global__ void splatVoxelsTrilinear(const float4* __restrict__ pos, float* grid, float size, int x, int y, int z) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d.count)return;

    float4 p = __ldg(&pos[i]);

    float gx = (p.x - d.minX) / size - 0.5f;
    float gy = (p.y - d.minY) / size - 0.5f;
    float gz = (p.z - d.minZ) / size - 0.5f;

    int x0 = (int)floorf(gx);
    int y0 = (int)floorf(gy);
    int z0 = (int)floorf(gz);

    float fx = gx - x0;
    float fy = gy - y0;
    float fz = gz - z0;
    float density = (p.w > 0.0001f) ? p.w : d.particlemass;

    float validWeight = 0.0f;
    for (int dz = 0; dz <= 1; ++dz) {
        int zz = z0 + dz;
        if (zz < 0 || zz >= z) continue;
        float wz = dz ? fz : 1.0f - fz;

        for (int dy = 0; dy <= 1; ++dy) {
            int yy = y0 + dy;
            if (yy < 0 || yy >= y) continue;
            float wy = dy ? fy : 1.0f - fy;

            for (int dx = 0; dx <= 1; ++dx) {
                int xx = x0 + dx;
                if (xx < 0 || xx >= x) continue;
                float wx = dx ? fx : 1.0f - fx;
                validWeight += wx * wy * wz;
            }
        }
    }

    if (validWeight <= 0.0f)
        return;

    float normDensity = density / validWeight;
    for (int dz = 0; dz <= 1; ++dz) {
        int zz = z0 + dz;
        if (zz < 0 || zz >= z) continue;
        float wz = dz ? fz : 1.0f - fz;

        for (int dy = 0; dy <= 1; ++dy) {
            int yy = y0 + dy;
            if (yy < 0 || yy >= y) continue;
            float wy = dy ? fy : 1.0f - fy;

            for (int dx = 0; dx <= 1; ++dx) {
                int xx = x0 + dx;
                if (xx < 0 || xx >= x) continue;
                float wx = dx ? fx : 1.0f - fx;
                int idx = zz * y * x + yy * x + xx;
                atomicAdd(&grid[idx], normDensity * wx * wy * wz);
            }
        }
    }
}




extern "C" void reallocgrid() {
    settings.x = (int)ceilf((settings.maxX - settings.minX) / settings.voxelSize);
    settings.y = (int)ceilf((settings.maxY - settings.minY) / settings.voxelSize);
    settings.z = (int)ceilf((settings.maxz - settings.minZ) / settings.voxelSize);
    if (settings.x < 1) settings.x = 1;
    if (settings.y < 1) settings.y = 1;
    if (settings.z < 1) settings.z = 1;

    if (voxelgrid) {
        cudaFree(voxelgrid);
        voxelgrid = nullptr;
    }
    size_t gridSize = (size_t)settings.x * (size_t)settings.y * (size_t)settings.z * sizeof(float);
    cudaMalloc(&voxelgrid, gridSize);
    cudaMemset(voxelgrid, 0, gridSize);
}

float tamfov = tanf(settings.Fov * 0.5f * 3.14159f / 180.0f);

__device__ __forceinline__ float voxelValue(const float* __restrict__ grid, int x, int y, int z, int ix, int iy, int iz)
{
    if (ix < 0 || ix >= x || iy < 0 || iy >= y || iz < 0 || iz >= z)
        return 0.0f;
    return __ldg(&grid[iz * y * x + iy * x + ix]);
}

__device__ __forceinline__ float sampleVoxelsTrilinear(const float* __restrict__ grid, float3 pos, float size, int x, int y, int z)
{
    float gx = (pos.x - d.minX) / size - 0.5f;
    float gy = (pos.y - d.minY) / size - 0.5f;
    float gz = (pos.z - d.minZ) / size - 0.5f;

    int x0 = (int)floorf(gx);
    int y0 = (int)floorf(gy);
    int z0 = (int)floorf(gz);

    float fx = gx - x0;
    float fy = gy - y0;
    float fz = gz - z0;

    float c000 = voxelValue(grid, x, y, z, x0,     y0,     z0);
    float c100 = voxelValue(grid, x, y, z, x0 + 1, y0,     z0);
    float c010 = voxelValue(grid, x, y, z, x0,     y0 + 1, z0);
    float c110 = voxelValue(grid, x, y, z, x0 + 1, y0 + 1, z0);
    float c001 = voxelValue(grid, x, y, z, x0,     y0,     z0 + 1);
    float c101 = voxelValue(grid, x, y, z, x0 + 1, y0,     z0 + 1);
    float c011 = voxelValue(grid, x, y, z, x0,     y0 + 1, z0 + 1);
    float c111 = voxelValue(grid, x, y, z, x0 + 1, y0 + 1, z0 + 1);

    float c00 = lerp(c000, c100, fx);
    float c10 = lerp(c010, c110, fx);
    float c01 = lerp(c001, c101, fx);
    float c11 = lerp(c011, c111, fx);
    float c0 = lerp(c00, c10, fy);
    float c1 = lerp(c01, c11, fy);
    return lerp(c0, c1, fz);
}


__device__ __forceinline__ float3 calcNormal(
    const float* __restrict__ grid, float3 pos, float size, int x, int y, int z)
{
    float eps = size;  // one voxel width
    float dx = sampleVoxelsTrilinear(grid, { pos.x + eps, pos.y, pos.z }, size, x, y, z)
        - sampleVoxelsTrilinear(grid, { pos.x - eps, pos.y, pos.z }, size, x, y, z);
    float dy = sampleVoxelsTrilinear(grid, { pos.x, pos.y + eps, pos.z }, size, x, y, z)
        - sampleVoxelsTrilinear(grid, { pos.x, pos.y - eps, pos.z }, size, x, y, z);
    float dz = sampleVoxelsTrilinear(grid, { pos.x, pos.y, pos.z + eps }, size, x, y, z)
        - sampleVoxelsTrilinear(grid, { pos.x, pos.y, pos.z - eps }, size, x, y, z);
    return normalize({ -dx, -dy, -dz });  // negative = points outward
}

__device__ __forceinline__ float3 getSkyColor(float3 rd, float3 sunDir) {
    float sunDot = fmaxf(dot(rd, sunDir), 0.0f);
    float horizon = fmaxf(rd.y, 0.0f);

    float3 zenith = { 0.1f, 0.3f, 0.8f };
    float  t = clamp(sunDir.y + 0.3f, 0.0f, 1.0f);
    float3 horizCol = { 0.9f * (1.0f - t) + 0.7f * t,
                       0.5f * (1.0f - t) + 0.8f * t,
                       0.2f * (1.0f - t) + 0.9f * t };

    float  hpow = powf(horizon, 0.5f);
    float3 sky = { horizCol.x + (zenith.x - horizCol.x) * hpow,
                   horizCol.y + (zenith.y - horizCol.y) * hpow,
                   horizCol.z + (zenith.z - horizCol.z) * hpow };

    float mie = powf(sunDot, 8.0f) * 0.4f;
    sky.x += 1.0f * mie;
    sky.y += 0.8f * mie;
    sky.z += 0.5f * mie;

    if (sunDot > 0.9997f) { sky = { 1.5f, 1.6f, 0.9f }; }

    float ground = clamp(-rd.y * 8.0f, 0.0f, 1.0f);
    sky.x = sky.x * (1.0f - ground) + 0.15f * ground;
    sky.y = sky.y * (1.0f - ground) + 0.12f * ground;
    sky.z = sky.z * (1.0f - ground) + 0.10f * ground;

    return sky;
}
 
__device__ float hashFloor(float px, float pz) {
    float v = sinf(px * 127.1f + pz * 311.7f) * 43758.5453f;
    return v - floorf(v);
}

__device__ float3 sampleFloorColor(float wx, float wz, float3 sundir) {
    float2 tile = { floorf(wx / d.tilesize), floorf(wz / d.tilesize) };

    float checker = fmodf(tile.x + tile.y, 2.0f);
  

    float3 col;
    if (wx > d.centerx && wz > d.centerz) col = d.col4;
    else if (wx < d.centerx && wz > d.centerz) col = d.col3;
    else if (wx < d.centerx && wz < d.centerz) col = d.col2;
    else                                                  col = d.col1;

    float3 baseColor =  col * 1.1f ;

    float rnd = hashFloor(tile.x, tile.y);
    baseColor.x += (rnd - 0.5f) * d.variationStrength;
    baseColor.y += (rnd - 0.5f) * d.variationStrength;
    baseColor.z += (rnd - 0.5f) * d.variationStrength;

    float diff = fmaxf(dot({ 0.f,1.f,0.f }, normalize(sundir)), 0.0f);
    float light = fmaxf(diff, 0.02f);

    return {
        clamp(baseColor.x * light, 0.f, 1.f),
        clamp(baseColor.y * light, 0.f, 1.f),
        clamp(baseColor.z * light, 0.f, 1.f)
    };
}

__global__ void reymarch(uchar4* output,float* grid,int sw,int sh,
	float3 campos, float3 forward, float3 right, float3 up, float aspect, int x, int y, int z, float size, float halftanfov,float3 sundir
    ){
	int px = blockIdx.x * blockDim.x + threadIdx.x;
	int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= sw || py >= sh) return;

    float u = (2 * (px + 0.5f) / sw - 1);
	float v = (1 - 2 * (py + 0.5f) / sh);

	u *= halftanfov* aspect;
    v *= halftanfov;
	float3 dir = forward + u * right + v * up;
	dir = normalize(dir);
    float density = 0.0f;
	uchar4 color = { 0, 0, 0, 255 };
    float3 invDir = { 1.f / dir.x, 1.f / dir.y, 1.f / dir.z };

    float t0x = (d.minX - campos.x) * invDir.x;
    float t1x = (d.maxX - campos.x) * invDir.x;
    float tMinX = fminf(t0x, t1x);
    float tMaxX = fmaxf(t0x, t1x);

    float t0y = (d.minY - campos.y) * invDir.y;
    float t1y = (d.maxY - campos.y) * invDir.y;
    float tMinY = fminf(t0y, t1y);
    float tMaxY = fmaxf(t0y, t1y);

    float t0z = (d.minZ - campos.z) * invDir.z;
    float t1z = (d.maxZ - campos.z) * invDir.z;
    float tMinZ = fminf(t0z, t1z);
    float tMaxZ = fmaxf(t0z, t1z);

    float t_enter = fmaxf(fmaxf(tMinX, tMinY), tMinZ);
    float t_exit = fminf(fminf(tMaxX, tMaxY), tMaxZ);

    if (t_exit < t_enter || t_exit < 0.f) {
        output[py * sw + px] = { 0, 0, 0, 0}; 
        return;
    }

    float stepsize = d.stepsize;
    float t_start = fmaxf(t_enter, 0.0f);
    float t = t_start + fminf(stepsize * 0.5f, fmaxf((t_exit - t_start) * 0.5f, 0.0001f));
    int steps = (int)((t_exit - t) / stepsize) + 1;
    for (int i = 0; i < steps; i++) {
        if (t > t_exit) break;
        float3 pos = campos + dir * t;
        t += stepsize;
        density = sampleVoxelsTrilinear(grid, pos, size, x, y, z);
        if (density > d.densityoffset) {
            float depth = t_exit - t;
            float3 n = calcNormal(grid, pos, size, x, y, z);//normals
            float3 viewDir = normalize(-dir);

			float diffuse = max(dot(n, sundir), 0.0f);

			float3 h = normalize(sundir + viewDir); 

            float specular = pow(max(dot(n, h), 0.0), 256);

            float F0 = powf(((1.003f - 1.333f) / (1.003f + 1.333f)), 2.0f);

			float fresnel = F0 + (1.0f - F0) * powf(1.0f - max(dot(n, viewDir), 0.0f), 5.0f);

            float3 reflDir = dir - 2 * dot(dir, n) * n;
			float3 skyColor = getSkyColor(-reflDir, sundir);

            float eta = 1.0f / 1.33f;
            float cosI = -dot(n, dir);
            float sinT2 = eta * eta * (1.0f - cosI * cosI);
            float cosT = sqrtf(1.0f - sinT2);
            float3 refrDir = eta * dir + (eta * cosI - cosT) * n;
            refrDir = normalize(refrDir);

            float3 refrColor;
            if (refrDir.y < 0.0f) {
                float floorY = d.minY - 1.0f;
                float tFloor = (floorY - pos.y) / refrDir.y;
                float3 floorHit = pos + refrDir * tFloor;

                refrColor = sampleFloorColor(floorHit.x, floorHit.z, sundir);
            }
            else {
                refrColor = getSkyColor(refrDir,sundir);
            }
            float3 absCoeff = { d.extinctionR, d.extinctionG, d.extinctionB };
            float optical = depth * d.scale;
            float3 transmittance = {
                 expf(-absCoeff.x * optical),
                 expf(-absCoeff.y * optical),
                 expf(-absCoeff.z * optical)
                         };
            float3 tintedRefr = {
                 refrColor.x * transmittance.x,
                 refrColor.y * transmittance.y,
                 refrColor.z * transmittance.z
            };

            float3 color{ .05f,.35f,.55f };

			float3 ambient = skyColor * 0.2f;

            float3 finalColor = ambient+fresnel *skyColor+(1.0f- fresnel) * tintedRefr + specular;//sun color 255

            finalColor.x = clamp(finalColor.x, 0.0f, 1.0f);
            finalColor.y = clamp(finalColor.y, 0.0f, 1.0f);
            finalColor.z = clamp(finalColor.z, 0.0f, 1.0f);

            output[py * sw + px] = {
                (unsigned char)(finalColor.x * 255.0f),
                (unsigned char)(finalColor.y * 255.0f),
                (unsigned char)(finalColor.z * 255.0f),
               255,
            };

            

            return;


        }
    }
    
        output[py * sw + px] = { 255, 255, 255, 0 };
}



cudaGraphicsResource* rayPBOResource = nullptr;

static unsigned int s_rayPBO = 0;
static unsigned int s_rayTex = 0;

extern "C" void unregisterraymarch() {
    if (rayPBOResource) {
        cudaGraphicsUnregisterResource(rayPBOResource);
        rayPBOResource = nullptr;
    }
    s_rayPBO = 0;
    s_rayTex = 0;
}

extern "C" void initraymarch(unsigned int rayPBO, unsigned int rayTex) {
    unregisterraymarch();
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("VRAM free: %.2f MB / %.2f MB\n", free_mem / 1024.0 / 1024.0, total_mem / 1024.0 / 1024.0);

    s_rayPBO = rayPBO;
    s_rayTex = rayTex;
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&rayPBOResource, rayPBO, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        printf("ERROR: cudaGraphicsGLRegisterBuffer(rayPBO): %s\n", cudaGetErrorString(err));
        rayPBOResource = nullptr;
    }
}

extern "C" void render() {
    int blocks = (settings.count + THREADS - 1) / THREADS;
    if (blocks <= 0)
        return;

   

    if (settings.shaderType != 2) {
    cudaError_t err = cudaGraphicsMapResources(1, &g_vboResource, 0);
    if (err != cudaSuccess) {
        printf("ERROR: cudaGraphicsMapResources(VBO): %s\n", cudaGetErrorString(err));
        return;
    }

    GLVertex* d_vbo = nullptr;
    size_t nbytes = 0;
    err = cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &nbytes, g_vboResource);
    if (err != cudaSuccess) {
        printf("ERROR: cudaGraphicsResourceGetMappedPointer(VBO): %s\n", cudaGetErrorString(err));
        cudaGraphicsUnmapResources(1, &g_vboResource, 0);
        return;
    }
        packToVBOKernel << <blocks, THREADS >> > (
            settings.count, positions, velocity, accelration,
            d_vbo, settings.heateffect, settings.heatMultiplier,
            settings.fixedDt, settings.cold, settings.size, settings.particlecolorR, settings.particlecolorG, settings.particlecolorB);
    cudaGraphicsUnmapResources(1, &g_vboResource, 0);
    }
}

void raymarchingrender() {

    int blocks = (settings.count + THREADS - 1) / THREADS;
    if (blocks <= 0)
        return;

    if (settings.shaderType == 2) {
        if (settings.shaderType == 2) {
            if (!voxelgrid || !rayPBOResource)
                return;

            size_t voxelBytes = (size_t)settings.x * (size_t)settings.y * (size_t)settings.z * sizeof(float);
            cudaMemset(voxelgrid, 0, voxelBytes);
            splatVoxelsTrilinear << <blocks, THREADS >> > (positions, voxelgrid, settings.voxelSize, settings.x, settings.y, settings.z);

            cudaError_t err = cudaGraphicsMapResources(1, &rayPBOResource, 0);
            if (err != cudaSuccess) {
                printf("ERROR: cudaGraphicsMapResources(rayPBO): %s\n", cudaGetErrorString(err));
                return;
            }

            uchar4* d_pixels = nullptr;
            size_t numBytes = 0;
            err = cudaGraphicsResourceGetMappedPointer((void**)&d_pixels, &numBytes, rayPBOResource);
            if (err != cudaSuccess) {
                printf("ERROR: cudaGraphicsResourceGetMappedPointer(rayPBO): %s\n", cudaGetErrorString(err));
                cudaGraphicsUnmapResources(1, &rayPBOResource, 0);
                return;
            }

            float tamfov = tanf(settings.Fov * 0.5f * 3.14159f / 180.0f);
            int sw = (int)settings.sw;
            int sh = (int)settings.sh;
            dim3 block(16, 16);
            dim3 grid((sw + block.x - 1) / block.x, (sh + block.y - 1) / block.y);
            reymarch << <grid, block >> > (
                d_pixels, voxelgrid,
                sw, sh,
                settings.campos, settings.Forward, settings.Right, settings.Up,
                settings.Aspect,
                settings.x, settings.y, settings.z,
                settings.voxelSize,
                tamfov, settings.sundir);

            err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("ERROR: reymarch launch: %s\n", cudaGetErrorString(err));
            }

            cudaGraphicsUnmapResources(1, &rayPBOResource, 0);

            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, s_rayPBO);
            glBindTexture(GL_TEXTURE_2D, s_rayTex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, sw, sh, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            return;
        }
    }
}



extern "C" void computephysics(float dt)
{
    int blocks = (settings.count + THREADS - 1) / THREADS;
    int totalBodies = settings.count;
    // cudaError_t err;
    float d_cellsize = settings.h * settings.cellSize; // tweaak it gng

    //  float subdt = settings.fixedDt / settings.substeps;
    float deltaTime = dt / settings.substeps;


    if (settings.nopause)
    {
        for (int i = 0; i < settings.substeps; i++)
        {




            // update pos
            updateKernel<<<blocks, THREADS>>>( deltaTime,positions, velocity, accelration,make_float4(0.0f,0.0f,0.0f,settings.ylevel),settings.zoneradius,settings.turbulence,settings.omega,settings.steer);
            // acelrations reset

            if (settings.sph)
            {
                // builds grid and sorted arrays with pridicted positions
                 buildDynamicGrid(d_cellsize, positions, deltaTime); 

                // uses p pos for stability
                computeDensity<<<blocks, THREADS>>>( d_cellsize, positions_sorted, velocity_sorted, HASH_TABLE_SIZE, d_cellStart, d_cellEnd, d_particleIndex);



                if (settings.shaderType == 2) {

                   // cudaMemset(voxelgrid, 0, settings.x * settings.y * settings.z * sizeof(float));


                    //splatVoxelsTrilinear << <blocks, THREADS >> > (positions_sorted, voxelgrid, settings.voxelSize, settings.x, settings.y, settings.z);
                }







                // reads from pridicted pos and writes back to orginal velocity array with velocity verlet 2nd step
                computePressure<<<blocks, THREADS>>>( d_cellsize,positions_sorted, accelration_sorted, velocity_sorted,  HASH_TABLE_SIZE, d_cellStart, d_cellEnd, d_particleIndex,ncount,xsph_delta);


               






                


                scatterarray << <blocks, THREADS >> > (totalBodies, deltaTime, accelration_sorted, accelration, velocity, d_particleIndex,xsph_delta);

            
            }

        }
        // DEBUG INFO no always active
        static int framecount = 0;
        ++framecount;
        if (framecount >= 100 && settings.debug == true)
        {
		

            int izero = 0, ibig = INT_MAX;
            float fzero = 0.0f, fbig = FLT_MAX, fnbig = -FLT_MAX;

            cudaMemcpyToSymbol(min_nb, &ibig, sizeof(int));
            cudaMemcpyToSymbol(max_nb, &izero, sizeof(int));
            cudaMemcpyToSymbol(avg_nb, &izero, sizeof(int));

            cudaMemcpyToSymbol(min_Density, &fbig, sizeof(float));
            cudaMemcpyToSymbol(max_Density, &fnbig, sizeof(float));
            cudaMemcpyToSymbol(avg_Density, &fzero, sizeof(float));

            cudaMemcpyToSymbol(min_nearDensity, &fbig, sizeof(float));
            cudaMemcpyToSymbol(max_nearDensity, &fnbig, sizeof(float));
            cudaMemcpyToSymbol(avg_nearDensity, &fzero, sizeof(float));

            debug<<<blocks, THREADS>>>(totalBodies, positions_sorted, velocity_sorted, ncount);
            cudaDeviceSynchronize();

            // read back
            int h_minN, h_maxN, h_sumN;
            float h_minD, h_maxD, h_sumD;
            float h_minND, h_maxND, h_sumND;

            cudaMemcpyFromSymbol(&h_minN, min_nb, sizeof(int));
            cudaMemcpyFromSymbol(&h_maxN, max_nb, sizeof(int));
            cudaMemcpyFromSymbol(&h_sumN, avg_nb, sizeof(int));

            cudaMemcpyFromSymbol(&h_minD, min_Density, sizeof(float));
            cudaMemcpyFromSymbol(&h_maxD, max_Density, sizeof(float));
            cudaMemcpyFromSymbol(&h_sumD, avg_Density, sizeof(float));

            cudaMemcpyFromSymbol(&h_minND, min_nearDensity, sizeof(float));
            cudaMemcpyFromSymbol(&h_maxND, max_nearDensity, sizeof(float));
            cudaMemcpyFromSymbol(&h_sumND, avg_nearDensity, sizeof(float));

            // push to settings
            settings.min_n = h_minN;
            settings.max_n = h_maxN;
            settings.avg_n = (float)h_sumN / totalBodies;

            settings.min_density = h_minD;
            settings.max_density = h_maxD;
            settings.avg_density = h_sumD / totalBodies;

            settings.min_neardensity = h_minND;
            settings.max_neardensity = h_maxND;
            settings.avg_neardensity = h_sumND / totalBodies;

            framecount = 0;
        }
        //   cudaMemcpy(&settings.samplen, ncount+2, sizeof(int), cudaMemcpyDeviceToHost);

        // emiter
        // frametime for stability
        // prevents buffer overlow when particles reach 98.5% of buffer size or the maxframetime set by user is reached, also prevents adding too many particles at once which causes instability, also adding particles gradually looks better
        if (settings.addParticle == true)
        {
            static float frametime = 0.0f;
            static int framecount = 0;
            framecount++;
            frametime += dt;
            if (frametime >= settings.flowrate)
            {
                addparticles<<<blocks, THREADS>>>(
                                                  positions, velocity, accelration);
                settings.samplecount += settings.flowcount;
                settings.totalBodies += settings.flowcount;
                settings.count = settings.totalBodies;
                syncstruct();
                frametime = 0;
            }
        }
    }
    else {
        // if paused check if any particle is out of box,stay paused until all particles are back in box to prevent instability, also allows user to fix parameters while paused and see the effect immediately without worrying about particles flying away, also allows user to add particles while paused and see them spawn without flying away
        cudaMemset(d_cob, 0, sizeof(int));

        checkOutOfBounds << <blocks, THREADS >> > (totalBodies, positions, d_cob);

        int h_result = 0;
        cudaMemcpy(&h_result, d_cob, sizeof(int), cudaMemcpyDeviceToHost);
        settings.h_cob = (h_result != 0);
    }

    raymarchingrender();

    
}



