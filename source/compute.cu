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

    cudaFree(d_cob);
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
    const float4 *__restrict__ pos,
    const float4 *__restrict__ vel,
    float4 *acl,

    GLVertex *vbo, bool heat, float heatMultipler, float dt, float heatDecay, float size)
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

    float fpx = p.x,
          fpy = p.y,
          fpz = p.z;
    float rad = size; // size
    float fcr = c.x * (1.0f / 255.0f);
    float fcg = c.y * (1.0f / 255.0f);
    float fcb = c.z * (1.0f / 255.0f);

    // Matches the offsets used in the old CPU drawAll() loop
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

    static bool firstFrame = true;
    if (firstFrame)
    {
        cudaMemset(d_cellStart, -1, HASH_TABLE_SIZE * sizeof(int));
        cudaMemset(d_cellEnd, -1, HASH_TABLE_SIZE * sizeof(int));
        firstFrame = false;
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

    pos[i].w = fmaxf(rho, mindensity);
    vel[i].w = fmaxf(rhon, mindensity );
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
    float pn_i = d.nearpressure * v.w ;
   
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
    float NpressuretermRho_i = pn_i / (nrho_i * nrho_i);
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

                        float pressureterm = pressuretermRho_i + p_j / (rho_j * rho_j);
                        float npressureterm = NpressuretermRho_i + np_j / (nrho_j * nrho_j);
                        // float pressureterm = (p_i + p_j)/2;

                        float m_j = d.particlemass; // particle mass

                        force += -m_j * pressureterm * gradW * dir;

                        force += -m_j * npressureterm * gradW * dir;
                        
                        
                            float3 vxj = make_float3(vj.x, vj.y, vj.z);
                            float3 vij = (vxj - vi);

                            float lapW = d.viscK * x;
                            float viscosityCoeff = d.viscstrength;
                            visc += viscosityCoeff * m_j * vij / rho_j * lapW;

                            float v = d.h2 - r2;
                            float W = d.pollycoef6 * v * v * v;

                            

                            float coeff = (2.0f * d.particlemass / (rho_j * rho_i)) * W;
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


__global__ void scatterarray(int numParticles,float dt,float4* aclin,float4* aclout,float4* vel,int* particleindex,float3* vs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numParticles)
        return;
    int org = particleindex[i]; // where this particle came from in the original unsorted array
  float4 accl=__ldg(&aclin[i]);
  float3 vx = vs[i];
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
__global__ void updateKernel( float dt,float4 *pos, float4 *vel, float4 *acl
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


    p.x += vl.x* dt;
    p.y += vl.y* dt;
    p.z += vl.z* dt;

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
            // update positipons

            updateKernel<<<blocks, THREADS>>>( deltaTime,positions, velocity, accelration);
            // acelrations reset

            if (settings.sph)
            {
                // builds grid and sorted arrays with pridicted positions
                 buildDynamicGrid(d_cellsize, positions, deltaTime); 

                // uses p pos for stability
                computeDensity<<<blocks, THREADS>>>( d_cellsize, positions_sorted, velocity_sorted, HASH_TABLE_SIZE, d_cellStart, d_cellEnd, d_particleIndex);

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

    
}

extern "C" void render() {
   
    int blocks = (settings.count + THREADS - 1) / THREADS;

    cudaGraphicsMapResources(1, &g_vboResource, 0);
    
        GLVertex* d_vbo = nullptr;
        size_t nbytes = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&d_vbo, &nbytes, g_vboResource);

        packToVBOKernel << <blocks, THREADS >> > (
            settings.count, positions, velocity, accelration,
            d_vbo, settings.heateffect, settings.heatMultiplier,
            settings.fixedDt, settings.cold, settings.size);
    

        cudaGraphicsUnmapResources(1, &g_vboResource, 0);
}