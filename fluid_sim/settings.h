#pragma once
#include<iostream>

#include <vector_types.h>
struct param {
	// === FLOAT VARIABLES (4 bytes each) ===
	// Grouped for cache locality and alignment
	float3 campos = {0.0f,0.0f,0.0f};
	float3 Forward = {0.0f, 0.0f, 1.0f};
	float3 Up = {0.0f, 1.0f, 0.0f};
	float3 Right = {1.0f, 0.0f, 0.0f};
	float3 sundir = { 0.4f, 0.6f, 0.3f };

	float sw = 0.0f;
	float sh = 0.0f;
	float fixedDt = 1 / 60.0f;
	float Fov = 0.0f;
	float Aspect = 0.0f;
	float size = 1.0f;
	float particleMass = 0.20f;
	float cold = 4.500f;
	float spacing = 0.48f;
	float heatMultiplier = 15.0f;
	float h = 4.0f;
	float h2 = h * h;
	float rest_density = 0.01800f;
	float pressure = 140.0f;
	float nearpressure = 40.0f;
	float visc = 0.0f;
	float gravityforce = 98.0f;
	float pi = 3.14159265358979323846f;
	float sample_ms = 0.0f;
	float minX = -125.0f;
	float maxX = 125.0f;
	float minY = 0.0f;
	float maxY = 100.0f;
	float minZ = -50.0f;
	float maxz = 50.0f;
	float restitution = 0.8f;
	float flowrate = 0.018f;
	float pollycoef6 = 0.0f;
	float spikycoef = 0.0f;
	float Sdensity = 0.0f;
	float cellSize = 1.0f;
	float ndensity = 0.0f;
	float spikygradv = 0.0f;
	float viscosity = 0.0f;
	float wx = 0.0f;
	float wy = 0.0f;
	float wz = 0.0f;
	float avgFps = 0.0f;
	float minFps = 1000.0f;
	float maxFps = 0.0f;
	float fpsTimer = 0.0f;
	float accumulator = 0.0f;
	float fps = 0.0f;
	float walldst = 0.25f;
	float wallrep = 58.0f;
	float changelimit = 50.0f;
	float epsilon = 0.018f;
	float chnageamount = 2000.0f;
	float tilesize = 5.0f;
	float variationStrength = 0.f;
	float color1R = 1.0f, color1G = 1.0f, color1B = 1.0f;
	float color2R = 1.0f, color2G = 1.0f, color2B = 1.0f;
	float color3R = 1.0f, color3G = 1.0f, color3B = 1.0f;
	float color4R = 1.0f, color4G = 1.0f, color4B = 1.0f;
	float floorbounds = 600.0f;
	float floorbounx = floorbounds * 0.5f;
	float floorboun_x = -floorbounds * 0.5f;
	float floorbounz = floorbounds * 0.5f;
	float floorboun_z = -floorbounds * 0.5f;

	float gaussSigma = 4.0f;
	float bgColorR = 0.11373f, bgColorG = 0.11373f, bgColorB = 0.11373f;

	float shallowColorR = 0.05f, shallowColorG = 0.55f, shallowColorB = 0.65f;
	float deepColorR = 0.01f, deepColorG = 0.06f, deepColorB = 0.18f;
	float absorption = 2.0f;

	float blurSigma = 14.0f;

	float neargrad = 0.0f;
	float blurDepthFall = 10.75f;
	float particlecolorR=0.0f,particlecolorG =0.1f,particlecolorB=0.9f;
	float boundsSizeX = 5.0f, boundsSizeY = 5.0f, boundsSizeZ = 5.0f;
	float skyZenithR = 0.53f, skyZenithG = 0.81f, skyZenithB = 0.98f;
	float skyHorizonR = 0.85f, skyHorizonG = 0.91f, skyHorizonB = 0.97f;
	float maxframetime = 16.67;
	float min_density, max_density, avg_density = 0;
	float min_neardensity, max_neardensity, avg_neardensity = 0;
	float extinctionR = 1.f, extinctionG = (float)28/255, extinctionB = (float)28 / 255;
	float mx = 75.0f, mz = 50.0f, my = 75.0f;
	float nx = -75.0f, nz = -50.0f, ny = 25.0f;
	float expandx = 0.0f, expandy = 0.0f, expandz = 0.0f;
	float movex = 0.0f, movey = 0.0f, movez = 0.0f;

	float sunIntensity = 1.5f;
	float refrStrength = 0.008f;
	float vr = 0.2f, vg = 0.1f, vb = 0.9f;
	//float shadowStrength = 1.5f;

	float blurWorldRadius = 1.3870f;    // world-space kernel radius
	float blurStrength = 0.250f;    // sigma scale factor
	float blurDiffStrength = 0.1f;     // depth-similarity falloff
	float refrMult = 1.8f;     // refraction ray march scale
	float scale = 0.13f;
	float densityoffset = 0.09f;
	float voxelSize = 2.0f;
	int dy = 25;
	double fuc_ms = 0.0;
	float stepsize = 0.50f;
	float ramfree = 0.0f;
	float ramtotal = 0.0f;
	float depth = 2.0f;
	float zoneradius = 5.0f;
	float ylevel = 2.0f;
	float omega = 1.0f;
	float steer = 0.50f;
    int x = (int)ceil((maxX - minX) / voxelSize);
	int y = (int)ceil((maxY - minY) / voxelSize);
	int z = (int)ceil((maxz - minZ) / voxelSize);
	
	
	int totalBodies = 60000;
	int maxparticles = totalBodies * 5;
	int count = totalBodies;
	int min_n, max_n, avg_n = 0;
	int samplecount = totalBodies;
	int flowcount = 18;
	int substeps = 3;
	int fpsCount = 0;
	int pressureMode = 0;
	int shaderType = 2;
	int samplen = 1;
	int blurMaxRadius = 32;    
	int cframe = 30;// screen-space pixel radius cap

	bool sph = true;
	bool pause = true;
	bool heateffect = true;
	bool addParticle = false;
	bool boundingBox = true;
	bool debug = false;
	bool recordSim = false;
	bool spawnstate = true;
	bool cf = false;
	bool h_cob;
	bool gui = true;
	bool movingbox = false;
	bool ff = true;
	bool turbulence = false;
};
extern param settings;

struct data {
	float3 col1;
	float3 col2;
	float3 col3;
	float3 col4;
	float centerx; 	float centerz;
	float dt;
	float downf;
	float particlemass;
	float minX, minY, minZ, maxX, maxY, maxZ;
	float mx, nx, my, ny, mz, nz;
	float restitution;
	float h;
	float spacing;
	float ndensity;
	float sdensity;
	float h2;
	float pollycoef6;
	float spikycoef;
	float rep;
	float dst;
	float pressure;
	float nearpressure;
	float extinctionR , extinctionG , extinctionB ;
	float restDensity;
	float nearrestdensity;
	float spikyGradv;
	float viscK;
	float viscstrength;
	float surfacetension;
	float epsilon;
	float neargrad;
	float scale;
	float vr, vg, vb;
	float densityoffset;
	float tilesize;
	float depth;
	float stepsize;
	float variationStrength ;
	
	int count;
	int flowcount;
};

extern data gpudata;
