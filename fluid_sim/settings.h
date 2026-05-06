#pragma once
struct param {
	// === FLOAT VARIABLES (4 bytes each) ===
	// Grouped for cache locality and alignment
	float fixedDt = 1 / 120.0f;
	float simspeed = 1.0f;
	float size = 1.0f;
	float particleMass = 0.20f;
	float cold = 4.500f;
	float spacing = 0.71f;
	float heatMultiplier = 15.0f;
	float h = 4.0f;
	float h2 = h * h;
	float rest_density = 0.01800f;
	float pressure = 250.0f;
	float nearpressure = 400.0f;
	float visc = 0.0529f;
	float gravityforce = 20.0f;
	float pi = 3.14159265358979323846f;
	float sample_ms = 0.0f;
	float minX = -125.0f;
	float maxX = 125.0f;
	float minY = 0.0f;
	float maxY = 100.0f;
	float minZ = -50.0f;
	float maxz = 50.0f;
	float restitution = 0.8f;
	float pollycoef6 = 0.0f;
	float spikycoef = 0.0f;
	float Sdensity = 0.0f;
	float cellSize = 1.2f;
	float nearRestDensity = 0.153f;//not used atall wasted variable :D
	//float spikycoef2 = 0.0f;
	float ndensity = 0.0f;
	float spikygradv = 0.0f;
	float viscosity = 0.0f;
	float airdrag = 0.35f;
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

	float tilesize = 5.0f;
	float variationStrength = 0.15f;
	float color1R = 0.608f, color1G = 0.361f, color1B = 0.851f;
	float color2R = 0.851f, color2G = 0.361f, color2B = 0.361f;
	float color3R = 0.3f, color3G = 0.851f, color3B = 0.608f;
	float color4R = 0.851f, color4G = 0.792f, color4B = 0.361f;
	float floorbounds = 600.0f;
	float floorbounx = floorbounds * 0.5f;
	float floorboun_x = -floorbounds * 0.5f;
	float floorbounz = floorbounds * 0.5f;
	float floorboun_z = -floorbounds * 0.5f;

	float gaussSigma = 4.0f;
	float bgColorR = 0.11373f, bgColorG = 0.11373f, bgColorB = 0.11373f;

	float shallowColorR = 0.05f, shallowColorG = 0.55f, shallowColorB = 0.65f;
	float deepColorR = 0.01f, deepColorG = 0.06f, deepColorB = 0.18f;
	float absorption = 0.8f;

	float blurSigma = 14.0f;

	
	float blurDepthFall = 10.75f;

	float boundsSizeX = 5.0f, boundsSizeY = 5.0f, boundsSizeZ = 5.0f;
	float skyZenithR = 0.53f, skyZenithG = 0.81f, skyZenithB = 0.98f;
	float skyHorizonR = 0.85f, skyHorizonG = 0.91f, skyHorizonB = 0.97f;
	float maxframetime = 16.67;
	float min_density, max_density, avg_density = 0;
	float min_neardensity, max_neardensity, avg_neardensity = 0;
	float extinctionR = 1.80f, extinctionG = 0.50f, extinctionB = 0.30f;
	float mx = 75.0f, mz = 50.0f, my = 75.0f;
	float nx = -75.0f, nz = -50.0f, ny = 25.0f;
	float expandx = 0.0f, expandy = 0.0f, expandz = 0.0f;
	float movex = 0.0f, movey = 0.0f, movez = 0.0f;

	float sunIntensity = 1.5f;
	float refrStrength = 0.008f;

	//float shadowStrength = 1.5f;

	float blurWorldRadius = 1.3870f;    // world-space kernel radius
	float blurStrength = 0.250f;    // sigma scale factor
	float blurDiffStrength = 0.1f;     // depth-similarity falloff
	float refrMult = 1.8f;     // refraction ray march scale

	double fuc_ms = 0.0;

	// === INT VARIABLES (4 bytes each) ===
	int totalBodies = 60000;
	int maxparticles = totalBodies * 5;
	int count = totalBodies;
	int min_n, max_n, avg_n = 0;
	int samplecount = totalBodies;
	int flowcount = 18;
	int substeps = 1;
	int fpsCount = 0;
	int pressureMode = 0;
	int shaderType = 1;
	int samplen = 1;
	int blurMaxRadius = 32;    
	int cframe = 30;// screen-space pixel radius cap

	// === BOOL VARIABLES (1 byte each) ===
	bool sph = true;
	bool nopause = true;
	bool heateffect = true;
	bool addParticle = false;
	bool boundingBox = true;
	bool debug = false;
	bool recordSim = false;
	bool spawnstate = true;
	bool cf = false;
	bool h_cob;
	bool gui = true;
};
extern param settings;

struct data {
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
	float restDensity;
	float spikyGradv;
	float viscK;
	float viscstrength;
	
	int count;
	int flowcount;
};

extern data gpudata;