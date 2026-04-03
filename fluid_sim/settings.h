#pragma once
struct param {
	// === FLOAT VARIABLES (4 bytes each) ===
	// Grouped for cache locality and alignment
	float fixedDt = 1 / 120.0f;
	float simspeed = 1.0f;
	float size = 1.0f;
	float particleMass = 1.0f;
	float cold = 4.500f;
	float spacing = 1.01f;
	float heatMultiplier = 15.0f;
	float h = 3.50f;
	float h2 = h * h;
	float rest_density = 0.1460f;
	float pressure = 10000.0f;
	float nearpressure = 75000.0f;
	float visc = 0.055f;
	float gravityforce = 150.8f;
	float pi = 3.14159265358979323846f;
	float sample_ms = 0.0f;
	float minX = -125.0f;
	float maxX = 125.0f;
	float minY = -50.0f;
	float maxY = 50.0f;
	float minZ = -50.0f;
	float maxz = 50.0f;
	float restitution = 0.8f;
	float pollycoef6 = 0.0f;
	float spikycoef = 0.0f;
	float Sdensity = 0.0f;
	float cellSize = 1.0f;
	float nearRestDensity = 0.153f;
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
	float walldst = 0.90f;
	float wallrep = 58.0f;

	float tilesize = 5.0f;
	float variationStrength = 0.15f;
	float color1R = 0.608f, color1G = 0.361f, color1B = 0.851f;
	float color2R = 0.851f, color2G = 0.361f, color2B = 0.361f;
	float color3R = 0.3f, color3G = 0.851f, color3B = 0.608f;
	float color4R = 0.851f, color4G = 0.792f, color4B = 0.361f;
	float floorbounds = 600.0f;
	float floorbounx = floorbounds*0.5f;   // +X extent for floor
	float floorboun_x = -floorbounds*0.5f; // -X extent for floor
	float floorbouny = floorbounds * 0.5f;   // +Y extent for floor
	float floorboun_y = -floorbounds * 0.5f; // -Y extent for floor
	
	// ── Rendering ────────────────────────────────────────────────────────────
	float gaussSigma = 4.0f;
	float bgColorR = 0.11373f, bgColorG = 0.11373f, bgColorB = 0.11373f;

	float shallowColorR = 0.10196f, shallowColorG = 0.45f, shallowColorB = 0.525f;  // cyan/turquoise
	float deepColorR = 0.047f, deepColorG = 0.1686f, deepColorB = 0.2588f;           // dark ocean blue
	float absorption = 4.35f;

	float blurSigma = 18.0f;
	float blurDepthFall = 3.0f;
	float boundsSizeX = 5.0f, boundsSizeY = 5.0f, boundsSizeZ = 5.0f;
	float skyZenithR = 0.121f, skyZenithG = 0.3450f, skyZenithB = 0.5686f;   // bright sky blue
	float skyHorizonR = 0.60f, skyHorizonG = 0.80f, skyHorizonB = 0.95f; // light horizon cyan
	float reflStrength = 0.72f;
	float maxframetime = 16.67;
	float min_density, max_density, avg_density = 0;
	float min_neardensity, max_neardensity, avg_neardensity = 0;
	float extinctionR = 0.45f, extinctionG = 0.18f, extinctionB = 0.08f;
	float mx = 50.0f, my = 50.0f, mz = 25.0f;
	float nx = -50.0f, ny = -50.0f, nz = -25.0f;
	float expandx = 0.0f, expandy = 0.0f, expandz = 0.0f;
	float movex = 0.0f, movey = 0.0f, movez = 0.0f;
	double fuc_ms = 0.0;
	// === INT VARIABLES (4 bytes each) ===
	int totalBodies = 30000;
	int maxparticles = totalBodies * 5;
	int count = totalBodies;
	int min_n, max_n, avg_n = 0;
	int samplecount = totalBodies;
	int flowcount = 10;
	int substeps = 2;
	int fpsCount = 0;
	int pressureMode = 0;
	int shaderType = 0;
	int samplen = 1;
	// === BOOL VARIABLES (1 byte each) ===
	// Grouped together to minimize padding
	bool sph = true;
	
	bool nopause = true;
	bool heateffect = true;
	bool addParticle = false;
	bool boundingBox = true;
	bool debug = false;
	bool recordSim = false;
	bool spawnstate = true;
};
extern param settings;