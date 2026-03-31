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
	float rest_density = 0.1036f;
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
	
	// ── Rendering ────────────────────────────────────────────────────────────
	float gaussSigma = 4.0f;
	float bgColorR = 0.11373f, bgColorG = 0.11373f, bgColorB = 0.11373f;

	float shallowColorR = 0.28f, shallowColorG = 0.62f, shallowColorB = 0.92f;
	float deepColorR = 0.02f, deepColorG = 0.10f, deepColorB = 0.38f;
	float absorption = 4.35f;

	float blurSigma = 14.0f;
	float blurDepthFall = 5.5f;
	float boundsSizeX = 5.0f, boundsSizeY = 5.0f, boundsSizeZ = 5.0f;
	float skyZenithR = 0.42f, skyZenithG = 0.62f, skyZenithB = 0.95f;
	float skyHorizonR = 0.58f, skyHorizonG = 0.72f, skyHorizonB = 0.88f;
	float skyGroundR = 0.186f, skyGroundG = 0.159f, skyGroundB = 0.186f;
	float reflStrength = 0.72f;
	float sunIntensity = 8.0f;
	float maxframetime = 16.67;
	float sunInvSize = 200.0f;
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
	int   gaussIterations = 2;
	int   gaussRadius = 8;
	int samplecount = totalBodies;
	int flowcount = 10;
	int substeps = 2;
	int fpsCount = 0;
	int pressureMode = 0;
	int shaderType = 1;
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
};
extern param settings;