#pragma once
struct param {
	// === FLOAT VARIABLES (4 bytes each) ===
	// Grouped for cache locality and alignment
	float fixedDt = 1 / 120.0f;
	float simspeed = 1.0f;
	float size = 1.0f;
	float particleMass = 1.0f;
	float cold = 4.500f;
	float conductionrate = 4.0f;
	float heatMultiplier = 15.0f;
	float h = 3.5f;
	float h2 = h * h;
	float rest_density = 0.12f;
	float pressure = 4500.0f;
	float nearpressure = 100.0f;
	float visc = 6.0f;
	float gravityforce = 150.8f;
	float pi = 3.14159265358979323846f;
	float sample_ms = 0.0f;
	float minX = -50.0f;
	float maxX = 250.0f;
	float minY = -50.0f;
	float maxY = 50.0f;
	float minZ = 0.0f;
	float maxz = 100.0f;
	float restitution = 0.8f;
	float pollycoef6 = 0.0f;
	float spikycoef = 0.0f;
	float Sdensity = 0.0f;
	float rad = 1.0f;
	//float spikycoef2 = 0.0f;
	float ndensity = 0.0f;
	float spikygradv = 0.0f;
	float viscosity = 0.0f;
	float visclap = 0.0f;
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
	float wallrep = 50.0f;
	
	// ── Rendering ────────────────────────────────────────────────────────────
	float bgColorR = 0.11373f, bgColorG = 0.11373f, bgColorB = 0.11373f;

	// Water body colours (screen-space mode)
	float shallowColorR = 0.0f, shallowColorG = 0.59216f, shallowColorB = 0.68235f;
	float deepColorR = 0.01176f, deepColorG = 0.21569f, deepColorB = 0.25098f;
	float absorption = 4.30f;     // Beer-Lambert coefficient

	// Blur settings (screen-space mode)
	float blurSigma = 14.0f;     // Gaussian sigma in pixels (radius fixed at 8)
	float blurDepthFall = 2.0f;    // Bilateral depth-edge sharpness

	// Sky environment for reflection (screen-space mode)
	float skyZenithR = 0.05f, skyZenithG = 0.15f, skyZenithB = 0.45f;
	float skyHorizonR = 0.0f, skyHorizonG = 0.25096f, skyHorizonB = 0.46667f;
	float reflStrength = 0.70f;

	float maxframetime = 16.67;

	double fuc_ms = 0.0;
	// === INT VARIABLES (4 bytes each) ===
	int totalBodies = 20000;
	int maxparticles = totalBodies * 5;
	int count = totalBodies;
	int A = 0;
	int star = 0;
	
	int samplecount = totalBodies;
	int flowcount = 5;
	int substeps = 4;
	int fpsCount = 0;
	int pressureMode = 0;
	int shaderType = 1;

	// === BOOL VARIABLES (1 byte each) ===
	// Grouped together to minimize padding
	bool colisionFun = true;
	
	bool nopause = true;
	bool heateffect = true;
	bool addParticle = false;
	bool boundingBox = true;
	
	
	bool recordSim = false;
};
extern param settings;