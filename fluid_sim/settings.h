#pragma once
struct param {
	// === FLOAT VARIABLES (4 bytes each) ===
	// Grouped for cache locality and alignment
	float fixedDt = 1 / 120.0f;
	float simspeed = 1.0f;
	float size = 1.0f;
	float particleMass = 1.0f;
	float cold = 4.500f;
	float cr = 4.0f;
	float heatMultiplier = 15.0f;
	float h = 3.5f;
	float h2 = h * h;
	float rest_density = 3.0f;
	float pressure = 100.0f;
	float nearpressure = 5000.0f;
	float visc = 6.0f;
	float downf = 150.0f;
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
	
	// ── Rendering ────────────────────────────────────────────────────────────
	float bgColorR = 0.40f, bgColorG = 0.40f, bgColorB = 0.40f;

	// Water body colours (screen-space mode)
	float shallowColorR = 0.25f, shallowColorG = 0.75f, shallowColorB = 1.0f;
	float deepColorR = 0.0f, deepColorG = 0.0f, deepColorB = 0.0f;
	float absorption = 1.4f;     // Beer-Lambert coefficient

	// Blur settings (screen-space mode)
	float blurSigma = 6.0f;     // Gaussian sigma in pixels (radius fixed at 8)
	float blurDepthFall = 2.0f;    // Bilateral depth-edge sharpness

	// Sky environment for reflection (screen-space mode)
	float skyZenithR = 0.05f, skyZenithG = 0.15f, skyZenithB = 0.45f;
	float skyHorizonR = 0.55f, skyHorizonG = 0.75f, skyHorizonB = 0.90f;
	float reflStrength = 0.70f;

	double fuc_ms = 0.0;
	// === INT VARIABLES (4 bytes each) ===
	int totalBodies = 20000;
	int maxparticles = totalBodies * 5;
	int count = totalBodies;
	int A = 0;
	int star = 0;
	int rc = 25;
	int gc = 50;
	int bc = 255;
	int samplecount = totalBodies;
	int flowcount = 5;
	int substeps = 1;
	int fpsCount = 0;
	int pressureMode = 0;
	int shaderType = 0;

	// === BOOL VARIABLES (1 byte each) ===
	// Grouped together to minimize padding
	bool colisionFun = true;
	bool updateFun = true;
	bool nopause = true;
	bool heateffect = true;
	bool addParticle = false;
	bool pressureClamp = false;
};
extern param settings;