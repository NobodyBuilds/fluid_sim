#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	void registerBodies(int n, float h, float Size, float Mass, int R, int G, int B, float maxX, float maxY, float maxz, float minX, float minY, float minZ);

	void initgpu(int count);
	void  freeDynamicGrid();
	void freegpu();
	
	void initDynamicGrid(int totalbodies);

	void computephysics(int n, float dt, float h, float h2, float pollycoef6, float spikycoef, float gradv, float viscK, float sdensity, float ndensity, float rest_density, float pressure, float k_,
		float hmulti, float cold, float br, float bg, float bb, float maxX, float maxY, float maxZ, float minX, float minY, float minZ, float restitution, float downwardforce, int isstar, float visc);
	void registerGLBuffer(unsigned int vboId);
	void unregisterGLBuffer();

#ifdef __cplusplus
}
#endif

