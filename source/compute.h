#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	void  updatebodies(float dt, int count, float cold, float MAX_HEAT, float mx, float Mx, float my, float My, float mz, float Mz, float res, float downf);

	void registerBodies(int n, float h, float Size, float Mass, int R, int G, int B, float maxX, float maxY, float maxz, float minX, float minY, float minZ);
	void updatearray(int count, float* px, float* py, float* pz, float* size, int* r, int* g, int* b);
	void initgpu(int count);
	void  freeDynamicGrid();
	void freegpu();
	void heating(int totalbodies, float dt, float hmulti, float cold,int rc,int gc,int bc);
	void stepsph(int totalbodies, float dt, float h, float pressure, float rst_density, float mx, float my, float mz, float maxx, float mY, float mZ, float visc,float k, float h2,
		float pollycoef6, float spikycoef, float gradv, float viscK, float sdensity, float ndensity);
	void initDynamicGrid(int totalbodies);
#ifdef __cplusplus
}
#endif