#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	void  updatebodies(float dt, int count, float cold,  float MAX_HEAT,float mx,float Mx,float my ,float My,float mz,float Mz,float res,float downf);
	
	void copyarray( int count,
		float* px, float* py, float* pz,
		float* vx, float* vy, float* vz,
		float* ax, float* ay, float* az,
		float* ox, float* oy, float* oz,
		float* fx, float* fy, float* fz,
		float* size,
		float* mass,
		int* is,
		int* r, int* g, int* b,
		int* br, int* bg, int* bb,
		float* heat,
		float* density,
		float* pressure
		);
	void updatearray( int count, float* px, float* py, float* pz, float* size, int* r, int* g, int* b,float* heat);
	void initgpu(int count);
	void  freeDynamicGrid();
	
	void heating(int totalbodies, float dt, float hmulti, float cold);
	void stepsph(int totalbodies, float dt, float h, float pressure, float rst_density, float gamma, float av,float bv,float mx,float my,float mz,float maxx,float mY,float mZ,float st,float dm);
	void initDynamicGrid(int totalbodies);
#ifdef __cplusplus
}
#endif
