#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	void registerBodies();

	bool initgpu(int count);
	void  freeDynamicGrid();
	void freegpu();
	
	bool initDynamicGrid(int totalbodies);

	void computephysics(float dt);
	void registerGLBuffer(unsigned int vboId);
	void unregisterGLBuffer();
	
#ifdef __cplusplus
}
#endif

