#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	void registerBodies();

	void initgpu(int count);
	void  freeDynamicGrid();
	void freegpu();
	
	void initDynamicGrid(int totalbodies);

	void computephysics(float dt);
	void registerGLBuffer(unsigned int vboId);
	void unregisterGLBuffer();
	
#ifdef __cplusplus
}
#endif

