# Fluid Renderer GLSL Fix TODO

## Steps:

- [x] Step 1: Edit SSF_COMPOSITE_FRAG in fluid_renderer.h - Fix variable scope (move declarations before early-out) and syntax (split multi-line inits). **Now compile & test!**
- [ ] Step 2: Test compile (user runs build).
- [ ] Step 3: Verify render with shaderType=0 (no crashes, fluid surface smooth).
- [ ] Step 4: Optional - Fix projScale blur bug.
- [ ] Complete: attempt_completion.

Current: Starting Step 1.
