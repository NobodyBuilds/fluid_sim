#version 330 core
out vec4 FragColor;

uniform vec3 u_SunDir;       // normalized, points TOWARD sun
uniform vec2 u_Resolution;
uniform mat4 u_InvViewProj; // for reconstructing world ray dir
uniform float u_SunIntensity;

vec3 getSkyColor(vec3 rd, vec3 sunDir) {
    float sunDot = max(dot(rd, sunDir), 0.0);
    float horizon = max(rd.y, 0.0);

    // Rayleigh — blue sky + reddish horizon
    vec3 zenith  = vec3(0.1, 0.3, 0.8);
    vec3 horizCol = mix(vec3(0.9, 0.5, 0.2), vec3(0.7, 0.8, 0.9),
                        clamp(sunDir.y + 0.3, 0.0, 1.0)); // redder at sunset
    vec3 sky = mix(horizCol, zenith, pow(horizon, 0.5));

    // Mie — sun glow
 float mie = pow(sunDot, 8.0) * 0.4 * u_SunIntensity; 
    sky += vec3(1.0, 0.8, 0.5) * mie;

    // Sun disk
    float disk = step(0.9997, sunDot); // hard cutoff, tweak for size
   sky = mix(sky, vec3(1.5, 1.6, 0.9) * u_SunIntensity, disk);

    // Below horizon: ground fog
    float ground = clamp(-rd.y * 8.0, 0.0, 1.0);
    sky = mix(sky, vec3(0.15, 0.12, 0.1), ground);

    return sky;
}

void main() {
    // Reconstruct ray direction from screen UV
    vec2 uv = (gl_FragCoord.xy / u_Resolution) * 2.0 - 1.0;
    vec4 rayClip = vec4(uv, 1.0, 1.0);
    vec4 rayWorld = u_InvViewProj * rayClip;
    vec3 rd = normalize(rayWorld.xyz / rayWorld.w);

    FragColor = vec4(getSkyColor(rd, u_SunDir), 1.0);
}