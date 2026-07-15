uniform sampler2D texturePosition;
uniform float uPointSize;
uniform float uTime;

// Connectome harmonics: per-dot signed field value (R channel), sampled
// here so the fragment shader can color dots by the oscillating eigenmode.
uniform sampler2D tHarmonicField;
uniform float uHarmonicColor;

varying vec2 vUV;
varying float vHarmonic;   // R: signed displacement field (coolwarm)
varying float vHarmonicU;  // G: phase-wheel parameter u∈[0,1] (Kuramoto)

void main() {
    vUV = uv;
    vec2 hf = uHarmonicColor > 0.5 ? texture2D(tHarmonicField, uv).rg : vec2(0.0);
    vHarmonic = hf.r;
    vHarmonicU = hf.g;
    vec4 pos = texture2D(texturePosition, uv);
    
    vec4 mvPosition = modelViewMatrix * vec4(pos.xyz, 1.0);
    gl_Position = projectionMatrix * mvPosition;
    
    // Scale point size with perspective so near particles appear larger.
    // Factor of 30 keeps them at a reasonable pixel size (e.g. ~18px at distance 10).
    gl_PointSize = uPointSize * (30.0 / -mvPosition.z);
}
