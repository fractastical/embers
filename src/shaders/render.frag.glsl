// ═══════════════════════════════════════════════════════════════════════
// RENDER FRAGMENT SHADER — Controls how each particle dot looks.
//
// Each particle is rendered as a gl.POINTS primitive, so the GPU gives
// us a square quad for each point. gl_PointCoord lets us know where
// we are within that quad (0,0 = top-left, 1,1 = bottom-right).
//
// We create a soft, star-like glow by combining two layers:
// - A bright "core" that fades quickly (the dot's center)
// - A wider "glow" that fades gradually (the soft halo)
// This makes particles look like luminous points rather than hard discs.
//
// COLOR MODES:
//   White (0): Subtle tension-driven warm↔cool tint on white base.
//   Color (1): Plutchik emotion wheel coloring via VAD.
//     Uses valence (sentiment) and arousal to position on
//     the Plutchik circumplex, mapping to distinct hues:
//       Happy    → Gold/Yellow    (V+, A+)
//       Surprise → Amber/Orange   (V+, A++)
//       Angry    → Red            (V−, A++)
//       Disgust  → Green          (V−, A mid)
//       Fear     → Purple         (V−, A+)
//       Sad      → Blue           (V−, A−)
//       Neutral  → Warm White     (desaturated)
// ═══════════════════════════════════════════════════════════════════════

uniform vec3 uColor;        // Base particle color (fallback)
uniform float uAlpha;       // Overall opacity multiplier
uniform float uColorMode;   // 0.0 = white/tension, 1.0 = color (sentiment)
uniform float uTime;        // Animation time for subtle hue drift
uniform float uRolloff;     // Spectral rolloff → edge softness (0=soft, 1=crisp)
uniform float uBrightness;  // Overall brightness multiplier (tunable)
uniform float uCoreWeight;  // Core dot intensity (tunable)
uniform float uGlowWeight;  // Glow halo intensity (tunable)

// Color channel uniforms
uniform float uTension;     // 0–1, from spectral centroid (0=relaxed, 1=tense)
uniform float uSentiment;   // −1 to +1, valence from keyword classifier / SER
uniform float uEnergy;      // 0–1, from RMS
uniform float uEmotionalIntensity; // 0–1, from classifier
uniform float uEmotionArousal;     // 0–1, from SER (calm→excited)
uniform float uEmotionDominance;   // 0–1, from SER (submissive→dominant)

// Connectome harmonics coloring
uniform float uHarmonicColor;      // 0.0 = off, 1.0 = color dots by harmonic field
uniform float uHarmonicColorMode;  // 0.0 = coolwarm (signed), 1.0 = phase wheel (Kuramoto)

varying vec2 vUV;           // Per-particle UV from vertex shader
varying float vHarmonic;    // Per-particle harmonic field value, signed [-1, 1]
varying float vHarmonicU;   // Per-particle phase-wheel parameter u∈[0,1]

// ── COOLWARM DIVERGING COLORMAP ──────────────────────────────────────
// Saturated diverging map tuned for ADDITIVE blending with many overlapping
// dots per node: red (positive) ↔ dark (zero crossing / nodal line) ↔ blue
// (negative). Keeping the off-hue channels low means stacked sprites keep
// their hue instead of summing to white, and the |field| magnitude drives
// brightness so the oscillating "highs" pulse and nodal lines fade out.
vec3 coolwarm(float t) {
    float m = clamp(abs(t), 0.0, 1.0);
    vec3 warm = vec3(1.0, 0.16, 0.10);   // positive lobe → red
    vec3 cold = vec3(0.12, 0.42, 1.0);   // negative lobe → blue
    vec3 hue = t >= 0.0 ? warm : cold;
    // Small dark-neutral floor keeps zero-field nodes faintly visible.
    return mix(vec3(0.05), hue, m);
}

// ── KURAMOTO PHASE WHEEL ─────────────────────────────────────────────
// Matches the source viewer's phaseColor(): a cyclic blue→purple→pink→red
// arc. Input u = 0.5 - 0.5·cos(θ) ∈ [0,1], so synchronized (in-phase)
// oscillators share a colour and phase patterns flow across the graph.
vec3 phaseWheel(float u) {
    float h = fract((205.0 + 175.0 * u) / 360.0);   // 205° → 380°(=20°)
    // Inline HSV→RGB (S=0.76, V=0.98) so this stays self-contained.
    vec3 p = abs(fract(h + vec3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return 0.98 * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), 0.76);
}

// ── HSL → RGB CONVERSION ─────────────────────────────────────────────
vec3 hsl2rgb(float h, float s, float l) {
    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    float hp = h * 6.0;
    float x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
    float m = l - c * 0.5;

    vec3 rgb;
    if      (hp < 1.0) rgb = vec3(c, x, 0.0);
    else if (hp < 2.0) rgb = vec3(x, c, 0.0);
    else if (hp < 3.0) rgb = vec3(0.0, c, x);
    else if (hp < 4.0) rgb = vec3(0.0, x, c);
    else if (hp < 5.0) rgb = vec3(x, 0.0, c);
    else               rgb = vec3(c, 0.0, x);

    return rgb + m;
}

// ── PLUTCHIK EMOTION WHEEL ───────────────────────────────────────────
// Maps (valence, arousal) → hue on the Plutchik circumplex model.
//
// The 2D affect space (valence on X, arousal on Y) is divided into
// angular regions, each assigned a perceptually distinct hue.
// This gives smooth, continuous transitions between emotion colors.
//
// Emotion positions in V-A space (from ser-worker.ts VAD table):
//   Happy:    V=+0.8, A=0.7  → Gold    (45°)
//   Surprise: V=+0.3, A=0.8  → Amber   (30°)
//   Angry:    V=−0.6, A=0.8  → Red     (0°)
//   Fear:     V=−0.7, A=0.7  → Purple  (280°)
//   Disgust:  V=−0.7, A=0.4  → Green   (120°)
//   Sad:      V=−0.6, A=0.2  → Blue    (220°)
//   Neutral:  V= 0.0, A=0.2  → Warm    (desaturated)
float emotionHue(float valence, float arousal) {
    // Remap arousal from [0,1] to centered [-0.5, +0.5]
    float a = arousal - 0.5;
    float v = valence;

    // Compute angle in V-A space (radians, -π to +π)
    float angle = atan(a, v);

    // Normalize angle from [-π, +π] to [0, 1]
    float normAngle = (angle + 3.14159) / 6.28318;

    // Piecewise hue mapping — each angular region maps to a color
    float hue;
    if (normAngle < 0.12) {
        // Sad region (V−, A−) → Blue (0.61)
        hue = 0.61;
    } else if (normAngle < 0.30) {
        // Transition: Sad→Happy (crossing through positive valence, low arousal)
        hue = mix(0.61, 0.125, (normAngle - 0.12) / 0.18);
    } else if (normAngle < 0.42) {
        // Happy region (V+, A+) → Gold (0.125 = 45°)
        hue = mix(0.125, 0.083, (normAngle - 0.30) / 0.12);
    } else if (normAngle < 0.52) {
        // Surprise region (V+, strong A+) → Amber (0.083 = 30°)
        hue = mix(0.083, 0.03, (normAngle - 0.42) / 0.10);
    } else if (normAngle < 0.64) {
        // Angry region (V−, A++) → Red (0.0 = 0°)
        hue = mix(0.03, 0.0, (normAngle - 0.52) / 0.12);
    } else if (normAngle < 0.74) {
        // Fear region (V−, A+) → Purple (0.78 = 280°)
        // Jump through hue = 1.0 to wrap from red to purple
        hue = mix(1.0, 0.78, (normAngle - 0.64) / 0.10);
    } else if (normAngle < 0.86) {
        // Disgust region (V−, A mid) → Green (0.33 = 120°)
        hue = mix(0.78, 0.33, (normAngle - 0.74) / 0.12);
    } else {
        // Wrap: Disgust→Sad (Green back to Blue)
        hue = mix(0.33, 0.61, (normAngle - 0.86) / 0.14);
    }

    return fract(hue);
}

void main() {
    // Distance from center of the point quad (0.5, 0.5)
    float dist = length(gl_PointCoord - vec2(0.5));

    // Discard pixels outside the circle (turns square quad into circle)
    if (dist > 0.5) discard;

    // ── ROLLOFF → EDGE SOFTNESS ───────────────────────────────────
    float edgeSoftness = mix(0.45, 0.15, uRolloff);
    float core = 1.0 - smoothstep(0.0, edgeSoftness, dist);
    float glow = 1.0 - smoothstep(edgeSoftness * 0.67, 0.5, dist);
    float alpha = (core * uCoreWeight + glow * uGlowWeight) * uAlpha;
    if (alpha < 0.01) discard;

    // ── COLOR SYSTEM ──────────────────────────────────────────────
    vec3 finalColor;

    if (uHarmonicColor > 0.5) {
        // ── CONNECTOME HARMONIC MODE ──────────────────────────────
        if (uHarmonicColorMode > 0.5) {
            // Kuramoto: colour each dot by its oscillator phase, so
            // synchronized (connected/nearby) regions share a hue and
            // phase patterns flow across the connectome.
            finalColor = phaseWheel(vHarmonicU);
        } else {
            // Eigen/mix standing wave: coolwarm by signed field. Gain
            // spreads the mostly-small eigenmode values (mean |ψ|≈0.12).
            finalColor = coolwarm(vHarmonic * 2.4);
        }
    } else if (uColorMode > 0.5) {
        // ── COLOR MODE: Plutchik Emotion Wheel ────────────────────
        //
        // Map (valence, arousal) → hue via the Plutchik circumplex.
        // Saturation scales with emotional intensity.
        // Lightness modulated by dominance.

        float sentAbs = abs(uSentiment);

        // Effective arousal: blend SER arousal with text-based emotionalIntensity.
        // When SER isn't running, emotionalIntensity from the classifier
        // distinguishes angry (high) from sad (low), giving us color diversity.
        // When SER IS running, its arousal takes precedence.
        float effectiveArousal = max(uEmotionArousal, uEmotionalIntensity);

        // Emotional strength: how far from neutral are we?
        float emotionStrength = max(sentAbs, effectiveArousal * 0.6);

        // Compute hue from Plutchik emotion wheel using effective arousal
        float hue = emotionHue(uSentiment, effectiveArousal);

        // Saturation: near-neutral = desaturated, strong emotion = vivid.
        // With additive blending on a dark background, overlapping particles
        // wash out high-lightness colors to white. Pushing saturation to 0.95
        // and using a steeper ramp keeps hues vivid even at particle density.
        float sat = smoothstep(0.0, 0.25, emotionStrength) * 0.95;

        // Lightness: base 0.50 (was 0.65 — too pastel with additive blending).
        // Lower base preserves hue identity when particles overlap.
        //   - dominance pushes brighter (dominant) or dimmer (submissive)
        //   - strong emotion brightens moderately
        float lit = 0.50 + uEmotionDominance * 0.10 + emotionStrength * 0.08;

        // Per-particle hue variation for organic feel (very subtle ±0.03)
        float hueVariation = (vUV.x * 0.37 + vUV.y * 0.23 + uTime * 0.02);
        hue += (fract(hueVariation) - 0.5) * 0.06;

        finalColor = hsl2rgb(fract(hue), sat, lit);
    } else {
        // ── WHITE MODE: tension-driven warm ↔ cool baseline ───────
        vec3 warmBase = vec3(1.0, 0.95, 0.88);    // slightly golden
        vec3 coolBase = vec3(0.88, 0.93, 1.0);     // slightly icy
        finalColor = mix(warmBase, coolBase, uTension);

        // Subtle sentiment overlay in white mode too (max 15%)
        float sentStrength = abs(uSentiment) * 0.15;
        vec3 sentShift = uSentiment > 0.0
            ? vec3(1.0, 0.97, 0.9)    // positive = warm
            : vec3(0.9, 0.93, 1.0);   // negative = cool
        finalColor = mix(finalColor, sentShift, sentStrength);
    }

    // ── ENERGY GLOW (both modes) ──────────────────────────────────
    // Louder voice = brighter particles (up to 30% boost)
    float energyGlow = 1.0 + uEnergy * 0.3;
    finalColor *= energyGlow;

    // Color modulation: core area full brightness, outer glow dimmer
    gl_FragColor = vec4(finalColor * uBrightness * (core * 0.5 + 0.5), alpha);
}
