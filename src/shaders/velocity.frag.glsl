uniform float uTime;
uniform float uNoiseAmplitude; 
uniform float uNoiseFrequency; 
uniform float uDrag; 
uniform sampler2D tMorphTarget;
uniform float uSpringK;
uniform float uFormationScale;
uniform float uAbstraction;

// Audio Uniforms
uniform float uEnergy;
uniform float uTension;
uniform float uUrgency;
uniform float uBreathiness;
uniform float uTextureComplexity;

// Curve shaping mode uniforms
uniform float uEnergyCurveMode;     // 0.0 = linear, 1.0 = power curve
uniform float uUrgencyCurveMode;    // 0.0 = linear, 1.0 = smoothstep threshold
uniform float uUrgencyThresholdLow; // Lower edge of smoothstep (default 0.3)
uniform float uUrgencyThresholdHigh;// Upper edge of smoothstep (default 0.8)

// Breathing
uniform float uBreathingAmplitude;

// Interaction
uniform vec3 uPointerPos;
uniform float uPointerActive;
uniform float uDelta;

// Pointer repulsion — driven by TuningConfig for real-time adjustment.
uniform float uRepulsionRadius;
uniform float uRepulsionStrength;

// Sentiment Movement — modulates physics via LMA Effort framework.
// uSentimentMovement: smoothed sentiment value (−1 = angry/sad, +1 = happy)
// uSentimentMovementIntensity: user-adjustable strength (0 = off, 1 = full)
uniform float uSentimentMovement;
uniform float uSentimentMovementIntensity;

// Connectome harmonics — per-dot oscillating scalar field.
// tHarmonicField.r carries a signed field in [-1, 1] (see ConnectomeHarmonics.ts).
// The field displaces each dot's spring target along its outward normal,
// so the cloud oscillates in graph-Laplacian eigenmode fashion.
uniform sampler2D tHarmonicField;
uniform float uHarmonicActive;
uniform float uHarmonicAmp;

// [Insert Simplex Noise Functions Here - same as before]
// Simplex 3D Noise 
// by Ian McEwan, Ashima Arts
vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

float snoise(vec3 v){ 
  const vec2  C = vec2(1.0/6.0, 1.0/3.0) ;
  const vec4  D = vec4(0.0, 0.5, 1.0, 2.0);

// First corner
  vec3 i  = floor(v + dot(v, C.yyy) );
  vec3 x0 = v - i + dot(i, C.xxx) ;

// Other corners
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min( g.xyz, l.zxy );
  vec3 i2 = max( g.xyz, l.zxy );

  //  x0 = x0 - 0.0 + 0.0 * C 
  vec3 x1 = x0 - i1 + 1.0 * C.xxx;
  vec3 x2 = x0 - i2 + 2.0 * C.xxx;
  vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;

// Permutations
  i = mod(i, 289.0 ); 
  vec4 p = permute( permute( permute( 
             i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0 )) 
           + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

// Gradients
// ( N*N points uniformly over a square, mapped onto an octahedron.)
  float n_ = 1.0/7.0; // N=7
  vec3  ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);  //  mod(p,N*N)

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_ );    // mod(j,N)

  vec4 x = x_ *ns.x + ns.yyyy;
  vec4 y = y_ *ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4( x.xy, y.xy );
  vec4 b1 = vec4( x.zw, y.zw );

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);

//Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
  p0 *= norm.x;
  p1 *= norm.y;
  p2 *= norm.z;
  p3 *= norm.w;

// Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), 
                                dot(p2,x2), dot(p3,x3) ) );
}

vec3 snoiseVec3( vec3 x ){
  float s  = snoise(vec3( x ));
  float s1 = snoise(vec3( x.y - 19.1 , x.z + 33.4 , x.x + 47.2 ));
  float s2 = snoise(vec3( x.z + 74.2 , x.x - 124.5 , x.y + 99.4 ));
  return vec3( s , s1 , s2 );
}

vec3 curlNoise( vec3 p ){
  const float e = 0.001;
  vec3 dx = vec3( e   , 0.0 , 0.0 );
  vec3 dy = vec3( 0.0 , e   , 0.0 );
  vec3 dz = vec3( 0.0 , 0.0 , e   );

  vec3 p_x0 = snoiseVec3( p - dx );
  vec3 p_x1 = snoiseVec3( p + dx );
  vec3 p_y0 = snoiseVec3( p - dy );
  vec3 p_y1 = snoiseVec3( p + dy );
  vec3 p_z0 = snoiseVec3( p - dz );
  vec3 p_z1 = snoiseVec3( p + dz );

  float x = p_y1.z - p_y0.z - p_z1.y + p_z0.y;
  float y = p_z1.x - p_z0.x - p_x1.z + p_x0.z;
  float z = p_x1.y - p_x0.y - p_y1.x + p_y0.x;

  const float divisor = 1.0 / ( 2.0 * e );
  return normalize( vec3( x , y , z ) * divisor );
}

// ═══════════════════════════════════════════════════════════════════
// A2 MOTION PLAN INSERTION POINT
// primitives.glsl and motion-plan.glsl are prepended to this shader
// by ParticleSystem.ts at construction time. The functions they
// define (dispatchPrimitive, evaluateMotionPlan, etc.) are available
// here if the shader source includes them.
// ═══════════════════════════════════════════════════════════════════
// __MOTION_PLAN_FUNCTIONS_MARKER__

void main() {
    vec2 uv = gl_FragCoord.xy / resolution.xy;

    // ── READ CURRENT STATE ────────────────────────────────────────────
    vec4 selfPosition = texture2D(texturePosition, uv);
    vec4 selfVelocity = texture2D(textureVelocity, uv);

    vec3 position = selfPosition.xyz;
    vec3 velocity = selfVelocity.xyz;

    // ── CLAMP AUDIO INPUTS ────────────────────────────────────────────
    // Safety: all audio features must stay in [0, 1].
    float safeEnergy      = clamp(uEnergy, 0.0, 1.0);
    float safeTension     = clamp(uTension, 0.0, 1.0);
    float safeUrgency     = clamp(uUrgency, 0.0, 1.0);
    float safeBreathiness = clamp(uBreathiness, 0.0, 1.0);

    // ═══════════════════════════════════════════════════════════════════
    // FEATURE → VISUAL MAPPING (each feature has ONE primary effect)
    //
    //   ENERGY      → ring expansion + breathing amplitude + speed
    //   TENSION     → curl noise frequency (tighter swirls) + color*
    //   URGENCY     → noise turbulence amplitude (chaos/jitter)
    //   BREATHINESS → drag reduction + Z-axis spread (floatier, 3D)
    //
    //   * Color is handled in UniformBridge, not here.
    // ═══════════════════════════════════════════════════════════════════

    // ── SENTIMENT → MOVEMENT QUALITY (LMA Effort Framework) ──────────
    // Based on Laban Movement Analysis:
    //   Positive sentiment → Light weight, Free flow, Sustained time
    //   Mild negative      → Heavy weight, Bound flow, Sustained time
    //   Strong negative    → Strong weight, Sudden time, Direct space
    // References: Shafir et al. (2016), Chi et al. (2000) EMOTE model
    float sentMov = uSentimentMovement * uSentimentMovementIntensity;
    float sentAbs = abs(sentMov);

    float smSpringOffset  = 0.0;
    float smDragOffset    = 0.0;
    float smNoiseAmpOff   = 0.0;
    float smNoiseFreqOff  = 0.0;
    float smBreathOff     = 0.0;

    if (sentMov > 0.0) {
        // JOY profile: lighter spring, freer flow, calmer noise, bigger breath
        smSpringOffset  = -sentMov * 4.0;
        smDragOffset    = -sentMov * 3.0;
        smNoiseAmpOff   = -sentMov * 0.5;
        smNoiseFreqOff  = -sentMov * 0.5;
        smBreathOff     =  sentMov * 2.0;
    } else if (sentMov < 0.0) {
        // Negative: interpolate between SAD (mild) and ANGRY (strong)
        float angerRatio = smoothstep(0.3, 0.8, sentAbs);
        // SAD: heavy, bound, quiet, slow
        // ANGRY: firm, snappy, chaotic, tight swirls
        smSpringOffset  = mix(sentAbs * 2.5,  sentAbs * 1.5, angerRatio);
        smDragOffset    = mix(sentAbs * 2.5, -sentAbs * 1.5, angerRatio);
        smNoiseAmpOff   = mix(-sentAbs * 0.3, sentAbs * 2.5, angerRatio);
        smNoiseFreqOff  = mix(-sentAbs * 0.3, sentAbs * 1.5, angerRatio);
        smBreathOff     = mix(-sentAbs * 0.8, sentAbs * 1.0, angerRatio);
    }

    // ── SOFT CLAMPS (safety net for extreme sentiment × intensity) ────
    // Prevent overcorrection from strong words at high intensity settings.
    // Caps chosen so the effect reads as emotionally intense, not glitchy.
    smSpringOffset  = clamp(smSpringOffset, -5.0, 4.0);
    smDragOffset    = clamp(smDragOffset,   -4.0, 4.0);
    smNoiseAmpOff   = clamp(smNoiseAmpOff,  -0.8, 3.0);
    smNoiseFreqOff  = clamp(smNoiseFreqOff, -0.8, 2.0);
    smBreathOff     = clamp(smBreathOff,    -1.0, 3.0);

    // ── TENSION → CURL FREQUENCY ONLY ─────────────────────────────────
    // Higher tension = tighter curl patterns. This is the ONLY thing
    // tension does in the shader. Color shift is handled in UniformBridge.
    // The range 0.8-2.0 goes from lazy swirls to tight, nervous curls.
    float tensionFreq = uNoiseFrequency + safeTension * 1.2 + smNoiseFreqOff;

    // ── CURL NOISE COMPUTATION ────────────────────────────────────────
    // Curl noise produces divergence-free vector fields — organic swirls
    // without convergence. Time offset makes the field evolve.
    vec3 curl = curlNoise(position * tensionFreq + uTime * 0.15);

    // ── ENERGY → BREATHING + EXPANSION ────────────────────────────────
    // Energy modulates the morph target position (the particle's "home").
    // This is inherently safe because the spring always pulls particles
    // toward the target — they follow it outward but can NEVER escape.
    //
    // Three energy effects:
    //   1. Breathing amplitude: louder → bigger pulse (up to +1.5)
    //   2. Breathing speed: louder → faster pulse
    //   3. Ring expansion: louder → ring grows outward (up to 3.5 units)
    float dynamicBreathingAmp = uBreathingAmplitude + safeEnergy * 1.5 + smBreathOff;
    float breathSpeed = 0.2 + safeEnergy * 0.8;
    float phase = uTime * breathSpeed * 6.28 + uv.x * 6.28 + uv.y * 3.14;
    vec3 breathOffset = normalize(position) * sin(phase) * dynamicBreathingAmp;

    // ── MORPH TARGET (modified by energy) ─────────────────────────────
    vec3 targetPosRaw = texture2D(tMorphTarget, uv).xyz;
    // Scale the morph target by the formation scale. This is the "ring radius"
    // slider but works for ALL shapes — 1.0 = default, 0.5 = half, 2.0 = double.
    targetPosRaw *= uFormationScale;
    vec3 radialDir = normalize(targetPosRaw);
    // ── ENERGY → EXPANSION (dual-mode) ────────────────────────────
    // Linear mode: safeEnergy * 3.5 (current, responsive)
    // Power mode:  pow(energy, 1.5) * 3.0 (Prompt-4, quiets small values)
    float linearExpansion = safeEnergy * 3.5;
    float powerExpansion = pow(safeEnergy, 1.5) * 3.0;
    float energyExpansion = mix(linearExpansion, powerExpansion, uEnergyCurveMode);
    vec3 targetPos = targetPosRaw + breathOffset + radialDir * energyExpansion;

    // ── A2: MOTION PLAN DISPLACEMENT ──────────────────────────────────
    // When a motion plan is active, evaluate the assigned primitive for
    // this particle's part and add the displacement to the spring target.
    // The existing spring-damper then pulls particles toward the
    // animated target — free overshoot, settle, and follow-through.
    #ifdef MOTION_PLAN_ENABLED
    if (uMotionPlanActive > 0.5) {
        vec3 motionDisplacement = evaluateMotionPlan(uv, position, targetPosRaw, uTime);

        // Audio modulation on motion plan output
        float pitchGate = step(0.5, uPitchConfidence);
        motionDisplacement.y += uPitchDeviation * 0.3 * pitchGate;
        float arousalScale = mix(0.7, 1.5, clamp(uEmotionArousal, 0.0, 1.0));
        motionDisplacement *= arousalScale;
        motionDisplacement.y += uEmotionValence * 0.2;

        targetPos += motionDisplacement;
    }
    #endif

    // ── CONNECTOME HARMONICS DISPLACEMENT ─────────────────────────────
    // Sample this dot's harmonic field value and push its spring target
    // along the outward (radial) normal. Standing-wave eigenmodes make the
    // cloud ripple/breathe in connectome-harmonic fashion; the spring then
    // supplies organic overshoot and settle.
    if (uHarmonicActive > 0.5) {
        float harmonicField = texture2D(tHarmonicField, uv).r;
        vec3 harmonicDir = length(radialDir) > 0.0001 ? radialDir : normalize(position + 1e-4);
        targetPos += harmonicDir * harmonicField * uHarmonicAmp;
    }

    // ── BREATHINESS → Z-AXIS SPREAD ───────────────────────────────────
    // Breathy speech makes the ring "puff out" in the Z axis, creating
    // a more 3D, airy appearance. Each particle gets a unique Z offset
    // based on its UV position so the spread looks organic, not uniform.
    float zSpread = safeBreathiness * 1.2 * sin(uv.x * 13.37 + uv.y * 7.91);
    targetPos.z += zSpread;

    // ── SPRING FORCE ──────────────────────────────────────────────────
    // Hooke's Law: pulls particles toward their audio-modulated home.
    // This is the fundamental safety net — particles always return.
    //
    // A2: Pitch-based stiffness modulation — rising pitch tightens springs,
    // falling pitch loosens them (gated by confidence).
    vec3 springDir = targetPos - position;
    #ifdef MOTION_PLAN_ENABLED
    float pitchStiffness = 1.0 + uPitchDeviation * 0.3 * step(0.5, uPitchConfidence);
    float effectiveSpringK = max(0.5, (uSpringK + smSpringOffset) * pitchStiffness);
    #else
    float effectiveSpringK = max(0.5, uSpringK + smSpringOffset);
    #endif
    vec3 springF = springDir * ((1.0 - uAbstraction) * effectiveSpringK);

    // ── URGENCY → NOISE TURBULENCE ────────────────────────────────────
    // Urgency (spectral flux) drives the AMPLITUDE of curl noise.
    // High urgency = rapid speech changes, consonants, transients
    //              → particles jitter and scatter chaotically.
    // Low urgency  = sustained vowels, silence
    //              → particles drift gently.
    //
    // Base shimmer (0.06) is always present for organic life.
    // Urgency can push noise amplitude up to 1.8 — very visible chaos.
    // ── URGENCY → NOISE TURBULENCE (dual-mode) ───────────────────
    // Linear mode: safeUrgency * 1.8 (current, always responsive)
    // Smoothstep mode: gated by threshold, mild speech has NO effect
    float baseNoise = uNoiseAmplitude * 0.25;
    float linearUrgency = safeUrgency * 1.8;
    float smoothstepUrgency = smoothstep(uUrgencyThresholdLow, uUrgencyThresholdHigh, safeUrgency) * 0.8;
    float urgencyNoise = mix(linearUrgency, smoothstepUrgency, uUrgencyCurveMode);
    float abstractionNoise = uNoiseAmplitude * uAbstraction;

    // ── TEXTURE COMPLEXITY → SECOND NOISE OCTAVE ─────────────────
    // MFCCs capture vocal richness. High complexity = add higher-frequency
    // noise variations so particles shimmer with more detail.
    float textureNoise = uTextureComplexity * 0.4;

    float effectiveNoiseAmp = max(0.0, baseNoise + urgencyNoise + abstractionNoise + textureNoise + smNoiseAmpOff);
    vec3 noiseF = curl * effectiveNoiseAmp;

    // ── BREATHINESS → DRAG REDUCTION ──────────────────────────────────
    // Breathy speech makes particles floatier by reducing damping.
    // Drag range: 2.5 (default, snappy) → 0.5 (very floaty).
    // This is breathiness's OTHER visual effect (alongside Z-spread).
    float dynamicDrag = max(0.3, uDrag - safeBreathiness * 2.0 + smDragOffset);

    // ── SUM ALL FORCES ────────────────────────────────────────────────
    vec3 force = springF + noiseF;

    // ── POINTER REPULSION ─────────────────────────────────────────────
    // Radius and strength are now uniforms driven by TuningConfig,
    // so they can be adjusted in real time from the tuning panel.
    vec3 toParticle = position - uPointerPos;
    float dist = length(toParticle);

    if (dist < uRepulsionRadius && uPointerActive > 0.5) {
        vec3 repulsion = normalize(toParticle) * uRepulsionStrength
                       * smoothstep(uRepulsionRadius, 0.0, dist);
        force += repulsion;
    }

    // ── INTEGRATE VELOCITY ────────────────────────────────────────────
    velocity += force * uDelta;

    // ── DRAG (DAMPING) ────────────────────────────────────────────────
    velocity *= clamp(1.0 - dynamicDrag * uDelta, 0.0, 1.0);

    // ── VELOCITY HARD LIMIT (SAFETY) ──────────────────────────────────
    // Final safety net: cap particle speed to prevent any edge case
    // from sending particles to infinity.
    float speed = length(velocity);
    if (speed > 10.0) {
        velocity = normalize(velocity) * 10.0;
    }

    gl_FragColor = vec4(velocity, 1.0);
}
