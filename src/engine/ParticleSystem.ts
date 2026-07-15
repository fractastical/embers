import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/examples/jsm/misc/GPUComputationRenderer.js';
import positionFrag from '../shaders/position.frag.glsl?raw';
import velocityFrag from '../shaders/velocity.frag.glsl?raw';
import renderVert from '../shaders/render.vert.glsl?raw';
import renderFrag from '../shaders/render.frag.glsl?raw';
import { MorphTargets } from './MorphTargets';
import { TuningConfig, IS_MOBILE } from '../services/TuningConfig';
import { buildMotionPlanShader } from './particle-system-extensions';

export class ParticleSystem {
    renderer: THREE.WebGLRenderer;
    gpuCompute: GPUComputationRenderer;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- GPUComputationRenderer variable type is not exported
    positionVariable: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    velocityVariable: any;
    particles: THREE.Points;
    size: number;
    time: number;
    morphTargets: MorphTargets;

    // Track which morph target is currently active.
    // External systems (SemanticBackend, debug UI) can read this
    // to know the current shape without querying the texture.
    currentTarget: string = 'ring';

    // TuningConfig reference — used to read parameter values every frame.
    // This is the pattern used by game engines: a central config that
    // all subsystems poll, rather than prop-drilling individual values.
    private config: TuningConfig;

    constructor(renderer: THREE.WebGLRenderer, config: TuningConfig, size: number = 128) {
        this.renderer = renderer;
        this.config = config;
        this.size = size;
        this.time = 0;
        this.morphTargets = new MorphTargets(size);

        // Initialize GPUComputationRenderer
        this.gpuCompute = new GPUComputationRenderer(size, size, renderer);

        // iOS Safari cannot render to float32 textures — use half-float instead.
        // HalfFloatType (16-bit) provides sufficient precision for particle
        // positions and velocities while being universally supported on mobile GPUs.
        if (IS_MOBILE) {
            this.gpuCompute.setDataType(THREE.HalfFloatType);
            console.log('[ParticleSystem] 📱 Using HalfFloatType for mobile GPU compatibility');
        }

        // Create initial textures
        const dtPosition = this.gpuCompute.createTexture();
        const dtVelocity = this.gpuCompute.createTexture();
        this.initTextures(dtPosition, dtVelocity);

        // Add variables — use enhanced velocity shader with A2 motion plan functions
        const enhancedVelocityFrag = buildMotionPlanShader(velocityFrag);
        this.velocityVariable = this.gpuCompute.addVariable('textureVelocity', enhancedVelocityFrag, dtVelocity);
        this.positionVariable = this.gpuCompute.addVariable('texturePosition', positionFrag, dtPosition);

        // Dependencies
        this.gpuCompute.setVariableDependencies(this.velocityVariable, [this.positionVariable, this.velocityVariable]);
        this.gpuCompute.setVariableDependencies(this.positionVariable, [this.positionVariable, this.velocityVariable]);

        // Get initial morph target (ring) from the pre-baked cache
        const tMorphTarget = this.morphTargets.getTarget('ring');

        // Uniforms
        this.velocityVariable.material.uniforms.uTime = { value: 0.0 };
        this.velocityVariable.material.uniforms.uNoiseAmplitude = { value: 0.25 };
        this.velocityVariable.material.uniforms.uNoiseFrequency = { value: 0.8 };
        this.velocityVariable.material.uniforms.uDrag = { value: 2.5 };
        this.velocityVariable.material.uniforms.tMorphTarget = { value: tMorphTarget };
        this.velocityVariable.material.uniforms.uSpringK = { value: 1.5 };
        this.velocityVariable.material.uniforms.uFormationScale = { value: config.get('formationScale') };
        this.velocityVariable.material.uniforms.uAbstraction = { value: 0.0 };

        // Feature Uniforms
        this.velocityVariable.material.uniforms.uEnergy = { value: 0.0 };
        this.velocityVariable.material.uniforms.uTension = { value: 0.0 };
        this.velocityVariable.material.uniforms.uUrgency = { value: 0.0 };
        this.velocityVariable.material.uniforms.uBreathiness = { value: 0.0 };
        this.velocityVariable.material.uniforms.uTextureComplexity = { value: 0.0 };

        // Curve shaping mode uniforms — toggled from sidebar
        this.velocityVariable.material.uniforms.uEnergyCurveMode = { value: 0.0 };
        this.velocityVariable.material.uniforms.uUrgencyCurveMode = { value: 0.0 };
        this.velocityVariable.material.uniforms.uUrgencyThresholdLow = { value: 0.3 };
        this.velocityVariable.material.uniforms.uUrgencyThresholdHigh = { value: 0.8 };

        // New Uniforms for Breathing & Interaction
        this.velocityVariable.material.uniforms.uBreathingAmplitude = { value: 0.08 };
        this.velocityVariable.material.uniforms.uPointerPos = { value: new THREE.Vector3(9999, 9999, 9999) };
        this.velocityVariable.material.uniforms.uPointerActive = { value: 0.0 };

        // Repulsion uniforms — driven by TuningConfig for real-time adjustment.
        // These replace the hardcoded values that were previously in the shader.
        this.velocityVariable.material.uniforms.uRepulsionRadius = { value: config.get('repulsionRadius') };
        this.velocityVariable.material.uniforms.uRepulsionStrength = { value: config.get('repulsionStrength') };

        // Sentiment Movement — LMA-based physics modulation driven by speech sentiment.
        this.velocityVariable.material.uniforms.uSentimentMovement = { value: 0.0 };
        this.velocityVariable.material.uniforms.uSentimentMovementIntensity = { value: config.get('sentimentMovementIntensity') };

        // Connectome harmonics uniforms — driven by HarmonicsManager.
        // A 1×1 zero texture is bound by default so the sampler is always
        // valid; uHarmonicActive=0 keeps the code path inert until enabled.
        const harmonicDefault = new THREE.DataTexture(
            new Float32Array([0, 0, 0, 0]), 1, 1, THREE.RGBAFormat, THREE.FloatType,
        );
        harmonicDefault.needsUpdate = true;
        this.velocityVariable.material.uniforms.tHarmonicField = { value: harmonicDefault };
        this.velocityVariable.material.uniforms.uHarmonicActive = { value: 0.0 };
        this.velocityVariable.material.uniforms.uHarmonicAmp = { value: 0.6 };

        this.velocityVariable.material.uniforms.uDelta = { value: 0.016 };
        this.positionVariable.material.uniforms.uDelta = { value: 0.016 };

        // Initialize
        const error = this.gpuCompute.init();
        if (error !== null) {
            console.error('GPUComputationRenderer init error:', error);
            throw new Error(`GPUComputationRenderer failed to init: ${error}`);
        }
        console.log('[ParticleSystem] GPUComputationRenderer initialized OK, size:', size, '×', size, '=', size * size, 'particles');

        // Create Render Geometry
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(size * size * 3);
        const uvs = new Float32Array(size * size * 2);

        let p = 0;
        for (let j = 0; j < size; j++) {
            for (let i = 0; i < size; i++) {
                uvs[p * 2] = i / (size - 1);
                uvs[p * 2 + 1] = j / (size - 1);
                positions[p * 3] = 0;
                positions[p * 3 + 1] = 0;
                positions[p * 3 + 2] = 0;
                p++;
            }
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

        // Render Material
        const material = new THREE.ShaderMaterial({
            uniforms: {
                texturePosition: { value: null },
                textureVelocity: { value: null },
                uColor: { value: new THREE.Color(1.0, 1.0, 1.0) },
                uAlpha: { value: 0.7 },
                uPointSize: { value: 3.0 },
                uColorMode: { value: 0.0 },  // 0.0 = white, 1.0 = color (sentiment)
                uTime: { value: 0.0 },       // For subtle hue animation
                uRolloff: { value: 0.5 },    // Spectral rolloff → edge softness
                uTension: { value: 0.0 },    // Spectral centroid → warm/cool color
                uEnergy: { value: 0.0 },      // RMS → brightness boost
                uSentiment: { value: 0.0 },   // Sentiment color shift (−1 to +1) → maps to valence
                uEmotionalIntensity: { value: 0.0 }, // Emotional intensity (0=sad, 1=angry)
                uEmotionArousal: { value: 0.0 },     // SER arousal (0=calm, 1=excited)
                uEmotionDominance: { value: 0.0 },    // SER dominance (0=submissive, 1=dominant)
                uBrightness: { value: 1.0 },   // Overall brightness multiplier
                uCoreWeight: { value: 0.8 },   // Core dot intensity
                uGlowWeight: { value: 0.4 },   // Glow halo intensity
                // Connectome harmonics coloring — shares the field texture
                // with the velocity shader; HarmonicsManager toggles the mode.
                tHarmonicField: { value: harmonicDefault },
                uHarmonicColor: { value: 0.0 },
                uHarmonicColorMode: { value: 1.0 },  // 0 = coolwarm, 1 = phase wheel
            },
            vertexShader: renderVert,
            fragmentShader: renderFrag,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });

        this.particles = new THREE.Points(geometry, material);
        this.particles.matrixAutoUpdate = false;
        this.particles.updateMatrix();
    }

    initTextures(texturePosition: THREE.DataTexture, textureVelocity: THREE.DataTexture) {
        const posArray = texturePosition.image.data;
        const velArray = textureVelocity.image.data;

        if (!posArray || !velArray) return;

        for (let k = 0, kl = posArray.length; k < kl; k += 4) {
            // Position: ring formation
            const theta = Math.random() * Math.PI * 2;
            const r = 2.5 + Math.random() * 0.5;
            const x = r * Math.cos(theta);
            const y = r * Math.sin(theta);
            const z = (Math.random() - 0.5) * 0.2;

            posArray[k + 0] = x;
            posArray[k + 1] = y;
            posArray[k + 2] = z;
            posArray[k + 3] = Math.random();

            // Velocity: zero initial
            velArray[k + 0] = 0;
            velArray[k + 1] = 0;
            velArray[k + 2] = 0;
            velArray[k + 3] = 0;
        }
    }

    /**
     * Phase 1: Write config baselines to shader uniforms.
     *
     * This sets the "ground truth" from TuningConfig sliders BEFORE
     * UniformBridge applies its modulations (emotion overrides, transition
     * choreography, audio features). Must run before UniformBridge.update().
     *
     * Previously this was combined with compute() in a single update(),
     * which caused UniformBridge's writes to be silently overwritten
     * by the config baseline before the GPU ever saw them.
     */
    writeConfigUniforms(deltaTime: number): void {
        this.time += deltaTime;

        const velUniforms = this.velocityVariable.material.uniforms;
        const renderUniforms = (this.particles.material as THREE.ShaderMaterial).uniforms;

        // Physics uniforms (velocity shader)
        velUniforms.uSpringK.value = this.config.get('springK');
        velUniforms.uDrag.value = this.config.get('drag');
        velUniforms.uNoiseAmplitude.value = this.config.get('noiseAmplitude');
        velUniforms.uNoiseFrequency.value = this.config.get('noiseFrequency');
        velUniforms.uBreathingAmplitude.value = this.config.get('breathingAmplitude');
        velUniforms.uAbstraction.value = this.config.get('abstraction');

        // Pointer interaction uniforms
        velUniforms.uRepulsionRadius.value = this.config.get('repulsionRadius');
        velUniforms.uRepulsionStrength.value = this.config.get('repulsionStrength');

        // Formation scale — scales all morph target positions
        velUniforms.uFormationScale.value = this.config.get('formationScale');

        // Appearance uniforms (render shader)
        renderUniforms.uPointSize.value = this.config.get('pointSize');
        renderUniforms.uAlpha.value = this.config.get('pointOpacity');
        renderUniforms.uBrightness.value = this.config.get('pointBrightness');
        renderUniforms.uCoreWeight.value = this.config.get('coreWeight');
        renderUniforms.uGlowWeight.value = this.config.get('glowWeight');

        // Time and delta
        velUniforms.uTime.value = this.time;
        renderUniforms.uTime.value = this.time;
        this.positionVariable.material.uniforms.uDelta.value = deltaTime;
        velUniforms.uDelta.value = deltaTime;
    }

    /**
     * Phase 2: Run GPU compute and update render textures.
     *
     * Call this AFTER UniformBridge.update() has applied its modulations
     * on top of the config baselines. This ensures the GPU simulation
     * sees the fully modulated uniform values.
     */
    computeAndRender(): void {
        // Update GPGPU
        this.gpuCompute.compute();

        // Update Render Uniforms
        (this.particles.material as THREE.ShaderMaterial).uniforms.texturePosition.value =
            this.gpuCompute.getCurrentRenderTarget(this.positionVariable).texture;
        (this.particles.material as THREE.ShaderMaterial).uniforms.textureVelocity.value =
            this.gpuCompute.getCurrentRenderTarget(this.velocityVariable).texture;
    }

    /**
     * Combined update — writes config, then computes.
     * Kept for backward compatibility with tests and any code
     * that doesn't use the split-phase protocol.
     */
    update(deltaTime: number) {
        this.writeConfigUniforms(deltaTime);
        this.computeAndRender();
    }

    setPointer(position: THREE.Vector3, active: boolean) {
        this.velocityVariable.material.uniforms.uPointerPos.value.copy(position);
        this.velocityVariable.material.uniforms.uPointerActive.value = active ? 1.0 : 0.0;
    }

    /**
     * Switch to a named morph target.
     *
     * The spring forces in velocity.frag.glsl automatically animate the
     * transition — particles flow smoothly from their current positions
     * to the new target positions over ~1-2 seconds (controlled by uSpringK).
     *
     * @param name - One of the MORPH_TARGET_NAMES (e.g. 'quadruped', 'wave')
     */
    setTarget(name: string) {
        const texture = this.morphTargets.getTarget(name);
        this.velocityVariable.material.uniforms.tMorphTarget.value = texture;
        this.currentTarget = name;
        console.log(`[ParticleSystem] Morph target → "${name}"`);
    }

    /**
     * Set a raw DataTexture as the morph target (for server-generated shapes).
     * Bypasses MorphTargets name-based lookup.
     *
     * @param texture - RGBA Float32 DataTexture (textureSize × textureSize)
     * @param label - Optional label for logging (e.g., the noun that was requested)
     */
    setTargetTexture(texture: THREE.DataTexture, label: string = 'server') {
        this.velocityVariable.material.uniforms.tMorphTarget.value = texture;
        this.currentTarget = label;
        console.log(`[ParticleSystem] Morph target → "${label}" (server texture)`);
    }

    /**
     * Set a blended morph target interpolated between two shapes.
     *
     * Used for the abstraction spectrum: e.g. blend between a concrete
     * shape (quadruped) and a fluid shape (scatter) based on
     * abstractionLevel. The result is a temporary lerped texture.
     *
     * @param targetA - First shape (blend=0 → 100% this)
     * @param targetB - Second shape (blend=1 → 100% this)
     * @param blend   - Interpolation factor [0, 1]
     */
    blendTargets(targetA: string, targetB: string, blend: number) {
        const texture = this.morphTargets.blendTargets(targetA, targetB, blend);
        this.velocityVariable.material.uniforms.tMorphTarget.value = texture;
        this.currentTarget = `${targetA}↔${targetB}@${blend.toFixed(2)}`;
    }

    resize() {
        // No-op for now
    }

    /**
     * Get the velocity shader uniforms object.
     * Used by MotionPlanManager to add/update motion plan uniforms.
     */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- uniforms value types are heterogeneous
    getVelocityUniforms(): Record<string, { value: any }> {
        return this.velocityVariable.material.uniforms;
    }

    /**
     * Get the render (point) material uniforms object.
     * Used by HarmonicsManager to color dots by the harmonic field.
     */
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- uniforms value types are heterogeneous
    getRenderUniforms(): Record<string, { value: any }> {
        return (this.particles.material as THREE.ShaderMaterial).uniforms;
    }

    dispose() {
        // Dispose GPUComputationRenderer's internal render targets.
        // GPUComputationRenderer uses the main renderer's GL context,
        // but still creates render target textures we must free.
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- GPUComputationRenderer internals are untyped
        if (this.gpuCompute && (this.gpuCompute as unknown as Record<string, unknown>).variables) {
            const vars = (this.gpuCompute as unknown as { variables: Array<{ renderTargets?: Array<{ dispose(): void }> }> }).variables;
            for (const v of vars) {
                if (v.renderTargets) {
                    for (const rt of v.renderTargets) {
                        rt.dispose();
                    }
                }
            }
        }

        if (this.particles) {
            this.particles.geometry.dispose();
            if (Array.isArray(this.particles.material)) {
                this.particles.material.forEach(m => m.dispose());
            } else {
                this.particles.material.dispose();
            }
        }
    }
}
