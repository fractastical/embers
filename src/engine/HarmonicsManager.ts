/**
 * HarmonicsManager — companion to ParticleSystem (mirrors MotionPlanManager).
 *
 * Owns a ConnectomeHarmonics engine and bridges it to the velocity shader:
 *   - injects/updates the harmonics uniforms (tHarmonicField, uHarmonicActive,
 *     uHarmonicAmp)
 *   - advances the per-frame field evaluation
 *   - exposes enable/disable + mode/mix/speed/amp controls
 *
 * The velocity shader adds `tHarmonicField.r · outwardNormal · uHarmonicAmp`
 * to each dot's spring target, so the dots oscillate in connectome-harmonic
 * fashion while the existing spring-damper physics provides organic
 * overshoot and settle.
 */

import type * as THREE from 'three';
import { ConnectomeHarmonics } from './ConnectomeHarmonics';

export class HarmonicsManager {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- heterogeneous uniform values
    private uniforms: Record<string, { value: any }>;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any -- heterogeneous uniform values
    private renderUniforms: Record<string, { value: any }> | null;
    readonly harmonics: ConnectomeHarmonics;
    private active = false;

    constructor(
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- matches getVelocityUniforms()
        uniforms: Record<string, { value: any }>,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- matches getRenderUniforms()
        renderUniforms?: Record<string, { value: any }>,
        textureSize = 128,
    ) {
        this.uniforms = uniforms;
        this.renderUniforms = renderUniforms ?? null;
        this.harmonics = new ConnectomeHarmonics(textureSize);

        // Register uniforms if the shader/system hasn't already.
        this.uniforms.tHarmonicField = this.uniforms.tHarmonicField ?? { value: null };
        this.uniforms.uHarmonicActive = this.uniforms.uHarmonicActive ?? { value: 0.0 };
        this.uniforms.uHarmonicAmp = this.uniforms.uHarmonicAmp ?? { value: 0.6 };
    }

    /** Load the connectome data and bind the field texture (motion + color). */
    async load(url?: string): Promise<void> {
        await this.harmonics.load(url);
        this.uniforms.tHarmonicField.value = this.harmonics.fieldTexture;
        if (this.renderUniforms?.tHarmonicField) {
            this.renderUniforms.tHarmonicField.value = this.harmonics.fieldTexture;
        }
    }

    /** Whether the harmonics displacement is currently applied. */
    get isActive(): boolean {
        return this.active;
    }

    /** The anatomical connectome layout texture (use as a morph target). */
    get connectomeTarget(): THREE.DataTexture {
        return this.harmonics.connectomeTarget;
    }

    enable(): void {
        this.active = true;
        this.uniforms.uHarmonicActive.value = 1.0;
        if (this.renderUniforms?.uHarmonicColor) this.renderUniforms.uHarmonicColor.value = 1.0;
    }

    disable(): void {
        this.active = false;
        this.uniforms.uHarmonicActive.value = 0.0;
        if (this.renderUniforms?.uHarmonicColor) this.renderUniforms.uHarmonicColor.value = 0.0;
    }

    /** Toggle just the field coloring (independent of motion). */
    setColor(on: boolean): void {
        if (this.renderUniforms?.uHarmonicColor) this.renderUniforms.uHarmonicColor.value = on ? 1.0 : 0.0;
    }

    toggle(): boolean {
        if (this.active) this.disable();
        else this.enable();
        return this.active;
    }

    setMode(index: number): void {
        this.harmonics.setMode(index);
    }

    setMix(indices: number[], coeffs?: number[], phases?: number[]): void {
        this.harmonics.setMix(indices, coeffs, phases);
    }

    /** World-space displacement amplitude (feeds uHarmonicAmp). */
    set amp(v: number) {
        this.uniforms.uHarmonicAmp.value = v;
    }
    get amp(): number {
        return this.uniforms.uHarmonicAmp.value as number;
    }

    /** Temporal speed multiplier. */
    set speed(v: number) {
        this.harmonics.speed = v;
    }
    get speed(): number {
        return this.harmonics.speed;
    }

    /** Advance the field evaluation. Call every frame with the sim clock. */
    update(t: number): void {
        if (!this.active) return;
        this.harmonics.update(t);
    }

    dispose(): void {
        this.harmonics.dispose();
    }
}
