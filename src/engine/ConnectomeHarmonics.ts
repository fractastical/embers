/**
 * ConnectomeHarmonics — CPU engine that turns pre-computed connectome
 * harmonics (graph-Laplacian eigenmodes of a structural connectome) into
 * a per-particle oscillating scalar field for the dot cloud.
 *
 * DATA SOURCE
 * ───────────
 * `public/connectome_harmonics.json` is produced by the separate
 * `connectome_harmonics` repo (build_connectome_data.py). It contains:
 *   - nodes3d : N×3 anatomical node coordinates (MNI RAS, millimetres)
 *   - modes   : K×N eigenvector values ψ_k(node) of the normalized Laplacian
 *   - eigvals : K eigenvalues λ_k
 *   - networks/hemi/meta : metadata (unused here)
 *
 * OSCILLATOR MODEL (ported verbatim from the source repo's viewer)
 * ───────────────────────────────────────────────────────────────
 * Each mode oscillates at its own temporal frequency derived from its
 * eigenvalue:  ω_k = 0.8 + 2.2·√λ_k.
 *
 *   single mode:  f_i(t) = ψ_k(i) · cos(t·speed·ω_k)
 *   superposition: f_i(t) = Σ_m c_m · ψ_{k_m}(i) · cos(t·speed·ω_{k_m} + φ_m)
 *
 * The raw eigenvector entries are tiny (max |ψ| ≈ 0.1), so each mode is
 * normalized to unit peak amplitude at load time; the world-space
 * displacement magnitude is controlled downstream by `uHarmonicAmp`.
 *
 * OUTPUT
 * ──────
 * The scalar field is scattered — via a precomputed dot→node map — into the
 * R channel of an RGBA float DataTexture (`fieldTexture`) sized to match the
 * particle GPGPU textures (128×128). The velocity shader samples it and
 * displaces each dot's spring target along its outward normal.
 *
 * A companion `connectomeTarget` DataTexture places the dots at the actual
 * anatomical node layout so the cloud can literally *become* the connectome
 * while it oscillates.
 */

import * as THREE from 'three';

interface HarmonicsData {
    nodes: number[][];
    nodes3d?: number[][];
    eigvals: number[];
    modes: number[][];
    networks?: string[];
    hemi?: string[];
    meta?: Record<string, unknown>;
}

/** Target radius (world units) the connectome layout is scaled to fit. */
const WORLD_RADIUS = 2.7;

/** Positional jitter (world units) applied to dots sharing a node. Kept small
 *  and center-weighted so each parcel reads as a crisp glowing node rather
 *  than a diffuse blob. */
const NODE_JITTER = 0.035;

export class ConnectomeHarmonics {
    /** Texture edge length (matches ParticleSystem size, e.g. 128). */
    readonly size: number;
    /** Total dot count (size²). */
    readonly count: number;

    /** Number of connectome nodes (parcels). */
    nodeCount = 0;
    /** Number of available harmonic modes. */
    modeCount = 0;
    /** Whether JSON has finished loading. */
    loaded = false;

    /** Per-mode eigenvector, normalized to unit peak (|ψ| ≤ 1). */
    private modes: Float32Array[] = [];
    /** Temporal angular frequency per mode: 0.8 + 2.2·√λ_k. */
    private omega: number[] = [];
    /** World-space node positions (nodeCount×3, interleaved xyz). */
    private nodePos: Float32Array = new Float32Array(0);
    /** Node index assigned to each dot (length = count). */
    private dotNode: Int32Array;
    /** Scratch buffer for the per-node field (length = nodeCount). */
    private field: Float32Array = new Float32Array(0);

    /** RGBA float texture; R channel carries the signed field in [-1, 1]. */
    readonly fieldTexture: THREE.DataTexture;
    private fieldData: Float32Array;

    /** RGBA float morph target placing dots at the anatomical layout. */
    readonly connectomeTarget: THREE.DataTexture;
    private targetData: Float32Array;

    /** Playback controls. */
    speed = 1.0;
    /** Time scrub (seconds); advanced by update(). */
    private modeIndex = 3;
    private mix: { indices: number[]; coeffs: number[]; phases: number[] } | null = null;

    constructor(size = 128) {
        this.size = size;
        this.count = size * size;
        this.dotNode = new Int32Array(this.count);

        this.fieldData = new Float32Array(this.count * 4);
        this.fieldTexture = new THREE.DataTexture(
            this.fieldData, size, size, THREE.RGBAFormat, THREE.FloatType,
        );
        this.fieldTexture.needsUpdate = true;

        this.targetData = new Float32Array(this.count * 4);
        this.connectomeTarget = new THREE.DataTexture(
            this.targetData, size, size, THREE.RGBAFormat, THREE.FloatType,
        );
        this.connectomeTarget.needsUpdate = true;
    }

    /**
     * Fetch and parse the harmonics JSON, normalize modes, and precompute
     * the world-space node layout, the dot→node map, and the connectome
     * morph target texture.
     */
    async load(url = `${import.meta.env.BASE_URL ?? '/'}connectome_harmonics.json`): Promise<void> {
        const res = await fetch(url);
        if (!res.ok) {
            throw new Error(`[ConnectomeHarmonics] fetch failed: ${res.status} ${url}`);
        }
        const data = (await res.json()) as HarmonicsData;
        this.ingest(data);
        this.loaded = true;
        console.log(
            `[ConnectomeHarmonics] loaded ${this.nodeCount} nodes, ${this.modeCount} modes`,
        );
    }

    private ingest(data: HarmonicsData): void {
        const coords = data.nodes3d ?? data.nodes.map(([x, y]) => [x, y, 0]);
        this.nodeCount = coords.length;
        this.field = new Float32Array(this.nodeCount);

        // ── Normalize modes to unit peak amplitude ──────────────────────
        this.modeCount = data.modes.length;
        this.modes = data.modes.map((mode) => {
            const arr = new Float32Array(mode);
            let peak = 0;
            for (let i = 0; i < arr.length; i++) peak = Math.max(peak, Math.abs(arr[i]));
            if (peak > 1e-9) for (let i = 0; i < arr.length; i++) arr[i] /= peak;
            return arr;
        });
        this.omega = data.eigvals.map((lambda) => 0.8 + 2.2 * Math.sqrt(Math.max(0, lambda)));

        // ── Center + scale node coords into the dot-cloud world ──────────
        // MNI RAS → three.js: x = R (left/right), y = S (up), z = A (depth).
        const centroid = [0, 0, 0];
        for (const c of coords) { centroid[0] += c[0]; centroid[1] += c[1]; centroid[2] += c[2]; }
        centroid[0] /= this.nodeCount; centroid[1] /= this.nodeCount; centroid[2] /= this.nodeCount;

        this.nodePos = new Float32Array(this.nodeCount * 3);
        let maxR = 1e-6;
        for (let n = 0; n < this.nodeCount; n++) {
            const x = coords[n][0] - centroid[0];
            const s = coords[n][2] - centroid[2];
            const a = coords[n][1] - centroid[1];
            this.nodePos[n * 3 + 0] = x;   // R → x
            this.nodePos[n * 3 + 1] = s;   // S → y (up)
            this.nodePos[n * 3 + 2] = a;   // A → z (depth)
            maxR = Math.max(maxR, Math.hypot(x, s, a));
        }
        const scale = WORLD_RADIUS / maxR;
        for (let i = 0; i < this.nodePos.length; i++) this.nodePos[i] *= scale;

        // ── Assign every dot to a node (even blocks) + build morph target ─
        // Deterministic per-dot jitter keeps clusters volumetric but stable.
        for (let i = 0; i < this.count; i++) {
            const node = Math.min(this.nodeCount - 1, Math.floor((i * this.nodeCount) / this.count));
            this.dotNode[i] = node;
            this.targetData[i * 4 + 0] = this.nodePos[node * 3 + 0] + this.gauss(i, 11) * NODE_JITTER;
            this.targetData[i * 4 + 1] = this.nodePos[node * 3 + 1] + this.gauss(i, 29) * NODE_JITTER;
            this.targetData[i * 4 + 2] = this.nodePos[node * 3 + 2] + this.gauss(i, 47) * NODE_JITTER;
            this.targetData[i * 4 + 3] = 1;
        }
        this.connectomeTarget.needsUpdate = true;
    }

    /** Cheap deterministic pseudo-random in [0,1) from an integer + salt. */
    private hash(i: number, salt: number): number {
        const x = Math.sin((i + 1) * 12.9898 + salt * 78.233) * 43758.5453;
        return x - Math.floor(x);
    }

    /**
     * Deterministic center-weighted noise in roughly [-1, 1] (sum of four
     * uniforms → approximately Gaussian), so jittered dots concentrate near
     * the node center and clusters read as crisp points, not diffuse blobs.
     */
    private gauss(i: number, salt: number): number {
        const s = this.hash(i, salt) + this.hash(i, salt + 101)
            + this.hash(i, salt + 211) + this.hash(i, salt + 307);
        return (s / 4 - 0.5) * 2;
    }

    // ── Playback selection ────────────────────────────────────────────

    /** Oscillate a single harmonic mode (0-indexed). */
    setMode(index: number): void {
        this.modeIndex = Math.max(0, Math.min(this.modeCount - 1, index));
        this.mix = null;
    }

    /**
     * Oscillate a weighted superposition of modes (standing-wave field).
     * @param indices - mode indices to combine
     * @param coeffs  - per-mode amplitude (defaults to 1/(rank+1) falloff)
     * @param phases  - per-mode temporal phase offset in radians (defaults 0)
     */
    setMix(indices: number[], coeffs?: number[], phases?: number[]): void {
        const idx = indices.filter((k) => k >= 0 && k < this.modeCount);
        this.mix = {
            indices: idx,
            coeffs: coeffs ?? idx.map((_, r) => 1 / (r + 1)),
            phases: phases ?? idx.map(() => 0),
        };
    }

    /** Current selection description (for HUD/debug). */
    get selection(): string {
        return this.mix
            ? `mix[${this.mix.indices.map((k) => k + 1).join(',')}]`
            : `mode ${this.modeIndex + 1}`;
    }

    // ── Per-frame field evaluation ──────────────────────────────────────

    /**
     * Evaluate the harmonic field at time `t` (seconds) and scatter it into
     * the field texture's R channel via the dot→node map.
     */
    update(t: number): void {
        if (!this.loaded) return;
        const field = this.field;

        if (this.mix && this.mix.indices.length > 0) {
            field.fill(0);
            const { indices, coeffs, phases } = this.mix;
            for (let m = 0; m < indices.length; m++) {
                const k = indices[m];
                const mode = this.modes[k];
                const wave = coeffs[m] * Math.cos(t * this.speed * this.omega[k] + phases[m]);
                for (let n = 0; n < this.nodeCount; n++) field[n] += wave * mode[n];
            }
            // Normalize the superposition to unit peak for a stable amplitude.
            let peak = 1e-9;
            for (let n = 0; n < this.nodeCount; n++) peak = Math.max(peak, Math.abs(field[n]));
            for (let n = 0; n < this.nodeCount; n++) field[n] /= peak;
        } else {
            const k = this.modeIndex;
            const mode = this.modes[k];
            const phase = Math.cos(t * this.speed * this.omega[k]);
            for (let n = 0; n < this.nodeCount; n++) field[n] = mode[n] * phase;
        }

        const out = this.fieldData;
        const dotNode = this.dotNode;
        for (let i = 0; i < this.count; i++) out[i * 4] = field[dotNode[i]];
        this.fieldTexture.needsUpdate = true;
    }

    dispose(): void {
        this.fieldTexture.dispose();
        this.connectomeTarget.dispose();
    }
}
