/**
 * useThreeScene — Three.js scene, camera, renderer, and particle system lifecycle.
 *
 * Handles:
 * - Scene + camera + renderer creation
 * - ParticleSystem initialization with GPUComputationRenderer
 * - UniformBridge + SemanticBackend wiring
 * - Animation loop (60fps) with motion blur, camera updates, pointer interaction
 * - WebGL context loss recovery (bumps canvasKey)
 * - Full cleanup on unmount
 */

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { ParticleSystem } from '../engine/ParticleSystem';
import { HarmonicsManager } from '../engine/HarmonicsManager';
import { UniformBridge } from '../engine/UniformBridge';
import { SemanticBackend } from '../services/SemanticBackend';
import { SERManager } from '../audio/SERManager';
import type { CameraType } from '../components/TuningPanel';
import type { Singletons } from './useSingletons';

export interface ThreeSceneRefs {
    particleSystem: React.MutableRefObject<ParticleSystem | null>;
    uniformBridge: React.MutableRefObject<UniformBridge | null>;
    semanticBackend: React.MutableRefObject<SemanticBackend | null>;
    serManager: React.MutableRefObject<SERManager | null>;
}

export function useThreeScene(
    canvasRef: React.RefObject<HTMLCanvasElement | null>,
    canvasKey: number,
    setCanvasKey: React.Dispatch<React.SetStateAction<number>>,
    cameraType: CameraType,
    singletons: Singletons,
): ThreeSceneRefs {
    const particleSystemRef = useRef<ParticleSystem | null>(null);
    const uniformBridgeRef = useRef<UniformBridge | null>(null);
    const semanticBackendRef = useRef<SemanticBackend | null>(null);
    const serManagerRef = useRef<SERManager | null>(null);
    const animationFrameIdRef = useRef<number>(0);
    const raycasterRef = useRef(new THREE.Raycaster());
    const pointerRef = useRef(new THREE.Vector2());
    const isPointerActiveRef = useRef(false);

    const {
        audioEngine, speechEngine, tuningConfig, classifier,
        workspaceEngine, sessionLogger, serverClient,
    } = singletons;

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        // ── SCENE SETUP ──────────────────────────────────────────────────────
        const scene = new THREE.Scene();

        // Motion blur: orthographic quad rendered before every particle pass.
        const fadeScene = new THREE.Scene();
        const fadeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
        const fadeMaterial = new THREE.MeshBasicMaterial({
            color: 0x1a1a1a,
            transparent: true,
            opacity: 0.08,
        });
        const fadePlane = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), fadeMaterial);
        fadeScene.add(fadePlane);

        // ── CAMERA ────────────────────────────────────────────────────────
        const initialZ = tuningConfig.get('cameraZ');
        const aspect = window.innerWidth / window.innerHeight;

        let camera: THREE.PerspectiveCamera | THREE.OrthographicCamera;
        if (cameraType === 'orthographic') {
            const frustumHalf = initialZ * Math.tan((75 / 2) * (Math.PI / 180));
            camera = new THREE.OrthographicCamera(
                -frustumHalf * aspect, frustumHalf * aspect,
                frustumHalf, -frustumHalf,
                0.1, 1000,
            );
        } else {
            camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        }
        camera.position.z = initialZ;

        // ── RENDERER ─────────────────────────────────────────────────────────
        let renderer: THREE.WebGLRenderer;
        try {
            renderer = new THREE.WebGLRenderer({
                canvas,
                antialias: true,
                failIfMajorPerformanceCaveat: false,
            });
        } catch (e) {
            console.error('WebGL context creation failed — bumping canvas key for recovery:', e);
            setCanvasKey(k => k + 1);
            return;
        }

        renderer.setSize(window.innerWidth, window.innerHeight);
        // Cap pixel ratio on mobile to reduce GPU memory pressure.
        // iPhone 14/15/16 report DPR=3, producing ~4.7M-pixel framebuffers
        // that push past iOS Safari's ~300 MB WebGL ceiling when combined
        // with 16K-particle GPU compute. DPR=2 is visually identical on phone screens.
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, isMobile ? 2 : window.devicePixelRatio));
        renderer.setClearColor(0x1a1a1a);
        renderer.autoClear = false;

        // Listen for GPU context loss mid-session and trigger recovery
        const handleContextLost = (e: Event) => {
            e.preventDefault(); // Allow context restoration attempt
            console.warn('WebGL context lost — bumping canvas key for recovery');
            setCanvasKey(k => k + 1);
        };
        canvas.addEventListener('webglcontextlost', handleContextLost);

        // ── PARTICLE SYSTEM ───────────────────────────────────────────────────
        let particles: ParticleSystem;
        try {
            particles = new ParticleSystem(renderer, tuningConfig, 128);
        } catch (e) {
            console.error('ParticleSystem init failed — bumping canvas key for recovery:', e);
            renderer.dispose();
            renderer.forceContextLoss();
            setCanvasKey(k => k + 1);
            return;
        }
        scene.add(particles.particles);
        particleSystemRef.current = particles;

        // ── CONNECTOME HARMONICS ──────────────────────────────────────
        // Loads graph-Laplacian eigenmodes of a structural connectome and
        // drives the dots as an oscillating standing-wave field. Off by
        // default; toggle with the "h" key (see keyboard shortcuts below).
        const harmonicsManager = new HarmonicsManager(
            particles.getVelocityUniforms(), particles.getRenderUniforms(), particles.size,
        );
        let harmonicMode = 3;
        let harmonicTargetBeforeEnable = particles.currentTarget;
        harmonicsManager.load()
            .then(() => {
                harmonicsManager.setMode(harmonicMode);
                console.log('[useThreeScene] connectome harmonics loaded — press "h" to toggle');
            })
            .catch((e) => console.warn('[useThreeScene] harmonics load failed:', e));
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (window as any).__harmonics = harmonicsManager;

        // ── UNIFORM BRIDGE ────────────────────────────────────────────────────
        const uniformBridge = new UniformBridge(audioEngine, particles, tuningConfig, workspaceEngine);
        uniformBridgeRef.current = uniformBridge;

        // ── SEMANTIC BACKEND ──────────────────────────────────────────
        const semanticBackend = new SemanticBackend(
            speechEngine, classifier, particles, uniformBridge,
            sessionLogger, serverClient, audioEngine, tuningConfig,
        );
        semanticBackendRef.current = semanticBackend;

        // ── SER MANAGER (Speech Emotion Recognition) ──────────────
        // Bridges AudioEngine's mic stream → SER Web Worker for
        // prosodic emotion detection. Polls until AudioEngine is ready
        // (i.e., user has clicked the mic button).
        const serManager = new SERManager(audioEngine, uniformBridge);
        serManagerRef.current = serManager;
        let serPollCount = 0;
        const serPollId = setInterval(() => {
            if (serManager.active) {
                clearInterval(serPollId);
                return;
            }
            serManager.start();
            serPollCount++;
            if (serPollCount > 60) { // Give up after ~2 minutes
                clearInterval(serPollId);
                console.warn('[SER Manager] Gave up waiting for AudioEngine after 2 min');
            }
        }, 2000);

        // ── SERVER WARM-UP ────────────────────────────────────────────
        serverClient?.warmUp();

        // Expose on window for console debugging
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (window as any).__particles = particles;
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        (window as any).__semantic = semanticBackend;

        // ── RESIZE HANDLER ────────────────────────────────────────────────────
        const handleResize = () => {
            const newAspect = window.innerWidth / window.innerHeight;
            if (camera instanceof THREE.PerspectiveCamera) {
                camera.aspect = newAspect;
            } else {
                const frustumHalf = camera.position.z * Math.tan((75 / 2) * (Math.PI / 180));
                camera.left = -frustumHalf * newAspect;
                camera.right = frustumHalf * newAspect;
                camera.top = frustumHalf;
                camera.bottom = -frustumHalf;
            }
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            if (particleSystemRef.current) {
                particleSystemRef.current.resize();
            }
        };
        window.addEventListener('resize', handleResize);

        // ── INTERACTION HANDLERS ──────────────────────────────────────────────
        const updatePointer = (clientX: number, clientY: number) => {
            pointerRef.current.x = (clientX / window.innerWidth) * 2 - 1;
            pointerRef.current.y = -(clientY / window.innerHeight) * 2 + 1;
            isPointerActiveRef.current = true;
        };
        const handleMouseMove = (e: MouseEvent) => updatePointer(e.clientX, e.clientY);
        const handleTouchMove = (e: TouchEvent) => {
            if (e.touches.length > 0) updatePointer(e.touches[0].clientX, e.touches[0].clientY);
        };
        const handlePointerLeave = () => {
            isPointerActiveRef.current = false;
            particleSystemRef.current?.setPointer(new THREE.Vector3(9999, 9999, 9999), false);
        };

        window.addEventListener('mousemove', handleMouseMove);
        window.addEventListener('touchstart', handleTouchMove, { passive: true });
        window.addEventListener('touchmove', handleTouchMove);
        window.addEventListener('mouseup', handlePointerLeave);
        window.addEventListener('touchend', handlePointerLeave);

        // ── KEYBOARD SHORTCUTS FOR SHAPE CYCLING (DEV TOOL) ───────────────────
        const shapeKeys: Record<string, string> = {
            '1': 'ring', '2': 'sphere', '3': 'quadruped', '4': 'humanoid',
            '5': 'scatter', '6': 'dual-attract', '7': 'wave', '8': 'starburst',
            '9': 'tree', '0': 'mountain', '-': 'building', '=': 'bird',
        };
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

            // Connectome harmonics controls:
            //   h       → toggle oscillation (forms/leaves the connectome layout)
            //   [ / ]   → previous / next single harmonic mode
            //   m       → low-mode standing-wave superposition
            if (e.key === 'h' && harmonicsManager.harmonics.loaded && particleSystemRef.current) {
                const nowActive = harmonicsManager.toggle();
                if (nowActive) {
                    harmonicTargetBeforeEnable = particleSystemRef.current.currentTarget;
                    particleSystemRef.current.setTargetTexture(harmonicsManager.connectomeTarget, 'connectome');
                } else {
                    const restore = shapeKeys[Object.keys(shapeKeys)[0]];
                    particleSystemRef.current.setTarget(
                        harmonicTargetBeforeEnable && harmonicTargetBeforeEnable !== 'connectome'
                            ? harmonicTargetBeforeEnable
                            : (restore ?? 'ring'),
                    );
                }
                return;
            }
            if ((e.key === '[' || e.key === ']') && harmonicsManager.harmonics.loaded) {
                harmonicMode += e.key === ']' ? 1 : -1;
                if (harmonicMode < 0) harmonicMode = 0;
                harmonicsManager.setMode(harmonicMode);
                console.log(`[harmonics] ${harmonicsManager.harmonics.selection}`);
                return;
            }
            if (e.key === 'm' && harmonicsManager.harmonics.loaded) {
                harmonicsManager.setMix([0, 1, 2, 3], undefined, [0, Math.PI / 2, Math.PI, Math.PI / 3]);
                console.log(`[harmonics] ${harmonicsManager.harmonics.selection}`);
                return;
            }

            const shapeName = shapeKeys[e.key];
            if (shapeName && particleSystemRef.current) {
                particleSystemRef.current.setTarget(shapeName);
            }
        };
        window.addEventListener('keydown', handleKeyDown);

        // ── ANIMATION LOOP ────────────────────────────────────────────────────
        let lastTime = performance.now();
        const planeZ = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
        const targetVec = new THREE.Vector3();

        let logAudioAccum = 0;
        let logWorkspaceAccum = 0;

        const animate = () => {
            animationFrameIdRef.current = requestAnimationFrame(animate);

            const now = performance.now();
            const dt = Math.min((now - lastTime) / 1000, 0.1);
            lastTime = now;

            // Periodic session logging
            logAudioAccum += dt;
            logWorkspaceAccum += dt;
            if (logAudioAccum >= 0.2) {
                logAudioAccum = 0;
                const f = audioEngine.getFeatures();
                sessionLogger?.log('audio', {
                    energy: f.energy, tension: f.tension, urgency: f.urgency,
                    breathiness: f.breathiness, flatness: f.flatness,
                });
            }
            if (logWorkspaceAccum >= 0.5) {
                logWorkspaceAccum = 0;
                const ws = workspaceEngine?.getState();
                if (ws) {
                    sessionLogger?.log('workspace', {
                        coherence: ws.coherence, entropy: ws.entropy, arousal: ws.arousal,
                        abstractionLevel: ws.abstractionLevel, dominantConcept: ws.dominantConcept,
                        timeSinceLastUtterance: ws.timeSinceLastUtterance,
                    });
                }
            }

            // WorkspaceEngine updates cognitive state
            workspaceEngine?.update(
                dt,
                audioEngine.getFeatures(),
                semanticBackendRef.current?.lastState || null,
            );

            // ── SPLIT-PHASE UPDATE PROTOCOL ──────────────────────────────
            // Phase 1: ParticleSystem writes config baselines to uniforms.
            // Phase 2: SemanticBackend + UniformBridge apply modulations on top.
            // Phase 3: ParticleSystem runs GPU compute with fully modulated values.
            //
            // Previously, ParticleSystem.update() did phases 1+3 together,
            // which meant UniformBridge's emotion/transition overrides were
            // silently overwritten before the GPU ever saw them.
            if (particleSystemRef.current) {
                // Phase 1: Write config baselines (spring, drag, noise, etc.)
                particleSystemRef.current.writeConfigUniforms(dt);
            }

            // Phase 2: Semantic pipeline sets overrides, UniformBridge modulates
            semanticBackendRef.current?.update(dt);
            uniformBridgeRef.current?.update();

            // Refresh the connectome-harmonic field texture for this frame
            // (uses the sim clock so oscillation is framerate-independent).
            if (particleSystemRef.current) {
                harmonicsManager.update(particleSystemRef.current.time);
            }

            // Update camera Z from TuningConfig slider
            const z = tuningConfig.get('cameraZ');
            if (camera.position.z !== z) {
                camera.position.z = z;
                if (camera instanceof THREE.OrthographicCamera) {
                    const fH = z * Math.tan((75 / 2) * (Math.PI / 180));
                    const a = window.innerWidth / window.innerHeight;
                    camera.left = -fH * a;
                    camera.right = fH * a;
                    camera.top = fH;
                    camera.bottom = -fH;
                    camera.updateProjectionMatrix();
                }
            }

            if (particleSystemRef.current) {
                if (isPointerActiveRef.current) {
                    raycasterRef.current.setFromCamera(pointerRef.current, camera);
                    raycasterRef.current.ray.intersectPlane(planeZ, targetVec);
                    particleSystemRef.current.setPointer(targetVec, true);
                }
                // Phase 3: GPU compute sees fully modulated uniforms
                particleSystemRef.current.computeAndRender();
            }

            // Render: motion blur fade → clear depth → particles
            fadeMaterial.opacity = tuningConfig.get('trailLength');
            renderer.render(fadeScene, fadeCamera);
            renderer.clearDepth();
            renderer.render(scene, camera);
        };

        animate();

        // ── CLEANUP ───────────────────────────────────────────────────────────
        return () => {
            cancelAnimationFrame(animationFrameIdRef.current);

            window.removeEventListener('resize', handleResize);
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('touchstart', handleTouchMove);
            window.removeEventListener('touchmove', handleTouchMove);
            window.removeEventListener('mouseup', handlePointerLeave);
            window.removeEventListener('touchend', handlePointerLeave);
            window.removeEventListener('keydown', handleKeyDown);
            canvas.removeEventListener('webglcontextlost', handleContextLost);

            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            delete (window as any).__particles;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            delete (window as any).__semantic;
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            delete (window as any).__harmonics;
            harmonicsManager.dispose();

            if (semanticBackendRef.current) {
                semanticBackendRef.current.dispose();
                semanticBackendRef.current = null;
            }

            fadeMaterial.dispose();
            fadePlane.geometry.dispose();

            if (particleSystemRef.current) {
                particleSystemRef.current.dispose();
                particleSystemRef.current = null;
            }

            clearInterval(serPollId);
            serManager.stop();
            serManagerRef.current = null;

            renderer.dispose();
            renderer.forceContextLoss();
        };
        // canvasKey / cameraType change triggers a full teardown + remount.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [canvasKey, cameraType]);

    return { particleSystem: particleSystemRef, uniformBridge: uniformBridgeRef, semanticBackend: semanticBackendRef, serManager: serManagerRef };
}
