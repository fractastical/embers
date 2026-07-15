/**
 * SpeechEngine — Hybrid speech-to-text with Web Speech API + Deepgram fallback.
 * Lazy detection on first start(). Handles Safari isFinal bugs and iOS PWA failures.
 */

import { WebSocketSTTClient } from '../audio/WebSocketSTTClient';
import type { TranscriptEvent } from '../audio/types';
import type { AudioEngine } from './AudioEngine';

// Re-export so existing consumers (Canvas, AnalysisPanel, GhostTranscript, etc.)
// can keep importing from this module without changing their import paths.
export type { TranscriptEvent };

export type STTStatus = 'off' | 'listening' | 'restarting' | 'error' | 'unsupported' | 'connecting-ws';

type TranscriptCallback = (event: TranscriptEvent) => void;
type StatusCallback = (status: STTStatus, errorDetail?: string) => void;

interface SpeechRecognitionEvent extends Event {
    results: SpeechRecognitionResultList;
    resultIndex: number;
}

interface SpeechRecognitionResultList {
    length: number;
    item(index: number): SpeechRecognitionResult;
    [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
    isFinal: boolean;
    length: number;
    item(index: number): SpeechRecognitionAlternative;
    [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
    transcript: string;
    confidence: number;
}

interface SpeechRecognitionInstance extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start(): void;
    stop(): void;
    abort(): void;
    onresult: ((event: SpeechRecognitionEvent) => void) | null;
    onend: (() => void) | null;
    onerror: ((event: Event & { error: string }) => void) | null;
    onstart: (() => void) | null;
}

const FATAL_ERRORS = new Set(['not-allowed', 'service-not-allowed', 'language-not-supported']);
const MAX_RETRIES = 5;
const SAFARI_FINAL_TIMEOUT_MS = 750;
// After this many consecutive restarts that never produced a transcript,
// treat Web Speech as effectively unavailable. This happens in Chromium/
// Electron builds with no Google speech backend (the Cursor in-app browser,
// some Linux Chromium/Brave builds) where every session throws "network".
// Rather than loop forever with no feedback, fall back to Deepgram (if a key
// is configured) or the manual text-input UI.
const MAX_DRY_RESTARTS_BEFORE_DEGRADE = 4;

type WebSpeechProbeResult = 'untested' | 'works' | 'broken';

export class SpeechEngine {
    isSupported: boolean;
    private _isRunning = false;
    private _status: STTStatus = 'off';
    private _lastError: string = '';
    private statusListeners: Set<StatusCallback> = new Set();
    private recognition: SpeechRecognitionInstance | null = null;
    private listeners: Set<TranscriptCallback> = new Set();
    private restartTimer: ReturnType<typeof setTimeout> | null = null;
    private restartDebounceTimer: ReturnType<typeof setTimeout> | null = null;
    private intentionallyStopped = false;
    private retryCount = 0;
    private dryRestartCount = 0;
    private safariFinalTimer: ReturnType<typeof setTimeout> | null = null;
    private lastInterimText = '';
    private lastFinalText = '';
    private readonly isSafari: boolean;
    private wsClient: WebSocketSTTClient | null = null;
    private _wsTranscriptUnsub: (() => void) | null = null;
    private _wsStatusUnsub: (() => void) | null = null;
    private usingWebSocket = false;
    private webSpeechProbe: WebSpeechProbeResult = 'untested';
    private readonly deepgramApiKey: string;
    private readonly isIOS: boolean;
    private audioEngineRef: AudioEngine | null = null;

    constructor() {
        this.isSupported = 'SpeechRecognition' in window || 'webkitSpeechRecognition' in window;
        const ua = navigator.userAgent;
        this.isSafari = /Safari/.test(ua) && !/Chrome/.test(ua);
        this.isIOS = /iPhone|iPad|iPod/.test(ua);
        this.deepgramApiKey = (typeof import.meta !== 'undefined' &&
            import.meta.env?.VITE_DEEPGRAM_API_KEY) || '';

        if (!this.isSupported) {
            this._status = 'unsupported';
        }

        console.log(
            `[SpeechEngine] Web Speech API ${this.isSupported ? '✅ supported' : '❌ not supported'}` +
            (this.isSafari ? ' (Safari detected — workarounds active)' : '') +
            (this.deepgramApiKey ? ' | Deepgram fallback available' : ' | ⚠️ No VITE_DEEPGRAM_API_KEY — WebSocket fallback disabled')
        );
    }

    // ── PUBLIC API ───────────────────────────────────────────────────

    /**
     * Provide a reference to AudioEngine so that the WebSocket fallback
     * can reuse its AudioContext and MediaStream instead of creating
     * duplicates (critical on iOS where extra contexts crash the tab).
     */
    setAudioEngine(audioEngine: AudioEngine): void {
        this.audioEngineRef = audioEngine;
    }

    get isRunning(): boolean {
        return this._isRunning;
    }

    start(): void {
        if (this._isRunning) {
            console.log('[SpeechEngine] Already running, skipping start()');
            return;
        }

        if (!this.isSupported && !this.deepgramApiKey) {
            this._isRunning = true;
            this.setStatus('unsupported');
            console.log('[SpeechEngine] Started in text-input fallback mode');
            return;
        }

        if (this.isSupported && this.webSpeechProbe === 'works') {
            this.intentionallyStopped = false;
            this.createRecognition();
            return;
        }

        if (this.webSpeechProbe === 'broken') {
            this.startWebSocketFallback();
            return;
        }

        if (this.isSupported && this.webSpeechProbe === 'untested') {
            this.intentionallyStopped = false;
            this.createRecognition();
            return;
        }

        if (!this.isSupported && this.deepgramApiKey) {
            this.startWebSocketFallback();
            return;
        }
    }

    stop(): void {
        this.intentionallyStopped = true;
        this._isRunning = false;

        if (this.restartTimer !== null) {
            clearTimeout(this.restartTimer);
            this.restartTimer = null;
        }
        if (this.restartDebounceTimer !== null) {
            clearTimeout(this.restartDebounceTimer);
            this.restartDebounceTimer = null;
        }
        if (this.safariFinalTimer !== null) {
            clearTimeout(this.safariFinalTimer);
            this.safariFinalTimer = null;
        }

        if (this.recognition) {
            try {
                this.recognition.abort();
            } catch {
                // abort() can throw if recognition is already stopped; safe to ignore
            }
            this.recognition = null;
        }
        if (this.wsClient) {
            this.wsClient.destroy();
            this.wsClient = null;
        }
        if (this._wsTranscriptUnsub) {
            this._wsTranscriptUnsub();
            this._wsTranscriptUnsub = null;
        }
        if (this._wsStatusUnsub) {
            this._wsStatusUnsub();
            this._wsStatusUnsub = null;
        }

        this.usingWebSocket = false;
        this.setStatus('off');
        console.log('[SpeechEngine] Stopped');
    }

    onTranscript(callback: TranscriptCallback): () => void {
        this.listeners.add(callback);
        return () => this.listeners.delete(callback);
    }

    onStatusChange(callback: StatusCallback): () => void {
        this.statusListeners.add(callback);
        callback(this._status, this._lastError);
        return () => this.statusListeners.delete(callback);
    }

    get status(): STTStatus {
        return this._status;
    }

    get lastError(): string {
        return this._lastError;
    }

    submitText(text: string): void {
        if (!text.trim()) return;
        this.emit({
            text: text.trim(),
            isFinal: true,
            timestamp: Date.now(),
        });
    }

    /**
     * Give up on live speech recognition and expose the manual text-input
     * fallback. Called when Web Speech repeatedly fails to reach a speech
     * backend and no Deepgram key is configured, so the user always gets a
     * visible, usable input instead of a silent retry loop. Flipping
     * `isSupported` to false is what makes UIOverlay reveal the text box;
     * keeping `_isRunning` true keeps the mic session (and box) visible.
     */
    private degradeToTextInput(reason: string): void {
        if (this.restartTimer !== null) {
            clearTimeout(this.restartTimer);
            this.restartTimer = null;
        }
        if (this.restartDebounceTimer !== null) {
            clearTimeout(this.restartDebounceTimer);
            this.restartDebounceTimer = null;
        }
        this.intentionallyStopped = true;
        this.recognition = null;
        this.isSupported = false;
        this._isRunning = true;
        this._lastError = reason;
        this.webSpeechProbe = 'broken';
        this.setStatus('unsupported', reason);
        console.warn(
            `[SpeechEngine] ⛔ Web Speech unavailable ("${reason}") after ` +
            `${this.dryRestartCount} failed attempts — switching to text-input fallback. ` +
            `For live voice, open the app in Google Chrome (Web Speech needs Google's ` +
            `speech backend, which the in-app/Electron browser lacks), or set ` +
            `VITE_DEEPGRAM_API_KEY to enable the WebSocket fallback.`,
        );
    }

    private createRecognition(): void {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const SpeechRecognitionCtor = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
        if (!SpeechRecognitionCtor) return;

        const recognition: SpeechRecognitionInstance = new SpeechRecognitionCtor();

        recognition.continuous = !this.isIOS;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            this._isRunning = true;
            this.retryCount = 0;
            if (this.webSpeechProbe === 'untested') {
                this.webSpeechProbe = 'works';
                console.log('[SpeechEngine] ✅ Web Speech API probe: works');
            }
            if (this.restartDebounceTimer !== null) {
                clearTimeout(this.restartDebounceTimer);
                this.restartDebounceTimer = null;
            }
            this.setStatus('listening');
            console.log('[SpeechEngine] 🎤 Recognition started');
        };

        let sessionHadResults = false;

        recognition.onresult = (event: SpeechRecognitionEvent) => {
            sessionHadResults = true;

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                const transcript = result[0].transcript;
                const isFinal = result.isFinal;

                // ── SAFARI WORKAROUND: Deduplication ─────────────────
                // Safari sometimes repeats the entire transcript after speech ends.
                if (this.isSafari && isFinal && transcript === this.lastFinalText) {
                    console.log('[SpeechEngine] Safari dedup: skipping repeated final transcript');
                    continue;
                }

                // ── SAFARI WORKAROUND: isFinal timeout ───────────────
                // Safari sometimes never sets isFinal=true. We track the
                // last interim result and force-finalize after 750ms of silence.
                if (this.isSafari && !isFinal) {
                    this.lastInterimText = transcript;

                    // Reset the finalization timer
                    if (this.safariFinalTimer) clearTimeout(this.safariFinalTimer);
                    this.safariFinalTimer = setTimeout(() => {
                        this.safariFinalTimer = null;
                        if (this.lastInterimText) {
                            console.log('[SpeechEngine] Safari workaround: force-finalizing after timeout');
                            const forcedEvent: TranscriptEvent = {
                                text: this.lastInterimText,
                                isFinal: true,
                                timestamp: Date.now(),
                            };
                            this.lastFinalText = this.lastInterimText;
                            this.lastInterimText = '';
                            this.emit(forcedEvent);
                        }
                    }, SAFARI_FINAL_TIMEOUT_MS);
                }

                if (isFinal) {
                    this.lastFinalText = transcript;
                    this.lastInterimText = '';
                    // Cancel Safari timer — real final arrived
                    if (this.safariFinalTimer) {
                        clearTimeout(this.safariFinalTimer);
                        this.safariFinalTimer = null;
                    }
                }

                const transcriptEvent: TranscriptEvent = {
                    text: transcript,
                    isFinal,
                    timestamp: Date.now(),
                };

                this.emit(transcriptEvent);
            }
        };

        recognition.onend = () => {
            if (!this.intentionallyStopped) {
                if (sessionHadResults) {
                    this.dryRestartCount = 0;
                } else {
                    this.dryRestartCount++;
                }

                // Web Speech keeps restarting but never yields a transcript
                // (typically repeated "network" errors on a browser with no
                // usable speech backend). Stop the silent loop and give the
                // user a working alternative + visible feedback.
                if (this.dryRestartCount >= MAX_DRY_RESTARTS_BEFORE_DEGRADE) {
                    if (this.deepgramApiKey && !this.usingWebSocket) {
                        console.warn(
                            '[SpeechEngine] Web Speech unavailable — switching to WebSocket (Deepgram) fallback',
                        );
                        this.intentionallyStopped = true;
                        this.recognition = null;
                        this.startWebSocketFallback();
                        return;
                    }
                    if (!this.deepgramApiKey) {
                        this.degradeToTextInput(this._lastError || 'network');
                        return;
                    }
                }

                const baseDelay = 300;
                const restartDelay = Math.min(
                    baseDelay * Math.pow(2, this.dryRestartCount),
                    10000,
                );

                if (this.dryRestartCount > 3) {
                    console.warn(
                        `[SpeechEngine] ${this.dryRestartCount} dry restarts in a row — ` +
                        `backing off to ${restartDelay}ms. Speech service may be unavailable.`
                    );
                } else {
                    console.log('[SpeechEngine] Recognition ended — auto-restarting...');
                }

                this.restartDebounceTimer = setTimeout(() => {
                    this.restartDebounceTimer = null;
                    if (!this.intentionallyStopped) {
                        this.setStatus('restarting');
                    }
                }, 800);

                this.restartTimer = setTimeout(() => {
                    this.restartTimer = null;
                    if (!this.intentionallyStopped) {
                        this.createRecognition();
                    }
                }, restartDelay);
            } else {
                this._isRunning = false;
                if (this._status !== 'unsupported') {
                    this.setStatus('off');
                }
                console.log('[SpeechEngine] Recognition ended (intentional stop)');
            }
        };

        recognition.onerror = (event: Event & { error: string }) => {
            const errorType = event.error;

            if (errorType === 'aborted') return;

            if (errorType === 'no-speech') {
                console.log('[SpeechEngine] No speech detected — will auto-restart');
                return;
            }

            // ── FATAL ERRORS → try WebSocket fallback ────────────────
            if (FATAL_ERRORS.has(errorType)) {
                this._lastError = errorType;

                // Cache probe result
                if (this.webSpeechProbe === 'untested') {
                    this.webSpeechProbe = 'broken';
                    console.log(`[SpeechEngine] Web Speech API probe: broken (${errorType})`);
                }

                // Try WebSocket fallback before giving up
                if (this.deepgramApiKey) {
                    console.log(`[SpeechEngine] ⚠️ Web Speech fatal error "${errorType}" — switching to WebSocket fallback`);
                    this.intentionallyStopped = true; // prevent onend from restarting
                    this.recognition = null;
                    this.startWebSocketFallback();
                    return;
                }

                // No fallback available — text input mode
                this.isSupported = false;
                this.setStatus('unsupported', errorType);
                this.intentionallyStopped = true;
                this._isRunning = true;
                console.warn(`[SpeechEngine] ⛔ Fatal error: "${errorType}" — text-input fallback`);
                return;
            }

            // ── RECOVERABLE ERRORS ───────────────────────────────────
            this._lastError = errorType;
            this.setStatus('error', errorType);
            console.warn(`[SpeechEngine] ⚠️ Error: "${errorType}" (retry ${this.retryCount + 1}/${MAX_RETRIES})`);

            if (!this.intentionallyStopped && this.retryCount < MAX_RETRIES) {
                const delay = Math.min(1000 * Math.pow(2, this.retryCount), 16000);
                this.retryCount++;
                this.restartTimer = setTimeout(() => {
                    this.restartTimer = null;
                    if (!this.intentionallyStopped) {
                        console.log(`[SpeechEngine] Retrying after error (attempt ${this.retryCount}/${MAX_RETRIES})...`);
                        this.createRecognition();
                    }
                }, delay);
            } else if (this.retryCount >= MAX_RETRIES) {
                // Max retries on Web Speech — try WebSocket as last resort
                if (this.deepgramApiKey && !this.usingWebSocket) {
                    console.log('[SpeechEngine] Max Web Speech retries — switching to WebSocket');
                    this.startWebSocketFallback();
                } else {
                    console.warn(`[SpeechEngine] ⛔ Max retries (${MAX_RETRIES}) reached — giving up`);
                    this._isRunning = false;
                }
            }
        };

        // ── START ────────────────────────────────────────────────────
        this.recognition = recognition;

        try {
            recognition.start();
        } catch (e) {
            console.error('[SpeechEngine] Failed to start recognition:', e);

            // If first-time probe fails with an exception, try WebSocket
            if (this.webSpeechProbe === 'untested') {
                this.webSpeechProbe = 'broken';
                if (this.deepgramApiKey) {
                    console.log('[SpeechEngine] Web Speech start() threw — trying WebSocket fallback');
                    this.startWebSocketFallback();
                    return;
                }
            }

            this._isRunning = false;
        }
    }

    // ── WEBSOCKET FALLBACK ──────────────────────────────────────────

    /**
     * Start the WebSocket STT fallback (Deepgram). Called when Web Speech
     * API is unavailable, broken, or has exhausted retries.
     */
    private startWebSocketFallback(): void {
        if (this.usingWebSocket) return;
        if (!this.deepgramApiKey) {
            console.warn('[SpeechEngine] No Deepgram API key — cannot start WebSocket fallback');
            this.setStatus('unsupported', 'no-api-key');
            this._isRunning = true;
            this.isSupported = false;
            return;
        }

        this.usingWebSocket = true;
        this.setStatus('connecting-ws');
        console.log('[SpeechEngine] 🔌 Starting WebSocket STT fallback (Deepgram)');

        // Create WebSocket client if not already created
        if (!this.wsClient) {
            // On iOS, reuse AudioEngine's AudioContext and MediaStream
            // to avoid creating a second context (iOS strictly limits these).
            const sharedCtx = this.audioEngineRef?.audioContext ?? undefined;
            // AudioEngine stores its MediaStream source; extract the stream from it.
            const sharedStream = this.audioEngineRef?.source?.mediaStream ?? undefined;
            this.wsClient = new WebSocketSTTClient(this.deepgramApiKey, sharedCtx, sharedStream);

            // Wire up transcript events → our listeners
            this._wsTranscriptUnsub = this.wsClient.onTranscript((event) => {
                this.emit(event);
            });

            // Wire up status changes
            this._wsStatusUnsub = this.wsClient.onStatusChange((wsStatus) => {
                switch (wsStatus) {
                    case 'connecting':
                    case 'reconnecting':
                        this.setStatus('connecting-ws');
                        break;
                    case 'listening':
                        this.setStatus('listening');
                        break;
                    case 'error':
                        this.setStatus('error', 'websocket');
                        break;
                    case 'closed':
                        if (this._isRunning) {
                            this.setStatus('error', 'ws-closed');
                        }
                        break;
                }
            });
        }

        // Start mic capture + WebSocket connection
        this.wsClient.start().catch((err) => {
            console.error('[SpeechEngine] WebSocket fallback failed to start:', err);
            this.usingWebSocket = false;
            // Final fallback: text input
            this.isSupported = false;
            // Preserve the actual error name so the UI can show contextual messages
            const errorDetail = (err instanceof DOMException && err.name === 'NotAllowedError')
                ? 'not-allowed'
                : 'ws-failed';
            this.setStatus('unsupported', errorDetail);
            this._isRunning = true;
        });

        this._isRunning = true;
    }

    // ── INTERNAL ────────────────────────────────────────────────────

    /**
     * Emit a TranscriptEvent to all registered listeners.
     */
    private emit(event: TranscriptEvent): void {
        const prefix = event.isFinal ? '✅ FINAL' : '💬 interim';
        console.log(`[SpeechEngine] ${prefix}: "${event.text}"`);

        for (const listener of this.listeners) {
            try {
                listener(event);
            } catch (e) {
                console.error('[SpeechEngine] Listener error:', e);
            }
        }
    }

    /**
     * Update the STT status and notify all status listeners.
     */
    private setStatus(status: STTStatus, errorDetail?: string): void {
        this._status = status;
        if (errorDetail) {
            this._lastError = errorDetail;
        }
        for (const listener of this.statusListeners) {
            try {
                listener(status, errorDetail);
            } catch (e) {
                console.error('[SpeechEngine] Status listener error:', e);
            }
        }
    }
}
