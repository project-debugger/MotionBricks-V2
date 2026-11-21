// Controller settings shared by the page (main.js) and the timing test (build/test/sim.html).
export const PRESETS = {
  // third-person-shooter feel: quick plans and blends, steering assist, time-warped starts/stops/turns, leaning into curves
  shooter: { responsiveness: 0.06, facingResponsiveness: 0.06, predictionHorizon: 0.65, replanInterval: 6, acceleration: 16, deceleration: 20, smoothing: 0, transitionDuration: 0.14, steering: 220, snap: 0.35, lean: 0.8 },
  responsive: { responsiveness: 0.06, facingResponsiveness: 0.06, predictionHorizon: 0.65, replanInterval: 6, acceleration: 16, deceleration: 20, smoothing: 0, transitionDuration: 0.14, steering: 0, snap: 0, lean: 0 },
  balanced: { responsiveness: 0.12, facingResponsiveness: 0.10, predictionHorizon: 0.8, replanInterval: 10, acceleration: 9, deceleration: 12, smoothing: 0, transitionDuration: 0.18, steering: 0, snap: 0, lean: 0 },
  smooth: { responsiveness: 0.24, facingResponsiveness: 0.22, predictionHorizon: 1.1, replanInterval: 14, acceleration: 6, deceleration: 7, smoothing: 0.35, transitionDuration: 0.3, steering: 0, snap: 0, lean: 0.5 },
};
export const DEFAULTS = {
  model: 'balanced', character: 'human', animalMotion: 'network', preset: 'shooter', walkStyle: 'natural', walkSpeed: 1.4, runSpeed: 3.4, sprintSpeed: 5.5, crouchSpeed: 0.9, inertia: 1.0, momentum: 0.15,
  turnRate: 540, replanLead: 3, inputReplanThreshold: 0.35, minTokens: 6, maxTokens: 14, durationMode: 'horizon',
  velocityMatching: true, playbackRate: 1.0, variation: 0.15, footLocking: true, anatomical: true, fingers: true,
  fingerCurlRun: 55, fingerFollow: 10, torsoCalm: 0.4, settingsVersion: 3, lookAt: true, instant: true, bodyFX: true, autoQuality: true, ...PRESETS.shooter,
};
