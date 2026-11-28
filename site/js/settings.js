// Controller settings shared by the page (main.js) and the timing test (build/test/sim.html).
export const PRESETS = {
  // AI only (default): the network's own motion and timing, as in NVIDIA's MotionBricks demos. The controller only says
  // where to go (a target pose window from motion capture, placed along your path); the network chooses how long it takes,
  // so speed, stride and timing are its own. No steering warps, time-warps, procedural lean/look/foot IK, joint clamping
  // or pre-generated transitions; only upstream MotionBricks' endpoint correction is applied.
  ai: { pureAI: true, aiSlack: 1, responsiveness: 0.12, facingResponsiveness: 0.10, predictionHorizon: 0.9, replanInterval: 10, acceleration: 9, deceleration: 12, smoothing: 0, transitionDuration: 0.18, steering: 0, snap: 0, lean: 0 },
  // third-person-shooter feel: quick plans and blends, steering assist, time-warped starts/stops/turns, leaning into curves
  shooter: { pureAI: false, responsiveness: 0.06, facingResponsiveness: 0.06, predictionHorizon: 0.65, replanInterval: 6, acceleration: 16, deceleration: 20, smoothing: 0, transitionDuration: 0.14, steering: 220, snap: 0.35, lean: 0.8 },
  responsive: { pureAI: false, responsiveness: 0.06, facingResponsiveness: 0.06, predictionHorizon: 0.65, replanInterval: 6, acceleration: 16, deceleration: 20, smoothing: 0, transitionDuration: 0.14, steering: 0, snap: 0, lean: 0 },
  balanced: { pureAI: false, responsiveness: 0.12, facingResponsiveness: 0.10, predictionHorizon: 0.8, replanInterval: 10, acceleration: 9, deceleration: 12, smoothing: 0, transitionDuration: 0.18, steering: 0, snap: 0, lean: 0 },
  smooth: { pureAI: false, responsiveness: 0.24, facingResponsiveness: 0.22, predictionHorizon: 1.1, replanInterval: 14, acceleration: 6, deceleration: 7, smoothing: 0.35, transitionDuration: 0.3, steering: 0, snap: 0, lean: 0.5 },
};
export const DEFAULTS = {
  model: 'balanced', preset: 'ai', walkStyle: 'natural', walkSpeed: 1.4, runSpeed: 3.4, sprintSpeed: 5.5, crouchSpeed: 0.9, inertia: 1.0, momentum: 0.15,
  turnRate: 540, replanLead: 3, inputReplanThreshold: 0.35, minTokens: 6, maxTokens: 14, durationMode: 'horizon',
  velocityMatching: true, playbackRate: 1.0, variation: 0.15, footLocking: true, anatomical: true, fingers: true,
  fingerCurlRun: 55, fingerFollow: 10, torsoCalm: 0.4, settingsVersion: 4, lookAt: true, instant: true, bodyFX: true, autoQuality: true, ...PRESETS.ai,
};
export const PRESET_LABELS = { ai: 'AI only', shooter: 'Shooter', responsive: 'Responsive', balanced: 'Balanced', smooth: 'Smooth' };
