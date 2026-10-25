// WAMR3D math values in JavaScript: `wamr3d/math`.
//
//   import { init, World } from 'wamr3d';
//   import { Vector3, Quaternion } from 'wamr3d/math';
//
// The vectors, quaternions, Euler angles, matrices, boxes and spheres of the object API,
// without the rest of it and without the WebAssembly module: they need no `init()`. A
// small operation costs less in JavaScript than a call into WebAssembly, so use these for
// per-value math next to the low-level API, and hand the results to the engine in batches
// (`world.setTransforms` and the other typed-array calls).

export * from './compat/values.js';
