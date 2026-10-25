// Vectors, quaternions, Euler angles, matrices, boxes and spheres of the object API, kept in
// JavaScript. They are small values that applications create and change constantly; doing
// that work across the WebAssembly boundary made each call several times slower than
// three.js. The engine gets them as numbers or arrays where it needs them (see
// `math.js`).
//
// Behaviour follows three.js (MIT License, Copyright 2010-2026 three.js authors); numbers
// are 64-bit here, as in three.js.

// One module per class under `values/`; this module gathers them.

export * from './values/vector2.js';
export * from './values/vector3.js';
export * from './values/vector4.js';
export * from './values/quaternion.js';
export * from './values/euler.js';
export * from './values/matrix3.js';
export * from './values/matrix4.js';
export * from './values/box3.js';
export * from './values/sphere.js';
