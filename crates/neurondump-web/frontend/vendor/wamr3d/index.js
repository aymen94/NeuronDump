// WAMR3D entry point.
//
//   import { init, World, Vector3 } from 'wamr3d';
//   await init();
//
// Everything exported here except `init`, `attributeView`, `memory` and the scalar
// helpers is implemented in the WebAssembly module; this file only loads it.

import initWasm, { seededRandom, randInt, randFloat, randFloatSpread, generateUUID, setRandomSeed, World } from './wamr3d_wasm.js';

export * from './wamr3d_wasm.js';

// The per-object transform setters are called once per object and property, so they use
// the module's unchecked forms, which return a flag instead of an error object: the glue
// of a fallible call allocates an array on every call. A dead handle still throws.
const deadObject = () => new Error( 'object handle is not alive' );

World.prototype.setPosition = function ( object, x, y, z ) {

	if ( ! this.__setPosition( object, x, y, z ) ) throw deadObject();

};

World.prototype.setScale = function ( object, x, y, z ) {

	if ( ! this.__setScale( object, x, y, z ) ) throw deadObject();

};

World.prototype.setQuaternion = function ( object, x, y, z, w ) {

	if ( ! this.__setQuaternion( object, x, y, z, w ) ) throw deadObject();

};

let wasm;

// The scalar helpers run in JavaScript: a call into the module costs more than the
// arithmetic, and doubles give the same results as three.js. These local exports take
// precedence over the module's functions of the same name. `seededRandom` and the random
// helpers stay in the module so that their sequences are shared with the engine.

const DEG2RAD = Math.PI / 180, RAD2DEG = 180 / Math.PI;

export function degToRad( degrees ) {

	return degrees * DEG2RAD;

}

export function radToDeg( radians ) {

	return radians * RAD2DEG;

}

export function clamp( value, min, max ) {

	return Math.max( min, Math.min( max, value ) );

}

export function lerp( x, y, t ) {

	return ( 1 - t ) * x + t * y;

}

export function inverseLerp( x, y, value ) {

	return x !== y ? ( value - x ) / ( y - x ) : 0;

}

export function mapLinear( x, a1, a2, b1, b2 ) {

	return b1 + ( x - a1 ) * ( b2 - b1 ) / ( a2 - a1 );

}

export function damp( x, y, lambda, dt ) {

	return lerp( x, y, 1 - Math.exp( - lambda * dt ) );

}

export function euclideanModulo( n, m ) {

	return ( ( n % m ) + m ) % m;

}

export function pingpong( x, length = 1 ) {

	return length - Math.abs( euclideanModulo( x, length * 2 ) - length );

}

export function smoothstep( x, min, max ) {

	if ( x <= min ) return 0;
	if ( x >= max ) return 1;
	x = ( x - min ) / ( max - min );
	return x * x * ( 3 - 2 * x );

}

export function smootherstep( x, min, max ) {

	if ( x <= min ) return 0;
	if ( x >= max ) return 1;
	x = ( x - min ) / ( max - min );
	return x * x * x * ( x * ( x * 6 - 15 ) + 10 );

}

export function isPowerOfTwo( value ) {

	return value > 0 && Number.isInteger( value ) && 2 ** Math.round( Math.log2( value ) ) === value;

}

export function ceilPowerOfTwo( value ) {

	return Math.pow( 2, Math.ceil( Math.log( value ) / Math.LN2 ) );

}

export function floorPowerOfTwo( value ) {

	return Math.pow( 2, Math.floor( Math.log( value ) / Math.LN2 ) );

}

/** The scalar helpers, also available as individual exports. */
export const MathUtils = Object.freeze( {
	degToRad, radToDeg, clamp, lerp, inverseLerp, mapLinear, damp, pingpong, smoothstep, smootherstep,
	euclideanModulo, isPowerOfTwo, ceilPowerOfTwo, floorPowerOfTwo, seededRandom: ( seed ) => seededRandom( seed ),
	randInt: ( low, high ) => randInt( low, high ), randFloat: ( low, high ) => randFloat( low, high ),
	randFloatSpread: ( range ) => randFloatSpread( range ), generateUUID: () => generateUUID()
} );

/**
 * Loads and instantiates the WebAssembly module. Call once, before anything else.
 *
 * With no argument the .wasm file next to this module is used: read from disk under
 * Node.js, fetched in a browser. Pass a URL, Response, ArrayBuffer or
 * WebAssembly.Module to load it from somewhere else.
 */
export async function init( source ) {

	if ( wasm !== undefined ) return;

	if ( source === undefined && typeof process !== 'undefined' && process.versions?.node ) {

		const { readFile } = await import( 'node:fs/promises' );
		source = await readFile( new URL( './wamr3d_wasm_bg.wasm', import.meta.url ) );

	}

	wasm = await initWasm( source === undefined ? undefined : { module_or_path: source } );

	// The module has no random source of its own; without this every page load would draw
	// the same random sequence.
	const seed = new Uint32Array( 2 );
	if ( globalThis.crypto?.getRandomValues ) globalThis.crypto.getRandomValues( seed );
	else seed.set( [ Math.random() * 0x100000000, Math.random() * 0x100000000 ] );
	setRandomSeed( seed[ 0 ], seed[ 1 ] );

}

/** The module's linear memory. */
export function memory() {

	if ( wasm === undefined ) throw new Error( 'WAMR3D: call init() first' );
	return wasm.memory;

}

/**
 * A Float32Array over a geometry attribute's data inside the module's memory: no copy,
 * and writes go straight to the engine. Call world.markAttributeNeedsUpdate() after writing.
 *
 * The view becomes unusable (length 0) whenever the module's memory grows, which any
 * allocation can cause, and stale when the attribute is replaced or its geometry
 * disposed. Create it right before use, or again after other engine calls.
 */
export function attributeView( world, geometry, name ) {

	return new Float32Array( memory().buffer, world.attributePtr( geometry, name ), world.attributeLength( geometry, name ) );

}
