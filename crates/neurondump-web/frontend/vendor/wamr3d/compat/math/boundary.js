// The engine boundary: engine copies of JavaScript values, and compat classes over engine classes.
//
// Part of the math classes; `../math.js` gathers them.

import * as W from '../../index.js';
import { Box3, Euler, Matrix3, Matrix4, Quaternion, Sphere, Vector2, Vector3, Vector4 } from '../values.js';

// An engine copy of a JavaScript value, for an engine method's argument. Copies are
// pushed to `temporaries`, which the caller frees after the call. Other arguments,
// engine objects included, pass unchanged.
export function toEngine( value, temporaries ) {

	if ( value === null || typeof value !== 'object' || value.__wbg_ptr !== undefined ) return value;
	let copy;

	if ( value.isVector3 ) copy = new W.Vector3( value.x, value.y, value.z );
	else if ( value.isMatrix4 ) ( copy = new W.Matrix4() ).fromArray( value.elements );
	else if ( value.isQuaternion ) copy = new W.Quaternion( value.x, value.y, value.z, value.w );
	else if ( value.isVector2 ) copy = new W.Vector2( value.x, value.y );
	else if ( value.isVector4 ) copy = new W.Vector4( value.x, value.y, value.z, value.w );
	else if ( value.isMatrix3 ) ( copy = new W.Matrix3() ).fromArray( value.elements );
	else if ( value.isEuler ) copy = new W.Euler( value.x, value.y, value.z, value.order );
	else if ( value.isBox3 ) ( copy = new W.Box3() ).set( toEngine( value.min, temporaries ), toEngine( value.max, temporaries ) );
	else if ( value.isSphere ) copy = new W.Sphere( toEngine( value.center, temporaries ), value.radius );
	else return value;

	temporaries.push( copy );
	return copy;

}

export function free( temporaries ) {

	for ( const value of temporaries ) value.free();

}

// One reused engine vector per slot, for engine calls that read a vector and keep nothing.
const scratch = [];

export function engineVector( v, slot = 0 ) {

	const vector = scratch[ slot ] ??= new W.Vector3( 0, 0, 0 );
	vector.set( v.x, v.y, v.z );
	return vector;

}

const compatByBase = new Map();
const NULLABLE = new Set( [ 'intersectSphere', 'distanceToPlane', 'intersectPlane', 'intersectBox', 'intersectTriangle', 'getBarycoord' ] );

// A value returned by the engine as a class of this module: JavaScript values are copied
// out and the engine value freed; other engine values take the module's class.
function adopt( value ) {

	if ( value === null || typeof value !== 'object' ) return value;

	if ( value instanceof W.Vector3 ) return release( value, new Vector3( value.x, value.y, value.z ) );
	if ( value instanceof W.Matrix4 ) return release( value, new Matrix4().fromArray( value.toArray() ) );
	if ( value instanceof W.Quaternion ) return release( value, new Quaternion( value.x, value.y, value.z, value.w ) );
	if ( value instanceof W.Vector2 ) return release( value, new Vector2( value.x, value.y ) );
	if ( value instanceof W.Vector4 ) return release( value, new Vector4( value.x, value.y, value.z, value.w ) );
	if ( value instanceof W.Matrix3 ) return release( value, new Matrix3().fromArray( value.toArray() ) );
	if ( value instanceof W.Euler ) return release( value, new Euler( value.x, value.y, value.z, value.order ) );
	if ( value instanceof W.Box3 ) return release( value, new Box3( adopt( value.min ), adopt( value.max ) ) );
	if ( value instanceof W.Sphere ) return release( value, new Sphere( adopt( value.center ), value.radius ) );

	const prototype = compatByBase.get( Object.getPrototypeOf( value ) );
	if ( prototype ) Object.setPrototypeOf( value, prototype );
	return value;

}

function release( engineValue, value ) {

	engineValue.free();
	return value;

}

// Gives a class the calling conventions of its engine base class plus: methods return
// `this` for chaining, a trailing `target` argument receives the result, methods that
// can fail return null, and JavaScript values are accepted wherever the engine takes its
// own.
export function extend( Compat, Base, name ) {

	compatByBase.set( Base.prototype, Compat.prototype );
	Compat.prototype[ 'is' + name ] = true;

	for ( const key of Object.getOwnPropertyNames( Base.prototype ) ) {

		if ( key === 'constructor' || key === 'free' || key.startsWith( '__' ) || Object.hasOwn( Compat.prototype, key ) ) continue;
		const descriptor = Object.getOwnPropertyDescriptor( Base.prototype, key );

		if ( descriptor.get ) {

			Object.defineProperty( Compat.prototype, key, { get() {

				return adopt( descriptor.get.call( this ) );

			}, set: descriptor.set, configurable: true } );
			continue;

		}

		const method = descriptor.value;
		if ( typeof method !== 'function' ) continue;
		const arity = method.length;

		Compat.prototype[ key ] = function () {

			// Reuse the arguments object instead of allocating a rest array on every call.
			const args = arguments;
			const target = args.length > arity ? args[ arity ] : null, offset = args[ arity + 1 ] || 0;
			if ( args.length > arity ) args.length = arity;
			let temporaries = null;

			for ( let i = 0; i < args.length; i ++ ) {

				const value = args[ i ];
				if ( value !== null && typeof value === 'object' && value.__wbg_ptr === undefined && ! ArrayBuffer.isView( value ) ) args[ i ] = toEngine( value, temporaries ??= [] );

			}

			let result;
			switch ( args.length ) {

				case 0: result = method.call( this ); break;
				case 1: result = method.call( this, args[ 0 ] ); break;
				case 2: result = method.call( this, args[ 0 ], args[ 1 ] ); break;
				case 3: result = method.call( this, args[ 0 ], args[ 1 ], args[ 2 ] ); break;
				case 4: result = method.call( this, args[ 0 ], args[ 1 ], args[ 2 ], args[ 3 ] ); break;
				default: result = method.apply( this, args );

			}

			if ( temporaries ) free( temporaries );

			if ( result === undefined ) return NULLABLE.has( key ) ? null : this;

			if ( target !== null && target !== undefined && typeof result === 'object' ) {

				if ( ArrayBuffer.isView( result ) ) {

					for ( let i = 0; i < result.length; i ++ ) target[ offset + i ] = result[ i ];
					return target;

				}

				if ( typeof target.copy === 'function' ) {

					const value = adopt( result );
					target.copy( value );
					value.free?.();
					return target;

				}

			}

			return adopt( result );

		};

	}

}
